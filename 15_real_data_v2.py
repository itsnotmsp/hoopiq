"""
Step 15 v2 — REAL Per-Game Player Logs from ESPN
-------------------------------------------------
Drop-in replacement for 15_real_data_final.py.

What changed:
  - OLD: fetched season AVERAGES, then synthesized 40 fake games per player by
         adding Gaussian noise around those averages. Game logs were not real.
  - NEW: fetches actual per-game logs from ESPN's gamelog endpoint:
           /athletes/{id}/gamelog
         Each row in the output parquet is a real game the player actually played.

Output schema (identical to the old script — downstream code keeps working):
  PLAYER_ID, PLAYER_NAME, PLAYER_TEAM, TEAM, GAME_ID, GAME_DATE, SEASON,
  OPP, HOME, RESULT, MIN, PTS, REB, AST, STL, BLK, TOV,
  FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, FPTS

Run:
  python 15_real_data_v2.py

Then retrain prop models on the real data:
  python 7_player_model.py
  python 5_api_server.py
"""

import asyncio
import re
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

console = Console()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ESPN_TEAMS   = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
ESPN_ROSTER  = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
ESPN_GAMELOG = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{athlete_id}/gamelog"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.espn.com/",
}

# ESPN gamelog stat labels we care about (label → our schema column)
# ESPN typical labels: MIN, FG, FG%, 3PT, 3P%, FT, FT%, REB, AST, BLK, STL, PF, TO, PTS
# Some labels are "X-Y" strings (made-attempted) which we split.
SPLIT_FIELDS = {"FG", "3PT", "FT"}  # "made-attempted" strings


def _to_float(x, default=0.0):
    """Coerce ESPN stat string to float, tolerating None / '-' / ''."""
    if x is None:
        return default
    s = str(x).strip()
    if not s or s in {"-", "--", "DNP", "NP"}:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _parse_made_attempted(s):
    """'10-22' → (10.0, 22.0). Returns (0.0, 0.0) on failure."""
    if not s or not isinstance(s, str):
        return 0.0, 0.0
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$", s)
    if not m:
        return 0.0, 0.0
    return float(m.group(1)), float(m.group(2))


def _parse_score_string(s):
    """ESPN result strings look like 'W 102-95' or 'L 95-102'. Returns ('W'|'L', None)."""
    if not s:
        return None
    s = str(s).strip()
    if s.startswith(("W", "w")):
        return "W"
    if s.startswith(("L", "l")):
        return "L"
    return None


async def fetch_all_teams(client):
    r = await client.get(ESPN_TEAMS, headers=HEADERS, timeout=15.0)
    r.raise_for_status()
    teams = []
    for sport in r.json().get("sports", []):
        for league in sport.get("leagues", []):
            for td in league.get("teams", []):
                t = td.get("team", {})
                teams.append({"id": t.get("id"), "abbr": t.get("abbreviation")})
    return teams


async def fetch_team_roster(client, team_id):
    try:
        r = await client.get(ESPN_ROSTER.format(team_id=team_id), headers=HEADERS, timeout=15.0)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    team_abbr = (data.get("team") or {}).get("abbreviation", "")
    athletes = []
    for entry in data.get("athletes", []):
        if "items" in entry:
            for a in entry["items"]:
                athletes.append({
                    "id":   a.get("id"),
                    "name": a.get("displayName") or a.get("fullName"),
                    "team": team_abbr,
                })
        elif "id" in entry:
            athletes.append({
                "id":   entry.get("id"),
                "name": entry.get("displayName") or entry.get("fullName"),
                "team": team_abbr,
            })
    return [a for a in athletes if a.get("id") and a.get("name")]


async def fetch_gamelog(client, athlete_id):
    """Returns the raw gamelog JSON for a player, or None on failure."""
    try:
        r = await client.get(
            ESPN_GAMELOG.format(athlete_id=athlete_id),
            headers=HEADERS,
            timeout=20.0,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def parse_gamelog(raw, athlete):
    """
    Extract per-game stat rows from an ESPN gamelog payload.

    Structure:
      raw["events"]                    : dict {event_id: {gameDate, opponent, atVs, score, ...}}
      raw["seasonTypes"][i]["categories"][j]["events"]
                                       : list of {eventId, stats: [...]}
      raw["labels"]                    : list of stat names matching the stats arrays
                                         e.g. ["MIN","FG","FG%","3PT","3P%","FT","FT%",
                                               "REB","AST","BLK","STL","PF","TO","PTS"]
    """
    if not raw:
        return []

    labels = raw.get("labels") or raw.get("names") or []
    events_meta = raw.get("events") or {}
    season_types = raw.get("seasonTypes") or []
    if not labels or not season_types:
        return []

    rows = []

    # Walk every season type → category → event, collect stat arrays keyed by eventId
    for st in season_types:
        season_year = st.get("season") or st.get("year")
        # Only include current/most-recent season — usually seasonTypes[0] is current.
        # We accept all season types here; downstream filters by date if needed.
        for cat in st.get("categories") or []:
            for ev in cat.get("events") or []:
                ev_id = str(ev.get("eventId") or ev.get("id") or "")
                stats = ev.get("stats") or []
                if not ev_id or not stats:
                    continue

                # Map labels → values
                stat_map = {}
                for i, label in enumerate(labels):
                    stat_map[label] = stats[i] if i < len(stats) else None

                meta = events_meta.get(ev_id) or {}
                # Game date — ESPN gives ISO 8601 with timezone
                date_iso = meta.get("gameDate") or meta.get("date") or ""
                try:
                    game_date = pd.to_datetime(date_iso).date()
                except Exception:
                    continue  # skip rows with no parseable date

                # Opponent
                opp_obj = meta.get("opponent") or {}
                opp_abbr = opp_obj.get("abbreviation") or opp_obj.get("displayName", "")[:3].upper()

                # Home/Away — "atVs" is "@" for away, "vs" for home
                at_vs = (meta.get("atVs") or meta.get("homeAwaySymbol") or "").strip().lower()
                is_home = at_vs in {"vs", "vs."}

                # Result
                result = _parse_score_string(meta.get("score") or meta.get("gameResult"))

                # Pull individual stats
                minutes = _to_float(stat_map.get("MIN"))

                # Skip DNPs (zero minutes and zero points → likely DNP/injury)
                pts_raw = _to_float(stat_map.get("PTS"))
                if minutes == 0 and pts_raw == 0:
                    continue

                fgm, fga   = _parse_made_attempted(stat_map.get("FG"))
                fg3m, fg3a = _parse_made_attempted(stat_map.get("3PT"))
                ftm, fta   = _parse_made_attempted(stat_map.get("FT"))

                reb = _to_float(stat_map.get("REB"))
                ast = _to_float(stat_map.get("AST"))
                stl = _to_float(stat_map.get("STL"))
                blk = _to_float(stat_map.get("BLK"))
                tov = _to_float(stat_map.get("TO"))
                pts = pts_raw

                # ESPN gamelog only provides TOTAL rebounds, not the
                # offensive/defensive split. Previously this fabricated
                # DREB=REB, OREB=0 — a FALSE signal (claims every rebound was
                # defensive) that added noise to any feature using these.
                # Instead use the league-average split (~22% offensive) as a
                # neutral estimate, and flag it so the model can be told these
                # are derived, not measured. The prop model drops these from
                # its feature set entirely (see REB_SPLIT_ESTIMATED).
                oreb = round(reb * 0.22, 1)
                dreb = round(reb * 0.78, 1)
                reb_split_estimated = 1

                # DraftKings fantasy points
                fpts = pts * 1.0 + reb * 1.25 + ast * 1.5 + stl * 2.0 + blk * 2.0 - tov * 0.5
                if pts >= 10 and reb >= 10:
                    fpts += 1.5
                if pts >= 10 and reb >= 10 and ast >= 10:
                    fpts += 3.0

                rows.append({
                    "PLAYER_ID":   int(athlete["id"]),
                    "PLAYER_NAME": athlete["name"],
                    "PLAYER_TEAM": athlete["team"],
                    "TEAM":        athlete["team"],
                    "GAME_ID":     ev_id,
                    "GAME_DATE":   pd.to_datetime(game_date),
                    "SEASON":      int(season_year) if season_year else 2026,
                    "OPP":         opp_abbr or "",
                    "HOME":        bool(is_home),
                    "RESULT":      result or "",
                    "MIN":         round(minutes, 1),
                    "PTS":         int(pts),
                    "REB":         int(reb),
                    "AST":         int(ast),
                    "STL":         round(stl, 1),
                    "BLK":         round(blk, 1),
                    "TOV":         round(tov, 1),
                    "FGM":         int(fgm),
                    "FGA":         int(fga),
                    "FG3M":        int(fg3m),
                    "FG3A":        int(fg3a),
                    "FTM":         int(ftm),
                    "FTA":         int(fta),
                    "OREB":        round(oreb, 1),
                    "DREB":        round(dreb, 1),
                    "REB_SPLIT_ESTIMATED": reb_split_estimated,
                    "FPTS":        round(fpts, 2),
                })

    # Deduplicate on GAME_ID — playoffs and regular season can repeat eventIds
    # across season types in some payloads.
    if rows:
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="first")
        rows = df.to_dict("records")

    return rows


async def run():
    console.print("[bold orange1]HoopIQ Real Per-Game Logs Pipeline (v2)[/bold orange1]\n")

    async with httpx.AsyncClient() as client:
        teams = await fetch_all_teams(client)
        console.print(f"Found {len(teams)} teams")

        all_athletes = []
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), MofNCompleteColumn(), console=console,
        ) as prog:
            task = prog.add_task("Rosters", total=len(teams))
            for t in teams:
                prog.update(task, description=f"[cyan]{t['abbr']}[/cyan]")
                all_athletes.extend(await fetch_team_roster(client, t["id"]))
                prog.advance(task)

        console.print(f"\n[green]✓[/green] {len(all_athletes)} active players\n")
        console.print("[bold]Fetching real game logs from ESPN...[/bold]")

        sem = asyncio.Semaphore(10)  # 10 concurrent — ESPN rate-limits hard above this

        async def process(athlete):
            async with sem:
                raw = await fetch_gamelog(client, athlete["id"])
                if not raw:
                    return None, 0
                rows = parse_gamelog(raw, athlete)
                return rows, len(rows)

        all_rows = []
        success_count = 0
        zero_game_count = 0

        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), MofNCompleteColumn(), console=console,
        ) as prog:
            task = prog.add_task("Players", total=len(all_athletes))

            tasks = [process(a) for a in all_athletes]
            for fut in asyncio.as_completed(tasks):
                rows, n = await fut
                if rows:
                    all_rows.extend(rows)
                    success_count += 1
                else:
                    zero_game_count += 1
                prog.advance(task)

        console.print(
            f"\n[green]✓[/green] Got real game logs for "
            f"{success_count}/{len(all_athletes)} players "
            f"([dim]{zero_game_count} had no log data[/dim])"
        )

        if not all_rows:
            console.print("[red]No data generated! Check network/ESPN access.[/red]")
            return

        df = pd.DataFrame(all_rows)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

        out = DATA_DIR / "player_logs.parquet"
        if out.exists():
            backup = DATA_DIR / "player_logs_synthetic_backup.parquet"
            if not backup.exists():
                out.rename(backup)
                console.print(f"[dim]Backed up old (synthetic) data to {backup.name}[/dim]")

        df.to_parquet(out, index=False)

        console.print(f"\n[bold green]✓ Saved {len(df):,} REAL game-log rows[/bold green]")
        console.print(f"  Players:     {df['PLAYER_NAME'].nunique()}")
        console.print(f"  Date range:  {df['GAME_DATE'].min().date()} → {df['GAME_DATE'].max().date()}")
        console.print(f"  Games/player (median): {df.groupby('PLAYER_NAME').size().median():.0f}")
        console.print(
            f"  League avg PTS: {df['PTS'].mean():.1f}  "
            f"REB: {df['REB'].mean():.1f}  AST: {df['AST'].mean():.1f}"
        )

        console.print("\n[bold]Top 20 scorers (REAL per-game averages):[/bold]")
        top = (
            df.groupby("PLAYER_NAME")
              .agg(GP=("PTS", "size"), PPG=("PTS", "mean"),
                   RPG=("REB", "mean"), APG=("AST", "mean"),
                   MPG=("MIN", "mean"))
              .query("GP >= 10")
              .sort_values("PPG", ascending=False)
              .head(20)
        )
        for name, row in top.iterrows():
            console.print(
                f"  {name:30s}  {int(row['GP']):3d} GP   "
                f"{row['PPG']:5.1f} PPG   {row['RPG']:4.1f} RPG   "
                f"{row['APG']:4.1f} APG   {row['MPG']:4.1f} MPG"
            )

        # Sanity check: spot-check a known player
        for check_name in ["Tobias Harris", "Cade Cunningham", "Nikola Jokic"]:
            sub = df[df["PLAYER_NAME"].str.lower() == check_name.lower()]
            if len(sub):
                last5 = sub.sort_values("GAME_DATE").tail(5)
                console.print(
                    f"\n[dim]Sanity check — {check_name} last 5 games "
                    f"(MIN/PTS/REB/AST):[/dim]"
                )
                for _, r in last5.iterrows():
                    console.print(
                        f"  {r['GAME_DATE'].date()}  vs {r['OPP']:>4s}   "
                        f"{r['MIN']:>4.1f} / {r['PTS']:>2d} / "
                        f"{r['REB']:>2d} / {r['AST']:>2d}"
                    )

        console.print("\n[bold orange1]Now run:[/bold orange1]")
        console.print("  [cyan]python 7_player_model.py[/cyan]   ← retrain on real data")
        console.print("  [cyan]python 5_api_server.py[/cyan]    ← restart server")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
