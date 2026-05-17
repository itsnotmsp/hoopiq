"""
Step 2 — Historical Data Pipeline (ESPN API)
----------------------------------------------
Uses the same ESPN API as Step 1 — proven to work, fast, no rate limits.
Pulls historical game logs for all 30 NBA teams across 2 seasons.

Usage:
    python 2_data_pipeline.py             # pull 2022-23 + 2023-24 seasons
    python 2_data_pipeline.py --update    # add new games only

Output:
    data/game_logs.parquet    — one row per team per game
    data/team_stats.parquet   — team season summaries
"""

import argparse
import asyncio
import time
import logging
from pathlib import Path

import httpx
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()
logging.basicConfig(level=logging.WARNING)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ESPN season year = end year of the season (2023 = 2022-23)
# 2021-2026 = six seasons. Deliberately excludes 2020 (COVID bubble: neutral
# sites, no travel/crowds, compressed schedule — structurally different from
# normal NBA and known to hurt models predicting normal games).
SEASON_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

ESPN_BASE    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
TEAMS_URL    = f"{ESPN_BASE}/teams"
SCHEDULE_URL = f"{ESPN_BASE}/teams/{{team_id}}/schedule"
SUMMARY_URL  = f"{ESPN_BASE}/summary"   # ?event={game_id}, returns full box score

HEADERS = {"User-Agent": "HoopIQ/1.0", "Accept": "application/json"}

# All 30 NBA team ESPN IDs (stable, won't change)
NBA_TEAMS = [
    (1,  "ATL"), (2,  "BOS"), (3,  "NOP"), (4,  "CHI"), (5,  "CLE"),
    (6,  "DAL"), (7,  "DEN"), (8,  "DET"), (9,  "GSW"), (10, "HOU"),
    (11, "IND"), (12, "LAC"), (13, "LAL"), (14, "MIA"), (15, "MIL"),
    (16, "MIN"), (17, "BKN"), (18, "NYK"), (19, "ORL"), (20, "PHI"),
    (21, "PHX"), (22, "POR"), (23, "SAC"), (24, "SAS"), (25, "OKC"),
    (26, "UTA"), (27, "MEM"), (28, "WAS"), (29, "TOR"), (30, "OKC"),
]
# Deduplicate by team id
NBA_TEAMS = list({tid: (tid, abbr) for tid, abbr in NBA_TEAMS}.values())


# ---------------------------------------------------------------------------
# ESPN fetchers
# ---------------------------------------------------------------------------

async def fetch_team_schedule(client: httpx.AsyncClient, team_id: int, season_year: int,
                              seasontype: int = 2) -> list[dict]:
    """Fetch all games for one team in one season.

    seasontype: 1 = preseason, 2 = regular season, 3 = playoffs.
    Call once with 2 and once with 3 to get the full year.
    """
    try:
        r = await client.get(
            SCHEDULE_URL.format(team_id=team_id),
            params={"season": season_year, "seasontype": seasontype},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    events = data.get("events", [])
    rows = []

    for ev in events:
        comps = ev.get("competitions", [{}])[0]
        competitors = comps.get("competitors", [])

        team_comp = next((c for c in competitors if int(c.get("id", 0)) == team_id), None)
        opp_comp  = next((c for c in competitors if int(c.get("id", 0)) != team_id), None)

        if not team_comp or not opp_comp:
            continue

        status = comps.get("status", {}).get("type", {})
        if not status.get("completed", False):
            continue  # skip future/live games

        is_home = team_comp.get("homeAway") == "home"
        # ESPN schedule API returns score as dict {"value":110} or plain int
        def parse_score(raw):
            if isinstance(raw, dict):
                return int(raw.get("value", 0) or 0)
            return int(raw or 0)
        team_score = parse_score(team_comp.get("score", 0))
        opp_score  = parse_score(opp_comp.get("score", 0))
        won = team_score > opp_score

        # Extract linescores (per-quarter points)
        linescores = team_comp.get("linescores", [])
        q_pts = [int(ls.get("value", 0)) for ls in linescores]

        stats_raw = team_comp.get("statistics", [])
        stat_map = {s.get("name"): s.get("displayValue", "") for s in stats_raw}

        def s(key, default=0.0):
            v = stat_map.get(key, "")
            try:
                return float(str(v).replace("%", "")) if v else default
            except:
                return default

        opp_team = opp_comp.get("team", {})
        matchup = (
            f"{data.get('team', {}).get('abbreviation', '')} vs. {opp_team.get('abbreviation', '')}"
            if is_home else
            f"{data.get('team', {}).get('abbreviation', '')} @ {opp_team.get('abbreviation', '')}"
        )

        rows.append({
            "GAME_ID":           ev.get("id"),
            "TEAM_ID":           team_id,
            "TEAM_ABBREVIATION": data.get("team", {}).get("abbreviation", ""),
            "GAME_DATE":         ev.get("date", "")[:10],
            "MATCHUP":           matchup,
            "IS_HOME":           is_home,
            "OPP_ABBR":          opp_team.get("abbreviation", ""),
            "WL":                "W" if won else "L",
            "WIN":               int(won),
            "PTS":               team_score,
            "OPP_PTS":           opp_score,
            "PLUS_MINUS":        team_score - opp_score,
            # Stats from ESPN (available for completed games)
            "FGM":    s("fieldGoalsMade"),
            "FGA":    s("fieldGoalsAttempted"),
            "FG_PCT": s("fieldGoalPct"),
            "FG3M":   s("threePointFieldGoalsMade"),
            "FG3A":   s("threePointFieldGoalsAttempted"),
            "FG3_PCT":s("threePointFieldGoalPct"),
            "FTM":    s("freeThrowsMade"),
            "FTA":    s("freeThrowsAttempted"),
            "FT_PCT": s("freeThrowPct"),
            "REB":    s("totalRebounds"),
            "OREB":   s("offensiveRebounds"),
            "DREB":   s("defensiveRebounds"),
            "AST":    s("assists"),
            "STL":    s("steals"),
            "BLK":    s("blocks"),
            "TOV":    s("turnovers"),
            "PF":     s("foulsPersonal"),
            "SEASON": f"{season_year-1}-{str(season_year)[2:]}",
            "SEASON_YEAR": season_year,
        })

    return rows


async def fetch_boxscore_stats(client: httpx.AsyncClient, game_id: str) -> dict:
    """
    Fetch full box-score stats for one game.

    Why this exists: ESPN's /teams/{id}/schedule endpoint returns playoff games
    with PTS/score filled in but ALL the box-score fields (FGA, FTA, OREB, TOV,
    etc.) as 0. Those stats are only available via the /summary endpoint per
    game. Without a second pass, anything that uses FGA/FTA/etc. (pace, true
    shooting, possessions) silently returns 0 for every playoff game.

    Returns: {team_id_int: {stat_name: value}} or empty dict on error.
    """
    try:
        r = await client.get(SUMMARY_URL, params={"event": game_id}, timeout=15.0)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    out = {}
    for team_box in data.get("boxscore", {}).get("teams", []):
        try:
            tid = int(team_box.get("team", {}).get("id", 0))
        except (TypeError, ValueError):
            continue
        if not tid:
            continue
        # ESPN box-score stats are a flat list of {name, displayValue} dicts.
        # Naming convention sometimes differs from the schedule endpoint —
        # we'll map common variants to the schedule names downstream code uses.
        stats = {}
        for s in team_box.get("statistics", []):
            name = s.get("name") or s.get("label") or s.get("abbreviation") or ""
            val  = s.get("displayValue", "")
            if name:
                stats[name] = val
        out[tid] = stats
    return out


def _parse_stat(stat_map: dict, *keys, default: float = 0.0) -> float:
    """Try a list of possible stat names (different endpoints use different ones)
    and return the first hit as a float."""
    for k in keys:
        v = stat_map.get(k, "")
        if v in (None, "", "-"):
            continue
        s = str(v)
        # Some stats arrive as "fieldGoalsMade-fieldGoalsAttempted" pair "12-30"
        if "-" in s and not s.startswith("-"):
            # caller will handle made/attempted pairs separately
            continue
        try:
            return float(s.replace("%", ""))
        except ValueError:
            continue
    return default


def _parse_made_attempted(stat_map: dict, key: str) -> tuple[float, float]:
    """ESPN sometimes encodes shots as 'made-attempted' (e.g. '40-87'). Parse both."""
    v = stat_map.get(key, "")
    if isinstance(v, str) and "-" in v:
        parts = v.split("-", 1)
        try:
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            pass
    return 0.0, 0.0


async def backfill_box_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every game where stats are missing (FGA == 0 despite PTS > 0 — the
    classic playoff data shape), pull the box score and fill the stat columns.
    Concurrent with a semaphore so we don't slam ESPN.
    """
    if "FGA" not in df.columns:
        return df

    # Find rows that need backfilling
    needs = df[(df["FGA"] == 0) & (df["PTS"] > 0)]
    if not len(needs):
        console.print("[dim]No box-score backfill needed.[/dim]")
        return df

    game_ids = needs["GAME_ID"].unique().tolist()
    console.print(
        f"[yellow]Backfilling box-score stats for {len(game_ids)} games "
        f"(playoff games + any others missing detail)...[/yellow]"
    )

    sem = asyncio.Semaphore(8)   # 8 concurrent requests is fine for ESPN

    async def fetch_one(client, gid):
        async with sem:
            return gid, await fetch_boxscore_stats(client, str(gid))

    async with httpx.AsyncClient(headers=HEADERS) as client:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), MofNCompleteColumn(), console=console,
        ) as prog:
            task = prog.add_task("Fetching box scores...", total=len(game_ids))
            tasks = [fetch_one(client, gid) for gid in game_ids]
            box_by_game = {}
            for fut in asyncio.as_completed(tasks):
                gid, box = await fut
                if box:
                    box_by_game[gid] = box
                prog.advance(task)

    # Apply the backfilled stats back into the DataFrame.
    # df has two rows per GAME_ID (one per team), keyed by (GAME_ID, TEAM_ID).
    updated = 0
    for idx, row in needs.iterrows():
        gid, tid = row["GAME_ID"], int(row["TEAM_ID"])
        box = box_by_game.get(gid)
        if not box or tid not in box:
            continue
        stats = box[tid]

        # ESPN /summary uses different stat names than /schedule. Map both.
        fgm, fga   = _parse_made_attempted(stats, "fieldGoalsMade-fieldGoalsAttempted")
        if fga == 0:
            fga = _parse_stat(stats, "fieldGoalsAttempted")
            fgm = _parse_stat(stats, "fieldGoalsMade")
        fg3m, fg3a = _parse_made_attempted(stats, "threePointFieldGoalsMade-threePointFieldGoalsAttempted")
        if fg3a == 0:
            fg3a = _parse_stat(stats, "threePointFieldGoalsAttempted")
            fg3m = _parse_stat(stats, "threePointFieldGoalsMade")
        ftm, fta   = _parse_made_attempted(stats, "freeThrowsMade-freeThrowsAttempted")
        if fta == 0:
            fta = _parse_stat(stats, "freeThrowsAttempted")
            ftm = _parse_stat(stats, "freeThrowsMade")

        df.at[idx, "FGM"]    = fgm
        df.at[idx, "FGA"]    = fga
        df.at[idx, "FG_PCT"] = round(100 * fgm / fga, 1) if fga else 0
        df.at[idx, "FG3M"]   = fg3m
        df.at[idx, "FG3A"]   = fg3a
        df.at[idx, "FG3_PCT"]= round(100 * fg3m / fg3a, 1) if fg3a else 0
        df.at[idx, "FTM"]    = ftm
        df.at[idx, "FTA"]    = fta
        df.at[idx, "FT_PCT"] = round(100 * ftm / fta, 1) if fta else 0
        df.at[idx, "REB"]    = _parse_stat(stats, "rebounds", "totalRebounds")
        df.at[idx, "OREB"]   = _parse_stat(stats, "offensiveRebounds")
        df.at[idx, "DREB"]   = _parse_stat(stats, "defensiveRebounds")
        df.at[idx, "AST"]    = _parse_stat(stats, "assists")
        df.at[idx, "STL"]    = _parse_stat(stats, "steals")
        df.at[idx, "BLK"]    = _parse_stat(stats, "blocks")
        df.at[idx, "TOV"]    = _parse_stat(stats, "turnovers")
        df.at[idx, "PF"]     = _parse_stat(stats, "foulsPersonal", "personalFouls", "fouls")
        updated += 1

    console.print(f"[green]✓[/green] Filled stats for {updated} team-game rows")
    return df


async def fetch_all_seasons(season_years: list[int]) -> pd.DataFrame:
    """Pull all teams × all seasons (regular season + playoffs) concurrently."""
    all_rows = []
    # 2 seasontypes per season-year: regular + playoffs
    SEASONTYPES = [(2, "regular"), (3, "playoffs")]
    total = len(NBA_TEAMS) * len(season_years) * len(SEASONTYPES)

    async with httpx.AsyncClient(headers=HEADERS) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("Fetching schedules...", total=total)

            for season_year in season_years:
                for seasontype_id, seasontype_label in SEASONTYPES:
                    tasks = [
                        fetch_team_schedule(client, tid, season_year, seasontype_id)
                        for tid, _ in NBA_TEAMS
                    ]
                    results = await asyncio.gather(*tasks)

                    season_rows = 0
                    for rows in results:
                        all_rows.extend(rows)
                        season_rows += len(rows)
                        prog.advance(task)

                    console.print(
                        f"  [green]✓[/green] {season_year-1}-{str(season_year)[2:]} "
                        f"{seasontype_label}: {season_rows} game-team rows"
                    )

    if not all_rows:
        raise RuntimeError("No data pulled. Check internet connection.")

    df = pd.DataFrame(all_rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # Backfill box-score stats for games that came in without them (mainly playoffs)
    df = await backfill_box_scores(df)
    return df


# ---------------------------------------------------------------------------
# Also fetch regular season to get standings-style team stats
# ---------------------------------------------------------------------------

async def fetch_team_stats(season_years: list[int]) -> pd.DataFrame:
    """Pull team season totals from ESPN standings."""
    rows = []
    async with httpx.AsyncClient(headers=HEADERS) as client:
        for year in season_years:
            try:
                r = await client.get(
                    f"{ESPN_BASE}/standings",
                    params={"season": year},
                    timeout=15.0,
                )
                r.raise_for_status()
                data = r.json()
                for group in data.get("children", []):
                    for entry in group.get("standings", {}).get("entries", []):
                        team = entry.get("team", {})
                        stats = {s["name"]: s.get("value") for s in entry.get("stats", [])}
                        rows.append({
                            "TEAM_ID":    team.get("id"),
                            "TEAM":       team.get("abbreviation"),
                            "SEASON":     year,
                            **stats,
                        })
                console.print(f"  [green]✓[/green] Standings {year}: {len(rows)} teams")
            except Exception as e:
                console.print(f"  [yellow]Standings {year}: {e}[/yellow]")

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

async def update_game_logs() -> pd.DataFrame:
    out_path = DATA_DIR / "game_logs.parquet"
    if not out_path.exists():
        console.print("[yellow]No existing data — running full pull.[/yellow]")
        return await fetch_all_seasons(SEASON_YEARS)

    existing = pd.read_parquet(out_path)
    existing["GAME_DATE"] = pd.to_datetime(existing["GAME_DATE"])
    last_date = existing["GAME_DATE"].max()
    # Refresh the *current* season (regular + playoffs) — was previously
    # hardcoded to [2024] which is why playoff games never showed up.
    current = max(SEASON_YEARS)
    console.print(
        f"Existing data through {last_date.date()}. "
        f"Refreshing {current-1}-{str(current)[2:]} (regular + playoffs)..."
    )

    new_df = await fetch_all_seasons([current])
    new_df["GAME_DATE"] = pd.to_datetime(new_df["GAME_DATE"])
    new_rows = new_df[new_df["GAME_DATE"] > last_date]

    if len(new_rows):
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
        # Backfill any rows still missing box-score stats (existing playoff
        # games from before this fix shipped will be caught here).
        combined = await backfill_box_scores(combined)
        combined.to_parquet(out_path, index=False)
        console.print(f"[green]+{len(new_rows)} new rows → {len(combined):,} total[/green]")
        return combined

    # No new games — but existing data may still have playoff rows with empty
    # stats from earlier pipeline runs. Run backfill once anyway.
    existing = await backfill_box_scores(existing)
    existing.to_parquet(out_path, index=False)
    console.print("[dim]No new games. Backfill applied to existing data.[/dim]")
    return existing


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> None:
    console.print("\n[bold]Dataset Summary[/bold]")
    console.print(f"  Total rows   : {len(df):,}")
    console.print(f"  Unique games : {df['GAME_ID'].nunique():,}")
    console.print(f"  Seasons      : {sorted(df['SEASON'].unique())}")
    console.print(f"  Date range   : {df['GAME_DATE'].min().date()} → {df['GAME_DATE'].max().date()}")
    console.print(f"  Teams        : {df['TEAM_ABBREVIATION'].nunique()}")
    console.print(f"  Avg PTS/game : {df['PTS'].mean():.1f}")
    console.print(f"  Home win %   : {df[df['IS_HOME']]['WIN'].mean()*100:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="HoopIQ ESPN Data Pipeline")
    parser.add_argument("--update", action="store_true", help="Incremental update only")
    args = parser.parse_args()

    console.print("[bold orange1]HoopIQ Data Pipeline[/bold orange1] — ESPN API\n")
    console.print(f"[dim]Pulling seasons: {[f'{y-1}-{str(y)[2:]}' for y in SEASON_YEARS]}[/dim]\n")

    if args.update:
        df = await update_game_logs()
    else:
        df = await fetch_all_seasons(SEASON_YEARS)

        out = DATA_DIR / "game_logs.parquet"
        df.to_parquet(out, index=False)
        console.print(f"\n[bold green]Saved {len(df):,} rows → {out}[/bold green]")

        stats_df = await fetch_team_stats(SEASON_YEARS)
        if len(stats_df):
            stats_out = DATA_DIR / "team_stats.parquet"
            stats_df.to_parquet(stats_out, index=False)
            console.print(f"[bold green]Saved standings → {stats_out}[/bold green]")

    summarize(df)
    console.print("\n[bold green]Done! Run: python 3_feature_engineering.py[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
