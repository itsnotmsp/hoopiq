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
SEASON_YEARS = [2023, 2024, 2025, 2026]

ESPN_BASE    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
TEAMS_URL    = f"{ESPN_BASE}/teams"
SCHEDULE_URL = f"{ESPN_BASE}/teams/{{team_id}}/schedule"

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

async def fetch_team_schedule(client: httpx.AsyncClient, team_id: int, season_year: int) -> list[dict]:
    """Fetch all games for one team in one season."""
    try:
        r = await client.get(
            SCHEDULE_URL.format(team_id=team_id),
            params={"season": season_year, "seasontype": 2},  # 2=regular, 3=playoffs
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


async def fetch_all_seasons(season_years: list[int]) -> pd.DataFrame:
    """Pull all teams × all seasons concurrently (ESPN handles the load fine)."""
    all_rows = []
    total = len(NBA_TEAMS) * len(season_years)

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
                # Fetch all 30 teams concurrently for this season
                tasks = [
                    fetch_team_schedule(client, tid, season_year)
                    for tid, _ in NBA_TEAMS
                ]
                results = await asyncio.gather(*tasks)

                season_rows = 0
                for rows in results:
                    all_rows.extend(rows)
                    season_rows += len(rows)
                    prog.advance(task)

                console.print(f"  [green]✓[/green] {season_year-1}-{str(season_year)[2:]} season: {season_rows} game-team rows")

    if not all_rows:
        raise RuntimeError("No data pulled. Check internet connection.")

    df = pd.DataFrame(all_rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
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
    console.print(f"Existing data through {last_date.date()}. Fetching 2024 updates...")

    new_df = await fetch_all_seasons([2024])
    new_df["GAME_DATE"] = pd.to_datetime(new_df["GAME_DATE"])
    new_rows = new_df[new_df["GAME_DATE"] > last_date]

    if len(new_rows):
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
        combined.to_parquet(out_path, index=False)
        console.print(f"[green]+{len(new_rows)} new rows → {len(combined):,} total[/green]")
        return combined

    console.print("[dim]No new games found.[/dim]")
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
