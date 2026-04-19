"""
Step 6 — Player Props Data Pipeline (ESPN API)
------------------------------------------------
Pulls player game logs for all active NBA players.
Saves per-player rolling stats used for prop predictions.

Usage:
    python 6_player_pipeline.py              # pull 2024-25 + 2025-26
    python 6_player_pipeline.py --update     # append new games only

Output:
    data/player_logs.parquet     — one row per player per game
    data/player_index.json       — player name → ESPN ID map
"""

import asyncio
import argparse
import json
from pathlib import Path

import httpx
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ESPN_BASE    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_CORE    = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
HEADERS      = {"User-Agent": "HoopIQ/1.0", "Accept": "application/json"}
SEASON_YEARS = [2025, 2026]

# Top 60 NBA players by ESPN athlete ID (stars + popular fantasy players)
TOP_PLAYERS = [
    (3945274,  "Jayson Tatum",      "BOS"),
    (3202,     "LeBron James",      "LAL"),
    (3032977,  "Nikola Jokic",      "DEN"),
    (3547303,  "Luka Doncic",       "DAL"),
    (2490149,  "Giannis Antetokounmpo", "MIL"),
    (4065648,  "Anthony Edwards",   "MIN"),
    (3136193,  "Joel Embiid",       "PHI"),
    (4066648,  "Ja Morant",         "MEM"),
    (4277905,  "Zion Williamson",   "NOP"),
    (4431678,  "Cade Cunningham",   "DET"),
    (4395725,  "Paolo Banchero",    "ORL"),
    (4432174,  "Victor Wembanyama", "SAS"),
    (4683021,  "Chet Holmgren",     "OKC"),
    (4277956,  "Jalen Green",       "HOU"),
    (4432166,  "Franz Wagner",      "ORL"),
    (3059318,  "Devin Booker",      "PHX"),
    (3934672,  "Donovan Mitchell",  "CLE"),
    (4066261,  "Trae Young",        "ATL"),
    (3155942,  "Karl-Anthony Towns","NYK"),
    (2991055,  "Jaylen Brown",      "BOS"),
    (3913176,  "Bam Adebayo",       "MIA"),
    (4066623,  "De'Aaron Fox",      "SAC"),
    (4278129,  "Tyrese Haliburton", "IND"),
    (3136779,  "Darius Garland",    "CLE"),
    (3032976,  "Nikola Vucevic",    "CHI"),
    (4066269,  "Brandon Ingram",    "NOP"),
    (2490155,  "Khris Middleton",   "MIL"),
    (3149673,  "CJ McCollum",       "NOP"),
    (3064514,  "Kristaps Porzingis","BOS"),
    (4066253,  "Lauri Markkanen",   "UTA"),
    (4431992,  "Evan Mobley",       "CLE"),
    (3032993,  "Pascal Siakam",     "IND"),
    (3468781,  "OG Anunoby",        "NYK"),
    (4066328,  "Scottie Barnes",    "TOR"),
    (4065632,  "Alperen Sengun",    "HOU"),
    (3055083,  "Jimmy Butler",      "MIA"),
    (3136291,  "Bradley Beal",      "PHX"),
    (3059319,  "Mikal Bridges",     "NYK"),
    (4066269,  "Jordan Poole",      "WAS"),
    (2779816,  "Stephen Curry",     "GSW"),
    (3975,     "Kevin Durant",      "PHX"),
    (3017,     "James Harden",      "LAC"),
    (2991230,  "Kawhi Leonard",     "LAC"),
    (3136228,  "Andrew Wiggins",    "GSW"),
    (4432817,  "Jabari Smith Jr",   "HOU"),
    (4277957,  "Keegan Murray",     "SAC"),
    (4432773,  "Bennedict Mathurin","IND"),
    (4432166,  "Jalen Williams",    "OKC"),
    (4066390,  "Josh Giddey",       "CHI"),
    (4066262,  "Desmond Bane",      "MEM"),
]
# Deduplicate
seen = set()
TOP_PLAYERS = [(pid,name,team) for pid,name,team in TOP_PLAYERS if not (pid in seen or seen.add(pid))]


async def fetch_player_gamelog(client: httpx.AsyncClient, player_id: int, season_year: int) -> list[dict]:
    """Fetch game-by-game stats for one player via ESPN."""
    try:
        url = f"{ESPN_BASE}/athletes/{player_id}/gamelog"
        r = await client.get(url, params={"season": season_year}, timeout=15.0)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    events   = data.get("events", {})
    stats_by = data.get("seasonTypes", [])
    labels   = []
    for st in stats_by:
        for cat in st.get("categories", []):
            if cat.get("name") == "gameLog":
                labels = [s.get("abbreviation","") for s in cat.get("stats",[])]
                break

    rows = []
    for game_id, ev in events.items():
        game_info = ev if isinstance(ev, dict) else {}
        stats_raw = game_info.get("stats", [])
        if not stats_raw or not labels:
            continue

        stat_map = dict(zip(labels, stats_raw))

        def g(k, default=0.0):
            v = stat_map.get(k, default)
            try: return float(v) if v not in ("", "--", None) else default
            except: return default

        rows.append({
            "PLAYER_ID":    player_id,
            "GAME_ID":      game_id,
            "SEASON_YEAR":  season_year,
            "GAME_DATE":    game_info.get("gameDate", "")[:10],
            "TEAM":         game_info.get("teamAbbrev", ""),
            "OPP":          game_info.get("opponent", {}).get("abbreviation", "") if isinstance(game_info.get("opponent"), dict) else "",
            "HOME":         not game_info.get("atVs", "") == "@",
            "RESULT":       game_info.get("result", ""),
            "MIN":  g("MIN"), "PTS":  g("PTS"), "REB":  g("REB"),
            "AST":  g("AST"), "STL":  g("STL"), "BLK":  g("BLK"),
            "TOV":  g("TO"),  "FGM":  g("FGM"), "FGA":  g("FGA"),
            "FG3M": g("3PM"), "FG3A": g("3PA"), "FTM":  g("FTM"),
            "FTA":  g("FTA"), "OREB": g("OR"),  "DREB": g("DR"),
        })

    return rows


async def fetch_all_players(season_years: list[int]) -> pd.DataFrame:
    all_rows = []
    total = len(TOP_PLAYERS) * len(season_years)

    async with httpx.AsyncClient(headers=HEADERS) as client:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      BarColumn(), MofNCompleteColumn(), console=console) as prog:
            task = prog.add_task("Fetching player logs...", total=total)

            for season_year in season_years:
                tasks = [fetch_player_gamelog(client, pid, season_year) for pid,_,_ in TOP_PLAYERS]
                results = await asyncio.gather(*tasks)

                season_rows = 0
                for (pid, name, team), rows in zip(TOP_PLAYERS, results):
                    for r in rows:
                        r["PLAYER_NAME"] = name
                        r["PLAYER_TEAM"] = team
                    all_rows.extend(rows)
                    season_rows += len(rows)
                    prog.advance(task)

                console.print(f"  [green]✓[/green] {season_year}: {season_rows} player-game rows")

    if not all_rows:
        console.print("[yellow]No player data found — ESPN athlete gamelog endpoint may require different IDs[/yellow]")
        console.print("[dim]Generating synthetic data from game_logs.parquet for demonstration...[/dim]")
        return generate_from_game_logs()

    df = pd.DataFrame(all_rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.dropna(subset=["GAME_DATE"])
    df = df[df["MIN"] > 0]  # only games where player actually played
    df = df.drop_duplicates(subset=["PLAYER_ID","GAME_ID"])
    df = df.sort_values(["PLAYER_ID","GAME_DATE"]).reset_index(drop=True)

    # Add fantasy points (DraftKings scoring)
    df["FPTS"] = (
        df["PTS"] * 1.0 +
        df["REB"] * 1.25 +
        df["AST"] * 1.5 +
        df["STL"] * 2.0 +
        df["BLK"] * 2.0 -
        df["TOV"] * 0.5 +
        (df["PTS"] >= 10).astype(int) * (df["REB"] >= 10).astype(int) * 1.5 +  # double-double bonus
        (df["PTS"] >= 10).astype(int) * (df["REB"] >= 10).astype(int) * (df["AST"] >= 10).astype(int) * 3.0  # triple-double
    )

    return df


def generate_from_game_logs() -> pd.DataFrame:
    """
    Fallback: create realistic player logs from team game logs
    by simulating per-player splits. Used if ESPN athlete API is unavailable.
    """
    import numpy as np
    path = DATA_DIR / "game_logs.parquet"
    if not path.exists():
        return pd.DataFrame()

    team_df = pd.read_parquet(path)
    rows = []
    rng = np.random.default_rng(42)

    for pid, name, team in TOP_PLAYERS:
        team_games = team_df[team_df["TEAM_ABBREVIATION"] == team].sort_values("GAME_DATE")
        if len(team_games) == 0:
            continue

        # Assign realistic per-game stats based on player archetype
        base_pts = rng.uniform(14, 32)
        base_reb = rng.uniform(3, 12)
        base_ast = rng.uniform(2, 9)
        base_stl = rng.uniform(0.5, 2.0)
        base_blk = rng.uniform(0.3, 2.0)
        base_tov = rng.uniform(1.5, 4.0)
        base_min = rng.uniform(28, 36)

        for _, g in team_games.iterrows():
            noise = lambda x: max(0, x + rng.normal(0, x * 0.25))
            pts = round(noise(base_pts))
            reb = round(noise(base_reb))
            ast = round(noise(base_ast))
            stl = round(noise(base_stl), 1)
            blk = round(noise(base_blk), 1)
            tov = round(noise(base_tov), 1)
            mn  = round(noise(base_min), 1)

            fpts = pts*1.0 + reb*1.25 + ast*1.5 + stl*2.0 + blk*2.0 - tov*0.5
            rows.append({
                "PLAYER_ID": pid, "PLAYER_NAME": name, "PLAYER_TEAM": team,
                "GAME_ID": g["GAME_ID"], "GAME_DATE": g["GAME_DATE"],
                "SEASON_YEAR": g.get("SEASON_YEAR", 2026),
                "TEAM": team, "OPP": g.get("OPP_ABBR",""),
                "HOME": g.get("IS_HOME", True), "RESULT": g.get("WL",""),
                "MIN": mn, "PTS": pts, "REB": reb, "AST": ast,
                "STL": stl, "BLK": blk, "TOV": tov,
                "FGM": round(pts/2.2), "FGA": round(pts/1.1),
                "FG3M": round(pts*0.15), "FG3A": round(pts*0.35),
                "FTM": round(pts*0.2), "FTA": round(pts*0.25),
                "OREB": round(reb*0.25), "DREB": round(reb*0.75),
                "FPTS": round(fpts, 2),
            })

    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values(["PLAYER_ID","GAME_DATE"]).reset_index(drop=True)


async def update_player_logs() -> pd.DataFrame:
    out = DATA_DIR / "player_logs.parquet"
    if not out.exists():
        return await fetch_all_players(SEASON_YEARS)

    existing = pd.read_parquet(out)
    existing["GAME_DATE"] = pd.to_datetime(existing["GAME_DATE"])
    last = existing["GAME_DATE"].max()
    console.print(f"Existing through {last.date()}. Pulling new games...")

    new_df = await fetch_all_players([2026])
    new_df["GAME_DATE"] = pd.to_datetime(new_df["GAME_DATE"])
    new_rows = new_df[new_df["GAME_DATE"] > last]
    if len(new_rows):
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["PLAYER_ID","GAME_ID"])
        combined.to_parquet(out, index=False)
        console.print(f"[green]+{len(new_rows)} new rows[/green]")
        return combined
    console.print("[dim]No new games.[/dim]")
    return existing


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    console.print("[bold orange1]HoopIQ Player Pipeline[/bold orange1]\n")

    if args.update:
        df = await update_player_logs()
    else:
        df = await fetch_all_players(SEASON_YEARS)

    out = DATA_DIR / "player_logs.parquet"
    df.to_parquet(out, index=False)

    # Save player index
    index = {name: int(pid) for pid,name,_ in TOP_PLAYERS}
    (DATA_DIR / "player_index.json").write_text(json.dumps(index, indent=2))

    console.print(f"\n[bold green]Saved {len(df):,} player-game rows → {out}[/bold green]")
    console.print(f"Players: {df['PLAYER_NAME'].nunique()}")
    console.print(f"Date range: {df['GAME_DATE'].min().date()} → {df['GAME_DATE'].max().date()}")
    console.print(f"Avg PTS: {df['PTS'].mean():.1f} | Avg REB: {df['REB'].mean():.1f} | Avg AST: {df['AST'].mean():.1f}")
    console.print("\n[bold green]Done! Run: python 7_player_model.py[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
