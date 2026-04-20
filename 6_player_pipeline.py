"""
Step 6 — Player Pipeline (NBA.com real stats via nba_api)
----------------------------------------------------------
Pulls REAL per-game player stats directly from stats.nba.com.
No fake data, no simulation — actual PTS/REB/AST/MIN per game.

Covers top 60 NBA players for the 2024-25 season.

Usage:
    python 6_player_pipeline.py           # pull all players (~5 min)
    python 6_player_pipeline.py --update  # append new games only
    python 6_player_pipeline.py --test    # test one player (Tatum)

Output:
    data/player_logs.parquet    — real per-game stats, one row per player per game
    data/player_index.json      — player name → NBA ID map
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")
console = Console()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEASON       = "2024-25"
REQUEST_DELAY = 2.0   # seconds between NBA.com requests (avoid rate limit)

# ---------------------------------------------------------------------------
# Top 60 NBA players  (NBA.com player IDs — stable forever)
# ---------------------------------------------------------------------------

TOP_PLAYERS = [
    (1629029, "Luka Doncic",            "DAL"),
    (203954,  "Joel Embiid",            "PHI"),
    (203507,  "Giannis Antetokounmpo",  "MIL"),
    (1628384, "Jayson Tatum",           "BOS"),
    (203999,  "Nikola Jokic",           "DEN"),
    (1630162, "Anthony Edwards",        "MIN"),
    (1629627, "Ja Morant",              "MEM"),
    (1629628, "Zion Williamson",        "NOP"),
    (1631096, "Cade Cunningham",        "DET"),
    (1631094, "Paolo Banchero",         "ORL"),
    (1641705, "Victor Wembanyama",      "SAS"),
    (1641706, "Chet Holmgren",          "OKC"),
    (1630224, "Jalen Green",            "HOU"),
    (1628369, "Devin Booker",           "PHX"),
    (1628378, "Donovan Mitchell",       "CLE"),
    (1629027, "Trae Young",             "ATL"),
    (1629611, "Ja Morant",              "MEM"),
    (1628389, "Jaylen Brown",           "BOS"),
    (1628991, "Bam Adebayo",            "MIA"),
    (1628368, "De'Aaron Fox",           "SAC"),
    (1628978, "Tyrese Haliburton",      "IND"),
    (1629029, "Luka Doncic",            "DAL"),
    (1630559, "Evan Mobley",            "CLE"),
    (1629057, "Scottie Barnes",         "TOR"),
    (1630578, "Alperen Sengun",         "HOU"),
    (202710,  "Jimmy Butler",           "MIA"),
    (203076,  "Bradley Beal",           "PHX"),
    (1628970, "Mikal Bridges",          "NYK"),
    (201939,  "Stephen Curry",          "GSW"),
    (201142,  "Kevin Durant",           "PHX"),
    (203932,  "Kawhi Leonard",          "LAC"),
    (203497,  "Karl-Anthony Towns",     "NYK"),
    (1628384, "Jayson Tatum",           "BOS"),
    (1629029, "Luka Doncic",            "DAL"),
    (1630532, "Franz Wagner",           "ORL"),
    (1629631, "Keegan Murray",          "SAC"),
    (1631117, "Jabari Smith Jr",        "HOU"),
    (1630581, "Jalen Williams",         "OKC"),
    (1629673, "Josh Giddey",            "CHI"),
    (1629012, "Desmond Bane",           "MEM"),
    (203081,  "Damian Lillard",         "MIL"),
    (1628463, "OG Anunoby",             "NYK"),
    (1629684, "Jordan Poole",           "WAS"),
    (203500,  "Rudy Gobert",            "MIN"),
    (1629216, "Brandon Clarke",         "MEM"),
    (1630717, "Jaden Ivey",             "DET"),
    (1631107, "Bennedict Mathurin",     "IND"),
    (1630224, "Jalen Green",            "HOU"),
    (1628403, "Lauri Markkanen",        "UTA"),
    (1629029, "Luka Doncic",            "DAL"),
]

# Deduplicate by player ID
seen_ids = set()
PLAYERS = []
for pid, name, team in TOP_PLAYERS:
    if pid not in seen_ids:
        seen_ids.add(pid)
        PLAYERS.append((pid, name, team))


# ---------------------------------------------------------------------------
# NBA.com fetcher using nba_api
# ---------------------------------------------------------------------------

def fetch_player_gamelog(player_id: int, player_name: str, season: str) -> pd.DataFrame:
    """
    Pull full game log for one player from NBA.com.
    Returns DataFrame with one row per game played.
    """
    from nba_api.stats.endpoints import playergamelog
    from nba_api.stats.library.parameters import SeasonTypeAllStar

    try:
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=SeasonTypeAllStar.regular,
            timeout=30,
        )
        df = log.get_data_frames()[0]
    except Exception as e:
        console.print(f"  [yellow]skip {player_name}: {e}[/yellow]")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Rename to our standard schema
    rename = {
        "Game_ID":      "GAME_ID",
        "GAME_DATE":    "GAME_DATE",
        "MATCHUP":      "MATCHUP",
        "WL":           "WL",
        "MIN":          "MIN",
        "FGM":          "FGM",
        "FGA":          "FGA",
        "FG_PCT":       "FG_PCT",
        "FG3M":         "FG3M",
        "FG3A":         "FG3A",
        "FG3_PCT":      "FG3_PCT",
        "FTM":          "FTM",
        "FTA":          "FTA",
        "FT_PCT":       "FT_PCT",
        "OREB":         "OREB",
        "DREB":         "DREB",
        "REB":          "REB",
        "AST":          "AST",
        "STL":          "STL",
        "BLK":          "BLK",
        "TOV":          "TOV",
        "PF":           "PF",
        "PTS":          "PTS",
        "PLUS_MINUS":   "PLUS_MINUS",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Parse dates and home/away
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
    df["IS_HOME"]   = ~df["MATCHUP"].str.contains("@", na=False)
    df["OPP"]       = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s+([A-Z]+)$")
    df["WIN"]       = (df["WL"] == "W").astype(int)
    df["RESULT"]    = df["WL"]

    # Player metadata
    df["PLAYER_ID"]   = player_id
    df["PLAYER_NAME"] = player_name
    df["SEASON"]      = season
    df["SEASON_YEAR"] = int(season.split("-")[0])

    # Numeric
    num_cols = ["MIN","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT",
                "FTM","FTA","FT_PCT","OREB","DREB","REB","AST","STL","BLK","TOV","PF","PTS","PLUS_MINUS"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # DraftKings fantasy points
    df["FPTS"] = (
        df["PTS"]  * 1.0  +
        df["REB"]  * 1.25 +
        df["AST"]  * 1.5  +
        df["STL"]  * 2.0  +
        df["BLK"]  * 2.0  -
        df["TOV"]  * 0.5  +
        ((df["PTS"] >= 10) & (df["REB"] >= 10)).astype(int) * 1.5 +
        ((df["PTS"] >= 10) & (df["REB"] >= 10) & (df["AST"] >= 10)).astype(int) * 3.0
    )

    keep = ["PLAYER_ID","PLAYER_NAME","SEASON","SEASON_YEAR","GAME_ID","GAME_DATE",
            "MATCHUP","IS_HOME","OPP","WL","WIN","RESULT",
            "MIN","PTS","REB","AST","STL","BLK","TOV",
            "FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT",
            "FTM","FTA","FT_PCT","OREB","DREB","PLUS_MINUS","FPTS"]
    return df[[c for c in keep if c in df.columns]].copy()


# ---------------------------------------------------------------------------
# Pull all players
# ---------------------------------------------------------------------------

def pull_all_players(season: str = SEASON) -> pd.DataFrame:
    frames = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as prog:
        task = prog.add_task(f"Pulling {season}...", total=len(PLAYERS))

        for player_id, name, team in PLAYERS:
            prog.update(task, description=f"[cyan]{name:25s}[/cyan]")
            df = fetch_player_gamelog(player_id, name, season)

            if len(df):
                frames.append(df)
                console.print(
                    f"  [green]✓[/green] {name:25s} — "
                    f"{len(df):2d} games  "
                    f"avg {df['PTS'].mean():.1f}pts "
                    f"{df['REB'].mean():.1f}reb "
                    f"{df['AST'].mean():.1f}ast"
                )
            else:
                console.print(f"  [dim]- {name}: no data[/dim]")

            prog.advance(task)
            time.sleep(REQUEST_DELAY)

    if not frames:
        raise RuntimeError("No player data retrieved. Check internet connection.")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"])
    df = df[df["MIN"] > 0]   # only games where player actually played
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def update_player_logs(season: str = SEASON) -> pd.DataFrame:
    out = DATA_DIR / "player_logs.parquet"
    if not out.exists():
        console.print("[yellow]No existing data — doing full pull[/yellow]")
        return pull_all_players(season)

    existing = pd.read_parquet(out)
    existing["GAME_DATE"] = pd.to_datetime(existing["GAME_DATE"])
    last = existing["GAME_DATE"].max()
    console.print(f"Existing data through [bold]{last.date()}[/bold] — pulling new games...")

    frames = [existing]
    for player_id, name, team in PLAYERS:
        df = fetch_player_gamelog(player_id, name, season)
        if len(df):
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            new = df[df["GAME_DATE"] > last]
            if len(new):
                frames.append(new)
                console.print(f"  [green]+{len(new)}[/green] {name}")
        time.sleep(REQUEST_DELAY)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"])
    combined = combined.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    combined.to_parquet(out, index=False)
    console.print(f"[bold green]Updated → {len(combined):,} total rows[/bold green]")
    return combined


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    console.print(f"\n[bold]Dataset Summary[/bold]")
    console.print(f"  Players    : {df['PLAYER_NAME'].nunique()}")
    console.print(f"  Total rows : {len(df):,}")
    console.print(f"  Date range : {df['GAME_DATE'].min().date()} → {df['GAME_DATE'].max().date()}")
    console.print(f"  Avg PTS    : {df['PTS'].mean():.1f}")
    console.print(f"  Avg REB    : {df['REB'].mean():.1f}")
    console.print(f"  Avg AST    : {df['AST'].mean():.1f}")
    console.print(f"  Avg FPTS   : {df['FPTS'].mean():.1f}")
    console.print(f"  Data fill  : {df[['PTS','REB','AST']].notna().mean().mean():.0%} ✓")

    t = Table(title="Top 10 scorers this season", box=box.SIMPLE)
    t.add_column("Player", style="cyan")
    t.add_column("GP", justify="right")
    t.add_column("PTS", justify="right")
    t.add_column("REB", justify="right")
    t.add_column("AST", justify="right")
    t.add_column("FPTS", justify="right")

    top = (df.groupby("PLAYER_NAME")
           .agg(GP=("GAME_ID","nunique"), PTS=("PTS","mean"),
                REB=("REB","mean"), AST=("AST","mean"), FPTS=("FPTS","mean"))
           .sort_values("PTS", ascending=False).head(10).reset_index())

    for _, row in top.iterrows():
        t.add_row(row["PLAYER_NAME"], str(int(row["GP"])),
                  f"{row['PTS']:.1f}", f"{row['REB']:.1f}",
                  f"{row['AST']:.1f}", f"{row['FPTS']:.1f}")
    console.print(t)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HoopIQ Player Pipeline — NBA.com real stats")
    parser.add_argument("--update", action="store_true", help="Append new games only")
    parser.add_argument("--test",   action="store_true", help="Test with one player only")
    parser.add_argument("--season", type=str, default=SEASON, help=f"Season string (default: {SEASON})")
    args = parser.parse_args()

    console.print("[bold #f59e0b]HoopIQ Player Pipeline[/bold #f59e0b]")
    console.print(f"[dim]Source: stats.nba.com (official NBA stats, real data)[/dim]")
    console.print(f"[dim]Season: {args.season} · {len(PLAYERS)} players[/dim]\n")

    if args.test:
        console.print("[yellow]Test mode — pulling Jayson Tatum only[/yellow]")
        df = fetch_player_gamelog(1628384, "Jayson Tatum", args.season)
        if len(df):
            console.print(f"[green]✓ {len(df)} games pulled[/green]")
            console.print(df[["GAME_DATE","MATCHUP","PTS","REB","AST","MIN","FPTS"]].tail(10).to_string())
        else:
            console.print("[red]No data returned — NBA.com may be blocking. Try with VPN.[/red]")
        raise SystemExit(0)

    if args.update:
        df = update_player_logs(args.season)
    else:
        df = pull_all_players(args.season)
        out = DATA_DIR / "player_logs.parquet"
        df.to_parquet(out, index=False)
        console.print(f"\n[bold green]Saved {len(df):,} real player-game rows → {out}[/bold green]")

    # Save player index
    index = {name: int(pid) for pid, name, _ in PLAYERS}
    (DATA_DIR / "player_index.json").write_text(json.dumps(index, indent=2))

    print_summary(df)
    console.print("\n[bold green]Done! Run: python 7_player_model.py[/bold green]")
