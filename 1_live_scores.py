"""
Step 1 — Live Scores via ESPN Public API
-----------------------------------------
Polls ESPN every 30s during live games, returns structured game objects.
No API key required. Safe to call from frontend or backend.

Usage:
    python 1_live_scores.py               # one-time fetch, pretty print
    python 1_live_scores.py --watch       # continuous poll loop
    python 1_live_scores.py --json        # output raw JSON
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Optional
import httpx
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

HEADERS = {
    "User-Agent": "HoopIQ/1.0",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# Data models (plain dicts — easy to JSON-serialize)
# ---------------------------------------------------------------------------

def parse_competitor(comp: dict) -> dict:
    team = comp.get("team", {})
    records = comp.get("records", [{}])
    record_str = records[0].get("summary", "0-0") if records else "0-0"
    wins, losses = (record_str.split("-") + ["0"])[:2]
    return {
        "id": team.get("id"),
        "abbr": team.get("abbreviation", "UNK"),
        "name": team.get("displayName", "Unknown"),
        "score": int(comp.get("score", 0)),
        "home_away": comp.get("homeAway", "home"),
        "wins": int(wins),
        "losses": int(losses),
        "record": record_str,
        "winner": comp.get("winner", False),
    }


def parse_game(event: dict) -> dict:
    comps = event.get("competitions", [{}])[0]
    competitors = comps.get("competitors", [])

    home = next((parse_competitor(c) for c in competitors if c.get("homeAway") == "home"), {})
    away = next((parse_competitor(c) for c in competitors if c.get("homeAway") == "away"), {})

    status = comps.get("status", {})
    status_type = status.get("type", {})
    situation = comps.get("situation", {})

    odds_list = comps.get("odds", [])
    odds = odds_list[0] if odds_list else {}

    return {
        "game_id": event.get("id"),
        "name": event.get("name", ""),
        "date": event.get("date"),
        "home": home,
        "away": away,
        "status": {
            "state": status_type.get("state", "pre"),       # pre | in | post
            "description": status_type.get("description"),
            "detail": status.get("displayClock", "0:00"),
            "period": status.get("period", 0),
            "completed": status_type.get("completed", False),
        },
        "situation": {
            "possession": situation.get("possession"),      # team id with ball
            "last_play": situation.get("lastPlay", {}).get("text", ""),
            "down_distance": situation.get("downDistanceText", ""),
        },
        "odds": {
            "spread": odds.get("details", ""),
            "over_under": odds.get("overUnder", 0.0),
            "home_moneyline": odds.get("homeTeamOdds", {}).get("moneyLine", None),
            "away_moneyline": odds.get("awayTeamOdds", {}).get("moneyLine", None),
        },
        "venue": comps.get("venue", {}).get("fullName", ""),
        "attendance": comps.get("attendance", 0),
    }


# ---------------------------------------------------------------------------
# API fetchers
# ---------------------------------------------------------------------------

async def fetch_scoreboard(date: Optional[str] = None) -> list[dict]:
    """Return all games for a given date (YYYYMMDD) or today if None."""
    params = {}
    if date:
        params["dates"] = date

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(ESPN_SCOREBOARD, params=params, headers=HEADERS)
        r.raise_for_status()
        data = r.json()

    events = data.get("events", [])
    return [parse_game(e) for e in events]


async def fetch_game_detail(game_id: str) -> dict:
    """Return full box score + play-by-play for a specific game."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(ESPN_SUMMARY, params={"event": game_id}, headers=HEADERS)
        r.raise_for_status()
        raw = r.json()

    plays_raw = raw.get("plays", [])
    plays = [
        {
            "period": p.get("period", {}).get("number"),
            "clock": p.get("clock", {}).get("displayValue"),
            "type": p.get("type", {}).get("text"),
            "text": p.get("text"),
            "score_home": p.get("homeScore"),
            "score_away": p.get("awayScore"),
        }
        for p in plays_raw[-50:]  # last 50 plays
    ]

    boxscore = raw.get("boxscore", {})
    players_raw = boxscore.get("players", [])
    teams_stats = []
    for team_block in players_raw:
        team_info = team_block.get("team", {})
        stats_list = team_block.get("statistics", [])
        athletes = []
        for stat_group in stats_list:
            for athlete in stat_group.get("athletes", []):
                a = athlete.get("athlete", {})
                stats = athlete.get("stats", [])
                stat_labels = stat_group.get("labels", [])
                stat_map = dict(zip(stat_labels, stats))
                athletes.append({
                    "name": a.get("displayName"),
                    "position": a.get("position", {}).get("abbreviation"),
                    "starter": athlete.get("starter", False),
                    "active": athlete.get("active", True),
                    "stats": stat_map,
                })
        teams_stats.append({
            "team": team_info.get("abbreviation"),
            "athletes": athletes,
        })

    return {
        "game_id": game_id,
        "plays": plays,
        "box_score": teams_stats,
    }


# ---------------------------------------------------------------------------
# Live game filter helpers
# ---------------------------------------------------------------------------

def live_games(games: list[dict]) -> list[dict]:
    return [g for g in games if g["status"]["state"] == "in"]


def completed_games(games: list[dict]) -> list[dict]:
    return [g for g in games if g["status"]["state"] == "post"]


def upcoming_games(games: list[dict]) -> list[dict]:
    return [g for g in games if g["status"]["state"] == "pre"]


# ---------------------------------------------------------------------------
# Rich pretty-print
# ---------------------------------------------------------------------------

def print_scoreboard(games: list[dict]) -> None:
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    console.rule(f"[bold orange1]🏀 HoopIQ Live Scoreboard[/bold orange1]  [dim]{now}[/dim]")

    if not games:
        console.print("[dim]No games found.[/dim]")
        return

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold dim")
    table.add_column("Away", style="cyan", width=22)
    table.add_column("Score", justify="center", width=10)
    table.add_column("Home", style="cyan", width=22)
    table.add_column("Status", justify="center", width=18)
    table.add_column("Spread", justify="center", width=12)
    table.add_column("O/U", justify="center", width=8)

    state_color = {"pre": "dim", "in": "bold green", "post": "dim"}

    for g in games:
        st = g["status"]
        odds = g["odds"]
        state = st["state"]
        color = state_color.get(state, "white")

        if state == "in":
            status_str = f"Q{st['period']} {st['detail']}"
        elif state == "post":
            status_str = "Final"
        else:
            date_str = g["date"][11:16] if g["date"] else "TBD"
            status_str = f"{date_str} ET"

        score_str = f"{g['away']['score']} - {g['home']['score']}" if state != "pre" else "vs"

        table.add_row(
            f"{g['away']['abbr']}  {g['away']['record']}",
            f"[{color}]{score_str}[/{color}]",
            f"{g['home']['abbr']}  {g['home']['record']}",
            f"[{color}]{status_str}[/{color}]",
            odds["spread"] or "—",
            str(odds["over_under"]) if odds["over_under"] else "—",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Poll loop
# ---------------------------------------------------------------------------

async def watch(interval: int = 30) -> None:
    console.print(f"[dim]Polling every {interval}s. Ctrl+C to stop.[/dim]\n")
    while True:
        try:
            games = await fetch_scoreboard()
            print_scoreboard(games)
            live = live_games(games)
            if live:
                console.print(f"[green]{len(live)} game(s) live[/green]")
            await asyncio.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped.[/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="HoopIQ Live Scores")
    parser.add_argument("--watch", action="store_true", help="Continuous poll")
    parser.add_argument("--json",  action="store_true", help="Raw JSON output")
    parser.add_argument("--date",  type=str, default=None, help="Date YYYYMMDD")
    parser.add_argument("--game",  type=str, default=None, help="Game ID for detail")
    args = parser.parse_args()

    if args.game:
        detail = await fetch_game_detail(args.game)
        print(json.dumps(detail, indent=2))
        return

    if args.watch:
        await watch()
        return

    games = await fetch_scoreboard(args.date)

    if args.json:
        print(json.dumps(games, indent=2))
    else:
        print_scoreboard(games)


if __name__ == "__main__":
    asyncio.run(main())
