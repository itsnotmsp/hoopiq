"""
Step 9 — Real Odds Pipeline (The Odds API)
-------------------------------------------
Pulls live NBA odds from 70+ bookmakers including DraftKings, FanDuel, BetMGM.
Covers: moneyline, spread, totals, player props (PTS/REB/AST over/under).

Usage:
    python 9_odds_pipeline.py              # fetch today's odds + props
    python 9_odds_pipeline.py --games      # game odds only
    python 9_odds_pipeline.py --props      # player props only
    python 9_odds_pipeline.py --live       # live in-game odds

Output:
    data/odds_games.json     — moneyline, spread, totals per game
    data/odds_props.json     — player prop lines per game
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

API_KEY   = "6098fee47d0139939b30ddaad819fbf9"
BASE      = "https://api.the-odds-api.com/v4"
SPORT     = "basketball_nba"
REGIONS   = "us"
ODDS_FMT  = "american"

# Best books to pull from (prioritize sharp/liquid books)
PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet", "bovada"]

HEADERS = {"User-Agent": "HoopIQ/1.0"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(endpoint: str, params: dict = {}) -> dict:
    url = f"{BASE}/{endpoint}"
    params["apiKey"] = API_KEY
    r = SESSION.get(url, params=params, timeout=15)
    remaining = r.headers.get("x-requests-remaining", "?")
    used = r.headers.get("x-requests-used", "?")
    console.print(f"  [dim]API quota: {remaining} remaining ({used} used)[/dim]")
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Fetch game odds (moneyline + spread + totals)
# ---------------------------------------------------------------------------

def fetch_game_odds() -> list[dict]:
    console.print("[cyan]Fetching game odds (moneyline, spread, totals)...[/cyan]")
    data = api_get(f"sports/{SPORT}/odds", {
        "regions": REGIONS,
        "markets": "h2h,spreads,totals",
        "oddsFormat": ODDS_FMT,
        "bookmakers": ",".join(PREFERRED_BOOKS),
    })

    games = []
    for ev in data:
        game = {
            "id":           ev["id"],
            "home_team":    ev["home_team"],
            "away_team":    ev["away_team"],
            "commence_time": ev["commence_time"],
            "moneyline":    {},
            "spread":       {},
            "total":        {},
            "bookmakers":   [],
        }

        for book in ev.get("bookmakers", []):
            bkey = book["key"]
            game["bookmakers"].append(bkey)
            for market in book.get("markets", []):
                mkey = market["key"]
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}

                if mkey == "h2h":
                    home_price = outcomes.get(ev["home_team"], {}).get("price")
                    away_price = outcomes.get(ev["away_team"], {}).get("price")
                    if home_price and bkey not in game["moneyline"]:
                        game["moneyline"] = {
                            "home": home_price,
                            "away": away_price,
                            "book": bkey,
                        }

                elif mkey == "spreads":
                    home_out = outcomes.get(ev["home_team"], {})
                    if home_out and bkey not in game["spread"]:
                        game["spread"] = {
                            "home_line": home_out.get("point"),
                            "home_price": home_out.get("price"),
                            "away_line": outcomes.get(ev["away_team"], {}).get("point"),
                            "away_price": outcomes.get(ev["away_team"], {}).get("price"),
                            "book": bkey,
                        }

                elif mkey == "totals":
                    over = outcomes.get("Over", {})
                    if over and "total" not in game.get("total", {}):
                        game["total"] = {
                            "line": over.get("point"),
                            "over_price": over.get("price"),
                            "under_price": outcomes.get("Under", {}).get("price"),
                            "book": bkey,
                        }

        games.append(game)
        console.print(
            f"  [green]✓[/green] {ev['away_team']} @ {ev['home_team']} — "
            f"ML: {game['moneyline'].get('home','?')}/{game['moneyline'].get('away','?')} | "
            f"Spread: {game['spread'].get('home_line','?')} | "
            f"O/U: {game['total'].get('line','?')}"
        )

    out = DATA_DIR / "odds_games.json"
    out.write_text(json.dumps(games, indent=2))
    console.print(f"\n[bold green]Saved {len(games)} game odds → {out}[/bold green]")
    return games


# ---------------------------------------------------------------------------
# Fetch player props (per event)
# ---------------------------------------------------------------------------

def fetch_player_props(games: list[dict]) -> list[dict]:
    console.print("\n[cyan]Fetching player props (PTS/REB/AST)...[/cyan]")
    all_props = []

    prop_markets = "player_points,player_rebounds,player_assists,player_points_alternate"

    for g in games:
        event_id = g["id"]
        matchup = f"{g['away_team']} @ {g['home_team']}"
        console.print(f"  {matchup}...")

        try:
            data = api_get(
                f"sports/{SPORT}/events/{event_id}/odds",
                {
                    "regions": REGIONS,
                    "markets": prop_markets,
                    "oddsFormat": ODDS_FMT,
                    "bookmakers": ",".join(PREFERRED_BOOKS),
                },
            )

            props_by_player: dict[str, dict] = {}

            for book in data.get("bookmakers", []):
                bkey = book["key"]
                for market in book.get("markets", []):
                    mkey = market["key"]
                    stat = (
                        "PTS" if "points" in mkey else
                        "REB" if "rebounds" in mkey else
                        "AST" if "assists" in mkey else
                        None
                    )
                    if not stat:
                        continue

                    for outcome in market.get("outcomes", []):
                        player = outcome.get("description") or outcome.get("name", "")
                        side   = outcome.get("name", "")   # "Over" or "Under"
                        line   = outcome.get("point")
                        price  = outcome.get("price")

                        if not player or line is None:
                            continue

                        if player not in props_by_player:
                            props_by_player[player] = {
                                "player": player,
                                "game_id": event_id,
                                "matchup": matchup,
                                "home_team": g["home_team"],
                                "away_team": g["away_team"],
                                "props": {},
                            }

                        if stat not in props_by_player[player]["props"]:
                            props_by_player[player]["props"][stat] = {}

                        entry = props_by_player[player]["props"][stat]
                        if "line" not in entry:
                            entry["line"] = line
                            entry["book"] = bkey

                        if side == "Over" and "over_price" not in entry:
                            entry["over_price"] = price
                        elif side == "Under" and "under_price" not in entry:
                            entry["under_price"] = price

            game_props = list(props_by_player.values())
            all_props.extend(game_props)
            console.print(f"    [green]✓[/green] {len(game_props)} players with props")

        except Exception as e:
            console.print(f"    [yellow]Props unavailable: {e}[/yellow]")

        time.sleep(0.5)

    out = DATA_DIR / "odds_props.json"
    out.write_text(json.dumps(all_props, indent=2))
    console.print(f"\n[bold green]Saved {len(all_props)} player prop lines → {out}[/bold green]")
    return all_props


# ---------------------------------------------------------------------------
# Fetch live in-game odds
# ---------------------------------------------------------------------------

def fetch_live_odds() -> list[dict]:
    console.print("[cyan]Fetching live in-game odds...[/cyan]")
    try:
        data = api_get(f"sports/{SPORT}/odds-live", {
            "regions": REGIONS,
            "markets": "h2h",
            "oddsFormat": ODDS_FMT,
        })
        console.print(f"  [green]{len(data)} live games found[/green]")
        return data
    except Exception as e:
        console.print(f"  [yellow]No live odds: {e}[/yellow]")
        return []


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_game_odds(games: list[dict]) -> None:
    console.rule("[bold #f59e0b]Game Odds — NBA[/bold #f59e0b]")
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold dim")
    t.add_column("Matchup", style="cyan", width=30)
    t.add_column("ML Home", justify="center", width=10)
    t.add_column("ML Away", justify="center", width=10)
    t.add_column("Spread", justify="center", width=10)
    t.add_column("O/U", justify="center", width=10)
    t.add_column("Book", justify="center", width=12)

    for g in games:
        ml = g.get("moneyline", {})
        sp = g.get("spread", {})
        ou = g.get("total", {})
        t.add_row(
            f"{g['away_team']} @ {g['home_team']}",
            str(ml.get("home", "—")),
            str(ml.get("away", "—")),
            str(sp.get("home_line", "—")),
            str(ou.get("line", "—")),
            ml.get("book", "—"),
        )
    console.print(t)


def print_props(props: list[dict], top_n: int = 20) -> None:
    console.rule("[bold #f59e0b]Player Props[/bold #f59e0b]")
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold dim")
    t.add_column("Player", style="cyan", width=22)
    t.add_column("Matchup", width=22)
    t.add_column("PTS line", justify="center", width=10)
    t.add_column("REB line", justify="center", width=10)
    t.add_column("AST line", justify="center", width=10)

    for p in props[:top_n]:
        pr = p.get("props", {})
        t.add_row(
            p["player"],
            p["matchup"],
            str(pr.get("PTS", {}).get("line", "—")),
            str(pr.get("REB", {}).get("line", "—")),
            str(pr.get("AST", {}).get("line", "—")),
        )
    console.print(t)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HoopIQ Odds Pipeline")
    parser.add_argument("--games",  action="store_true", help="Game odds only")
    parser.add_argument("--props",  action="store_true", help="Player props only")
    parser.add_argument("--live",   action="store_true", help="Live in-game odds")
    args = parser.parse_args()

    console.print("[bold #f59e0b]HoopIQ Odds Pipeline[/bold #f59e0b]")
    console.print(f"[dim]Source: The Odds API · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}[/dim]\n")

    if args.live:
        fetch_live_odds()
    elif args.games:
        games = fetch_game_odds()
        print_game_odds(games)
    elif args.props:
        games = fetch_game_odds()
        props = fetch_player_props(games)
        print_props(props)
    else:
        games = fetch_game_odds()
        print_game_odds(games)
        props = fetch_player_props(games)
        print_props(props)
        console.print("\n[bold green]Done! Run: python 5_api_server.py[/bold green]")
