"""
Step 10 — Real-Time Odds Integration (The Odds API)
------------------------------------------------------
Fetches game odds + player props + live in-game odds from The Odds API.
Caches results to avoid burning through your API quota.

Adds these endpoints to your API:
  GET /odds/games           — moneyline, spread, totals for all games
  GET /odds/player_props    — PTS/REB/AST player props
  GET /odds/live            — live in-game odds (updates every 60s)
  GET /odds/usage           — your remaining API requests

Usage:
    1. Add your API key to config.json
    2. Run: python 10_odds_integration.py   (test fetch)
    3. The endpoints get auto-mounted in 5_api_server.py
"""

import asyncio
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import httpx
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

CONFIG_PATH = Path("config.json")
CACHE_DIR   = Path("data/odds_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "basketball_nba"

# Cache TTLs (seconds)
# Trade-off: longer TTL = lower API spend, slower reaction to line moves.
# Pre-game lines settle hours before tipoff so 10–90 min staleness is fine.
GAMES_TTL    = 600      # 10 min for game odds (was 5)
PROPS_TTL    = 5400     # 90 min for player props (was 5) — biggest quota sink
LIVE_TTL     = 60       # 1 min for live odds (unchanged — lines move fast in-game)

# Markets
GAME_MARKETS  = "h2h,spreads,totals"
PROP_MARKETS  = ["player_points", "player_rebounds", "player_assists"]
BOOKMAKERS    = "draftkings,fanduel,betmgm"

HEADERS = {"User-Agent": "HoopIQ/1.0"}


def get_api_key() -> str:
    if os.environ.get("ODDS_API_KEY"):
        return os.environ["ODDS_API_KEY"]
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
        key = cfg.get("odds_api_key", "")
        if key and key != "YOUR_ODDS_API_KEY_HERE":
            return key
    raise ValueError("No Odds API key. Set ODDS_API_KEY env var or update config.json")


# ─────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────

def cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def cache_get(key: str, ttl: int) -> Optional[dict]:
    """Return cached data if fresh, else None."""
    p = cache_path(key)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > ttl:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def cache_set(key: str, data: dict) -> None:
    cache_path(key).write_text(json.dumps(data))


# ─────────────────────────────────────────────────────────────────
# API fetchers
# ─────────────────────────────────────────────────────────────────

async def _request(path: str, params: dict) -> dict:
    """Internal: make a request to The Odds API with rate limit tracking."""
    api_key = get_api_key()
    params["apiKey"] = api_key

    async with httpx.AsyncClient(headers=HEADERS) as client:
        r = await client.get(f"{BASE_URL}{path}", params=params, timeout=15.0)
        r.raise_for_status()
        data = r.json()

        # Track rate limit
        remaining = r.headers.get("x-requests-remaining")
        used      = r.headers.get("x-requests-used")
        if remaining:
            cache_set("usage", {
                "remaining": int(remaining),
                "used": int(used or 0),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            })

        return data


async def fetch_game_odds(use_cache: bool = True) -> list[dict]:
    """Fetch moneyline, spread, totals for all NBA games."""
    if use_cache:
        cached = cache_get("games", GAMES_TTL)
        if cached:
            return cached

    raw = await _request(f"/sports/{SPORT}/odds", {
        "regions":    "us",
        "markets":    GAME_MARKETS,
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "american",
        "dateFormat": "iso",
    })

    games = []
    for g in raw:
        parsed = {
            "game_id":   g.get("id"),
            "home_team": g.get("home_team"),
            "away_team": g.get("away_team"),
            "commence":  g.get("commence_time"),
            "moneyline": {},
            "spread":    {},
            "total":     {},
        }

        for bm in g.get("bookmakers", []):
            bm_key = bm["key"]
            for m in bm.get("markets", []):
                if m["key"] == "h2h":
                    for o in m.get("outcomes", []):
                        parsed["moneyline"].setdefault(o["name"], {})[bm_key] = o["price"]
                elif m["key"] == "spreads":
                    for o in m.get("outcomes", []):
                        parsed["spread"].setdefault(o["name"], {})[bm_key] = {
                            "point": o.get("point"), "price": o.get("price")
                        }
                elif m["key"] == "totals":
                    for o in m.get("outcomes", []):
                        parsed["total"].setdefault(o["name"], {})[bm_key] = {
                            "point": o.get("point"), "price": o.get("price")
                        }

        # Calculate consensus prices
        for team, prices in parsed["moneyline"].items():
            vals = list(prices.values())
            parsed["moneyline"][team]["consensus"] = round(sum(vals) / len(vals)) if vals else None

        games.append(parsed)

    cache_set("games", games)
    return games


async def fetch_live_odds(use_cache: bool = True) -> list[dict]:
    """Fetch live in-game odds. Shorter cache for freshness."""
    if use_cache:
        cached = cache_get("live", LIVE_TTL)
        if cached:
            return cached

    raw = await _request(f"/sports/{SPORT}/odds", {
        "regions":    "us",
        "markets":    GAME_MARKETS,
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "american",
        "dateFormat": "iso",
    })

    # Filter to only live games (commence_time in past, not too long ago)
    now = datetime.now(timezone.utc)
    live = []
    for g in raw:
        try:
            commence = datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
            mins_in = (now - commence).total_seconds() / 60
            if 0 < mins_in < 180:  # game started, less than 3 hours ago
                live.append({
                    "game_id":   g["id"],
                    "home_team": g["home_team"],
                    "away_team": g["away_team"],
                    "minutes_in": round(mins_in),
                    "bookmakers": g.get("bookmakers", []),
                })
        except Exception:
            continue

    cache_set("live", live)
    return live


async def fetch_player_props(game_id: str, use_cache: bool = True) -> dict:
    """Fetch PTS/REB/AST props for a single game. Costs 3 API requests."""
    cache_key = f"props_{game_id}"
    if use_cache:
        cached = cache_get(cache_key, PROPS_TTL)
        if cached:
            return cached

    api_key = get_api_key()
    all_props = {"game_id": game_id, "props": {}}

    async with httpx.AsyncClient(headers=HEADERS) as client:
        for market in PROP_MARKETS:
            try:
                r = await client.get(
                    f"{BASE_URL}/sports/{SPORT}/events/{game_id}/odds",
                    params={
                        "apiKey":     api_key,
                        "regions":    "us",
                        "markets":    market,
                        "oddsFormat": "american",
                        "bookmakers": BOOKMAKERS,
                    },
                    timeout=15.0,
                )
                r.raise_for_status()
                data = r.json()
                stat = market.replace("player_", "").upper()  # POINTS, REBOUNDS, ASSISTS

                player_lines = {}
                for bm in data.get("bookmakers", []):
                    for m in bm.get("markets", []):
                        if m["key"] != market:
                            continue
                        for o in m.get("outcomes", []):
                            player = o.get("description") or o.get("name", "")
                            side   = o.get("name", "")  # "Over" or "Under"
                            point  = o.get("point", 0)
                            price  = o.get("price", 0)

                            if player not in player_lines:
                                player_lines[player] = {"line": point, "over": {}, "under": {}}
                            player_lines[player]["line"] = point
                            if side == "Over":
                                player_lines[player]["over"][bm["key"]] = price
                            elif side == "Under":
                                player_lines[player]["under"][bm["key"]] = price

                all_props["props"][stat] = player_lines

                # Track usage
                remaining = r.headers.get("x-requests-remaining")
                if remaining:
                    cache_set("usage", {
                        "remaining": int(remaining),
                        "used": int(r.headers.get("x-requests-used", 0)),
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                    })
            except Exception as e:
                console.print(f"[yellow]Props {market} error: {e}[/yellow]")
                all_props["props"][market.replace("player_", "").upper()] = {}

    cache_set(cache_key, all_props)
    return all_props


def get_usage() -> dict:
    cached = cache_get("usage", 86400)  # cache for 24h
    return cached or {"remaining": "unknown", "used": "unknown"}


# ─────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────

def print_games(games: list[dict]) -> None:
    table = Table(title="NBA Game Odds (consensus)", box=box.SIMPLE_HEAVY)
    table.add_column("Matchup", style="cyan", width=35)
    table.add_column("ML Away", justify="center")
    table.add_column("ML Home", justify="center")
    table.add_column("Spread (Home)", justify="center")
    table.add_column("Total", justify="center")
    table.add_column("Time", justify="center")

    for g in games:
        ml_away = g["moneyline"].get(g["away_team"], {}).get("consensus", "—")
        ml_home = g["moneyline"].get(g["home_team"], {}).get("consensus", "—")

        # Spread (home perspective, first bookmaker)
        sp_data = g["spread"].get(g["home_team"], {})
        sp_first = next(iter([v for k, v in sp_data.items()]), None)
        spread_str = f"{sp_first['point']:+.1f}" if sp_first else "—"

        # Total (over)
        ov_data = g["total"].get("Over", {})
        ov_first = next(iter([v for k, v in ov_data.items()]), None)
        total_str = f"O {ov_first['point']}" if ov_first else "—"

        try:
            t = datetime.fromisoformat(g["commence"].replace("Z", "+00:00"))
            time_str = t.strftime("%I:%M %p")
        except:
            time_str = "—"

        away = g["away_team"].split()[-1]
        home = g["home_team"].split()[-1]
        ml_away_s = f"+{ml_away}" if isinstance(ml_away, int) and ml_away > 0 else str(ml_away)
        ml_home_s = f"+{ml_home}" if isinstance(ml_home, int) and ml_home > 0 else str(ml_home)

        table.add_row(f"{away} @ {home}", ml_away_s, ml_home_s, spread_str, total_str, time_str)

    console.print(table)


def print_usage(usage: dict) -> None:
    console.print(
        f"\n[dim]API Quota: {usage.get('used','?')} used, "
        f"[bold green]{usage.get('remaining','?')}[/bold green] remaining[/dim]"
    )


async def test_run():
    console.print("[bold orange1]Testing The Odds API integration...[/bold orange1]\n")

    try:
        games = await fetch_game_odds()
        console.print(f"[green]✓[/green] Fetched {len(games)} games")
        print_games(games)

        if games:
            console.print(f"\n[bold]Fetching player props for first game...[/bold]")
            props = await fetch_player_props(games[0]["game_id"])
            n_pts = len(props["props"].get("POINTS", {}))
            n_reb = len(props["props"].get("REBOUNDS", {}))
            n_ast = len(props["props"].get("ASSISTS", {}))
            console.print(f"[green]✓[/green] Props: {n_pts} PTS, {n_reb} REB, {n_ast} AST lines")

            # Show top 5 PTS lines
            pts_lines = props["props"].get("POINTS", {})
            if pts_lines:
                console.print("\n[bold]Top 5 PTS prop lines:[/bold]")
                for name, info in list(pts_lines.items())[:5]:
                    over_avg  = sum(info["over"].values())/len(info["over"]) if info["over"] else "—"
                    under_avg = sum(info["under"].values())/len(info["under"]) if info["under"] else "—"
                    console.print(f"  {name:25s} O/U {info['line']:5.1f}  "
                                  f"Over {over_avg if isinstance(over_avg,str) else f'+{int(over_avg)}' if over_avg>0 else int(over_avg)}  "
                                  f"Under {under_avg if isinstance(under_avg,str) else f'+{int(under_avg)}' if under_avg>0 else int(under_avg)}")

        console.print(f"\n[bold]Live games...[/bold]")
        live = await fetch_live_odds()
        console.print(f"[green]✓[/green] {len(live)} live games")

        usage = get_usage()
        print_usage(usage)

        console.print("\n[bold green]All endpoints working! Push to Railway.[/bold green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_run())
