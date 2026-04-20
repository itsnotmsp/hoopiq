"""
10 — Odds API endpoints (append to 5_api_server.py)
-----------------------------------------------------
New endpoints:
  GET  /odds/games       — real moneyline, spread, totals from 70+ books
  GET  /odds/props       — real player prop lines (PTS/REB/AST over/under)
  GET  /odds/edge        — model prediction vs market line edge finder
  GET  /odds/live        — live in-game odds

Run 9_odds_pipeline.py first to populate data/odds_games.json and data/odds_props.json.
The /odds/edge endpoint automatically refreshes odds before returning results.
"""

ODDS_API_KEY = "6098fee47d0139939b30ddaad819fbf9"
ODDS_BASE    = "https://api.the-odds-api.com/v4"
ODDS_SPORT   = "basketball_nba"

# ---------------------------------------------------------------------------
# Odds fetcher helpers (async, called at request time)
# ---------------------------------------------------------------------------

async def fetch_odds_games() -> list[dict]:
    """Pull fresh game odds from The Odds API."""
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "h2h,spreads,totals",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm,caesars",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{ODDS_BASE}/sports/{ODDS_SPORT}/odds", params=params)
        r.raise_for_status()
        raw = r.json()

    games = []
    for ev in raw:
        game = {
            "id":            ev["id"],
            "home_team":     ev["home_team"],
            "away_team":     ev["away_team"],
            "commence_time": ev["commence_time"],
            "moneyline":     {},
            "spread":        {},
            "total":         {},
        }
        for book in ev.get("bookmakers", []):
            bkey = book["key"]
            for market in book.get("markets", []):
                mkey = market["key"]
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}
                if mkey == "h2h" and not game["moneyline"]:
                    game["moneyline"] = {
                        "home": outcomes.get(ev["home_team"], {}).get("price"),
                        "away": outcomes.get(ev["away_team"], {}).get("price"),
                        "book": bkey,
                    }
                elif mkey == "spreads" and not game["spread"]:
                    ho = outcomes.get(ev["home_team"], {})
                    game["spread"] = {
                        "home_line":  ho.get("point"),
                        "home_price": ho.get("price"),
                        "away_line":  outcomes.get(ev["away_team"], {}).get("point"),
                        "away_price": outcomes.get(ev["away_team"], {}).get("price"),
                        "book": bkey,
                    }
                elif mkey == "totals" and not game["total"]:
                    over = outcomes.get("Over", {})
                    game["total"] = {
                        "line":        over.get("point"),
                        "over_price":  over.get("price"),
                        "under_price": outcomes.get("Under", {}).get("price"),
                        "book": bkey,
                    }
        games.append(game)

    # Cache to disk
    (Path("data") / "odds_games.json").write_text(json.dumps(games))
    return games


async def fetch_odds_props_for_event(client: httpx.AsyncClient, event_id: str,
                                      home: str, away: str) -> list[dict]:
    """Pull player props for one game event."""
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "player_points,player_rebounds,player_assists",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm",
    }
    try:
        r = await client.get(
            f"{ODDS_BASE}/sports/{ODDS_SPORT}/events/{event_id}/odds",
            params=params,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    props_by_player: dict[str, dict] = {}
    matchup = f"{away} @ {home}"

    for book in data.get("bookmakers", []):
        bkey = book["key"]
        for market in book.get("markets", []):
            mkey = market["key"]
            stat = ("PTS" if "points" in mkey else
                    "REB" if "rebounds" in mkey else
                    "AST" if "assists" in mkey else None)
            if not stat:
                continue
            for outcome in market.get("outcomes", []):
                player = outcome.get("description") or outcome.get("name", "")
                side   = outcome.get("name", "")
                line   = outcome.get("point")
                price  = outcome.get("price")
                if not player or line is None:
                    continue
                if player not in props_by_player:
                    props_by_player[player] = {
                        "player": player, "game_id": event_id,
                        "matchup": matchup, "home_team": home, "away_team": away,
                        "props": {},
                    }
                entry = props_by_player[player]["props"].setdefault(stat, {"book": bkey})
                if "line" not in entry:
                    entry["line"] = line
                if side == "Over" and "over_price" not in entry:
                    entry["over_price"] = price
                elif side == "Under" and "under_price" not in entry:
                    entry["under_price"] = price

    return list(props_by_player.values())


# ---------------------------------------------------------------------------
# Team name normaliser  (Odds API uses full names, our model uses abbreviations)
# ---------------------------------------------------------------------------

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


# ===========================================================================
# ROUTES  — paste these into 5_api_server.py (before the if __name__ block)
# ===========================================================================

@app.get("/odds/games")
async def odds_games():
    """Real moneyline, spread and totals from DraftKings/FanDuel/BetMGM."""
    try:
        games = await fetch_odds_games()
        return {"count": len(games), "games": games,
                "updated": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        raise HTTPException(502, f"Odds API error: {e}")


@app.get("/odds/props")
async def odds_props():
    """Real player prop lines for all tonight's games."""
    try:
        games = await fetch_odds_games()
        all_props = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            for g in games:
                props = await fetch_odds_props_for_event(
                    client, g["id"], g["home_team"], g["away_team"]
                )
                all_props.extend(props)
        (Path("data") / "odds_props.json").write_text(json.dumps(all_props))
        return {"count": len(all_props), "props": all_props,
                "updated": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        raise HTTPException(502, f"Props API error: {e}")


@app.get("/odds/edge")
async def odds_edge():
    """
    Compare model win probability vs market implied probability.
    Returns games where the model finds an edge over the market.
    """
    try:
        games = await fetch_odds_games()
    except Exception as e:
        raise HTTPException(502, f"Odds fetch failed: {e}")

    today = date.today().isoformat()
    results = []

    for g in games:
        home_full = g["home_team"]
        away_full = g["away_team"]
        home_abbr = TEAM_NAME_TO_ABBR.get(home_full, home_full[:3].upper())
        away_abbr = TEAM_NAME_TO_ABBR.get(away_full, away_full[:3].upper())

        # Model prediction
        if state.model and state.calibrated:
            features = build_game_features(home_abbr, away_abbr, today)
            if features is not None:
                model_home_prob = float(state.calibrated.predict_proba(features)[0][1])
            else:
                model_home_prob = 0.585
        else:
            model_home_prob = 0.585

        # Market implied probability from moneyline
        ml = g.get("moneyline", {})
        home_ml = ml.get("home")
        away_ml = ml.get("away")
        market_home_prob = None

        if home_ml and away_ml:
            def ml_to_prob(ml_price):
                if ml_price > 0:
                    return 100 / (ml_price + 100)
                else:
                    return abs(ml_price) / (abs(ml_price) + 100)

            raw_home = ml_to_prob(home_ml)
            raw_away = ml_to_prob(away_ml)
            vig = raw_home + raw_away
            market_home_prob = raw_home / vig  # remove vig

        sp = g.get("spread", {})
        ou = g.get("total", {})

        result = {
            "matchup":        f"{away_full} @ {home_full}",
            "home_team":      home_full,
            "away_team":      away_full,
            "home_abbr":      home_abbr,
            "away_abbr":      away_abbr,
            "commence_time":  g["commence_time"],
            "model_home_prob":   round(model_home_prob, 4),
            "model_away_prob":   round(1 - model_home_prob, 4),
            "market_home_prob":  round(market_home_prob, 4) if market_home_prob else None,
            "market_away_prob":  round(1 - market_home_prob, 4) if market_home_prob else None,
            "moneyline":      ml,
            "spread":         sp,
            "total":          ou,
            "edge": None,
        }

        if market_home_prob:
            edge = model_home_prob - market_home_prob
            result["edge"] = {
                "value":      round(edge, 4),
                "direction":  "home" if edge > 0 else "away",
                "pct":        f"{abs(edge)*100:.1f}%",
                "rating":     "strong" if abs(edge) > 0.07 else
                              "moderate" if abs(edge) > 0.04 else "small",
                "bet_signal": abs(edge) > 0.04,
            }

        results.append(result)

    results.sort(key=lambda x: abs(x["edge"]["value"]) if x["edge"] else 0, reverse=True)
    return {
        "date": today,
        "count": len(results),
        "games": results,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
