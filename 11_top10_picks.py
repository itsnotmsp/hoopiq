"""
Step 11 — Top 10 Best Prop Picks
----------------------------------
Adds /props/top10 endpoint to your API server.

Picks are ranked by:
  - Edge vs Vegas line (when available)
  - Recent form trend (last 5 vs last 10)
  - Consistency score (low variance = more reliable)
  - Model confidence

Returns the 10 highest-confidence prop picks for tonight.

INSTALLATION:
    1. Open your 5_api_server.py
    2. Find: @app.get("/props/starts")
    3. Add the entire ROUTE_CODE block BEFORE that line
    4. Restart your server: python 5_api_server.py
"""

ROUTE_CODE = '''
@app.get("/props/top10")
async def top_picks(date_str: Optional[str] = None):
    """Return the top 10 highest-confidence prop picks for tonight."""
    if not state.prop_models or state.player_log_cache is None:
        raise HTTPException(503, "Prop models not loaded.")

    game_date = date_str or date.today().isoformat()

    try:
        games = await get_espn_games(game_date)
    except Exception as e:
        raise HTTPException(502, str(e))

    if not games:
        return {"message": "No games today.", "picks": []}

    active_teams = set()
    team_to_opp = {}
    team_home = {}
    for g in games:
        active_teams.add(g["home_abbr"])
        active_teams.add(g["away_abbr"])
        team_to_opp[g["home_abbr"]] = g["away_abbr"]
        team_to_opp[g["away_abbr"]] = g["home_abbr"]
        team_home[g["home_abbr"]] = True
        team_home[g["away_abbr"]] = False

    df = state.player_log_cache
    players_tonight = df[df["PLAYER_TEAM"].isin(active_teams)]["PLAYER_NAME"].unique()

    # Try to load Vegas lines (optional)
    vegas_lines = {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("odds_mod", "10_odds_integration.py")
        odds_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(odds_mod)
        odds_games = await odds_mod.fetch_game_odds()
        for og in odds_games[:5]:  # limit to 5 games to save API calls
            try:
                props = await odds_mod.fetch_player_props(og["game_id"])
                for stat_key, players in props.get("props", {}).items():
                    for player_name, info in players.items():
                        vegas_lines[(player_name.lower(), stat_key)] = info["line"]
            except Exception:
                continue
    except Exception:
        pass

    picks = []
    for player_name in players_tonight:
        p_df = df[df["PLAYER_NAME"] == player_name]
        if len(p_df) < 10:
            continue

        team = p_df["PLAYER_TEAM"].iloc[-1]
        opp = team_to_opp.get(team, "")
        is_home = team_home.get(team, True)

        features = build_player_features(player_name, opp, is_home, game_date)
        if features is None:
            continue

        projections = {}
        for stat, model in state.prop_models.items():
            projections[stat] = float(model.predict(features)[0])

        last5 = p_df.tail(5)
        last10 = p_df.tail(10)

        for stat in ["PTS", "REB", "AST"]:
            if stat not in projections:
                continue

            proj = projections[stat]
            avg10 = float(last10[stat].mean())
            avg5 = float(last5[stat].mean())
            std5 = float(last5[stat].std()) if len(last5) >= 3 else proj * 0.2

            if proj < 5:
                continue

            form_trend = (avg5 - avg10) / max(avg10, 1.0)
            consistency = 1.0 - min(std5 / max(avg5, 1.0), 1.0)

            stat_key_map = {"PTS": "POINTS", "REB": "REBOUNDS", "AST": "ASSISTS"}
            line_key = (player_name.lower(), stat_key_map[stat])
            vegas_line = vegas_lines.get(line_key)

            edge = None
            edge_pct = 0
            pick_side = None
            if vegas_line:
                edge = proj - vegas_line
                edge_pct = (edge / vegas_line) * 100 if vegas_line else 0
                pick_side = "OVER" if edge > 0 else "UNDER"

            base_score = consistency * 50
            form_score = max(-15, min(15, form_trend * 100))
            edge_score = abs(edge_pct) * 1.5 if edge else 0
            confidence = min(99, max(0, base_score + form_score + edge_score + 20))

            if vegas_line:
                if abs(edge_pct) < 3:
                    continue
                elif abs(edge_pct) >= 10:
                    recommendation = "STRONG " + pick_side
                else:
                    recommendation = pick_side
            else:
                if proj > avg10 * 1.05:
                    recommendation = "OVER (model)"
                    pick_side = "OVER"
                else:
                    continue

            picks.append({
                "player": player_name,
                "team": team,
                "opponent": opp,
                "home": is_home,
                "stat": stat,
                "projection": round(proj, 1),
                "vegas_line": vegas_line,
                "edge": round(edge, 2) if edge is not None else None,
                "edge_pct": round(edge_pct, 1) if vegas_line else None,
                "pick": pick_side,
                "recommendation": recommendation,
                "confidence": round(confidence),
                "form": {
                    "avg_last_5": round(avg5, 1),
                    "avg_last_10": round(avg10, 1),
                    "trending": "up" if form_trend > 0.05 else "down" if form_trend < -0.05 else "flat",
                    "consistency": round(consistency * 100),
                },
            })

    def sort_key(p):
        edge = abs(p["edge_pct"]) if p.get("edge_pct") else 0
        return (edge, p["confidence"], p["form"]["consistency"])

    picks.sort(key=sort_key, reverse=True)
    top10 = picks[:10]

    return {
        "date": game_date,
        "games": len(games),
        "vegas_lines_loaded": len(vegas_lines) > 0,
        "total_candidates": len(picks),
        "top_picks": top10,
    }
'''

# Auto-install: try to inject this into local 5_api_server.py
import sys
from pathlib import Path

def install():
    server_path = Path("5_api_server.py")
    if not server_path.exists():
        print("Error: 5_api_server.py not found in current directory")
        return False

    content = server_path.read_text()

    if '/props/top10' in content:
        print("Already installed!")
        return True

    # Find /props/starts and inject before it
    marker = '@app.get("/props/starts")'
    if marker in content:
        content = content.replace(marker, ROUTE_CODE.strip() + '\n\n\n' + marker)
        server_path.write_text(content)
        print("Installed /props/top10 endpoint")
        return True

    # Fallback: add before the main block
    main_marker = 'if __name__ == "__main__":'
    if main_marker in content:
        content = content.replace(main_marker, ROUTE_CODE.strip() + '\n\n\n' + main_marker)
        server_path.write_text(content)
        print("Installed /props/top10 endpoint (added before main)")
        return True

    print("Could not find a place to inject. Add manually.")
    print(ROUTE_CODE)
    return False


if __name__ == "__main__":
    install()
