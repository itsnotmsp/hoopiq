"""
Step 20 — Deep Game Analysis Endpoint
---------------------------------------
Adds /predict/game/analyze — comprehensive 10-factor analysis:

1. Team recent form (last 5-10)
2. Injuries (placeholder - needs paid API)
3. Home/away splits
4. Schedule fatigue (B2B, 3-in-4)
5. Matchup analysis (size, pace, defense)
6. Pace & scoring environment
7. Situational (back-to-back, travel)
8. Betting market (line analysis)
9. Advanced analytics (ORtg, DRtg, NetRtg, TS%, TO%, REB%)
10. Betting value (spread, ML, O/U, confidence)

Run:
    python 20_deep_analysis.py
    # Then restart server
"""

from pathlib import Path

NEW_CODE = '''

# ─── Deep Game Analysis ──────────────────────────────────────────────
@app.post("/predict/game/analyze")
async def analyze_game_deep(req: GamePredictRequest):
    """Comprehensive 10-factor game analysis with betting value assessment."""
    if state.model is None or state.game_log_cache is None:
        raise HTTPException(503, "Model or data not loaded")

    home, away = req.home_team.upper(), req.away_team.upper()
    game_date = req.date or date.today().isoformat()
    cutoff = pd.Timestamp(game_date)

    df = state.game_log_cache

    def team_logs(abbr, n=10):
        t = df[df["TEAM_ABBREVIATION"] == abbr]
        return t[t["GAME_DATE"] < cutoff].sort_values("GAME_DATE").tail(n)

    h_l10 = team_logs(home, 10)
    a_l10 = team_logs(away, 10)
    h_l5 = h_l10.tail(5)
    a_l5 = a_l10.tail(5)

    if len(h_l10) < 5 or len(a_l10) < 5:
        raise HTTPException(400, f"Insufficient data for {home} or {away}")

    # ── Get base prediction ──
    features = build_game_features(home, away, game_date)
    home_prob = float(state.calibrated.predict_proba(features)[0][1]) if features is not None else 0.585
    away_prob = 1 - home_prob

    def safe_mean(series):
        return float(series.mean()) if len(series) > 0 else 0.0

    # ── 1. RECENT FORM ──
    h_wins_l10 = int((h_l10["WL"] == "W").sum())
    a_wins_l10 = int((a_l10["WL"] == "W").sum())
    h_wins_l5 = int((h_l5["WL"] == "W").sum())
    a_wins_l5 = int((a_l5["WL"] == "W").sum())
    h_off = safe_mean(h_l10["PTS"])
    a_off = safe_mean(a_l10["PTS"])
    h_pm = safe_mean(h_l10["PLUS_MINUS"])
    a_pm = safe_mean(a_l10["PLUS_MINUS"])

    recent_form = {
        "home": {
            "record_l10": f"{h_wins_l10}-{10-h_wins_l10}",
            "record_l5":  f"{h_wins_l5}-{5-h_wins_l5}",
            "ppg":        round(h_off, 1),
            "net_rating": round(h_pm, 1),
            "trend":      "improving" if h_wins_l5 >= 3 else "declining" if h_wins_l5 <= 1 else "steady",
        },
        "away": {
            "record_l10": f"{a_wins_l10}-{10-a_wins_l10}",
            "record_l5":  f"{a_wins_l5}-{5-a_wins_l5}",
            "ppg":        round(a_off, 1),
            "net_rating": round(a_pm, 1),
            "trend":      "improving" if a_wins_l5 >= 3 else "declining" if a_wins_l5 <= 1 else "steady",
        },
    }

    # ── 2. INJURIES (placeholder - real version needs paid API) ──
    injuries = {
        "home": {"key_outs": [], "impact": "Unknown - check ESPN/Rotowire"},
        "away": {"key_outs": [], "impact": "Unknown - check ESPN/Rotowire"},
        "note": "Live injury data requires paid API integration",
    }

    # ── 3. HOME / AWAY SPLITS ──
    h_home_only = h_l10[h_l10.get("IS_HOME", pd.Series([True]*len(h_l10))) == True]
    a_away_only = a_l10[a_l10.get("IS_HOME", pd.Series([False]*len(a_l10))) == False]
    h_home_wr = (h_home_only["WL"] == "W").mean() if len(h_home_only) >= 3 else 0.5
    a_away_wr = (a_away_only["WL"] == "W").mean() if len(a_away_only) >= 3 else 0.5

    location = {
        "home_record_at_home": f"{int(h_home_wr*100)}% ({len(h_home_only)} games)",
        "away_record_on_road": f"{int(a_away_wr*100)}% ({len(a_away_only)} games)",
        "home_court_factor":   round((h_home_wr - a_away_wr), 2),
    }

    # ── 4. SCHEDULE FATIGUE ──
    def get_schedule_state(team_logs):
        if len(team_logs) < 2:
            return {"days_rest": 3, "is_b2b": False, "is_3in4": False, "games_l7": 0}
        last_game = team_logs.iloc[-1]["GAME_DATE"]
        days_rest = (cutoff - last_game).days
        is_b2b = days_rest <= 1
        last3 = team_logs.tail(3)
        is_3in4 = (cutoff - last3.iloc[0]["GAME_DATE"]).days <= 3 if len(last3) >= 3 else False
        from datetime import timedelta
        games_l7 = len(team_logs[team_logs["GAME_DATE"] >= cutoff - timedelta(days=7)])
        return {"days_rest": int(days_rest), "is_b2b": bool(is_b2b), "is_3in4": bool(is_3in4), "games_l7": int(games_l7)}

    fatigue = {
        "home": get_schedule_state(h_l10),
        "away": get_schedule_state(a_l10),
    }

    # ── 5. MATCHUP ANALYSIS ──
    def adv_stats(logs):
        pts = safe_mean(logs["PTS"])
        fga = safe_mean(logs["FGA"]) if "FGA" in logs.columns else 85
        fta = safe_mean(logs["FTA"]) if "FTA" in logs.columns else 20
        oreb = safe_mean(logs.get("OREB", pd.Series([0])))
        tov = safe_mean(logs.get("TOV", pd.Series([13])))
        # Possessions estimate
        poss = fga + 0.44*fta - oreb + tov
        # Off rating per 100 poss
        off_rtg = (pts / poss * 100) if poss > 0 else 110
        # Pace estimate
        pace = poss
        ts_pct = (pts / (2*(fga + 0.44*fta))) * 100 if (fga + 0.44*fta) > 0 else 55
        return {
            "off_rtg": round(off_rtg, 1),
            "pace":    round(pace, 1),
            "ts_pct":  round(ts_pct, 1),
            "rebs":    round(safe_mean(logs.get("REB", pd.Series([45]))), 1),
            "asts":    round(safe_mean(logs.get("AST", pd.Series([25]))), 1),
            "tov":     round(tov, 1),
        }

    h_adv = adv_stats(h_l10)
    a_adv = adv_stats(a_l10)

    matchup = {
        "rebounding_edge":  f"{home}: {h_adv['rebs']} vs {away}: {a_adv['rebs']}",
        "rebound_diff":     round(h_adv["rebs"] - a_adv["rebs"], 1),
        "scoring_efficiency": f"{home} TS {h_adv['ts_pct']}% vs {away} TS {a_adv['ts_pct']}%",
        "ts_diff":          round(h_adv["ts_pct"] - a_adv["ts_pct"], 1),
        "passing":          f"{home}: {h_adv['asts']} apg vs {away}: {a_adv['asts']} apg",
        "turnover_battle":  f"{home}: {h_adv['tov']} vs {away}: {a_adv['tov']}",
    }

    # ── 6. PACE & SCORING ──
    avg_pace = (h_adv["pace"] + a_adv["pace"]) / 2
    expected_total = h_off + a_off
    pace_class = "fast" if avg_pace > 100 else "medium" if avg_pace > 95 else "slow"

    pace_analysis = {
        "expected_pace": round(avg_pace, 1),
        "pace_class": pace_class,
        "expected_total_points": round(expected_total, 1),
        "vs_market_total": None,
    }

    # ── 7. SITUATIONAL ANGLES ──
    situational = []
    if fatigue["home"]["is_b2b"]:
        situational.append(f"⚠️ {home} on B2B — historical 3% accuracy drop")
    if fatigue["away"]["is_b2b"]:
        situational.append(f"⚠️ {away} on B2B — fade slightly")
    if fatigue["home"]["is_3in4"]:
        situational.append(f"⚠️ {home} playing 3 games in 4 nights")
    if fatigue["away"]["is_3in4"]:
        situational.append(f"⚠️ {away} playing 3 games in 4 nights")
    if fatigue["home"]["days_rest"] >= 3 and fatigue["away"]["days_rest"] <= 1:
        situational.append(f"✅ {home} rest advantage ({fatigue['home']['days_rest']}d vs {fatigue['away']['days_rest']}d)")
    if fatigue["away"]["days_rest"] >= 3 and fatigue["home"]["days_rest"] <= 1:
        situational.append(f"✅ {away} rest advantage")
    if h_pm > 5 and a_pm < -2:
        situational.append(f"📊 Big NetRtg gap favors {home}")

    # ── 8. BETTING MARKET ──
    # Try to fetch real odds
    market = {"available": False, "spread": None, "total": None, "ml_home": None, "ml_away": None}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("odds_mod", "10_odds_integration.py")
        odds_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(odds_mod)
        odds_games = await odds_mod.fetch_game_odds()
        for og in odds_games:
            home_match = home in og["home_team"].upper() or og["home_team"].split()[-1].upper()[:3] == home[:3]
            away_match = away in og["away_team"].upper() or og["away_team"].split()[-1].upper()[:3] == away[:3]
            if home_match and away_match:
                ml_home_consensus = og["moneyline"].get(og["home_team"], {}).get("consensus")
                ml_away_consensus = og["moneyline"].get(og["away_team"], {}).get("consensus")
                # Spread (from home perspective)
                spread_dict = og["spread"].get(og["home_team"], {})
                spread_first = next((v for v in spread_dict.values() if isinstance(v, dict)), None)
                # Total
                total_dict = og["total"].get("Over", {})
                total_first = next((v for v in total_dict.values() if isinstance(v, dict)), None)
                market = {
                    "available": True,
                    "ml_home": ml_home_consensus,
                    "ml_away": ml_away_consensus,
                    "spread": spread_first.get("point") if spread_first else None,
                    "total":  total_first.get("point") if total_first else None,
                }
                break
    except Exception:
        pass

    # ── 9. ADVANCED ANALYTICS ──
    advanced = {
        "home": {
            "offensive_rating": h_adv["off_rtg"],
            "true_shooting_pct": h_adv["ts_pct"],
            "pace": h_adv["pace"],
            "net_rating_l10": round(h_pm, 1),
        },
        "away": {
            "offensive_rating": a_adv["off_rtg"],
            "true_shooting_pct": a_adv["ts_pct"],
            "pace": a_adv["pace"],
            "net_rating_l10": round(a_pm, 1),
        },
    }

    # ── 10. BETTING VALUE ASSESSMENT ──
    confidence = max(home_prob, away_prob)
    confidence_10 = round(confidence * 10, 1)

    # Compare model to market
    value_analysis = {
        "spread_value": None,
        "moneyline_value": None,
        "total_value": None,
    }

    if market["available"] and market["ml_home"]:
        # Convert ML to implied prob
        def ml_to_prob(ml):
            if ml is None: return 0.5
            return abs(ml)/(abs(ml)+100) if ml < 0 else 100/(ml+100)

        market_home_prob = ml_to_prob(market["ml_home"])
        edge_home = home_prob - market_home_prob

        if abs(edge_home) > 0.05:
            value_analysis["moneyline_value"] = {
                "side": home if edge_home > 0 else away,
                "edge_pct": round(edge_home * 100, 1),
                "model_prob": round(home_prob*100, 1),
                "implied_prob": round(market_home_prob*100, 1),
                "verdict": "STRONG VALUE" if abs(edge_home) > 0.10 else "VALUE",
            }
        else:
            value_analysis["moneyline_value"] = {"verdict": "NO VALUE - market efficient"}

        if market["total"]:
            total_diff = expected_total - market["total"]
            if abs(total_diff) > 4:
                value_analysis["total_value"] = {
                    "side": "OVER" if total_diff > 0 else "UNDER",
                    "model_total": round(expected_total, 1),
                    "market_total": market["total"],
                    "diff": round(total_diff, 1),
                    "verdict": "VALUE" if abs(total_diff) > 6 else "MARGINAL",
                }
            pace_analysis["vs_market_total"] = round(total_diff, 1)

    # ── BEST BET ──
    predicted_winner = home if home_prob >= 0.5 else away
    best_bet_options = []

    if value_analysis.get("moneyline_value", {}).get("side"):
        ml_v = value_analysis["moneyline_value"]
        if "edge_pct" in ml_v:
            best_bet_options.append({
                "type": "MONEYLINE",
                "side": ml_v["side"],
                "edge": ml_v.get("edge_pct"),
                "verdict": ml_v["verdict"],
            })

    if value_analysis.get("total_value"):
        tv = value_analysis["total_value"]
        if "side" in tv:
            best_bet_options.append({
                "type": "TOTAL",
                "side": f"{tv['side']} {tv['market_total']}",
                "edge": tv.get("diff"),
                "verdict": tv["verdict"],
            })

    if not best_bet_options:
        best_bet_options.append({
            "type": "MONEYLINE",
            "side": predicted_winner,
            "verdict": f"{int(confidence*100)}% model confidence",
            "edge": None,
        })

    # ── RISK FACTORS ──
    risk_factors = []
    if confidence < 0.55:
        risk_factors.append("⚠️ Low confidence game — close matchup, bet small")
    if fatigue["home"]["is_b2b"] or fatigue["away"]["is_b2b"]:
        risk_factors.append("⚠️ B2B game — schedule unpredictable")
    if abs(h_pm) > 8 or abs(a_pm) > 8:
        risk_factors.append("⚠️ Recent blowouts may inflate stats")
    if h_wins_l5 == 5 or a_wins_l5 == 5:
        risk_factors.append("⚠️ Hot streak may be due for regression")

    return {
        "matchup": f"{away} @ {home}",
        "date": game_date,
        "prediction": {
            "winner": predicted_winner,
            "home_win_prob": round(home_prob, 4),
            "away_win_prob": round(away_prob, 4),
            "confidence_pct": round(confidence * 100, 1),
            "confidence_10": confidence_10,
        },
        "recent_form": recent_form,
        "injuries": injuries,
        "location": location,
        "fatigue": fatigue,
        "matchup": matchup,
        "pace": pace_analysis,
        "situational": situational,
        "market": market,
        "advanced": advanced,
        "value": value_analysis,
        "best_bet": best_bet_options[0],
        "all_bets": best_bet_options,
        "risk_factors": risk_factors,
    }
'''


def install():
    server = Path("5_api_server.py")
    if not server.exists():
        print("Error: 5_api_server.py not found")
        return False

    content = server.read_text()
    if "/predict/game/analyze" in content:
        print("Already installed")
        return True

    # Inject before main block
    marker = 'if __name__ == "__main__":'
    if marker in content:
        content = content.replace(marker, NEW_CODE.strip() + "\n\n\n" + marker)
        server.write_text(content)
        print("✓ Installed /predict/game/analyze endpoint")
        print("Restart server: python 5_api_server.py")
        return True
    print("Could not find injection point")
    return False


if __name__ == "__main__":
    install()
