"""HoopIQ FastAPI Server with all endpoints."""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional

import httpx
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("hoopiq")

MODEL_DIR  = Path("models")
DATA_DIR   = Path("data")
MODEL_PATH = MODEL_DIR / "xgb_model.json"
CALIB_PATH = MODEL_DIR / "calibrator.joblib"
FEAT_LIST  = MODEL_DIR / "feature_list.json"
EVAL_PATH  = MODEL_DIR / "eval_report.json"

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
HTTP_HEADERS    = {"User-Agent": "HoopIQ/1.0", "Accept": "application/json"}

app = FastAPI(title="HoopIQ API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_origin_regex=".*",
    allow_credentials=False, allow_methods=["*"],
    allow_headers=["*"], expose_headers=["*"],
)


class ModelState:
    model = None
    calibrated = None
    feature_cols: list = []
    eval_report: dict = {}
    game_log_cache = None
    prop_models: dict = {}
    prop_feat_cols: list = []
    prop_eval: list = []
    player_log_cache = None
    player_index: dict = {}

state = ModelState()


@app.on_event("startup")
async def load_models():
    if MODEL_PATH.exists():
        state.model = xgb.XGBClassifier()
        state.model.load_model(str(MODEL_PATH))
        state.calibrated = joblib.load(CALIB_PATH)
        state.feature_cols = json.loads(FEAT_LIST.read_text())
        state.eval_report = json.loads(EVAL_PATH.read_text()) if EVAL_PATH.exists() else {}
        log.info(f"Game model loaded — {len(state.feature_cols)} features")

    if (DATA_DIR / "game_logs.parquet").exists():
        df = pd.read_parquet(DATA_DIR / "game_logs.parquet")
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        state.game_log_cache = df
        log.info(f"Game log cache — {len(df):,} rows")

    for target in ["PTS","REB","AST","FPTS"]:
        path = MODEL_DIR / f"prop_{target.lower()}.json"
        if path.exists():
            m = xgb.XGBRegressor()
            m.load_model(str(path))
            state.prop_models[target] = m
            log.info(f"Prop model loaded: {target}")

    p = MODEL_DIR / "prop_feature_list.json"
    if p.exists(): state.prop_feat_cols = json.loads(p.read_text())
    p = MODEL_DIR / "prop_eval.json"
    if p.exists(): state.prop_eval = json.loads(p.read_text())

    if (DATA_DIR / "player_logs.parquet").exists():
        df = pd.read_parquet(DATA_DIR / "player_logs.parquet")
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        state.player_log_cache = df
        log.info(f"Player log cache — {len(df):,} rows")

    if (DATA_DIR / "player_index.json").exists():
        state.player_index = json.loads((DATA_DIR / "player_index.json").read_text())


# ─── ESPN Helper ───
async def get_espn_games(game_date=None):
    params = {}
    if game_date: params["dates"] = game_date.replace("-","")
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(ESPN_SCOREBOARD, params=params, headers=HTTP_HEADERS)
        r.raise_for_status()
    games = []
    for ev in r.json().get("events",[]):
        comp = ev.get("competitions",[{}])[0]
        comps = comp.get("competitors",[])
        home = next((c for c in comps if c.get("homeAway")=="home"),{})
        away = next((c for c in comps if c.get("homeAway")=="away"),{})
        status = comp.get("status",{}).get("type",{})
        odds_list = comp.get("odds",[])
        odds = odds_list[0] if odds_list else {}
        games.append({
            "game_id": ev.get("id"),
            "home_abbr": home.get("team",{}).get("abbreviation",""),
            "away_abbr": away.get("team",{}).get("abbreviation",""),
            "home_score": int(home.get("score",0) or 0),
            "away_score": int(away.get("score",0) or 0),
            "home_record": home.get("records",[{}])[0].get("summary","") if home.get("records") else "",
            "away_record": away.get("records",[{}])[0].get("summary","") if away.get("records") else "",
            "status": status.get("state","pre"),
            "status_desc": status.get("description",""),
            "period": comp.get("status",{}).get("period",0),
            "clock": comp.get("status",{}).get("displayClock",""),
            "spread": odds.get("details",""),
            "over_under": odds.get("overUnder"),
            "venue": comp.get("venue",{}).get("fullName",""),
            "date": ev.get("date",""),
        })
    return games


# ─── Feature builders ───
def build_game_features(home, away, game_date):
    if state.game_log_cache is None: return None
    df = state.game_log_cache
    cutoff = pd.Timestamp(game_date)
    def team_feats(abbr):
        t = df[df["TEAM_ABBREVIATION"]==abbr]
        past = t[t["GAME_DATE"]<cutoff].sort_values("GAME_DATE")
        if len(past)<3: return {}
        l5=past.tail(5); l10=past.tail(10); feat={}
        for c in ["PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT","OREB","DREB","REB","AST","STL","BLK","TOV","PF","PLUS_MINUS"]:
            if c in past.columns:
                feat[f"ROLL5_{c}"]=l5[c].mean()
                feat[f"ROLL10_{c}"]=l10[c].mean()
        feat["REST_DAYS"]=min((cutoff-past["GAME_DATE"].iloc[-1]).days,10)
        hm=past[past.get("IS_HOME",pd.Series(True,index=past.index))==True].tail(10)
        aw=past[past.get("IS_HOME",pd.Series(False,index=past.index))==False].tail(10)
        feat["HOME_WIN_RATE"]=(hm["WL"]=="W").mean() if len(hm)>=3 else 0.5
        feat["AWAY_WIN_RATE"]=(aw["WL"]=="W").mean() if len(aw)>=3 else 0.5
        feat["HOME_AVG_PTS"]=hm["PTS"].mean() if len(hm)>=3 else past["PTS"].mean()
        feat["AWAY_AVG_PTS"]=aw["PTS"].mean() if len(aw)>=3 else past["PTS"].mean()
        feat["FORM_WIN_RATE"]=(l10["WL"]=="W").mean() if len(l10)>=3 else 0.5
        feat["FORM_NET_RTG"]=l10["PLUS_MINUS"].mean() if len(l10)>=3 else 0.0
        return feat
    h=team_feats(home); a=team_feats(away)
    if not h or not a: return None
    row={}
    for c,v in h.items(): row[f"H_{c}"]=v
    for c,v in a.items(): row[f"A_{c}"]=v
    for c in h:
        if f"H_{c}" in row and f"A_{c}" in row:
            row[f"DIFF_{c}"]=row[f"H_{c}"]-row[f"A_{c}"]
    vec=np.array([row.get(f,0.0) for f in state.feature_cols],dtype=np.float32)
    return vec.reshape(1,-1)


def build_player_features(player_name, opp_team, is_home, game_date):
    if state.player_log_cache is None or not state.prop_feat_cols: return None
    df = state.player_log_cache
    cutoff = pd.Timestamp(game_date)
    p_df = df[df["PLAYER_NAME"].str.lower() == player_name.lower()]
    if len(p_df)==0: p_df = df[df["PLAYER_NAME"].str.lower().str.contains(player_name.lower())]
    if len(p_df)==0: return None
    past = p_df[p_df["GAME_DATE"]<cutoff].sort_values("GAME_DATE")
    if len(past)<3: return None
    row = {}
    for c in ["PTS","REB","AST","STL","BLK","TOV","MIN","FGM","FGA","FG3M","FPTS"]:
        if c not in past.columns: continue
        for w in [3,5,10]: row[f"ROLL{w}_{c}"] = past[c].tail(w).mean()
        row[f"STD5_{c}"] = past[c].tail(5).std() if len(past)>=5 else 0.0
    row["REST_DAYS"] = min((cutoff - past["GAME_DATE"].iloc[-1]).days, 10)
    row["IS_HOME"] = int(is_home)
    row["GAME_NUM"] = len(past) + 1
    row["FORM_WIN_RATE"] = (past["RESULT"].tail(5) == "W").mean() if "RESULT" in past.columns else 0.5
    if "FGM" in past.columns and "FGA" in past.columns:
        fga = past["FGA"].tail(5).mean()
        row["ROLL5_FG_PCT"] = (past["FGM"].tail(5).mean() / fga) if fga > 0 else 0.45
    if state.game_log_cache is not None:
        opp_past = state.game_log_cache[(state.game_log_cache["TEAM_ABBREVIATION"]==opp_team)&(state.game_log_cache["GAME_DATE"]<cutoff)].tail(5)
        row["OPP_PTS_ALLOWED"] = opp_past["PTS"].mean() if len(opp_past)>=3 else 110.0
    vec = np.array([row.get(f, 0.0) for f in state.prop_feat_cols], dtype=np.float32)
    return vec.reshape(1,-1)


def confidence_label(prob):
    if prob>=0.70: return "HIGH"
    elif prob>=0.58: return "MEDIUM"
    return "LOW"


def grade_start_sit(proj_fpts, avg_fpts):
    diff = proj_fpts - avg_fpts
    pct = diff / max(avg_fpts, 1.0)
    if pct >= 0.15: return {"grade":"A","recommendation":"START","reason":f"Projected {proj_fpts:.1f} FPTS (+{diff:.1f} vs avg)"}
    elif pct >= 0.05: return {"grade":"B","recommendation":"START","reason":f"Projected {proj_fpts:.1f} FPTS, slight edge"}
    elif pct >= -0.10: return {"grade":"C","recommendation":"FLEX","reason":f"Projected {proj_fpts:.1f} FPTS, near average"}
    else: return {"grade":"D","recommendation":"SIT","reason":f"Projected {proj_fpts:.1f} FPTS ({diff:.1f} vs avg)"}


# ─── Schemas ───
class GamePredictRequest(BaseModel):
    home_team: str
    away_team: str
    date: Optional[str] = None
    spread: Optional[float] = None
    over_under: Optional[float] = None

class BatchPredictRequest(BaseModel):
    date: Optional[str] = None

class PropRequest(BaseModel):
    player_name: str
    opp_team: str
    is_home: bool = True
    date: Optional[str] = None
    pts_line: Optional[float] = None
    reb_line: Optional[float] = None
    ast_line: Optional[float] = None

class PredictionResult(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    predicted_winner: str
    confidence: str
    model_version: str = "xgb_v2"


# ─── Routes ───
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "prop_models": list(state.prop_models.keys()),
        "cache_rows": len(state.game_log_cache) if state.game_log_cache is not None else 0,
        "player_rows": len(state.player_log_cache) if state.player_log_cache is not None else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/info")
async def model_info():
    return {
        "n_features": len(state.feature_cols),
        "evaluation": state.eval_report.get("holdout",{}),
        "cv": state.eval_report.get("cv",{}).get("summary",{}),
        "prop_models": {e["target"]: {"mae": e["mae"], "r2": e["r2"]} for e in state.prop_eval} if state.prop_eval else {},
    }


@app.get("/games/today")
async def games_today():
    try:
        et = timezone(timedelta(hours=-5))
        today_et = datetime.now(timezone.utc).astimezone(et).date()
        date_str = today_et.strftime("%Y%m%d")
        games = await get_espn_games(date_str)
        return {"date": today_et.isoformat(), "timezone": "America/New_York", "games": games, "count": len(games)}
    except Exception as e:
        raise HTTPException(502, f"ESPN error: {e}")


@app.post("/predict/game", response_model=PredictionResult)
async def predict_game(req: GamePredictRequest):
    if state.model is None: raise HTTPException(503, "Model not loaded")
    game_date = req.date or date.today().isoformat()
    home, away = req.home_team.upper(), req.away_team.upper()
    features = build_game_features(home, away, game_date)
    home_prob = float(state.calibrated.predict_proba(features)[0][1]) if features is not None else 0.585
    away_prob = 1.0 - home_prob
    return PredictionResult(
        home_team=home, away_team=away,
        home_win_prob=round(home_prob,4), away_win_prob=round(away_prob,4),
        predicted_winner=home if home_prob>=0.5 else away,
        confidence=confidence_label(max(home_prob, away_prob)),
    )


@app.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    if state.model is None: raise HTTPException(503, "Model not loaded")
    et = timezone(timedelta(hours=-5))
    today_et = datetime.now(timezone.utc).astimezone(et).date()
    game_date = req.date or today_et.isoformat()
    try: games = await get_espn_games(game_date)
    except Exception as e: raise HTTPException(502, str(e))
    results = []
    for g in games:
        features = build_game_features(g["home_abbr"], g["away_abbr"], game_date)
        home_prob = float(state.calibrated.predict_proba(features)[0][1]) if features is not None else 0.585
        away_prob = 1.0 - home_prob
        winner = g["home_abbr"] if home_prob>=0.5 else g["away_abbr"]
        results.append({
            "game_id": g["game_id"], "matchup": f"{g['away_abbr']} @ {g['home_abbr']}",
            "home_team": g["home_abbr"], "away_team": g["away_abbr"],
            "home_win_prob": round(home_prob,4), "away_win_prob": round(away_prob,4),
            "predicted_winner": winner, "confidence": confidence_label(max(home_prob, away_prob)),
            "spread": g["spread"], "over_under": g["over_under"], "status": g["status"],
        })
    return {"date": game_date, "predictions": results, "model_accuracy_season": state.eval_report.get("holdout",{}).get("accuracy")}


@app.get("/predict/live")
async def predict_live():
    if state.model is None: raise HTTPException(503, "Model not loaded")
    games = await get_espn_games()
    live = [g for g in games if g["status"]=="in"]
    if not live: return {"message": "No live games.", "games": []}
    today = date.today().isoformat()
    results = []
    for g in live:
        features = build_game_features(g["home_abbr"], g["away_abbr"], today)
        home_prob = float(state.calibrated.predict_proba(features)[0][1]) if features is not None else 0.585
        score_diff = g["home_score"] - g["away_score"]
        adjusted = float(np.clip(home_prob + np.tanh(score_diff/12.0)*0.10, 0.05, 0.95))
        results.append({
            "game_id": g["game_id"], "matchup": f"{g['away_abbr']} @ {g['home_abbr']}",
            "score": f"{g['away_score']} - {g['home_score']}",
            "period": g["period"], "clock": g["clock"],
            "pre_game_home_prob": round(home_prob,4), "live_home_prob": round(adjusted,4),
            "predicted_winner": g["home_abbr"] if adjusted>=0.5 else g["away_abbr"],
            "confidence": confidence_label(max(adjusted, 1.0-adjusted)),
        })
    return {"live_games": len(results), "predictions": results}


@app.post("/props/player")
async def predict_player_props(req: PropRequest):
    if not state.prop_models: raise HTTPException(503, "Prop models not loaded")
    game_date = req.date or date.today().isoformat()
    features = build_player_features(req.player_name, req.opp_team, req.is_home, game_date)
    if features is None: raise HTTPException(404, f"Player not found or insufficient history")
    projections = {t: round(max(0, float(m.predict(features)[0])), 1) for t,m in state.prop_models.items()}
    lines = {"PTS": req.pts_line, "REB": req.reb_line, "AST": req.ast_line}
    prop_picks = {}
    for stat, line in lines.items():
        if line and stat in projections:
            proj = projections[stat]; edge = proj - line
            prop_picks[stat] = {
                "line": line, "projection": proj,
                "pick": "OVER" if edge>0 else "UNDER",
                "edge": round(edge,1),
                "confidence": min(95, max(50, 50 + abs(edge)*5)),
            }
    avg_fpts = None
    if state.player_log_cache is not None:
        df = state.player_log_cache
        p_df = df[df["PLAYER_NAME"].str.lower().str.contains(req.player_name.lower())]
        if len(p_df)>=5: avg_fpts = float(p_df["FPTS"].tail(10).mean())
    proj_fpts = projections.get("FPTS", 0)
    start_sit = grade_start_sit(proj_fpts, avg_fpts or proj_fpts*0.95)
    return {
        "player": req.player_name, "opponent": req.opp_team, "is_home": req.is_home, "date": game_date,
        "projections": projections, "prop_picks": prop_picks,
        "fantasy": {"projected_fpts": proj_fpts, "avg_fpts_last10": round(avg_fpts,1) if avg_fpts else None,
                    "scoring": "DraftKings (PTS×1 + REB×1.25 + AST×1.5 + STL×2 + BLK×2 - TOV×0.5)"},
        "start_sit": start_sit,
    }


@app.get("/props/fantasy")
async def fantasy_lineup(date_str: Optional[str] = None):
    if not state.prop_models or state.player_log_cache is None: raise HTTPException(503, "Models not loaded")
    et = timezone(timedelta(hours=-5))
    today_et = datetime.now(timezone.utc).astimezone(et).date()
    game_date = date_str or today_et.isoformat()
    try: games = await get_espn_games(game_date.replace("-",""))
    except Exception as e: raise HTTPException(502, str(e))
    active = set(); team_to_opp = {}; team_home = {}
    for g in games:
        active.add(g["home_abbr"]); active.add(g["away_abbr"])
        team_to_opp[g["home_abbr"]] = g["away_abbr"]; team_to_opp[g["away_abbr"]] = g["home_abbr"]
        team_home[g["home_abbr"]] = True; team_home[g["away_abbr"]] = False
    if not active: return {"message": "No games today.", "players": []}
    df = state.player_log_cache
    players = df[df["PLAYER_TEAM"].isin(active)]["PLAYER_NAME"].unique()
    results = []
    for name in players:
        p_df = df[df["PLAYER_NAME"]==name]
        if len(p_df)<5: continue
        team = p_df["PLAYER_TEAM"].iloc[-1]
        opp = team_to_opp.get(team,""); is_home = team_home.get(team, True)
        features = build_player_features(name, opp, is_home, game_date)
        if features is None: continue
        proj_fpts = float(state.prop_models["FPTS"].predict(features)[0]) if "FPTS" in state.prop_models else 0
        proj_pts = float(state.prop_models["PTS"].predict(features)[0]) if "PTS" in state.prop_models else 0
        proj_reb = float(state.prop_models["REB"].predict(features)[0]) if "REB" in state.prop_models else 0
        proj_ast = float(state.prop_models["AST"].predict(features)[0]) if "AST" in state.prop_models else 0
        avg_fpts = float(p_df["FPTS"].tail(10).mean())
        ss = grade_start_sit(proj_fpts, avg_fpts)
        results.append({
            "player": name, "team": team, "opponent": opp, "home": is_home,
            "proj_pts": round(max(0,proj_pts),1), "proj_reb": round(max(0,proj_reb),1),
            "proj_ast": round(max(0,proj_ast),1), "proj_fpts": round(max(0,proj_fpts),1),
            "avg_fpts_last10": round(avg_fpts,1),
            "grade": ss["grade"], "recommendation": ss["recommendation"],
        })
    results.sort(key=lambda x: x["proj_fpts"], reverse=True)
    return {"date": game_date, "games": len(games), "players": results,
            "top_plays": [r for r in results if r["grade"]=="A"][:10]}


@app.get("/props/top10")
async def top_picks(date_str: Optional[str] = None):
    if not state.prop_models or state.player_log_cache is None: raise HTTPException(503, "Models not loaded")
    et = timezone(timedelta(hours=-5))
    today_et = datetime.now(timezone.utc).astimezone(et).date()
    game_date = date_str or today_et.isoformat()
    try: games = await get_espn_games(game_date.replace("-",""))
    except Exception as e: raise HTTPException(502, str(e))
    if not games: return {"message": "No games today.", "picks": []}
    active = set(); team_to_opp = {}; team_home = {}
    for g in games:
        active.add(g["home_abbr"]); active.add(g["away_abbr"])
        team_to_opp[g["home_abbr"]] = g["away_abbr"]; team_to_opp[g["away_abbr"]] = g["home_abbr"]
        team_home[g["home_abbr"]] = True; team_home[g["away_abbr"]] = False
    df = state.player_log_cache
    players = df[df["PLAYER_TEAM"].isin(active)]["PLAYER_NAME"].unique()

    # Try Vegas lines (optional)
    vegas_lines = {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("odds_mod", "10_odds_integration.py")
        odds_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(odds_mod)
        odds_games = await odds_mod.fetch_game_odds()
        for og in odds_games[:5]:
            try:
                props = await odds_mod.fetch_player_props(og["game_id"])
                for stat_key, plist in props.get("props",{}).items():
                    for pname, info in plist.items():
                        vegas_lines[(pname.lower(), stat_key)] = info["line"]
            except: continue
    except: pass

    picks = []
    for name in players:
        p_df = df[df["PLAYER_NAME"]==name]
        if len(p_df)<10: continue
        team = p_df["PLAYER_TEAM"].iloc[-1]
        opp = team_to_opp.get(team,""); is_home = team_home.get(team, True)
        features = build_player_features(name, opp, is_home, game_date)
        if features is None: continue
        projections = {s: float(m.predict(features)[0]) for s,m in state.prop_models.items()}
        last5 = p_df.tail(5); last10 = p_df.tail(10)
        for stat in ["PTS","REB","AST"]:
            if stat not in projections: continue
            proj = projections[stat]
            if proj < 5: continue
            avg10 = float(last10[stat].mean()); avg5 = float(last5[stat].mean())
            std5 = float(last5[stat].std()) if len(last5)>=3 else proj*0.2
            form_trend = (avg5-avg10)/max(avg10,1.0)
            consistency = 1.0 - min(std5/max(avg5,1.0), 1.0)
            stat_key = {"PTS":"POINTS","REB":"REBOUNDS","AST":"ASSISTS"}[stat]
            vegas_line = vegas_lines.get((name.lower(), stat_key))
            edge = None; edge_pct = 0; pick_side = None
            if vegas_line:
                edge = proj - vegas_line; edge_pct = (edge/vegas_line)*100 if vegas_line else 0
                pick_side = "OVER" if edge>0 else "UNDER"
            confidence = min(99, max(0, consistency*50 + max(-15,min(15,form_trend*100)) + (abs(edge_pct)*1.5 if edge else 0) + 20))
            if vegas_line:
                if abs(edge_pct)<3: continue
                rec = ("STRONG " if abs(edge_pct)>=10 else "")+pick_side
            else:
                if proj > avg10*1.05: rec = "OVER (model)"; pick_side = "OVER"
                else: continue
            picks.append({
                "player": name, "team": team, "opponent": opp, "home": is_home,
                "stat": stat, "projection": round(proj,1), "vegas_line": vegas_line,
                "edge": round(edge,2) if edge is not None else None,
                "edge_pct": round(edge_pct,1) if vegas_line else None,
                "pick": pick_side, "recommendation": rec, "confidence": round(confidence),
                "form": {"avg_last_5": round(avg5,1), "avg_last_10": round(avg10,1),
                         "trending": "up" if form_trend>0.05 else "down" if form_trend<-0.05 else "flat",
                         "consistency": round(consistency*100)},
            })
    picks.sort(key=lambda p: (abs(p.get("edge_pct") or 0), p["confidence"], p["form"]["consistency"]), reverse=True)
    return {"date": game_date, "games": len(games),
            "vegas_lines_loaded": len(vegas_lines)>0,
            "total_candidates": len(picks), "top_picks": picks[:10]}


@app.get("/props/starts")
async def start_sit(date_str: Optional[str] = None):
    data = await fantasy_lineup(date_str)
    players = data.get("players",[])
    return {
        "date": data.get("date"),
        "must_start": [p for p in players if p["recommendation"]=="START" and p["grade"]=="A"][:10],
        "start": [p for p in players if p["recommendation"]=="START" and p["grade"]=="B"][:10],
        "flex": [p for p in players if p["recommendation"]=="FLEX"][:8],
        "sit": [p for p in players if p["recommendation"]=="SIT"][:8],
    }


# ─── Odds endpoints ───
async def _load_odds_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("odds_mod", "10_odds_integration.py")
    if spec is None: raise FileNotFoundError("10_odds_integration.py not found")
    odds_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(odds_mod)
    return odds_mod


@app.get("/odds/games")
async def odds_games():
    try:
        odds = await _load_odds_module()
        return {"games": await odds.fetch_game_odds(), "usage": odds.get_usage()}
    except Exception as e: raise HTTPException(502, f"Odds error: {e}")


@app.get("/odds/live")
async def odds_live():
    try:
        odds = await _load_odds_module()
        return {"live_games": await odds.fetch_live_odds(), "usage": odds.get_usage()}
    except Exception as e: raise HTTPException(502, f"Odds error: {e}")


@app.get("/odds/player_props/{game_id}")
async def odds_player_props(game_id: str):
    try:
        odds = await _load_odds_module()
        props = await odds.fetch_player_props(game_id)
        return {**props, "usage": odds.get_usage()}
    except Exception as e: raise HTTPException(502, f"Props error: {e}")


@app.get("/odds/usage")
async def odds_usage():
    try:
        odds = await _load_odds_module()
        return odds.get_usage()
    except: return {"remaining": "unknown", "used": "unknown"}


# ─── History Tracking ────────────────────────────────────────────────────
HISTORY_PATH = DATA_DIR / "predictions_log.json"


def load_history():
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except:
            return []
    return []


def save_history(records):
    HISTORY_PATH.write_text(json.dumps(records, indent=2))


class HistoryEntry(BaseModel):
    type: str  # "game" or "prop"
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    matchup: Optional[str] = None
    predicted_winner: Optional[str] = None
    confidence: Optional[str] = None
    home_win_prob: Optional[float] = None
    away_win_prob: Optional[float] = None
    player: Optional[str] = None
    stat: Optional[str] = None
    pick: Optional[str] = None
    line: Optional[float] = None
    projection: Optional[float] = None
    notes: Optional[str] = None


class ResultUpdate(BaseModel):
    result: str  # "WIN", "LOSS", "PENDING"
    actual_score: Optional[str] = None
    actual_value: Optional[float] = None


@app.post("/history/log")
async def log_prediction(entry: HistoryEntry):
    """Log a prediction to track later."""
    records = load_history()
    new_record = {
        "id": f"pred_{int(datetime.now().timestamp() * 1000)}",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "result": "PENDING",
        **entry.dict(exclude_none=True),
    }
    records.insert(0, new_record)
    save_history(records[:500])  # keep last 500
    return {"id": new_record["id"], "logged": True}


@app.get("/history")
async def get_history(filter_type: Optional[str] = None, limit: int = 100):
    """Get prediction history with stats."""
    records = load_history()

    if filter_type:
        records = [r for r in records if r.get("type") == filter_type]

    # Compute win rate stats
    completed = [r for r in records if r.get("result") in ("WIN", "LOSS")]
    wins = sum(1 for r in completed if r.get("result") == "WIN")
    losses = sum(1 for r in completed if r.get("result") == "LOSS")
    pending = sum(1 for r in records if r.get("result") == "PENDING")

    win_rate = (wins / len(completed) * 100) if completed else 0

    # Streak calculation
    streak = 0
    streak_type = None
    for r in records:
        if r.get("result") == "PENDING":
            continue
        if streak_type is None:
            streak_type = r["result"]
            streak = 1
        elif r["result"] == streak_type:
            streak += 1
        else:
            break

    return {
        "stats": {
            "total": len(records),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": round(win_rate, 1),
            "streak": streak,
            "streak_type": streak_type,
        },
        "records": records[:limit],
    }


@app.post("/history/{record_id}/result")
async def update_result(record_id: str, update: ResultUpdate):
    """Mark a prediction as WIN or LOSS."""
    records = load_history()
    found = False
    for r in records:
        if r.get("id") == record_id:
            r["result"] = update.result.upper()
            r["resolved_at"] = datetime.now(timezone.utc).isoformat()
            if update.actual_score:
                r["actual_score"] = update.actual_score
            if update.actual_value is not None:
                r["actual_value"] = update.actual_value
            found = True
            break

    if not found:
        raise HTTPException(404, "Record not found")

    save_history(records)
    return {"updated": True, "id": record_id}


@app.delete("/history/{record_id}")
async def delete_record(record_id: str):
    records = load_history()
    new_records = [r for r in records if r.get("id") != record_id]
    if len(new_records) == len(records):
        raise HTTPException(404, "Record not found")
    save_history(new_records)
    return {"deleted": True}


# ─── Game Prediction with Reasoning ──────────────────────────────────────

def _generate_reasoning(home, away, home_prob, features_vec):
    """Generate human-readable reasons for the prediction."""
    if state.game_log_cache is None:
        return ["Prediction based on team stats"]

    df = state.game_log_cache
    cutoff = pd.Timestamp(date.today().isoformat())

    def team_recent(abbr):
        t = df[(df["TEAM_ABBREVIATION"] == abbr) & (df["GAME_DATE"] < cutoff)]
        return t.sort_values("GAME_DATE").tail(10)

    h_df = team_recent(home)
    a_df = team_recent(away)

    if len(h_df) < 5 or len(a_df) < 5:
        return ["Insufficient recent data for detailed analysis"]

    reasons = []
    winner = home if home_prob >= 0.5 else away
    loser  = away if winner == home else home
    winner_df = h_df if winner == home else a_df
    loser_df  = a_df if winner == home else h_df

    # 1. Form / record
    w_wins = (winner_df["WL"] == "W").sum()
    l_wins = (loser_df["WL"] == "W").sum()
    if w_wins > l_wins:
        reasons.append(f"{winner} is {w_wins}-{10-w_wins} in last 10 games vs {loser}\'s {l_wins}-{10-l_wins}")

    # 2. Net rating (point differential)
    w_pm = winner_df["PLUS_MINUS"].mean()
    l_pm = loser_df["PLUS_MINUS"].mean()
    if abs(w_pm - l_pm) > 2:
        diff = w_pm - l_pm
        reasons.append(f"{winner} has a +{diff:.1f} point differential edge over {loser}")

    # 3. Scoring
    w_pts = winner_df["PTS"].mean()
    l_pts = loser_df["PTS"].mean()
    if w_pts - l_pts > 3:
        reasons.append(f"{winner} averaging {w_pts:.1f} PPG vs {loser}\'s {l_pts:.1f}")
    elif l_pts - w_pts > 3:
        reasons.append(f"{loser} actually scores more ({l_pts:.1f} vs {w_pts:.1f}) but model favors {winner} on other factors")

    # 4. Three-point shooting
    if "FG3_PCT" in winner_df.columns:
        w_3p = winner_df["FG3_PCT"].mean()
        l_3p = loser_df["FG3_PCT"].mean()
        if w_3p - l_3p > 0.02:
            reasons.append(f"{winner} shooting better from 3 ({w_3p*100:.1f}% vs {l_3p*100:.1f}%)")

    # 5. Home court
    if winner == home:
        h_home_games = h_df[h_df.get("IS_HOME", pd.Series([False]*len(h_df))) == True]
        if len(h_home_games) >= 3:
            home_wr = (h_home_games["WL"] == "W").mean()
            if home_wr > 0.6:
                reasons.append(f"{home} is {int(home_wr*100)}% at home in their last home games")

    # 6. Recent momentum
    last3_w = (winner_df.tail(3)["WL"] == "W").sum()
    last3_l = (loser_df.tail(3)["WL"] == "W").sum()
    if last3_w >= 2 and last3_l <= 1:
        reasons.append(f"{winner} has {last3_w}-{3-last3_w} momentum in last 3 games")

    # 7. Confidence framing
    confidence_pct = max(home_prob, 1 - home_prob) * 100
    if confidence_pct >= 70:
        reasons.insert(0, f"Strong {int(confidence_pct)}% confidence pick — model sees clear edge")
    elif confidence_pct >= 60:
        reasons.insert(0, f"Moderate {int(confidence_pct)}% confidence — close matchup with slight edge")
    else:
        reasons.insert(0, f"Low {int(confidence_pct)}% confidence — toss-up, bet small if at all")

    return reasons[:5]


@app.post("/predict/game/explain")
async def predict_game_with_reasoning(req: GamePredictRequest):
    """Predict winner with detailed reasoning."""
    if state.model is None:
        raise HTTPException(503, "Model not loaded")
    game_date = req.date or date.today().isoformat()
    home, away = req.home_team.upper(), req.away_team.upper()
    features = build_game_features(home, away, game_date)
    home_prob = float(state.calibrated.predict_proba(features)[0][1]) if features is not None else 0.585
    away_prob = 1.0 - home_prob
    winner = home if home_prob >= 0.5 else away

    reasoning = _generate_reasoning(home, away, home_prob, features)

    return {
        "home_team": home,
        "away_team": away,
        "home_win_prob": round(home_prob, 4),
        "away_win_prob": round(away_prob, 4),
        "predicted_winner": winner,
        "confidence": confidence_label(max(home_prob, away_prob)),
        "reasoning": reasoning,
        "model_version": "xgb_v2_explained",
    }


@app.post("/predict/batch/explain")
async def predict_batch_with_reasoning(req: BatchPredictRequest):
    """Batch predict all games with reasoning included."""
    if state.model is None:
        raise HTTPException(503, "Model not loaded")
    et = timezone(timedelta(hours=-5))
    today_et = datetime.now(timezone.utc).astimezone(et).date()
    game_date = req.date or today_et.isoformat()
    try:
        games = await get_espn_games(game_date.replace("-",""))
    except Exception as e:
        raise HTTPException(502, str(e))

    results = []
    for g in games:
        features = build_game_features(g["home_abbr"], g["away_abbr"], game_date)
        home_prob = float(state.calibrated.predict_proba(features)[0][1]) if features is not None else 0.585
        away_prob = 1.0 - home_prob
        winner = g["home_abbr"] if home_prob >= 0.5 else g["away_abbr"]
        reasoning = _generate_reasoning(g["home_abbr"], g["away_abbr"], home_prob, features)
        results.append({
            "game_id": g["game_id"],
            "matchup": f"{g['away_abbr']} @ {g['home_abbr']}",
            "home_team": g["home_abbr"],
            "away_team": g["away_abbr"],
            "home_win_prob": round(home_prob, 4),
            "away_win_prob": round(away_prob, 4),
            "predicted_winner": winner,
            "confidence": confidence_label(max(home_prob, away_prob)),
            "reasoning": reasoning,
            "spread": g["spread"],
            "over_under": g["over_under"],
            "status": g["status"],
        })
    return {
        "date": game_date,
        "predictions": results,
        "model_accuracy_season": state.eval_report.get("holdout",{}).get("accuracy"),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("5_api_server:app", host="0.0.0.0", port=port, reload=False, log_level="info")
