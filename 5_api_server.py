import asyncio
import json
import logging
from datetime import datetime, timezone, date
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

app = FastAPI(title="HoopIQ Prediction API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ModelState:
    # Game winner model
    model = None
    calibrated = None
    feature_cols: list = []
    eval_report: dict = {}
    game_log_cache = None
    # Player prop models
    prop_models: dict = {}          # {"PTS": xgb, "REB": xgb, ...}
    prop_feat_cols: list = []
    prop_eval: list = []
    player_log_cache = None
    player_index: dict = {}         # name → player_id

state = ModelState()


@app.on_event("startup")
async def load_models():
    # Game winner model
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

    # Player prop models
    for target in ["PTS","REB","AST","FPTS"]:
        path = MODEL_DIR / f"prop_{target.lower()}.json"
        if path.exists():
            m = xgb.XGBRegressor()
            m.load_model(str(path))
            state.prop_models[target] = m
            log.info(f"Prop model loaded: {target}")

    prop_feat_path = MODEL_DIR / "prop_feature_list.json"
    if prop_feat_path.exists():
        state.prop_feat_cols = json.loads(prop_feat_path.read_text())

    prop_eval_path = MODEL_DIR / "prop_eval.json"
    if prop_eval_path.exists():
        state.prop_eval = json.loads(prop_eval_path.read_text())

    if (DATA_DIR / "player_logs.parquet").exists():
        df = pd.read_parquet(DATA_DIR / "player_logs.parquet")
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        state.player_log_cache = df
        log.info(f"Player log cache — {len(df):,} rows, {df['PLAYER_NAME'].nunique()} players")

    if (DATA_DIR / "player_index.json").exists():
        state.player_index = json.loads((DATA_DIR / "player_index.json").read_text())


# -----------------------------------------------------------------------
# ESPN helpers
# -----------------------------------------------------------------------

async def get_espn_games(game_date=None):
    params = {}
    if game_date:
        params["dates"] = game_date.replace("-","")
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(ESPN_SCOREBOARD, params=params, headers=HTTP_HEADERS)
        r.raise_for_status()
    events = r.json().get("events",[])
    games = []
    for ev in events:
        comp = ev.get("competitions",[{}])[0]
        comps = comp.get("competitors",[])
        home = next((c for c in comps if c.get("homeAway")=="home"),{})
        away = next((c for c in comps if c.get("homeAway")=="away"),{})
        status = comp.get("status",{}).get("type",{})
        odds_list = comp.get("odds",[])
        odds = odds_list[0] if odds_list else {}
        games.append({
            "game_id":    ev.get("id"),
            "home_abbr":  home.get("team",{}).get("abbreviation",""),
            "away_abbr":  away.get("team",{}).get("abbreviation",""),
            "home_score": int(home.get("score",0) or 0),
            "away_score": int(away.get("score",0) or 0),
            "home_record": home.get("records",[{}])[0].get("summary","") if home.get("records") else "",
            "away_record": away.get("records",[{}])[0].get("summary","") if away.get("records") else "",
            "status":      status.get("state","pre"),
            "status_desc": status.get("description",""),
            "period":      comp.get("status",{}).get("period",0),
            "clock":       comp.get("status",{}).get("displayClock",""),
            "spread":      odds.get("details",""),
            "over_under":  odds.get("overUnder"),
            "venue":       comp.get("venue",{}).get("fullName",""),
            "date":        ev.get("date",""),
        })
    return games


# -----------------------------------------------------------------------
# Game winner inference
# -----------------------------------------------------------------------

def build_game_features(home, away, game_date):
    if state.game_log_cache is None:
        return None
    df = state.game_log_cache
    cutoff = pd.Timestamp(game_date)

    def team_feats(abbr):
        t = df[df["TEAM_ABBREVIATION"]==abbr]
        past = t[t["GAME_DATE"]<cutoff].sort_values("GAME_DATE")
        if len(past)<3: return {}
        l5=past.tail(5); l10=past.tail(10)
        feat={}
        for col in ["PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT","OREB","DREB","REB","AST","STL","BLK","TOV","PF","PLUS_MINUS"]:
            if col in past.columns:
                feat[f"ROLL5_{col}"]=l5[col].mean()
                feat[f"ROLL10_{col}"]=l10[col].mean()
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
    for col,val in h.items(): row[f"H_{col}"]=val
    for col,val in a.items(): row[f"A_{col}"]=val
    for col in h:
        if f"H_{col}" in row and f"A_{col}" in row:
            row[f"DIFF_{col}"]=row[f"H_{col}"]-row[f"A_{col}"]
    vec=np.array([row.get(f,0.0) for f in state.feature_cols],dtype=np.float32)
    return vec.reshape(1,-1)


def confidence_label(prob):
    if prob>=0.70: return "HIGH"
    elif prob>=0.58: return "MEDIUM"
    return "LOW"


# -----------------------------------------------------------------------
# Player prop inference
# -----------------------------------------------------------------------

def build_player_features(player_name: str, opp_team: str, is_home: bool, game_date: str) -> Optional[np.ndarray]:
    if state.player_log_cache is None or not state.prop_feat_cols:
        return None

    df = state.player_log_cache
    cutoff = pd.Timestamp(game_date)

    player_df = df[df["PLAYER_NAME"].str.lower() == player_name.lower()]
    if len(player_df) == 0:
        # Try partial match
        player_df = df[df["PLAYER_NAME"].str.lower().str.contains(player_name.lower())]
    if len(player_df) == 0:
        return None

    past = player_df[player_df["GAME_DATE"] < cutoff].sort_values("GAME_DATE")
    if len(past) < 3:
        return None

    row = {}
    stat_cols = ["PTS","REB","AST","STL","BLK","TOV","MIN","FGM","FGA","FG3M","FPTS"]

    for col in stat_cols:
        if col not in past.columns:
            continue
        for w in [3,5,10]:
            row[f"ROLL{w}_{col}"] = past[col].tail(w).mean()
        row[f"STD5_{col}"] = past[col].tail(5).std() if len(past)>=5 else 0.0

    row["REST_DAYS"] = min((cutoff - past["GAME_DATE"].iloc[-1]).days, 10)
    row["IS_HOME"] = int(is_home)
    row["GAME_NUM"] = len(past) + 1
    row["FORM_WIN_RATE"] = (past["RESULT"].tail(5) == "W").mean() if "RESULT" in past.columns else 0.5

    if "FGM" in past.columns and "FGA" in past.columns:
        fga = past["FGA"].tail(5).mean()
        row["ROLL5_FG_PCT"] = (past["FGM"].tail(5).mean() / fga) if fga > 0 else 0.45

    # Opponent defense
    if state.game_log_cache is not None:
        opp_df = state.game_log_cache
        opp_past = opp_df[(opp_df["TEAM_ABBREVIATION"]==opp_team) & (opp_df["GAME_DATE"]<cutoff)].tail(5)
        row["OPP_PTS_ALLOWED"] = opp_past["PTS"].mean() if len(opp_past)>=3 else 110.0

    vec = np.array([row.get(f, 0.0) for f in state.prop_feat_cols], dtype=np.float32)
    return vec.reshape(1,-1)


def grade_start_sit(proj_fpts: float, avg_fpts: float) -> dict:
    diff = proj_fpts - avg_fpts
    pct = diff / max(avg_fpts, 1.0)
    if pct >= 0.15:
        return {"grade": "A", "recommendation": "START", "reason": f"Projected {proj_fpts:.1f} FPTS (+{diff:.1f} vs avg)"}
    elif pct >= 0.05:
        return {"grade": "B", "recommendation": "START", "reason": f"Projected {proj_fpts:.1f} FPTS, slight edge"}
    elif pct >= -0.10:
        return {"grade": "C", "recommendation": "FLEX", "reason": f"Projected {proj_fpts:.1f} FPTS, near average"}
    else:
        return {"grade": "D", "recommendation": "SIT", "reason": f"Projected {proj_fpts:.1f} FPTS ({diff:.1f} vs avg)"}


# -----------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------

class GamePredictRequest(BaseModel):
    home_team: str
    away_team: str
    date: Optional[str] = None
    spread: Optional[float] = None
    over_under: Optional[float] = None

class BatchPredictRequest(BaseModel):
    date: Optional[str] = None

class PropRequest(BaseModel):
    player_name: str = Field(..., example="Jayson Tatum")
    opp_team: str    = Field(..., example="MIL")
    is_home: bool    = Field(True)
    date: Optional[str] = None
    pts_line: Optional[float] = Field(None, description="Vegas over/under for points")
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


# -----------------------------------------------------------------------
# Routes — Game Winner
# -----------------------------------------------------------------------

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
        games = await get_espn_games()
        return {"date": date.today().isoformat(), "games": games, "count": len(games)}
    except Exception as e:
        raise HTTPException(502, f"ESPN API error: {e}")

@app.post("/predict/game", response_model=PredictionResult)
async def predict_game(req: GamePredictRequest):
    if state.model is None:
        raise HTTPException(503, "Model not loaded.")
    game_date = req.date or date.today().isoformat()
    home = req.home_team.upper()
    away = req.away_team.upper()
    features = build_game_features(home, away, game_date)
    home_prob = float(state.calibrated.predict_proba(features)[0][1]) if features is not None else 0.585
    away_prob = 1.0 - home_prob
    winner = home if home_prob >= 0.5 else away
    return PredictionResult(
        home_team=home, away_team=away,
        home_win_prob=round(home_prob,4), away_win_prob=round(away_prob,4),
        predicted_winner=winner, confidence=confidence_label(max(home_prob,away_prob)),
    )

@app.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    if state.model is None:
        raise HTTPException(503, "Model not loaded.")
    game_date = req.date or date.today().isoformat()
    try:
        games = await get_espn_games(game_date)
    except Exception as e:
        raise HTTPException(502, str(e))
    results = []
    for g in games:
        features = build_game_features(g["home_abbr"], g["away_abbr"], game_date)
        home_prob = float(state.calibrated.predict_proba(features)[0][1]) if features is not None else 0.585
        away_prob = 1.0 - home_prob
        winner = g["home_abbr"] if home_prob >= 0.5 else g["away_abbr"]
        results.append({
            "game_id": g["game_id"], "matchup": f"{g['away_abbr']} @ {g['home_abbr']}",
            "home_team": g["home_abbr"], "away_team": g["away_abbr"],
            "home_win_prob": round(home_prob,4), "away_win_prob": round(away_prob,4),
            "predicted_winner": winner, "confidence": confidence_label(max(home_prob,away_prob)),
            "spread": g["spread"], "over_under": g["over_under"], "status": g["status"],
        })
    return {"date": game_date, "predictions": results,
            "model_accuracy_season": state.eval_report.get("holdout",{}).get("accuracy")}

@app.get("/predict/live")
async def predict_live():
    if state.model is None:
        raise HTTPException(503, "Model not loaded.")
    games = await get_espn_games()
    live = [g for g in games if g["status"]=="in"]
    if not live:
        return {"message": "No live games right now.", "games": []}
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
            "confidence": confidence_label(max(adjusted,1.0-adjusted)),
        })
    return {"live_games": len(results), "predictions": results}


# -----------------------------------------------------------------------
# Routes — Player Props
# -----------------------------------------------------------------------

@app.post("/props/player")
async def predict_player_props(req: PropRequest):
    """Predict PTS, REB, AST, FPTS for one player with over/under analysis."""
    if not state.prop_models:
        raise HTTPException(503, "Prop models not loaded. Run 7_player_model.py first.")

    game_date = req.date or date.today().isoformat()
    features = build_player_features(req.player_name, req.opp_team, req.is_home, game_date)

    if features is None:
        raise HTTPException(404, f"Player '{req.player_name}' not found or insufficient history.")

    projections = {}
    for target, model in state.prop_models.items():
        proj = float(model.predict(features)[0])
        projections[target] = round(max(0, proj), 1)

    # Over/under analysis
    lines = {"PTS": req.pts_line, "REB": req.reb_line, "AST": req.ast_line}
    prop_picks = {}
    for stat, line in lines.items():
        if line is not None and stat in projections:
            proj = projections[stat]
            edge = proj - line
            pick = "OVER" if edge > 0 else "UNDER"
            confidence = min(95, max(50, 50 + abs(edge) * 5))
            prop_picks[stat] = {
                "line": line, "projection": proj, "pick": pick,
                "edge": round(edge, 1), "confidence": round(confidence),
            }

    # Fantasy points & start/sit
    avg_fpts = None
    if state.player_log_cache is not None:
        df = state.player_log_cache
        p_df = df[df["PLAYER_NAME"].str.lower().str.contains(req.player_name.lower())]
        if len(p_df) >= 5:
            avg_fpts = float(p_df["FPTS"].tail(10).mean())

    proj_fpts = projections.get("FPTS", 0)
    start_sit = grade_start_sit(proj_fpts, avg_fpts or proj_fpts * 0.95)

    return {
        "player": req.player_name,
        "opponent": req.opp_team,
        "is_home": req.is_home,
        "date": game_date,
        "projections": projections,
        "prop_picks": prop_picks,
        "fantasy": {
            "projected_fpts": proj_fpts,
            "avg_fpts_last10": round(avg_fpts, 1) if avg_fpts else None,
            "scoring": "DraftKings (PTS×1 + REB×1.25 + AST×1.5 + STL×2 + BLK×2 - TOV×0.5)",
        },
        "start_sit": start_sit,
    }


@app.get("/props/fantasy")
async def fantasy_lineup(date_str: Optional[str] = None):
    """Return fantasy recommendations for all players in tonight's games."""
    if not state.prop_models or state.player_log_cache is None:
        raise HTTPException(503, "Prop models not loaded.")

    game_date = date_str or date.today().isoformat()
    try:
        games = await get_espn_games(game_date)
    except Exception as e:
        raise HTTPException(502, str(e))

    # Get all teams playing tonight
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

    if not active_teams:
        return {"message": "No games today.", "players": []}

    # Score all players on active teams
    df = state.player_log_cache
    players_tonight = df[df["PLAYER_TEAM"].isin(active_teams)]["PLAYER_NAME"].unique()

    results = []
    for player_name in players_tonight:
        p_df = df[df["PLAYER_NAME"]==player_name]
        if len(p_df) < 5:
            continue
        team = p_df["PLAYER_TEAM"].iloc[-1]
        opp = team_to_opp.get(team, "")
        is_home = team_home.get(team, True)

        features = build_player_features(player_name, opp, is_home, game_date)
        if features is None:
            continue

        proj_fpts = float(state.prop_models["FPTS"].predict(features)[0]) if "FPTS" in state.prop_models else 0
        proj_pts  = float(state.prop_models["PTS"].predict(features)[0])  if "PTS"  in state.prop_models else 0
        proj_reb  = float(state.prop_models["REB"].predict(features)[0])  if "REB"  in state.prop_models else 0
        proj_ast  = float(state.prop_models["AST"].predict(features)[0])  if "AST"  in state.prop_models else 0
        avg_fpts  = float(p_df["FPTS"].tail(10).mean())
        start_sit = grade_start_sit(proj_fpts, avg_fpts)

        results.append({
            "player": player_name, "team": team, "opponent": opp, "home": is_home,
            "proj_pts": round(max(0,proj_pts),1), "proj_reb": round(max(0,proj_reb),1),
            "proj_ast": round(max(0,proj_ast),1), "proj_fpts": round(max(0,proj_fpts),1),
            "avg_fpts_last10": round(avg_fpts,1),
            "grade": start_sit["grade"],
            "recommendation": start_sit["recommendation"],
        })

    results.sort(key=lambda x: x["proj_fpts"], reverse=True)
    return {
        "date": game_date,
        "games": len(games),
        "players": results,
        "top_plays": [r for r in results if r["grade"] == "A"][:10],
    }


@app.get("/props/starts")
async def start_sit_recommendations(date_str: Optional[str] = None):
    """Simplified start/sit list for fantasy players."""
    data = await fantasy_lineup(date_str)
    players = data.get("players", [])
    return {
        "date": data.get("date"),
        "must_start": [p for p in players if p["recommendation"]=="START" and p["grade"]=="A"][:10],
        "start":      [p for p in players if p["recommendation"]=="START" and p["grade"]=="B"][:10],
        "flex":       [p for p in players if p["recommendation"]=="FLEX"][:8],
        "sit":        [p for p in players if p["recommendation"]=="SIT"][:8],
    }


from pathlib import Path

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("5_api_server:app", host="0.0.0.0", port=port, reload=False, log_level="info")