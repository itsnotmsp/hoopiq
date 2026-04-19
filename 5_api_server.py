"""
Step 5 — FastAPI Prediction Server
-------------------------------------
Production-ready REST API for NBA game winner predictions.
Loads trained XGBoost model, pulls live data, returns predictions.

Endpoints:
  GET  /health                     — liveness check
  GET  /games/today                — today's games with live scores
  POST /predict/game               — predict winner for a matchup
  POST /predict/batch              — predict all games for a date
  GET  /predict/live               — live game predictions + current score
  GET  /model/info                 — model metadata & accuracy

Usage:
    pip install -r requirements.txt
    python 5_api_server.py

    # With auto-reload for development:
    uvicorn 5_api_server:app --reload --port 8000

Test:
    curl http://localhost:8000/games/today
    curl -X POST http://localhost:8000/predict/game \
         -H "Content-Type: application/json" \
         -d '{"home_team": "BOS", "away_team": "MIL"}'
"""

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
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hoopiq")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR  = Path("models")
DATA_DIR   = Path("data")
MODEL_PATH = MODEL_DIR / "xgb_model.json"
CALIB_PATH = MODEL_DIR / "calibrator.joblib"
FEAT_LIST  = MODEL_DIR / "feature_list.json"
EVAL_PATH  = MODEL_DIR / "eval_report.json"

# ---------------------------------------------------------------------------
# ESPN API
# ---------------------------------------------------------------------------
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
HTTP_HEADERS    = {"User-Agent": "HoopIQ/1.0", "Accept": "application/json"}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="HoopIQ Prediction API",
    description="AI-powered NBA game predictions backed by XGBoost + live data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
class ModelState:
    model: Optional[xgb.XGBClassifier] = None
    calibrated = None
    feature_cols: list[str] = []
    eval_report: dict = {}
    game_log_cache: Optional[pd.DataFrame] = None
    last_cache_date: Optional[date] = None

state = ModelState()


# ---------------------------------------------------------------------------
# Startup: load model + feature data
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def load_model():
    if MODEL_PATH.exists():
        state.model = xgb.XGBClassifier()
        state.model.load_model(str(MODEL_PATH))
        state.calibrated = joblib.load(CALIB_PATH)
        state.feature_cols = json.loads(FEAT_LIST.read_text())
        state.eval_report = json.loads(EVAL_PATH.read_text()) if EVAL_PATH.exists() else {}
        log.info(f"Model loaded — {len(state.feature_cols)} features")
    else:
        log.warning("No trained model found at models/xgb_model.json. Run 4_train_model.py first.")

    if (DATA_DIR / "game_logs.parquet").exists():
        df = pd.read_parquet(DATA_DIR / "game_logs.parquet")
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        state.game_log_cache = df
        log.info(f"Game log cache loaded — {len(df):,} rows")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class GamePredictRequest(BaseModel):
    home_team: str = Field(..., example="BOS", description="3-letter team abbreviation")
    away_team: str = Field(..., example="MIL")
    date: Optional[str]  = Field(None, example="2024-04-18", description="ISO date, defaults to today")
    spread: Optional[float] = Field(None, description="Vegas spread (home perspective, e.g. -5.5)")
    over_under: Optional[float] = Field(None, description="Over/under total")


class BatchPredictRequest(BaseModel):
    date: Optional[str] = Field(None, description="ISO date YYYY-MM-DD, defaults to today")


class PredictionResult(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    predicted_winner: str
    confidence: str          # HIGH / MEDIUM / LOW
    model_version: str = "xgb_v1"


# ---------------------------------------------------------------------------
# Feature builder for inference
# ---------------------------------------------------------------------------

def build_inference_features(home: str, away: str, game_date: str) -> Optional[np.ndarray]:
    """
    Build the same features the model was trained on, but for a single matchup.
    Pulls rolling stats from the cached game log.
    Returns None if insufficient data.
    """
    if state.game_log_cache is None:
        return None

    df = state.game_log_cache
    cutoff = pd.Timestamp(game_date)

    def team_features(abbr: str, is_home: bool) -> dict:
        team_df = df[df["TEAM_ABBREVIATION"] == abbr]
        past = team_df[team_df["GAME_DATE"] < cutoff].sort_values("GAME_DATE")

        if len(past) < 3:
            return {}

        last5  = past.tail(5)
        last10 = past.tail(10)

        feat = {}
        stat_cols = ["PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT",
                     "FTM","FTA","FT_PCT","OREB","DREB","REB","AST","STL","BLK","TOV","PF","PLUS_MINUS"]

        for col in stat_cols:
            if col in past.columns:
                feat[f"ROLL5_{col}"]  = last5[col].mean()
                feat[f"ROLL10_{col}"] = last10[col].mean()

        # Rest days
        last_game = past["GAME_DATE"].iloc[-1]
        feat["REST_DAYS"] = min((cutoff - last_game).days, 10)

        # Home/away splits
        home_df = past[past.get("IS_HOME", pd.Series(True, index=past.index)) == True].tail(10)
        away_df = past[past.get("IS_HOME", pd.Series(False, index=past.index)) == False].tail(10)
        feat["HOME_WIN_RATE"] = (home_df["WL"] == "W").mean() if len(home_df) >= 3 else 0.5
        feat["AWAY_WIN_RATE"] = (away_df["WL"] == "W").mean() if len(away_df) >= 3 else 0.5
        feat["HOME_AVG_PTS"]  = home_df["PTS"].mean() if len(home_df) >= 3 else past["PTS"].mean()
        feat["AWAY_AVG_PTS"]  = away_df["PTS"].mean() if len(away_df) >= 3 else past["PTS"].mean()

        # Season form
        feat["FORM_WIN_RATE"] = (last10["WL"] == "W").mean() if len(last10) >= 3 else 0.5
        feat["FORM_NET_RTG"]  = last10["PLUS_MINUS"].mean() if len(last10) >= 3 else 0.0

        return feat

    def h2h_features(team: str, opp: str) -> dict:
        team_df = df[(df["TEAM_ABBREVIATION"] == team) & (df["OPP_ABBR"] == opp)]
        past_h2h = team_df[team_df["GAME_DATE"] < cutoff].tail(10)
        if len(past_h2h) < 2:
            return {"H2H_WIN_RATE": 0.5, "H2H_AVG_MARGIN": 0.0}
        return {
            "H2H_WIN_RATE": (past_h2h["WL"] == "W").mean(),
            "H2H_AVG_MARGIN": past_h2h["PLUS_MINUS"].mean(),
        }

    h_raw = team_features(home, True)
    a_raw = team_features(away, False)
    h2h   = h2h_features(home, away)

    if not h_raw or not a_raw:
        return None

    row = {}
    for col, val in h_raw.items():
        row[f"H_{col}"] = val
    for col, val in a_raw.items():
        row[f"A_{col}"] = val
    for col, val in h2h.items():
        row[f"H_{col}"] = val

    # Diff features
    for col in h_raw:
        h_key = f"H_{col}"
        a_key = f"A_{col}"
        if h_key in row and a_key in row:
            row[f"DIFF_{col}"] = row[h_key] - row[a_key]

    # Align to model feature list, fill missing with 0
    vec = np.array([row.get(f, 0.0) for f in state.feature_cols], dtype=np.float32)
    return vec.reshape(1, -1)


def confidence_label(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.58:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# ESPN helpers (async)
# ---------------------------------------------------------------------------

async def get_espn_games(game_date: Optional[str] = None) -> list[dict]:
    params = {}
    if game_date:
        params["dates"] = game_date.replace("-", "")
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(ESPN_SCOREBOARD, params=params, headers=HTTP_HEADERS)
        r.raise_for_status()
    events = r.json().get("events", [])
    games = []
    for ev in events:
        comp = ev.get("competitions", [{}])[0]
        comps = comp.get("competitors", [])
        home = next((c for c in comps if c.get("homeAway") == "home"), {})
        away = next((c for c in comps if c.get("homeAway") == "away"), {})
        status = comp.get("status", {}).get("type", {})
        odds_list = comp.get("odds", [])
        odds = odds_list[0] if odds_list else {}
        games.append({
            "game_id":    ev.get("id"),
            "home_abbr":  home.get("team", {}).get("abbreviation", ""),
            "away_abbr":  away.get("team", {}).get("abbreviation", ""),
            "home_score": int(home.get("score", 0)),
            "away_score": int(away.get("score", 0)),
            "home_record": home.get("records", [{}])[0].get("summary", "") if home.get("records") else "",
            "away_record": away.get("records", [{}])[0].get("summary", "") if away.get("records") else "",
            "status":      status.get("state", "pre"),
            "status_desc": status.get("description", ""),
            "period":      comp.get("status", {}).get("period", 0),
            "clock":       comp.get("status", {}).get("displayClock", ""),
            "spread":      odds.get("details", ""),
            "over_under":  odds.get("overUnder"),
            "venue":       comp.get("venue", {}).get("fullName", ""),
            "date":        ev.get("date", ""),
        })
    return games


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "cache_rows": len(state.game_log_cache) if state.game_log_cache is not None else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/info")
async def model_info():
    if not state.eval_report:
        raise HTTPException(503, "Model not loaded")
    return {
        "n_features": len(state.feature_cols),
        "evaluation": state.eval_report.get("holdout", {}),
        "cv": state.eval_report.get("cv", {}).get("summary", {}),
        "params": state.eval_report.get("params", {}),
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
        raise HTTPException(503, "Model not loaded. Run 4_train_model.py first.")

    game_date = req.date or date.today().isoformat()
    home = req.home_team.upper()
    away = req.away_team.upper()

    features = build_inference_features(home, away, game_date)

    if features is None:
        # Fall back to a league-average home-court advantage estimate
        log.warning(f"Insufficient history for {away}@{home} — using home court prior")
        home_prob = 0.585
    else:
        probs = state.calibrated.predict_proba(features)[0]
        home_prob = float(probs[1])

    away_prob = 1.0 - home_prob
    winner = home if home_prob >= 0.5 else away

    return PredictionResult(
        home_team=home,
        away_team=away,
        home_win_prob=round(home_prob, 4),
        away_win_prob=round(away_prob, 4),
        predicted_winner=winner,
        confidence=confidence_label(max(home_prob, away_prob)),
    )


@app.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    if state.model is None:
        raise HTTPException(503, "Model not loaded.")

    game_date = req.date or date.today().isoformat()

    try:
        games = await get_espn_games(game_date)
    except Exception as e:
        raise HTTPException(502, f"ESPN fetch failed: {e}")

    results = []
    for g in games:
        features = build_inference_features(g["home_abbr"], g["away_abbr"], game_date)
        if features is None:
            home_prob = 0.585
        else:
            probs = state.calibrated.predict_proba(features)[0]
            home_prob = float(probs[1])

        away_prob = 1.0 - home_prob
        winner = g["home_abbr"] if home_prob >= 0.5 else g["away_abbr"]

        results.append({
            "game_id": g["game_id"],
            "matchup": f"{g['away_abbr']} @ {g['home_abbr']}",
            "home_team": g["home_abbr"],
            "away_team": g["away_abbr"],
            "home_win_prob": round(home_prob, 4),
            "away_win_prob": round(away_prob, 4),
            "predicted_winner": winner,
            "confidence": confidence_label(max(home_prob, away_prob)),
            "spread": g["spread"],
            "over_under": g["over_under"],
            "status": g["status"],
        })

    return {
        "date": game_date,
        "predictions": results,
        "model_accuracy_season": state.eval_report.get("holdout", {}).get("accuracy"),
    }


@app.get("/predict/live")
async def predict_live():
    """Return predictions for all currently in-progress games."""
    if state.model is None:
        raise HTTPException(503, "Model not loaded.")

    games = await get_espn_games()
    live = [g for g in games if g["status"] == "in"]

    if not live:
        return {"message": "No live games right now.", "games": []}

    today = date.today().isoformat()
    results = []
    for g in live:
        features = build_inference_features(g["home_abbr"], g["away_abbr"], today)
        home_prob = float(state.calibrated.predict_proba(features)[0][1]) if features is not None else 0.585

        score_diff = g["home_score"] - g["away_score"]
        score_adjustment = np.tanh(score_diff / 12.0) * 0.10
        adjusted_prob = float(np.clip(home_prob + score_adjustment, 0.05, 0.95))

        results.append({
            "game_id": g["game_id"],
            "matchup": f"{g['away_abbr']} @ {g['home_abbr']}",
            "score": f"{g['away_score']} - {g['home_score']}",
            "period": g["period"],
            "clock": g["clock"],
            "pre_game_home_prob": round(home_prob, 4),
            "live_home_prob": round(adjusted_prob, 4),
            "predicted_winner": g["home_abbr"] if adjusted_prob >= 0.5 else g["away_abbr"],
            "confidence": confidence_label(max(adjusted_prob, 1.0 - adjusted_prob)),
        })

    return {"live_games": len(results), "predictions": results}


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", 8000))
    uvicorn.run("5_api_server:app", host="0.0.0.0", port=port, reload=False, log_level="info")
