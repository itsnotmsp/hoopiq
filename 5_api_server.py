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
ESPN_INJURIES   = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
HTTP_HEADERS    = {"User-Agent": "HoopIQ/1.0", "Accept": "application/json"}

# Simple in-process injury cache (ESPN updates a few times daily; 30 min TTL).
_injury_cache = {"data": None, "fetched_at": 0.0}


async def fetch_injuries() -> dict:
    """
    Live NBA injuries from ESPN's (free, no-auth) injuries endpoint.

    Returns: { "TEAM_ABBR": [ {name, status, detail}, ... ] }
    status is typically "Out", "Day-To-Day", "Out (Injury Management)", etc.

    NOTE: this is LIVE context only — it is NOT a trained model feature.
    Historical per-game injury data would be required for that, which ESPN
    does not expose. Use this to eyeball-adjust bets, not as model accuracy.
    """
    import time
    now = time.time()
    if _injury_cache["data"] is not None and (now - _injury_cache["fetched_at"]) < 1800:
        return _injury_cache["data"]

    out: dict = {}
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(ESPN_INJURIES, headers=HTTP_HEADERS)
            r.raise_for_status()
            data = r.json()
        # ESPN shape: { injuries: [ { team:{abbreviation}, injuries:[ {athlete:{displayName}, status, details:{type}} ] } ] }
        for team_block in data.get("injuries", []):
            team = (team_block.get("team") or {}).get("abbreviation", "")
            if not team:
                continue
            entries = []
            for inj in team_block.get("injuries", []):
                ath = (inj.get("athlete") or {}).get("displayName", "")
                status = inj.get("status", "") or (inj.get("type") or {}).get("description", "")
                detail = (inj.get("details") or {}).get("type", "") or inj.get("shortComment", "")
                if ath:
                    entries.append({"name": ath, "status": status, "detail": detail})
            if entries:
                out[team.upper()] = entries
    except Exception as e:
        log.warning(f"injury fetch failed: {e}")
        out = {}

    _injury_cache["data"] = out
    _injury_cache["fetched_at"] = now
    return out

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
    prop_features_by_stat: dict = {}
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
    p = MODEL_DIR / "prop_features_by_stat.json"
    if p.exists(): state.prop_features_by_stat = json.loads(p.read_text())
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
        # EWM recency feature — MUST match training:
        #   training = groupby(PLAYER).transform(x.shift(1).ewm(halflife=3,
        #              min_periods=2).mean()) then the row's value.
        # `past` is already all games strictly before the cutoff (= shift(1)),
        # so the EWM of past[c] with halflife=3 and its LAST value reproduces
        # the training-time number exactly. Omitting this was the bug that
        # made every projection ~half the player's real average.
        if len(past) >= 2:
            row[f"EWM_{c}"] = past[c].ewm(halflife=3, min_periods=2).mean().iloc[-1]
        else:
            row[f"EWM_{c}"] = past[c].mean()

    # TREND_* = ROLL3 / ROLL10 ratio (role expanding vs shrinking), clipped
    # to [0.3, 3.0] then NaN→1.0, exactly as in training.
    for c in ["PTS", "MIN", "FGA"]:
        r3, r10 = row.get(f"ROLL3_{c}"), row.get(f"ROLL10_{c}")
        if r3 is not None and r10:
            row[f"TREND_{c}"] = float(np.clip(r3 / r10, 0.3, 3.0)) if r10 else 1.0
        else:
            row[f"TREND_{c}"] = 1.0

    # Player schedule/fatigue (training: REST_DAYS<=1 → B2B; games in last 7d)
    last_date = past["GAME_DATE"].iloc[-1]
    _rest = min((cutoff - last_date).days, 10)
    row["IS_B2B"] = int(_rest <= 1)
    row["GAMES_LAST_7D"] = int((past["GAME_DATE"] >= (cutoff - pd.Timedelta(days=7))).sum())

    # Minutes anchor + per-minute production rates (training formulas exactly)
    r5m, r10m = row.get("ROLL5_MIN"), row.get("ROLL10_MIN")
    if r5m is not None and r10m is not None:
        row["MIN_PROJ"] = 0.6*r5m + 0.4*r10m
        for c in ["PTS", "REB", "AST"]:
            r5 = row.get(f"ROLL5_{c}")
            if r5 is not None and r5m:
                row[f"{c}_PER_MIN"] = float(np.clip(r5 / r5m, 0, 2.0)) if r5m else 0.0
            else:
                row[f"{c}_PER_MIN"] = 0.0
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
        # OPP_PACE: possessions ≈ FGA + 0.44*FTA - OREB + TOV, last-5 mean.
        # Matches training; default to league-average 99 if data missing.
        if len(opp_past) >= 2 and all(c in opp_past.columns for c in ["FGA","FTA","OREB","TOV"]):
            poss = (opp_past["FGA"] + 0.44*opp_past["FTA"]
                    - opp_past["OREB"] + opp_past["TOV"])
            row["OPP_PACE"] = float(poss.mean())
        else:
            row["OPP_PACE"] = 99.0
    else:
        row["OPP_PACE"] = 99.0
    # Return the full feature dict. Per-stat models now use different feature
    # subsets (REB dropped the pace/usage block, AST kept it), so the caller
    # builds the right vector per model from prop_features_by_stat.
    vec = np.array([row.get(f, 0.0) for f in state.prop_feat_cols], dtype=np.float32)
    return vec.reshape(1,-1), row


def _vec_for_stat(row: dict, stat: str):
    """Build the feature vector a specific prop model expects, using its
    saved per-stat feature list. Falls back to the union list."""
    feats = state.prop_features_by_stat.get(stat) or state.prop_feat_cols
    return np.array([row.get(f, 0.0) for f in feats], dtype=np.float32).reshape(1, -1)


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
    er = state.eval_report or {}
    cv_summary = (er.get("cv") or {}).get("summary", {})
    # Headline accuracy = CV mean (honest generalization estimate), falling
    # back to holdout only if CV is unavailable. The old code surfaced the
    # single-slice holdout, which over-claimed by ~7 points.
    headline = er.get("headline_accuracy")
    if headline is None:
        headline = cv_summary.get("accuracy_mean") or (er.get("holdout") or {}).get("accuracy")
    return {
        "n_features": len(state.feature_cols),
        "headline_accuracy": headline,
        "headline_source": er.get("headline_source", "cv_mean"),
        "evaluation": er.get("holdout", {}),   # kept for transparency / debugging
        "cv": cv_summary,
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
    return {"date": game_date, "predictions": results, "model_accuracy_season": (state.eval_report.get("headline_accuracy") or (state.eval_report.get("cv") or {}).get("summary",{}).get("accuracy_mean"))}


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
    _fr = build_player_features(req.player_name, req.opp_team, req.is_home, game_date)
    if _fr is None: raise HTTPException(404, f"Player not found or insufficient history")
    _, _row = _fr
    projections = {t: round(max(0, float(m.predict(_vec_for_stat(_row, t))[0])), 1)
                   for t,m in state.prop_models.items()}
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
        _fr = build_player_features(name, opp, is_home, game_date)
        if _fr is None: continue
        _, _row = _fr
        proj_fpts = float(state.prop_models["FPTS"].predict(_vec_for_stat(_row,"FPTS"))[0]) if "FPTS" in state.prop_models else 0
        proj_pts = float(state.prop_models["PTS"].predict(_vec_for_stat(_row,"PTS"))[0]) if "PTS" in state.prop_models else 0
        proj_reb = float(state.prop_models["REB"].predict(_vec_for_stat(_row,"REB"))[0]) if "REB" in state.prop_models else 0
        proj_ast = float(state.prop_models["AST"].predict(_vec_for_stat(_row,"AST"))[0]) if "AST" in state.prop_models else 0
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

    # Try Vegas lines (optional) — pulls player props for up to 5 upcoming games.
    # Each game costs 3 Odds-API requests (one per market). Cached 30 min.
    vegas_lines = {}      # (player_lower, stat_key) -> line
    vegas_prices = {}     # (player_lower, stat_key) -> {"over_prices": {bm: american}, "under_prices": {...}}

    def _amer_to_dec(am):
        try:
            n = float(am)
        except (TypeError, ValueError):
            return None
        if n == 0:
            return None
        return round((n / 100 + 1) if n > 0 else (100 / abs(n) + 1), 2)

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("odds_mod", "10_odds_integration.py")
        odds_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(odds_mod)
        odds_games = await odds_mod.fetch_game_odds()
        for og in odds_games[:5]:
            try:
                props = await odds_mod.fetch_player_props(og["game_id"])
                for stat_key, plist in props.get("props", {}).items():
                    for pname, info in plist.items():
                        key = (pname.lower(), stat_key)
                        vegas_lines[key] = info["line"]
                        # Pick best (highest) price per side as the displayed odds,
                        # mirroring how a bettor would shop the line.
                        over_prices  = info.get("over", {})  or {}
                        under_prices = info.get("under", {}) or {}
                        best_over_am  = max(over_prices.values(),  key=lambda x: x or -9999) if over_prices  else None
                        best_under_am = max(under_prices.values(), key=lambda x: x or -9999) if under_prices else None
                        vegas_prices[key] = {
                            "over_decimal":  _amer_to_dec(best_over_am),
                            "under_decimal": _amer_to_dec(best_under_am),
                            "over_american":  best_over_am,
                            "under_american": best_under_am,
                            "bookmakers": sorted(set(list(over_prices.keys()) + list(under_prices.keys()))),
                        }
            except Exception:
                continue
    except Exception:
        pass

    # Pull live injuries ONCE for this slate. Players who are OUT must never
    # be recommended (the bet auto-loses — they don't play). Day-to-day /
    # questionable players are kept but flagged and barred from high-confidence.
    inj_all = await fetch_injuries()
    def _injury_status(player_name, team_abbr):
        for e in inj_all.get(team_abbr, []):
            if e["name"].lower() == player_name.lower():
                s = (e.get("status") or "").lower()
                if "out" in s:
                    return "OUT", e
                if "day" in s or "question" in s or "doubt" in s:
                    return "RISK", e
                return "RISK", e   # any listed status → treat as risky
        return "OK", None

    picks = []
    for name in players:
        p_df = df[df["PLAYER_NAME"]==name]
        if len(p_df)<10: continue
        team = p_df["PLAYER_TEAM"].iloc[-1]
        # ── INJURY GATE ──
        inj_state, inj_entry = _injury_status(name, team)
        if inj_state == "OUT":
            continue   # never recommend a prop for a player who isn't playing
        opp = team_to_opp.get(team,""); is_home = team_home.get(team, True)
        _fr = build_player_features(name, opp, is_home, game_date)
        if _fr is None: continue
        _, _row = _fr
        projections = {s: float(m.predict(_vec_for_stat(_row, s))[0]) for s,m in state.prop_models.items()}
        last5 = p_df.tail(5); last10 = p_df.tail(10)
        for stat in ["PTS","REB","AST"]:
            if stat not in projections: continue
            proj = projections[stat]
            if proj < 5: continue
            avg10 = float(last10[stat].mean()); avg5 = float(last5[stat].mean())
            std5 = float(last5[stat].std()) if len(last5)>=3 else proj*0.2
            # Wider, more honest volatility read: spread over last 10 games,
            # not just 5. A player like Wembanyama (4,39,19,27,19,...) has a
            # huge std — his "average" describes none of his games and any
            # projection is a meaningless midpoint of a bimodal range.
            vals10 = last10[stat].dropna().astype(float)
            std10 = float(vals10.std()) if len(vals10) >= 4 else std5
            game_range = (float(vals10.max() - vals10.min())
                          if len(vals10) >= 4 else std5 * 3)

            stat_key = {"PTS":"POINTS","REB":"REBOUNDS","AST":"ASSISTS"}[stat]
            vegas_line = vegas_lines.get((name.lower(), stat_key))

            # ── UNPROJECTABLE-PLAYER FILTER ──
            # The bet is only meaningful if the gap between projection and line
            # is BIGGER than the player's own game-to-game noise. If a player
            # swings ±13 pts night to night and the line is 4 pts from the
            # projection, OVER/UNDER is a coin flip no matter how good the
            # projection model is — the outcome lives entirely inside the
            # noise. We REFUSE to emit a pick in that case rather than dress
            # up a coin flip as "high confidence".
            if vegas_line:
                edge_abs = abs(proj - vegas_line)
                # Require the edge to clear 75% of one standard deviation AND
                # the player not to be wildly bimodal (range < 4x std would be
                # roughly normal; far above that = explosive/dud pattern).
                noise = max(std10, 1e-6)
                if edge_abs < 0.75 * noise:
                    continue  # edge drowned by the player's own variance
                if game_range > 6.0 * noise and noise > 3.0:
                    continue  # bimodal boom/bust player — unprojectable
                # Hard volatility ceiling for scoring: a PTS std over ~9 means
                # explosive scorers (stars who go 5 or 40). Skip unless the
                # edge is enormous (line very far from projection).
                if stat == "PTS" and std10 > 9.0 and edge_abs < 1.25 * std10:
                    continue

            form_trend = (avg5-avg10)/max(avg10,1.0)
            consistency = 1.0 - min(std10/max(avg5,1.0), 1.0)
            edge = None; edge_pct = 0; pick_side = None
            if vegas_line:
                edge = proj - vegas_line; edge_pct = (edge/vegas_line)*100 if vegas_line else 0
                pick_side = "OVER" if edge>0 else "UNDER"
            confidence = min(99, max(0, consistency*50 + max(-15,min(15,form_trend*100)) + (abs(edge_pct)*1.5 if edge else 0) + 20))
            if vegas_line:
                if abs(edge_pct)<3: continue
                rec = ("STRONG " if abs(edge_pct)>=10 else "")+pick_side
            else:
                continue  # No real sportsbook line; skip phantom picks
            # Build human-friendly reasons
            stat_word = {"PTS":"points","REB":"rebounds","AST":"assists"}[stat]
            reasons = []
            simple = ""

            if vegas_line:
                if pick_side == "OVER":
                    simple = f"Bet {name} to score MORE than {vegas_line} {stat_word} tonight"
                else:
                    simple = f"Bet {name} to score LESS than {vegas_line} {stat_word} tonight"
                reasons.append(f"Sportsbook line is {vegas_line} {stat_word}, but model projects {proj:.1f}")
            else:
                simple = f"Model expects {name} to score over {round(avg10)} {stat_word}"

            if avg5 > avg10 * 1.05:
                reasons.append(f"Heating up: {avg5:.1f} {stat_word} last 5 games vs {avg10:.1f} season avg")
            elif avg5 < avg10 * 0.95:
                reasons.append(f"Cooling off: only {avg5:.1f} {stat_word} last 5 vs {avg10:.1f} season avg")
            else:
                reasons.append(f"Consistent: {avg5:.1f} {stat_word} L5, {avg10:.1f} L10")

            if consistency > 0.75:
                reasons.append(f"Reliable performer ({round(consistency*100)}% consistency)")
            elif consistency < 0.5:
                reasons.append(f"⚠️ Risky pick — only {round(consistency*100)}% consistency")

            if vegas_line and abs(edge_pct) >= 10:
                if pick_side == "OVER":
                    reasons.append(f"Big edge: projecting {abs(edge_pct):.0f}% over the line")
                else:
                    reasons.append(f"Big edge: projecting {abs(edge_pct):.0f}% below the line")

            if is_home:
                reasons.append("Playing at home (small boost)")
            else:
                reasons.append("Road game (slight headwind)")

            confidence_word = ("Very high confidence" if confidence>=80 else
                             "Good confidence" if confidence>=65 else
                             "Moderate confidence" if confidence>=50 else
                             "Low confidence — bet small")

            # High-confidence flag: pick only fires when the model has a
            # real edge AND the player is reliable. Combines:
            #  • model confidence ≥ 65
            #  • |edge_pct| ≥ 15 (real gap to Vegas, not 2% noise)
            #  • consistency ≥ 65 (player hits their average reliably)
            # Bets matching all three should hit higher than the open set.
            cons_pct = round(consistency * 100)
            high_conf = (
                confidence >= 65
                and (edge_pct is not None and abs(edge_pct) >= 15)
                and cons_pct >= 65
                and inj_state == "OK"   # day-to-day players can't be high-confidence
            )

            # Surface the injury risk prominently in the reasons list.
            if inj_state == "RISK" and inj_entry:
                reasons.insert(0,
                    f"⚠️ INJURY RISK — {inj_entry['name']} listed "
                    f"{inj_entry.get('status','day-to-day')}. Minutes uncertain; "
                    f"projection may not hold.")

            picks.append({
                "player": name, "team": team, "opponent": opp, "home": is_home,
                "stat": stat, "stat_label": stat_word,
                "projection": round(proj,1), "vegas_line": vegas_line,
                "edge": round(edge,2) if edge is not None else None,
                "edge_pct": round(edge_pct,1) if vegas_line else None,
                "pick": pick_side, "recommendation": rec, "confidence": round(confidence),
                "confidence_label": confidence_word,
                "high_confidence": high_conf,
                "injury_status": inj_state,   # "OK" or "RISK" (OUT players already filtered)
                "injury_detail": (
                    f"{inj_entry['name']} — {inj_entry.get('status','')}"
                    if inj_entry else None
                ),
                "simple_explanation": simple,
                "reasons": reasons[:5],
                "form": {"avg_last_5": round(avg5,1), "avg_last_10": round(avg10,1),
                         "trending": "up" if form_trend>0.05 else "down" if form_trend<-0.05 else "flat",
                         "consistency": cons_pct},
                # Real Vegas prices when available; null otherwise. Frontend
                # uses these to populate the bet-logging modal.
                "odds": vegas_prices.get((name.lower(), stat_key), {
                    "over_decimal": None, "under_decimal": None,
                    "over_american": None, "under_american": None,
                    "bookmakers": [],
                }),
            })
    picks.sort(key=lambda p: (abs(p.get("edge_pct") or 0), p["confidence"], p["form"]["consistency"]), reverse=True)

    # ── All-OVER sanity check ──
    # If nearly every pick is OVER, the model is likely projecting high into
    # conservative (shaded) sportsbook lines rather than finding real edges.
    # A trustworthy slate is roughly balanced between overs and unders.
    top = picks[:10]
    over_n = sum(1 for p in top if (p.get("pick") or "").upper() == "OVER")
    bias_warning = None
    if len(top) >= 5:
        over_frac = over_n / len(top)
        if over_frac >= 0.85:
            bias_warning = (
                f"⚠️ {over_n}/{len(top)} picks are OVER. Sportsbooks shade prop "
                f"lines low and juice the over. An all-OVER slate usually means "
                f"the model is projecting high into shaded lines, NOT finding real "
                f"edges. Treat these as low-trust — bet tiny or skip."
            )
        elif over_frac <= 0.15:
            bias_warning = (
                f"⚠️ {len(top)-over_n}/{len(top)} picks are UNDER — unusually "
                f"one-sided. Same caution as an all-OVER slate."
            )

    return {"date": game_date, "games": len(games),
            "vegas_lines_loaded": len(vegas_lines)>0,
            "over_count": over_n, "slate_size": len(top),
            "bias_warning": bias_warning,
            "total_candidates": len(picks), "top_picks": top}


@app.get("/props/degen")
async def degen_props(
    min_decimal: float = 3.0,
    min_hit_rate: float = 0.25,
    min_edge: float = 0.10,
    limit: int = 15,
):
    """
    High-payout alt-line picks where the model thinks the empirical hit rate
    beats the breakeven implied by the offered odds.

    Math:
      breakeven_p = 1 / decimal_odds
      edge = empirical_p / breakeven_p - 1
    A pick is included only if `edge >= min_edge`. Default 10% edge over
    breakeven — conservative; raise it to be stricter.

    Params:
      min_decimal  — minimum decimal odds to consider (default 3.0 ≈ +200).
                     Lower this to include slightly safer lines.
      min_hit_rate — minimum empirical P(hit) from player's last 10 games.
                     Below this it's a true lottery ticket, not a +EV bet.
      min_edge     — empirical_p must exceed breakeven_p by this multiplier
                     (0.10 = 10% edge).
      limit        — top N picks to return, ranked by EV per $1 staked.

    Quota cost: 3 API requests per game on first call, then cached PROPS_TTL.
    """
    if state.player_log_cache is None:
        raise HTTPException(503, "Player log cache not loaded")

    df = state.player_log_cache

    # Load the odds module
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("odds_mod", "10_odds_integration.py")
        odds_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(odds_mod)
    except Exception as e:
        raise HTTPException(503, f"Odds module load failed: {e}")

    try:
        odds_games = await odds_mod.fetch_game_odds()
    except Exception as e:
        return {"degen_picks": [], "error": f"No games from odds: {e}"}

    if not odds_games:
        return {"degen_picks": [], "games": 0, "note": "No upcoming games."}

    def amer_to_dec(am):
        try: n = float(am)
        except (TypeError, ValueError): return None
        if n == 0: return None
        return round((n/100 + 1) if n > 0 else (100/abs(n) + 1), 3)

    STAT_COLS = {"POINTS": "PTS", "REBOUNDS": "REB", "ASSISTS": "AST"}

    # Live injuries — degen bets on an OUT player are guaranteed losses.
    inj_all = await fetch_injuries()
    # Flatten to a fast name→status lookup across all teams.
    injured_lookup = {}
    for _team, _entries in inj_all.items():
        for _e in _entries:
            s = (_e.get("status") or "").lower()
            injured_lookup[_e["name"].lower()] = (
                "OUT" if "out" in s else "RISK"
            )

    picks = []
    for og in odds_games[:5]:  # cap at 5 games to control quota burn
        try:
            alt = await odds_mod.fetch_player_props_alt(og["game_id"])
        except Exception:
            continue

        for stat_label, players in (alt.get("props") or {}).items():
            stat_col = STAT_COLS.get(stat_label)
            if not stat_col:
                continue

            for player_name, info in (players or {}).items():
                # ── INJURY GATE ── never surface a degen bet on an OUT player.
                inj = injured_lookup.get(player_name.lower())
                if inj == "OUT":
                    continue
                # Last 10 games for this player — empirical distribution.
                p_logs = df[df["PLAYER_NAME"].str.lower() == player_name.lower()]
                if len(p_logs) < 5:
                    continue
                last10 = p_logs.sort_values("GAME_DATE").tail(10)
                if stat_col not in last10.columns:
                    continue
                values = last10[stat_col].astype(float).values
                if len(values) < 8:
                    # Empirical hit rate off <8 games is too noisy to trust,
                    # especially for boom/bust players. Require a real sample.
                    continue

                avg_l10 = float(values.mean())
                std_l10 = float(values.std())
                # Coefficient of variation: std relative to mean. A player
                # with CV > ~0.6 is wildly inconsistent — a 10-game empirical
                # hit rate is mostly which explosions happened to land in the
                # window, not a real probability. Skip these entirely.
                cv = std_l10 / max(avg_l10, 1.0)
                if cv > 0.65:
                    continue

                for offer in info.get("lines", []):
                    point = offer.get("point")
                    side  = offer.get("side")  # "Over" or "Under"
                    if point is None or side not in ("Over", "Under"):
                        continue

                    # Best (highest) decimal odds across bookmakers for this leg.
                    prices = offer.get("prices") or {}
                    decimals = [amer_to_dec(p) for p in prices.values()]
                    decimals = [d for d in decimals if d is not None]
                    if not decimals:
                        continue
                    best_dec = max(decimals)
                    if best_dec < min_decimal:
                        continue

                    # Empirical P(hit): fraction of last-10 games where the
                    # side would have won at this line.
                    if side == "Over":
                        hits = int((values >  point).sum())
                    else:
                        hits = int((values <  point).sum())
                    p_hit = hits / len(values)
                    if p_hit < min_hit_rate:
                        continue

                    breakeven_p = 1.0 / best_dec
                    edge = (p_hit / breakeven_p) - 1.0
                    if edge < min_edge:
                        continue

                    # Expected value per $1 staked.
                    ev_per_dollar = p_hit * (best_dec - 1) - (1 - p_hit)

                    # Best bookmaker for this price (so user knows where to bet)
                    best_book = max(prices.items(),
                                    key=lambda kv: amer_to_dec(kv[1]) or 0)[0]

                    picks.append({
                        "player":       player_name,
                        "stat":         stat_col,
                        "stat_label":   stat_label.lower(),
                        "side":         side.upper(),       # OVER / UNDER
                        "line":         point,
                        "decimal_odds": best_dec,
                        "american_odds": prices[best_book],
                        "best_book":    best_book,
                        "all_prices":   prices,
                        "l10_avg":      round(avg_l10, 1),
                        "l10_hits":     hits,
                        "l10_games":    len(values),
                        "hit_rate":     round(p_hit * 100, 1),
                        "breakeven_rate": round(breakeven_p * 100, 1),
                        "edge_pct":     round(edge * 100, 1),
                        "ev_per_dollar": round(ev_per_dollar, 3),
                        "payout_on_100": round(100 * (best_dec - 1), 2),
                        "injury_risk":  inj == "RISK",   # day-to-day; minutes uncertain
                        "game_id":      og["game_id"],
                    })

    # Rank by EV per dollar (better proxy than raw edge — accounts for variance)
    picks.sort(key=lambda p: p["ev_per_dollar"], reverse=True)
    return {
        "games_scanned":   min(len(odds_games), 5),
        "total_candidates": len(picks),
        "min_decimal":      min_decimal,
        "min_hit_rate":     min_hit_rate,
        "min_edge":         min_edge,
        "degen_picks":      picks[:limit],
        "warning": (
            "These are high-variance bets. Use smaller stakes than your standard "
            "props (e.g. $25 instead of $100). +EV is realized over hundreds of "
            "bets, not dozens — expect cold streaks even when picks are good."
        ),
    }


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
# History is stored at HOOPIQ_HISTORY_DIR if set (used in production for a
# Railway Volume mounted at /data so predictions survive redeploys), otherwise
# alongside the bundled data files. Parquet/model files stay in DATA_DIR
# (they're read-only and ship with the container).
HISTORY_DIR  = Path(os.environ.get("HOOPIQ_HISTORY_DIR", str(DATA_DIR)))
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = HISTORY_DIR / "predictions_log.json"

# Seed the volume on first run: if the volume's history file is missing but
# the bundled data/predictions_log.json exists (from git), copy it across so
# you don't start from zero after enabling the volume.
_BUNDLED = DATA_DIR / "predictions_log.json"
if not HISTORY_PATH.exists() and _BUNDLED.exists() and HISTORY_PATH != _BUNDLED:
    HISTORY_PATH.write_text(_BUNDLED.read_text())
    print(f"[history] Seeded {HISTORY_PATH} from bundled {_BUNDLED}")


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
    # Betting fields ───────────────────────────────────────────────────────
    side: Optional[str] = None             # e.g. "BOS", "NYK", "OVER", "UNDER"
    odds_decimal: Optional[float] = None   # 1.85, 2.04, etc.
    stake: Optional[float] = None          # default $100, set in /history/log
    closing_decimal: Optional[float] = None  # captured automatically at game start


def compute_pl(record: dict) -> float:
    """Profit/loss for a resolved record. Pending → 0."""
    if record.get("result") not in ("WIN", "LOSS"):
        return 0.0
    stake = float(record.get("stake") or 0)
    if stake <= 0:
        return 0.0
    odds = record.get("odds_decimal")
    if record["result"] == "LOSS":
        return -stake
    # WIN
    if odds is None or odds <= 1:
        return 0.0  # no odds recorded → can't compute profit
    return round(stake * (odds - 1), 2)


class ResultUpdate(BaseModel):
    result: str  # "WIN", "LOSS", "PENDING"
    actual_score: Optional[str] = None
    actual_value: Optional[float] = None


@app.post("/history/log")
async def log_prediction(entry: HistoryEntry):
    """Log a prediction to track later."""
    records = load_history()
    payload = entry.dict(exclude_none=True)
    # Default $100 stake when odds were captured at log time. Without odds we
    # can't compute P/L on a win, so stake stays 0 unless caller set it.
    if payload.get("odds_decimal") and not payload.get("stake"):
        payload["stake"] = 100.0
    new_record = {
        "id": f"pred_{int(datetime.now().timestamp() * 1000)}",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "result": "PENDING",
        "profit_loss": 0.0,
        **payload,
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

    # CLV (closing-line value) — only entries where both opening (your) odds
    # and closing odds were captured. CLV per bet, in %:
    #   100 * (closing_decimal / opening_decimal - 1) for back side
    # We average across all bets that have both fields. Positive CLV means
    # you consistently beat the market — strongest long-term profitability
    # indicator known, more meaningful than win rate on small samples.
    clv_records = [r for r in records
                   if r.get("odds_decimal") and r.get("closing_decimal")
                   and r["odds_decimal"] > 1 and r["closing_decimal"] > 1]
    if clv_records:
        clv_pcts = [
            100 * (r["closing_decimal"] / r["odds_decimal"] - 1)
            for r in clv_records
        ]
        avg_clv = sum(clv_pcts) / len(clv_pcts)
        clv_beats = sum(1 for c in clv_pcts if c > 0)
        clv_beat_rate = round(100 * clv_beats / len(clv_pcts), 1)
    else:
        avg_clv = 0.0
        clv_beat_rate = 0.0

    return {
        "stats": {
            "total": len(records),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": round(win_rate, 1),
            "streak": streak,
            "streak_type": streak_type,
            # Betting aggregates (only entries with recorded stake count)
            "total_staked":   round(sum(float(r.get("stake") or 0)
                                        for r in completed), 2),
            "total_profit":   round(sum(float(r.get("profit_loss") or 0)
                                        for r in completed), 2),
            "pending_at_risk": round(sum(float(r.get("stake") or 0)
                                         for r in records
                                         if r.get("result") == "PENDING"), 2),
            "roi_pct": round(
                (sum(float(r.get("profit_loss") or 0) for r in completed)
                 / sum(float(r.get("stake") or 0) for r in completed) * 100)
                if sum(float(r.get("stake") or 0) for r in completed) > 0
                else 0,
                2,
            ),
            # Closing-line value
            "clv_pct": round(avg_clv, 2),
            "clv_beat_rate": clv_beat_rate,
            "clv_sample": len(clv_records),
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
            r["profit_loss"] = compute_pl(r)
            found = True
            break

    if not found:
        raise HTTPException(404, "Record not found")

    save_history(records)
    return {"updated": True, "id": record_id}


@app.post("/history/auto_update")
async def auto_update_history():
    """Check ESPN for finished games and auto-mark pending GAME predictions WIN/LOSS."""
    records = load_history()
    et = timezone(timedelta(hours=-5))

    updated = 0
    games_checked = set()
    final_games = {}

    # Find all unique game-prediction dates we need to check
    pending_games = [r for r in records
                     if r.get("type") == "game" and r.get("result") == "PENDING"]

    if not pending_games:
        return {"message": "No pending predictions to update", "updated": 0}

    # Get unique dates from pending records
    dates_to_check = set()
    for r in pending_games:
        try:
            logged_date = datetime.fromisoformat(r["logged_at"].replace("Z","+00:00")).astimezone(et).date()
            dates_to_check.add(logged_date.strftime("%Y%m%d"))
        except:
            pass

    # Fetch ESPN for each date
    for date_str in dates_to_check:
        try:
            games = await get_espn_games(date_str)
            for g in games:
                if g.get("status") == "post":  # final
                    home = g["home_abbr"]
                    away = g["away_abbr"]
                    home_score = g["home_score"]
                    away_score = g["away_score"]
                    actual_winner = home if home_score > away_score else away
                    key = (home, away)
                    final_games[key] = {
                        "actual_winner": actual_winner,
                        "score": f"{away_score}-{home_score}",
                    }
        except Exception:
            continue

    # Update pending records
    for r in records:
        if r.get("type") == "game" and r.get("result") == "PENDING":
            key = (r.get("home_team"), r.get("away_team"))
            if key in final_games:
                actual = final_games[key]
                # Bet resolves based on the user's chosen side (manual-choice
                # at log time); fall back to predicted_winner for legacy rows.
                bet_side = r.get("side") or r.get("predicted_winner")
                r["result"] = "WIN" if bet_side == actual["actual_winner"] else "LOSS"
                r["actual_score"] = actual["score"]
                r["actual_winner"] = actual["actual_winner"]
                r["resolved_at"] = datetime.now(timezone.utc).isoformat()
                r["profit_loss"] = compute_pl(r)
                updated += 1

    save_history(records)

    return {
        "updated": updated,
        "games_checked": len(final_games),
        "pending_remaining": sum(1 for r in records if r.get("result") == "PENDING"),
    }


@app.delete("/history/{record_id}")
async def delete_record(record_id: str):
    records = load_history()
    new_records = [r for r in records if r.get("id") != record_id]
    if len(new_records) == len(records):
        raise HTTPException(404, "Record not found")
    save_history(new_records)
    return {"deleted": True}


def _amer_to_dec_safe(am):
    """American → decimal odds. Returns None on bad input."""
    try:
        n = float(am)
    except (TypeError, ValueError):
        return None
    if n == 0:
        return None
    return round((n / 100 + 1) if n > 0 else (100 / abs(n) + 1), 4)


@app.post("/history/capture_closing_lines")
async def capture_closing_lines():
    """
    Capture the current sportsbook line as the 'closing line' for any pending
    bet that doesn't have one yet. Intended to be called shortly before / at
    tipoff so the line we record is genuinely the close.

    Why this matters: CLV (closing-line value = how much you beat the close
    by) is the single most predictive measure of long-term betting skill.
    A bettor who consistently gets +CLV will profit even with a sub-55% hit
    rate; a bettor who consistently gives up -CLV will lose even at 60%+.
    """
    records = load_history()
    pending_needing_clv = [
        r for r in records
        if r.get("result") == "PENDING"
        and r.get("odds_decimal")
        and not r.get("closing_decimal")
    ]
    if not pending_needing_clv:
        return {"updated": 0, "message": "No pending bets need a closing line."}

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("odds_mod", "10_odds_integration.py")
        odds_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(odds_mod)
        odds_games = await odds_mod.fetch_game_odds()
    except Exception as e:
        raise HTTPException(503, f"Odds module failed: {e}")

    # Build a fast lookup: full team name → odds payload for the game it's in.
    by_team = {}
    for og in odds_games:
        by_team[og["home_team"].lower()] = og
        by_team[og["away_team"].lower()] = og

    # Map common abbreviations to Odds API full names (mirrors backend ODDS_TEAM_NAMES)
    ODDS_TEAM_NAMES = {
        "ATL":"Atlanta Hawks","BOS":"Boston Celtics","BKN":"Brooklyn Nets","BRK":"Brooklyn Nets",
        "CHA":"Charlotte Hornets","CHO":"Charlotte Hornets","CHI":"Chicago Bulls",
        "CLE":"Cleveland Cavaliers","DAL":"Dallas Mavericks","DEN":"Denver Nuggets",
        "DET":"Detroit Pistons","GSW":"Golden State Warriors","GS":"Golden State Warriors",
        "HOU":"Houston Rockets","IND":"Indiana Pacers","LAC":"Los Angeles Clippers",
        "LAL":"Los Angeles Lakers","MEM":"Memphis Grizzlies","MIA":"Miami Heat",
        "MIL":"Milwaukee Bucks","MIN":"Minnesota Timberwolves",
        "NOP":"New Orleans Pelicans","NO":"New Orleans Pelicans",
        "NYK":"New York Knicks","NY":"New York Knicks",
        "OKC":"Oklahoma City Thunder","ORL":"Orlando Magic","PHI":"Philadelphia 76ers",
        "PHX":"Phoenix Suns","PHO":"Phoenix Suns","POR":"Portland Trail Blazers",
        "SAC":"Sacramento Kings","SAS":"San Antonio Spurs","SA":"San Antonio Spurs",
        "TOR":"Toronto Raptors","UTA":"Utah Jazz","UTAH":"Utah Jazz",
        "WAS":"Washington Wizards","WSH":"Washington Wizards",
    }

    updated = 0
    for r in pending_needing_clv:
        # Only game-type bets get auto-captured for now. Props CLV would need
        # a player-props API call per bet, which we don't want to spend quota
        # on; props CLV can be added manually later if needed.
        if r.get("type") != "game":
            continue
        side_abbr = r.get("side")
        if not side_abbr:
            continue
        side_full = ODDS_TEAM_NAMES.get(side_abbr, side_abbr)
        og = by_team.get(side_full.lower())
        if not og:
            continue
        ml_dict = og["moneyline"].get(side_full) or {}
        ml_consensus = ml_dict.get("consensus")
        closing_dec = _amer_to_dec_safe(ml_consensus)
        if closing_dec:
            r["closing_decimal"] = closing_dec
            r["closing_captured_at"] = datetime.now(timezone.utc).isoformat()
            updated += 1

    if updated:
        save_history(records)
    return {"updated": updated, "candidates": len(pending_needing_clv)}


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
        "model_accuracy_season": (state.eval_report.get("headline_accuracy") or (state.eval_report.get("cv") or {}).get("summary",{}).get("accuracy_mean")),
    }


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

    # Team abbreviation aliases (handles NYK↔NY, GSW↔GS, etc.)
    ALIAS = {
        "NYK": ["NYK","NY"], "NY": ["NY","NYK"],
        "BKN": ["BKN","BRK"], "BRK": ["BRK","BKN"],
        "GSW": ["GSW","GS"], "GS": ["GS","GSW"],
        "NOP": ["NOP","NO"], "NO": ["NO","NOP"],
        "PHX": ["PHX","PHO"], "PHO": ["PHO","PHX"],
        "SAS": ["SAS","SA"], "SA": ["SA","SAS"],
        "WAS": ["WAS","WSH"], "WSH": ["WSH","WAS"],
        "CHA": ["CHA","CHO"], "CHO": ["CHO","CHA"],
    }

    def team_logs(abbr, n=10):
        candidates = ALIAS.get(abbr, [abbr])
        t = df[df["TEAM_ABBREVIATION"].isin(candidates)]
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

    # ── 2. INJURIES (live ESPN data — context only, not a model feature) ──
    inj_all = await fetch_injuries()
    home_inj = inj_all.get(home, [])
    away_inj = inj_all.get(away, [])

    def _summarize(entries):
        # Highlight players who are fully OUT (biggest betting impact).
        outs = [e for e in entries
                if "out" in (e.get("status", "").lower())]
        dtd = [e for e in entries
               if "day" in (e.get("status", "").lower())]
        if not entries:
            return {"key_outs": [], "day_to_day": [],
                    "impact": "No reported injuries"}
        return {
            "key_outs": [f"{e['name']} ({e['status']})" for e in outs],
            "day_to_day": [f"{e['name']} ({e['status']})" for e in dtd],
            "impact": (
                f"{len(outs)} OUT, {len(dtd)} day-to-day"
                if (outs or dtd) else "Reported but status unclear"
            ),
        }

    injuries = {
        "home": _summarize(home_inj),
        "away": _summarize(away_inj),
        "note": (
            "Live ESPN injuries. This is decision context for YOU — it does "
            "not adjust the model's probability (no historical injury data to "
            "train on). Weigh key OUTs manually before betting."
        ),
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

    # Map team abbreviations (what users pass in) → Odds API full team names.
    # The Odds API uses full names like "New York Knicks", not abbreviations.
    ODDS_TEAM_NAMES = {
        "ATL": "Atlanta Hawks", "BOS": "Boston Celtics",
        "BKN": "Brooklyn Nets", "BRK": "Brooklyn Nets",
        "CHA": "Charlotte Hornets", "CHO": "Charlotte Hornets",
        "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
        "DET": "Detroit Pistons",
        "GSW": "Golden State Warriors", "GS": "Golden State Warriors",
        "HOU": "Houston Rockets", "IND": "Indiana Pacers",
        "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers",
        "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
        "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans", "NO":  "New Orleans Pelicans",
        "NYK": "New York Knicks",     "NY":  "New York Knicks",
        "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
        "PHI": "Philadelphia 76ers",
        "PHX": "Phoenix Suns",        "PHO": "Phoenix Suns",
        "POR": "Portland Trail Blazers",
        "SAC": "Sacramento Kings",
        "SAS": "San Antonio Spurs",   "SA":  "San Antonio Spurs",
        "TOR": "Toronto Raptors",
        "UTA": "Utah Jazz",           "UTAH": "Utah Jazz",
        "WAS": "Washington Wizards",  "WSH": "Washington Wizards",
    }

    home_full = ODDS_TEAM_NAMES.get(home, home)
    away_full = ODDS_TEAM_NAMES.get(away, away)

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("odds_mod", "10_odds_integration.py")
        odds_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(odds_mod)
        odds_games = await odds_mod.fetch_game_odds()

        # Match the game by team SET (order-independent), so the user can pass
        # home/away in either order and still get market data.
        wanted = {home_full.lower(), away_full.lower()}
        for og in odds_games:
            og_pair = {og["home_team"].lower(), og["away_team"].lower()}
            if og_pair != wanted:
                continue

            # Look up market values keyed by the USER's stated home/away — so
            # market.ml_home is always the moneyline for the team the user
            # passed as home_team, regardless of who is actually hosting.
            ml_home_dict = og["moneyline"].get(home_full) or {}
            ml_away_dict = og["moneyline"].get(away_full) or {}
            ml_home_consensus = ml_home_dict.get("consensus")
            ml_away_consensus = ml_away_dict.get("consensus")

            spread_dict_home = og["spread"].get(home_full) or {}
            spread_first = next((v for v in spread_dict_home.values() if isinstance(v, dict)), None)

            total_dict = og["total"].get("Over") or {}
            total_first = next((v for v in total_dict.values() if isinstance(v, dict)), None)

            # Note: og["home_team"] is the actual host (sportsbook's perspective),
            # which may differ from the user's home_team. Surfacing it lets the
            # frontend show "PHI hosting NYK" if the user queried NYK as home.
            market = {
                "available": True,
                "ml_home":  ml_home_consensus,
                "ml_away":  ml_away_consensus,
                "spread":   spread_first.get("point") if spread_first else None,
                "total":    total_first.get("point") if total_first else None,
                "actual_host": og["home_team"],
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

    # value_analysis fields are None when no market data was available, so guard
    # against that with `or {}` before chaining .get() calls.
    ml_v = value_analysis.get("moneyline_value") or {}
    if ml_v.get("side") and "edge_pct" in ml_v:
        best_bet_options.append({
            "type": "MONEYLINE",
            "side": ml_v["side"],
            "edge": ml_v.get("edge_pct"),
            "verdict": ml_v.get("verdict"),
        })

    tv = value_analysis.get("total_value") or {}
    if tv.get("side"):
        best_bet_options.append({
            "type": "TOTAL",
            "side": f"{tv['side']} {tv.get('market_total')}",
            "edge": tv.get("diff"),
            "verdict": tv.get("verdict"),
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
        "home_team": home,
        "away_team": away,
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
        "matchup_analysis": matchup,
        "pace": pace_analysis,
        "situational": situational,
        "market": market,
        "advanced": advanced,
        "value": value_analysis,
        "best_bet": best_bet_options[0],
        "all_bets": best_bet_options,
        "risk_factors": risk_factors,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("5_api_server:app", host="0.0.0.0", port=port, reload=False, log_level="info")
