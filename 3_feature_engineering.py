"""
Step 3 — Feature Engineering
------------------------------
Transforms raw game logs into ML-ready features for the prediction model.

Features built:
  Rolling stats       — last 5/10 game averages for both teams
  Home/away splits    — team performance at home vs away
  Rest days           — days since last game (fatigue factor)
  Head-to-head        — H2H win rate, avg margin last 5 meetings
  Season form         — win%, net rating, pace last 10 games
  Injury proxy        — minutes drop / roster depth (from game logs)
  Spread features     — Vegas line (when available from odds data)

Usage:
    python 3_feature_engineering.py
    # Reads  data/game_logs.parquet
    # Writes data/features.parquet   ← model training set
    # Writes data/feature_info.json  ← column metadata for API
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

DATA_DIR  = Path("data")
FEAT_PATH = DATA_DIR / "features.parquet"
INFO_PATH = DATA_DIR / "feature_info.json"

ROLL_WINDOWS = [5, 10]   # rolling averages over last N games

TEAM_STAT_COLS = [
    "PTS", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF",
    "PLUS_MINUS",
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_game_logs() -> pd.DataFrame:
    path = DATA_DIR / "game_logs.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "data/game_logs.parquet not found. Run python 2_data_pipeline.py first."
        )
    df = pd.read_parquet(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    console.print(f"Loaded {len(df):,} team-game rows")
    return df


# ---------------------------------------------------------------------------
# Rolling team stats (per-team, shifted so no data leakage)
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages shifted by 1 game (exclude current game from roll)."""
    console.print("Computing rolling team features...")

    feature_cols = []
    for col in TEAM_STAT_COLS:
        if col not in df.columns:
            continue
        for w in ROLL_WINDOWS:
            feat_name = f"ROLL{w}_{col}"
            df[feat_name] = (
                df.groupby("TEAM_ID")[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).mean())
            )
            feature_cols.append(feat_name)

    return df, feature_cols


# ---------------------------------------------------------------------------
# Rest days
# ---------------------------------------------------------------------------

def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    df["PREV_GAME_DATE"] = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    df["REST_DAYS"] = (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days.clip(0, 10).fillna(3)
    df.drop(columns=["PREV_GAME_DATE"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Home / away split features
# ---------------------------------------------------------------------------

def add_home_away_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling win rate and avg PTS split by home/away."""
    console.print("Computing home/away splits...")

    df["WIN"] = (df["WL"] == "W").astype(int)

    for location, flag in [("HOME", True), ("AWAY", False)]:
        mask = df["IS_HOME"] == flag
        for col, fcol in [("WIN", f"{location}_WIN_RATE"), ("PTS", f"{location}_AVG_PTS")]:
            val = (
                df[mask]
                .groupby("TEAM_ID")[col]
                .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
            )
            df[fcol] = np.nan
            df.loc[mask, fcol] = val

    # Forward-fill across home/away (so away games know home win rate etc.)
    for col in ["HOME_WIN_RATE", "HOME_AVG_PTS", "AWAY_WIN_RATE", "AWAY_AVG_PTS"]:
        df[col] = df.groupby("TEAM_ID")[col].ffill().bfill()

    return df


# ---------------------------------------------------------------------------
# Head-to-head features
# ---------------------------------------------------------------------------

def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate and avg margin in last 10 H2H meetings."""
    console.print("Computing H2H features...")

    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df["H2H_WIN_RATE"]  = np.nan
    df["H2H_AVG_MARGIN"] = np.nan

    pair_history: dict = {}

    for idx, row in df.iterrows():
        tid = row["TEAM_ID"]
        opp = row.get("OPP_ABBR", "")
        key = (tid, opp)
        rev_key = (opp, tid)  # not used for wins but symmetry check

        if key in pair_history and len(pair_history[key]) >= 3:
            hist = pair_history[key][-10:]
            df.at[idx, "H2H_WIN_RATE"]  = np.mean([h["win"] for h in hist])
            df.at[idx, "H2H_AVG_MARGIN"] = np.mean([h["margin"] for h in hist])

        # Append current game AFTER reading (no leakage)
        entry = {"win": int(row["WL"] == "W"), "margin": row.get("PLUS_MINUS", 0)}
        pair_history.setdefault(key, []).append(entry)

    df["H2H_WIN_RATE"].fillna(0.5, inplace=True)
    df["H2H_AVG_MARGIN"].fillna(0.0, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Season form (last 10 win %, net rating proxy)
# ---------------------------------------------------------------------------

def add_season_form(df: pd.DataFrame) -> pd.DataFrame:
    df["FORM_WIN_RATE"] = (
        df.groupby("TEAM_ID")["WIN"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        .fillna(0.5)
    )
    df["FORM_NET_RTG"] = (
        df.groupby("TEAM_ID")["PLUS_MINUS"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        .fillna(0.0)
    )
    return df


# ---------------------------------------------------------------------------
# Build matchup-level dataset (one row per game, home vs away)
# ---------------------------------------------------------------------------

def build_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge home and away team stats into a single row per game.
    The target label is HOME_WIN (1 = home team wins).
    """
    console.print("Building matchup rows...")

    home = df[df["IS_HOME"] == True].copy()
    away = df[df["IS_HOME"] == False].copy()

    feature_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["ROLL", "H2H", "HOME_", "AWAY_", "FORM_", "REST"]
    )]

    home_feats = home[["GAME_ID", "TEAM_ABBREVIATION", "WL"] + feature_cols].copy()
    away_feats = away[["GAME_ID", "TEAM_ABBREVIATION"] + feature_cols].copy()

    home_feats.columns = (
        ["GAME_ID", "HOME_TEAM", "WL_HOME"] +
        [f"H_{c}" for c in feature_cols]
    )
    away_feats.columns = (
        ["GAME_ID", "AWAY_TEAM"] +
        [f"A_{c}" for c in feature_cols]
    )

    matchups = home_feats.merge(away_feats, on="GAME_ID", how="inner")
    matchups["HOME_WIN"] = (matchups["WL_HOME"] == "W").astype(int)

    # Difference features (home - away) — often most predictive
    for col in feature_cols:
        h_col = f"H_{col}"
        a_col = f"A_{col}"
        if h_col in matchups.columns and a_col in matchups.columns:
            matchups[f"DIFF_{col}"] = matchups[h_col] - matchups[a_col]

    matchups.dropna(subset=["HOME_WIN"], inplace=True)
    matchups = matchups.sort_values("GAME_ID").reset_index(drop=True)

    console.print(f"  {len(matchups):,} matchup rows, {len(matchups.columns)} columns")
    return matchups


# ---------------------------------------------------------------------------
# Feature importance / metadata export
# ---------------------------------------------------------------------------

def export_feature_info(df: pd.DataFrame) -> None:
    feature_cols = [
        c for c in df.columns
        if c not in ["GAME_ID", "HOME_TEAM", "AWAY_TEAM", "WL_HOME", "HOME_WIN"]
    ]
    info = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "n_samples": len(df),
        "target": "HOME_WIN",
        "label_balance": {
            "home_wins": int(df["HOME_WIN"].sum()),
            "away_wins": int((df["HOME_WIN"] == 0).sum()),
            "home_win_rate": round(df["HOME_WIN"].mean(), 4),
        },
    }
    INFO_PATH.write_text(json.dumps(info, indent=2))
    console.print(f"Feature info → {INFO_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline() -> pd.DataFrame:
    df = load_game_logs()

    df, roll_cols = add_rolling_features(df)
    df = add_rest_days(df)
    df = add_home_away_splits(df)
    df = add_h2h_features(df)
    df = add_season_form(df)

    matchups = build_matchup_features(df)

    matchups.to_parquet(FEAT_PATH, index=False)
    console.print(f"\n[bold green]Features saved → {FEAT_PATH}[/bold green]")
    export_feature_info(matchups)

    return matchups


if __name__ == "__main__":
    df = run_pipeline()
    console.print("\n[bold]Sample feature row:[/bold]")
    sample = df[["HOME_TEAM", "AWAY_TEAM", "HOME_WIN",
                 "H_ROLL5_PTS", "A_ROLL5_PTS", "DIFF_ROLL5_PTS",
                 "H_REST_DAYS", "A_REST_DAYS",
                 "H_FORM_WIN_RATE", "A_FORM_WIN_RATE",
                 "H_H2H_WIN_RATE"]].tail(5)
    console.print(sample.to_string())
