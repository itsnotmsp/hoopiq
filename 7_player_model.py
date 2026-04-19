"""
Step 7 — Player Prop Models (XGBoost)
---------------------------------------
Trains separate XGBoost regressors for PTS, REB, AST, and FPTS.
Generates over/under predictions, fantasy scores, and start/sit grades.

Usage:
    python 7_player_model.py            # train all models
    python 7_player_model.py --eval     # evaluate saved models

Output:
    models/prop_pts.json         — points model
    models/prop_reb.json         — rebounds model
    models/prop_ast.json         — assists model
    models/prop_fpts.json        — fantasy points model
    models/prop_feature_list.json
    models/prop_eval.json
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
warnings.filterwarnings("ignore")

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGETS = ["PTS", "REB", "AST", "FPTS"]

ROLL_WINDOWS = [3, 5, 10]

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


def load_player_logs() -> pd.DataFrame:
    path = DATA_DIR / "player_logs.parquet"
    if not path.exists():
        raise FileNotFoundError("data/player_logs.parquet not found. Run 6_player_pipeline.py first.")
    df = pd.read_parquet(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    console.print(f"Loaded {len(df):,} player-game rows, {df['PLAYER_NAME'].nunique()} players")
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    console.print("Engineering player features...")
    feat_cols = []

    stat_cols = ["PTS","REB","AST","STL","BLK","TOV","MIN","FGM","FGA","FG3M","FPTS"]

    for col in stat_cols:
        if col not in df.columns:
            continue
        for w in ROLL_WINDOWS:
            name = f"ROLL{w}_{col}"
            df[name] = (
                df.groupby("PLAYER_ID")[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=max(1, w//2)).mean())
            )
            feat_cols.append(name)

        # Standard deviation (consistency metric)
        std_name = f"STD5_{col}"
        df[std_name] = (
            df.groupby("PLAYER_ID")[col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std().fillna(0))
        )
        feat_cols.append(std_name)

    # Rest days
    df["REST_DAYS"] = (
        df.groupby("PLAYER_ID")["GAME_DATE"]
        .transform(lambda x: x.diff().dt.days.clip(0, 10).fillna(3))
    )
    feat_cols.append("REST_DAYS")

    # Home/away
    df["IS_HOME"] = df["HOME"].astype(int)
    feat_cols.append("IS_HOME")

    # Season game number (fatigue over season)
    df["GAME_NUM"] = df.groupby(["PLAYER_ID", "SEASON_YEAR"]).cumcount() + 1
    feat_cols.append("GAME_NUM")

    # Win rate (team playing well = more min, better stats)
    df["WIN"] = (df["RESULT"] == "W").astype(int)
    df["FORM_WIN_RATE"] = (
        df.groupby("PLAYER_ID")["WIN"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean().fillna(0.5))
    )
    feat_cols.append("FORM_WIN_RATE")

    # FG% trend
    if "FGM" in df.columns and "FGA" in df.columns:
        df["FG_PCT"] = df["FGM"] / df["FGA"].replace(0, np.nan)
        df["ROLL5_FG_PCT"] = (
            df.groupby("PLAYER_ID")["FG_PCT"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean().fillna(0.45))
        )
        feat_cols.append("ROLL5_FG_PCT")

    # Opponent defensive features (from team game logs if available)
    team_path = DATA_DIR / "game_logs.parquet"
    if team_path.exists():
        team_df = pd.read_parquet(team_path)
        team_df["GAME_DATE"] = pd.to_datetime(team_df["GAME_DATE"])
        # Opponent avg points allowed (rolling)
        opp_pts = (
            team_df.groupby(["TEAM_ABBREVIATION","GAME_DATE"])["PTS"].mean().reset_index()
        )
        opp_pts_roll = {}
        for team, grp in opp_pts.groupby("TEAM_ABBREVIATION"):
            grp = grp.sort_values("GAME_DATE")
            opp_pts_roll[team] = grp.set_index("GAME_DATE")["PTS"].rolling(5, min_periods=2).mean().shift(1)

        def get_opp_pts_allowed(row):
            opp = row.get("OPP","")
            date = row["GAME_DATE"]
            if opp in opp_pts_roll:
                series = opp_pts_roll[opp]
                idx = series.index.searchsorted(date, side="left")
                if idx > 0:
                    return float(series.iloc[idx-1])
            return 110.0  # league average

        df["OPP_PTS_ALLOWED"] = df.apply(get_opp_pts_allowed, axis=1)
        feat_cols.append("OPP_PTS_ALLOWED")

    df[feat_cols] = df[feat_cols].fillna(0.0)
    console.print(f"  {len(feat_cols)} features built")
    return df, feat_cols


def train_prop_model(df: pd.DataFrame, target: str, feat_cols: list[str]) -> dict:
    console.print(f"\nTraining [cyan]{target}[/cyan] model...")

    valid = df.dropna(subset=[target])
    valid = valid[valid[target] >= 0]
    X = valid[feat_cols].values
    y = valid[target].values

    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    r2  = r2_score(y_val, preds)

    console.print(f"  MAE: [green]{mae:.2f}[/green]  R²: [green]{r2:.3f}[/green]  (n={len(X_val):,})")

    path = MODEL_DIR / f"prop_{target.lower()}.json"
    model.save_model(str(path))

    return {"target": target, "mae": round(mae,3), "r2": round(r2,3), "n_val": len(X_val)}


def print_importance(target: str, feat_cols: list[str]) -> None:
    path = MODEL_DIR / f"prop_{target.lower()}.json"
    if not path.exists():
        return
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    scores = model.feature_importances_
    top = sorted(zip(feat_cols, scores), key=lambda x: x[1], reverse=True)[:8]

    table = Table(title=f"{target} Top Features", box=box.SIMPLE, show_header=True)
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", justify="right")
    max_s = top[0][1] if top else 1
    for feat, score in top:
        table.add_row(feat, f"{score:.4f}", "█" * int(score/max_s*20))
    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    df = load_player_logs()
    df, feat_cols = engineer_features(df)

    results = []
    for target in TARGETS:
        if target not in df.columns:
            console.print(f"[yellow]Skipping {target} — column not found[/yellow]")
            continue
        metrics = train_prop_model(df, target, feat_cols)
        results.append(metrics)
        print_importance(target, feat_cols)

    # Save feature list and eval
    (MODEL_DIR / "prop_feature_list.json").write_text(json.dumps(feat_cols))
    (MODEL_DIR / "prop_eval.json").write_text(json.dumps(results, indent=2))

    console.print("\n[bold green]All prop models saved:[/bold green]")
    for r in results:
        console.print(f"  {r['target']:6s} — MAE: {r['mae']:.2f}  R²: {r['r2']:.3f}")

    console.print("\n[bold orange1]Done! Add prop endpoints to your API and push to Railway.[/bold orange1]")


if __name__ == "__main__":
    main()
