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
    df["GAME_NUM"] = df.groupby(["PLAYER_ID", "SEASON"]).cumcount() + 1
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

        # ── NEW #1: Opponent PACE (possessions ≈ counting-stat opportunity) ──
        # More possessions → more shots/rebounds/assists available to everyone.
        # Pace ≈ FGA + 0.44*FTA - OREB + TOV  (standard possessions estimate).
        if all(c in team_df.columns for c in ["FGA","FTA","OREB","TOV"]):
            team_df["POSS"] = (
                team_df["FGA"] + 0.44*team_df["FTA"]
                - team_df["OREB"] + team_df["TOV"]
            )
            opp_pace = (
                team_df.groupby(["TEAM_ABBREVIATION","GAME_DATE"])["POSS"]
                .mean().reset_index()
            )
            pace_roll = {}
            for team, grp in opp_pace.groupby("TEAM_ABBREVIATION"):
                grp = grp.sort_values("GAME_DATE")
                pace_roll[team] = (grp.set_index("GAME_DATE")["POSS"]
                                   .rolling(5, min_periods=2).mean().shift(1))

            def get_opp_pace(row):
                opp = row.get("OPP","")
                date = row["GAME_DATE"]
                if opp in pace_roll:
                    s = pace_roll[opp]
                    idx = s.index.searchsorted(date, side="left")
                    if idx > 0:
                        return float(s.iloc[idx-1])
                return 99.0  # league-average possessions
            df["OPP_PACE"] = df.apply(get_opp_pace, axis=1)
            feat_cols.append("OPP_PACE")

    # ── NEW #2: Usage / role trend (heating up vs cooling down) ──
    # ROLL3 / ROLL10 ratio per stat. >1 = recent role expanding, <1 = shrinking.
    # Captures rotation changes the season-long average misses.
    for col in ["PTS", "MIN", "FGA"]:
        r3, r10 = f"ROLL3_{col}", f"ROLL10_{col}"
        if r3 in df.columns and r10 in df.columns:
            tname = f"TREND_{col}"
            df[tname] = (df[r3] / df[r10].replace(0, np.nan)).clip(0.3, 3.0).fillna(1.0)
            feat_cols.append(tname)

    # ── NEW #3: Player back-to-back & heavy schedule ──
    # Stars often see reduced minutes on the 2nd night of a B2B.
    df["IS_B2B"] = (df["REST_DAYS"] <= 1).astype(int)
    feat_cols.append("IS_B2B")
    df["GAMES_LAST_7D"] = (
        df.groupby("PLAYER_ID")["GAME_DATE"]
        .transform(lambda s: s.diff().dt.days.le(7).rolling(4, min_periods=1).sum())
        .fillna(0)
    )
    feat_cols.append("GAMES_LAST_7D")

    # ── NEW #4: Minutes-anchored expectation ──
    # Counting stats scale ~linearly with minutes. Anchor on a robust
    # minutes estimate (recent + season blend) so a minutes change moves
    # the projection directly instead of being one weak signal among many.
    if "ROLL5_MIN" in df.columns and "ROLL10_MIN" in df.columns:
        df["MIN_PROJ"] = (0.6*df["ROLL5_MIN"] + 0.4*df["ROLL10_MIN"]).fillna(
            df.get("ROLL10_MIN", 0)
        )
        feat_cols.append("MIN_PROJ")
        # Per-minute production rate (last 5) — separates rate from volume.
        for col in ["PTS", "REB", "AST"]:
            r5 = f"ROLL5_{col}"
            if r5 in df.columns:
                rate = f"{col}_PER_MIN"
                df[rate] = (df[r5] / df["ROLL5_MIN"].replace(0, np.nan)).clip(0, 2.0).fillna(0)
                feat_cols.append(rate)

    df[feat_cols] = df[feat_cols].fillna(0.0)
    console.print(f"  {len(feat_cols)} features built")
    return df, feat_cols


def _cv_r2(X, y, n_splits=5):
    """Mean time-series CV R² for a given feature matrix."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, te in tscv.split(X):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        scores.append(r2_score(y[te], m.predict(X[te])))
    return float(np.mean(scores))


# The four features added in the pace/usage/B2B/minutes upgrade. Some help
# certain stats (AST loves pace+usage) and hurt others (REB regressed with
# them). We test per-stat and keep them only if CV says they help THAT stat.
NEW_FEATURE_PREFIXES = ("OPP_PACE", "TREND_", "IS_B2B", "GAMES_LAST_7D",
                        "MIN_PROJ", "_PER_MIN")


def select_features_for_target(df: pd.DataFrame, target: str,
                               feat_cols: list[str]) -> list[str]:
    """
    Per-stat feature selection. Compare CV R² with vs without the new
    feature block. Keep the new features only if they don't hurt THIS stat.
    This is why REB will automatically shed the features that regressed it
    while AST keeps the ones that gave it +0.13 R².
    """
    valid = df.dropna(subset=[target])
    valid = valid[valid[target] >= 0]
    y = valid[target].values

    base_cols = [c for c in feat_cols
                 if not any(c.startswith(p) or c.endswith("_PER_MIN")
                            for p in NEW_FEATURE_PREFIXES)]

    full_r2 = _cv_r2(valid[feat_cols].values, y)
    base_r2 = _cv_r2(valid[base_cols].values, y)

    if full_r2 >= base_r2 - 0.002:   # new features help (or are neutral)
        console.print(
            f"  [{target}] full set CV R²={full_r2:.3f} ≥ base {base_r2:.3f} "
            f"→ [green]keeping new features[/green]"
        )
        return feat_cols
    else:
        console.print(
            f"  [{target}] full set CV R²={full_r2:.3f} < base {base_r2:.3f} "
            f"→ [yellow]dropping new features for this stat[/yellow]"
        )
        return base_cols


def train_prop_model(df: pd.DataFrame, target: str, feat_cols: list[str]) -> dict:
    console.print(f"\nTraining [cyan]{target}[/cyan] model...")

    # Per-stat feature selection (fixes the REB regression automatically).
    feat_cols = select_features_for_target(df, target, feat_cols)

    valid = df.dropna(subset=[target])
    valid = valid[valid[target] >= 0]
    X = valid[feat_cols].values
    y = valid[target].values

    # ── Time-series cross-validation (honest generalization estimate) ──
    # Single-split R² can be optimistic or pessimistic depending on which
    # slice you happen to test on — same lesson as the game-model leak fix.
    # 5 expanding-window folds give the real picture.
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes, cv_r2s = [], []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        p = m.predict(X[te])
        fold_mae = mean_absolute_error(y[te], p)
        fold_r2  = r2_score(y[te], p)
        cv_maes.append(fold_mae)
        cv_r2s.append(fold_r2)
        console.print(
            f"  Fold {fold}: MAE={fold_mae:.2f}  R²={fold_r2:.3f}  (n={len(te):,})"
        )

    cv_mae  = float(np.mean(cv_maes))
    cv_r2   = float(np.mean(cv_r2s))
    cv_r2_sd = float(np.std(cv_r2s))
    console.print(
        f"  [bold]CV mean: MAE=[green]{cv_mae:.2f}[/green]  "
        f"R²=[green]{cv_r2:.3f}[/green] ± {cv_r2_sd:.3f}[/bold]  ← honest number"
    )

    # Final model: train on first 85% chronologically, validate last 15% as a
    # last sanity check, then ship. CV mean above is the number to trust.
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    r2  = r2_score(y_val, preds)

    console.print(
        f"  Single-split holdout: MAE={mae:.2f}  R²={r2:.3f}  (n={len(X_val):,}) "
        f"[dim](sanity only — trust the CV mean)[/dim]"
    )

    path = MODEL_DIR / f"prop_{target.lower()}.json"
    model.save_model(str(path))

    return {
        "target": target,
        "mae": round(cv_mae, 3),          # report CV as the headline
        "r2": round(cv_r2, 3),
        "r2_std": round(cv_r2_sd, 3),
        "holdout_mae": round(mae, 3),     # kept for transparency
        "holdout_r2": round(r2, 3),
        "n_val": len(X_val),
        "metric_source": "cv_mean",
        "features": feat_cols,            # per-stat selected feature list
    }


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

    # Save feature lists. Each stat may use a different set now (REB drops
    # the new features, AST keeps them), so we save BOTH a per-stat map and
    # the union list (for backward-compat with anything reading the old key).
    per_stat_features = {r["target"]: r["features"] for r in results}
    (MODEL_DIR / "prop_feature_list.json").write_text(json.dumps(feat_cols))
    (MODEL_DIR / "prop_features_by_stat.json").write_text(
        json.dumps(per_stat_features, indent=2)
    )
    # Strip the bulky feature list out of eval (keep eval readable)
    for r in results:
        r.pop("features", None)
    (MODEL_DIR / "prop_eval.json").write_text(json.dumps(results, indent=2))

    console.print("\n[bold green]All prop models saved:[/bold green]")
    for r in results:
        console.print(f"  {r['target']:6s} — MAE: {r['mae']:.2f}  R²: {r['r2']:.3f}")

    console.print("\n[bold orange1]Done! Add prop endpoints to your API and push to Railway.[/bold orange1]")


if __name__ == "__main__":
    main()
