"""
Step 4 — Model Training (XGBoost)
-----------------------------------
Trains an XGBoost classifier to predict NBA game winners.
Includes time-series cross-validation, hyperparameter tuning,
calibrated probabilities, and full evaluation report.

Usage:
    python 4_train_model.py             # train + evaluate
    python 4_train_model.py --tune      # run full hyperparam search (slow)
    python 4_train_model.py --eval-only # load saved model and evaluate

Output:
    models/xgb_model.json         — trained XGBoost model
    models/calibrator.joblib      — Platt scaling calibrator
    models/feature_list.json      — ordered feature list for inference
    models/eval_report.json       — accuracy, AUC, calibration metrics
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, brier_score_loss,
    log_loss, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
warnings.filterwarnings("ignore")

DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEAT_PATH   = DATA_DIR / "features.parquet"
INFO_PATH   = DATA_DIR / "feature_info.json"
MODEL_PATH  = MODEL_DIR / "xgb_model.json"
CALIB_PATH  = MODEL_DIR / "calibrator.joblib"
FEAT_LIST   = MODEL_DIR / "feature_list.json"
EVAL_PATH   = MODEL_DIR / "eval_report.json"

# Default hyperparameters (strong baseline, no tuning needed)
DEFAULT_PARAMS = {
    "n_estimators": 600,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.80,
    "colsample_bytree": 0.70,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


# ---------------------------------------------------------------------------
# Load features
# ---------------------------------------------------------------------------

def load_features() -> tuple[pd.DataFrame, list[str], str]:
    if not FEAT_PATH.exists():
        raise FileNotFoundError("data/features.parquet not found. Run 3_feature_engineering.py first.")

    df = pd.read_parquet(FEAT_PATH)
    info = json.loads(INFO_PATH.read_text())

    feature_cols = [c for c in info["feature_columns"] if c in df.columns]

    # Drop rows where the target is missing, but fill NaN features with 0
    # (balldontlie doesn't supply box-score stats, so many cols are NaN —
    #  we keep all rows and let XGBoost handle the zeros gracefully)
    df = df.dropna(subset=["HOME_WIN"])
    df[feature_cols] = df[feature_cols].fillna(0.0)

    console.print(f"Loaded {len(df):,} matchups, {len(feature_cols)} features")
    return df, feature_cols, "HOME_WIN"


# ---------------------------------------------------------------------------
# Time-series CV evaluation (never test on future data seen during train)
# ---------------------------------------------------------------------------

def time_series_cv(df: pd.DataFrame, feature_cols: list[str], target: str, n_splits: int = 5) -> dict:
    """Walk-forward validation: train on past, test on next block."""
    console.print(f"\nRunning {n_splits}-fold time-series CV...")

    X = df[feature_cols].values
    y = df[target].values

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)  # gap=5 avoids overlap at fold boundary
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBClassifier(**DEFAULT_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        metrics = {
            "fold": fold + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "auc": round(roc_auc_score(y_test, probs), 4),
            "log_loss": round(log_loss(y_test, probs), 4),
            "brier": round(brier_score_loss(y_test, probs), 4),
        }
        fold_metrics.append(metrics)

        console.print(
            f"  Fold {fold+1}: acc={metrics['accuracy']:.3f}  "
            f"AUC={metrics['auc']:.3f}  logloss={metrics['log_loss']:.3f}"
        )

    avg = {
        "accuracy_mean": round(np.mean([m["accuracy"] for m in fold_metrics]), 4),
        "accuracy_std": round(np.std([m["accuracy"] for m in fold_metrics]), 4),
        "auc_mean": round(np.mean([m["auc"] for m in fold_metrics]), 4),
        "logloss_mean": round(np.mean([m["log_loss"] for m in fold_metrics]), 4),
        "brier_mean": round(np.mean([m["brier"] for m in fold_metrics]), 4),
    }
    console.print(
        f"\n[bold]CV Summary:[/bold] acc={avg['accuracy_mean']:.3f}±{avg['accuracy_std']:.3f}  "
        f"AUC={avg['auc_mean']:.3f}  logloss={avg['logloss_mean']:.3f}"
    )
    return {"folds": fold_metrics, "summary": avg}


# ---------------------------------------------------------------------------
# Train final model on all data + calibrate probabilities
# ---------------------------------------------------------------------------

def train_final(df: pd.DataFrame, feature_cols: list[str], target: str) -> xgb.XGBClassifier:
    console.print("\nTraining final model on full dataset...")
    X = df[feature_cols].values
    y = df[target].values

    # Hold out last 10% for final sanity check (chronological)
    split = int(len(X) * 0.90)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = xgb.XGBClassifier(**DEFAULT_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    console.print(
        f"\n[bold green]Holdout accuracy: {accuracy_score(y_val, val_preds):.3f}  "
        f"AUC: {roc_auc_score(y_val, val_probs):.3f}[/bold green]"
    )

    return model, X_val, y_val


# ---------------------------------------------------------------------------
# Calibrate probabilities using Platt scaling
# ---------------------------------------------------------------------------

def calibrate(model: xgb.XGBClassifier, X_cal: np.ndarray, y_cal: np.ndarray):
    """Wrap model in an isotonic calibrator for sharper probability estimates."""
    console.print("Calibrating probabilities (isotonic)...")
    calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    calibrated.fit(X_cal, y_cal)
    return calibrated


# ---------------------------------------------------------------------------
# Feature importance report
# ---------------------------------------------------------------------------

def print_importance(model: xgb.XGBClassifier, feature_cols: list[str], top_n: int = 20) -> None:
    scores = model.feature_importances_
    pairs = sorted(zip(feature_cols, scores), key=lambda x: x[1], reverse=True)[:top_n]

    table = Table(title="Top Feature Importances", box=box.SIMPLE, show_header=True)
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", justify="right")
    table.add_column("Bar")

    max_score = pairs[0][1] if pairs else 1.0
    for feat, score in pairs:
        bar_len = int((score / max_score) * 30)
        table.add_row(feat, f"{score:.4f}", "█" * bar_len)

    console.print(table)


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(model, calibrated, feature_cols: list[str], cv_results: dict, X_val, y_val) -> None:
    model.save_model(str(MODEL_PATH))
    joblib.dump(calibrated, CALIB_PATH)
    FEAT_LIST.write_text(json.dumps(feature_cols))

    # Final evaluation on holdout
    probs = calibrated.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    eval_report = {
        "holdout": {
            "accuracy": round(accuracy_score(y_val, preds), 4),
            "auc": round(roc_auc_score(y_val, probs), 4),
            "log_loss": round(log_loss(y_val, probs), 4),
            "brier": round(brier_score_loss(y_val, probs), 4),
            "n_samples": len(y_val),
        },
        "cv": cv_results,
        "params": DEFAULT_PARAMS,
        "n_features": len(feature_cols),
    }
    EVAL_PATH.write_text(json.dumps(eval_report, indent=2))

    console.print(f"\n[bold green]Saved:[/bold green]")
    console.print(f"  {MODEL_PATH}")
    console.print(f"  {CALIB_PATH}")
    console.print(f"  {FEAT_LIST}")
    console.print(f"  {EVAL_PATH}")


# ---------------------------------------------------------------------------
# Hyperparameter tuning (optional, slow)
# ---------------------------------------------------------------------------

def tune_hyperparams(df: pd.DataFrame, feature_cols: list[str], target: str) -> dict:
    """Basic grid search over key XGBoost hyperparameters."""
    from sklearn.model_selection import RandomizedSearchCV

    console.print("[yellow]Running hyperparameter search (this may take 10–20 min)...[/yellow]")

    X = df[feature_cols].values
    y = df[target].values
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    param_dist = {
        "n_estimators": [400, 600, 800],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "min_child_weight": [3, 5, 7],
        "gamma": [0, 0.1, 0.2],
    }

    base = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    tscv = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(base, param_dist, n_iter=30, cv=tscv, scoring="roc_auc",
                                 n_jobs=-1, verbose=1, random_state=42)
    search.fit(X_train, y_train)

    best = search.best_params_
    console.print(f"[green]Best params: {best}[/green]")
    console.print(f"Best CV AUC: {search.best_score_:.4f}")
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HoopIQ Model Training")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--eval-only", action="store_true", help="Load saved model and evaluate")
    args = parser.parse_args()

    df, feature_cols, target = load_features()

    if args.eval_only:
        if not MODEL_PATH.exists():
            console.print("[red]No saved model found. Run training first.[/red]")
            raise SystemExit(1)
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        calibrated = joblib.load(CALIB_PATH)
        X = df[feature_cols].values
        y = df[target].values
        probs = calibrated.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        console.print(f"Overall accuracy : {accuracy_score(y, preds):.4f}")
        console.print(f"AUC              : {roc_auc_score(y, probs):.4f}")
        raise SystemExit(0)

    if args.tune:
        best_params = tune_hyperparams(df, feature_cols, target)
        DEFAULT_PARAMS.update(best_params)

    cv_results = time_series_cv(df, feature_cols, target)
    model, X_val, y_val = train_final(df, feature_cols, target)
    print_importance(model, feature_cols)
    calibrated = calibrate(model, X_val, y_val)
    save_artifacts(model, calibrated, feature_cols, cv_results, X_val, y_val)

    console.print("\n[bold orange1]Training complete! Run python 5_api_server.py to serve predictions.[/bold orange1]")
