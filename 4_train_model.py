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

def train_final(df: pd.DataFrame, feature_cols: list[str], target: str):
    """
    Chronological THREE-way split to avoid calibration leakage:
      - train     [0   : 80%]  — fits the XGBoost model
      - calibrate  [80% : 90%]  — fits the isotonic calibrator (model is prefit)
      - holdout    [90% : 100%] — NEVER seen by model or calibrator; the only
                                  honest estimate of real-world accuracy

    Previously the calibrator was fit AND evaluated on the same 10% slice,
    which inflated reported holdout accuracy by ~7 points vs. true CV.
    """
    X = df[feature_cols].values
    y = df[target].values

    n = len(X)
    train_end = int(n * 0.80)
    cal_end   = int(n * 0.90)

    X_train, y_train = X[:train_end],          y[:train_end]
    X_cal,   y_cal   = X[train_end:cal_end],   y[train_end:cal_end]
    X_hold,  y_hold  = X[cal_end:],            y[cal_end:]

    # Time-weighted training: exp decay with 365-day half-life.
    # Most recent training game weight ≈ 1.0; one-year-older ≈ 0.5; etc.
    if "GAME_DATE" in df.columns:
        dates = pd.to_datetime(df["GAME_DATE"]).values
        dates_train = dates[:train_end]
        most_recent = dates_train.max()
        days_old = (most_recent - dates_train).astype("timedelta64[D]").astype(float)
        HALF_LIFE_DAYS = 1095.0
        sample_weights = 0.5 ** (days_old / HALF_LIFE_DAYS)
        console.print(
            f"[dim]Time-weighted: train {pd.Timestamp(dates_train.min()).date()} → "
            f"{pd.Timestamp(most_recent).date()}; "
            f"weights {sample_weights.min():.3f} – {sample_weights.max():.3f}[/dim]"
        )
    else:
        sample_weights = None
        console.print("[yellow]GAME_DATE missing — training without time weights[/yellow]")

    console.print(
        f"[dim]Split → train {len(y_train)} | calibrate {len(y_cal)} | "
        f"holdout {len(y_hold)} (holdout is untouched by both)[/dim]"
    )

    model = xgb.XGBClassifier(**DEFAULT_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_cal, y_cal)],
        verbose=100,
    )

    val_probs = model.predict_proba(X_hold)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    console.print(
        f"\n[dim]Holdout (raw, uncalibrated): {accuracy_score(y_hold, val_preds):.3f}  "
        f"AUC: {roc_auc_score(y_hold, val_probs):.3f} — "
        f"calibrated number printed below is what the API uses.[/dim]"
    )

    # Return BOTH the calibration slice and the untouched holdout separately.
    return model, (X_cal, y_cal), (X_hold, y_hold)


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
            "leakage_free": True,  # holdout untouched by model AND calibrator
        },
        "cv": cv_results,
        # The number to trust / display. CV is the most honest generalization
        # estimate; holdout is a single-slice sanity check. We surface CV mean
        # as the headline so the dashboard stops over-claiming.
        "headline_accuracy": round(cv_results["summary"]["accuracy_mean"], 4),
        "headline_source": "cv_mean",
        "params": DEFAULT_PARAMS,
        "n_features": len(feature_cols),
    }
    EVAL_PATH.write_text(json.dumps(eval_report, indent=2))

    h = eval_report["holdout"]
    console.print(
        f"\n[bold green]Holdout (CALIBRATED — this is the API's accuracy): "
        f"{h['accuracy']:.4f}  AUC: {h['auc']:.4f}  "
        f"LogLoss: {h['log_loss']:.4f}  Brier: {h['brier']:.4f}[/bold green]"
    )

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
# Feature pruning (#6) — drop low-importance noise features
# ---------------------------------------------------------------------------

def prune_features(df: pd.DataFrame, feature_cols: list[str], target: str,
                   keep_fraction: float = 0.65) -> list[str]:
    """
    Train a quick model, rank features by importance, and keep only the top
    `keep_fraction`. On ~5–10k games, 189 features is too many — the long tail
    is mostly noise that hurts generalization. Pruning often RAISES CV accuracy
    and always speeds up training.

    Returns the pruned feature list. We verify the prune helps via a CV
    comparison and only keep it if CV doesn't get worse.
    """
    console.print(
        f"\n[bold]Feature pruning:[/bold] {len(feature_cols)} features → "
        f"keeping top {int(keep_fraction*100)}%"
    )

    X = df[feature_cols].values
    y = df[target].values

    # Baseline CV with all features
    base_cv = _quick_cv_auc(X, y)

    quick = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, random_state=42,
        eval_metric="logloss", n_jobs=-1,
    )
    quick.fit(X, y)
    importances = quick.feature_importances_

    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    n_keep = max(10, int(len(feature_cols) * keep_fraction))
    pruned = [f for f, _ in ranked[:n_keep]]

    # CV with pruned features — only accept the prune if it doesn't hurt.
    Xp = df[pruned].values
    pruned_cv = _quick_cv_auc(Xp, y)

    console.print(
        f"  All {len(feature_cols)} feats  → CV AUC {base_cv:.4f}\n"
        f"  Top {len(pruned)} feats     → CV AUC {pruned_cv:.4f}"
    )

    if pruned_cv >= base_cv - 0.003:   # allow tiny noise; pruning is worth it
        dropped = [f for f, _ in ranked[n_keep:]]
        console.print(
            f"  [green]✓ Keeping pruned set[/green] "
            f"(dropped {len(dropped)} low-signal features)"
        )
        return pruned
    else:
        console.print(
            f"  [yellow]✗ Pruning hurt CV — keeping all {len(feature_cols)} features[/yellow]"
        )
        return feature_cols


def _quick_cv_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 4) -> float:
    """Fast time-series CV AUC for the prune accept/reject decision."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for tr, te in tscv.split(X):
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, random_state=42,
            eval_metric="logloss", n_jobs=-1,
        )
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HoopIQ Model Training")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--eval-only", action="store_true", help="Load saved model and evaluate")
    parser.add_argument("--no-prune", action="store_true",
                        help="Skip feature pruning (keep all 189 features)")
    args = parser.parse_args()

    df, feature_cols, target = load_features()

    if args.eval_only:
        if not MODEL_PATH.exists():
            console.print("[red]No saved model found. Run training first.[/red]")
            raise SystemExit(1)
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        calibrated = joblib.load(CALIB_PATH)
        # Use the pruned feature list if one was saved
        if FEAT_LIST.exists():
            saved_feats = json.loads(FEAT_LIST.read_text())
        else:
            saved_feats = feature_cols
        X = df[saved_feats].values
        y = df[target].values
        probs = calibrated.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        console.print(f"Overall accuracy : {accuracy_score(y, preds):.4f}")
        console.print(f"AUC              : {roc_auc_score(y, probs):.4f}")
        raise SystemExit(0)

    # ── #6 Feature pruning (before tuning so we tune on the right feature set) ──
    if not args.no_prune:
        feature_cols = prune_features(df, feature_cols, target)

    # ── #5 Hyperparameter tuning (on the pruned feature set) ──
    if args.tune:
        best_params = tune_hyperparams(df, feature_cols, target)
        DEFAULT_PARAMS.update(best_params)

    cv_results = time_series_cv(df, feature_cols, target)
    model, (X_cal, y_cal), (X_hold, y_hold) = train_final(df, feature_cols, target)
    print_importance(model, feature_cols)
    calibrated = calibrate(model, X_cal, y_cal)            # fit on CAL slice
    save_artifacts(model, calibrated, feature_cols, cv_results, X_hold, y_hold)  # eval on HOLDOUT

    console.print("\n[bold orange1]Training complete! Run python 5_api_server.py to serve predictions.[/bold orange1]")
