import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

DATA = Path("data")
MODELS = Path("models")

def load_data():
    feats = pd.read_parquet(DATA / "features.parquet")
    dates = pd.read_parquet(DATA / "game_logs.parquet")[["GAME_ID","GAME_DATE"]].drop_duplicates()
    df = feats.merge(dates, on="GAME_ID", how="left")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)
    fl = json.loads((MODELS / "feature_list.json").read_text())
    fl = [f for f in fl if f in df.columns]
    return df, fl

def walk_forward(df, feats, start=None, retrain_every=30, min_train=1500):
    X = df[feats].values
    y = df["HOME_WIN"].values.astype(int)
    dates = df["GAME_DATE"].values
    start_idx = min_train
    if start:
        after = np.argmax(dates >= np.datetime64(pd.Timestamp(start)))
        start_idx = max(min_train, int(after))
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                  reg_alpha=0.1, reg_lambda=1.5, eval_metric="logloss",
                  random_state=42, n_jobs=-1, tree_method="hist")
    preds, probs, actuals = [], [], []
    i, model = start_idx, None
    while i < len(df):
        if model is None or (i - start_idx) % retrain_every == 0:
            model = xgb.XGBClassifier(**params)
            model.fit(X[:i], y[:i])
        end = min(i + retrain_every, len(df))
        p = model.predict_proba(X[i:end])[:, 1]
        probs.extend(p); preds.extend((p >= 0.5).astype(int))
        actuals.extend(y[i:end]); i = end
    return np.array(actuals), np.array(preds), np.array(probs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    args = ap.parse_args()
    df, feats = load_data()
    print("Loaded", len(df), "games,", len(feats), "features,",
          df["GAME_DATE"].min().date(), "to", df["GAME_DATE"].max().date())
    print("Walk-forward (train only on the past)...")
    actuals, preds, probs = walk_forward(df, feats, start=args.start)
    n = len(actuals)
    acc = accuracy_score(actuals, preds)
    auc = roc_auc_score(actuals, probs)
    brier = brier_score_loss(actuals, probs)
    base = max(actuals.mean(), 1 - actuals.mean())
    print("Out-of-sample games :", n)
    print("Walk-forward acc    :", round(acc, 4))
    print("Naive baseline      :", round(base, 4))
    print("Edge over baseline  :", round((acc - base) * 100, 2), "pts")
    print("AUC                 :", round(auc, 4))
    print("Brier (lower better):", round(brier, 4))
    print("Calibration (pred vs actual):")
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0: continue
        pr, ac = probs[m].mean(), actuals[m].mean()
        flag = "" if abs(pr - ac) < 0.05 else "  off"
        print("  ", round(lo,1), "-", round(hi,1), " n=", int(m.sum()),
              " pred=", round(pr,3), " actual=", round(ac,3), flag)
    print("flat-110 ROI (SIGNAL PROXY, NOT PROFIT):")
    for e in [0.0, 0.05, 0.08, 0.10, 0.15]:
        bm = np.abs(probs - 0.5) >= e
        if bm.sum() == 0: continue
        won = ((probs[bm] >= 0.5).astype(int) == actuals[bm])
        roi = np.where(won, 1.00, -1.10).sum() / (1.10 * bm.sum()) * 100
        print("  min_edge=", e, " bets=", int(bm.sum()),
              " win%=", round(won.mean()*100,1), " roi%=", round(roi,2))
    print("NOTE: ROI assumes -110 on every game. Real games are not.")
    print("Shows SIGNAL, not profit. Real profit needs stored closing lines.")

if __name__ == "__main__":
    main()
