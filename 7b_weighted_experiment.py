"""
Step 7b - Time-Weighted Player Prop Experiment

Trains two XGBoost regressors per stat on identical features:
  - baseline: every game weighted equally
  - weighted: exp-decay (half-weight every 90 days) + 3x bonus for 2026 playoffs

Reports MAE/R^2 delta. Negative MAE delta = weighting helps.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_parquet('data/player_logs.parquet')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
print(f"Loaded {len(df)} player-game rows. Range: {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")

# Build rolling features per player
def add_features(g):
    g = g.copy()
    for col in ['PTS', 'REB', 'AST', 'MIN']:
        g[f'{col}_avg_5'] = g[col].shift(1).rolling(5, min_periods=1).mean()
        g[f'{col}_avg_10'] = g[col].shift(1).rolling(10, min_periods=1).mean()
        g[f'{col}_avg_season'] = g[col].shift(1).expanding(min_periods=1).mean()
        g[f'{col}_std_10'] = g[col].shift(1).rolling(10, min_periods=2).std()
    return g

print("Building features...")
df = df.groupby('PLAYER_ID', group_keys=False).apply(add_features)

feature_cols = [c for c in df.columns if c.endswith(('_avg_5', '_avg_10', '_avg_season', '_std_10'))]
df['IS_HOME_NUM'] = df['HOME'].astype(int)
feature_cols.append('IS_HOME_NUM')

df_clean = df.dropna(subset=feature_cols + ['PTS', 'REB', 'AST']).copy()
df_clean = df_clean.sort_values('GAME_DATE').reset_index(drop=True)
print(f"After dropping NaN: {len(df_clean)} rows")

split = int(len(df_clean) * 0.8)
train = df_clean.iloc[:split]
test = df_clean.iloc[split:]
print(f"Train: {len(train)} ({train['GAME_DATE'].min().date()} to {train['GAME_DATE'].max().date()})")
print(f"Test:  {len(test)} ({test['GAME_DATE'].min().date()} to {test['GAME_DATE'].max().date()})")

# Compute sample weights for the "weighted" run
max_date = train['GAME_DATE'].max()
days_ago = (max_date - train['GAME_DATE']).dt.days.values
weights = np.exp(-days_ago / 90.0)
playoff_2026 = (train['GAME_DATE'] >= '2026-04-18').values
weights = weights * np.where(playoff_2026, 3.0, 1.0)
print(f"2026 playoff games in train: {playoff_2026.sum()}")
print(f"Weight range: min={weights.min():.3f}, max={weights.max():.3f}")

results = []
for stat in ['PTS', 'REB', 'AST']:
    X_train = train[feature_cols].values
    y_train = train[stat].values
    X_test = test[feature_cols].values
    y_test = test[stat].values

    base = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    base.fit(X_train, y_train)
    p_b = base.predict(X_test)

    wt = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    wt.fit(X_train, y_train, sample_weight=weights)
    p_w = wt.predict(X_test)

    results.append({
        'stat': stat,
        'mae_base': round(mean_absolute_error(y_test, p_b), 3),
        'mae_wtd':  round(mean_absolute_error(y_test, p_w), 3),
        'mae_delta': round(mean_absolute_error(y_test, p_w) - mean_absolute_error(y_test, p_b), 3),
        'r2_base':  round(r2_score(y_test, p_b), 3),
        'r2_wtd':   round(r2_score(y_test, p_w), 3),
        'r2_delta': round(r2_score(y_test, p_w) - r2_score(y_test, p_b), 3),
    })

print("\n" + "=" * 70)
print("Time-Weighted vs Baseline (same features, same model, same data)")
print("=" * 70)
print(pd.DataFrame(results).to_string(index=False))
print("=" * 70)
print("mae_delta NEGATIVE  = weighting helps")
print("mae_delta POSITIVE  = weighting hurts")
print("r2_delta  POSITIVE  = weighted model explains more variance")
