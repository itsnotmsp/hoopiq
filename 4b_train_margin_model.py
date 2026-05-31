"""
Step 4b - Margin Regression Model (XGBoost)
Predicts home team's expected margin (HOME_PTS - AWAY_PTS).
Companion to 4_train_model.py which predicts HOME_WIN probability.

Output: models/xgb_margin_model.json
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# Load features + game logs and merge on GAME_ID
features = pd.read_parquet('data/features.parquet')
game_logs = pd.read_parquet('data/game_logs.parquet')

# Home team margin is PLUS_MINUS for the IS_HOME=True row of each game
home_margins = (game_logs[game_logs['IS_HOME'] == True]
                [['GAME_ID', 'GAME_DATE', 'PLUS_MINUS']]
                .rename(columns={'PLUS_MINUS': 'HOME_MARGIN'}))
df = features.merge(home_margins, on='GAME_ID', how='inner').sort_values('GAME_DATE').reset_index(drop=True)

print(f"Loaded {len(df)} games. Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
print(f"Margin stats: mean={df['HOME_MARGIN'].mean():.2f}, std={df['HOME_MARGIN'].std():.2f}")

# Chronological 80/20 split (no leakage)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# Drop targets and IDs from feature set
EXCLUDE = ['GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_WIN', 'HOME_MARGIN', 'WL_HOME']
feature_cols = [c for c in df.columns if c not in EXCLUDE]
# Defensive: drop any non-numeric feature columns XGBoost cannot handle
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

X_train = train_df[feature_cols]
y_train = train_df['HOME_MARGIN']
X_test = test_df[feature_cols]
y_test = test_df['HOME_MARGIN']

print(f"Training on {len(X_train)} games, testing on {len(X_test)} games.")
print(f"Feature count: {len(feature_cols)}")

# Train (no early stopping = no leakage)
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Baseline: predict the average home margin from training (HCA-only model)
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

print("\n" + "=" * 55)
print("Margin Model - Walk-Forward Holdout Results")
print("=" * 55)
print(f"MAE:  {mae:.2f} pts (baseline mean-only: {baseline_mae:.2f})")
print(f"RMSE: {rmse:.2f} pts (baseline mean-only: {baseline_rmse:.2f})")
print(f"R^2:  {r2:.3f}")
print(f"MAE reduction vs baseline: {baseline_mae - mae:.2f} pts")
print("=" * 55)

print("""
INTERPRETATION GUIDE:
- Vegas spreads typically have RMSE ~11.5 against actual NBA margins.
- RMSE around 11 or below = competitive with Vegas (has signal).
- RMSE 12+ = model isn't capturing margin signal well.
- MAE reduction of 1+ point vs baseline = real edge in feature engineering.
""")

# Feature importance for inspection
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False).head(15)
print("Top 15 features by importance:")
print(importance.to_string(index=False))

# Save model and feature order
Path('models').mkdir(exist_ok=True)
model.save_model('models/xgb_margin_model.json')
with open('models/xgb_margin_features.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

print(f"\nSaved model to models/xgb_margin_model.json")
print(f"Saved feature column order to models/xgb_margin_features.json")
print("Next: review the MAE/RMSE above. If reasonable, we'll wire it into 5_api_server.py.")
