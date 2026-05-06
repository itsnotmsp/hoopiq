#!/bin/bash
# HoopIQ repo cleanup script
# Removes scratch files, debug scripts, and outdated patchers
# Keeps only what's needed for production

set -e
cd ~/Desktop/hoopiq-fresh

echo "🧹 Cleaning HoopIQ repo..."

# Remove debug & one-off patcher scripts (no longer needed)
echo "Removing debug & patcher scripts..."
rm -f debug_espn.py
rm -f debug_espn2.py
rm -f fix_modelstate.py
rm -f restore_server.py
rm -f 8_prop_api_update.py
rm -f 11_top10_picks.py
rm -f 12_balldontlie_pipeline.py
rm -f 13_espn_player_stats.py
rm -f 14_real_data_v2.py
rm -f 16_history_endpoints.py
rm -f 17_simple_props.py

# Remove backup files
echo "Removing backups..."
rm -f data/player_logs_synthetic_backup.parquet
rm -f data/players_real.json
rm -f data/debug_*.json
rm -rf data/odds_cache

# Remove __pycache__
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
rm -rf logs/

# Make sure config.json is gitignored (NEVER push API keys)
if ! grep -q "^config.json$" .gitignore 2>/dev/null; then
  echo "config.json" >> .gitignore
fi

# Update .gitignore with everything we want excluded
cat > .gitignore << 'GITIGNORE'
# Python
venv/
__pycache__/
*.pyc
*.pyo
.Python

# Environment & secrets
.env
config.json

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp

# Logs
logs/
*.log

# Cache
data/odds_cache/
data/debug_*.json
data/*_backup.parquet
GITIGNORE

# Update README with clean instructions
cat > README.md << 'README'
# HoopIQ — NBA Predictions

AI-powered NBA game and player prop predictions using XGBoost.

**Live API:** https://hoopiq-production.up.railway.app
**Dashboard:** Open `hoopiq_dashboard.html` in browser

## Features

- 🏀 Game winner predictions (XGBoost, ~69% accuracy)
- 🎯 Player prop predictions (PTS, REB, AST, FPTS)
- ⭐ Top 10 best prop picks tonight (ranked by edge)
- 💡 Plain-English explanations for every pick
- 📊 Real-time odds from DraftKings/FanDuel/BetMGM
- 📝 History tracking — log picks, mark wins/losses
- 🔴 Live in-game odds and predictions

## Quick Start

```bash
# 1. Install dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Add API keys to config.json
cp config.example.json config.json
# Edit config.json with your keys

# 3. Build data pipeline (one time)
python 2_data_pipeline.py        # NBA team game logs
python 15_real_data_final.py     # Real player stats from ESPN
python 3_feature_engineering.py  # Build ML features
python 4_train_model.py          # Train game winner model
python 7_player_model.py         # Train prop models

# 4. Start the server
python 5_api_server.py
# → http://localhost:8000
```

## Files

| File | Purpose |
|------|---------|
| `1_live_scores.py` | Poll ESPN for live game scores |
| `2_data_pipeline.py` | Pull historical NBA game logs |
| `3_feature_engineering.py` | Build ML features from game logs |
| `4_train_model.py` | Train game winner XGBoost model |
| `5_api_server.py` | FastAPI prediction server (main) |
| `6_player_pipeline.py` | Player roster mapper |
| `7_player_model.py` | Train player prop models |
| `9_odds_pipeline.py` | The Odds API integration |
| `10_odds_integration.py` | Live odds module for API |
| `15_real_data_final.py` | Real player stats from ESPN |
| `hoopiq_dashboard.html` | Frontend dashboard |
| `Dockerfile` | Railway deployment config |
| `requirements.txt` | Python dependencies |
| `config.example.json` | Template for API keys |

## Configuration

Create `config.json` (gitignored):
```json
{
  "odds_api_key": "your_key_from_the-odds-api.com",
  "balldontlie_api_key": "optional"
}
```

## Deployment

Auto-deploys to Railway on `git push`. Set environment variables in Railway:
- `ODDS_API_KEY` (from the-odds-api.com)
- `PORT` = 8000

## API Endpoints

```
GET  /health                  Server status
GET  /games/today             Today's NBA games
POST /predict/game            Predict single game
POST /predict/game/explain    Predict with reasoning
POST /predict/batch           Predict all today's games
GET  /predict/live            Live in-game predictions
POST /props/player            Predict player props
GET  /props/top10             Top 10 best picks tonight
GET  /props/fantasy           Fantasy rankings
GET  /props/starts            Start/sit recommendations
GET  /odds/games              Real game odds
GET  /odds/live               Live odds
POST /history/log             Log a prediction
GET  /history                 Get tracked predictions
POST /history/{id}/result     Mark WIN/LOSS
DELETE /history/{id}          Delete record
```

## Tech Stack

- **Backend:** FastAPI + XGBoost + pandas
- **Data:** ESPN API + The Odds API
- **Deploy:** Railway (Docker)
- **Frontend:** Vanilla HTML/JS
README

# Create config.example.json template
cat > config.example.json << 'CONFIG'
{
  "odds_api_key": "GET_FREE_KEY_AT_the-odds-api.com",
  "balldontlie_api_key": "OPTIONAL_GET_AT_balldontlie.io"
}
CONFIG

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Files in repo:"
ls -1 *.py *.html *.txt *.md *.json Dockerfile 2>/dev/null | sort
echo ""
echo "Next: review changes and push:"
echo "  git status"
echo "  git add -A"
echo "  git commit -m 'Clean up repo for v1 release'"
echo "  git push"
