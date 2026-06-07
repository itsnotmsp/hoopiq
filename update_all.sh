#!/bin/bash
# HoopIQ full data refresh + retrain pipeline
# Runs every step in sequence; stops immediately if any step fails.
#
# IMPORTANT: Stop the running server (Ctrl+C in run.sh terminal) BEFORE running this.
# After this completes, restart the server with ./run.sh

set -e

cd ~/Code/hoopiq-fresh
source venv/bin/activate

echo ""
echo "========================================="
echo "[1/5] Pulling latest team game logs..."
echo "========================================="
python 2_data_pipeline.py

echo ""
echo "========================================="
echo "[2/5] Pulling latest player game logs..."
echo "(this takes several minutes - do NOT Ctrl+C)"
echo "========================================="
python 15_real_data_v2.py

echo ""
echo "========================================="
echo "[3/5] Rebuilding game features..."
echo "========================================="
python 3_feature_engineering.py

echo ""
echo "========================================="
echo "[4/5] Retraining game model..."
echo "========================================="
python 4_train_model.py

echo ""
echo "========================================="
echo "[5/5] Retraining player prop models..."
echo "========================================="
python 7_player_model.py

echo ""
echo "========================================="
echo "All steps completed successfully."
echo "Now restart the server: ./run.sh"
echo "========================================="
