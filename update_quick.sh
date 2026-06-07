#!/bin/bash
# HoopIQ quick data refresh (no retrain). Use daily.
# IMPORTANT: Stop the server (Ctrl+C in run.sh terminal) BEFORE running.
# Restart with ./run.sh after.

set -e

cd ~/Code/hoopiq-fresh
source venv/bin/activate

echo ""
echo "[1/2] Pulling latest team game logs..."
python 2_data_pipeline.py

echo ""
echo "[2/2] Pulling latest player game logs..."
echo "(takes several minutes - do NOT Ctrl+C)"
python 15_real_data_v2.py

echo ""
echo "Quick refresh complete. Now restart the server: ./run.sh"
