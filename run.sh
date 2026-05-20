#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -d "venv" ]; then source venv/bin/activate; fi
if [ ! -f "models/xgb_model.json" ]; then
  echo "ERROR: models/xgb_model.json missing."
  exit 1
fi
(sleep 2; open "file://$(pwd)/hoopiq_dashboard.html" 2>/dev/null || true) &
echo ""
echo "================================================================"
echo " HoopIQ local server"
echo " API:        http://localhost:8000"
echo " Dashboard:  opening in your browser..."
echo ""
echo " FIRST TIME ONLY: in the dashboard, click Settings (gear)"
echo " and set the API URL to:  http://localhost:8000"
echo ""
echo " Stop the server with Ctrl+C."
echo "================================================================"
echo ""
exec uvicorn 5_api_server:app --host 127.0.0.1 --port 8000 --reload
