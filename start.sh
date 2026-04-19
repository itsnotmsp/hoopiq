#!/bin/sh
exec uvicorn 5_api_server:app --host 0.0.0.0 --port "${PORT:-8000}"
