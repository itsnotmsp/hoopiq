"""
Run once to backfill the betting fields on your existing predictions_log.json.

Existing entries don't have recorded odds, so they're marked:
    stake = 0
    odds_decimal = null
    profit_loss = 0
This keeps them visible in history but excludes them from ROI calculations.

Usage:
    python migrate_history_for_betting.py
"""

import json
from pathlib import Path

LOG = Path("data/predictions_log.json")

records = json.loads(LOG.read_text())
changed = 0
for r in records:
    # Only stamp entries that don't already have these fields.
    if "stake" not in r:
        r["stake"] = 0
        changed += 1
    if "odds_decimal" not in r:
        r["odds_decimal"] = None
    if "profit_loss" not in r:
        r["profit_loss"] = 0
    if "side" not in r:
        # Best-effort backfill: game predictions get the model's pick as side,
        # prop predictions get whatever "pick" was (OVER/UNDER).
        r["side"] = r.get("predicted_winner") or r.get("pick")

LOG.write_text(json.dumps(records, indent=2))
print(f"Migrated {changed} of {len(records)} records.")
print(f"All have stake/odds_decimal/profit_loss/side fields now.")
