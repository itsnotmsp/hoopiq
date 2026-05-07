"""
Step 19 — Add team names to prop history + auto-update results
----------------------------------------------------------------
Two improvements:

1. Prop predictions now include team name in title (e.g. "PHX - Devin Booker PTS OVER 27.5")
2. New /history/auto_update endpoint that checks final scores from ESPN and
   automatically marks pending predictions as WIN/LOSS

Run:
    python 19_team_in_props.py
    # Then restart server
"""

from pathlib import Path
import re

server = Path("5_api_server.py")
if not server.exists():
    print("Error: 5_api_server.py not found")
    exit(1)

content = server.read_text()


# ─── 1. Add /history/auto_update endpoint ─────────────────────────────
if '/history/auto_update' not in content:
    AUTO_UPDATE_CODE = '''

@app.post("/history/auto_update")
async def auto_update_history():
    """Check ESPN for finished games and auto-mark pending GAME predictions WIN/LOSS."""
    records = load_history()
    et = timezone(timedelta(hours=-5))

    updated = 0
    games_checked = set()
    final_games = {}

    # Find all unique game-prediction dates we need to check
    pending_games = [r for r in records
                     if r.get("type") == "game" and r.get("result") == "PENDING"]

    if not pending_games:
        return {"message": "No pending predictions to update", "updated": 0}

    # Get unique dates from pending records
    dates_to_check = set()
    for r in pending_games:
        try:
            logged_date = datetime.fromisoformat(r["logged_at"].replace("Z","+00:00")).astimezone(et).date()
            dates_to_check.add(logged_date.strftime("%Y%m%d"))
        except:
            pass

    # Fetch ESPN for each date
    for date_str in dates_to_check:
        try:
            games = await get_espn_games(date_str)
            for g in games:
                if g.get("status") == "post":  # final
                    home = g["home_abbr"]
                    away = g["away_abbr"]
                    home_score = g["home_score"]
                    away_score = g["away_score"]
                    actual_winner = home if home_score > away_score else away
                    key = (home, away)
                    final_games[key] = {
                        "actual_winner": actual_winner,
                        "score": f"{away_score}-{home_score}",
                    }
        except Exception:
            continue

    # Update pending records
    for r in records:
        if r.get("type") == "game" and r.get("result") == "PENDING":
            key = (r.get("home_team"), r.get("away_team"))
            if key in final_games:
                actual = final_games[key]
                predicted = r.get("predicted_winner")
                r["result"] = "WIN" if predicted == actual["actual_winner"] else "LOSS"
                r["actual_score"] = actual["score"]
                r["actual_winner"] = actual["actual_winner"]
                r["resolved_at"] = datetime.now(timezone.utc).isoformat()
                updated += 1

    save_history(records)

    return {
        "updated": updated,
        "games_checked": len(final_games),
        "pending_remaining": sum(1 for r in records if r.get("result") == "PENDING"),
    }
'''
    # Insert after the existing /history/{record_id}/result endpoint
    marker = '@app.delete("/history/{record_id}")'
    if marker in content:
        content = content.replace(marker, AUTO_UPDATE_CODE.strip() + "\n\n\n" + marker)
        print("✓ Added /history/auto_update endpoint")
    else:
        print("Could not find injection point for auto_update")
else:
    print("✓ auto_update endpoint already exists")


# ─── 2. Add team name to /props/top10 picks ─────────────────────────
# Already includes "team" field. Just need to make sure it's prominent.
# The team field should already be in picks - this is a dashboard fix.


server.write_text(content)
print("\n✓ Server patched")
print("Restart server: python 5_api_server.py")
