"""
Quick fix: rename 'matchup' dict in analyze endpoint to 'matchup_analysis'
to avoid collision with the matchup string field.
"""
from pathlib import Path

server = Path("5_api_server.py")
content = server.read_text()

# Rename the SECOND "matchup": matchup, line in the analyze endpoint
old = '''        "fatigue": fatigue,
        "matchup": matchup,
        "pace": pace_analysis,'''

new = '''        "fatigue": fatigue,
        "matchup_analysis": matchup,
        "pace": pace_analysis,'''

if old in content:
    content = content.replace(old, new)
    
    # Also add home_team/away_team for cleaner JS access
    old_top = '''    return {
        "matchup": f"{away} @ {home}",
        "date": game_date,'''
    new_top = '''    return {
        "matchup": f"{away} @ {home}",
        "home_team": home,
        "away_team": away,
        "date": game_date,'''
    if old_top in content:
        content = content.replace(old_top, new_top)
    
    server.write_text(content)
    print("✓ Fixed: 'matchup' (dict) renamed to 'matchup_analysis'")
    print("✓ Added: 'home_team' and 'away_team' fields for cleaner access")
    print("\nRestart server: python 5_api_server.py")
else:
    print("Pattern not found - already fixed?")
