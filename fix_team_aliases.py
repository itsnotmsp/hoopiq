"""
Quick fix: Add team abbreviation aliases to /predict/game/analyze
Some teams use NYK vs NY, GSW vs GS, etc. — this makes lookups tolerant.
"""
from pathlib import Path

server = Path("5_api_server.py")
content = server.read_text()

old = '''    def team_logs(abbr, n=10):
        t = df[df["TEAM_ABBREVIATION"] == abbr]
        return t[t["GAME_DATE"] < cutoff].sort_values("GAME_DATE").tail(n)'''

new = '''    # Team abbreviation aliases (handles NYK↔NY, GSW↔GS, etc.)
    ALIAS = {
        "NYK": ["NYK","NY"], "NY": ["NY","NYK"],
        "BKN": ["BKN","BRK"], "BRK": ["BRK","BKN"],
        "GSW": ["GSW","GS"], "GS": ["GS","GSW"],
        "NOP": ["NOP","NO"], "NO": ["NO","NOP"],
        "PHX": ["PHX","PHO"], "PHO": ["PHO","PHX"],
        "SAS": ["SAS","SA"], "SA": ["SA","SAS"],
        "WAS": ["WAS","WSH"], "WSH": ["WSH","WAS"],
        "CHA": ["CHA","CHO"], "CHO": ["CHO","CHA"],
    }

    def team_logs(abbr, n=10):
        candidates = ALIAS.get(abbr, [abbr])
        t = df[df["TEAM_ABBREVIATION"].isin(candidates)]
        return t[t["GAME_DATE"] < cutoff].sort_values("GAME_DATE").tail(n)'''

if old in content:
    content = content.replace(old, new)
    server.write_text(content)
    print("✓ Patched team abbreviation aliases")
    print("Restart server: python 5_api_server.py")
else:
    print("Pattern not found - aliases may already be added")
