"""
Step 17 — Make Top 10 Picks Simple + Add Reasons
--------------------------------------------------
Patches the /props/top10 endpoint to add:
  - simple_explanation: "Bet X to score MORE than Y points tonight"
  - reasons: 3-5 bullet points why this is a good pick
  - confidence_label: "Very high confidence" instead of just a number

Run:
    python 17_simple_props.py
    # Then restart your server
"""

from pathlib import Path
import re

server = Path("5_api_server.py")
if not server.exists():
    print("Error: 5_api_server.py not found in current directory")
    exit(1)

content = server.read_text()

if '"simple_explanation"' in content:
    print("✓ Already patched")
    exit(0)

# Find the top10 endpoint - look for picks.append({
match = re.search(r'(\s+)picks\.append\(\{\s*\n(\s+)"player":', content)
if not match:
    print("Could not find picks.append() in your server file")
    print("Make sure /props/top10 endpoint exists")
    exit(1)

# Match the entire picks.append({...}) block
# Find the dict by matching balanced braces from picks.append({
start_idx = content.find('picks.append({')
if start_idx == -1:
    print("Could not locate picks.append({")
    exit(1)

# Walk forward to find the matching })
depth = 0
end_idx = start_idx
in_picks_call = False
for i in range(start_idx, len(content)):
    c = content[i]
    if c == '{':
        depth += 1
        in_picks_call = True
    elif c == '}':
        depth -= 1
        if depth == 0 and in_picks_call:
            # Find the closing ) after the }
            for j in range(i+1, min(i+20, len(content))):
                if content[j] == ')':
                    end_idx = j + 1
                    break
            break

old_block = content[start_idx:end_idx]

new_block = '''# Build human-friendly reasons
            stat_word = {"PTS":"points","REB":"rebounds","AST":"assists"}[stat]
            reasons = []
            simple = ""

            if vegas_line:
                if pick_side == "OVER":
                    simple = f"Bet {name} to score MORE than {vegas_line} {stat_word} tonight"
                else:
                    simple = f"Bet {name} to score LESS than {vegas_line} {stat_word} tonight"
                reasons.append(f"Sportsbook line is {vegas_line} {stat_word}, but model projects {proj:.1f}")
            else:
                simple = f"Model expects {name} to score over {round(avg10)} {stat_word}"

            if avg5 > avg10 * 1.05:
                reasons.append(f"Heating up: {avg5:.1f} {stat_word} last 5 games vs {avg10:.1f} season avg")
            elif avg5 < avg10 * 0.95:
                reasons.append(f"Cooling off: only {avg5:.1f} {stat_word} last 5 vs {avg10:.1f} season avg")
            else:
                reasons.append(f"Consistent: {avg5:.1f} {stat_word} L5, {avg10:.1f} L10")

            if consistency > 0.75:
                reasons.append(f"Reliable performer ({round(consistency*100)}% consistency)")
            elif consistency < 0.5:
                reasons.append(f"⚠️ Risky pick — only {round(consistency*100)}% consistency")

            if vegas_line and abs(edge_pct) >= 10:
                if pick_side == "OVER":
                    reasons.append(f"Big edge: projecting {abs(edge_pct):.0f}% over the line")
                else:
                    reasons.append(f"Big edge: projecting {abs(edge_pct):.0f}% below the line")

            if is_home:
                reasons.append("Playing at home (small boost)")
            else:
                reasons.append("Road game (slight headwind)")

            confidence_word = ("Very high confidence" if confidence>=80 else
                             "Good confidence" if confidence>=65 else
                             "Moderate confidence" if confidence>=50 else
                             "Low confidence — bet small")

            picks.append({
                "player": name, "team": team, "opponent": opp, "home": is_home,
                "stat": stat, "stat_label": stat_word,
                "projection": round(proj,1), "vegas_line": vegas_line,
                "edge": round(edge,2) if edge is not None else None,
                "edge_pct": round(edge_pct,1) if vegas_line else None,
                "pick": pick_side, "recommendation": rec, "confidence": round(confidence),
                "confidence_label": confidence_word,
                "simple_explanation": simple,
                "reasons": reasons[:5],
                "form": {"avg_last_5": round(avg5,1), "avg_last_10": round(avg10,1),
                         "trending": "up" if form_trend>0.05 else "down" if form_trend<-0.05 else "flat",
                         "consistency": round(consistency*100)},
            })'''

content = content.replace(old_block, new_block)
server.write_text(content)
print("✓ Patched /props/top10 with simple explanations and reasons")
print("\nRestart your server: python 5_api_server.py")
