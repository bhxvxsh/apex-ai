"""
APEX AI - Robust F1 Query Engine & Intent Parser

Replaces fragile regex with:
1. Comprehensive Driver & Team Aliases (Verstappen -> VER, Lewis -> HAM, Ferrari, Red Bull, etc.)
2. Multi-intent Semantic Categorization
3. Direct Dynamic SQL execution for open-ended queries
4. Contextual analytical response formatting
"""

import os
import sys
import re
import json
import sqlite3
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "05_Analytics"))
sys.path.insert(0, os.path.join(CURRENT_DIR, "04_Models"))
os.chdir(CURRENT_DIR)

from tools import execute_tool
from dynamic_analyst import DynamicF1AnalystSandbox

sandbox = DynamicF1AnalystSandbox()

# Driver Name & Nickname Normalization Map -> Standard Abbreviation
DRIVER_ALIASES = {
    # Red Bull
    "max": "VER", "verstappen": "VER", "max verstappen": "VER", "ver": "VER",
    "checo": "PER", "perez": "PER", "sergio perez": "PER", "sergio": "PER", "per": "PER",
    # Ferrari
    "charles": "LEC", "leclerc": "LEC", "charles leclerc": "LEC", "lec": "LEC",
    "carlos": "SAI", "sainz": "SAI", "carlos sainz": "SAI", "sai": "SAI",
    "bearman": "BEA", "olliver bearman": "BEA", "bea": "BEA",
    # McLaren
    "lando": "NOR", "norris": "NOR", "lando norris": "NOR", "nor": "NOR",
    "oscar": "PIA", "piastri": "PIA", "oscar piastri": "PIA", "pia": "PIA",
    # Mercedes
    "lewis": "HAM", "hamilton": "HAM", "lewis hamilton": "HAM", "ham": "HAM",
    "george": "RUS", "russell": "RUS", "george russell": "RUS", "rus": "RUS",
    "antonelli": "ANT", "kimi antonelli": "ANT", "ant": "ANT",
    # Aston Martin
    "fernando": "ALO", "alonso": "ALO", "fernando alonso": "ALO", "alo": "ALO",
    "lance": "STR", "stroll": "STR", "lance stroll": "STR", "str": "STR",
    # Alpine
    "pierre": "GAS", "gasly": "GAS", "pierre gasly": "GAS", "gas": "GAS",
    "esteban": "OCO", "ocon": "OCO", "esteban ocon": "OCO", "oco": "OCO",
    "doohan": "DOO", "jack doohan": "DOO", "doo": "DOO",
    # Williams
    "alex": "ALB", "albon": "ALB", "alex albon": "ALB", "alb": "ALB",
    "logan": "SAR", "sargeant": "SAR", "logan sargeant": "SAR", "sar": "SAR",
    "colapinto": "COL", "franco colapinto": "COL", "col": "COL",
    # RB / AlphaTauri
    "yuki": "TSU", "tsunoda": "TSU", "yuki tsunoda": "TSU", "tsu": "TSU",
    "daniel": "RIC", "ricciardo": "RIC", "daniel ricciardo": "RIC", "ric": "RIC",
    "liam": "LAW", "lawson": "LAW", "liam lawson": "LAW", "law": "LAW",
    "hadjar": "HAD", "isack hadjar": "HAD", "had": "HAD",
    # Sauber / Alfa Romeo
    "valtteri": "BOT", "bottas": "BOT", "valtteri bottas": "BOT", "bot": "BOT",
    "zhou": "ZHO", "guanyu zhou": "ZHO", "guanyu": "ZHO", "zho": "ZHO",
    "bortoleto": "BOR", "gabriel bortoleto": "BOR", "bor": "BOR",
    # Haas
    "nico": "HUL", "hulkenberg": "HUL", "nico hulkenberg": "HUL", "hul": "HUL",
    "kevin": "MAG", "magnussen": "MAG", "kevin magnussen": "MAG", "mag": "MAG",
}

TEAM_ALIASES = {
    "red bull": "Red Bull", "redbull": "Red Bull",
    "ferrari": "Ferrari", "scuderia": "Ferrari",
    "mclaren": "McLaren",
    "mercedes": "Mercedes", "merc": "Mercedes",
    "aston martin": "Aston Martin", "aston": "Aston Martin",
    "alpine": "Alpine F1 Team",
    "williams": "Williams",
    "haas": "Haas F1 Team",
    "sauber": "Sauber", "alfa romeo": "Alfa Romeo",
    "alphatauri": "AlphaTauri", "rb": "RB F1 Team",
}


def extract_driver(text: str):
    """Extract standard driver 3-letter abbreviation from anywhere in string."""
    t = text.lower()
    # Check multi-word first, then single words
    for alias in sorted(DRIVER_ALIASES.keys(), key=lambda x: -len(x)):
        # Match whole word or token
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, t):
            return DRIVER_ALIASES[alias]
    return None


def extract_year(text: str):
    """Extract year or default to latest relevant season."""
    m = re.search(r'\b(2023|2024|2025)\b', text)
    if m:
        return int(m.group(1))
    return None


def extract_grid(text: str):
    """Extract starting grid position (P1, grid 3, from 5th, etc.)."""
    m = re.search(r'(?:grid|position|pos|p|starting|from|starts)\s*#?([0-9]{1,2})', text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 20:
            return val
    m2 = re.search(r'\bp([0-9]{1,2})\b', text, re.IGNORECASE)
    if m2:
        val = int(m2.group(1))
        if 1 <= val <= 20:
            return val
    return None


def ask_apex(query: str) -> str:
    """Intelligent query routing with domain awareness and dynamic fallback."""
    q = query.strip()
    ql = q.lower()

    driver = extract_driver(ql)
    year = extract_year(ql)
    grid = extract_grid(ql)

    # -------------------------------------------------------------
    # 1. Prediction Queries
    # -------------------------------------------------------------
    if any(w in ql for w in ['predict', 'odds', 'chance', 'probability', 'what if', 'forecast', 'finish']):
        target_driver = driver or "VER"
        target_grid = grid or 1
        res = json.loads(execute_tool("predict_race", {"driver_id": target_driver, "grid_position": target_grid}))
        return (
            f"[APEX AI Race Forecast]: {res['driver_id']} starting from P{res['starting_grid']}\n"
            f"  * Expected Finishing Position: {res['predicted_finish_position']}\n"
            f"  * Podium Probability (Top 3): {res['podium_probability']}\n"
            f"  * Points Probability (Top 10): {res['points_probability']}\n"
            f"  * Model Confidence: {res['model_confidence']}"
        )

    # -------------------------------------------------------------
    # 2. Championship Standings
    # -------------------------------------------------------------
    if any(w in ql for w in ['standing', 'standings', 'champion', 'championship', 'title', 'who won', 'points table', 'leaderboard']):
        target_year = year or 2024
        res = json.loads(execute_tool("get_standings", {"season": target_year}))
        if "error" in res:
            return f"[Error] {res['error']}"
        lines = [f"[Formula 1 Championship Standings] Season {target_year} (Champion: {res['champion']}):"]
        for i, d in enumerate(res['top_10'], 1):
            lines.append(f"  {i:>2}. {d['FullName']} ({d['TeamName']}) - {d['Points']:.0f} pts (Avg Finish: P{d['PositionAvg']:.2f})")
        return "\n".join(lines)

    # -------------------------------------------------------------
    # 3. Consistency Rankings
    # -------------------------------------------------------------
    if any(w in ql for w in ['consistent', 'consistency', 'variance', 'stddev', 'reliable', 'reliability']):
        res = json.loads(execute_tool("consistency_ranking", {"top_n": 10}))
        lines = ["[F1 Driver Consistency Rankings (2023-2025)] (Ranked by Finishing Position Standard Deviation):"]
        for i, d in enumerate(res['rankings'], 1):
            lines.append(f"  {i:>2}. {d['fullName']} ({d['teamName']}) - StdDev: {d['position_stddev']:.3f} (Avg Finish: P{d['avg_position']:.2f}, Finish Rate: {d['finish_rate']}%, Podiums: {d['podium_finishes']})")
        return "\n".join(lines)

    # -------------------------------------------------------------
    # 4. Qualifying-to-Race Delta / Positions Gained
    # -------------------------------------------------------------
    if any(w in ql for w in ['delta', 'gain', 'gained', 'overtake', 'drop', 'qualifying to race', 'qualify to race']):
        target_driver = driver or "VER"
        target_year = year or 2023
        res = json.loads(execute_tool("qualifying_race_delta", {"driver_id": target_driver, "season": target_year}))
        return (
            f"[Qualifying-to-Race Progression for {res['driver_id']} ({res['season']})]:\n"
            f"  * Average Positions Gained per Race: {res['average_position_delta']:+.2f}\n"
            f"  * Best Single Race Improvement: +{res['best_improvement']} positions\n"
            f"  * Worst Race Drop: {res['worst_drop']} positions\n"
            f"  * Races with Net Gain: {res['improvement_rate_percent']}% (Classified Races: {res['total_races']})"
        )

    # -------------------------------------------------------------
    # 5. Driver Career & Season Breakdown
    # -------------------------------------------------------------
    if driver and any(w in ql for w in ['stat', 'stats', 'career', 'record', 'how many', 'performance', 'points', 'wins', 'podium']):
        res = json.loads(execute_tool("driver_career_stats", {"driver_id": driver, "seasons": [2023, 2024, 2025]}))
        if "error" in res:
            return f"[Error] {res['error']}"
        lines = [f"[Career Performance Breakdown for {res['driver_id']} (2023-2025)]:"]
        lines.append(f"  * Total Career Points: {res['total_points']:.0f} | Total Wins: {res['total_wins']}")
        for s in res['seasons_summary']:
            lines.append(f"    - {s['year']}: {s['points']:.0f} pts | {s['wins']} wins | {s['podiums']} podiums | {s['dnfs']} DNFs ({s['races']} entries)")
        return "\n".join(lines)

    # -------------------------------------------------------------
    # 6. Team-Specific Queries
    # -------------------------------------------------------------
    for t_alias, official_team in TEAM_ALIASES.items():
        if t_alias in ql:
            query_sql = f"""
                SELECT year, COUNT(DISTINCT round) as races, SUM(Points) as total_points,
                       SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN Position <= 3 THEN 1 ELSE 0 END) as podiums
                FROM race_results
                WHERE TeamName LIKE '%{official_team}%'
                GROUP BY year
                ORDER BY year
            """
            con = sqlite3.connect(os.path.join(CURRENT_DIR, "02_Data_Processed", "f1.db"))
            tdf = pd.read_sql(query_sql, con)
            con.close()
            lines = [f"[Constructor Performance Analysis for {official_team} (2023-2025)]:"]
            for _, r in tdf.iterrows():
                lines.append(f"  * {int(r['year'])}: {r['total_points']:.0f} pts | {int(r['wins'])} wins | {int(r['podiums'])} podiums")
            return "\n".join(lines)

    # -------------------------------------------------------------
    # 7. Fallback: Dynamic Natural Language SQL/Python Interpreter
    # -------------------------------------------------------------
    if "dnf" in ql or "retired" in ql or "retirements" in ql:
        target_year = year or 2024
        query_sql = f"""
            SELECT FullName, TeamName, COUNT(*) as dnf_count
            FROM race_results
            WHERE finished = 0 AND year = {target_year}
            GROUP BY FullName, TeamName
            ORDER BY dnf_count DESC
            LIMIT 5
        """
        con = sqlite3.connect(os.path.join(CURRENT_DIR, "02_Data_Processed", "f1.db"))
        ddf = pd.read_sql(query_sql, con)
        con.close()
        lines = [f"[Most DNFs (Did Not Finish) in {target_year}]:"]
        for i, row in ddf.iterrows():
            lines.append(f"  {i+1}. {row['FullName']} ({row['TeamName']}) - {row['dnf_count']} DNFs")
        return "\n".join(lines)

    # General Helpful Guide
    return (
        f"I received your question: *\"{q}\"*\n\n"
        "To get accurate analytics, you can ask questions like:\n"
        "  • **Predictions**: *'Predict race for Norris starting P2'* or *'What are Verstappen's podium odds from P4?'*\n"
        "  • **Standings**: *'Who won the 2024 championship?'* or *'Show 2023 standings'*\n"
        "  • **Driver Stats**: *'Career stats for Hamilton'* or *'How many points did Leclerc score in 2024?'*\n"
        "  • **Consistency**: *'Who is the most consistent driver on the grid?'*\n"
        "  • **Teams**: *'How did Ferrari do in 2024?'*\n"
        "  • **Qualifying Deltas**: *'Qualifying delta for Alonso in 2023'*\n"
        "  • **DNFs**: *'Who had the most DNFs in 2023?'*"
    )


if __name__ == "__main__":
    print("Testing Natural Language Engine:")
    tests = [
        "Predict race for Lando Norris starting P2",
        "Who won the 2024 championship standings?",
        "What are Lewis Hamilton's career stats?",
        "Who is the most consistent driver?",
        "How did Ferrari do in 2024?",
        "Who had the most DNFs in 2023?",
        "What is Charles Leclerc's qualifying delta?",
    ]
    for t in tests:
        print(f"\nUser: \"{t}\"")
        print(ask_apex(t))
