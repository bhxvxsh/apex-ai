"""
APEX AI - Local Rule-Based & Offline LLM Dispatcher

Allows running natural-language F1 queries locally without needing an expensive GPU:
- Direct intent matcher & query router
- Calls structured tools and formats clean responses
- Serves as the fallback / local driver for Colab or local inference
"""

import os
import sys
import re
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "05_Analytics"))
from tools import execute_tool, TOOL_SCHEMAS


def parse_query_intent(query: str):
    q = query.lower()

    # 1. Prediction intent
    # "predict verstappen from P3", "what if norris starts 1st", "predict race for HAM grid 5"
    m_grid = re.search(r'(grid|from|starts|p)\s*([0-9]+)', q)
    if 'predict' in q or 'what if' in q or 'finish' in q:
        grid = int(m_grid.group(2)) if m_grid else 1
        driver = 'VER'
        for d in ['ver', 'ham', 'nor', 'lec', 'sai', 'rus', 'alo', 'per', 'pia']:
            if d in q:
                driver = d.upper()
                break
        return "predict_race", {"driver_id": driver, "grid_position": grid}

    # 2. Standings intent
    # "who won 2023", "standings 2024", "who is the champion in 2023"
    m_stand = re.search(r'(standings|champion|who won|points table).*?(2023|2024|2025)', q)
    if m_stand or 'standings' in q:
        year = 2023
        if '2024' in q:
            year = 2024
        elif '2025' in q:
            year = 2025
        return "get_standings", {"season": year}

    # 3. Consistency intent
    # "most consistent driver", "who is consistent", "consistency ranking"
    if 'consistent' in q or 'consistency' in q:
        return "consistency_ranking", {"top_n": 10}

    # 4. Qualifying-race delta intent
    # "qualifying delta for verstappen", "positions gained by norris"
    m_delta = re.search(r'(delta|gain|positions gained|drop|qualifying.*race).*?([a-z]{3})', q)
    if m_delta:
        driver = m_delta.group(2).upper()
        return "qualifying_race_delta", {"driver_id": driver, "season": 2023}

    # 5. Career / multi-season stats
    # "how many wins does verstappen have", "stats for hamilton", "ver career"
    for d_name in ['ver', 'ham', 'nor', 'lec', 'sai', 'rus', 'alo', 'per', 'pia']:
        if d_name in q:
            return "driver_career_stats", {"driver_id": d_name.upper(), "seasons": [2023, 2024, 2025]}

    return None, None


def ask_apex(query: str) -> str:
    """Natural language query interface for APEX AI."""
    tool_name, args = parse_query_intent(query)
    if not tool_name:
        return (
            "I couldn't identify the exact F1 analytical query. You can ask me:\n"
            "  - 'Predict race for VER starting P1'\n"
            "  - 'Show me the 2023 championship standings'\n"
            "  - 'Who is the most consistent driver on the grid?'\n"
            "  - 'What are Hamilton's career stats across 2023-2025?'\n"
            "  - 'What is the qualifying delta for NOR in 2023?'"
        )

    res_raw = execute_tool(tool_name, args)
    data = json.loads(res_raw)

    # Format human-friendly response
    if tool_name == "predict_race":
        return (
            f"[APEX AI Race Prediction]: {data['driver_id']} starting from P{data['starting_grid']}\n"
            f"  * Expected Finishing Position: {data['predicted_finish_position']}\n"
            f"  * Podium Probability (Top 3): {data['podium_probability']}\n"
            f"  * Points Probability (Top 10): {data['points_probability']}\n"
            f"  * Model Confidence: {data['model_confidence']}"
        )
    elif tool_name == "get_standings":
        lines = [f"[F1 Championship Standings] Season {data['season']} (Champion: {data['champion']}):"]
        for i, d in enumerate(data['top_10'], 1):
            lines.append(f"  {i}. {d['FullName']} ({d['TeamName']}) - {d['Points']:.0f} pts (Avg Pos: {d['PositionAvg']:.2f})")
        return "\n".join(lines)
    elif tool_name == "consistency_ranking":
        lines = ["[F1 Driver Consistency Rankings (2023-2025)]:"]
        for i, d in enumerate(data['rankings'], 1):
            lines.append(f"  {i}. {d['fullName']} ({d['teamName']}) - Pos StdDev: {d['position_stddev']} (Avg Pos: {d['avg_position']}, Finish Rate: {d['finish_rate']}%)")
        return "\n".join(lines)
    elif tool_name == "driver_career_stats":
        lines = [f"[Career Performance for {data['driver_id']} (2023-2025)]:"]
        lines.append(f"  Total Points: {data['total_points']:.0f} | Total Wins: {data['total_wins']}")
        for s in data['seasons_summary']:
            lines.append(f"  * {s['year']}: {s['points']:.0f} pts, {s['wins']} wins, {s['podiums']} podiums ({s['races']} races)")
        return "\n".join(lines)
    elif tool_name == "qualifying_race_delta":
        return (
            f"[Qualifying-to-Race Deltas for {data['driver_id']} ({data['season']})]:\n"
            f"  * Average Positions Gained: {data['average_position_delta']:+.2f}\n"
            f"  * Best Gain: +{data['best_improvement']} positions\n"
            f"  * Worst Drop: {data['worst_drop']} positions\n"
            f"  * Improvement Rate: {data['improvement_rate_percent']}% of races"
        )
    else:
        return json.dumps(data, indent=2)


if __name__ == "__main__":
    print("=" * 60)
    print(" APEX AI CONVERSATIONAL QUERY DEMO")
    print("=" * 60)
    test_queries = [
        "Predict race finish for VER starting P1",
        "Who won the 2023 championship standings?",
        "Who is the most consistent driver on the grid?",
        "Show me stats for HAM",
    ]
    for q in test_queries:
        print(f"\nUser: \"{q}\"")
        print(ask_apex(q))
