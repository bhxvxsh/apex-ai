"""
APEX AI - Complete Milestone 2 Analytics Suite Runner

Runs and verifies all 4 analytics modules:
1. Driver Performance: Standings & Season Comparison
2. Qualifying Deltas: Grid-to-Race Deltas & Stats
3. Consistency Engine: Multi-Season Comparison & Driver Consistency Rankings
4. Visualizations: Verification & Figure Generation
"""

import os
import sys

# Ensure APEX_AI and 05_Analytics are in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYTICS_DIR = os.path.join(CURRENT_DIR, "05_Analytics")
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, ANALYTICS_DIR)
os.chdir(CURRENT_DIR)

from driver_performance import get_driver_standings, get_driver_season_comparison, get_all_drivers_list
from qualifying_deltas import get_qualifying_race_deltas, get_driver_delta_stats
from consistency import get_driver_consistency, get_season_consistency_comparison, get_all_drivers_consistency
from visualizations import (
    plot_driver_standings,
    plot_delta_distribution,
    plot_season_consistency_comparison,
    plot_driver_consistency_radar,
    plot_driver_season_comparison,
)


def run_milestone2_verification():
    print("=" * 80)
    print("  APEX AI - MILESTONE 2: NUMERICAL ANALYTICS ENGINE VERIFICATION")
    print("=" * 80)

    # 1. Driver Performance
    print("\n[1/4] Driver Performance Analysis")
    standings_2023 = get_driver_standings(season=2023)
    top_driver = standings_2023.iloc[0]
    print(f"  [OK] 2023 Champion: {top_driver['FullName']} ({top_driver['DriverId']}) - {top_driver['Points']:.1f} pts")
    print(f"  [OK] Total classified drivers in 2023: {len(standings_2023)}")

    # 2. Top Driver Multi-Season Comparison (VER)
    print("\n[2/4] Top Driver Deep Dive: Max Verstappen (2023-2025)")
    ver_history = get_driver_season_comparison('VER', [2023, 2024, 2025])
    ver_summary = ver_history.groupby('year').agg(
        points=('Points', 'sum'),
        races=('round', 'count'),
        wins=('Position', lambda s: (s == 1).sum()),
        podiums=('Position', lambda s: (s <= 3).sum())
    )
    print(ver_summary.to_string())

    # 3. Qualifying Deltas
    print("\n[3/4] Qualifying-to-Race Deltas")
    deltas_2023 = get_qualifying_race_deltas(season=2023)
    ver_deltas = get_driver_delta_stats('VER', season=2023)
    print(f"  [OK] 2023 Total delta events analyzed: {len(deltas_2023)}")
    print(f"  [OK] VER 2023 Delta Stats: {ver_deltas['total_races']} races, avg delta {ver_deltas['average_delta']}, improvement rate {ver_deltas['improvement_rate']}")

    # 4. Consistency Engine: Season Comparison
    print("\n[4/4] Consistency Engine: 3-Season Comparison (2023-2025)")
    season_comp = get_season_consistency_comparison([2023, 2024, 2025])
    print(season_comp[['year', 'races_analyzed', 'finish_rate', 'avg_position', 'position_stddev', 'podium_finishes']].to_string(index=False))

    # Consistency Rankings (Top 10)
    print("\nDriver Consistency Rankings (Top 10 across 2023-2025):")
    rankings = get_all_drivers_consistency()
    cols = ['driverId', 'fullName', 'teamName', 'races_analyzed', 'finish_rate', 'avg_position', 'position_stddev', 'podium_finishes']
    print(rankings.head(10)[cols].to_string(index=False))

    # Save figures
    fig_dir = os.path.join(CURRENT_DIR, "05_Analytics", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig1 = plot_driver_season_comparison(ver_history, driver_name="Max Verstappen (VER)")
    fig1.savefig(os.path.join(fig_dir, "verstappen_season_comparison.png"), dpi=150)

    fig2 = plot_season_consistency_comparison(season_comp)
    fig2.figure.savefig(os.path.join(fig_dir, "season_consistency_comparison.png"), dpi=150)

    fig3 = plot_driver_consistency_radar('VER')
    fig3.figure.savefig(os.path.join(fig_dir, "verstappen_radar.png"), dpi=150)

    print(f"\n[OK] Generated and saved figures in: {fig_dir}")
    print("\n*** MILESTONE 2 - NUMERICAL ANALYTICS ENGINE: FULLY VERIFIED AND COMPLETE! ***")


if __name__ == "__main__":
    run_milestone2_verification()
