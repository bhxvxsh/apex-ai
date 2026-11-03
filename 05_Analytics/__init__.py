"""
 APEX AI Analytics Engine

Package for Formula 1 data analytics using SQLite database from the
Data Pipeline (Milestone 1).

Sub-modules:
- driver_performance: Driver standings and season comparisons
- qualifying_deltas: Qualifying-to-race position analysis
- consistency: Consistency metrics and comparisons
- visualizations: Plots and charts using matplotlib/seaborn

Example usage:
    from apex_ai_analytics import (
        get_driver_standings,
        get_qualifying_race_deltas,
        get_driver_consistency,
        plot_driver_standings
    )

    # Get 2023 standings
    standings = get_driver_standings(season=2023)
    print(standings.head(10))

    # Analyze Verstappen's qualifying-to-race deltas
    deltas = get_qualifying_race_deltas(season=2023)
    print(deltas.head())

    # Check consistency
    consistent = get_driver_consistency('VER')
    print(consistent)
"""

# Re-export key functions for convenience
from .driver_performance import (
    get_driver_standings,
    get_driver_season_comparison,
    get_all_drivers_list,
)
from .qualifying_deltas import (
    get_qualifying_race_deltas,
    get_driver_delta_stats,
)
from .consistency import (
    get_driver_consistency,
    get_season_consistency_comparison,
    get_all_drivers_consistency,
)
from .visualizations import (
    plot_driver_standings,
    plot_delta_distribution,
    plot_season_consistency_comparison,
    plot_driver_consistency_radar,
    plot_driver_season_comparison,
)

__all__ = [
    'get_driver_standings',
    'get_driver_season_comparison',
    'get_all_drivers_list',
    'get_qualifying_race_deltas',
    'get_driver_delta_stats',
    'get_driver_consistency',
    'get_season_consistency_comparison',
    'get_all_drivers_consistency',
    'plot_driver_standings',
    'plot_delta_distribution',
    'plot_season_consistency_comparison',
    'plot_driver_consistency_radar',
    'plot_driver_season_comparison',
]