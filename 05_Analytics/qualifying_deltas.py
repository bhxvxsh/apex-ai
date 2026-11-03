"""
 Qualifying-to-Race Delta Analysis

Computes the gap between qualifying position and race position for each driver,
measuring how well qualifying translates to race performance.
"""

import sqlite3
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def get_qualifying_race_deltas(season: Optional[int] = None,
                               minimum_laps: int = 1) -> pd.DataFrame:
    """
    Compute qualifying-to-race position deltas for all drivers.

    A positive delta means the driver improved (qualifying position worse than
    race position, e.g., qualified 5th, finished 3rd → delta = -2).
    A negative delta means the driver dropped (qualifying good, race worse).

    Parameters
    ----------
    season : int or None
        Filter to a specific season. If None, use all seasons.
    minimum_laps : int
        Only include races where the driver completed at least this many laps.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: DriverId, FullName, TeamName, year,
        event_name, round, GridPosition, Position, Delta,
        DeltaInterpretation
    """
    con = sqlite3.connect('02_Data_Processed/f1.db')

    # Get qualifying results with position info
    qual_query = """
        SELECT DriverId, year, event_name, round, Position AS GridPosition
        FROM qualifying_results
    """
    params = []
    if season is not None:
        qual_query += " WHERE year = ?"
        params = [season]
    else:
        qual_query += " "

    qual_df = pd.read_sql(qual_query, con, params=params)

    # Get race results with position info
    race_query = """
        SELECT DriverId, year, event_name, round, Position, Laps, finished
        FROM race_results
        WHERE finished = 1
    """
    params_r = []
    if season is not None:
        race_query += " AND year = ?"
        params_r = [season]

    race_df = pd.read_sql(race_query, con, params=params_r)

    con.close()

    # Only include drivers who finished the race (completed minimum laps)
    if minimum_laps > 0:
        race_df = race_df[race_df.Laps >= minimum_laps]

    # Merge qualifying with race results
    merged = qual_df.merge(
        race_df[['DriverId', 'year', 'event_name', 'round', 'Position', 'Laps']],
        on=['DriverId', 'year', 'event_name', 'round'],
        how='inner',
    )

    # Compute delta: qualifying position minus race position
    # Positive = improved (started worse, finished better)
    # Negative = dropped (started better, finished worse)
    merged['Delta'] = merged['GridPosition'] - merged['Position']

    # Interpretation
    conditions = [
        merged['Delta'] > 0,   # Improved
        merged['Delta'] == 0,  # Same position
        merged['Delta'] < 0,   # Dropped
    ]
    choices = ['Improved', 'Same Position', 'Dropped']
    merged['DeltaInterpretation'] = np.select(conditions, choices, default='Unknown')

    # Add driver info
    merged['FullName'] = merged['DriverId'].map(
        lambda d: get_driver_name(d)
    )
    merged['TeamName'] = merged['DriverId'].map(
        lambda d: get_driver_team(d)
    )

    # Sort by year, round, delta (largest improvements first)
    merged = merged.sort_values(['year', 'round', 'Delta'], ascending=[True, True, False]).reset_index(drop=True)

    return merged


def get_driver_delta_stats(driver_id: str,
                           season: Optional[int] = None) -> Dict:
    """
    Get summary statistics for a driver's qualifying-to-race deltas.

    Parameters
    ----------
    driver_id : str
        The driver's driver ID
    season : int or None
        If provided, only this season

    Returns
    -------
    Dict with keys: total_races, improvements, drops, average_delta,
                    best_improvement, worst_drop, improvement_rate
    """
    con = sqlite3.connect('02_Data_Processed/f1.db')

    qual_query = """
        SELECT Position AS GridPosition
        FROM qualifying_results
        WHERE (DriverId = ? OR Abbreviation = ?)
    """
    params = [driver_id, driver_id]

    race_query = """
        SELECT Position, finished, Laps
        FROM race_results
        WHERE (DriverId = ? OR Abbreviation = ?) AND finished = 1
    """
    params_r = [driver_id, driver_id]

    if season is not None:
        qual_query += " AND year = ?"
        params.append(season)
        race_query += " AND year = ?"
        params_r.append(season)

    qual_df = pd.read_sql(qual_query, con, params=params)
    race_df = pd.read_sql(race_query, con, params=params_r)

    con.close()

    if race_df.empty:
        return {
            'total_races': 0,
            'improvements': 0,
            'drops': 0,
            'average_delta': 0,
            'best_improvement': 0,
            'worst_drop': 0,
            'improvement_rate': 0,
        }

    # Need qualifying positions for the same events
    # Merge on year + event_name + round
    # First get qualifying with event/round info
    con2 = sqlite3.connect('02_Data_Processed/f1.db')
    qual_full = pd.read_sql(
        """
        SELECT DriverId, year, event_name, round, Position AS GridPosition
        FROM qualifying_results
        WHERE (DriverId = ? OR Abbreviation = ?)
    """, con2, params=[driver_id, driver_id])

    race_full = pd.read_sql(
        """
        SELECT year, event_name, round, Position, finished, Laps
        FROM race_results
        WHERE (DriverId = ? OR Abbreviation = ?) AND finished = 1
    """, con2, params=[driver_id, driver_id])
    con2.close()

    if season is not None:
        qual_full = qual_full[qual_full.year == season]
        race_full = race_full[race_full.year == season]

    # Merge
    merged = qual_full.merge(
        race_full[['year', 'event_name', 'round', 'Position', 'Laps']],
        on=['year', 'event_name', 'round'],
        how='inner',
    )

    if merged.empty:
        return {
            'total_races': 0,
            'improvements': 0,
            'drops': 0,
            'average_delta': 0,
            'best_improvement': 0,
            'worst_drop': 0,
            'improvement_rate': 0,
        }

    merged['Delta'] = merged['GridPosition'] - merged['Position']

    total_races = len(merged)
    improvements = int((merged['Delta'] > 0).sum())
    drops = int((merged['Delta'] < 0).sum())
    valid_deltas = merged['Delta'].dropna()
    avg_delta = float(valid_deltas.mean()) if not valid_deltas.empty else 0.0
    best_improvement = int(valid_deltas.max()) if not valid_deltas.empty else 0
    worst_drop = int(valid_deltas.min()) if not valid_deltas.empty else 0
    improvement_rate = improvements / total_races if total_races > 0 else 0

    return {
        'total_races': total_races,
        'improvements': improvements,
        'drops': drops,
        'average_delta': round(avg_delta, 3),
        'best_improvement': best_improvement,
        'worst_drop': worst_drop,
        'improvement_rate': round(improvement_rate, 3),
    }


def get_driver_name(driver_id: str) -> str:
    """Helper: get driver full name from dataset."""
    con = sqlite3.connect('02_Data_Processed/f1.db')
    cur = con.cursor()
    cur.execute("SELECT FullName FROM race_results WHERE DriverId = ? OR Abbreviation = ? LIMIT 1", (driver_id, driver_id))
    row = cur.fetchone()
    con.close()
    return row[0] if row else "Unknown"


def get_driver_team(driver_id: str) -> str:
    """Helper: get driver team from dataset."""
    con = sqlite3.connect('02_Data_Processed/f1.db')
    cur = con.cursor()
    cur.execute("SELECT TeamName FROM race_results WHERE DriverId = ? OR Abbreviation = ? LIMIT 1", (driver_id, driver_id))
    row = cur.fetchone()
    con.close()
    return row[0] if row else "Unknown"