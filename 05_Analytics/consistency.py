"""
 Consistency Metrics Analysis

Measures driver performance consistency across seasons using position
variance, finish rates, and lap completion statistics.
"""

import sqlite3
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def get_driver_consistency(driver_id: str,
                           seasons: Optional[List[int]] = None) -> Dict:
    """
    Compute comprehensive consistency metrics for a driver.

    Parameters
    ----------
    driver_id : str
        The driver's driver ID (e.g., 'VER', 'HAM', 'PER')
    seasons : list of int or None
        Specific seasons to analyze. If None, uses all available.

    Returns
    -------
    Dict with consistency metrics:
        - races_analyzed: number of races included
        - finish_rate: percentage of races finished
        - position_stddev: standard deviation of race positions (lower = more consistent)
        - position_variance: variance of race positions
        - avg_position: average finishing position
        - podium_finishes: count of top-3 finishes
        - top_10_finishes: count of top-10 finishes
        - did_not_finish: count of DNFs
        - avg_laps_completed: average laps finished per race
        - completion_rate: percentage of total possible laps completed
    """
    con = sqlite3.connect('02_Data_Processed/f1.db')

    query = """
        SELECT Position, Laps, finished, year, event_name, round
        FROM race_results
        WHERE (DriverId = ? OR Abbreviation = ?)
    """
    params = [driver_id, driver_id]

    if seasons:
        query += " AND year IN (" + ",".join("?" * len(seasons)) + ")"
        params.extend(seasons)

    df = pd.read_sql(query, con, params=params)
    con.close()

    if df.empty:
        return {
            'races_analyzed': 0,
            'finish_rate': 0.0,
            'position_stddev': 0.0,
            'position_variance': 0.0,
            'avg_position': 0.0,
            'podium_finishes': 0,
            'top_10_finishes': 0,
            'did_not_finish': 0,
            'avg_laps_completed': 0.0,
            'completion_rate': 0.0,
        }

    total_races = len(df)
    finished = df[df.finished == 1]
    dnf = df[df.finished == 0]

    total_races = len(df)
    finished = df[df.finished == 1]
    dnf = df[df.finished == 0]

    # Position statistics (only finished races for stddev/mean)
    finished_positions = finished['Position'].astype(float)

    position_stddev = float(finished_positions.std()) if len(finished_positions) > 1 else 0.0
    position_variance = float(finished_positions.var()) if len(finished_positions) > 0 else 0.0
    avg_position = float(finished_positions.mean()) if len(finished_positions) > 0 else 0.0

    podium_finishes = int((finished['Position'] <= 3).sum())
    top_10_finishes = int((finished['Position'] <= 10).sum())
    did_not_finish = int((df['finished'] == 0).sum())

    avg_laps_completed = float(df['Laps'].mean()) if 'Laps' in df.columns else 0.0

    # Completion rate: total laps completed / (total races * typical race laps)
    # Estimate typical race laps from the data
    typical_race_laps = float(df['Laps'].median()) if 'Laps' in df.columns else 50.0
    max_possible_laps = total_races * typical_race_laps
    completed_laps = float(df['Laps'].sum()) if 'Laps' in df.columns else 0.0
    completion_rate = round(completed_laps / max_possible_laps * 100, 2) if max_possible_laps > 0 else 0.0

    return {
        'races_analyzed': total_races,
        'finish_rate': round(len(finished) / total_races * 100, 1) if total_races > 0 else 0.0,
        'position_stddev': round(position_stddev, 3),
        'position_variance': round(position_variance, 3),
        'avg_position': round(avg_position, 2),
        'podium_finishes': podium_finishes,
        'top_10_finishes': top_10_finishes,
        'did_not_finish': did_not_finish,
        'avg_laps_completed': round(avg_laps_completed, 1),
        'completion_rate': completion_rate,
    }


def get_season_consistency_comparison(seasons: List[int]) -> pd.DataFrame:
    """
    Compare consistency across multiple seasons.

    Parameters
    ----------
    seasons : list of int
        List of season years to compare

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year, avg_position, position_stddev,
        podium_finishes, top_10_finishes, finish_rate, races_analyzed
        Sorted by avg_position (ascending = best).
    """
    results = []
    for year in sorted(seasons):
        con = sqlite3.connect('02_Data_Processed/f1.db')
        query = """
            SELECT Position, Laps, finished
            FROM race_results
            WHERE year = ?
        """
        df = pd.read_sql(query, con, params=[year])
        con.close()

        if df.empty:
            results.append({
                'year': year,
                'avg_position': None,
                'position_stddev': None,
                'position_variance': None,
                'podium_finishes': 0,
                'top_10_finishes': 0,
                'finish_rate': 0.0,
                'races_analyzed': 0,
            })
            continue

        total = len(df)
        finished = df[df.finished == 1]
        finished_pos = finished['Position'].astype(float)

        results.append({
            'year': year,
            'avg_position': round(finished_pos.mean(), 2) if len(finished_pos) > 0 else None,
            'position_stddev': round(finished_pos.std(), 3) if len(finished_pos) > 1 else 0.0,
            'position_variance': round(finished_pos.var(), 3) if len(finished_pos) > 0 else 0.0,
            'podium_finishes': int((finished['Position'] <= 3).sum()),
            'top_10_finishes': int((finished['Position'] <= 10).sum()),
            'finish_rate': round(len(finished) / total * 100, 1) if total > 0 else 0.0,
            'races_analyzed': total,
        })

    return pd.DataFrame(results).sort_values('avg_position', ascending=True)


def get_all_drivers_consistency(seasons: Optional[List[int]] = None
                               ) -> pd.DataFrame:
    """
    Compute consistency metrics for ALL drivers in the dataset.

    Parameters
    ----------
    seasons : list of int or None
        If None, all seasons. If provided, only these seasons.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per driver, columns:
        driverId, fullName, teamName, races_analyzed, finish_rate,
        position_stddev, avg_position, podium_finishes, top_10_finishes
        Sorted by position_stddev (ascending = most consistent).
    """
    con = sqlite3.connect('02_Data_Processed/f1.db')
    query = """
        SELECT DriverId, FullName, TeamName, Position, Laps, finished, year
        FROM race_results
    """
    params = []

    if seasons:
        query += " AND year IN (" + ",".join("?" * len(seasons)) + ")"
        params.extend(seasons)

    query += " ORDER BY DriverId, year, round"
    df = pd.read_sql(query, con, params=params)
    con.close()

    if df.empty:
        return pd.DataFrame()

    # Aggregate per driver
    driver_stats = []

    for driver_id in df['DriverId'].unique():
        driver_df = df[df['DriverId'] == driver_id]
        name = driver_df['FullName'].iloc[0]
        team = driver_df['TeamName'].iloc[0] if 'TeamName' in driver_df.columns else 'Unknown'

        stats = get_driver_consistency(driver_id, seasons=seasons)
        # Override with correct name/team
        stats['driverId'] = driver_id
        stats['fullName'] = name
        stats['teamName'] = team

        driver_stats.append(stats)

    result_df = pd.DataFrame(driver_stats)
    # Sort by consistency (lower position_stddev = more consistent)
    result_df = result_df.sort_values('position_stddev', ascending=True).reset_index(drop=True)

    return result_df