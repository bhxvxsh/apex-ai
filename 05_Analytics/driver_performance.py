"""
 Driver Performance Analysis

Compares drivers across seasons based on race results stored in f1.db.
Uses only stdlib sqlite3 + pandas for data manipulation.
"""

import sqlite3
from typing import Dict, List, Optional, Tuple
import pandas as pd


def get_driver_standings(season: Optional[int] = None,
                        minimum_laps: int = 0) -> pd.DataFrame:
    """
    Get driver championship standings for a season (or all seasons).

    Parameters
    ----------
    season : int or None
        If provided, only return data for this year. If None, all years.
    minimum_laps : int
        Minimum completed laps to qualify (filters out DNFs with 0 laps).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: DriverId, FullName, TeamName,
        Points, LapsCompleted, RacesEntered, PositionAvg,
        PointsPerRace, ConsistencyScore
    """
    con = sqlite3.connect('02_Data_Processed/f1.db')
    query = """
        SELECT DriverId, FullName, TeamName, Position, Points, Laps,
               year, event_name, round, finished
        FROM race_results
        WHERE finished = 1
    """
    if season is not None:
        query += " AND year = ?"
        params = [season]
    else:
        params = []

    df = pd.read_sql(query + " ORDER BY year, round", con, params=params)
    con.close()

    if minimum_laps > 0:
        df = df[df.Laps >= minimum_laps]

    # Aggregate per driver
    agg = df.groupby(['DriverId', 'FullName', 'TeamName']).agg(
        Points=('Points', 'sum'),
        LapsCompleted=('Laps', 'sum'),
        RacesEntered=('round', 'count'),
        PositionAvg=('Position', 'mean'),
        PointsPerRace=('Points', lambda x: x.sum() / len(x) if len(x) > 0 else 0),
    ).reset_index()

    # Consistency: lower standard deviation of position = more consistent
    pos_std = df.groupby('DriverId')['Position'].std().reset_index()
    pos_std.columns = ['DriverId', 'PositionStd']
    agg = agg.merge(pos_std, on='DriverId')

    # Sort by points descending
    agg = agg.sort_values('Points', ascending=False).reset_index(drop=True)

    return agg


def get_driver_season_comparison(driver_id: str,
                                 seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Get a driver's race-by-race performance across seasons.

    Parameters
    ----------
    driver_id : str
        The driver's driver ID (e.g., 'VER', 'HAM', 'PER')
    seasons : list of int or None
        Specific seasons to include. If None, uses all available.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year, round, event_name, Position,
        GridPosition, Points, Laps, finished
    """
    con = sqlite3.connect('02_Data_Processed/f1.db')
    query = """
        SELECT year, round, event_name, Position, GridPosition,
               Points, Laps, finished
        FROM race_results
        WHERE (DriverId = ? OR Abbreviation = ?)
    """
    params = [driver_id, driver_id]

    if seasons:
        query += " AND year IN (" + ",".join("?" * len(seasons)) + ")"
        params.extend(seasons)

    query += " ORDER BY year, round"
    df = pd.read_sql(query, con, params=params)
    con.close()
    return df


def get_all_drivers_list() -> List[Dict[str, str]]:
    """
    Get a complete list of unique drivers in the dataset.

    Returns
    -------
    List[Dict]
        List of dicts with keys: driverId, fullName, abbreviation,
        teamName, countryCode
    """
    con = sqlite3.connect('02_Data_Processed/f1.db')
    cur = con.cursor()
    cur.execute("""
        SELECT DISTINCT DriverId, FullName,
               (SELECT TeamName FROM race_results r2
                WHERE r2.DriverId = race_results.DriverId
                LIMIT 1) as TeamName,
               (SELECT CountryCode FROM race_results r3
                WHERE r3.DriverId = race_results.DriverId
                LIMIT 1) as CountryCode
        FROM race_results
        ORDER BY FullName
    """)
    rows = cur.fetchall()
    con.close()

    drivers = []
    for row in rows:
        drivers.append({
            'driverId': row[0],
            'fullName': row[1],
            'teamName': row[2] or 'Unknown',
            'countryCode': row[3] or 'Unknown',
        })
    return drivers