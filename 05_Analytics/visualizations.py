"""
 Data Visualization for APEX AI Analytics

Creates charts and plots from the analytics engine outputs using
matplotlib and seaborn. Designed to work on low-end PCs (8GB RAM)
with lightweight, publication-quality figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os

# Set style for all plots
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'lines.linewidth': 2,
})


def plot_driver_standings(standings_df: pd.DataFrame,
                         top_n: int = 15,
                         title: str = "F1 Championship Standings") -> plt.Axes:
    """
    Plot a horizontal bar chart of driver championship points.

    Parameters
    ----------
    standings_df : pd.DataFrame
        Output from get_driver_standings()
    top_n : int
        Number of top drivers to show
    title : str
        Plot title

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plot
    """
    # Take top N
    plot_df = standings_df.head(top_n).copy()

    fig, ax = plt.subplots()
    bars = ax.barh(plot_df['FullName'], plot_df['Points'],
                   color=sns.color_palette("viridis", len(plot_df)))

    ax.set_xlabel('Points')
    ax.set_ylabel('Driver')
    ax.set_title(title)
    ax.invert_yaxis()  # Highest points at top

    # Add points labels
    for i, (name, pts) in enumerate(zip(plot_df['FullName'], plot_df['Points'])):
        ax.text(pts + 1, i, str(int(pts)),
                va='center', fontsize=8)

    fig.tight_layout()
    return ax


def plot_delta_distribution(deltas_df: pd.DataFrame,
                            title: str = "Qualifying-to-Race Position Deltas") -> plt.Axes:
    """
    Plot histogram of qualifying-to-race position deltas.

    Parameters
    ----------
    deltas_df : pd.DataFrame
        Output from get_qualifying_race_deltas()
    title : str
        Plot title

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plot
    """
    fig, ax = plt.subplots()

    # Filter out NaN deltas
    valid_deltas = deltas_df['Delta'].dropna()

    sns.histplot(valid_deltas, bins=range(-25, 26, 1),
                 kde=True, ax=ax, color='steelblue')

    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Net Neutral')
    ax.set_xlabel('Delta (Qualify Pos - Race Pos)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

    # Add interpretation text
    improvements = (valid_deltas > 0).sum()
    drops = (valid_deltas < 0).sum()
    ax.text(0.02, 0.98, f'Improvements: {improvements} | Drops: {drops}',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    return ax


def plot_season_consistency_comparison(season_df: pd.DataFrame,
                                       title: str = "Season-to-Season Consistency Comparison") -> plt.Axes:
    """
    Bar chart comparing consistency metrics across seasons.

    Parameters
    ----------
    season_df : pd.DataFrame
        Output from get_season_consistency_comparison()
    title : str
        Plot title

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plot
    """
    # Only plot seasons with data
    plot_df = season_df.dropna(subset=['avg_position'])

    fig, ax = plt.subplots()

    bars = ax.bar(plot_df['year'].astype(str), plot_df['avg_position'],
                  color=sns.color_palette("muted", len(plot_df)))

    ax.set_xlabel('Season')
    ax.set_ylabel('Average Position (lower = better)')
    ax.set_title(title)
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df['year'].astype(str))

    # Add text on bars
    for i, (year, avg_pos) in enumerate(zip(plot_df['year'], plot_df['avg_position'])):
        if avg_pos is not None:
            ax.text(i, avg_pos + 0.1, f'{avg_pos:.2f}',
                    ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    return ax


def plot_driver_consistency_radar(driver_id: str,
                                  seasons: Optional[List[int]] = None,
                                  title: str = "Driver Consistency Profile") -> plt.Axes:
    """
    Create a radar/spider chart of driver consistency metrics.

    Note: This requires a moderate amount of memory. For low-end PCs,
    limit to 1-2 drivers at a time.

    Parameters
    ----------
    driver_id : str
        The driver's driver ID or abbreviation
    seasons : list of int or None
        Specific seasons. If None, all seasons.
    title : str
        Plot title

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plot
    """
    import sqlite3

    con = sqlite3.connect('02_Data_Processed/f1.db')
    query = """
        SELECT Position, Laps, finished
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
        fig, ax = plt.subplots()
        ax.set_title(f"No data for driver {driver_id}")
        return ax

    finished = df[df.finished == 1]

    # Compute metrics
    positions = finished['Position'].astype(float)
    avg_pos = positions.mean()
    std_pos = positions.std()
    var_pos = positions.var()
    podium = int((positions <= 3).sum())
    top10 = int((positions <= 10).sum())
    finish_rate = len(finished) / len(df) * 100 if len(df) > 0 else 0
    avg_laps = finished['Laps'].mean() if 'Laps' in finished.columns else 0

    max_position = 20  # F1 has 20 cars
    norm_avg_pos = min(avg_pos / max_position, 1.0)
    norm_finish_rate = 1 - (finish_rate / 100)  # lower is better for consistency
    norm_podium = 1 - (podium / max(len(finished), 1))  # lower variability = better
    norm_avg_laps = min(avg_laps / 70, 1.0)  # typical race ~70 laps

    # Number of variables
    N = 4

    # Compute angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Values (normalized, lower is better so we invert some)
    values = [norm_avg_pos, norm_finish_rate, norm_podium, norm_avg_laps]
    values += values[:1]  # Close the loop

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    ax.plot(angles, values, 'o-', linewidth=2, label=driver_id)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([
        'Avg Position', 'Finish Rate', 'Podium Rate', 'Avg Laps'
    ], size=8)

    ax.set_ylim(0, 1)
    ax.set_title(title, size=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    return ax


def plot_driver_season_comparison(comparison_df: pd.DataFrame,
                                  driver_name: str = "Driver",
                                  title: Optional[str] = None) -> plt.Figure:
    """
    Plot driver multi-season comparison showing points/wins per season and race finish positions.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from get_driver_season_comparison()
    driver_name : str
        Name or ID of driver for plot labels
    title : str, optional
        Overall chart title

    Returns
    -------
    matplotlib.figure.Figure
        The Figure object
    """
    if comparison_df.empty:
        fig, ax = plt.subplots()
        ax.set_title(f"No data for {driver_name}")
        return fig

    title = title or f"{driver_name} - Multi-Season Performance"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Total Points & Wins per Season
    season_summary = comparison_df.groupby('year').agg(
        points=('Points', 'sum'),
        wins=('Position', lambda s: (s == 1).sum()),
        podiums=('Position', lambda s: (s <= 3).sum()),
        races=('round', 'count')
    ).reset_index()

    bars = ax1.bar(season_summary['year'].astype(str), season_summary['points'],
                   color=sns.color_palette("viridis", len(season_summary)))
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Total Points')
    ax1.set_title(f'{driver_name} Championship Points by Season')
    for i, row in season_summary.iterrows():
        ax1.text(i, row['points'] + 8, f"{int(row['points'])} pts\n({int(row['wins'])} wins)",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 2: Finishing Positions by Round across Seasons
    for year, group in comparison_df.groupby('year'):
        finished_group = group[group['finished'] == 1]
        ax2.plot(finished_group['round'], finished_group['Position'], marker='o', label=f"{year}")

    ax2.set_xlabel('Round')
    ax2.set_ylabel('Finishing Position')
    ax2.set_title('Finishing Position Progression (lower = better)')
    ax2.invert_yaxis()
    ax2.legend()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig
