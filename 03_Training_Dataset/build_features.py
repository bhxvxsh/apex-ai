"""
APEX AI - Enhanced Feature Engineering & Dataset Builder

Generates high-signal Formula 1 features with strict temporal leak prevention:
- Qualifying lap performance: fastest lap time, pole delta in seconds, teammate delta.
- Driver rolling form: 3-race and 5-race rolling race & qualifying finishes, momentum trend.
- Track/race dynamics: historical positions gained from grid slot, finish rates, podium rates.
- Binary targets for high-precision classification: is_podium (Top-3), is_top10 (Points).
"""

import os
import sqlite3
import pandas as pd
import numpy as np


def load_raw_data(db_path: str) -> pd.DataFrame:
    """Load race and qualifying results merged by event and driver."""
    con = sqlite3.connect(db_path)

    q_df = pd.read_sql("""
        SELECT year, round, event_name, DriverId, Abbreviation, FullName, TeamName,
               Position as QualPosition, Q1_s, Q2_s, Q3_s
        FROM qualifying_results
    """, con)

    r_df = pd.read_sql("""
        SELECT year, round, event_name, DriverId, Abbreviation, TeamName,
               Position as RacePosition, GridPosition, Points, Laps, finished
        FROM race_results
    """, con)

    con.close()

    # Lap times & pole deltas
    q_df['best_q_s'] = q_df[['Q1_s', 'Q2_s', 'Q3_s']].min(axis=1)

    # Pole time per session
    pole_times = q_df.groupby(['year', 'round'])['best_q_s'].min().reset_index().rename(columns={'best_q_s': 'pole_time_s'})
    q_df = pd.merge(q_df, pole_times, on=['year', 'round'], how='left')
    q_df['pole_delta_s'] = (q_df['best_q_s'] - q_df['pole_time_s']).fillna(2.5)

    # Teammate best time per session
    team_best = q_df.groupby(['year', 'round', 'TeamName'])['best_q_s'].min().reset_index().rename(columns={'best_q_s': 'team_best_q_s'})
    q_df = pd.merge(q_df, team_best, on=['year', 'round', 'TeamName'], how='left')
    q_df['teammate_delta_s'] = (q_df['best_q_s'] - q_df['team_best_q_s']).fillna(0.5)

    # Merge qualifying and race results
    df = pd.merge(
        q_df,
        r_df[['year', 'round', 'DriverId', 'RacePosition', 'GridPosition', 'Points', 'Laps', 'finished']],
        on=['year', 'round', 'DriverId'],
        how='outer'
    )

    # Fill GridPosition with QualPosition if GridPosition is missing or None
    df['GridPosition'] = df['GridPosition'].fillna(df['QualPosition'])

    # Sort strictly chronologically
    df = df.sort_values(['year', 'round', 'QualPosition', 'RacePosition']).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling historical features strictly prior to the current event round
    to prevent look-ahead data leakage.
    """
    events = df[['year', 'round']].drop_duplicates().sort_values(['year', 'round']).reset_index(drop=True)
    events['event_idx'] = np.arange(len(events))
    df = pd.merge(df, events, on=['year', 'round'], how='left')

    features = []

    for idx, row in df.iterrows():
        cur_event_idx = row['event_idx']
        cur_driver = row['DriverId']
        cur_team = row['TeamName']

        # Historical data strictly prior to this event
        hist = df[df['event_idx'] < cur_event_idx]

        # Driver history
        d_hist = hist[hist['DriverId'] == cur_driver]
        # Team history
        t_hist = hist[hist['TeamName'] == cur_team]

        # 1. Driver rolling qualifying position
        d_qual = d_hist['QualPosition'].dropna()
        driver_qual_mean_all = float(d_qual.mean()) if len(d_qual) > 0 else 12.0
        driver_qual_mean_last3 = float(d_qual.tail(3).mean()) if len(d_qual) >= 1 else driver_qual_mean_all
        driver_qual_best = float(d_qual.min()) if len(d_qual) > 0 else 15.0

        # 2. Driver rolling race finish position
        d_race = d_hist[d_hist['finished'] == 1]['RacePosition'].dropna()
        driver_race_mean_all = float(d_race.mean()) if len(d_race) > 0 else 12.0
        driver_race_mean_last3 = float(d_race.tail(3).mean()) if len(d_race) >= 1 else driver_race_mean_all
        driver_finish_rate = float(d_hist['finished'].mean() * 100) if len(d_hist) > 0 else 85.0
        driver_podium_rate = float((d_race <= 3).mean() * 100) if len(d_race) > 0 else 0.0
        driver_top10_rate = float((d_race <= 10).mean() * 100) if len(d_race) > 0 else 50.0

        # 3. Driver momentum: is the driver outperforming their career average lately?
        driver_momentum = round(driver_race_mean_all - driver_race_mean_last3, 2)

        # 4. Driver historical grid-to-race delta (positions gained)
        d_deltas = (d_hist['GridPosition'] - d_hist['RacePosition']).dropna()
        driver_avg_gain = float(d_deltas.mean()) if len(d_deltas) > 0 else 0.0

        # 5. Team rolling metrics
        t_qual = t_hist['QualPosition'].dropna()
        team_qual_mean = float(t_qual.mean()) if len(t_qual) > 0 else 11.0
        team_qual_mean_last3 = float(t_qual.tail(6).mean()) if len(t_qual) >= 2 else team_qual_mean

        t_race = t_hist[t_hist['finished'] == 1]['RacePosition'].dropna()
        team_race_mean = float(t_race.mean()) if len(t_race) > 0 else 11.0
        team_points_ytd = float(t_hist[t_hist['year'] == row['year']]['Points'].sum()) if len(t_hist) > 0 else 0.0

        # 6. Driver points entering event
        driver_points_ytd = float(d_hist[d_hist['year'] == row['year']]['Points'].sum()) if len(d_hist) > 0 else 0.0

        # Targets
        is_podium = 1 if (row['finished'] == 1 and row['RacePosition'] <= 3.0) else 0
        is_top10 = 1 if (row['finished'] == 1 and row['RacePosition'] <= 10.0) else 0

        feat_dict = {
            'year': row['year'],
            'round': row['round'],
            'event_idx': cur_event_idx,
            'DriverId': cur_driver,
            'TeamName': cur_team,
            'GridPosition': row['GridPosition'],
            'QualPosition': row['QualPosition'],
            'pole_delta_s': round(float(row['pole_delta_s']), 3) if pd.notnull(row['pole_delta_s']) else 2.5,
            'teammate_delta_s': round(float(row['teammate_delta_s']), 3) if pd.notnull(row['teammate_delta_s']) else 0.5,
            'driver_qual_mean_all': round(driver_qual_mean_all, 2),
            'driver_qual_mean_last3': round(driver_qual_mean_last3, 2),
            'driver_qual_best': round(driver_qual_best, 2),
            'driver_race_mean_all': round(driver_race_mean_all, 2),
            'driver_race_mean_last3': round(driver_race_mean_last3, 2),
            'driver_momentum': driver_momentum,
            'driver_avg_gain': round(driver_avg_gain, 2),
            'driver_finish_rate': round(driver_finish_rate, 2),
            'driver_podium_rate': round(driver_podium_rate, 2),
            'driver_top10_rate': round(driver_top10_rate, 2),
            'driver_points_ytd': round(driver_points_ytd, 1),
            'team_qual_mean': round(team_qual_mean, 2),
            'team_qual_mean_last3': round(team_qual_mean_last3, 2),
            'team_race_mean': round(team_race_mean, 2),
            'team_points_ytd': round(team_points_ytd, 1),
            'RacePosition': row['RacePosition'],
            'finished': row['finished'],
            'is_podium': is_podium,
            'is_top10': is_top10,
        }
        features.append(feat_dict)

    feat_df = pd.DataFrame(features)
    return feat_df


def build_and_save_datasets(db_path: str = "02_Data_Processed/f1.db",
                            out_dir: str = "03_Training_Dataset"):
    """Build datasets and save train/test splits."""
    os.makedirs(out_dir, exist_ok=True)
    raw_df = load_raw_data(db_path)
    feat_df = engineer_features(raw_df)

    # Temporal split: Train on 2023-2024, Test on 2025
    train_df = feat_df[feat_df['year'].isin([2023, 2024])].reset_index(drop=True)
    test_df = feat_df[feat_df['year'] == 2025].reset_index(drop=True)

    full_path = os.path.join(out_dir, "full_features.csv")
    train_path = os.path.join(out_dir, "train_dataset.csv")
    test_path = os.path.join(out_dir, "test_dataset.csv")

    feat_df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Enhanced features created successfully!")
    print(f"  Full dataset:      {len(feat_df)} rows -> {full_path}")
    print(f"  Train (2023-2024): {len(train_df)} rows -> {train_path}")
    print(f"  Test  (2025):      {len(test_df)} rows -> {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    db_file = os.path.join(base_dir, "02_Data_Processed", "f1.db")
    dataset_dir = os.path.join(base_dir, "03_Training_Dataset")
    build_and_save_datasets(db_file, dataset_dir)
