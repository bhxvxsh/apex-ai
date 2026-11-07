"""
APEX AI - Enhanced Feature Engineering & Dataset Builder (Betterments 1 & 2)

Generates high-signal Formula 1 features with strict temporal leak prevention:
1. Corner-Level Micro-Sector Telemetry & Qualifying Dynamics (Betterment 1):
   - Relative pace efficiency (ratio to pole lap time)
   - Intra-session progression (Q1 to Q2/Q3 delta reduction %)
   - Q3 execution indicator (operating in maximum downforce/grip envelope)
   - Estimated corner apex speed efficiency benchmark
2. Circuit Aerodynamic Taxonomy & Physical Clustering (Betterment 2):
   - Circuit cluster ID (Low-Downforce, Thermal Deg, Mechanical Traction, Street Hybrid)
   - Asphalt tire stress index (2.1 - 5.0)
   - Full throttle duty cycle percentage (45.0% - 79.0%)
   - Decisive corner apex speed benchmark V_min (km/h)
   - Overtaking difficulty index & grid sensitivity interaction
   - Thermal degradation exposure interaction
3. Driver & Team Rolling Longitudinal Form:
   - 3-race and career rolling averages for race & qualifying finishes
   - Momentum trend (career mean minus recent mean)
   - Historical grid-to-finish position delta gains
   - Podium and points finish conversion rates
4. Supervised ML Targets:
   - RacePosition, QualPosition, is_podium (Top-3), is_top10 (Points)
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np

# Ensure 05_Analytics is in path to import circuit_clusters
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
ANALYTICS_DIR = os.path.join(BASE_DIR, "05_Analytics")
if ANALYTICS_DIR not in sys.path:
    sys.path.insert(0, ANALYTICS_DIR)

from circuit_clusters import get_circuit_features, CIRCUIT_TAXONOMY


def load_raw_data(db_path: str) -> pd.DataFrame:
    """Load race and qualifying results merged by event and driver with telemetry proxies."""
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

    # Betterment 1: Micro-Sector Telemetry & Progression metrics
    q_df['q3_executed'] = q_df['Q3_s'].notnull().astype(int)
    # Session progression: lap time improvement from Q1 to best downstream session (in percent)
    q_prog = ((q_df['Q1_s'] - q_df['best_q_s']) / q_df['Q1_s'] * 100).clip(lower=0.0)
    q_df['q_session_progression'] = q_prog.fillna(0.0)

    # Pole time per session
    pole_times = q_df.groupby(['year', 'round'])['best_q_s'].min().reset_index().rename(columns={'best_q_s': 'pole_time_s'})
    q_df = pd.merge(q_df, pole_times, on=['year', 'round'], how='left')
    q_df['pole_delta_s'] = (q_df['best_q_s'] - q_df['pole_time_s']).fillna(2.5)

    # Relative pace efficiency (1.0 = pole, >1.0 = slower)
    q_df['q_pace_efficiency'] = (q_df['best_q_s'] / q_df['pole_time_s']).fillna(1.035)

    # Teammate best time per session
    team_best = q_df.groupby(['year', 'round', 'TeamName'])['best_q_s'].min().reset_index().rename(columns={'best_q_s': 'team_best_q_s'})
    q_df = pd.merge(q_df, team_best, on=['year', 'round', 'TeamName'], how='left')
    q_df['teammate_delta_s'] = (q_df['best_q_s'] - q_df['team_best_q_s']).fillna(0.5)

    # Merge qualifying and race results cleanly with coalesced event_name and TeamName
    df = pd.merge(
        q_df,
        r_df[['year', 'round', 'event_name', 'DriverId', 'TeamName', 'RacePosition', 'GridPosition', 'Points', 'Laps', 'finished']],
        on=['year', 'round', 'DriverId'],
        how='outer',
        suffixes=('_q', '_r')
    )

    df['event_name'] = df['event_name_q'].fillna(df['event_name_r']).fillna("Unknown Grand Prix")
    df['TeamName'] = df['TeamName_q'].fillna(df['TeamName_r']).fillna("Unknown Team")

    # Fill GridPosition with QualPosition if GridPosition is missing or None
    df['GridPosition'] = df['GridPosition'].fillna(df['QualPosition']).fillna(15.0)

    # Sort strictly chronologically
    df = df.sort_values(['year', 'round', 'QualPosition', 'RacePosition']).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling historical features strictly prior to the current event round
    to prevent look-ahead data leakage, and enrich with physical circuit taxonomy.
    """
    events = df[['year', 'round']].drop_duplicates().sort_values(['year', 'round']).reset_index(drop=True)
    events['event_idx'] = np.arange(len(events))
    df = pd.merge(df, events, on=['year', 'round'], how='left')

    features = []

    for idx, row in df.iterrows():
        cur_event_idx = row['event_idx']
        cur_driver = row['DriverId']
        cur_team = row['TeamName']
        cur_event = str(row['event_name'])

        # Historical data strictly prior to this event (Zero temporal look-ahead leakage)
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

        # 3. Driver momentum: outperforming career average recently
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

        # -------------------------------------------------------------
        # Betterment 2: Circuit Taxonomy & Downforce Physics
        # -------------------------------------------------------------
        c_feat = get_circuit_features(cur_event)
        circuit_cluster_id = int(c_feat['cluster_id'])
        tire_stress_index = float(c_feat['tire_stress_index'])
        full_throttle_pct = float(c_feat['full_throttle_pct'])
        v_min_benchmark_kmh = float(c_feat['v_min_benchmark_kmh'])
        overtaking_diff_idx = float(c_feat.get('overtaking_difficulty_idx', 2.0))

        # -------------------------------------------------------------
        # Betterment 1: Micro-Sector Cornering Dynamics & Interaction Terms
        # -------------------------------------------------------------
        pole_delta = round(float(row['pole_delta_s']), 3) if pd.notnull(row['pole_delta_s']) else 2.5
        grid_pos = float(row['GridPosition']) if pd.notnull(row['GridPosition']) else 10.0
        apex_speed_eff = round(v_min_benchmark_kmh / (1.0 + pole_delta), 2)
        overtaking_sensitivity = round(grid_pos * overtaking_diff_idx, 2)
        tire_deg_stress_impact = round(tire_stress_index * ((20.0 - driver_race_mean_last3) / 20.0), 2)
        q_session_progression = round(float(row['q_session_progression']), 3) if pd.notnull(row.get('q_session_progression')) else 0.0
        q_pace_efficiency = round(float(row['q_pace_efficiency']), 4) if pd.notnull(row.get('q_pace_efficiency')) else 1.035
        q3_exec = int(row['q3_executed']) if pd.notnull(row.get('q3_executed')) else 0

        # Binary targets
        is_podium = 1 if (row['finished'] == 1 and row['RacePosition'] <= 3.0) else 0
        is_top10 = 1 if (row['finished'] == 1 and row['RacePosition'] <= 10.0) else 0

        feat_dict = {
            'year': row['year'],
            'round': row['round'],
            'event_idx': cur_event_idx,
            'event_name': cur_event,
            'DriverId': cur_driver,
            'TeamName': cur_team,
            'GridPosition': grid_pos,
            'QualPosition': row['QualPosition'],
            'pole_delta_s': pole_delta,
            'teammate_delta_s': round(float(row['teammate_delta_s']), 3) if pd.notnull(row['teammate_delta_s']) else 0.5,
            # Micro-Sector & Lap Execution Features (Betterment 1)
            'q_pace_efficiency': q_pace_efficiency,
            'q_session_progression': q_session_progression,
            'q3_executed': q3_exec,
            'apex_speed_eff': apex_speed_eff,
            # Circuit Physics & Aerodynamic Clustering (Betterment 2)
            'circuit_cluster_id': circuit_cluster_id,
            'tire_stress_index': tire_stress_index,
            'full_throttle_pct': full_throttle_pct,
            'v_min_benchmark_kmh': v_min_benchmark_kmh,
            'overtaking_difficulty_idx': overtaking_diff_idx,
            'overtaking_sensitivity': overtaking_sensitivity,
            'tire_deg_stress_impact': tire_deg_stress_impact,
            # Driver Rolling Metrics
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
            # Team Rolling Metrics
            'team_qual_mean': round(team_qual_mean, 2),
            'team_qual_mean_last3': round(team_qual_mean_last3, 2),
            'team_race_mean': round(team_race_mean, 2),
            'team_points_ytd': round(team_points_ytd, 1),
            # Target labels
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
