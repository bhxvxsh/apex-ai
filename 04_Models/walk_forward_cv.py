"""
APEX AI - Time-Series Expanding Walk-Forward Cross-Validation Engine (Betterment 3)

Implements rigorous, leak-free expanding window walk-forward validation across all
chronologically ordered Formula 1 Grand Prix events (2023–2025):
- Preserves absolute temporal causality: Train on events 0..t-1, evaluate on event t.
- Folds run across 54+ out-of-sample Grand Prix events.
- Evaluates Status-Quo Grid Baseline, Ridge, Random Forest, XGBoost, and Intra-Race Ranked XGBoost.
- Calculates statistical significance tests (Paired Student's t-test and Wilcoxon signed-rank test)
  comparing the APEX AI pipeline against the Formula 1 status-quo grid baseline.
- Generates publication-grade convergence figures and JSON metrics summary.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
ANALYTICS_DIR = os.path.join(BASE_DIR, "05_Analytics")
if ANALYTICS_DIR not in sys.path:
    sys.path.insert(0, ANALYTICS_DIR)

from circuit_clusters import get_circuit_features

# Feature schema from train_baseline
RACE_FEATURES = [
    'GridPosition',
    'pole_delta_s',
    'teammate_delta_s',
    'q_pace_efficiency',
    'q_session_progression',
    'q3_executed',
    'apex_speed_eff',
    'circuit_cluster_id',
    'tire_stress_index',
    'full_throttle_pct',
    'v_min_benchmark_kmh',
    'overtaking_difficulty_idx',
    'overtaking_sensitivity',
    'tire_deg_stress_impact',
    'driver_race_mean_all',
    'driver_race_mean_last3',
    'driver_momentum',
    'driver_avg_gain',
    'driver_finish_rate',
    'driver_podium_rate',
    'driver_top10_rate',
    'team_race_mean',
    'team_points_ytd',
]

CLF_FEATURES = [
    'GridPosition',
    'pole_delta_s',
    'teammate_delta_s',
    'q_pace_efficiency',
    'q_session_progression',
    'q3_executed',
    'apex_speed_eff',
    'circuit_cluster_id',
    'tire_stress_index',
    'full_throttle_pct',
    'overtaking_sensitivity',
    'tire_deg_stress_impact',
    'driver_race_mean_all',
    'driver_race_mean_last3',
    'driver_momentum',
    'driver_podium_rate',
    'driver_top10_rate',
    'team_race_mean',
    'team_points_ytd',
]


def run_walk_forward_cv(dataset_path: str = None, out_dir: str = None, warmup_events: int = 15):
    """
    Execute expanding-window walk-forward cross validation.
    """
    if dataset_path is None:
        dataset_path = os.path.join(BASE_DIR, "03_Training_Dataset", "full_features.csv")
    if out_dir is None:
        out_dir = CURRENT_DIR

    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    df = pd.read_csv(dataset_path)
    df = df[df['finished'] == 1].copy()

    # Identify distinct chronological events
    events_df = df[['year', 'round', 'event_name']].drop_duplicates().sort_values(['year', 'round']).reset_index(drop=True)
    events_df['event_order'] = np.arange(len(events_df))
    df = pd.merge(df, events_df[['year', 'round', 'event_order']], on=['year', 'round'], how='left')

    total_events = len(events_df)
    print(f"[Walk-Forward CV] Total Chronological Events: {total_events}")
    print(f"[Walk-Forward CV] Warm-Up Window: {warmup_events} events | Folds to evaluate: {total_events - warmup_events}")

    fold_results = []
    
    # Store predictions for paired statistical testing
    all_y_true = []
    all_y_grid = []
    all_y_xgb_rank = []

    for t in range(warmup_events, total_events):
        cur_event_info = events_df.iloc[t]
        year = int(cur_event_info['year'])
        rnd = int(cur_event_info['round'])
        event_name = str(cur_event_info['event_name'])

        train_data = df[df['event_order'] < t].dropna(subset=['RacePosition'] + RACE_FEATURES)
        test_data = df[df['event_order'] == t].dropna(subset=['RacePosition'] + RACE_FEATURES)

        if len(test_data) < 5:
            continue

        X_train, y_train = train_data[RACE_FEATURES], train_data['RacePosition']
        X_test, y_test = test_data[RACE_FEATURES], test_data['RacePosition']

        # 1. Status Quo Grid Baseline
        pred_grid = test_data['GridPosition'].values

        # 2. Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        pred_ridge = ridge.predict(X_test)

        # 3. Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)

        # 4. XGBoost Regressor
        model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        model_xgb.fit(X_train, y_train)
        pred_xgb_raw = model_xgb.predict(X_test)

        # 5. XGBoost + Intra-Race Ranking
        test_data_copy = test_data.copy()
        test_data_copy['pred_raw'] = pred_xgb_raw
        pred_xgb_ranked = test_data_copy['pred_raw'].rank(method='min').values

        # 6. Podium Classifier
        clf_pod = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42)
        clf_pod.fit(train_data[CLF_FEATURES], train_data['is_podium'])
        pred_pod = clf_pod.predict(test_data[CLF_FEATURES])
        pod_acc = float(accuracy_score(test_data['is_podium'], pred_pod) * 100)

        # Record errors
        mae_grid = float(mean_absolute_error(y_test, pred_grid))
        mae_ridge = float(mean_absolute_error(y_test, pred_ridge))
        mae_rf = float(mean_absolute_error(y_test, pred_rf))
        mae_xgb = float(mean_absolute_error(y_test, pred_xgb_raw))
        mae_xgb_ranked = float(mean_absolute_error(y_test, pred_xgb_ranked))

        acc1_xgb_ranked = float(np.mean(np.abs(y_test.values - pred_xgb_ranked) <= 1.0) * 100)
        acc2_xgb_ranked = float(np.mean(np.abs(y_test.values - pred_xgb_ranked) <= 2.0) * 100)
        acc3_xgb_ranked = float(np.mean(np.abs(y_test.values - pred_xgb_ranked) <= 3.0) * 100)

        # Append individual observations for global statistical tests
        all_y_true.extend(y_test.values.tolist())
        all_y_grid.extend(pred_grid.tolist())
        all_y_xgb_rank.extend(pred_xgb_ranked.tolist())

        c_feat = get_circuit_features(event_name)

        fold_results.append({
            'fold': t - warmup_events + 1,
            'event_order': t,
            'year': year,
            'round': rnd,
            'event_name': event_name,
            'circuit_cluster_id': c_feat['cluster_id'],
            'cluster_name': c_feat['cluster_name'],
            'n_finishers': len(test_data),
            'mae_grid': round(mae_grid, 3),
            'mae_ridge': round(mae_ridge, 3),
            'mae_rf': round(mae_rf, 3),
            'mae_xgb': round(mae_xgb, 3),
            'mae_xgb_ranked': round(mae_xgb_ranked, 3),
            'acc_within_1_pos': round(acc1_xgb_ranked, 1),
            'acc_within_2_pos': round(acc2_xgb_ranked, 1),
            'acc_within_3_pos': round(acc3_xgb_ranked, 1),
            'podium_accuracy': round(pod_acc, 1),
        })

    folds_df = pd.DataFrame(fold_results)

    # -------------------------------------------------------------
    # Statistical Rigor: Paired Hypothesis Tests
    # -------------------------------------------------------------
    grid_errors = np.abs(np.array(all_y_true) - np.array(all_y_grid))
    xgb_errors = np.abs(np.array(all_y_true) - np.array(all_y_xgb_rank))

    # Paired t-test
    t_stat, p_val_ttest = stats.ttest_rel(grid_errors, xgb_errors)
    # Wilcoxon signed-rank test (non-parametric)
    w_stat, p_val_wilcoxon = stats.wilcoxon(grid_errors, xgb_errors)

    mae_grid_mean = float(np.mean(grid_errors))
    mae_xgb_mean = float(np.mean(xgb_errors))
    error_reduction_pct = float(((mae_grid_mean - mae_xgb_mean) / mae_grid_mean) * 100)

    # Summary across all folds
    summary = {
        'total_walk_forward_folds': len(folds_df),
        'total_samples_evaluated': len(all_y_true),
        'warmup_events_count': warmup_events,
        'mean_mae_grid_baseline': round(mae_grid_mean, 3),
        'mean_mae_xgb_ranked': round(mae_xgb_mean, 3),
        'mean_mae_rf': round(float(folds_df['mae_rf'].mean()), 3),
        'mean_mae_ridge': round(float(folds_df['mae_ridge'].mean()), 3),
        'error_reduction_percentage': round(error_reduction_pct, 2),
        'acc_within_1_pos_mean': round(float(folds_df['acc_within_1_pos'].mean()), 2),
        'acc_within_2_pos_mean': round(float(folds_df['acc_within_2_pos'].mean()), 2),
        'acc_within_3_pos_mean': round(float(folds_df['acc_within_3_pos'].mean()), 2),
        'podium_accuracy_mean': round(float(folds_df['podium_accuracy'].mean()), 2),
        'statistical_significance': {
            'paired_t_statistic': round(float(t_stat), 4),
            'paired_t_p_value': float(p_val_ttest),
            'wilcoxon_w_statistic': round(float(w_stat), 4),
            'wilcoxon_p_value': float(p_val_wilcoxon),
            'statistically_significant_at_alpha_0_01': bool(p_val_ttest < 0.01 and p_val_wilcoxon < 0.01),
        },
        'cluster_performance_mae': folds_df.groupby('cluster_name')['mae_xgb_ranked'].mean().round(3).to_dict(),
        'folds_detail': fold_results,
    }

    # Save JSON summary
    out_json = os.path.join(out_dir, "walk_forward_metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------------------
    # Visualization 1: Cumulative Expanding Walk-Forward Convergence
    # -------------------------------------------------------------
    folds_df['cum_mae_grid'] = folds_df['mae_grid'].expanding().mean()
    folds_df['cum_mae_xgb'] = folds_df['mae_xgb_ranked'].expanding().mean()
    folds_df['cum_mae_rf'] = folds_df['mae_rf'].expanding().mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    x_axis = np.arange(1, len(folds_df) + 1)
    ax.plot(x_axis, folds_df['cum_mae_grid'], label='Status-Quo Grid Baseline', color='#e74c3c', linestyle='--', linewidth=2)
    ax.plot(x_axis, folds_df['cum_mae_rf'], label='Random Forest', color='#f39c12', linewidth=1.8, alpha=0.8)
    ax.plot(x_axis, folds_df['cum_mae_xgb'], label='XGBoost + Intra-Race Ranking (APEX AI)', color='#27ae60', linewidth=2.5)

    ax.set_title("Empirical Walk-Forward Validation: Cumulative MAE Convergence Across 54 F1 GPs", fontsize=12, fontweight='bold')
    ax.set_xlabel("Sequential Out-of-Sample Grand Prix Fold Number", fontsize=11)
    ax.set_ylabel("Cumulative Mean Absolute Error (Positions)", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    fig_path1 = os.path.join(out_dir, "figures", "walk_forward_mae_trend.png")
    fig.savefig(fig_path1, dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------
    # Visualization 2: Circuit Cluster Error Breakdown
    # -------------------------------------------------------------
    cluster_stats = folds_df.groupby('cluster_name')[['mae_grid', 'mae_xgb_ranked']].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cluster_stats))
    w = 0.35
    ax.bar(x - w/2, cluster_stats['mae_grid'], w, label='Grid Baseline', color='#d9534f')
    ax.bar(x + w/2, cluster_stats['mae_xgb_ranked'], w, label='APEX AI (Ranked XGBoost)', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_stats['cluster_name'], rotation=12, ha='right', fontsize=9.5)
    ax.set_ylabel("Mean Absolute Error (Lower is Better)")
    ax.set_title("Walk-Forward Error Margin by Aerodynamic Downforce Cluster")
    ax.legend()
    fig.tight_layout()
    fig_path2 = os.path.join(out_dir, "figures", "walk_forward_cluster_breakdown.png")
    fig.savefig(fig_path2, dpi=160)
    plt.close(fig)

    print("\n" + "=" * 80)
    print(" WALK-FORWARD CROSS-VALIDATION SUMMARY (BETTERMENT 3)")
    print("=" * 80)
    print(f"Evaluated Folds:             {len(folds_df)} Grand Prix events")
    print(f"Total Prediction Instances:  {len(all_y_true)} car finishes")
    print(f"Grid Baseline Mean MAE:      {mae_grid_mean:.3f} positions")
    print(f"APEX AI Ranked XGBoost MAE:  {mae_xgb_mean:.3f} positions")
    print(f"Relative Error Reduction:    {error_reduction_pct:.2f}%")
    print(f"Within ±1 Position Acc:      {summary['acc_within_1_pos_mean']}%")
    print(f"Within ±2 Position Acc:      {summary['acc_within_2_pos_mean']}%")
    print(f"Within ±3 Position Acc:      {summary['acc_within_3_pos_mean']}%")
    print(f"Average Podium Accuracy:     {summary['podium_accuracy_mean']}%")
    print(f"Paired t-statistic:          {t_stat:.4f} (p-value: {p_val_ttest:.2e})")
    print(f"Wilcoxon W-statistic:        {w_stat:.4f} (p-value: {p_val_wilcoxon:.2e})")
    print(f"Statistically Significant:   {summary['statistical_significance']['statistically_significant_at_alpha_0_01']}")
    print(f"Metrics saved to:            {out_json}")
    print(f"Figures saved to:            {fig_path1}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    run_walk_forward_cv()

