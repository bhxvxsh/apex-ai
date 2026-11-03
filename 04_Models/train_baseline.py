"""
APEX AI - High-Accuracy Machine Learning Pipeline

Trains, benchmarks, and serializes state-of-the-art F1 prediction models:
1. Race Finishing Position Prediction (Continuous & Intra-Race Ranked)
2. High-Precision Podium Classification (Top-3 Finish)
3. Points Finish Classification (Top-10 Finish)
4. Qualifying Position Prediction (Continuous & Ranked)

Models evaluated:
- Status-Quo Grid Baseline
- Ridge Regularized Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- LightGBM Regressor
- XGBoost Regressor (with Intra-Race Ranking)
- XGBoost Podium & Points Classifiers
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Compute regression metrics and position threshold accuracies."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    acc_within_1 = float(np.mean(np.abs(y_true - y_pred) <= 1.0) * 100)
    acc_within_2 = float(np.mean(np.abs(y_true - y_pred) <= 2.0) * 100)
    acc_within_3 = float(np.mean(np.abs(y_true - y_pred) <= 3.0) * 100)

    top3_acc = float(np.mean((y_pred <= 3.5) == (y_true <= 3.0)) * 100)
    top10_acc = float(np.mean((y_pred <= 10.5) == (y_true <= 10.0)) * 100)

    return {
        'model': model_name,
        'mae': round(float(mae), 3),
        'rmse': round(float(rmse), 3),
        'r2': round(float(r2), 3),
        'acc_within_1_pos': round(acc_within_1, 1),
        'acc_within_2_pos': round(acc_within_2, 1),
        'acc_within_3_pos': round(acc_within_3, 1),
        'top3_accuracy': round(top3_acc, 1),
        'top10_accuracy': round(top10_acc, 1),
    }


def train_race_models(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str):
    """Train and evaluate models for Race Finishing Position Prediction."""
    print("\n" + "=" * 80)
    print(" TASK 1: RACE FINISHING POSITION PREDICTION (2025 HELD-OUT TEST EVALUATION)")
    print("=" * 80)

    features = [
        'GridPosition',
        'pole_delta_s',
        'teammate_delta_s',
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

    target = 'RacePosition'

    tr = train_df[train_df['finished'] == 1].dropna(subset=[target] + features).copy()
    te = test_df[test_df['finished'] == 1].dropna(subset=[target] + features).copy()

    X_train, y_train = tr[features], tr[target]
    X_test, y_test = te[features], te[target]

    models = {
        'Status-Quo Grid Baseline': None,
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
        'LightGBM Regressor': lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1),
        'XGBoost Regressor': xgb.XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42),
    }

    results = []
    trained_artifacts = {}

    for name, model in models.items():
        if model is None:
            y_pred = te['GridPosition'].values
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_artifacts[name] = model

        metrics = evaluate_regression(y_test.values, y_pred, name)
        results.append(metrics)

    # Intra-Race Ranking with XGBoost
    xgb_model = trained_artifacts['XGBoost Regressor']
    te_copy = te.copy()
    te_copy['pred_raw'] = xgb_model.predict(X_test)
    te_copy['pred_rank'] = te_copy.groupby(['year', 'round'])['pred_raw'].rank(method='min')

    ranked_metrics = evaluate_regression(y_test.values, te_copy['pred_rank'].values, 'XGBoost + Intra-Race Ranking')
    results.append(ranked_metrics)

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # Feature Importance for XGBoost
    importances = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=importances.values, y=importances.index, ax=ax, palette='mako')
    ax.set_title("Race Finish Prediction: Feature Importances (XGBoost)")
    ax.set_xlabel("Feature Importance Score")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "feature_importance_race.png"), dpi=150)
    plt.close(fig)

    # Accuracy Comparison Bar Chart
    comp_df = res_df[res_df['model'].isin(['Status-Quo Grid Baseline', 'Ridge Regression', 'XGBoost + Intra-Race Ranking'])].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(comp_df))
    width = 0.25
    ax.bar(x - width, comp_df['acc_within_1_pos'], width, label='Within ±1 Pos (%)', color='#2b5c8f')
    ax.bar(x, comp_df['acc_within_2_pos'], width, label='Within ±2 Pos (%)', color='#3fa34d')
    ax.bar(x + width, comp_df['acc_within_3_pos'], width, label='Within ±3 Pos (%)', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df['model'], rotation=10, ha='right')
    ax.set_ylabel("Accuracy Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Prediction Accuracy Comparison Across Error Tolerances")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "accuracy_thresholds_comparison.png"), dpi=150)
    plt.close(fig)

    # Save artifacts
    joblib.dump(xgb_model, os.path.join(out_dir, "saved_models", "xgb_race_predictor.joblib"))
    joblib.dump(trained_artifacts['Random Forest'], os.path.join(out_dir, "saved_models", "rf_race_predictor.joblib"))

    return results, xgb_model


def train_classification_models(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str):
    """Train specialized classifiers for Podium (Top-3) and Points (Top-10)."""
    print("\n" + "=" * 80)
    print(" TASK 2: HIGH-STAKES CLASSIFICATION (PODIUM & POINTS FINISHES)")
    print("=" * 80)

    features = [
        'GridPosition',
        'pole_delta_s',
        'teammate_delta_s',
        'driver_race_mean_all',
        'driver_race_mean_last3',
        'driver_momentum',
        'driver_podium_rate',
        'driver_top10_rate',
        'team_race_mean',
        'team_points_ytd',
    ]

    tr = train_df.dropna(subset=['is_podium', 'is_top10'] + features).copy()
    te = test_df.dropna(subset=['is_podium', 'is_top10'] + features).copy()

    X_train, X_test = tr[features], te[features]

    # 1. Podium Classifier
    clf_podium = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    clf_podium.fit(X_train, tr['is_podium'])
    p_pred = clf_podium.predict(X_test)
    p_prob = clf_podium.predict_proba(X_test)[:, 1]

    podium_metrics = {
        'task': 'Podium Finish (Top-3)',
        'accuracy': round(float(accuracy_score(te['is_podium'], p_pred) * 100), 2),
        'roc_auc': round(float(roc_auc_score(te['is_podium'], p_prob) * 100), 2),
        'precision': round(float(precision_score(te['is_podium'], p_pred) * 100), 2),
        'recall': round(float(recall_score(te['is_podium'], p_pred) * 100), 2),
    }

    # 2. Points Classifier
    clf_top10 = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    clf_top10.fit(X_train, tr['is_top10'])
    t_pred = clf_top10.predict(X_test)
    t_prob = clf_top10.predict_proba(X_test)[:, 1]

    top10_metrics = {
        'task': 'Points Finish (Top-10)',
        'accuracy': round(float(accuracy_score(te['is_top10'], t_pred) * 100), 2),
        'roc_auc': round(float(roc_auc_score(te['is_top10'], t_prob) * 100), 2),
        'precision': round(float(precision_score(te['is_top10'], t_pred) * 100), 2),
        'recall': round(float(recall_score(te['is_top10'], t_pred) * 100), 2),
    }

    clf_df = pd.DataFrame([podium_metrics, top10_metrics])
    print(clf_df.to_string(index=False))

    # Save artifacts
    joblib.dump(clf_podium, os.path.join(out_dir, "saved_models", "xgb_podium_classifier.joblib"))
    joblib.dump(clf_top10, os.path.join(out_dir, "saved_models", "xgb_top10_classifier.joblib"))

    return [podium_metrics, top10_metrics]


def train_qualifying_models(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str):
    """Train and evaluate models for Qualifying Position Prediction."""
    print("\n" + "=" * 80)
    print(" TASK 3: QUALIFYING POSITION PREDICTION")
    print("=" * 80)

    features = [
        'driver_qual_mean_all',
        'driver_qual_mean_last3',
        'driver_qual_best',
        'team_qual_mean',
        'team_qual_mean_last3',
        'team_points_ytd',
    ]

    target = 'QualPosition'

    tr = train_df.dropna(subset=[target] + features).copy()
    te = test_df.dropna(subset=[target] + features).copy()

    X_train, y_train = tr[features], tr[target]
    X_test, y_test = te[features], te[target]

    models = {
        'Naive Rolling Mean Baseline': None,
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42),
        'XGBoost Regressor': xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
    }

    results = []
    trained_artifacts = {}

    for name, model in models.items():
        if model is None:
            y_pred = te['driver_qual_mean_last3'].values
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_artifacts[name] = model

        metrics = evaluate_regression(y_test.values, y_pred, name)
        results.append(metrics)

    # Intra-Session Ranked XGBoost
    xgb_qual = trained_artifacts['XGBoost Regressor']
    te_copy = te.copy()
    te_copy['pred_raw'] = xgb_qual.predict(X_test)
    te_copy['pred_rank'] = te_copy.groupby(['year', 'round'])['pred_raw'].rank(method='min')

    ranked_metrics = evaluate_regression(y_test.values, te_copy['pred_rank'].values, 'XGBoost + Intra-Session Ranking')
    results.append(ranked_metrics)

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # Save artifacts
    joblib.dump(xgb_qual, os.path.join(out_dir, "saved_models", "xgb_qualifying_predictor.joblib"))
    joblib.dump(trained_artifacts['Random Forest'], os.path.join(out_dir, "saved_models", "rf_qualifying_predictor.joblib"))

    return results


def run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    train_path = os.path.join(base_dir, "03_Training_Dataset", "train_dataset.csv")
    test_path = os.path.join(base_dir, "03_Training_Dataset", "test_dataset.csv")
    out_dir = os.path.join(base_dir, "04_Models")

    os.makedirs(os.path.join(out_dir, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded datasets: Train={len(train_df)} rows, Test={len(test_df)} rows")

    race_results, _ = train_race_models(train_df, test_df, out_dir)
    clf_results = train_classification_models(train_df, test_df, out_dir)
    qual_results = train_qualifying_models(train_df, test_df, out_dir)

    summary = {
        'phase': 'Phase 4 - Advanced ML Pipeline',
        'train_seasons': [2023, 2024],
        'test_season': 2025,
        'race_finishing_prediction_metrics': race_results,
        'classification_metrics': clf_results,
        'qualifying_prediction_metrics': qual_results,
    }

    summary_file = os.path.join(out_dir, "baseline_metrics_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Advanced ML pipeline complete! Summary written to: {summary_file}")


if __name__ == "__main__":
    run_pipeline()
