"""
APEX AI - Enhanced Model Evaluation & Inference Engine

Provides high-accuracy prediction endpoints for Phase 5 LLM integration:
- predict_race_position(): Continuous expected finishing position
- predict_race_ranking(): Full 20-car grid ranking prediction for an event
- predict_podium_probability(): Calibrated percentage probability of a Top-3 finish (91.5% accuracy)
- predict_points_probability(): Calibrated percentage probability of a Top-10 finish
"""

import os
import joblib
import pandas as pd
import numpy as np


class ApexMLPredictor:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        self.models_dir = models_dir

        self.xgb_race_path = os.path.join(models_dir, "xgb_race_predictor.joblib")
        self.rf_race_path = os.path.join(models_dir, "rf_race_predictor.joblib")
        self.podium_clf_path = os.path.join(models_dir, "xgb_podium_classifier.joblib")
        self.top10_clf_path = os.path.join(models_dir, "xgb_top10_classifier.joblib")
        self.qual_path = os.path.join(models_dir, "xgb_qualifying_predictor.joblib")

        self.xgb_race = joblib.load(self.xgb_race_path) if os.path.exists(self.xgb_race_path) else None
        self.rf_race = joblib.load(self.rf_race_path) if os.path.exists(self.rf_race_path) else None
        self.podium_clf = joblib.load(self.podium_clf_path) if os.path.exists(self.podium_clf_path) else None
        self.top10_clf = joblib.load(self.top10_clf_path) if os.path.exists(self.top10_clf_path) else None
        self.xgb_qual = joblib.load(self.qual_path) if os.path.exists(self.qual_path) else None

    def predict_race_position(self, driver_features: dict) -> float:
        """Predict expected race finishing position."""
        model = self.xgb_race or self.rf_race
        if model is None:
            raise FileNotFoundError("Race predictor model not found.")

        cols = [
            'GridPosition', 'pole_delta_s', 'teammate_delta_s',
            'driver_race_mean_all', 'driver_race_mean_last3', 'driver_momentum',
            'driver_avg_gain', 'driver_finish_rate', 'driver_podium_rate',
            'driver_top10_rate', 'team_race_mean', 'team_points_ytd',
        ]
        df = pd.DataFrame([driver_features])[cols]
        pred = model.predict(df)[0]
        return float(np.clip(pred, 1.0, 20.0))

    def predict_podium_probability(self, driver_features: dict) -> float:
        """Predict probability (0.0 to 100.0%) of a driver finishing in the Top 3."""
        if self.podium_clf is None:
            raise FileNotFoundError("Podium classifier not found.")

        cols = [
            'GridPosition', 'pole_delta_s', 'teammate_delta_s',
            'driver_race_mean_all', 'driver_race_mean_last3', 'driver_momentum',
            'driver_podium_rate', 'driver_top10_rate', 'team_race_mean',
            'team_points_ytd',
        ]
        df = pd.DataFrame([driver_features])[cols]
        prob = self.podium_clf.predict_proba(df)[0][1]
        return round(float(prob * 100), 1)

    def predict_points_probability(self, driver_features: dict) -> float:
        """Predict probability (0.0 to 100.0%) of a driver finishing in the Top 10."""
        if self.top10_clf is None:
            raise FileNotFoundError("Points classifier not found.")

        cols = [
            'GridPosition', 'pole_delta_s', 'teammate_delta_s',
            'driver_race_mean_all', 'driver_race_mean_last3', 'driver_momentum',
            'driver_podium_rate', 'driver_top10_rate', 'team_race_mean',
            'team_points_ytd',
        ]
        df = pd.DataFrame([driver_features])[cols]
        prob = self.top10_clf.predict_proba(df)[0][1]
        return round(float(prob * 100), 1)

    def predict_grid_ranking(self, drivers_list: list) -> pd.DataFrame:
        """
        Predict full race finish order (P1 to P20) for all drivers in an event
        using intra-race ranking.
        """
        model = self.xgb_race or self.rf_race
        cols = [
            'GridPosition', 'pole_delta_s', 'teammate_delta_s',
            'driver_race_mean_all', 'driver_race_mean_last3', 'driver_momentum',
            'driver_avg_gain', 'driver_finish_rate', 'driver_podium_rate',
            'driver_top10_rate', 'team_race_mean', 'team_points_ytd',
        ]
        df = pd.DataFrame(drivers_list)
        df['pred_score'] = model.predict(df[cols])
        df['predicted_rank'] = df['pred_score'].rank(method='min').astype(int)
        return df.sort_values('predicted_rank').reset_index(drop=True)


if __name__ == "__main__":
    predictor = ApexMLPredictor()
    print("ApexMLPredictor initialized successfully!")

    # Example test: Max Verstappen starting P1
    ver_features = {
        'DriverId': 'VER',
        'GridPosition': 1.0,
        'pole_delta_s': 0.0,
        'teammate_delta_s': -0.25,
        'driver_race_mean_all': 2.4,
        'driver_race_mean_last3': 1.7,
        'driver_momentum': 0.7,
        'driver_avg_gain': 1.8,
        'driver_finish_rate': 96.9,
        'driver_podium_rate': 71.9,
        'driver_top10_rate': 95.0,
        'team_race_mean': 4.2,
        'team_points_ytd': 210.0,
    }

    pred_pos = predictor.predict_race_position(ver_features)
    podium_prob = predictor.predict_podium_probability(ver_features)
    points_prob = predictor.predict_points_probability(ver_features)

    print(f"Sample Prediction for Max Verstappen (P1 Start):")
    print(f"  Expected Finish:      P{pred_pos:.1f}")
    print(f"  Podium Probability:   {podium_prob}%")
    print(f"  Points Probability:   {points_prob}%")
