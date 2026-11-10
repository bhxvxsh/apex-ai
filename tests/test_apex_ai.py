#!/usr/bin/env python3
"""
tests/test_apex_ai.py
=====================
Core Verification Test Suite for the APEX AI Predictive Telemetry Architecture.

Validates:
  - Feature matrix schema and domain invariants (23-feature dataset)
  - Supervised machine learning metrics and expanding walk-forward results
  - Baseline metrics summary across race finishing and classification
  - Alpaca instruction dataset format
  - CLI script syntax and runnable modules
  - Core analytics modules compilation
"""

import os
import json
import py_compile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestApexAICore(unittest.TestCase):
    """Core verification tests for the APEX AI codebase."""

    def test_01_feature_matrix_integrity(self):
        """Verifies full_features.csv exists, is non-empty, and has expected columns."""
        feat_path = ROOT_DIR / "03_Training_Dataset" / "full_features.csv"
        self.assertTrue(feat_path.is_file(), f"full_features.csv missing at {feat_path}")
        self.assertGreater(feat_path.stat().st_size, 50000, "full_features.csv file too small")

        import pandas as pd
        df = pd.read_csv(feat_path)
        self.assertGreater(len(df), 500, f"Expected >500 rows in feature matrix, found {len(df)}")
        self.assertIn("GridPosition", df.columns, "GridPosition missing from feature columns")
        self.assertIn("RacePosition", df.columns, "RacePosition missing from feature columns")
        self.assertIn("circuit_cluster_id", df.columns, "circuit_cluster_id missing")

    def test_02_walk_forward_metrics_validity(self):
        """Verifies walk_forward_metrics.json contains valid 49-fold statistics."""
        wf_path = ROOT_DIR / "04_Models" / "walk_forward_metrics.json"
        self.assertTrue(wf_path.is_file(), f"walk_forward_metrics.json missing at {wf_path}")

        with open(wf_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["total_walk_forward_folds"], 49)
        self.assertEqual(data["total_samples_evaluated"], 857)
        self.assertAlmostEqual(data["mean_mae_grid_baseline"], 2.711, delta=0.01)
        self.assertAlmostEqual(data["mean_mae_xgb_ranked"], 2.191, delta=0.01)
        self.assertAlmostEqual(data["error_reduction_percentage"], 19.16, delta=0.05)
        self.assertIn("statistical_significance", data)

    def test_03_baseline_metrics_summary(self):
        """Verifies baseline_metrics_summary.json exists and holds valid evaluation scores."""
        bm_path = ROOT_DIR / "04_Models" / "baseline_metrics_summary.json"
        self.assertTrue(bm_path.is_file(), f"baseline_metrics_summary.json missing at {bm_path}")

        with open(bm_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("race_finishing_prediction_metrics", data)
        self.assertIn("classification_metrics", data)
        self.assertGreater(len(data["race_finishing_prediction_metrics"]), 3)

    def test_04_alpaca_instruction_dataset(self):
        """Verifies f1_technical_alpaca.json contains valid instruction-input-output format."""
        alpaca_path = ROOT_DIR / "03_Training_Dataset" / "f1_technical_alpaca.json"
        self.assertTrue(alpaca_path.is_file(), f"f1_technical_alpaca.json missing at {alpaca_path}")

        with open(alpaca_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.assertIsInstance(records, list)
        self.assertGreater(len(records), 0)
        sample = records[0]
        self.assertIn("instruction", sample)
        self.assertIn("output", sample)

    def test_05_cli_scripts_syntax(self):
        """Verifies apex_cli.py and ask_apex_cli.py compile without syntax errors."""
        for cli_name in ["apex_cli.py", "ask_apex_cli.py"]:
            cli_path = ROOT_DIR / cli_name
            self.assertTrue(cli_path.is_file(), f"{cli_name} missing at {cli_path}")
            py_compile.compile(str(cli_path), doraise=True)

    def test_06_analytics_modules_syntax(self):
        """Verifies key analytics modules compile without syntax errors."""
        analytics_dir = ROOT_DIR / "05_Analytics"
        for mod_name in ["react_agent.py", "circuit_clusters.py", "tools.py", "dynamic_analyst.py"]:
            mod_path = analytics_dir / mod_name
            self.assertTrue(mod_path.is_file(), f"{mod_name} missing at {mod_path}")
            py_compile.compile(str(mod_path), doraise=True)

    def test_07_requirements_file_exists(self):
        """Verifies root requirements.txt is present and specifies core dependencies."""
        req_path = ROOT_DIR / "requirements.txt"
        self.assertTrue(req_path.is_file(), f"requirements.txt missing at {req_path}")
        content = req_path.read_text(encoding="utf-8")
        self.assertIn("fastf1", content)
        self.assertIn("xgboost", content)
        self.assertIn("pandas", content)

    def test_08_license_file_exists(self):
        """Verifies MIT LICENSE file is present at repository root."""
        lic_path = ROOT_DIR / "LICENSE"
        self.assertTrue(lic_path.is_file(), f"LICENSE missing at {lic_path}")
        content = lic_path.read_text(encoding="utf-8")
        self.assertIn("MIT License", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
