# APEX AI: Reproducibility & Replication Protocol

This document provides a comprehensive, deterministic replication protocol for the empirical results, physical telemetry models, and publication figures presented in the **APEX AI** research manuscript.

---

## 1. System & Hardware Specifications

The experiments and benchmarks were executed and validated under the following configuration:
- **Operating System**: Windows 11 Pro 64-bit / Ubuntu 22.04 LTS / macOS 14+ (Cross-Platform Compliant)
- **Processor**: AMD Ryzen 9 5950X (16 cores, 32 threads, 3.4 GHz base) or Intel Core i7/i9 equivalent
- **System Memory**: 32 GB DDR4/DDR5 RAM (minimum 16 GB required for expanding walk-forward folds)
- **GPU Acceleration**: NVIDIA GeForce RTX 3080 (10 GB VRAM) / NVIDIA Tesla T4 (16 GB VRAM on Google Colab)
- **Python Version**: Python 3.12 or Python 3.14 (Verified with standard CPython distribution)
- **CUDA Runtime**: CUDA 12.1+ / cuDNN 8.9+ (optional for XGBoost GPU trees, CPU fallback fully supported)

---

## 2. Environment Initialization & Dependency Setup

Clone the repository and instantiate an isolated virtual environment:

```bash
# 1. Clone repository
git clone https://github.com/APEX-AI-Research/APEX_AI.git
cd APEX_AI

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r 00_Setup/requirements.txt
```

Core library dependencies:
- `numpy>=1.26.0`, `scipy>=1.12.0`, `pandas>=2.2.0`
- `scikit-learn>=1.4.0`, `xgboost>=2.0.0`, `lightgbm>=4.3.0`
- `matplotlib>=3.8.0`, `seaborn>=0.13.0`
- `fastf1>=3.4.0`
- `pytest>=8.0.0`

---

## 3. Data Provenance & Integrity Audit

APEX AI uses official timing feeds archived in SQLite database `02_Data_Processed/f1.db`. To verify that the database has not suffered corruption, execute the integrity audit:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('02_Data_Processed/f1.db')
c = conn.cursor()
starts = c.execute('SELECT COUNT(*) FROM race_results').fetchone()[0]
finishers = c.execute('SELECT COUNT(*) FROM race_results WHERE finished = 1').fetchone()[0]
gps = c.execute('SELECT COUNT(DISTINCT year || \"-\" || round) FROM race_results').fetchone()[0]
print(f'Starts: {starts} (Expected 1278), Finishers: {finishers} (Expected 1119), GPs: {gps} (Expected 64)')
assert (starts, finishers, gps) == (1278, 1119, 64), 'Database scale verification failed!'
print('Database integrity verified successfully.')
"
```

---

## 4. Step-by-Step Pipeline Execution

### Stage 1: Feature Engineering & Circuit Clustering
Construct the 23-feature training matrix combining longitudinal form, micro-sector telemetry dynamics, and aerodynamic taxonomy:
```bash
# Run circuit aerodynamic clustering
python 03_Training_Dataset/circuit_taxonomy.py

# Extract micro-sector dynamics and compile full features
python 03_Training_Dataset/micro_sector_dynamics.py
```
*Generated Artifact*: `03_Training_Dataset/full_features.csv` (1,278 rows, 40 total columns including 23 model features).

### Stage 2: Supervised Baseline Training & Evaluation
Train the baseline suite (Ridge, Random Forest, LightGBM, XGBoost) and specialized binary classifiers on the held-out 2025 season test split:
```bash
python 04_Models/train_baseline.py
python 04_Models/evaluate_models.py
```
*Generated Artifact*: `04_Models/baseline_metrics_summary.json`.

### Stage 3: Expanding-Window Walk-Forward Cross-Validation
Execute leak-free walk-forward validation across 49 sequential Grand Prix folds:
```bash
python 04_Models/walk_forward_cv.py
```
*Generated Artifact*: `04_Models/walk_forward_metrics.json`.
*Verification Assertion*: Status-Quo Grid MAE = 2.711, APEX AI Ranked XGBoost MAE = 2.191, Relative Gain = 19.16%, $p = 6.24 \times 10^{-10}$.

### Stage 4: Publication Figure Generation (One-Command)
Generate all 7 publication figures in both vector `.pdf` and raster 300+ DPI `.png` format:
```bash
python 06_Research_Paper/scripts/generate_all_figures.py
```
*Target Directory*: `06_Research_Paper/figures/` (Figures 1 through 7 audited and rendered cleanly without typography overlap).

### Stage 5: Comprehensive Test Verification
Execute the test suites to ensure 100% compliance across all data invariants, manuscript claims, and package deliverables:
```bash
# 1. Execute full 19-test IEEE research paper test suite
pytest tests/test_e2e_research_paper.py -v

# 2. Execute companion 8-test academic outreach test suite
pytest tests/test_academic_submission.py -v

# 3. Or run master unified test runner
python tests/run_e2e_tests.py --suite all
```

---

## 5. Expected Output Verification Ledger

| Benchmark / Metric | Empirical Ground Truth | Acceptable Tolerance | Primary Source Artifact |
|:---|:---:|:---:|:---|
| **Database Starts** | `1,278` | Exact integer match | `02_Data_Processed/f1.db` |
| **Classified Finishers** | `1,119` | Exact integer match | `02_Data_Processed/f1.db` |
| **Championship GPs** | `64` | Exact integer match | `02_Data_Processed/f1.db` |
| **Walk-Forward Folds** | `49` | Exact integer match | `04_Models/walk_forward_metrics.json` |
| **Status-Quo Grid MAE** | `2.711` | $\pm 0.005$ | `04_Models/walk_forward_metrics.json` |
| **APEX AI Ranked MAE** | `2.191` | $\pm 0.005$ | `04_Models/walk_forward_metrics.json` |
| **Relative Error Reduction** | `19.16%` | $\pm 0.10\%$ | `04_Models/walk_forward_metrics.json` |
| **Paired $t$-test $p$-value** | `6.24e-10` | $\pm 1.0\times 10^{-10}$ | `04_Models/walk_forward_metrics.json` |
| **Wilcoxon $p$-value** | `1.92e-09` | $\pm 0.5\times 10^{-9}$ | `04_Models/walk_forward_metrics.json` |
| **Podium ROC-AUC** | `94.43%` | $\pm 0.20\%$ | `04_Models/baseline_metrics_summary.json` |
| **Podium Accuracy** | `90.43%` | $\pm 0.20\%$ | `04_Models/baseline_metrics_summary.json` |
| **Points ROC-AUC** | `81.78%` | $\pm 0.20\%$ | `04_Models/baseline_metrics_summary.json` |
| **Points Accuracy** | `75.22%` | $\pm 0.20\%$ | `04_Models/baseline_metrics_summary.json` |
| **QLoRA Trainable Params** | `41,943,040` | Exact integer match | `07_Notebooks/apex_ai_master_colab.ipynb` |
| **Figure Raster Density** | $\ge 300\text{ DPI}$ | Minimum threshold | `06_Research_Paper/figures/*.png` |
| **Figure Vector Validity** | `%PDF-1.4` header | Strict format header | `06_Research_Paper/figures/*.pdf` |
| **E2E Test Suite Status** | 19 / 19 Passed | 0 failures, 0 errors | `tests/test_e2e_research_paper.py` |
| **Academic Suite Status** | 8 / 8 Passed | 0 failures, 0 errors | `tests/test_academic_submission.py` |

---

## 6. Contact & Support
For replication inquiries or telemetry data access questions, please open a GitHub Issue or reach out to the corresponding authors via `08_Academic_Submission/MENTOR_COVER_LETTER.md`.
