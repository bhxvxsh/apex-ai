# APEX AI: Dual-Track End-to-End (E2E) Test Infrastructure Specification

**Document Version**: 1.0.0  
**Target Project**: APEX AI IEEE Research Paper & Visual Assets (`06_Research_Paper/`)  
**Parent Working Directory**: `d:\yeah def my nyc project\APEX_AI`  
**Test Suite Path**: `tests/test_e2e_research_paper.py` & `tests/run_e2e_tests.py`  
**Publication Status**: Published & Ready (`TEST_READY.md`)  

---

## 1. Executive Summary & Test Philosophy

### 1.1 Opaque-Box & Requirement-Driven Architecture
The APEX AI End-to-End (E2E) test harness operates strictly on an **opaque-box, requirement-driven verification methodology**. The test suite treats the paper compilation pipeline, asset generation scripts, LaTeX manuscript, Markdown replica, and empirical validation artifacts as black-box deliverables governed entirely by the formal specifications in `ORIGINAL_REQUEST.md` and `PROJECT.md`.

Key tenets:
1. **Zero Coupling to Internal Implementation Details**: Tests do not rely on transient script internals, private variable names, or ad-hoc plotting subroutines. Instead, they validate external deliverables, file contracts, exact output formats, metadata, and numerical consistency.
2. **Deterministic Authoritative Oracles**: Every test assertion is derived directly from authoritative physical files in the project workspace:
   - Primary database: `02_Data_Processed/f1.db` (64 Grand Prix events, 1,119 finishers).
   - Walk-forward temporal cross-validation metrics: `04_Models/walk_forward_metrics.json`.
   - Held-out evaluation & classification summary: `04_Models/baseline_metrics_summary.json`.
   - Domain feature definitions: `03_Training_Dataset/circuit_taxonomy.py`, `03_Training_Dataset/micro_sector_dynamics.py`.
   - QLoRA fine-tuning checkpoints: `07_Notebooks/apex_ai_master_colab.ipynb`.
3. **Dual Execution Environment**: The test suite is dual-compatible:
   - Automated test runners: Standard `pytest` via `python -m pytest tests/test_e2e_research_paper.py -v`.
   - Zero-dependency standalone runner: Built-in `python tests/run_e2e_tests.py` or `python tests/test_e2e_research_paper.py` using Python's standard library `unittest`, supporting headless CI/CD and air-gapped environments without external package installations.

---

## 2. Feature Inventory Mapping to Test Tiers

The 14 canonical project features defined in `PROJECT.md` map to the 4-Tier Test Architecture:

| Feature ID | Feature Description | Target Scope | Assigned Test Tier | Test Method / Verification Anchor |
|:---:|---|---|:---:|---|
| **F1** | IEEE Paper Structure & Preamble | `IEEEtran.cls`, `IEEEtran.bst`, `main.tex` | **Tier 1 & Tier 3** | File existence, IEEE standard class verification, 8-section layout audit |
| **F2** | Fig. 1: System Architecture Diagram | `fig1_system_architecture.py`, `.pdf`, `.png` | **Tier 1 & Tier 2** | Script compilation, execution exit code 0, 300+ DPI PNG, valid `%PDF-` header |
| **F3** | Fig. 2: Circuit Taxonomy & Radar | `fig2_circuit_taxonomy.py`, `.pdf`, `.png` | **Tier 1 & Tier 2** | Script compilation, execution exit code 0, 300+ DPI PNG, valid `%PDF-` header |
| **F4** | Fig. 3: Micro-Sector Dynamics Telemetry | `fig3_micro_sector_dynamics.py`, `.pdf`, `.png` | **Tier 1 & Tier 2** | Script compilation, execution exit code 0, 300+ DPI PNG, valid `%PDF-` header |
| **F5** | Fig. 4: Non-Linear Tire Degradation | `fig4_tire_degradation.py`, `.pdf`, `.png` | **Tier 1 & Tier 2** | Script compilation, execution exit code 0, 300+ DPI PNG, valid `%PDF-` header |
| **F6** | Fig. 5: Walk-Forward Scheme | `fig5_walk_forward_validation.py`, `.pdf`, `.png`| **Tier 1 & Tier 2** | Script compilation, execution exit code 0, 300+ DPI PNG, valid `%PDF-` header |
| **F7** | Fig. 6: Model Performance & ROC-AUC | `fig6_model_performance.py`, `.pdf`, `.png` | **Tier 1 & Tier 2** | Script compilation, execution exit code 0, 300+ DPI PNG, valid `%PDF-` header |
| **F8** | Fig. 7: Feature Importance & Ablation | `fig7_feature_ablation.py`, `.pdf`, `.png` | **Tier 1 & Tier 2** | Script compilation, execution exit code 0, 300+ DPI PNG, valid `%PDF-` header |
| **F9** | BibTeX Bibliography (`references.bib`) | `references.bib` (28 citations) | **Tier 1 & Tier 3** | File existence, non-emptiness, citation key parsing, citation count >= 25 |
| **F10** | Mathematical Formulation & Hypotheses | Sections IV, V ($H_1, H_2, H_3$) | **Tier 4** | Exact formulation verification, null/alternative statements, test statistics |
| **F11** | Empirical Tables & Metrics | Sections V, VI, VII | **Tier 3 & Tier 4** | Markdown vs LaTeX table alignment, exact cross-check against JSON/DB |
| **F12** | Complete LaTeX Manuscript (`main.tex`) | `06_Research_Paper/main.tex` | **Tier 1, 3, 4** | File existence, section structure, citation resolution, zero placeholders |
| **F13** | Markdown Manuscript (`PAPER.md`) | `06_Research_Paper/PAPER.md` | **Tier 1, 3, 4** | File existence, section structure, table consistency, zero placeholders |
| **F14** | Automated E2E Verification Suite | `tests/test_e2e_research_paper.py` | **Meta / All** | Autonomous continuous validation, clear error telemetry, pass/fail reporting |

---

## 3. 4-Tier Test Architecture Breakdown

### 3.1 Tier 1: Target Deliverable & Feature Coverage
Validates the physical presence, correct workspace topology, and non-emptiness of all publication deliverables:
- **Core LaTeX & Bibliography Files**:
  - `06_Research_Paper/IEEEtran.cls` (official IEEE Transactions document class v1.8b).
  - `06_Research_Paper/IEEEtran.bst` (official IEEE Transactions BibTeX style).
  - `06_Research_Paper/references.bib` (peer-reviewed bibliography).
  - `06_Research_Paper/main.tex` (complete LaTeX research paper).
  - `06_Research_Paper/PAPER.md` (complete Markdown research paper).
- **Standalone Reproducible Figure Scripts (`06_Research_Paper/scripts/`)**:
  - `generate_all_figures.py` (master batch runner).
  - `fig1_system_architecture.py`
  - `fig2_circuit_taxonomy.py`
  - `fig3_micro_sector_dynamics.py`
  - `fig4_tire_degradation.py`
  - `fig5_walk_forward_validation.py`
  - `fig6_model_performance.py`
  - `fig7_feature_ablation.py`
- **Publication-Ready Figure Outputs (`06_Research_Paper/figures/`)**:
  - 7 Vector PDF figures (`fig1_system_architecture.pdf` through `fig7_feature_importance_ablation.pdf`).
  - 7 High-Resolution Raster PNG figures (`fig1_system_architecture.png` through `fig7_feature_importance_ablation.png`).
  - All files must satisfy `size > 0` bytes and minimum content thresholds.

### 3.2 Tier 2: Boundary, Format, & Resolution Validation
Enforces IEEE publication standards on file formats and visual quality:
- **Raster Image Resolution & Metadata**:
  - Valid PNG signature (`\x89PNG\r\n\x1a\n`).
  - Resolution audit: image density must be $\ge 300\text{ DPI}$ along both horizontal and vertical axes (via `PIL.Image` info or `pHYs` chunk).
  - Dimension constraints: width $\ge 1200\text{ px}$, height $\ge 600\text{ px}$ to prevent degraded unreadable figures.
- **Vector Graphics Format Integrity**:
  - Valid PDF magic bytes (`%PDF-` in header).
  - Non-empty byte streams ($> 1,024$ bytes) ensuring plots are properly flushed to disk.
- **Script Executability & Reproducibility**:
  - Python abstract syntax tree (AST) compilation (`py_compile.compile`) across all 8 generation scripts.
  - Subprocess execution verification: executing scripts in standalone mode must exit with `returncode == 0`.

### 3.3 Tier 3: Cross-Feature Alignment & Internal Consistency
Ensures mathematical, visual, and bibliographic links across files:
- **Figure Reference Integrity**:
  - Parses every `\includegraphics{...}` in `main.tex` and validates that the referenced figure file exists in `06_Research_Paper/figures/`.
  - Parses every `![...](...)` in `PAPER.md` and validates that the referenced figure file exists.
  - Validates that LaTeX labels (`\label{fig:...}`) match corresponding figure definitions.
- **Citation Key Resolution**:
  - Parses all citation keys defined in `06_Research_Paper/references.bib` (`@article{key,`, `@inproceedings{key,`, etc.).
  - Extracts all `\cite{key1, key2}` citations from `main.tex`.
  - Asserts that **100% of cited keys resolve directly** to a BibTeX entry in `references.bib` (0 broken references).
  - Asserts that `references.bib` contains at least 25 peer-reviewed citations.
- **Structural Symmetry (LaTeX $\leftrightarrow$ Markdown)**:
  - Validates that `main.tex` and `PAPER.md` share identical 8-section IEEE topology:
    - Section I: Introduction
    - Section II: Related Work
    - Section III: System Architecture & Methodology
    - Section IV: Mathematical Formulation & Empirical Hypotheses
    - Section V: Experimental Setup & Walk-Forward Validation
    - Section VI: Results & Quantitative Analysis
    - Section VII: Discussion & Ablation Studies
    - Section VIII: Conclusion & Future Work
- **Table Data Synchronization**:
  - Compares numerical table cells between `main.tex` (`\begin{tabular}`) and `PAPER.md` (Markdown pipe tables) to guarantee zero discrepancies between formats.

### 3.4 Tier 4: Empirical Ground-Truth Validation & Hypotheses Verification
Strict cross-verification against authoritative ground-truth datasets and models in the project repository:
- **Empirical Ground-Truth Ingestion (`02_Data_Processed/f1.db`)**:
  - Total Grand Prix events = 64 across seasons 2023–2025.
  - Total race classification entries = 1,278 rows.
  - Total classified finishers (`finished = 1`) = 1,119.
- **Walk-Forward Cross-Validation Integrity (`04_Models/walk_forward_metrics.json`)**:
  - Exactly **49 walk-forward GP folds**.
  - Exactly **857 out-of-sample evaluated car finishes** ($1,119 - 262\text{ warmup} = 857$).
  - Warmup window = **15 initial Grand Prix events** (262 finishes).
  - APEX AI Mean Absolute Error (MAE) = **2.191 positions** (RMSE = 3.109).
  - Status-Quo Grid Baseline MAE = **2.711 positions** (RMSE = 3.874).
  - Relative Error Reduction = **19.16%** ($\frac{2.711 - 2.191}{2.711} \times 100$).
  - Paired Student's t-test statistic $t = 6.2559$, two-tailed $p = 6.236 \times 10^{-10}$ ($p = 6.24 \times 10^{-10}$).
  - Wilcoxon signed-rank statistic $W = 75,599.0$, $p = 1.924 \times 10^{-9}$.
  - Podium Accuracy = **90.43%**, ROC-AUC = **94.43%**.
  - Points Accuracy = **75.22%**, ROC-AUC = **81.78%**.
- **Circuit Aerodynamic Cluster MAEs**:
  - Low-Downforce Power (Cluster 1): **1.942 positions**.
  - Maximum-Downforce Mechanical Grip (Cluster 3): **2.090 positions**.
  - High-Downforce Thermal Degradation (Cluster 2): **2.130 positions**.
  - Hybrid Street / Stop-and-Go (Cluster 4): **2.402 positions**.
- **QLoRA Fine-Tuning Parameter Ground Truth (`07_Notebooks/`)**:
  - Base architecture: `unsloth/llama-3-8b-Instruct-bnb-4bit` (8.07B parameters).
  - LoRA rank $r = 16$, alpha $\alpha = 16$, dropout $= 0.0$.
  - Target projection layers: 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
  - Trainable parameters = **41,943,040** (~**41.94 Million**).
  - Trainable ratio = **0.5196%** (~**0.52%**).
  - Training schedule = 60 steps, learning rate = $2 \times 10^{-4}$, optimizer = `adamw_8bit`.
- **Formal Hypotheses Verification ($H_1, H_2, H_3$)**:
  - $H_1$: Telemetry-enriched machine learning significantly reduces position error relative to the status-quo grid baseline ($t = 6.2559, p = 6.24 \times 10^{-10}$).
  - $H_2$: Micro-sector telemetry and downforce clustering provide statistically decisive gains across aerodynamic regimes (Cluster 1 MAE 1.942 vs Cluster 4 MAE 2.402).
  - $H_3$: Autonomous ReAct multi-step reasoning calibration and podium prediction reliability surpass static heuristics ($z = 9.07, p < 10^{-19}$).
- **Strict Zero-Placeholder Audit**:
  - Complete elimination of incomplete draft markers: regex scanning enforces zero occurrences of `"TODO"`, `"TBD"`, `"lorem ipsum"`, `"FIXME"`, `"XXX"`, or unpopulated bracket templates `"[...]"` across both `main.tex` and `PAPER.md`.

---

## 4. Test Execution Guide

### 4.1 Pytest Command
Run from project root (`d:\yeah def my nyc project\APEX_AI` or `d:\yeah def my nyc project`):
```bash
python -m pytest tests/test_e2e_research_paper.py -v
```

### 4.2 Standalone Python Runner (Standard Library)
Run directly without external test dependencies:
```bash
python tests/run_e2e_tests.py
```
Or directly:
```bash
python tests/test_e2e_research_paper.py
```

### 4.3 Selective Tier Execution
Target individual test tiers:
```bash
# Tier 1 only (Deliverable coverage)
python -m pytest tests/test_e2e_research_paper.py -k "TestTier1" -v

# Tier 2 only (Resolution & boundary)
python -m pytest tests/test_e2e_research_paper.py -k "TestTier2" -v

# Tier 3 only (Cross-feature consistency & citations)
python -m pytest tests/test_e2e_research_paper.py -k "TestTier3" -v

# Tier 4 only (Empirical data & hypothesis ground truth)
python -m pytest tests/test_e2e_research_paper.py -k "TestTier4" -v
```

---

## 5. Maintenance & Escalation Protocol
- **Test Integrity**: Test logic derives strictly from `ORIGINAL_REQUEST.md` and repository ground-truth data. Tests must never be loosened or altered to fit an incomplete or erroneous implementation.
- **Implementation Bug Escalation**: If tests fail due to discrepancies in `main.tex`, missing figures, low image resolution, or corrupted ground-truth values, the failure is reported immediately to the implementing agents (`teamwork_coder` / orchestrator) with exact line citations and expected values.
