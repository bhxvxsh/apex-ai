# APEX AI: Executive Results & Empirical Benchmark Summary

**Study**: Multi-Modal Telemetry Ingestion, Physical Aerodynamic Clustering, and Walk-Forward Race Prediction  
**Evaluation**: 64 Grand Prix Events (2023–2025 FIA F1 World Championship), 1,119 Finishers, 49 Walk-Forward Folds ($N = 857$)  
**Core Finding**: APEX AI achieves **MAE 2.191**, a **19.16% relative error reduction** over Starting Grid Baseline ($p = 6.24 \times 10^{-10}$).

---

### Table 1: Comprehensive 7-Model Performance Benchmark

| # | Model Architecture | Paradigm / Key Hyperparameters | MAE (pos) | RMSE | Podium ROC-AUC | Points ROC-AUC | Acc ±1 | Acc ±2 | Rel. Gain vs Baseline |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Starting Grid Baseline** | Status-quo qualifying heuristic ($\hat{y} = P_{\text{grid}}$) | 2.711 | 3.874 | 93.96% | 87.53% | 42.12% | 58.11% | Baseline (0.0%) |
| 2 | **Historical Rolling Mean** | 3-event moving finishing average (`driver_race_mean_last3`) | 3.015 | 3.842 | 90.43% | 82.71% | 23.91% | 47.83% | -10.97% |
| 3 | **Ridge Regression** | L2-regularized linear model ($\alpha = 1.0$) | 2.228 | 3.145 | 94.21% | 88.11% | 28.14% | 57.29% | +17.82% |
| 4 | **Random Forest** | Bagged decision trees ($N = 150$, $\text{max\_depth} = 6$) | 2.288 | 3.190 | 94.70% | 88.32% | 31.04% | 56.48% | +15.60% |
| 5 | **LightGBM** | Leaf-wise GBDT ($\text{leaves} = 31$, $\eta = 0.05$) | 2.377 | 3.156 | 94.57% | 87.72% | 30.43% | 54.62% | +12.51% |
| 6 | **XGBoost** | Depth-wise GBDT ($\text{depth} = 3$, $\eta = 0.05$, $N = 120$) | 2.283 | 3.185 | 94.08% | 87.20% | 30.80% | 55.90% | +15.79% |
| 7 | **APEX AI Hybrid** | **Gradient Boosting + Intra-Race Ordinal Ranking** | **2.191** | **3.109** | **94.43%** | **81.78%** | **49.33%** | **67.21%** | **+19.16%** |

*Note*: Evaluated on 49 out-of-sample walk-forward folds ($N = 857$). Classification metrics derived from held-out 2025 test set and dedicated binary classifiers (Podium Top-3 Accuracy: **90.43%**, Points Top-10 Accuracy: **75.22%**).

---

### Table 2: Circuit Aerodynamic Cluster Breakdown (49 Folds)

| Cluster ID & Physical Regime | MAE (pos) | Folds | Representative Circuits | Dominant Physical Constraint |
|:---|:---:|:---:|:---|:---|
| **Cluster 1: Low-Downforce Power** | **1.942** | 11 | Monza, Spa, Las Vegas, Baku, Jeddah | High full throttle ($>71\%$), slipstream, top speeds $>340\text{ km/h}$ |
| **Cluster 3: Max-Downforce Mechanical Grip** | **2.090** | 9 | Monaco, Singapore, Hungaroring, Zandvoort | Low apex speed ($V_{\text{min}} \le 75\text{ km/h}$), severe track position premium |
| **Cluster 2: High-Downforce Thermal Deg** | **2.130** | 11 | Bahrain, Silverstone, Suzuka, Barcelona, Lusail | Severe lateral loads ($>4.5g$), rapid tire surface blistering |
| **Cluster 4: Hybrid Street / Stop-and-Go** | **2.402** | 18 | Miami, Montreal, Austin, Interlagos, Melbourne | Heavy braking, curb traction, high safety car probability |

---

### Statistical Hypothesis Testing & Component Ablation

- **Hypothesis 1 (Telemetry Dynamics vs Grid Baseline)**:
  - Paired Student's $t$-test: $t = 6.2559$, **$p = 6.24 \times 10^{-10}$** (Significant at $\alpha = 0.001$)
  - Wilcoxon signed-rank test: $W = 75,599.0$, **$p = 1.924 \times 10^{-9}$**
- **Hypothesis 2 (Physics & Micro-Sector Feature Attribution)**:
  - Longitudinal Rolling Baselines ($\mathcal{M}_0$): MAE $2.492$ (-8.08% error reduction)
  - + Circuit Aerodynamic Taxonomy: MAE $2.398$ (-11.55% error reduction)
  - + Micro-Sector Telemetry Dynamics: MAE $2.348$ (-13.39% error reduction)
  - + Intra-Race Discrete Ranking Filter: **MAE 2.191 (-19.16% error reduction)**
- **Hypothesis 3 (Tactical Podium Classification Calibration)**:
  - One-tailed Binomial test: $z = 9.07$, $p < 10^{-19}$ vs $68.50\%$ status-quo grid retention baseline.

---

### Computational Latency & Reproduction
- **Inference Latency**: 14.2 ms (full 20-car XGBoost prediction) | 1.62 s (5-step Llama-3-8B ReAct reasoning loop)
- **One-Command Figure Reproduction**: `python 06_Research_Paper/scripts/generate_all_figures.py`
- **One-Command Verification**: `pytest tests/test_e2e_research_paper.py -v`
