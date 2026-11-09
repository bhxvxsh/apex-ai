"""
APEX AI Research Paper - Figure 7: Feature Importance & Ablation Breakdown
Multi-panel visualization of feature gain importances, step-wise component ablation,
and downforce cluster-specific error reductions:
- Panel (a): Top 12 Predictive Features Ranked by Gain Importance across Model Domains
- Panel (b): Step-Wise Component Ablation Waterfall (Baseline 2.711 -> APEX AI 2.191)
- Panel (c): Circuit Aerodynamic Downforce Cluster MAE Breakdown
Produces both vector PDF and 300+ DPI raster PNG.
"""

import os
import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_figure(output_dir=None):
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(script_dir, "..", "figures"))
    os.makedirs(output_dir, exist_ok=True)

    # Set up IEEE publication styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 0.9

    fig = plt.figure(figsize=(13.5, 5.8), facecolor="white", dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.15, 1.05], wspace=0.28, left=0.06, right=0.96, top=0.88, bottom=0.12)

    ax_feat = fig.add_subplot(gs[0, 0])
    ax_ablation = fig.add_subplot(gs[0, 1])
    ax_cluster = fig.add_subplot(gs[0, 2])

    # -------------------------------------------------------------
    # (a) Top 12 Feature Gain Importances (xgb_race_predictor)
    # -------------------------------------------------------------
    features = [
        "driver_podium_rate",
        "GridPosition",
        "driver_race_mean_last3",
        "team_race_mean",
        "driver_race_mean_all",
        "team_points_ytd",
        "pole_delta_s",
        "overtaking_sensitivity",
        "q_pace_efficiency",
        "driver_finish_rate",
        "q_session_progression",
        "tire_deg_stress_impact"
    ][::-1]

    gains = [
        38.92,
        13.18,
        10.71,
        8.11,
        4.71,
        3.76,
        3.48,
        2.69,
        2.31,
        1.50,
        1.42,
        1.23
    ][::-1]

    # Color code by domain
    # Driver Form: Blue #0284c7
    # Qualifying/Grid: Green #16a34a
    # Team Baseline: Amber #f59e0b
    # Circuit/Physics: Red #dc2626
    domain_colors = {
        "driver_podium_rate": "#0284c7",
        "GridPosition": "#16a34a",
        "driver_race_mean_last3": "#0284c7",
        "team_race_mean": "#f59e0b",
        "driver_race_mean_all": "#0284c7",
        "team_points_ytd": "#f59e0b",
        "pole_delta_s": "#16a34a",
        "overtaking_sensitivity": "#dc2626",
        "q_pace_efficiency": "#16a34a",
        "driver_finish_rate": "#0284c7",
        "q_session_progression": "#16a34a",
        "tire_deg_stress_impact": "#dc2626"
    }
    bar_colors = [domain_colors[f] for f in features]

    ax_feat.set_facecolor("#f8fafc")
    ax_feat.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1", axis="x")

    y_pos = np.arange(len(features))
    bars = ax_feat.barh(y_pos, gains, color=bar_colors, height=0.68, edgecolor="#334155", linewidth=0.7)

    for bar, val in zip(bars, gains):
        w = bar.get_width()
        ax_feat.text(
            w + 0.6, bar.get_y() + bar.get_height() / 2.0,
            f"{val:.2f}%",
            va="center", ha="left", fontsize=7.2, weight="bold", color="#1e293b"
        )

    clean_names = [
        "Driver Career Podium Rate",
        "Qualifying Grid Position",
        "Rolling Form (Last 3 GPs)",
        "Team Constructor Mean",
        "Driver Career Mean Finish",
        "Team Points YTD",
        "Session Pole Lap Delta (s)",
        "Overtaking Sensitivity",
        "Relative Pace Efficiency",
        "Driver Race Finish Rate",
        "Intra-Session Progression",
        "Tire Deg Stress Impact"
    ][::-1]

    ax_feat.set_yticks(y_pos)
    ax_feat.set_yticklabels(clean_names, fontsize=7.8, color="#0f172a")
    ax_feat.set_xlabel("Relative Gain Feature Importance (%)", fontsize=9.2, weight="bold", color="#0f172a")
    ax_feat.set_title("(a) XGBoost Predictor Feature Importance Profile", fontsize=9.8, weight="bold", pad=8, color="#0f172a")
    ax_feat.set_xlim(0, 45)

    # Inset Legend for domains
    custom_legend = [
        plt.Line2D([0], [0], color="#0284c7", lw=4, label="Driver Form"),
        plt.Line2D([0], [0], color="#16a34a", lw=4, label="Qualifying/Grid"),
        plt.Line2D([0], [0], color="#f59e0b", lw=4, label="Constructor/Team"),
        plt.Line2D([0], [0], color="#dc2626", lw=4, label="Circuit Physics"),
    ]
    ax_feat.legend(handles=custom_legend, loc="lower right", fontsize=6.8, framealpha=0.92, edgecolor="#cbd5e1")

    # -------------------------------------------------------------
    # (b) Step-Wise Component Ablation Waterfall
    # -------------------------------------------------------------
    stages = [
        "Grid\nBaseline",
        "+ Rolling\nForm",
        "+ XGBoost\nNon-Linear",
        "+ Micro-Sector\nDynamics",
        "+ Circuit\nTaxonomy",
        "+ Intra-Race\nRanking"
    ]
    deltas = [2.711, -0.308, -0.055, -0.063, -0.042, -0.052]
    running_totals = [2.711, 2.403, 2.348, 2.285, 2.243, 2.191]

    ax_ablation.set_facecolor("#f8fafc")
    ax_ablation.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1", axis="y")

    x_idx = np.arange(len(stages))

    # Base bar (start)
    ax_ablation.bar(0, running_totals[0], color="#64748b", width=0.55, edgecolor="#334155", linewidth=0.9)
    ax_ablation.text(0, running_totals[0] + 0.04, f"{running_totals[0]:.3f}", ha="center", fontsize=7.8, weight="bold", color="#1e293b")

    # Waterfall steps
    for i in range(1, len(stages) - 1):
        prev = running_totals[i - 1]
        curr = running_totals[i]
        delta = deltas[i]
        # Falling bar
        ax_ablation.bar(i, abs(delta), bottom=curr, color="#10b981", width=0.55, edgecolor="#047857", linewidth=0.8)
        ax_ablation.text(i, curr - 0.08, f"{delta:+.3f}", ha="center", fontsize=7.2, weight="bold", color="#047857")

    # Final bar (Full APEX AI)
    final_idx = len(stages) - 1
    ax_ablation.bar(final_idx, running_totals[-1], color="#0284c7", width=0.55, edgecolor="#0369a1", linewidth=1.1)
    ax_ablation.text(final_idx, running_totals[-1] + 0.04, f"{running_totals[-1]:.3f}", ha="center", fontsize=8.0, weight="bold", color="#0284c7")

    # Annotate total delta
    ax_ablation.annotate(
        "Total Reduction:\n-0.520 pos (-19.16%)",
        (final_idx, running_totals[-1]), xytext=(final_idx - 1.8, 2.50),
        fontsize=7.8, weight="bold", color="#0369a1",
        arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#bae6fd")
    )

    ax_ablation.set_xticks(x_idx)
    ax_ablation.set_xticklabels(stages, fontsize=7.2, color="#0f172a")
    ax_ablation.set_ylabel("Out-of-Sample MAE (Positions)", fontsize=9.2, weight="bold", color="#0f172a")
    ax_ablation.set_title("(b) Step-Wise Component Ablation Waterfall", fontsize=9.8, weight="bold", pad=8, color="#0f172a")
    ax_ablation.set_ylim(2.0, 2.88)

    # -------------------------------------------------------------
    # (c) Circuit Aerodynamic Cluster Breakdown (MAE Comparison)
    # -------------------------------------------------------------
    clusters = [
        "Cluster 1\nLow-Downforce\nPower",
        "Cluster 3\nMax-Downforce\nMechanical",
        "Cluster 2\nHigh-Downforce\nThermal Deg",
        "Cluster 4\nHybrid\nStop-and-Go"
    ]
    grid_cluster_mae = [2.580, 2.620, 2.740, 2.890]
    apex_cluster_mae = [1.942, 2.090, 2.130, 2.402]
    improvements = [24.73, 20.23, 22.26, 16.89]

    ax_cluster.set_facecolor("#f8fafc")
    ax_cluster.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1", axis="y")

    w = 0.35
    c_x = np.arange(len(clusters))

    rects1 = ax_cluster.bar(c_x - w / 2.0, grid_cluster_mae, w, label="Grid Baseline", color="#94a3b8", edgecolor="#475569", linewidth=0.8)
    rects2 = ax_cluster.bar(c_x + w / 2.0, apex_cluster_mae, w, label="APEX AI Ranked", color="#0284c7", edgecolor="#0369a1", linewidth=0.8)

    for i in range(len(clusters)):
        ax_cluster.text(
            c_x[i] - w / 2.0, grid_cluster_mae[i] + 0.04,
            f"{grid_cluster_mae[i]:.2f}",
            ha="center", fontsize=7.2, color="#475569", weight="bold"
        )
        ax_cluster.text(
            c_x[i] + w / 2.0, apex_cluster_mae[i] + 0.04,
            f"{apex_cluster_mae[i]:.3f}",
            ha="center", fontsize=7.2, color="#0284c7", weight="bold"
        )
        # Percentage reduction badge
        ax_cluster.text(
            c_x[i], max(grid_cluster_mae[i], apex_cluster_mae[i]) + 0.16,
            f"-{improvements[i]:.1f}%",
            ha="center", fontsize=6.8, weight="bold", color="#15803d",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#dcfce7", edgecolor="#86efac", alpha=0.95)
        )

    ax_cluster.set_xticks(c_x)
    ax_cluster.set_xticklabels(clusters, fontsize=7.0, color="#0f172a")
    ax_cluster.set_ylabel("Mean Absolute Error (Positions)", fontsize=9.2, weight="bold", color="#0f172a")
    ax_cluster.set_title("(c) Performance Across Aerodynamic Clusters", fontsize=9.8, weight="bold", pad=8, color="#0f172a")
    ax_cluster.set_ylim(1.6, 3.25)
    ax_cluster.legend(loc="upper left", fontsize=7.2, framealpha=0.95, edgecolor="#cbd5e1")

    # Save both canonical and alias names
    pdf_path = os.path.join(output_dir, "fig7_feature_ablation.pdf")
    png_path = os.path.join(output_dir, "fig7_feature_ablation.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=305)
    plt.close(fig)

    # Alias matching PROJECT.md interface contract: fig7_feature_importance_ablation
    alias_pdf = os.path.join(output_dir, "fig7_feature_importance_ablation.pdf")
    alias_png = os.path.join(output_dir, "fig7_feature_importance_ablation.png")
    shutil.copy2(pdf_path, alias_pdf)
    shutil.copy2(png_path, alias_png)

    print(f"[SUCCESS] Figure 7 generated:\n  - {pdf_path}\n  - {png_path}\n  - {alias_pdf}\n  - {alias_png}")
    return pdf_path, png_path


if __name__ == "__main__":
    create_figure()
