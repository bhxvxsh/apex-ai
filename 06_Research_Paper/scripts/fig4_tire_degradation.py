"""
APEX AI Research Paper - Figure 4: Non-Linear Tire Degradation Modeling
Visualizes compound-specific thermal degradation dynamics, tire blistering cliffs,
and the empirical Tire-Efficiency Performance Index (TEPI):
- Panel (a): Stint Lap-Time Degradation Penalty Delta t_deg(l) across Soft, Medium, Hard
- Panel (b): Bulk Surface Temperature Evolution T_b and Thermal Blistering Envelope
- Panel (c): Tire-Efficiency Performance Index (TEPI) Driver Benchmarks
Produces both vector PDF and 300+ DPI raster PNG.
"""

import os
import sys
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

    fig = plt.figure(figsize=(13.0, 5.8), facecolor="white", dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.1, 0.9], wspace=0.28, left=0.06, right=0.96, top=0.88, bottom=0.12)

    ax_deg = fig.add_subplot(gs[0, 0])
    ax_temp = fig.add_subplot(gs[0, 1])
    ax_tepi = fig.add_subplot(gs[0, 2])

    laps = np.linspace(1, 38, 300)

    # -------------------------------------------------------------
    # (a) Stint Lap Degradation Penalty Delta t_deg(l) [s]
    # -------------------------------------------------------------
    # Soft (C4/C5): delta_0 = -0.75s, s_deg = 0.125 s/lap, cliff at lap 16
    # Medium (C3): delta_0 = 0.00s, s_deg = 0.075 s/lap, cliff at lap 26
    # Hard (C1/C2): delta_0 = +0.55s, s_deg = 0.042 s/lap, cliff at lap 35

    def deg_curve(l, d0, s_deg, l_cliff, cliff_steep):
        linear = d0 + s_deg * l
        cliff = 1.8 / (1.0 + np.exp(-(l - l_cliff) / cliff_steep))
        return linear + cliff

    deg_soft = deg_curve(laps, -0.75, 0.125, 17.0, 1.8)
    deg_medium = deg_curve(laps, 0.00, 0.075, 26.5, 2.2)
    deg_hard = deg_curve(laps, 0.55, 0.042, 34.5, 2.5)

    ax_deg.set_facecolor("#f8fafc")
    ax_deg.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")

    # Shaded Strategic Windows
    ax_deg.axvspan(14, 22, color="#fef3c7", alpha=0.55, label="Undercut Pit Window (Soft/Medium)")
    ax_deg.axhspan(2.5, 6.0, color="#fee2e2", alpha=0.45, label="Severe Pace Loss / Cliff Regime")

    ax_deg.plot(laps, deg_soft, color="#dc2626", lw=2.2, label=r"Soft Compound (C4/C5, $s_{deg} = 0.125$ s/lap)")
    ax_deg.plot(laps, deg_medium, color="#d97706", lw=2.2, label=r"Medium Compound (C3, $s_{deg} = 0.075$ s/lap)")
    ax_deg.plot(laps, deg_hard, color="#0284c7", lw=2.2, label=r"Hard Compound (C1/C2, $s_{deg} = 0.042$ s/lap)")

    # Cliff Callouts
    ax_deg.scatter([17.0], [deg_curve(17.0, -0.75, 0.125, 17.0, 1.8)], color="#dc2626", s=45, zorder=5)
    ax_deg.annotate(
        r"Soft Thermal Cliff" + "\n" + r"(Lap 17, $T_{surf} > 130^\circ$C)",
        (17.0, 2.25), xytext=(6.5, 3.4),
        fontsize=7.3, weight="bold", color="#991b1b",
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#fca5a5")
    )

    ax_deg.set_xlabel("Stint Tire Age (Laps Completed)", fontsize=9.2, weight="bold", color="#0f172a")
    ax_deg.set_ylabel(r"Lap-Time Degradation $\Delta t_{deg}(l)$ (seconds)", fontsize=9.2, weight="bold", color="#0f172a")
    ax_deg.set_title("(a) Compound Lap Degradation & Thermal Cliff Trajectory", fontsize=9.8, weight="bold", pad=8, color="#0f172a")
    ax_deg.set_xlim(1, 38)
    ax_deg.set_ylim(-1.0, 5.0)
    ax_deg.legend(loc="upper left", fontsize=7.2, framealpha=0.95, edgecolor="#cbd5e1")

    # -------------------------------------------------------------
    # (b) Bulk Tire Surface Temperature T_b (°C) vs Laps
    # -------------------------------------------------------------
    # Optimal Window: 105°C - 125°C
    # Soft overheats quickly past 130°C (thermal blistering)
    # Medium stabilizes in window, rises toward end of stint
    # Hard requires 2-3 warmup laps to reach 105°C, then remains in window

    t_soft = 90.0 + (138.0 - 90.0) * (1.0 - np.exp(-laps / 5.5)) + 0.35 * laps
    t_medium = 85.0 + (126.0 - 85.0) * (1.0 - np.exp(-laps / 7.0)) + 0.18 * laps
    t_hard = 75.0 + (116.0 - 75.0) * (1.0 - np.exp(-laps / 9.0)) + 0.10 * laps

    ax_temp.set_facecolor("#f8fafc")
    ax_temp.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")

    # Optimal and Blistering Zones
    ax_temp.axhspan(105, 125, color="#dcfce7", alpha=0.55, label=r"Optimal Grip Window ($105 - 125^\circ$C)")
    ax_temp.axhspan(130, 150, color="#fee2e2", alpha=0.55, label=r"Blistering / Overheating ($>130^\circ$C)")

    ax_temp.plot(laps, t_soft, color="#dc2626", lw=2.2, label="Soft Bulk Surface Temp")
    ax_temp.plot(laps, t_medium, color="#d97706", lw=2.2, label="Medium Bulk Surface Temp")
    ax_temp.plot(laps, t_hard, color="#0284c7", lw=2.2, label="Hard Bulk Surface Temp")

    # Crossing threshold annotation
    ax_temp.axhline(130.0, color="#ef4444", linestyle=":", lw=1.2)
    ax_temp.text(2.0, 131.5, r"Thermal Cliff Threshold $T_{cliff} = 130^\circ$C", fontsize=7.2, color="#b91c1c", weight="bold")

    ax_temp.set_xlabel("Stint Tire Age (Laps Completed)", fontsize=9.2, weight="bold", color="#0f172a")
    ax_temp.set_ylabel(r"Bulk Surface Temperature $T_b$ ($^\circ$C)", fontsize=9.2, weight="bold", color="#0f172a")
    ax_temp.set_title("(b) Tire Bulk Thermal Evolution & Operating Limits", fontsize=9.8, weight="bold", pad=8, color="#0f172a")
    ax_temp.set_xlim(1, 38)
    ax_temp.set_ylim(70, 146)
    ax_temp.legend(loc="lower right", fontsize=7.2, framealpha=0.95, edgecolor="#cbd5e1")

    # -------------------------------------------------------------
    # (c) Tire-Efficiency Performance Index (TEPI) Benchmarks
    # -------------------------------------------------------------
    # TEPI = Win Rate (%) / (mean s_deg * 100)
    # Max Verstappen: Win Rate 50.0%, mean s_deg = 0.052 -> TEPI = 9.62
    # Lando Norris: Win Rate 14.1%, mean s_deg = 0.061 -> TEPI = 2.31
    # Charles Leclerc: Win Rate 7.8%, mean s_deg = 0.078 -> TEPI = 1.00
    # Field Average: Win Rate 3.5%, mean s_deg = 0.082 -> TEPI = 0.43

    drivers = ["M. Verstappen", "L. Norris", "C. Leclerc", "Field Average"]
    tepi_scores = [9.62, 2.31, 1.00, 0.43]
    s_deg_vals = [0.052, 0.061, 0.078, 0.082]
    win_rates = [50.0, 14.1, 7.8, 3.5]
    colors = ["#0284c7", "#f59e0b", "#dc2626", "#94a3b8"]

    ax_tepi.set_facecolor("#f8fafc")
    ax_tepi.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1", axis="y")

    bars = ax_tepi.bar(drivers, tepi_scores, color=colors, width=0.55, edgecolor="#334155", linewidth=0.9, zorder=3)

    for bar, score, s_deg, wr in zip(bars, tepi_scores, s_deg_vals, win_rates):
        h = bar.get_height()
        ax_tepi.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.25,
            f"{score:.2f}",
            ha="center", va="bottom", fontsize=8.2, weight="bold", color="#0f172a"
        )
        ax_tepi.text(
            bar.get_x() + bar.get_width() / 2.0, h / 2.0 if h > 1.5 else 0.7,
            f"WR: {wr}%\n" + r"$s_{deg}$:" + f" {s_deg:.3f}",
            ha="center", va="center", fontsize=6.8, color="white" if h > 1.5 else "#1e293b",
            weight="bold"
        )

    ax_tepi.set_ylabel(r"Tire-Efficiency Performance Index (TEPI)", fontsize=9.2, weight="bold", color="#0f172a")
    ax_tepi.set_title("(c) Empirical Driver TEPI Benchmarks", fontsize=9.8, weight="bold", pad=8, color="#0f172a")
    ax_tepi.set_ylim(0, 11.2)
    ax_tepi.tick_params(axis="x", labelsize=8.0)
    plt.setp(ax_tepi.get_xticklabels(), rotation=18, ha="right")

    # Formula callout
    ax_tepi.text(
        0.05, 0.94,
        r"$\mathrm{TEPI} = \frac{\mathrm{Win\,Rate\,(\%)}}{\bar{s}_{deg} \times 100}$",
        transform=ax_tepi.transAxes, fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.95)
    )

    # Save PDF and PNG
    pdf_path = os.path.join(output_dir, "fig4_tire_degradation.pdf")
    png_path = os.path.join(output_dir, "fig4_tire_degradation.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=305)
    plt.close(fig)

    print(f"[SUCCESS] Figure 4 generated:\n  - {pdf_path}\n  - {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    create_figure()
