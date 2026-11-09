"""
APEX AI Research Paper - Figure 2: Circuit Aerodynamic & Physics Taxonomy
Visualizes the 4 physical aerodynamic/thermal clusters across the 23 Grand Prix circuits:
- Panel (a): 2D Parameter Space (Full Throttle Duty Cycle % vs Asphalt Tire Stress Index)
- Panel (b): Engineering Radar Profile across 5 Vehicle Dynamics Dimensions
Produces both vector PDF and 300+ DPI raster PNG.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Ensure access to APEX_AI analytics if available
script_dir = os.path.dirname(os.path.abspath(__file__))
apex_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
analytics_dir = os.path.join(apex_root, "05_Analytics")
if analytics_dir not in sys.path:
    sys.path.insert(0, analytics_dir)

try:
    from circuit_clusters import CIRCUIT_TAXONOMY, OVERTAKING_DIFFICULTY_MAP
except ImportError:
    # Autonomous self-contained fallback matching circuit_clusters.py verbatim
    CIRCUIT_TAXONOMY = {
        "Italian Grand Prix": {"cluster_id": 1, "cluster_name": "Low-Downforce Power", "downforce_level": "Ultra-Low", "tire_stress_index": 3.2, "full_throttle_pct": 78.5, "overtaking_difficulty": "Low", "v_min_benchmark_kmh": 72.0, "brake_energy_level": "High"},
        "Belgian Grand Prix": {"cluster_id": 1, "cluster_name": "Low-Downforce Power", "downforce_level": "Low-Medium", "tire_stress_index": 4.5, "full_throttle_pct": 71.2, "overtaking_difficulty": "Low", "v_min_benchmark_kmh": 165.0, "brake_energy_level": "Medium"},
        "Las Vegas Grand Prix": {"cluster_id": 1, "cluster_name": "Low-Downforce Power", "downforce_level": "Ultra-Low", "tire_stress_index": 2.8, "full_throttle_pct": 76.0, "overtaking_difficulty": "Low", "v_min_benchmark_kmh": 85.0, "brake_energy_level": "High"},
        "Azerbaijan Grand Prix": {"cluster_id": 1, "cluster_name": "Low-Downforce Power", "downforce_level": "Low", "tire_stress_index": 3.4, "full_throttle_pct": 72.0, "overtaking_difficulty": "Low-Medium", "v_min_benchmark_kmh": 82.0, "brake_energy_level": "High"},
        "Saudi Arabian Grand Prix": {"cluster_id": 1, "cluster_name": "Low-Downforce Power", "downforce_level": "Low-Medium", "tire_stress_index": 3.5, "full_throttle_pct": 79.0, "overtaking_difficulty": "Low-Medium", "v_min_benchmark_kmh": 115.0, "brake_energy_level": "High"},
        "Bahrain Grand Prix": {"cluster_id": 2, "cluster_name": "High-Downforce Thermal Degradation", "downforce_level": "High", "tire_stress_index": 4.8, "full_throttle_pct": 64.0, "overtaking_difficulty": "Medium", "v_min_benchmark_kmh": 68.0, "brake_energy_level": "High"},
        "Spanish Grand Prix": {"cluster_id": 2, "cluster_name": "High-Downforce Thermal Degradation", "downforce_level": "High", "tire_stress_index": 4.9, "full_throttle_pct": 65.5, "overtaking_difficulty": "High", "v_min_benchmark_kmh": 138.0, "brake_energy_level": "Medium"},
        "British Grand Prix": {"cluster_id": 2, "cluster_name": "High-Downforce Thermal Degradation", "downforce_level": "High", "tire_stress_index": 4.9, "full_throttle_pct": 70.0, "overtaking_difficulty": "Medium", "v_min_benchmark_kmh": 245.0, "brake_energy_level": "Low"},
        "Japanese Grand Prix": {"cluster_id": 2, "cluster_name": "High-Downforce Thermal Degradation", "downforce_level": "High", "tire_stress_index": 5.0, "full_throttle_pct": 68.0, "overtaking_difficulty": "High", "v_min_benchmark_kmh": 182.0, "brake_energy_level": "Medium"},
        "Qatar Grand Prix": {"cluster_id": 2, "cluster_name": "High-Downforce Thermal Degradation", "downforce_level": "High", "tire_stress_index": 5.0, "full_throttle_pct": 70.5, "overtaking_difficulty": "Medium", "v_min_benchmark_kmh": 195.0, "brake_energy_level": "Low-Medium"},
        "Monaco Grand Prix": {"cluster_id": 3, "cluster_name": "Maximum-Downforce Mechanical Grip", "downforce_level": "Maximum", "tire_stress_index": 2.1, "full_throttle_pct": 45.0, "overtaking_difficulty": "Extreme", "v_min_benchmark_kmh": 46.0, "brake_energy_level": "Low"},
        "Hungarian Grand Prix": {"cluster_id": 3, "cluster_name": "Maximum-Downforce Mechanical Grip", "downforce_level": "Maximum", "tire_stress_index": 3.8, "full_throttle_pct": 58.0, "overtaking_difficulty": "High", "v_min_benchmark_kmh": 92.0, "brake_energy_level": "High"},
        "Singapore Grand Prix": {"cluster_id": 3, "cluster_name": "Maximum-Downforce Mechanical Grip", "downforce_level": "Maximum", "tire_stress_index": 3.9, "full_throttle_pct": 52.0, "overtaking_difficulty": "High", "v_min_benchmark_kmh": 74.0, "brake_energy_level": "High"},
        "Dutch Grand Prix": {"cluster_id": 3, "cluster_name": "Maximum-Downforce Mechanical Grip", "downforce_level": "High-Maximum", "tire_stress_index": 4.1, "full_throttle_pct": 62.0, "overtaking_difficulty": "High", "v_min_benchmark_kmh": 110.0, "brake_energy_level": "Medium"},
        "Miami Grand Prix": {"cluster_id": 4, "cluster_name": "Hybrid Street / Stop-and-Go", "downforce_level": "Medium-High", "tire_stress_index": 3.7, "full_throttle_pct": 58.0, "overtaking_difficulty": "Medium-High", "v_min_benchmark_kmh": 78.0, "brake_energy_level": "Extreme"},
        "Australian Grand Prix": {"cluster_id": 4, "cluster_name": "Hybrid Street / Stop-and-Go", "downforce_level": "Medium", "tire_stress_index": 3.8, "full_throttle_pct": 68.0, "overtaking_difficulty": "Medium", "v_min_benchmark_kmh": 125.0, "brake_energy_level": "High"},
        "Canadian Grand Prix": {"cluster_id": 4, "cluster_name": "Hybrid Street / Stop-and-Go", "downforce_level": "Medium-Low", "tire_stress_index": 3.1, "full_throttle_pct": 60.0, "overtaking_difficulty": "Low-Medium", "v_min_benchmark_kmh": 65.0, "brake_energy_level": "Extreme"},
        "Austrian Grand Prix": {"cluster_id": 4, "cluster_name": "Hybrid Street / Stop-and-Go", "downforce_level": "Medium-High", "tire_stress_index": 3.9, "full_throttle_pct": 69.0, "overtaking_difficulty": "Low", "v_min_benchmark_kmh": 105.0, "brake_energy_level": "High"},
        "United States Grand Prix": {"cluster_id": 4, "cluster_name": "Hybrid Street / Stop-and-Go", "downforce_level": "Medium-High", "tire_stress_index": 4.2, "full_throttle_pct": 62.0, "overtaking_difficulty": "Low-Medium", "v_min_benchmark_kmh": 88.0, "brake_energy_level": "High"},
        "Mexico City Grand Prix": {"cluster_id": 4, "cluster_name": "Hybrid Street / Stop-and-Go", "downforce_level": "Maximum-Aerodynamic", "tire_stress_index": 3.3, "full_throttle_pct": 55.0, "overtaking_difficulty": "High", "v_min_benchmark_kmh": 70.0, "brake_energy_level": "Extreme"},
        "São Paulo Grand Prix": {"cluster_id": 4, "cluster_name": "Hybrid Street / Stop-and-Go", "downforce_level": "Medium-High", "tire_stress_index": 4.0, "full_throttle_pct": 65.0, "overtaking_difficulty": "Low-Medium", "v_min_benchmark_kmh": 95.0, "brake_energy_level": "Medium-High"},
        "Emilia Romagna Grand Prix": {"cluster_id": 4, "cluster_name": "Hybrid Street / Stop-and-Go", "downforce_level": "Medium-High", "tire_stress_index": 3.9, "full_throttle_pct": 66.0, "overtaking_difficulty": "High", "v_min_benchmark_kmh": 112.0, "brake_energy_level": "High"},
        "Chinese Grand Prix": {"cluster_id": 4, "cluster_name": "Hybrid Street / Stop-and-Go", "downforce_level": "Medium-High", "tire_stress_index": 4.4, "full_throttle_pct": 61.0, "overtaking_difficulty": "Low-Medium", "v_min_benchmark_kmh": 76.0, "brake_energy_level": "High"},
        "Abu Dhabi Grand Prix": {"cluster_id": 4, "cluster_name": "Hybrid Street / Stop-and-Go", "downforce_level": "Medium", "tire_stress_index": 3.6, "full_throttle_pct": 63.0, "overtaking_difficulty": "Medium", "v_min_benchmark_kmh": 82.0, "brake_energy_level": "High"},
    }
    OVERTAKING_DIFFICULTY_MAP = {"Low": 1.0, "Low-Medium": 1.5, "Medium": 2.0, "Medium-High": 2.5, "High": 3.0, "Extreme": 3.5}

BRAKE_ENERGY_MAP = {"Low": 1.0, "Low-Medium": 1.5, "Medium": 2.0, "Medium-High": 2.5, "High": 3.0, "Extreme": 4.0}


def create_figure(output_dir=None):
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(script_dir, "..", "figures"))
    os.makedirs(output_dir, exist_ok=True)

    # Set up publication font and style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 0.9

    fig = plt.figure(figsize=(13.0, 5.8), facecolor="white", dpi=300)

    # Grid: Panel (a) on left (60% width), Panel (b) polar radar on right (40% width)
    ax_scatter = fig.add_axes([0.07, 0.12, 0.52, 0.80])
    ax_radar = fig.add_axes([0.64, 0.12, 0.33, 0.80], polar=True)

    # Cluster Configuration & IEEE Palettes
    cluster_meta = {
        1: {"name": "Cluster 1: Low-Downforce Power", "color": "#0284c7", "marker": "o", "edge": "#0369a1"},
        2: {"name": "Cluster 2: High-Downforce Thermal Deg", "color": "#dc2626", "marker": "s", "edge": "#991b1b"},
        3: {"name": "Cluster 3: Max-Downforce Mechanical Grip", "color": "#16a34a", "marker": "^", "edge": "#15803d"},
        4: {"name": "Cluster 4: Hybrid Stop-and-Go", "color": "#d97706", "marker": "D", "edge": "#b45309"},
    }

    # Extract Data Points
    tracks = []
    x_throttles = []
    y_stresses = []
    c_ids = []
    v_mins = []
    overtakes = []
    brakes = []

    for name, data in CIRCUIT_TAXONOMY.items():
        if name == "Default":
            continue
        tracks.append(name.replace(" Grand Prix", ""))
        x_throttles.append(data["full_throttle_pct"])
        y_stresses.append(data["tire_stress_index"])
        c_ids.append(data["cluster_id"])
        v_mins.append(data["v_min_benchmark_kmh"])
        overtakes.append(OVERTAKING_DIFFICULTY_MAP.get(data["overtaking_difficulty"], 2.0))
        brakes.append(BRAKE_ENERGY_MAP.get(data.get("brake_energy_level", "Medium"), 2.0))

    x_throttles = np.array(x_throttles)
    y_stresses = np.array(y_stresses)
    c_ids = np.array(c_ids)
    v_mins = np.array(v_mins)
    overtakes = np.array(overtakes)
    brakes = np.array(brakes)

    # -------------------------------------------------------------
    # PANEL (a): 2D Parameter Space (Throttle % vs Tire Stress Index)
    # -------------------------------------------------------------
    ax_scatter.set_facecolor("#f8fafc")
    ax_scatter.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1", zorder=1)

    # Plot confidence ellipses per cluster
    for cid in [1, 2, 3, 4]:
        mask = (c_ids == cid)
        cx = x_throttles[mask]
        cy = y_stresses[mask]

        mean_x, mean_y = np.mean(cx), np.mean(cy)
        cov = np.cov(cx, cy)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height of ellipse (2 std deviations ~ 95% coverage)
        width, height = 2.4 * np.sqrt(np.maximum(vals, 1e-4))
        ell = Ellipse(
            xy=(mean_x, mean_y), width=max(width, 4.0), height=max(height, 0.45),
            angle=theta, facecolor=cluster_meta[cid]["color"], alpha=0.12,
            edgecolor=cluster_meta[cid]["color"], linestyle="-", linewidth=1.2, zorder=2
        )
        ax_scatter.add_patch(ell)

    # Scatter Points
    for cid in [1, 2, 3, 4]:
        mask = (c_ids == cid)
        meta = cluster_meta[cid]
        # Point size proportional to Apex Velocity (scaled 50 to 250)
        sizes = 45 + (v_mins[mask] / 245.0) * 170

        ax_scatter.scatter(
            x_throttles[mask], y_stresses[mask],
            s=sizes, c=meta["color"], marker=meta["marker"],
            edgecolors=meta["edge"], linewidths=1.2, alpha=0.90,
            label=f"{meta['name']} (n={np.sum(mask)})", zorder=4
        )

    # Annotate Selected Prominent Circuits
    annotations = {
        "Italian": (1.2, 0.12),
        "Belgian": (-4.5, 0.15),
        "Monaco": (1.2, -0.05),
        "British": (-4.0, 0.16),
        "Japanese": (1.2, 0.10),
        "Las Vegas": (1.2, -0.15),
        "Hungaroring": (-6.5, -0.15),
        "Singapore": (1.2, -0.12),
        "Bahrain": (-4.5, -0.18),
        "Canadian": (-5.0, -0.18),
        "Miami": (1.2, 0.12),
    }

    for track, x, y in zip(tracks, x_throttles, y_stresses):
        for key, (ox, oy) in annotations.items():
            if key in track:
                ax_scatter.annotate(
                    track, (x, y), xytext=(x + ox, y + oy),
                    fontsize=7.2, weight="bold", color="#1e293b",
                    arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.6, shrinkA=3, shrinkB=3),
                    zorder=5
                )

    ax_scatter.set_xlim(42.0, 82.5)
    ax_scatter.set_ylim(1.8, 5.4)
    ax_scatter.set_xlabel("Full-Throttle Duty Cycle $D_{throttle}$ (%)", fontsize=10, weight="bold", color="#0f172a")
    ax_scatter.set_ylabel("Asphalt Macro-Texture Tire Stress Index $T_{stress}$ (1 - 5)", fontsize=10, weight="bold", color="#0f172a")
    ax_scatter.set_title("(a) Aerodynamic Downforce & Tire Stress Parameter Space across 23 GPs", fontsize=10.5, weight="bold", pad=9, color="#0f172a")

    legend = ax_scatter.legend(loc="lower right", fontsize=8.0, framealpha=0.95, edgecolor="#cbd5e1")
    legend.get_frame().set_facecolor("white")

    # Inset Note on marker size
    ax_scatter.text(
        43.5, 5.15, r"Marker Size $\propto$ Micro-Sector $V_{min}$ Benchmark (46 to 245 km/h)",
        fontsize=7.3, style="italic", color="#475569",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#e2e8f0", alpha=0.9)
    )

    # -------------------------------------------------------------
    # PANEL (b): Engineering Radar Profile across 5 Dimensions
    # -------------------------------------------------------------
    categories = [
        "Full Throttle\nDuty Cycle (%)",
        "Tire Stress\nIndex",
        "Apex Velocity\n$V_{min}$ (km/h)",
        "Overtaking\nDifficulty",
        "Braking Energy\nSeverity"
    ]
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle loop

    ax_radar.set_facecolor("#f8fafc")
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)

    plt.sca(ax_radar)
    plt.xticks(angles[:-1], categories, fontsize=7.8, weight="bold", color="#1e293b")
    ax_radar.set_rlabel_position(0)
    plt.yticks([0.25, 0.50, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="#64748b", fontsize=6.8)
    plt.ylim(0, 1.08)
    ax_radar.grid(True, linestyle=":", color="#cbd5e1", linewidth=0.8)

    # Normalize dimensions to 0 - 1
    # Full throttle: 45 - 80%
    # Tire stress: 2.0 - 5.0
    # V_min: 45 - 250 km/h
    # Overtaking: 1.0 - 3.5
    # Brake energy: 1.0 - 4.0

    def norm_feat(vals, vmin, vmax):
        return np.clip((vals - vmin) / (vmax - vmin), 0.05, 1.0)

    for cid in [1, 2, 3, 4]:
        mask = (c_ids == cid)
        meta = cluster_meta[cid]

        m_throttle = np.mean(norm_feat(x_throttles[mask], 45.0, 80.0))
        m_stress = np.mean(norm_feat(y_stresses[mask], 2.0, 5.0))
        m_vmin = np.mean(norm_feat(v_mins[mask], 45.0, 245.0))
        m_overtake = np.mean(norm_feat(overtakes[mask], 1.0, 3.5))
        m_brake = np.mean(norm_feat(brakes[mask], 1.0, 4.0))

        values = [m_throttle, m_stress, m_vmin, m_overtake, m_brake]
        values += values[:1]

        ax_radar.plot(angles, values, color=meta["color"], linewidth=1.8, label=f"C{cid}")
        ax_radar.fill(angles, values, color=meta["color"], alpha=0.15)

    ax_radar.set_title("(b) 5-D Engineering Radar Profile", fontsize=10.5, weight="bold", pad=20, color="#0f172a")
    radar_legend = ax_radar.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=7.5, framealpha=0.9, edgecolor="#cbd5e1")

    # Save PDF and PNG
    pdf_path = os.path.join(output_dir, "fig2_circuit_taxonomy.pdf")
    png_path = os.path.join(output_dir, "fig2_circuit_taxonomy.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=305)
    plt.close(fig)

    print(f"[SUCCESS] Figure 2 generated:\n  - {pdf_path}\n  - {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    create_figure()
