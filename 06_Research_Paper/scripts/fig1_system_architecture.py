"""
APEX AI Research Paper - Figure 1: End-to-End System Architecture Diagram
Publication-quality multi-tier architecture diagram adhering to IEEE Transactions standards.
Produces both vector PDF and 300+ DPI raster PNG with clean, non-overlapping typography.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def create_figure(output_dir=None):
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(script_dir, "..", "figures"))
    os.makedirs(output_dir, exist_ok=True)

    # IEEE double-column width styling (16.0 x 9.6 inches for high clarity and breathing room)
    fig = plt.figure(figsize=(16.0, 9.6), facecolor="white", dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#f8fafc")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Color Palette - Professional Engineering Dark/Light Theme
    c_tier1_bg = "#eff6ff"      # Soft Blue
    c_tier1_border = "#2563eb"
    c_tier2_bg = "#f0fdf4"      # Soft Emerald
    c_tier2_border = "#059669"
    c_tier3_bg = "#fefce8"      # Soft Amber
    c_tier3_border = "#d97706"
    c_tier4_bg = "#faf5ff"      # Soft Purple
    c_tier4_border = "#7c3aed"
    c_tier5_bg = "#fff1f2"      # Soft Rose
    c_tier5_border = "#e11d48"

    card_bg = "#ffffff"
    text_dark = "#0f172a"
    text_muted = "#334155"

    # Title Banner
    ax.text(
        50, 97.0,
        "APEX AI: End-to-End Predictive Telemetry & Autonomous Strategic Reasoning Architecture",
        ha="center", va="center", fontsize=15.0, weight="bold", color="#0f172a",
        family="sans-serif"
    )
    ax.text(
        50, 94.4,
        "Multi-modal telemetry ingestion, physical aerodynamic clustering, intra-race ranking, and tool-augmented ReAct reasoning",
        ha="center", va="center", fontsize=9.8, style="italic", color="#475569",
        family="sans-serif"
    )

    # 5 vertical columns / tiers across the x axis:
    # Tier 1: x=2.0, w=17.0  (ends 19.0)
    # Tier 2: x=21.4, w=17.4 (ends 38.8)
    # Tier 3: x=41.2, w=17.8 (ends 59.0)
    # Tier 4: x=61.2, w=17.8 (ends 79.0)
    # Tier 5: x=81.2, w=16.8 (ends 98.0)
    #
    # Vertical grid:
    # Tier Box: y=3.0, h=88.5 (top=91.5)
    # Header:   y=84.5, h=6.5  (top=91.0)
    # Badge:    y=79.8, h=3.2  (top=83.0)
    # Cards:
    #   Card 1: y=61.0, h=17.2 (top=78.2)  -> 1.6 gap to badge
    #   Card 2: y=42.4, h=17.2 (top=59.6)  -> 1.4 gap to card 1
    #   Card 3: y=23.6, h=17.4 (top=41.0)  -> 1.4 gap to card 2
    #   Card 4: y=4.8,  h=17.4 (top=22.2)  -> 1.4 gap to card 3

    tiers = [
        {
            "name": "Tier 1: Data Ingestion & Storage",
            "subtitle": "Raw Feeds & Temporal Normalization",
            "x": 2.0, "w": 17.2, "y": 3.0, "h": 88.5,
            "bg": c_tier1_bg, "border": c_tier1_border,
            "badge": "64 GPs | 1,119 Finishes",
            "cards": [
                {
                    "title": "FastF1 & Timing Feeds",
                    "bullets": [
                        "• 10 Hz Car Telemetry (GPS, RPM)",
                        "• Throttle / Brake / Gear Channels",
                        "• Official FIA Timing Intervals"
                    ],
                    "y": 61.0, "h": 17.2
                },
                {
                    "title": "Data Normalizer",
                    "code": "clean_data.py",
                    "bullets": [
                        "• String Intervals to Floats (s)",
                        "• Status Flags (finished = 1 / 0)",
                        "• Normalized Qualifying Splits"
                    ],
                    "y": 42.4, "h": 17.2
                },
                {
                    "title": "Relational DB Engine",
                    "code": "f1.db (SQLite)",
                    "bullets": [
                        "• race_results (1,278 rows)",
                        "• qualifying_results (1,318 rows)",
                        "• 2023 - 2025 Multi-Season"
                    ],
                    "y": 23.6, "h": 17.4
                },
                {
                    "title": "Temporal Causality Guard",
                    "bullets": [
                        "• Zero Look-Ahead Leakage",
                        "• Strictly Ordered Historical Form",
                        "• Expanding Train/Test Windows"
                    ],
                    "y": 4.8, "h": 17.4
                }
            ]
        },
        {
            "name": "Tier 2: Feature Engineering",
            "subtitle": "Micro-Sectors & Physics Clustering",
            "x": 21.4, "w": 17.6, "y": 3.0, "h": 88.5,
            "bg": c_tier2_bg, "border": c_tier2_border,
            "badge": "23 Domain Features",
            "cards": [
                {
                    "title": "Micro-Sector Telemetry",
                    "bullets": [
                        r"• Pace Efficiency: $\eta_{pace} = t_{best} / t_{pole}$",
                        r"• Progression: $\Pi_{prog} = \Delta t_{Q1 \to Q3}$",
                        r"• Apex Speed: $\eta_{apex} = V_{min} / (1 + \Delta t)$"
                    ],
                    "y": 61.0, "h": 17.2
                },
                {
                    "title": "Circuit Aerodynamics",
                    "code": "circuit_clusters.py",
                    "bullets": [
                        "• 4 Physics Downforce Clusters",
                        r"• Full-Throttle Duty: $D_{thr} \in [45, 79]\%$",
                        r"• Tire Stress: $T_{stress} \in [2.1, 5.0]$",
                        r"• Overtaking Difficulty: $O_{diff}$"
                    ],
                    "y": 42.4, "h": 17.2
                },
                {
                    "title": "Driver & Team Baselines",
                    "bullets": [
                        "• Rolling Form (Last 3 / All-Time)",
                        "• Driver Momentum (Recent vs All)",
                        "• Historical Podium / Win Rates",
                        "• Teammate Qualifying Delta"
                    ],
                    "y": 23.6, "h": 17.4
                },
                {
                    "title": "Feature Matrix Export",
                    "code": "full_features.csv",
                    "bullets": [
                        "• 1,379 Rows x 40 Columns",
                        "• 1,119 Verified Classifications",
                        "• Standardized Numerical Tensors"
                    ],
                    "y": 4.8, "h": 17.4
                }
            ]
        },
        {
            "name": "Tier 3: Supervised ML Core",
            "subtitle": "Predictive Models & Ranking",
            "x": 41.2, "w": 18.0, "y": 3.0, "h": 88.5,
            "bg": c_tier3_bg, "border": c_tier3_border,
            "badge": r"MAE 2.191 | p = 6.24e-10",
            "cards": [
                {
                    "title": "Expanding Walk-Forward CV",
                    "code": "walk_forward_cv.py",
                    "bullets": [
                        "• 15 Warmup Events (262 cars)",
                        "• 49 Out-of-Sample GP Folds",
                        "• 857 Finishes Evaluated",
                        "• 19.16% Error Reduction"
                    ],
                    "y": 61.0, "h": 17.2
                },
                {
                    "title": "Intra-Race Ranked XGBoost",
                    "code": "xgb_race_predictor.joblib",
                    "bullets": [
                        "• Continuous Lap Pace Regression",
                        r"• Discrete Sort: $\hat{P}_i = \operatorname{rank}(\hat{y}_i)$",
                        "• MAE: 2.191 (vs 2.711 Baseline)",
                        r"• Acc $\pm$1 Pos: 49.3%, $\pm$2: 67.2%"
                    ],
                    "y": 42.4, "h": 17.2
                },
                {
                    "title": "High-Stakes Classifiers",
                    "bullets": [
                        "• Top-3 Podium Classifier:",
                        "    ROC-AUC: 94.43% | Acc: 90.43%",
                        "• Top-10 Points Classifier:",
                        "    ROC-AUC: 81.78% | Acc: 75.22%"
                    ],
                    "y": 23.6, "h": 17.4
                },
                {
                    "title": "Cluster-Specific Baselines",
                    "bullets": [
                        "• Low Downforce Power: 1.942",
                        "• High Downforce Thermal: 2.130",
                        "• Street Circuit Agility: 2.315"
                    ],
                    "y": 4.8, "h": 17.4
                }
            ]
        },
        {
            "name": "Tier 4: ReAct Autonomous Agent",
            "subtitle": "Tool-Augmented Strategic Reasoning",
            "x": 61.4, "w": 17.8, "y": 3.0, "h": 88.5,
            "bg": c_tier4_bg, "border": c_tier4_border,
            "badge": "5-Stage Reasoning Loop",
            "cards": [
                {
                    "title": "ReAct Multi-Step Engine",
                    "code": "react_agent.py",
                    "bullets": [
                        "• Step 1: Baseline Form Profiling",
                        "• Step 2: Aerodynamic Regime",
                        "• Step 3: Supervised ML Predict",
                        "• Step 4: Empirical Reflection",
                        "• Step 5: Strategic Thesis Synthesis"
                    ],
                    "y": 61.0, "h": 17.2
                },
                {
                    "title": "Unified Tool Registry",
                    "code": "tools.py",
                    "bullets": [
                        "• 8 Typed JSON Tool Schemas:",
                        "  predict_race, circuit_profile,",
                        "  driver_career_stats, standings,",
                        "  qualifying_race_delta, etc."
                    ],
                    "y": 42.4, "h": 17.2
                },
                {
                    "title": "Dynamic Sandbox (LATM)",
                    "code": "dynamic_analyst.py",
                    "bullets": [
                        "• LLMs-As-Tool-Makers Pattern",
                        "• Safe Python / SQL Execution",
                        "• On-the-Fly Telemetry Scripting",
                        "• Real-Time Output Interception"
                    ],
                    "y": 23.6, "h": 17.4
                },
                {
                    "title": "Decision Calibration",
                    "bullets": [
                        r"• Brier Reliability & ECE",
                        r"• Binomial Test: $z=9.07, p<10^{-19}$",
                        "• Calibrated Decision Confidence"
                    ],
                    "y": 4.8, "h": 17.4
                }
            ]
        },
        {
            "name": "Tier 5: LLM Layer & Strategy",
            "subtitle": "Fine-Tuning & Decision Output",
            "x": 81.4, "w": 16.6, "y": 3.0, "h": 88.5,
            "bg": c_tier5_bg, "border": c_tier5_border,
            "badge": "0.52% Parameter QLoRA",
            "cards": [
                {
                    "title": "Base LLM Model",
                    "bullets": [
                        "• Meta Llama-3-8B-Instruct",
                        "• 4-Bit NormalFloat (NF4)",
                        "• Double Quantization",
                        "• 2,048 Sequence Window"
                    ],
                    "y": 61.0, "h": 17.2
                },
                {
                    "title": "QLoRA Fine-Tuning",
                    "code": "f1_technical_alpaca.json",
                    "bullets": [
                        r"• LoRA Rank $r = 16, \alpha = 16$",
                        "• All 7 Linear Modules",
                        "• 41.94M Trainable Params",
                        "• Loss: 2.702 -> 0.0031 (-99.9%)",
                        "• Fast Unsloth Inference"
                    ],
                    "y": 42.4, "h": 17.2
                },
                {
                    "title": "Live Strategy Synthesis",
                    "bullets": [
                        "• Tactical Undercut Timing",
                        "• Thermal Cliff Alerts (>130°C)",
                        "• Tire Deg Slope Benchmarks",
                        "• Monte Carlo Win Probabilities"
                    ],
                    "y": 23.6, "h": 17.4
                },
                {
                    "title": "Publication Brief",
                    "bullets": [
                        "• Structured Strategy Memo",
                        "• Empirical Audit Trail",
                        "• Engineering Tradeoff Matrix"
                    ],
                    "y": 4.8, "h": 17.4
                }
            ]
        }
    ]

    # Draw Each Tier Container
    for t in tiers:
        # Tier Background Box
        tier_box = FancyBboxPatch(
            (t["x"], t["y"]), t["w"], t["h"],
            boxstyle="round,pad=0.2,rounding_size=1.2",
            facecolor=t["bg"], edgecolor=t["border"],
            linewidth=1.8, linestyle="-", zorder=1
        )
        ax.add_patch(tier_box)

        # Tier Header Banner
        header_h = 6.4
        header_y = t["y"] + t["h"] - header_h - 0.4
        header_box = FancyBboxPatch(
            (t["x"] + 0.3, header_y), t["w"] - 0.6, header_h,
            boxstyle="round,pad=0.1,rounding_size=0.8",
            facecolor=t["border"], edgecolor="none", zorder=2
        )
        ax.add_patch(header_box)

        ax.text(
            t["x"] + t["w"] / 2.0, header_y + header_h - 2.1,
            t["name"],
            ha="center", va="center", fontsize=8.8, weight="bold", color="white",
            family="sans-serif", zorder=3
        )
        ax.text(
            t["x"] + t["w"] / 2.0, header_y + 1.8,
            t["subtitle"],
            ha="center", va="center", fontsize=7.2, color="#f1f5f9",
            family="sans-serif", zorder=3
        )

        # Badge pill clearly separated below header
        badge_y = header_y - 4.0
        badge_h = 2.8
        badge_box = FancyBboxPatch(
            (t["x"] + 1.0, badge_y), t["w"] - 2.0, badge_h,
            boxstyle="round,pad=0.1,rounding_size=0.6",
            facecolor="#ffffff", edgecolor=t["border"],
            linewidth=1.2, zorder=2
        )
        ax.add_patch(badge_box)
        ax.text(
            t["x"] + t["w"] / 2.0, badge_y + badge_h / 2.0,
            t["badge"],
            ha="center", va="center", fontsize=7.2, weight="bold", color=t["border"],
            family="sans-serif", zorder=3
        )

        # Draw Cards inside Tier
        for card in t["cards"]:
            card_box = FancyBboxPatch(
                (t["x"] + 0.6, card["y"]), t["w"] - 1.2, card["h"],
                boxstyle="round,pad=0.15,rounding_size=0.8",
                facecolor=card_bg, edgecolor="#cbd5e1",
                linewidth=1.0, zorder=2
            )
            ax.add_patch(card_box)

            # Card Title
            card_top = card["y"] + card["h"]
            ax.text(
                t["x"] + 1.1, card_top - 1.8,
                card["title"],
                ha="left", va="center", fontsize=8.2, weight="bold", color=text_dark,
                family="sans-serif", zorder=3
            )

            # Optional Code/File tag
            bullet_y_start = card_top - 3.8
            if "code" in card:
                code_str = card["code"]
                badge_w = min(len(code_str) * 0.44 + 1.2, t["w"] - 2.6)
                code_badge = FancyBboxPatch(
                    (t["x"] + 1.1, card_top - 4.4), badge_w, 1.9,
                    boxstyle="round,pad=0.06,rounding_size=0.3",
                    facecolor="#f1f5f9", edgecolor="#94a3b8",
                    linewidth=0.7, zorder=3
                )
                ax.add_patch(code_badge)
                ax.text(
                    t["x"] + 1.6, card_top - 3.45,
                    code_str,
                    ha="left", va="center", fontsize=6.8, color="#0f172a",
                    family="monospace", weight="bold", zorder=4
                )
                bullet_y_start = card_top - 6.2

            # Card Bullets: precisely calculate spacing so lines never collide or exceed box
            n_bullets = len(card["bullets"])
            available_h = bullet_y_start - (card["y"] + 0.8)
            spacing = available_h / max(n_bullets, 1)
            spacing = min(spacing, 2.3)
            bullet_font_size = 6.9 if n_bullets <= 4 else 6.3

            for idx, bullet in enumerate(card["bullets"]):
                by = bullet_y_start - (idx * spacing) - 0.6
                ax.text(
                    t["x"] + 1.1, by,
                    bullet,
                    ha="left", va="center", fontsize=bullet_font_size, color=text_muted,
                    family="sans-serif", zorder=3
                )

    # Inter-Tier Connecting Arrows
    arrow_props = dict(
        arrowstyle="-|>", mutation_scale=16,
        linewidth=2.2, color="#0284c7", zorder=10
    )

    inter_arrows = [
        # Tier 1 to Tier 2
        (19.4, 51.0, 21.2, 51.0, "Filtered SQL\nDataset"),
        # Tier 2 to Tier 3
        (39.2, 51.0, 41.0, 51.0, "Engineered\nFeatures"),
        # Tier 3 to Tier 4
        (59.4, 51.0, 61.2, 51.0, "Model Inference\n& Probs"),
        # Tier 4 to Tier 5
        (79.4, 51.0, 81.2, 51.0, "Domain Guidance\n& Dynamic SQL"),
    ]

    for x1, y1, x2, y2, label in inter_arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), **arrow_props)
        ax.add_patch(arrow)
        ax.text(
            (x1 + x2) / 2.0, y1 + 3.2,
            label,
            ha="center", va="center", fontsize=6.4, weight="bold", color="#0369a1",
            family="sans-serif", zorder=11,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", edgecolor="#bae6fd", alpha=0.95)
        )

    # Bi-directional feedback loop arrow from Tier 4 Sandbox back to Tier 3 ML
    feedback_arrow = FancyArrowPatch(
        (61.4, 25.0), (59.2, 25.0),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.8, color="#7c3aed", zorder=10, linestyle="--"
    )
    ax.add_patch(feedback_arrow)
    ax.text(
        60.3, 27.4, "Re-Query\nPrediction",
        ha="center", va="center", fontsize=5.8, weight="bold", color="#6d28d9",
        family="sans-serif", zorder=11,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="#ddd6fe", alpha=0.95)
    )

    # Save PDF and PNG
    pdf_path = os.path.join(output_dir, "fig1_system_architecture.pdf")
    png_path = os.path.join(output_dir, "fig1_system_architecture.png")

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"Generated Figure 1 (Vector PDF): {pdf_path}")
    print(f"Generated Figure 1 (Raster PNG): {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    create_figure()
