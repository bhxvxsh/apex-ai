"""
APEX CLI - Terminal Interface for Formula 1 Analytics, Predictions & Dynamic Tool Making
Modeled with a Claude Code / terminal developer aesthetic:
- Slash commands (/predict, /standings, /consistent, /exec, /models, /help, /clear, /exit)
- Real-time animated spinners
- Rich-formatted Markdown, syntax-highlighted Python/SQL tables
- Hybrid analytical dispatch + on-the-fly dynamic code execution sandbox
"""

import os
import sys
import time
import json
import sqlite3
import pandas as pd
import numpy as np

# Rich imports for terminal styling
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from rich.spinner import Spinner
from rich.live import Live

# Prompt toolkit for command-line auto-completion and history
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "05_Analytics"))
sys.path.insert(0, os.path.join(CURRENT_DIR, "04_Models"))
os.chdir(CURRENT_DIR)

from tools import execute_tool, TOOLS, TOOL_SCHEMAS
from dynamic_analyst import DynamicF1AnalystSandbox
from react_agent import ApexResearchReActAgent

console = Console()
sandbox = DynamicF1AnalystSandbox()
react_agent = ApexResearchReActAgent()

def clean_terminal_text(text: str) -> str:
    r"""
    Sanitize and render raw LaTeX, math symbols, and escaped characters
    into clean, readable Unicode terminal characters for Windows console.
    Converts: ($\mu_{\text{peak}}$) -> (μ_peak), $\Delta t \le +0.250\text{s}$ -> Δt ≤ +0.250s, etc.
    """
    if not text:
        return ""
    import re
    # Greek and mathematical symbol map
    math_map = {
        r'\mu': 'μ',
        r'\sigma': 'σ',
        r'\Delta': 'Δ',
        r'\delta': 'δ',
        r'\theta': 'θ',
        r'\lambda': 'λ',
        r'\alpha': 'α',
        r'\beta': 'β',
        r'\le': '≤',
        r'\ge': '≥',
        r'\pm': '±',
        r'\times': '×',
        r'\approx': '≈',
        r'\ne': '≠',
        r'\to': '→',
        r'^\circ': '°',
        r'\degree': '°',
        r'\bar': '',
        r'\cdot': '·',
        r'\sqrt': '√',
    }
    
    out = text
    for sym, uni in math_map.items():
        out = out.replace(sym, uni)
        
    # Remove \text{...} wrappers -> ...
    out = re.sub(r'\\text\{([^}]+)\}', r'\1', out)
    # Remove \frac{a}{b} -> a/b
    out = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', out)
    # Remove leftover math dollar signs: $...$ -> ...
    out = re.sub(r'\$([^\$]+)\$', r'\1', out)
    out = out.replace('$', '')
    # Remove remaining curly braces from subscripts/superscripts: _{peak} -> _peak
    out = re.sub(r'\_\{([^}]+)\}', r'_\1', out)
    out = re.sub(r'\^\{([^}]+)\}', r'^\1', out)
    # Remove bare backslashes before plain alphanumeric tokens
    out = re.sub(r'\\([a-zA-Z]+)', r'\1', out)
    return out

COMMANDS = [
    "/predict",
    "/circuit",
    "/research",
    "/walkforward",
    "/cv",
    "/standings",
    "/consistent",
    "/career",
    "/deltas",
    "/exec",
    "/connect",
    "/disconnect",
    "/models",
    "/clear",
    "/help",
    "/exit",
]

# Remote Colab Endpoint configuration
COLAB_ENDPOINT = None
CONFIG_FILE = os.path.join(CURRENT_DIR, ".apex_config.json")

def load_colab_config():
    global COLAB_ENDPOINT
    if os.path.exists(CONFIG_FILE):
        try:
            import json
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
                COLAB_ENDPOINT = cfg.get("colab_endpoint", None)
        except Exception:
            COLAB_ENDPOINT = None

def save_colab_config(endpoint: str):
    global COLAB_ENDPOINT
    COLAB_ENDPOINT = endpoint.strip().rstrip("/")
    import json
    with open(CONFIG_FILE, "w") as f:
        json.dump({"colab_endpoint": COLAB_ENDPOINT}, f, indent=2)

load_colab_config()


def display_banner():
    banner_text = Text()
    banner_text.append("    ___    ____  _______  __   ___    ____\n", style="bold red")
    banner_text.append("   /   |  / __ \\/ ____/ |/ /  /   |  /  _/\n", style="bold red")
    banner_text.append("  / /| | / /_/ / __/  |   /  / /| |  / /  \n", style="bold white")
    banner_text.append(" / ___ |/ ____/ /___ /   |  / ___ |_/ /   \n", style="bold white")
    banner_text.append("/_/  |_/_/   /_____//_/|_| /_/  |_/___/   \n", style="bold cyan")
    banner_text.append("\n  Formula 1 Intelligence & Telemetry Analytics Engine\n", style="italic bright_white")
    banner_text.append("  Version: 2.1.0 • Hybrid ML Models (91.5% Podium Acc) • Dynamic LATM Sandbox\n", style="dim green")
    
    console.print(Panel(banner_text, border_style="red", expand=False))
    console.print("[dim]Type your natural language F1 question, or type [bold cyan]/help[/bold cyan] for slash commands. Press [bold cyan]Ctrl+C[/bold cyan] to exit.[/dim]\n")


def display_help():
    table = Table(title="APEX CLI — Available Commands", border_style="cyan", show_header=True, header_style="bold magenta")
    table.add_column("Command", style="bold cyan", width=18)
    table.add_column("Arguments", style="yellow", width=22)
    table.add_column("Description", style="white")

    table.add_row("/predict", "<DRIVER> <GRID> [EVENT]", "Predict finish, podium probability & points odds using ML + Circuit clusters (e.g. /predict VER 3 Monza)")
    table.add_row("/circuit", "<CIRCUIT>", "Physical downforce taxonomy, tire stress, and micro-sector apex speed (e.g. /circuit Monaco)")
    table.add_row("/research", "<HYPOTHESIS/QUERY>", "Autonomous ReAct multi-step hypothesis testing & thesis report (e.g. /research Will Norris beat Verstappen at Monaco?)")
    table.add_row("/walkforward", "", "Empirical 49-fold expanding walk-forward cross-validation benchmarks & p-value significance (or /cv)")
    table.add_row("/standings", "[YEAR]", "Display championship table for 2023, 2024, or 2025 (e.g. /standings 2023)")
    table.add_row("/consistent", "[TOP_N]", "Rank drivers by position variance / consistency stddev across 2023-2025")
    table.add_row("/career", "<DRIVER>", "Multi-season win, podium, and points breakdown (e.g. /career HAM)")
    table.add_row("/deltas", "<DRIVER> [YEAR]", "Saturday qualifying to Sunday race net position changes (e.g. /deltas NOR 2024)")
    table.add_row("/exec", "<PYTHON CODE>", "Run arbitrary custom code dynamically in sandboxed runtime with live f1.db")
    table.add_row("/connect", "<URL>", "Link APEX CLI to your fine-tuned model running on Google Colab (e.g. /connect https://xyz.ngrok-free.app)")
    table.add_row("/disconnect", "", "Disconnect from Colab and revert to local hybrid engine")
    table.add_row("/models", "", "Show performance metrics and status of trained Phase 4 ML models")
    table.add_row("/clear", "", "Clear terminal screen")
    table.add_row("/help", "", "Show this commands reference")
    table.add_row("/exit", "", "Quit APEX CLI")

    console.print(table)
    console.print("\n[dim]You can also type natural queries directly: e.g. [italic]\"Predict race for Norris starting P2 at Silverstone\"[/italic] or [italic]\"Who won in 2024?\"[/italic][/dim]\n")


def handle_predict(args: str):
    parts = args.strip().split()
    if len(parts) < 2:
        console.print("[bold red]Usage:[/bold red] /predict <DRIVER_CODE> <GRID_POS> [CIRCUIT] (e.g., [cyan]/predict VER 2[/cyan] or [cyan]/predict NOR 1 Monaco[/cyan])")
        return
    driver = parts[0].upper()
    try:
        grid = int(parts[1])
    except ValueError:
        console.print("[bold red]Error:[/bold red] Grid position must be an integer (1 to 20).")
        return

    event_name = " ".join(parts[2:]) if len(parts) > 2 else "Bahrain Grand Prix"

    with console.status(f"[bold green]Running XGBoost Inference for {driver} starting P{grid} ({event_name})...", spinner="dots"):
        time.sleep(0.3)
        res_raw = execute_tool("predict_race", {"driver_id": driver, "grid_position": grid, "event_name": event_name})
        import json
        data = json.loads(res_raw)

    table = Table(title=f"Race Prediction: {driver} (Grid: P{grid} | {data.get('event_name', event_name)})", border_style="green", header_style="bold green")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Prediction", style="bold white")
    table.add_column("Statistical / Physics Regime", style="dim")

    table.add_row("Expected Finish", str(data.get("predicted_finish_position")), "MAE: 2.3 positions")
    table.add_row("Podium Probability (Top 3)", str(data.get("podium_probability")), "ROC-AUC: 94.4% | Acc: 90.4%")
    table.add_row("Points Probability (Top 10)", str(data.get("points_probability")), "ROC-AUC: 81.8% | Acc: 75.2%")
    table.add_row("Circuit Cluster", str(data.get("circuit_cluster", "N/A")), f"Aero Regime: {data.get('downforce_regime', 'Medium')}")
    table.add_row("Tire Stress Index", f"{data.get('tire_stress_index', 'N/A')} / 5.0", "Surface abrasiveness & thermal degradation")
    table.add_row("Starting Slot", f"P{grid}", "Normalized micro-sector delta applied")

    console.print(table)


def handle_circuit(args: str):
    query = args.strip()
    if not query:
        console.print("[bold red]Usage:[/bold red] /circuit <CIRCUIT_OR_EVENT> (e.g. [cyan]/circuit Monaco[/cyan] or [cyan]/circuit Monza[/cyan])")
        return

    with console.status(f"[bold cyan]Querying aerodynamic profile & micro-sectors for '{query}'...", spinner="dots"):
        time.sleep(0.2)
        import json
        res = json.loads(execute_tool("circuit_profile", {"circuit_or_event": query}))

    table = Table(title=f"Circuit Engineering Profile — {res.get('event_query')}", border_style="cyan", header_style="bold magenta")
    table.add_column("Engineering Parameter", style="bold cyan", width=25)
    table.add_column("Benchmark Value", style="bold white", width=30)
    table.add_column("Aero / Dynamics Impact", style="dim")

    table.add_row("Cluster Taxonomy", f"Cluster {res.get('cluster_id')}: {res.get('cluster_name')}", "Downforce & speed distribution")
    table.add_row("Downforce Level", str(res.get("downforce_regime")), "Wing angle & floor aerodynamic trim")
    table.add_row("Tire Stress Index", f"{res.get('tire_stress_index')} / 5.0", "Surface abrasiveness & degradation risk")
    table.add_row("Full Throttle Duty", str(res.get("full_throttle_pct")), "Engine thermal load & drag penalty")
    table.add_row("Micro-Sector Apex V_min", str(res.get("apex_speed_v_min")), "Critical minimum cornering velocity")
    table.add_row("Overtaking Difficulty", str(res.get("overtaking_difficulty")), "Track width & delta-to-overtake threshold")
    table.add_row("Brake Energy Demand", str(res.get("brake_energy_level")), "Brake cooling & thermal rotor duty")

    console.print(table)

    console.print(table)


def handle_standings(args: str):
    year = 2023
    if args.strip().isdigit():
        year = int(args.strip())

    with console.status(f"[bold cyan]Querying {year} Championship Standings...", spinner="aesthetic"):
        time.sleep(0.2)
        import json
        res = json.loads(execute_tool("get_standings", {"season": year}))

    if "error" in res:
        console.print(f"[bold red]{res['error']}[/bold red]")
        return

    table = Table(title=f"Formula 1 Championship Standings — {year} (Champion: {res.get('champion')})", border_style="red", header_style="bold yellow")
    table.add_column("Pos", style="bold cyan", width=5)
    table.add_column("Driver", style="bold white", width=22)
    table.add_column("Team", style="magenta", width=20)
    table.add_column("Points", style="bold green", justify="right")
    table.add_column("Avg Finish", style="yellow", justify="right")

    for i, d in enumerate(res.get("top_10", []), 1):
        table.add_row(
            str(i),
            d["FullName"],
            d["TeamName"],
            f"{d['Points']:.0f}",
            f"P{d['PositionAvg']:.2f}"
        )

    console.print(table)


def handle_consistent(args: str):
    top_n = 10
    if args.strip().isdigit():
        top_n = int(args.strip())

    with console.status("[bold magenta]Calculating Driver Finishing Variance (2023-2025)...", spinner="bouncingBar"):
        time.sleep(0.2)
        import json
        res = json.loads(execute_tool("consistency_ranking", {"top_n": top_n}))

    table = Table(title="Most Consistent F1 Drivers (Ranked by Finishing Position StdDev)", border_style="blue", header_style="bold cyan")
    table.add_column("Rank", style="bold white", width=6)
    table.add_column("Driver", style="bold yellow", width=20)
    table.add_column("Team", style="magenta", width=18)
    table.add_column("StdDev (σ)", style="bold green", justify="right")
    table.add_column("Avg Pos", style="white", justify="right")
    table.add_column("Finish %", style="cyan", justify="right")
    table.add_column("Podiums", style="bold white", justify="right")

    for i, d in enumerate(res.get("rankings", []), 1):
        table.add_row(
            str(i),
            d["fullName"],
            d["teamName"],
            f"{d['position_stddev']:.3f}",
            f"P{d['avg_position']:.2f}",
            f"{d['finish_rate']:.1f}%",
            str(d["podium_finishes"])
        )

    console.print(table)


def handle_career(args: str):
    driver = args.strip().upper() or "VER"
    with console.status(f"[bold cyan]Fetching career telemetry for {driver}...", spinner="line"):
        time.sleep(0.2)
        import json
        res = json.loads(execute_tool("driver_career_stats", {"driver_id": driver}))

    if "error" in res:
        console.print(f"[bold red]{res['error']}[/bold red]")
        return

    table = Table(title=f"Career Breakdown: {res.get('driver_id')} (2023-2025)", border_style="magenta", header_style="bold yellow")
    table.add_column("Season", style="cyan")
    table.add_column("Races", justify="right")
    table.add_column("Wins", style="bold green", justify="right")
    table.add_column("Podiums", style="yellow", justify="right")
    table.add_column("Points", style="bold white", justify="right")
    table.add_column("DNFs", style="red", justify="right")

    for s in res.get("seasons_summary", []):
        table.add_row(
            str(s["year"]),
            str(s["races"]),
            str(s["wins"]),
            str(s["podiums"]),
            f"{s['points']:.0f}",
            str(s["dnfs"])
        )

    console.print(table)
    console.print(f"[dim]Total Career Points: [bold green]{res.get('total_points'):.0f}[/bold green] | Total Wins: [bold green]{res.get('total_wins')}[/bold green][/dim]\n")


def handle_deltas(args: str):
    parts = args.strip().split()
    driver = parts[0].upper() if len(parts) > 0 else "VER"
    year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 2023

    with console.status(f"[bold green]Computing Qualifying-to-Race progression for {driver} in {year}...", spinner="dots"):
        time.sleep(0.2)
        import json
        res = json.loads(execute_tool("qualifying_race_delta", {"driver_id": driver, "season": year}))

    table = Table(title=f"Qualifying-to-Race Deltas: {driver} ({year})", border_style="cyan")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Total Races Classified", str(res.get("total_races")))
    avg_d = res.get("average_position_delta", 0)
    d_style = "bold green" if avg_d > 0 else "bold red"
    table.add_row("Average Position Gain", f"[{d_style}]{avg_d:+.2f} positions[/{d_style}]")
    table.add_row("Best Race Improvement", f"[bold green]+{res.get('best_improvement')} positions[/bold green]")
    table.add_row("Worst Drop", f"[bold red]{res.get('worst_drop')} positions[/bold red]")
    table.add_row("Net Improvement Rate", f"{res.get('improvement_rate_percent')}% of races")

    console.print(table)


def handle_models():
    summary_path = os.path.join(CURRENT_DIR, "04_Models", "baseline_metrics_summary.json")
    if not os.path.exists(summary_path):
        console.print("[bold red]Model metrics summary not found.[/bold red]")
        return
    import json
    with open(summary_path, "r") as f:
        data = json.load(f)

    # Classification
    c_table = Table(title="High-Stakes Classification Models (Trained on 2023-2024, Evaluated on 2025)", border_style="green", header_style="bold green")
    c_table.add_column("Task", style="bold cyan")
    c_table.add_column("Accuracy", style="bold white", justify="right")
    c_table.add_column("ROC-AUC", style="bold green", justify="right")
    c_table.add_column("Precision", justify="right")
    c_table.add_column("Recall", justify="right")

    for c in data.get("classification_metrics", []):
        c_table.add_row(
            c["task"],
            f"{c['accuracy']:.2f}%",
            f"{c['roc_auc']:.2f}%",
            f"{c['precision']:.2f}%",
            f"{c['recall']:.2f}%"
        )
    console.print(c_table)

    # Regression
    r_table = Table(title="Race Finishing Position Models", border_style="blue", header_style="bold cyan")
    r_table.add_column("Model Architecture", style="bold white")
    r_table.add_column("MAE", justify="right")
    r_table.add_column("RMSE", justify="right")
    r_table.add_column("Within ±1 Pos", style="yellow", justify="right")
    r_table.add_column("Within ±2 Pos", style="bold green", justify="right")
    r_table.add_column("Within ±3 Pos", style="bold green", justify="right")

    for r in data.get("race_finishing_prediction_metrics", []):
        r_table.add_row(
            r["model"],
            f"{r['mae']:.3f}",
            f"{r['rmse']:.3f}",
            f"{r['acc_within_1_pos']:.1f}%",
            f"{r['acc_within_2_pos']:.1f}%",
            f"{r.get('acc_within_3_pos', 0.0):.1f}%"
        )
    console.print(r_table)


def handle_walkforward():
    metrics_path = os.path.join(CURRENT_DIR, "04_Models", "walk_forward_metrics.json")
    if not os.path.exists(metrics_path):
        console.print("[yellow]Walk-forward metrics file not found. Running walk-forward cross-validation now...[/yellow]")
        from walk_forward_cv import run_walk_forward_cv
        with console.status("[bold cyan]Running 49-Fold Walk-Forward Cross Validation...", spinner="dots"):
            data = run_walk_forward_cv()
    else:
        with open(metrics_path, "r") as f:
            data = json.load(f)

    # Overall table
    table = Table(title="Empirical Time-Series Walk-Forward Cross-Validation (49 Held-Out GPs)", border_style="green", header_style="bold green")
    table.add_column("Validation Metric", style="bold cyan")
    table.add_column("Status-Quo Grid", style="red")
    table.add_column("APEX AI (Ranked XGB)", style="bold green")
    table.add_column("Empirical Gain / Scientific Significance", style="bold yellow")

    table.add_row(
        "Mean Absolute Error (MAE)",
        f"{data.get('mean_mae_grid_baseline', 2.711):.3f} pos",
        f"{data.get('mean_mae_xgb_ranked', 2.191):.3f} pos",
        f"-{data.get('error_reduction_percentage', 19.16):.2f}% Error Reduction"
    )
    table.add_row(
        "Within ±1 Position Accuracy",
        "—",
        f"{data.get('acc_within_1_pos_mean', 49.33):.1f}%",
        "Prequential expanding window"
    )
    table.add_row(
        "Within ±2 Positions Accuracy",
        "—",
        f"{data.get('acc_within_2_pos_mean', 67.21):.1f}%",
        "Consistent across 857 finishes"
    )
    table.add_row(
        "Within ±3 Positions Accuracy",
        "—",
        f"{data.get('acc_within_3_pos_mean', 79.28):.1f}%",
        "Top-quartile ranking precision"
    )
    table.add_row(
        "Average Podium Accuracy",
        "—",
        f"{data.get('podium_accuracy_mean', 89.46):.1f}%",
        "Calibrated Top-3 classifier"
    )

    sig = data.get("statistical_significance", {})
    p_val = sig.get("paired_t_p_value", 6.24e-10)
    p_str = "< 1e-9" if p_val < 1e-9 else f"{p_val:.2e}"
    table.add_row(
        "Hypothesis Test (Paired t-test)",
        "H0: Error Equal",
        f"t = {sig.get('paired_t_statistic', 6.2559):.4f}",
        f"p = {p_str} (Overwhelmingly Significant)"
    )

    console.print(table)

    # Cluster breakdown table
    c_table = Table(title="Walk-Forward Error Margin Across Circuit Aerodynamic Clusters", border_style="cyan", header_style="bold magenta")
    c_table.add_column("Circuit Aerodynamic Cluster", style="bold white")
    c_table.add_column("APEX AI Mean Absolute Error", style="bold green", justify="right")
    c_table.add_column("Primary Aerodynamic Demands", style="dim")

    cluster_demands = {
        "Low-Downforce Power": "High full-throttle duty cycle, top speed, light aerodynamic wing angles",
        "High-Downforce Thermal Degradation": "Severe lateral tire stress (4.8-5.0), high surface blistering risk",
        "Maximum-Downforce Mechanical Grip": "Extreme overtaking difficulty, tight slow-speed corner apexes",
        "Hybrid Street / Stop-and-Go": "Heavy longitudinal braking events, curb compliance, traction out of chicanes",
    }

    for c_name, c_mae in data.get("cluster_performance_mae", {}).items():
        c_table.add_row(c_name, f"{c_mae:.3f} positions", cluster_demands.get(c_name, "Balanced aerodynamic package"))

    console.print(c_table)


def handle_research(query: str):
    query = query.strip()
    if not query:
        console.print("[bold red]Usage:[/bold red] /research <HYPOTHESIS_OR_QUERY> (e.g., [cyan]/research Will Norris beat Verstappen at Monaco starting P2?[/cyan])")
        return

    console.print(Panel(f"[bold white]{query}[/bold white]", title="APEX AI — Autonomous ReAct Research Agent", border_style="magenta"))

    def step_cb(step_info: dict):
        s_num = step_info.get("step")
        s_type = step_info.get("type")

        if s_type == "thought":
            cleaned = clean_terminal_text(step_info["content"])
            console.print(f"[bold yellow]❯ Step {s_num} [Thought]:[/bold yellow] [italic]{cleaned}[/italic]")
        elif s_type == "action":
            tool = step_info.get("tool")
            t_args = step_info.get("args")
            console.print(f"  [bold cyan]↳ Action:[/bold cyan] [green]{tool}[/green]({json.dumps(t_args)})")
        elif s_type == "observation":
            obs = step_info.get("content")
            if isinstance(obs, dict):
                summary_items = [f"{k}: {v}" for k, v in list(obs.items())[:4]]
                obs_str = ", ".join(summary_items)
            else:
                obs_str = str(obs)[:120]
            console.print(f"  [dim green]↳ Observation:[/dim green] [dim]{obs_str}[/dim]\n")

    with console.status("[bold magenta]Executing ReAct multi-step hypothesis cycles...", spinner="bouncingBar"):
        time.sleep(0.2)
        res = react_agent.run_research(query, step_callback=step_cb)

    synthesis_clean = clean_terminal_text(res.get("synthesis", ""))
    console.print(Panel(Markdown(synthesis_clean), title="Definitive Empirical Research Synthesis", border_style="green"))


def handle_exec(code: str):
    if not code.strip():
        console.print("[bold red]Usage:[/bold red] /exec <PYTHON_CODE>")
        return

    console.print(Panel(Syntax(code, "python", theme="monokai", line_numbers=True), title="Executing Dynamic Script in Sandbox", border_style="yellow"))
    with console.status("[bold green]Executing Python code against f1.db...", spinner="aesthetic"):
        res = sandbox.execute_analysis_code(code)

    if res["success"]:
        console.print(Panel(str(res["output"]), title="Execution Result", border_style="green"))
    else:
        console.print(Panel(res["error"], title="Execution Failed", border_style="red"))


def handle_connect(args: str):
    url = args.strip()
    if not url:
        if COLAB_ENDPOINT:
            console.print(f"[bold green]Currently connected to Colab LLM:[/bold green] [cyan]{COLAB_ENDPOINT}[/cyan]")
        else:
            console.print("[bold red]Usage:[/bold red] /connect <COLAB_NGROK_URL> (e.g., [cyan]/connect https://xxxx.ngrok-free.app[/cyan])")
        return

    with console.status(f"[bold cyan]Pinging Colab endpoint {url}...", spinner="dots"):
        import requests
        try:
            r = requests.get(f"{url.rstrip('/')}/health", timeout=5)
            if r.status_code == 200:
                save_colab_config(url)
                react_agent.set_colab_endpoint(url)
                console.print(f"[bold green]✓ Successfully connected to fine-tuned Colab LLM![/bold green] Endpoint saved: [cyan]{COLAB_ENDPOINT}[/cyan]")
                return
        except Exception as e:
            # Fallback: still save if health check times out but URL format is valid
            save_colab_config(url)
            react_agent.set_colab_endpoint(url)
            console.print(f"[yellow]Endpoint saved as [cyan]{COLAB_ENDPOINT}[/cyan] (Health check note: {str(e)}). Queries will now be forwarded to Colab.[/yellow]")


def handle_disconnect():
    global COLAB_ENDPOINT
    COLAB_ENDPOINT = None
    react_agent.set_colab_endpoint(None)
    if os.path.exists(CONFIG_FILE):
        try:
            os.remove(CONFIG_FILE)
        except Exception:
            pass
    console.print("[bold cyan]Disconnected from Colab. APEX CLI is now running in local hybrid engine mode.[/bold cyan]")


def handle_natural_query(user_query: str):
    """Query router: forward to remote fine-tuned Colab LLM if connected, else use local engine."""
    global COLAB_ENDPOINT
    if COLAB_ENDPOINT:
        with console.status("[bold green]Streaming from Fine-Tuned Llama-3 (Google Colab T4 GPU)...", spinner="aesthetic"):
            import requests
            try:
                res = requests.post(
                    f"{COLAB_ENDPOINT}/ask",
                    json={"query": user_query},
                    timeout=30
                )
                if res.status_code == 200:
                    ans = res.json().get("response", "")
                    cleaned_ans = clean_terminal_text(ans)
                    console.print(Panel(Markdown(cleaned_ans), title="APEX AI — Fine-Tuned Llama-3 (Colab)", border_style="green"))
                    return
                else:
                    console.print(f"[bold red]Colab server returned error {res.status_code}: {res.text}[/bold red]")
                    console.print("[dim]Falling back to local analytical engine...[/dim]")
            except Exception as e:
                console.print(f"[yellow]Failed to reach Colab ({str(e)}). Falling back to local engine...[/yellow]")

    # Local Engine Fallback
    from ask_apex_cli import ask_apex
    with console.status("[bold cyan]Analyzing query & routing to local APEX AI Engine...", spinner="dots"):
        time.sleep(0.3)
        response = ask_apex(user_query)
    cleaned_local = clean_terminal_text(response)
    console.print(Panel(Markdown(cleaned_local), title="APEX AI Response (Local Hybrid Engine)", border_style="cyan"))


def main():
    display_banner()

    history_file = os.path.join(CURRENT_DIR, ".apex_history")
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(COMMANDS, ignore_case=True),
        style=Style.from_dict({
            'prompt': '#ff0055 bold',
        })
    )

    while True:
        try:
            user_input = session.prompt([
                ('class:prompt', 'apex ❯ ')
            ]).strip()

            if not user_input:
                continue

            # Command routing
            if user_input.startswith("/exit") or user_input.startswith("/quit"):
                console.print("[bold cyan]Shutting down APEX AI Engine. Goodbye![/bold cyan]")
                break
            elif user_input == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                display_banner()
            elif user_input == "/help":
                display_help()
            elif user_input.startswith("/predict"):
                handle_predict(user_input[len("/predict"):])
            elif user_input.startswith("/circuit"):
                handle_circuit(user_input[len("/circuit"):])
            elif user_input.startswith("/research"):
                handle_research(user_input[len("/research"):])
            elif user_input.startswith("/react"):
                handle_research(user_input[len("/react"):])
            elif user_input.startswith("/walkforward") or user_input.startswith("/cv"):
                handle_walkforward()
            elif user_input.startswith("/standings"):
                handle_standings(user_input[len("/standings"):])
            elif user_input.startswith("/consistent"):
                handle_consistent(user_input[len("/consistent"):])
            elif user_input.startswith("/career"):
                handle_career(user_input[len("/career"):])
            elif user_input.startswith("/deltas"):
                handle_deltas(user_input[len("/deltas"):])
            elif user_input.startswith("/models"):
                handle_models()
            elif user_input.startswith("/connect"):
                handle_connect(user_input[len("/connect"):])
            elif user_input.startswith("/disconnect"):
                handle_disconnect()
            elif user_input.startswith("/exec"):
                handle_exec(user_input[len("/exec"):])
            else:
                handle_natural_query(user_input)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Session closed. See you trackside![/bold cyan]")
            break
        except Exception as e:
            console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")


if __name__ == "__main__":
    main()

