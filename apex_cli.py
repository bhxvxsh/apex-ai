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

console = Console()
sandbox = DynamicF1AnalystSandbox()

COMMANDS = [
    "/predict",
    "/standings",
    "/consistent",
    "/career",
    "/deltas",
    "/exec",
    "/models",
    "/help",
    "/clear",
    "/exit",
    "/quit",
]


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

    table.add_row("/predict", "<DRIVER> <GRID>", "Predict finish, podium probability & points odds using XGBoost (e.g. /predict VER 3)")
    table.add_row("/standings", "[YEAR]", "Display championship table for 2023, 2024, or 2025 (e.g. /standings 2023)")
    table.add_row("/consistent", "[TOP_N]", "Rank drivers by position variance / consistency stddev across 2023-2025")
    table.add_row("/career", "<DRIVER>", "Multi-season win, podium, and points breakdown (e.g. /career HAM)")
    table.add_row("/deltas", "<DRIVER> [YEAR]", "Saturday qualifying to Sunday race net position changes (e.g. /deltas NOR 2024)")
    table.add_row("/exec", "<PYTHON CODE>", "Run arbitrary custom code dynamically in sandboxed runtime with live f1.db")
    table.add_row("/models", "", "Show performance metrics and status of trained Phase 4 ML models")
    table.add_row("/clear", "", "Clear terminal screen")
    table.add_row("/help", "", "Show this commands reference")
    table.add_row("/exit", "", "Quit APEX CLI")

    console.print(table)
    console.print("\n[dim]You can also type natural queries directly: e.g. [italic]\"Predict race for Norris starting P2\"[/italic] or [italic]\"Who won in 2024?\"[/italic][/dim]\n")


def handle_predict(args: str):
    parts = args.strip().split()
    if len(parts) < 2:
        console.print("[bold red]Usage:[/bold red] /predict <DRIVER_CODE> <GRID_POS> (e.g., [cyan]/predict VER 2[/cyan])")
        return
    driver = parts[0].upper()
    try:
        grid = int(parts[1])
    except ValueError:
        console.print("[bold red]Error:[/bold red] Grid position must be an integer (1 to 20).")
        return

    with console.status(f"[bold green]Running XGBoost Inference for {driver} starting P{grid}...", spinner="dots"):
        time.sleep(0.3)
        res_raw = execute_tool("predict_race", {"driver_id": driver, "grid_position": grid})
        import json
        data = json.loads(res_raw)

    table = Table(title=f"Race Prediction: {driver} (Grid: P{grid})", border_style="green", header_style="bold green")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Prediction", style="bold white")
    table.add_column("Statistical Confidence", style="dim")

    table.add_row("Expected Finish", str(data.get("predicted_finish_position")), "MAE: 2.3 positions")
    table.add_row("Podium Probability (Top 3)", str(data.get("podium_probability")), "ROC-AUC: 94.0% | Acc: 91.5%")
    table.add_row("Points Probability (Top 10)", str(data.get("points_probability")), "ROC-AUC: 80.6%")
    table.add_row("Starting Slot", f"P{grid}", "Normalized pole delta applied")

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


def handle_natural_query(user_query: str):
    """Fallback natural-language router."""
    from ask_apex_cli import ask_apex
    with console.status("[bold cyan]Analyzing query & routing to APEX AI Engine...", spinner="dots"):
        time.sleep(0.3)
        response = ask_apex(user_query)
    console.print(Panel(response, title="APEX AI Response", border_style="cyan"))


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
