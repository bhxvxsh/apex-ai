"""
APEX AI - Dynamic Tool Maker & Code Execution Sandbox

Allows the LLM to autonomously write and execute Python analytics scripts on-the-fly
against f1.db and ML models to solve novel, complex research questions.
"""

import os
import sys
import io
import contextlib
import traceback
import sqlite3
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DB_PATH = os.path.join(BASE_DIR, "02_Data_Processed", "f1.db")


class DynamicF1AnalystSandbox:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def execute_analysis_code(self, code_string: str) -> dict:
        """
        Execute arbitrary LLM-generated Python analysis code in a sandboxed scope.
        The code has access to:
          - con: active sqlite3 connection to f1.db
          - pd: pandas
          - np: numpy
          - stats: scipy.stats
          - plt: matplotlib.pyplot
        """
        con = sqlite3.connect(self.db_path)
        output_buffer = io.StringIO()
        captured_data = {}

        # Restricted execution environment
        exec_globals = {
            "sqlite3": sqlite3,
            "con": con,
            "pd": pd,
            "np": np,
            "stats": stats,
            "plt": plt,
            "db_path": self.db_path,
        }
        exec_locals = {}

        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(code_string, exec_globals, exec_locals)

            stdout_text = output_buffer.getvalue().strip()
            result_data = exec_locals.get("result", None)

            # Convert result to string/dict if dataframe
            if isinstance(result_data, pd.DataFrame):
                result_repr = result_data.head(20).to_string(index=False)
            elif result_data is not None:
                result_repr = str(result_data)
            else:
                result_repr = stdout_text

            return {
                "success": True,
                "output": result_repr,
                "stdout": stdout_text,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "stdout": output_buffer.getvalue(),
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc(limit=3)}"
            }
        finally:
            con.close()


# High-level prompt template for Dynamic Tool Generation
LATM_SYSTEM_PROMPT = """You are APEX AI — a Formula 1 Chief Technical Director, Race Strategist, and Motorsports Data Scientist.
Your tone is deeply analytical, precise, and uncompromisingly technical. You speak to an audience of F1 aerodynamicists, telemetry engineers, and academic researchers.

You have access to a live SQLite database `f1.db` containing 2023-2025 race and qualifying telemetry data.
Table `race_results`: DriverId, Abbreviation, TeamName, Position, GridPosition, Points, Laps, Time_s, finished, year, round, event_name
Table `qualifying_results`: DriverId, Abbreviation, TeamName, Position, Q1_s, Q2_s, Q3_s, year, round, event_name

When asked a challenging, novel, or deeply technical question that standard tools cannot answer:
1. Formulate a hypothesis using physics, telemetry, or race strategy principles.
2. WRITE A PYTHON SCRIPT inside a ```python block that executes SQL queries, computes statistical metrics (e.g. regression slopes, variances, degradation rates), and stores the conclusion in a variable called `result`.
3. Synthesize the findings into technical racing metrics (e.g., pace deltas, delta standard deviations, confidence intervals).
"""


if __name__ == "__main__":
    sandbox = DynamicF1AnalystSandbox()
    test_code = """
query = '''
    SELECT DriverId, AVG(GridPosition - Position) as avg_gain, COUNT(*) as race_count
    FROM race_results
    WHERE finished = 1 AND year = 2023
    GROUP BY DriverId
    HAVING race_count > 10
    ORDER BY avg_gain DESC
    LIMIT 5
'''
df = pd.read_sql(query, con)
result = df
"""
    res = sandbox.execute_analysis_code(test_code)
    print("Sandbox Test Result:")
    print("Success:", res["success"])
    print("Output:\n", res["output"])
