"""
APEX AI - Milestone 1: Data Cleaning
Takes the raw FastF1 CSVs and produces clean, analysis-ready datasets.

Cleaning steps:
  1. Drop Pre-Season Testing (not real competition sessions)
  2. Normalise column names and types
  3. Convert lap/qualifying times to seconds (float) for maths
  4. Add a clean 'finished' flag (True = classified finisher)
  5. Save combined CSVs + a SQLite database

Why SQLite? It's a single file, no server, and lets the analytics layer
query with SQL later (e.g. "best lap time per driver per circuit").
"""

import glob
import os
import sqlite3

import pandas as pd

RAW_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_Data_Raw")
)
OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_Data_Processed")
)
os.makedirs(OUT_DIR, exist_ok=True)

# Sessions that are not real competition.
NON_COMPETITION = {"Pre-Season Testing"}


def time_to_seconds(series: pd.Series) -> pd.Series:
    """Convert pandas Timedelta (HH:MM:SS.fff) to seconds as float. NaN stays NaN."""
    return series.dt.total_seconds()


def clean_one(path: str, session_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1. Drop non-competition sessions.
    df = df[~df["event_name"].isin(NON_COMPETITION)].copy()

    # 2. Normalise types.
    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype(int)
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")

    # 3. Convert times to seconds.
    time_cols = [c for c in ["Time", "Q1", "Q2", "Q3"] if c in df.columns]
    for col in time_cols:
        td = pd.to_timedelta(df[col], errors="coerce")
        df[col + "_s"] = time_to_seconds(td)

    # 4. Finished flag: a driver who has a race Time is a classified finisher.
    #    NaN Time = DNF / retired.
    if "Time_s" in df.columns:
        df["finished"] = df["Time_s"].notna()

    # 5. Sort for predictable output.
    df = df.sort_values(["year", "round", "Position"]).reset_index(drop=True)

    df["session_type"] = session_label
    return df


def main() -> None:
    race_frames, quali_frames = [], []

    for path in sorted(glob.glob(os.path.join(RAW_DIR, "race_results_*.csv"))):
        race_frames.append(clean_one(path, "Race"))
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "qualifying_results_*.csv"))):
        quali_frames.append(clean_one(path, "Qualifying"))

    race = pd.concat(race_frames, ignore_index=True)
    quali = pd.concat(quali_frames, ignore_index=True)

    # Save clean CSVs.
    race_out = os.path.join(OUT_DIR, "race_results_clean.csv")
    quali_out = os.path.join(OUT_DIR, "qualifying_results_clean.csv")
    race.to_csv(race_out, index=False)
    quali.to_csv(quali_out, index=False)

    # Combined flat file (all sessions, one row per driver per session).
    combined = pd.concat([race, quali], ignore_index=True)
    combined_out = os.path.join(OUT_DIR, "all_sessions_clean.csv")
    combined.to_csv(combined_out, index=False)

    # SQLite database for the analytics layer.
    db_path = os.path.join(OUT_DIR, "f1.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    race.to_sql("race_results", conn, if_exists="replace", index=False)
    quali.to_sql("qualifying_results", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    print("=== Cleaned data summary ===")
    print(f"Race sessions:      {race['event_name'].nunique()} events, {len(race)} driver-results")
    print(f"Qualifying sessions:{quali['event_name'].nunique()} events, {len(quali)} driver-results")
    print(f"Seasons:            {sorted(combined['year'].unique())}")
    print(f"Drivers:            {combined['Abbreviation'].nunique()}")
    print(f"DNFs (race):        {(~race['finished']).sum()}")
    print()
    print("Files written:")
    for p in [race_out, quali_out, combined_out, db_path]:
        print(f"  {p} ({os.path.getsize(p):,} bytes)")


if __name__ == "__main__":
    main()