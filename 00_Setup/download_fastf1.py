"""
APEX AI - Milestone 1: Data Pipeline
Downloads F1 qualifying + race results for 2023-2025 using FastF1
and saves them as CSV files.

Why FastF1 for this milestone:
  - Installs and runs on your PC (pip install fastf1)
  - Qualifying + race results are FREE (no F1TV needed)
  - Lap-by-lap timing and telemetry are behind F1TV paywall,
    so we start with what is freely available and add OpenF1 later.

Rate-limit handling:
  FastF1 allows ~500 API calls per hour. Each session.load() burns
  several calls, so we:
    * save after EVERY session (never lose work)
    * skip sessions already saved (resume support)
    * back off + retry on rate-limit errors
    * print progress so you can watch it

Usage:
    python download_fastf1.py            # download everything, resume-safe
    python download_fastf1.py --year 2023 --type R   # one season + session type
"""

import argparse
import logging
import os
import sys
import time

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

# Silence FastF1's noisy logging; we only want real errors.
logging.getLogger("fastf1").setLevel(logging.ERROR)

OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_Data_Raw")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2023, 2024, 2025]
SESSION_TYPES = ["R", "Q"]  # R = Race, Q = Qualifying

# How long to wait when FastF1 tells us we're rate-limited.
RATE_LIMIT_SLEEP_SECONDS = 60
# Max times we'll retry a single session before giving up.
MAX_RETRIES = 3


def output_path(year: int, session_type: str) -> str:
    label = "race_results" if session_type == "R" else "qualifying_results"
    return os.path.join(OUTPUT_DIR, f"{label}_{year}.csv")


def load_existing(path: str) -> pd.DataFrame:
    """Load previously saved rows, if any (supports resuming)."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def fetch_season(year: int, session_type: str) -> None:
    """
    Download every session of one type in a season, saving incrementally.
    """
    path = output_path(year, session_type)
    existing = load_existing(path)
    already_done = set(existing["event_name"]) if not existing.empty else set()

    schedule = fastf1.get_event_schedule(year)
    label = "Race" if session_type == "R" else "Qualifying"
    print(f"\n=== {year} {label} sessions ===")

    new_rows: list[pd.DataFrame] = []

    for _, event in schedule.iterrows():
        event_name = event["EventName"]

        if event_name in already_done:
            print(f"  [skip] {event_name}: already downloaded")
            continue

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                session = fastf1.get_session(year, event_name, session_type)
                session.load()
                results = session.results
                if results is None or len(results) == 0:
                    print(f"  [empty] {event_name}: no results")
                    break  # nothing to retry

                results = results.copy()
                results["year"] = year
                results["session_type"] = label
                results["event_name"] = event_name
                results["round"] = event["RoundNumber"]
                new_rows.append(results)
                print(f"  [ok] {event_name}: {len(results)} drivers")
                success = True
                break

            except RateLimitExceededError:
                print(f"  [rate-limit] {event_name} attempt {attempt} - waiting {RATE_LIMIT_SLEEP_SECONDS}s")
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] {event_name} attempt {attempt}: {type(exc).__name__}: {str(exc)[:120]}")
                time.sleep(5)

        # Save after every session so a rate-limit crash never loses work.
        if new_rows:
            combined = pd.concat([existing, *new_rows], ignore_index=True)
            combined.to_csv(path, index=False)
            existing = combined
            new_rows = []

    if not existing.empty:
        print(f"  -> saved {len(existing)} rows to {path}")
    else:
        print(f"  -> no data for {year} {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download F1 results via FastF1")
    parser.add_argument("--year", type=int, default=None, help="Single season (default: all)")
    parser.add_argument("--type", choices=["R", "Q"], default=None,
                        help="Single session type (default: both)")
    args = parser.parse_args()

    years = [args.year] if args.year else YEARS
    types = [args.type] if args.type else SESSION_TYPES

    for year in years:
        for stype in types:
            try:
                fetch_season(year, stype)
            except RateLimitExceededError:
                print(f"\n[rate-limit] Hard limit hit for {year} {stype}. "
                      f"Data saved so far. Re-run this script to resume.")
                sys.exit(1)
            except Exception as exc:  # noqa: BLE001
                print(f"\n[error] {year} {stype}: {type(exc).__name__}: {str(exc)[:200]}")

    print("\n=== Download complete ===")
    for year in years:
        for stype in types:
            path = output_path(year, stype)
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"{path}: {len(df)} rows, {df['event_name'].nunique()} sessions")


if __name__ == "__main__":
    main()