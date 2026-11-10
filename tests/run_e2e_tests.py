#!/usr/bin/env python3
"""
APEX AI Test Runner
===================

Executes the automated test suite for the APEX AI architecture,
including feature integrity, models, analytics modules, and CLIs.

Usage:
  python tests/run_e2e_tests.py
"""

import sys
import os
import time
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import test_apex_ai


def main():
    print("=" * 80)
    print("APEX AI: AUTOMATED TEST SUITE")
    print("=" * 80)
    print(f"Project Root:   {ROOT_DIR}")
    print(f"Python Version: {sys.version.split()[0]} ({sys.executable})")
    print("-" * 80)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_apex_ai)

    start = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start

    print("\n" + "=" * 80)
    print(f"Tests Run: {result.testsRun} | Failures: {len(result.failures)} | Errors: {len(result.errors)}")
    print(f"Elapsed:   {elapsed:.3f}s")
    print("=" * 80)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
