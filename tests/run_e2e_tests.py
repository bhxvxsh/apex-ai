#!/usr/bin/env python3
"""
APEX AI Research Paper E2E Test Runner
=====================================

Executes the 4-Tier requirement-driven opaque-box verification test suite.
Zero external dependencies required (uses standard library `unittest`).

Usage:
  python tests/run_e2e_tests.py
  python tests/run_e2e_tests.py --tier 1
  python tests/run_e2e_tests.py --verbose
"""

import sys
import os
import time
import argparse
import unittest
from pathlib import Path

# Add project root and tests directory to path
TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import test_e2e_research_paper as suite_module


class ColoredText:
    """ANSI color formatting helpers."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def ok(cls, s): return f"{cls.GREEN}{s}{cls.RESET}"
    @classmethod
    def fail(cls, s): return f"{cls.RED}{s}{cls.RESET}"
    @classmethod
    def warn(cls, s): return f"{cls.YELLOW}{s}{cls.RESET}"
    @classmethod
    def info(cls, s): return f"{cls.CYAN}{s}{cls.RESET}"
    @classmethod
    def bold(cls, s): return f"{cls.BOLD}{s}{cls.RESET}"


def build_suite(selected_tier: int = None) -> unittest.TestSuite:
    """Builds a test suite filtered by tier or containing all tiers."""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    tier_classes = {
        1: suite_module.TestTier1Deliverables,
        2: suite_module.TestTier2ResolutionAndBoundary,
        3: suite_module.TestTier3CrossFeatureAlignment,
        4: suite_module.TestTier4EmpiricalConsistencyAndHypotheses,
    }

    if selected_tier in tier_classes:
        suite.addTests(loader.loadTestsFromTestCase(tier_classes[selected_tier]))
    else:
        for cls in tier_classes.values():
            suite.addTests(loader.loadTestsFromTestCase(cls))

    return suite


def run_e2e_test_suite(selected_tier: int = None, verbose: bool = False) -> int:
    """Executes the test suite and displays a formatted summary report."""
    print("=" * 80)
    print(ColoredText.bold(ColoredText.info("APEX AI IEEE RESEARCH PAPER: END-TO-END (E2E) TEST RUNNER")))
    print("=" * 80)
    print(f"Project Root:     {ROOT_DIR}")
    print(f"Target Directory: {ROOT_DIR / '06_Research_Paper'}")
    print(f"Python Version:   {sys.version.split()[0]} ({sys.executable})")
    print(f"Execution Target: {'All Tiers (1-4)' if not selected_tier else f'Tier {selected_tier} Only'}")
    print("-" * 80)

    start_time = time.time()
    test_suite = build_suite(selected_tier)

    # Run tests using TextTestRunner
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(ColoredText.bold("TEST SUITE EXECUTION SUMMARY"))
    print("=" * 80)

    total_tests = result.testsRun
    total_failures = len(result.failures)
    total_errors = len(result.errors)
    total_passed = total_tests - total_failures - total_errors

    print(f"Total Tests Run: {total_tests}")
    print(f"Passed:          {ColoredText.ok(str(total_passed))}")
    print(f"Failures:        {ColoredText.fail(str(total_failures)) if total_failures else '0'}")
    print(f"Errors:          {ColoredText.fail(str(total_errors)) if total_errors else '0'}")
    print(f"Elapsed Time:    {elapsed:.3f} seconds")
    print("-" * 80)

    if result.wasSuccessful():
        print(ColoredText.bold(ColoredText.ok(">>> OVERALL STATUS: ALL TEST TIERS PASSED [SUCCESS] <<<")))
        print("=" * 80)
        return 0
    else:
        print(ColoredText.bold(ColoredText.fail(">>> OVERALL STATUS: TEST VERIFICATION FAILED [FAIL] <<<")))
        print("=" * 80)
        return 1


def main():
    parser = argparse.ArgumentParser(description="APEX AI Research Paper E2E Test Suite Runner")
    parser.add_argument("--tier", "-t", type=int, choices=[1, 2, 3, 4], help="Run only specified test tier (1-4)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose test output")
    args = parser.parse_args()

    exit_code = run_e2e_test_suite(selected_tier=args.tier, verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
