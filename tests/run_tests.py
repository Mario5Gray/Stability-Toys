#!/usr/bin/env python3
"""
Test runner script.
Provides convenient commands for running different test suites.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd):
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_all_tests(args):
    """Run all tests."""
    cmd = ["pytest"]
    if args.verbose:
        cmd.append("-v")
    if args.coverage:
        cmd.extend(["--cov-report=html", "--cov-report=term"])
    return run_command(cmd)


def run_unit_tests(args):
    """Run only unit tests."""
    cmd = ["pytest", "-m", "unit"]
    if args.verbose:
        cmd.append("-v")
    return run_command(cmd)


def run_integration_tests(args):
    """Run only integration tests."""
    cmd = ["pytest", "-m", "integration"]
    if args.verbose:
        cmd.append("-v")
    return run_command(cmd)


def run_specific_test(args):
    """Run a specific test file or test."""
    cmd = ["pytest", args.test]
    if args.verbose:
        cmd.append("-v")
    return run_command(cmd)


def run_with_coverage(args):
    """Run tests with coverage report."""
    cmd = [
        "pytest",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=json",
    ]
    if args.verbose:
        cmd.append("-v")
    return run_command(cmd)


def run_fast(args):
    """Run only fast tests (exclude slow tests)."""
    cmd = ["pytest", "-m", "not slow"]
    if args.verbose:
        cmd.append("-v")
    return run_command(cmd)


def run_slow(args):
    """Run only slow tests."""
    cmd = ["pytest", "-m", "slow"]
    if args.verbose:
        cmd.append("-v")
    return run_command(cmd)


def run_watch(args):
    """Run tests in watch mode (requires pytest-watch)."""
    try:
        cmd = ["ptw", "--"]
        if args.verbose:
            cmd.append("-v")
        return run_command(cmd)
    except FileNotFoundError:
        print("Error: pytest-watch not installed. Install with: pip install pytest-watch")
        return 1


def run_parallel(args):
    """Run tests in parallel (requires pytest-xdist)."""
    try:
        cmd = ["pytest", "-n", str(args.workers)]
        if args.verbose:
            cmd.append("-v")
        return run_command(cmd)
    except FileNotFoundError:
        print("Error: pytest-xdist not installed. Install with: pip install pytest-xdist")
        return 1


def run_failed(args):
    """Run only previously failed tests."""
    cmd = ["pytest", "--lf"]
    if args.verbose:
        cmd.append("-v")
    return run_command(cmd)


def clean_cache(args):
    """Clean pytest cache and coverage files."""
    import shutil
    
    paths_to_clean = [
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "coverage.json",
        "__pycache__",
    ]
    
    for path in paths_to_clean:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                shutil.rmtree(path_obj)
                print(f"Removed directory: {path}")
            else:
                path_obj.unlink()
                print(f"Removed file: {path}")
    
    print("Cache cleaned!")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./run_tests.py all              # Run all tests
  ./run_tests.py unit             # Run only unit tests
  ./run_tests.py integration      # Run only integration tests
  ./run_tests.py coverage         # Run with coverage report
  ./run_tests.py fast             # Run fast tests only
  ./run_tests.py parallel -w 4    # Run tests in parallel with 4 workers
  ./run_tests.py failed           # Re-run only failed tests
  ./run_tests.py clean            # Clean cache files
        """
    )
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Test command")
    
    # All tests
    all_parser = subparsers.add_parser("all", help="Run all tests")
    all_parser.add_argument("-c", "--coverage", action="store_true", help="Include coverage")
    all_parser.set_defaults(func=run_all_tests)
    
    # Unit tests
    unit_parser = subparsers.add_parser("unit", help="Run unit tests only")
    unit_parser.set_defaults(func=run_unit_tests)
    
    # Integration tests
    integration_parser = subparsers.add_parser("integration", help="Run integration tests only")
    integration_parser.set_defaults(func=run_integration_tests)
    
    # Specific test
    specific_parser = subparsers.add_parser("specific", help="Run specific test file or test")
    specific_parser.add_argument("test", help="Test file or test path")
    specific_parser.set_defaults(func=run_specific_test)
    
    # Coverage
    coverage_parser = subparsers.add_parser("coverage", help="Run with coverage report")
    coverage_parser.set_defaults(func=run_with_coverage)
    
    # Fast tests
    fast_parser = subparsers.add_parser("fast", help="Run fast tests only (exclude slow)")
    fast_parser.set_defaults(func=run_fast)
    
    # Slow tests
    slow_parser = subparsers.add_parser("slow", help="Run slow tests only")
    slow_parser.set_defaults(func=run_slow)
    
    # Watch mode
    watch_parser = subparsers.add_parser("watch", help="Run tests in watch mode")
    watch_parser.set_defaults(func=run_watch)
    
    # Parallel
    parallel_parser = subparsers.add_parser("parallel", help="Run tests in parallel")
    parallel_parser.add_argument("-w", "--workers", type=int, default=4, help="Number of workers")
    parallel_parser.set_defaults(func=run_parallel)
    
    # Failed tests
    failed_parser = subparsers.add_parser("failed", help="Re-run only failed tests")
    failed_parser.set_defaults(func=run_failed)
    
    # Clean
    clean_parser = subparsers.add_parser("clean", help="Clean cache files")
    clean_parser.set_defaults(func=clean_cache)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
