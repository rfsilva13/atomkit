#!/usr/bin/env python3
"""
Test runner script for the Configuration class.

This script runs all tests and provides a summary of the test coverage.
"""

import subprocess
import sys
import os


def run_tests():
    """Run the Configuration tests and display results."""
    print("=" * 80)
    print("RUNNING UNIT TESTS FOR CONFIGURATION CLASS")
    print("=" * 80)

    # Change to the atomkit directory
    os.chdir("/home/rfsilva/EIEres/atomkit")

    # Run pytest with verbose output
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_configuration.py",
            "-v",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Summary
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nTest Coverage Summary:")
        print("- Basic Configuration creation and initialization")
        print("- Shell addition, removal, and manipulation")
        print("- String parsing (standard and compact notation)")
        print("- Element-based configuration creation (with mendeleev)")
        print("- Configuration operations (copy, compare, split)")
        print("- Hole and excitation generation")
        print("- X-ray labeling")
        print("- Error handling and edge cases")
        print("- All magic methods (__str__, __repr__, __eq__, etc.)")

        # Count test methods
        with open("tests/test_configuration.py", "r") as f:
            content = f.read()
            test_count = content.count("def test_")

        print(f"\nTotal tests: {test_count}")
        print("Test file: tests/test_configuration.py")

    else:
        print("\n" + "=" * 80)
        print("❌ SOME TESTS FAILED!")
        print("=" * 80)
        print(f"Exit code: {result.returncode}")

    return result.returncode == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
