#!/usr/bin/env python3
"""
Installation verification script for AtomKit.

Run this after installing AtomKit to verify that everything is set up correctly.
"""

import sys


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(
            f"✗ Python {version.major}.{version.minor}.{version.micro} (Requires Python 3.10+)"
        )
        return False


def check_imports():
    """Check if all required packages can be imported."""
    print("\nChecking core dependencies...")
    packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("mendeleev", "Mendeleev"),
    ]

    all_ok = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (not installed)")
            all_ok = False

    return all_ok


def check_optional_imports():
    """Check optional dependencies."""
    print("\nChecking optional dependencies...")
    packages = [
        ("matplotlib", "Matplotlib (plotting)"),
        ("seaborn", "Seaborn (enhanced plotting)"),
        ("colorlog", "Colorlog (colored logging)"),
    ]

    for module, name in packages:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"○ {name} (not installed, optional)")


def check_atomkit():
    """Check if AtomKit can be imported and basic functionality works."""
    print("\nChecking AtomKit installation...")

    try:
        import atomkit

        print("✓ AtomKit package")

        from atomkit import Configuration, Shell

        print("✓ Configuration and Shell classes")

        from atomkit.readers import read_fac

        print("✓ FAC readers")

        from atomkit.physics import energy_converter

        print("✓ Physics utilities")

        # Test basic functionality
        config = Configuration.from_string("1s2.2s2.2p6")
        assert config.total_electrons() == 10
        print("✓ Basic configuration parsing")

        energy = energy_converter.ev_to_rydberg(13.6)
        assert abs(energy - 1.0) < 0.01
        print("✓ Energy conversion")

        return True

    except Exception as e:
        print(f"✗ AtomKit check failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("AtomKit Installation Verification")
    print("=" * 60)

    checks = [
        check_python_version(),
        check_imports(),
        check_atomkit(),
    ]

    check_optional_imports()

    print("\n" + "=" * 60)
    if all(checks):
        print("✓ All checks passed! AtomKit is ready to use.")
        print("\nTo get started, try:")
        print("  python examples/basic_usage_example.py")
        print("\nOr run the test suite:")
        print("  pytest tests/ -v")
        print("=" * 60)
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        print("\nFor installation help, see:")
        print("  https://github.com/rfsilva13/atomkit#installation")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
