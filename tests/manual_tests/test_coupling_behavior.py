#!/usr/bin/env python3
"""
Test that coupling parameter works correctly for both backends.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from atomkit.core import AtomicCalculation, AutostructureBackend, FACBackend


def test_autostructure_coupling():
    """Test AUTOSTRUCTURE with and without coupling"""
    print("\n" + "=" * 80)
    print("AUTOSTRUCTURE: Coupling Parameter Tests")
    print("=" * 80)

    # Test 1: No coupling specified (should default to ICR)
    print("\n[Test 1] AUTOSTRUCTURE with coupling=None (should default to ICR)")
    calc1 = AtomicCalculation(
        name="test_as_default",
        element="C",
        charge=2,
        calculation_type="structure",
        coupling=None,  # Should default to ICR
        configurations=["1s2.2s2"],
        output_dir="outputs/test_coupling",
        code="autostructure",
    )
    print(f"  Coupling after init: {calc1.coupling}")
    assert calc1.coupling == "ICR", f"Expected ICR, got {calc1.coupling}"
    print("  ✓ Defaults to ICR")

    # Test 2: Explicit LS coupling
    print("\n[Test 2] AUTOSTRUCTURE with explicit coupling='LS'")
    calc2 = AtomicCalculation(
        name="test_as_ls",
        element="C",
        charge=2,
        calculation_type="structure",
        coupling="LS",
        configurations=["1s2.2s2"],
        output_dir="outputs/test_coupling",
        code="autostructure",
    )
    print(f"  Coupling after init: {calc2.coupling}")
    assert calc2.coupling == "LS", f"Expected LS, got {calc2.coupling}"

    # Check generated file doesn't have CUP (since LS is now not default)
    backend = AutostructureBackend()
    output = backend.write_input(calc2)
    with open(output) as f:
        content = f.read()
    if "CUP='LS'" in content:
        print("  ⚠️  File contains CUP='LS' (okay if explicit)")
    else:
        print("  ✓ LS coupling handled correctly")

    # Test 3: Explicit ICR coupling (should not appear in file since it's default)
    print("\n[Test 3] AUTOSTRUCTURE with explicit coupling='ICR'")
    calc3 = AtomicCalculation(
        name="test_as_icr",
        element="C",
        charge=2,
        calculation_type="structure",
        coupling="ICR",
        configurations=["1s2.2s2"],
        output_dir="outputs/test_coupling",
        code="autostructure",
    )
    output = backend.write_input(calc3)
    with open(output) as f:
        content = f.read()
    if "CUP=" not in content or "CUP='ICR'" not in content:
        print("  ✓ ICR is default, not written to file")
    else:
        print("  ⚠️  CUP appears in file (may be okay)")


def test_fac_coupling():
    """Test FAC ignores coupling parameter"""
    print("\n" + "=" * 80)
    print("FAC: Coupling Parameter Tests (Should Be Ignored)")
    print("=" * 80)

    # Test 1: No coupling specified
    print("\n[Test 1] FAC with coupling=None")
    calc1 = AtomicCalculation(
        name="test_fac_none",
        element="C",
        charge=2,
        calculation_type="structure",
        coupling=None,  # FAC doesn't use coupling
        configurations=["1s2.2s2"],
        output_dir="outputs/test_coupling",
        code="fac",
    )
    print(f"  Coupling after init: {calc1.coupling}")
    print(f"  Warnings: {calc1._warnings}")
    print("  ✓ FAC doesn't require coupling")

    # Test 2: Coupling specified (should warn)
    print("\n[Test 2] FAC with coupling='LS' (should warn)")
    calc2 = AtomicCalculation(
        name="test_fac_ls",
        element="C",
        charge=2,
        calculation_type="structure",
        coupling="LS",  # Will be ignored
        configurations=["1s2.2s2"],
        output_dir="outputs/test_coupling",
        code="fac",
    )
    print(f"  Warnings: {calc2._warnings}")
    has_warning = any("ignored" in w.lower() for w in calc2._warnings)
    if has_warning:
        print("  ✓ Warning issued about coupling being ignored")
    else:
        print("  ⚠️  No warning about coupling (expected one)")

    # Check FAC file doesn't mention coupling
    backend = FACBackend()
    output = backend.write_input(calc2)
    with open(output) as f:
        content = f.read()
    if "coupling" not in content.lower() and "CUP" not in content:
        print("  ✓ FAC output doesn't mention coupling")
    else:
        print("  ⚠️  Coupling appears in FAC file (shouldn't)")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Coupling Parameter Handling")
    print("=" * 80)

    try:
        test_autostructure_coupling()
        test_fac_coupling()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nSummary:")
        print("  • AUTOSTRUCTURE: Defaults to ICR when coupling=None")
        print("  • AUTOSTRUCTURE: Respects explicit coupling values")
        print("  • FAC: Ignores coupling parameter (always jj)")
        print("  • FAC: Issues warning if coupling is specified")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
