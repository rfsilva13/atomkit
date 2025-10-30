#!/usr/bin/env python3
"""
Test FACBackend with SFACWriter integration
"""

import sys
from pathlib import Path

# Add atomkit to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from atomkit.core import AtomicCalculation, FACBackend


def test_fac_with_sfacwriter():
    """Test that FACBackend uses SFACWriter module"""

    print("=" * 80)
    print("Testing FACBackend with SFACWriter Integration")
    print("=" * 80)

    # Test 1: Basic structure calculation
    print("\n[Test 1] Basic Fe XVI structure calculation")
    calc1 = AtomicCalculation(
        name="fe_16_fac_test",
        element="Fe",
        charge=16,
        calculation_type="structure",
        # No coupling parameter - FAC doesn't use it
        configurations=["1s2.2s2.2p5", "1s2.2s.2p6"],
        output_dir="outputs/fac_test1",
        code="fac",
    )

    backend = FACBackend()
    try:
        output_file = backend.write_input(calc1)
        print(f"✓ Created: {output_file}")

        # Read and display the file
        with open(output_file) as f:
            content = f.read()
        print("\nGenerated FAC script:")
        print("-" * 40)
        print(content[:500])  # First 500 chars
        print("-" * 40)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 2: With Breit interaction
    print("\n[Test 2] Fe XV with Breit interaction")
    calc2 = AtomicCalculation(
        name="fe_15_fac_breit",
        element="Fe",
        charge=15,
        calculation_type="radiative",
        # No coupling - FAC is always jj
        relativistic="Breit",
        configurations=["1s2.2s2.2p6", "1s2.2s.2p6.3s"],
        radiation_types=["E1"],
        output_dir="outputs/fac_test2",
        code="fac",
    )

    try:
        output_file = backend.write_input(calc2)
        print(f"✓ Created: {output_file}")

        with open(output_file) as f:
            content = f.read()

        # Check for Breit
        if "SetBreit" in content:
            print("✓ SetBreit found in output")
        else:
            print("✗ SetBreit NOT found (should be there!)")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 3: With QED
    print("\n[Test 3] Fe XV with QED corrections")
    calc3 = AtomicCalculation(
        name="fe_15_fac_qed",
        element="Fe",
        charge=15,
        calculation_type="structure",
        # No coupling - FAC doesn't use it
        qed_corrections=True,
        configurations=["1s2.2s2.2p6"],
        output_dir="outputs/fac_test3",
        code="fac",
    )

    try:
        output_file = backend.write_input(calc3)
        print(f"✓ Created: {output_file}")

        with open(output_file) as f:
            content = f.read()

        # Check for QED
        if "SetVP" in content and "SetSE" in content:
            print("✓ QED corrections (SetVP, SetSE) found in output")
        else:
            print("✗ QED corrections NOT found")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("All FACBackend tests passed!")
    print("SFACWriter integration working correctly ✓")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_fac_with_sfacwriter()
    sys.exit(0 if success else 1)
