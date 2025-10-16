#!/usr/bin/env python3
"""
Simple test - just test file generation directly without full imports.
"""

import sys

sys.path.insert(0, "/home/rfsilva/atomkit/src")

# Direct imports to avoid numpy issues
from atomkit.core.specs import CouplingScheme, RelativisticTreatment
from atomkit.core.calculation import AtomicCalculation
from atomkit.configuration import Configuration

print("✓ Imports successful")

# Test 1: LS coupling
print("\n" + "=" * 70)
print("Test 1: LS coupling, non-relativistic")
print("=" * 70)
calc = AtomicCalculation(
    element="Fe",
    charge=16,
    calculation_type="structure",
    coupling="LS",
    relativistic="none",
    configurations=[
        Configuration.from_string("2p6 3s0"),
        Configuration.from_string("2p5 3s1"),
    ],
    code="autostructure",
    output_dir="outputs/test1",
)
filepath = calc.write_input()
print(f"✓ Generated: {filepath}")

# Check file
with open(filepath) as f:
    content = f.read()
    print(content)
    assert "CUP='LS'" in content
    print(f"✓ File contains CUP='LS'")

# Test 2: IC + Breit (KEY TEST!)
print("\n" + "=" * 70)
print("Test 2: IC + Breit (should be CUP='IC' IBREIT=1, NOT ICR!)")
print("=" * 70)
calc = AtomicCalculation(
    element="Fe",
    charge=15,
    calculation_type="structure",
    coupling="IC",
    relativistic="Breit",
    configurations=[
        Configuration.from_string("2p6 3s1"),
    ],
    code="autostructure",
    output_dir="outputs/test2",
)
filepath = calc.write_input()
print(f"✓ Generated: {filepath}")

# Check file
with open(filepath) as f:
    content = f.read()
    print(content)
    assert "CUP='IC'" in content, "Should be CUP='IC'"
    assert "IBREIT=1" in content, "Should have IBREIT=1"
    assert "ICR" not in content, "Should NOT contain ICR"
    print(f"✓ File contains CUP='IC' IBREIT=1 (correct!)")

# Test 3: ICR coupling
print("\n" + "=" * 70)
print("Test 3: ICR (kappa-averaged)")
print("=" * 70)
calc = AtomicCalculation(
    element="Fe",
    charge=15,
    calculation_type="structure",
    coupling="ICR",
    relativistic="none",
    configurations=[
        Configuration.from_string("2p6 3s1"),
    ],
    code="autostructure",
    output_dir="outputs/test3",
)
filepath = calc.write_input()
print(f"✓ Generated: {filepath}")

# Check file
with open(filepath) as f:
    content = f.read()
    print(content)
    assert "CUP='ICR'" in content
    print(f"✓ File contains CUP='ICR'")

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nKey fix verified:")
print("  - IC + Breit → CUP='IC' IBREIT=1 (NOT ICR!)")
print("  - Output directories created automatically")
print("  - All AS coupling schemes work correctly")
