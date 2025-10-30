#!/usr/bin/env python3
"""
Test all parameter usage fixes

Tests:
1. FAC OptimizeRadial default behavior (should optimize g0 only)
2. FAC ConfigEnergy usage
3. FAC radiation_types mapping to multipole
4. FAC code_options integration
5. AS code_options integration
"""

import sys
from pathlib import Path

# Add atomkit to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from atomkit.core import AtomicCalculation, FACBackend, AutostructureBackend


def test_fac_default_optimization():
    """Test 1: FAC should optimize g0 by default"""
    print("=" * 80)
    print("Test 1: FAC Default Optimization (ground state only)")
    print("=" * 80)

    calc = AtomicCalculation(
        name="fac_default_opt",
        element="Fe",
        charge=16,
        calculation_type="structure",
        configurations=["1s2.2s2.2p6", "1s2.2s2.2p5.3s", "1s2.2s.2p6.3s"],
        output_dir="outputs/param_tests",
        code="fac",
    )

    backend = FACBackend()
    output_file = backend.write_input(calc)

    with open(output_file) as f:
        content = f.read()

    print(f"✓ Created: {output_file}")

    # Check for OptimizeRadial with only g0
    if "OptimizeRadial(['g0'])" in content:
        print("✅ PASS: OptimizeRadial(['g0']) found (only ground state)")
    elif "OptimizeRadial(" in content:
        print(f"⚠️  WARNING: Found OptimizeRadial but not ['g0'] only")
        # Print the line
        for line in content.split("\n"):
            if "OptimizeRadial" in line:
                print(f"   Found: {line.strip()}")
    else:
        print("❌ FAIL: No OptimizeRadial found!")

    print()
    return output_file


def test_fac_full_optimization():
    """Test 2: FAC with optimization='energy' should optimize all groups"""
    print("=" * 80)
    print("Test 2: FAC Full Optimization (all groups)")
    print("=" * 80)

    calc = AtomicCalculation(
        name="fac_full_opt",
        element="Fe",
        charge=16,
        calculation_type="structure",
        optimization="energy",  # Should optimize ALL groups
        configurations=["1s2.2s2.2p6", "1s2.2s2.2p5.3s"],
        output_dir="outputs/param_tests",
        code="fac",
    )

    backend = FACBackend()
    output_file = backend.write_input(calc)

    with open(output_file) as f:
        content = f.read()

    print(f"✓ Created: {output_file}")

    # Check for OptimizeRadial with all groups
    if "OptimizeRadial(['g0', 'g1'])" in content:
        print("✅ PASS: OptimizeRadial(['g0', 'g1']) found (all groups)")
    elif "OptimizeRadial(" in content:
        print(f"⚠️  WARNING: Found OptimizeRadial but not with all groups")
        for line in content.split("\n"):
            if "OptimizeRadial" in line:
                print(f"   Found: {line.strip()}")
    else:
        print("❌ FAIL: No OptimizeRadial found!")

    print()
    return output_file


def test_fac_config_energy():
    """Test 3: FAC should call ConfigEnergy"""
    print("=" * 80)
    print("Test 3: FAC ConfigEnergy Usage")
    print("=" * 80)

    calc = AtomicCalculation(
        name="fac_config_energy",
        element="O",
        charge=7,
        calculation_type="structure",
        configurations=["1s2.2s", "1s2.2p"],
        output_dir="outputs/param_tests",
        code="fac",
    )

    backend = FACBackend()
    output_file = backend.write_input(calc)

    with open(output_file) as f:
        content = f.read()

    print(f"✓ Created: {output_file}")

    # Check for ConfigEnergy calls
    config_energy_count = content.count("ConfigEnergy(")
    if config_energy_count >= 2:
        print(f"✅ PASS: Found {config_energy_count} ConfigEnergy calls")
        for i, line in enumerate(content.split("\n")):
            if "ConfigEnergy" in line:
                print(f"   Line {i+1}: {line.strip()}")
    else:
        print(
            f"❌ FAIL: Only found {config_energy_count} ConfigEnergy calls (expected 2)"
        )

    print()
    return output_file


def test_fac_radiation_types():
    """Test 4: FAC radiation_types should map to multipole"""
    print("=" * 80)
    print("Test 4: FAC radiation_types Mapping")
    print("=" * 80)

    calc = AtomicCalculation(
        name="fac_rad_types",
        element="Fe",
        charge=15,
        calculation_type="radiative",
        radiation_types=["E1", "M1", "E2"],  # Multiple types
        configurations=["1s2.2s2.2p6", "1s2.2s2.2p5.3s"],
        output_dir="outputs/param_tests",
        code="fac",
    )

    backend = FACBackend()
    output_file = backend.write_input(calc)

    with open(output_file) as f:
        content = f.read()

    print(f"✓ Created: {output_file}")

    # Check for separate TRTable calls for each radiation type
    if "tr_E1.b" in content and "tr_M1.b" in content and "tr_E2.b" in content:
        print("✅ PASS: Found separate TRTable files for E1, M1, E2")
        for line in content.split("\n"):
            if "TRTable" in line and "tr_" in line:
                print(f"   {line.strip()}")
    elif "TRTable" in content:
        print("⚠️  WARNING: Found TRTable but not separate files for each type")
        for line in content.split("\n"):
            if "TRTable" in line:
                print(f"   {line.strip()}")
    else:
        print("❌ FAIL: No TRTable found!")

    print()
    return output_file


def test_fac_code_options():
    """Test 5: FAC code_options should be used"""
    print("=" * 80)
    print("Test 5: FAC code_options Integration")
    print("=" * 80)

    calc = AtomicCalculation(
        name="fac_code_opts",
        element="Fe",
        charge=16,
        calculation_type="structure",
        configurations=["1s2.2s2.2p6"],
        code_options={
            "SetUTA": 1,  # Should call fac.SetUTA(1)
            "SetMS": (10, 20),  # Should call fac.SetMS(10, 20)
            "MaxLevels": 1000,  # Should add as comment (no SFACWriter method)
        },
        output_dir="outputs/param_tests",
        code="fac",
    )

    backend = FACBackend()
    output_file = backend.write_input(calc)

    with open(output_file) as f:
        content = f.read()

    print(f"✓ Created: {output_file}")

    # Check for SetUTA
    if "SetUTA(1)" in content:
        print("✅ PASS: SetUTA(1) found")
    else:
        print("❌ FAIL: SetUTA(1) not found")

    # Check for SetMS
    if "SetMS(10, 20)" in content:
        print("✅ PASS: SetMS(10, 20) found")
    else:
        print("❌ FAIL: SetMS(10, 20) not found")

    # Check for MaxLevels comment
    if "MaxLevels = 1000" in content:
        print("✅ PASS: MaxLevels = 1000 found in comments")
    else:
        print("⚠️  WARNING: MaxLevels not found in comments")

    print()
    return output_file


def test_as_code_options():
    """Test 6: AS code_options should be passed to ASWriter"""
    print("=" * 80)
    print("Test 6: AUTOSTRUCTURE code_options Integration")
    print("=" * 80)

    calc = AtomicCalculation(
        name="as_code_opts",
        element="Fe",
        charge=15,
        calculation_type="structure",
        coupling="IC",
        configurations=["1s2.2s2.2p6", "1s2.2s2.2p5.3s"],
        code_options={
            "SCFRAC": 0.85,  # Should go to SMINIM
            "NLAM": 10,  # Should go to SMINIM
            "IWGHT": 2,  # Should go to SMINIM
        },
        output_dir="outputs/param_tests",
        code="autostructure",
    )

    backend = AutostructureBackend()
    output_file = backend.write_input(calc)

    with open(output_file) as f:
        content = f.read()

    print(f"✓ Created: {output_file}")

    # Check for custom parameters in SMINIM
    if "SCFRAC=0.85" in content or "SCFRAC = 0.85" in content:
        print("✅ PASS: SCFRAC=0.85 found")
    else:
        print("❌ FAIL: SCFRAC=0.85 not found")

    if "NLAM=10" in content or "NLAM = 10" in content:
        print("✅ PASS: NLAM=10 found")
    else:
        print("❌ FAIL: NLAM=10 not found")

    if "IWGHT=2" in content or "IWGHT = 2" in content:
        print("✅ PASS: IWGHT=2 found")
    else:
        print("❌ FAIL: IWGHT=2 not found")

    print()
    return output_file


def main():
    """Run all parameter usage tests"""
    print("\n" + "=" * 80)
    print("PARAMETER USAGE TESTS")
    print("=" * 80 + "\n")

    files = []

    # FAC tests
    files.append(test_fac_default_optimization())
    files.append(test_fac_full_optimization())
    files.append(test_fac_config_energy())
    files.append(test_fac_radiation_types())
    files.append(test_fac_code_options())

    # AS tests
    files.append(test_as_code_options())

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nGenerated {len(files)} test files:")
    for f in files:
        print(f"  - {f}")
    print()
    print("Review the files to verify all parameters are used correctly!")
    print("=" * 80)


if __name__ == "__main__":
    main()
