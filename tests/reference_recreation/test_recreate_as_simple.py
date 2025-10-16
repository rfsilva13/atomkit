#!/usr/bin/env python3
"""
Simple test to recreate AUTOSTRUCTURE reference files.
Run with: conda activate atomkit && python test_recreate_as_simple.py
"""

import sys
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent / "src"))

from atomkit.core import AtomicCalculation, AutostructureBackend


def test_das_1():
    """das_1: Be-like C structure - simplest case"""
    print("\n" + "=" * 80)
    print("TEST: das_1 - Be-like C structure (simplest)")
    print("=" * 80)

    calc = AtomicCalculation(
        name="das_1_recreate",
        element="C",
        charge=2,  # Be-like = C2+ (C has 6 electrons, Be-like has 4)
        calculation_type="structure",
        coupling="LS",  # Default
        configurations=["1s2.2s2", "1s2.2s.2p", "1s2.2p2"],
        output_dir="outputs/as_reference_tests",
        code="autostructure",
    )

    backend = AutostructureBackend()
    output = backend.write_input(calc)

    print(f"✓ Generated: {output}")

    # Compare with original
    with open(output) as f:
        generated = f.read()

    ref_path = (
        Path(__file__).parent / "reference_examples" / "autostructure" / "das_1.dat"
    )
    with open(ref_path) as f:
        original = f.read()

    print("\n--- ORIGINAL das_1.dat ---")
    print(original)

    print("\n--- GENERATED das_1_recreate.dat ---")
    print(generated)

    print("\n--- COMPARISON ---")
    print(f"Original length: {len(original)} chars")
    print(f"Generated length: {len(generated)} chars")

    # Remove whitespace and compare
    orig_clean = original.strip()
    gen_clean = generated.strip()

    if orig_clean == gen_clean:
        print("✅ EXACT MATCH!")
        return True
    else:
        print("❌ MISMATCH - showing differences:")
        orig_lines = orig_clean.split("\n")
        gen_lines = gen_clean.split("\n")

        for i, (o, g) in enumerate(zip(orig_lines, gen_lines)):
            if o != g:
                print(f"  Line {i+1}:")
                print(f"    Original:  {repr(o)}")
                print(f"    Generated: {repr(g)}")

        if len(orig_lines) != len(gen_lines):
            print(f"  Different number of lines: {len(orig_lines)} vs {len(gen_lines)}")

        return False


def test_das_2():
    """das_2: Structure + radiative transitions"""
    print("\n" + "=" * 80)
    print("TEST: das_2 - IC coupling + E1 transitions")
    print("=" * 80)

    calc = AtomicCalculation(
        name="das_2_recreate",
        element="C",
        charge=2,
        calculation_type="radiative",
        coupling="IC",
        radiation_types=["E1"],
        configurations=["1s2.2s2", "1s2.2s.2p", "1s2.2p2"],
        output_dir="outputs/as_reference_tests",
        code="autostructure",
    )

    backend = AutostructureBackend()
    output = backend.write_input(calc)

    print(f"✓ Generated: {output}")

    # Compare with original
    with open(output) as f:
        generated = f.read()

    ref_path = (
        Path(__file__).parent / "reference_examples" / "autostructure" / "das_2.dat"
    )
    with open(ref_path) as f:
        original = f.read()

    print("\n--- ORIGINAL das_2.dat ---")
    print(original)

    print("\n--- GENERATED das_2_recreate.dat ---")
    print(generated)

    print("\n--- COMPARISON ---")
    orig_clean = original.strip()
    gen_clean = generated.strip()

    if orig_clean == gen_clean:
        print("✅ EXACT MATCH!")
        return True
    else:
        print("❌ MISMATCH")
        return False


def test_das_16():
    """das_16: Relativistic calculation with QED"""
    print("\n" + "=" * 80)
    print("TEST: das_16 - ICR coupling + E3 + relativistic corrections")
    print("=" * 80)

    calc = AtomicCalculation(
        name="das_16_recreate",
        element="W",  # Tungsten, Z=74
        charge=72,  # He-like
        calculation_type="radiative",
        coupling="ICR",
        radiation_types=["E3"],
        relativistic="retardation",  # IREL=2, IRTARD=1
        qed_corrections=True,  # QED=1
        nuclear_model="fermi",  # INUKE=1
        configurations=["1s2", "1s.2s"],
        code_options={
            "KCOR1": 1,
            "KCOR2": 1,
            "KUTSO": 0,
            "KUTSS": -9,
            "KUTOO": 1,
            "NLAM": 3,
        },
        output_dir="outputs/as_reference_tests",
        code="autostructure",
    )

    backend = AutostructureBackend()
    output = backend.write_input(calc)

    print(f"✓ Generated: {output}")

    # Compare with original
    with open(output) as f:
        generated = f.read()

    ref_path = (
        Path(__file__).parent / "reference_examples" / "autostructure" / "das_16.dat"
    )
    with open(ref_path) as f:
        original = f.read()

    print("\n--- ORIGINAL das_16.dat ---")
    print(original)

    print("\n--- GENERATED das_16_recreate.dat ---")
    print(generated)

    print("\n--- COMPARISON ---")
    orig_clean = original.strip()
    gen_clean = generated.strip()

    if orig_clean == gen_clean:
        print("✅ EXACT MATCH!")
        return True
    else:
        print("❌ MISMATCH")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("AUTOSTRUCTURE Reference File Recreation Test")
    print("=" * 80)

    results = []

    # Test basic examples that should work with current implementation
    results.append(("das_1", test_das_1()))
    results.append(("das_2", test_das_2()))
    results.append(("das_16", test_das_16()))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)
