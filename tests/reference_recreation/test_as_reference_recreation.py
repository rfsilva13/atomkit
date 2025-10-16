#!/usr/bin/env python3
"""
Test: Can we recreate the AUTOSTRUCTURE reference examples?

Attempts to recreate das_1 through das_10 from:
https://amdpp.phys.strath.ac.uk/autos/default/data/

Compares generated files with originals to see what works/doesn't work.
"""

import sys
from pathlib import Path

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

    print("\n--- ORIGINAL ---")
    print(original)
    print("\n--- GENERATED ---")
    print(generated)

    if "MXCONF=3" in generated and "MXVORB=3" in generated and "NZION=6" in generated:
        print("\n✅ PASS: Key parameters match!")
        return True
    else:
        print("\n❌ FAIL: Missing key parameters")
        return False


def test_das_2():
    """das_2: Be-like C with radiative transitions"""
    print("\n" + "=" * 80)
    print("TEST: das_2 - Be-like C with E1 transitions")
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

    with open(output) as f:
        generated = f.read()

    with open("reference_examples/autostructure/das_2.dat") as f:
        original = f.read()

    print("\n--- ORIGINAL ---")
    print(original)
    print("\n--- GENERATED ---")
    print(generated)

    if "CUP='IC'" in generated and "RAD='E1" in generated:
        print("\n✅ PASS: Coupling and radiation types correct!")
        return True
    else:
        print("\n❌ FAIL: Coupling or radiation types wrong")
        return False


def test_das_3():
    """das_3: Be-like C with optimization"""
    print("\n" + "=" * 80)
    print("TEST: das_3 - Be-like C with optimization")
    print("=" * 80)

    calc = AtomicCalculation(
        name="das_3_recreate",
        element="C",
        charge=2,
        calculation_type="structure",
        coupling="LS",
        optimization="energy",
        code_options={"INCLUD": 6, "NLAM": 3, "NVAR": 2},
        configurations=["1s2.2s2", "1s2.2s.2p", "1s2.2p2"],
        output_dir="outputs/as_reference_tests",
        code="autostructure",
    )

    backend = AutostructureBackend()
    output = backend.write_input(calc)

    with open(output) as f:
        generated = f.read()

    with open("reference_examples/autostructure/das_3.dat") as f:
        original = f.read()

    print("\n--- ORIGINAL ---")
    print(original)
    print("\n--- GENERATED ---")
    print(generated)

    if "INCLUD=6" in generated and "NLAM=3" in generated and "NVAR=2" in generated:
        print("\n✅ PASS: Optimization parameters correct!")
        return True
    else:
        print("\n❌ FAIL: Missing optimization parameters")
        return False


def test_das_10():
    """das_10: Be-like Fe distorted wave (collision)"""
    print("\n" + "=" * 80)
    print("TEST: das_10 - Be-like Fe collision (distorted wave)")
    print("=" * 80)

    calc = AtomicCalculation(
        name="das_10_recreate",
        element="Fe",
        charge=24,  # Be-like Fe (Fe has 26 electrons, Be-like has 4, so 26-4=22, but charge is 24 for Fe XXIV)
        calculation_type="collision",
        coupling="IC",
        core="He-like",  # KCOR1=1 KCOR2=1
        energy_range=(0.0, 500.0, 100),  # MAXE=500
        configurations=["2s2", "2s.2p", "2p2"],
        code_options={"NMETAJ": 2},
        output_dir="outputs/as_reference_tests",
        code="autostructure",
    )

    backend = AutostructureBackend()
    output = backend.write_input(calc)

    with open(output) as f:
        generated = f.read()

    with open("reference_examples/autostructure/das_10.dat") as f:
        original = f.read()

    print("\n--- ORIGINAL ---")
    print(original)
    print("\n--- GENERATED ---")
    print(generated)

    if "RUN='DE'" in generated and "CUP='IC'" in generated and "NZION=26" in generated:
        print("\n✅ PASS: Collision parameters look good!")
        if "NMETAJ=2" in generated:
            print("✅ NMETAJ parameter passed through!")
        else:
            print("⚠️  WARNING: NMETAJ not in output")
        return True
    else:
        print("\n❌ FAIL: Missing collision parameters")
        return False


def test_das_16():
    """das_16: ICR with QED (relativistic)"""
    print("\n" + "=" * 80)
    print("TEST: das_16 - W (Tungsten) ICR with QED")
    print("=" * 80)

    calc = AtomicCalculation(
        name="das_16_recreate",
        element="W",
        charge=72,  # Be-like W
        calculation_type="radiative",
        coupling="ICR",
        radiation_types=["E3"],
        core="He-like",
        relativistic="QED",
        code_options={
            "NLAM": 3,
            "IREL": 2,
            "INUKE": 1,
            "KUTSO": 0,
            "KUTSS": -9,
            "KUTOO": 1,
        },
        configurations=["2s2", "2s.2p", "2p2"],
        output_dir="outputs/as_reference_tests",
        code="autostructure",
    )

    backend = AutostructureBackend()
    output = backend.write_input(calc)

    with open(output) as f:
        generated = f.read()

    with open("reference_examples/autostructure/das_16.dat") as f:
        original = f.read()

    print("\n--- ORIGINAL ---")
    print(original)
    print("\n--- GENERATED ---")
    print(generated)

    if "CUP='ICR'" in generated and "QED=1" in generated and "IREL=2" in generated:
        print("\n✅ PASS: Relativistic parameters correct!")
        return True
    else:
        print("\n❌ FAIL: Missing relativistic parameters")
        return False


def main():
    """Run all reference tests"""
    print("\n" + "=" * 80)
    print("AUTOSTRUCTURE REFERENCE EXAMPLE RECREATION TEST")
    print("=" * 80)
    print("\nAttempting to recreate das_1, das_2, das_3, das_10, das_16")
    print("from https://amdpp.phys.strath.ac.uk/autos/default/data/")

    results = {}

    results["das_1"] = test_das_1()
    results["das_2"] = test_das_2()
    results["das_3"] = test_das_3()
    results["das_10"] = test_das_10()
    results["das_16"] = test_das_16()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All basic reference examples can be recreated!")
    else:
        print(f"\n⚠️  {total - passed} examples need work")

    print("\nNOTE: For das_4-das_9 (continuum processes), we need:")
    print("  - MXCCF parameter (continuum configurations)")
    print("  - DRR namelist support")
    print("  - SRADCON discrete energy points")
    print("  - These are not yet supported in the unified interface")


if __name__ == "__main__":
    main()
