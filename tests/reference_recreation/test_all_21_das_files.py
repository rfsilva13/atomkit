#!/usr/bin/env python3
"""
Comprehensive test: Recreate ALL 21 AUTOSTRUCTURE reference files.

This shows exactly which files can be recreated with the current unified interface
and which features are still missing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from atomkit.core import AtomicCalculation, AutostructureBackend


def test_das_1():
    """das_1: Be-like C structure - Basic structure (simplest)"""
    print("\n" + "=" * 80)
    print("das_1: Be-like C structure")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_1_recreate",
            element="C",
            charge=2,
            calculation_type="structure",
            coupling="LS",
            configurations=["1s2.2s2", "1s2.2s.2p", "1s2.2p2"],
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_2():
    """das_2: Be-like C structure + radiative (IC + E1)"""
    print("\n" + "=" * 80)
    print("das_2: IC coupling + E1 transitions")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_2_recreate",
            element="C",
            charge=2,
            calculation_type="radiative",
            coupling="IC",
            radiation_types=["E1"],
            configurations=["1s2.2s2", "1s2.2s.2p", "1s2.2p2"],
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_3():
    """das_3: Be-like C with optimization (INCLUD, NLAM)"""
    print("\n" + "=" * 80)
    print("das_3: Structure with optimization")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_3_recreate",
            element="C",
            charge=2,
            calculation_type="structure",
            coupling="LS",
            optimization="energy",
            configurations=["1s2.2s2", "1s2.2s.2p", "1s2.2p2"],
            code_options={"NLAM": 5, "NVAR": 6},
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_4():
    """das_4: KLL Auger (needs MXCCF for continuum)"""
    print("\n" + "=" * 80)
    print("das_4: Auger/autoionization (needs MXCCF)")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_4_recreate",
            element="C",
            charge=3,  # Li-like
            calculation_type="autoionization",
            coupling="IC",
            configurations=["1s2.2s"],
            code_options={"MXCCF": 3},  # N+1 continuum configs
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_5():
    """das_5: Photoionization (needs MXCCF)"""
    print("\n" + "=" * 80)
    print("das_5: Photoionization (needs MXCCF)")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_5_recreate",
            element="C",
            charge=3,  # Li-like
            calculation_type="photoionization",
            coupling="LS",
            configurations=["1s2.2s"],
            energy_range=(0.0, 100.0, 50),  # 0-100 eV, 50 points
            code_options={"MXCCF": 2},
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_6():
    """das_6: DR with DRR namelist"""
    print("\n" + "=" * 80)
    print("das_6: Dielectronic recombination (needs DRR)")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_6_recreate",
            element="C",
            charge=3,  # Li-like
            calculation_type="DR",
            coupling="IC",
            core="He-like",  # KCOR1=1, KCOR2=1
            configurations=["2s", "2p"],
            energy_range=(0.0, 50.0, 30),  # Energy range for DR
            code_options={"MXCCF": 3, "DRR": {"NMIN": 3, "NMAX": 20, "LMAX": 4}},
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_7():
    """das_7: Radiative recombination"""
    print("\n" + "=" * 80)
    print("das_7: Radiative recombination")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_7_recreate",
            element="C",
            charge=3,
            calculation_type="RR",
            coupling="LS",
            core="He-like",
            configurations=["2s", "2p"],
            energy_range=(0.0, 30.0, 20),  # Energy range for RR
            code_options={"DRR": {"NMIN": 3, "NMAX": 10, "LMAX": 2}},
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_8():
    """das_8: Inner-shell DR"""
    print("\n" + "=" * 80)
    print("das_8: Inner-shell DR")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_8_recreate",
            element="C",
            charge=3,
            calculation_type="DR",
            coupling="LS",
            configurations=[
                "1s2.2s",
                "1s2.2p",
                "1s.2s2",
                "1s.2s.2p",
                "1s.2p2",
                "2s2.2p",
            ],
            energy_range=(0.0, 100.0, 40),  # Energy range for inner-shell DR
            code_options={"MXCCF": 3, "DRR": {"NMIN": 3, "NMAX": 15}},
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_9():
    """das_9: Collision with meta-stable states (needs NMETA)"""
    print("\n" + "=" * 80)
    print("das_9: Collision with meta-stable (needs NMETA)")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_9_recreate",
            element="H",
            charge=0,
            calculation_type="collision",
            coupling="LS",
            configurations=["1s", "2s", "2p"],
            code_options={"NMETA": 3, "MINLT": 0, "MAXLT": 11},
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_10():
    """das_10: Collision with NMETAJ (IC coupling)"""
    print("\n" + "=" * 80)
    print("das_10: Collision with NMETAJ")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_10_recreate",
            element="Fe",
            charge=22,  # Be-like
            calculation_type="collision",
            coupling="IC",
            core="He-like",
            configurations=["2s2", "2s.2p", "2p2"],
            code_options={"NMETAJ": 2},
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_11():
    """das_11: Neutral Cr with NAST"""
    print("\n" + "=" * 80)
    print("das_11: Neutral Cr (complex config)")
    print("=" * 80)

    try:
        # This is complex - Cr with specific electron configs
        calc = AtomicCalculation(
            name="das_11_recreate",
            element="Cr",
            charge=0,
            calculation_type="radiative",
            coupling="LS",
            radiation_types=["E1"],
            configurations=[
                # Simplified version of the reference
                "1s2.2s2.2p6.3s2.3p6.3d5.4s1",
                "1s2.2s2.2p6.3s2.3p6.3d4.4s2",
            ],
            code_options={
                "KCOR1": 1,
                "KCOR2": -5,  # Negative means up to n=5
                "NAST": 2,
            },
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_12():
    """das_12: Custom STO orbitals (needs SRADWIN)"""
    print("\n" + "=" * 80)
    print("das_12: Custom STO orbitals")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_12_recreate",
            element="Be",
            charge=1,
            calculation_type="structure",
            configurations=[
                "1s2.2s",
                "1s2.2p",
                "1s2.3s",
                "1s2.3p",
                "1s2.3d",
                "1s2.4s",
                "1s2.4p",
                "1s2.4d",
                "1s2.4f",
                "1s.2s2",
                "1s.2s.2p",
                "1s.2p2",
            ],
            custom_orbitals_file="custom_orbitals.dat",  # Signals to use SRADWIN
            code_options={"MXVORB": 10, "MXCONF": 12, "SRADWIN_KEY": -10},
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_13():
    """das_13: Auto config generation with ICFG"""
    print("\n" + "=" * 80)
    print("das_13: Auto config generation (ICFG)")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_13_recreate",
            element="Fe",
            charge=22,  # Be-like Fe
            calculation_type="structure",
            auto_generate_configs=1,  # Single excitations
            min_occupation=[0] * 14,
            max_occupation=[2] * 14,
            base_config_promotions=([2] + [0] * 13, 2),  # 1s2, 2 promotions
            core="He-like",
            code_options={"MXVORB": -14, "MXCONF": 1},
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_14():
    """das_14: Pseudo-state expansion with NXTRA, LXTRA"""
    print("\n" + "=" * 80)
    print("das_14: Pseudo-state expansion (NXTRA/LXTRA)")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_14_recreate",
            element="C",
            charge=1,  # B-like
            calculation_type="structure",
            coupling="LS",
            auto_generate_configs=1,  # With ICFG
            n_extra_orbitals=10,  # NXTRA=10
            l_max_extra=2,  # LXTRA=2 (up to d orbitals)
            orthogonality="LPS",  # Laguerre pseudo-states
            min_occupation=[0, 0],
            max_occupation=[2, 2],
            base_config_promotions=([2, 1], 1),  # 1s2 2s, 1 promotion
            core="He-like",
            code_options={
                "MXVORB": 2,
                "MXCONF": 2,
                "NLAM": 4,
                "RADOUT": "YES",
            },
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_15():
    """das_15: R-matrix support files"""
    print("\n" + "=" * 80)
    print("das_15: R-matrix (needs KUTSO for R-matrix)")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_15_recreate",
            element="C",
            charge=2,
            calculation_type="radiative",
            coupling="IC",
            radiation_types=["E1", "E2", "M1", "M2"],  # 'ALL'
            configurations=["1s2.2s2", "1s2.2s.2p", "1s2.2p2"],
            code_options={"KUTSO": 0},  # R-matrix flag
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_16():
    """das_16: Relativistic with QED"""
    print("\n" + "=" * 80)
    print("das_16: ICR coupling + E3 + relativistic")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_16_recreate",
            element="W",  # Tungsten, Z=74
            charge=72,  # He-like
            calculation_type="radiative",
            coupling="ICR",
            radiation_types=["E3"],
            relativistic="retardation",
            qed_corrections=True,
            core="He-like",
            configurations=["2s", "2s.2p"],
            code_options={
                "KUTSO": 0,
                "KUTSS": -9,
                "KUTOO": 1,
                "NLAM": 3,
                "INUKE": 1,  # Fermi nuclear model
            },
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_17():
    """das_17: Plasma potential (PPOT, NDEN)"""
    print("\n" + "=" * 80)
    print("das_17: Plasma potential")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_17_recreate",
            element="C",
            charge=2,  # Be-like
            calculation_type="structure",
            configurations=["1s2.2s2", "1s2.2s.2p", "1s2.2p2"],
            plasma_potential="KS",  # Kohn-Sham
            plasma_density=1e24,  # cm^-3
            code_options={
                "MXCONF": 3,
                "MXVORB": 3,
                "NLAM": 3,
                "RAD": "YES",
                "plasma_temperature": -1e6,  # K
            },
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_18():
    """das_18: CA coupling with auto gen and NMETA"""
    print("\n" + "=" * 80)
    print("das_18: CA coupling + ICFG + NMETA")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_18_recreate",
            element="Ne",
            charge=8,  # He-like
            calculation_type="structure",
            coupling="CA",
            auto_generate_configs=2,  # Double excitations
            min_occupation=[0] * 12,
            max_occupation=[2] * 12,
            base_config_promotions=([2] + [0] * 11, 2),
            core="He-like",
            code_options={
                "MXVORB": -12,
                "MXCONF": 1,
                "NMETA": 10,
                "NLAM": 5,
            },
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_19():
    """das_19: Simplified Relaxed basis (BASIS='SRLX')"""
    print("\n" + "=" * 80)
    print("das_19: SRLX basis")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_19_recreate",
            element="C",
            charge=4,  # He-like
            calculation_type="DR",
            coupling="IC",
            orbital_basis="SRLX",  # Simplified relaxed basis
            configurations=["2s", "2s.2p"],
            energy_range=(0.0, 25.0, 15),
            code_options={
                "MXVORB": 3,
                "MXCONF": 2,
                "MXCCF": 1,
                "NLAM": 3,
                "PRINT": "FORM",
                "DRR": {"NMIN": 3, "NMAX": 15, "LMIN": 0, "LMAX": 0},
            },
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_20():
    """das_20: Inner-shell DR (CA coupling)"""
    print("\n" + "=" * 80)
    print("das_20: Inner-shell DR with CA coupling")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_20_recreate",
            element="C",
            charge=3,
            calculation_type="DR",
            coupling="CA",
            configurations=[
                "1s2.2s",
                "1s2.2p",
                "1s.2s2",
                "1s.2s.2p",
                "1s.2p2",
                "2s2.2p",
            ],
            energy_range=(0.0, 100.0, 40),  # Energy range for DR
            code_options={"MXCCF": 3, "DRR": {"NMIN": 3, "NMAX": 15}},
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


def test_das_21():
    """das_21: DR with ICFG for N+1 generation"""
    print("\n" + "=" * 80)
    print("das_21: DR with ICFG")
    print("=" * 80)

    try:
        calc = AtomicCalculation(
            name="das_21_recreate",
            element="C",
            charge=3,  # Li-like
            calculation_type="DR",
            coupling="IC",
            auto_generate_configs=2,  # Double excitations for N+1
            min_occupation=[0] * 10,
            max_occupation=[2] * 10,
            base_config_promotions=([2, 1] + [0] * 8, 1),  # 1s2 2s + 1 promotion
            energy_range=(0.0, 100.0, 40),
            core="He-like",
            code_options={
                "MXVORB": -10,
                "MXCONF": 2,
                "MXCCF": 2,
                "DRR": {"NMIN": 3, "NMAX": 15, "LMIN": 0, "LMAX": 10},
            },
            output_dir="outputs/all_21_das",
            code="autostructure",
        )
        backend = AutostructureBackend()
        output = backend.write_input(calc)
        print(f"✅ SUCCESS: {output}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, str(e)


if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE TEST: Recreate ALL 21 AUTOSTRUCTURE Reference Files")
    print("=" * 80)

    tests = [
        ("das_1", test_das_1),
        ("das_2", test_das_2),
        ("das_3", test_das_3),
        ("das_4", test_das_4),
        ("das_5", test_das_5),
        ("das_6", test_das_6),
        ("das_7", test_das_7),
        ("das_8", test_das_8),
        ("das_9", test_das_9),
        ("das_10", test_das_10),
        ("das_11", test_das_11),
        ("das_12", test_das_12),
        ("das_13", test_das_13),
        ("das_14", test_das_14),
        ("das_15", test_das_15),
        ("das_16", test_das_16),
        ("das_17", test_das_17),
        ("das_18", test_das_18),
        ("das_19", test_das_19),
        ("das_20", test_das_20),
        ("das_21", test_das_21),
    ]

    results = []
    for name, test_func in tests:
        success, error = test_func()
        results.append((name, success, error))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = []
    failed = []
    missing_features = {}

    for name, success, error in results:
        if success:
            successful.append(name)
            print(f"✅ {name}: SUCCESS")
        else:
            failed.append(name)
            if error and "Missing feature" in error:
                feature = error.split(":")[1].strip()
                if feature not in missing_features:
                    missing_features[feature] = []
                missing_features[feature].append(name)
                print(f"❌ {name}: {error}")
            else:
                print(f"❌ {name}: {error if error else 'Unknown error'}")

    print(f"\n{'='*80}")
    print(f"RESULTS: {len(successful)}/21 files can be recreated")
    print(f"{'='*80}")

    if missing_features:
        print(f"\nMISSING FEATURES NEEDED:")
        for feature, files in missing_features.items():
            print(f"  • {feature}")
            print(f"    Required by: {', '.join(files)}")

    sys.exit(0 if len(successful) == 21 else 1)
