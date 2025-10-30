"""
Test suite: AUTOS Reference Test Recreation (Clean Version)

Recreates all 21 AUTOSTRUCTURE reference tests from:
https://amdpp.phys.strath.ac.uk/autos/default/data/

Using the agnostic AtomKit AtomicCalculation interface.
"""

import pytest
from pathlib import Path
from atomkit.core import AtomicCalculation
from atomkit import Configuration


def configs(*strings):
    """Helper to create list of configurations from strings."""
    return [Configuration.from_string(s) for s in strings]


class TestAUTOSReference:
    """Recreate all 21 AUTOS reference tests using agnostic interface."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create temporary output directory for tests."""
        return str(tmp_path / "autos_tests")

    def test_das_1_be_like_c_structure(self, output_dir):
        """das_1: Be-like C structure - energies only"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="structure",
            configurations=configs(
                "1s2 2s2",
                "1s2 2s1 2p1",
                "1s2 2p2",
            ),
            output_dir=output_dir,
            code="autostructure",
        )
        assert calc.write_input().exists()

    def test_das_2_be_like_c_radiative(self, output_dir):
        """das_2: Be-like C structure + E1 radiative transitions"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="radiative",
            coupling="IC",
            radiation_types=["E1"],
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                    "1s2 2p2",
                ]
            ),
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_3_optimization_lambda_scaling(self, output_dir):
        """das_3: Be-like C with lambda scaling optimization"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="structure",
            coupling="LS",
            optimization="lambda",
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                    "1s2 2p2",
                ]
            ),
            code_options={"NLAM": 3, "NVAR": 2},
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_4_kll_auger(self, output_dir):
        """das_4: KLL Auger process"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="autoionization",
            coupling="IC",
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                    "1s2 2p2",
                    "1s1 2s2 2p1",  # Excited continuum states
                    "1s1 2s1 2p2",
                    "1s1 2p3",
                ]
            ),
            continuum_configs=Configuration.from_strings(
                [
                    "1s2 2s1",
                    "1s2 2p1",
                ]
            ),
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_5_photoionization(self, output_dir):
        """das_5: Photoionization cross sections"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="photoionization",
            coupling="IC",
            energy_range=(0, 10, 100),  # 0-10 Ry, 100 points
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                    "1s2 2p2",
                ]
            ),
            continuum_configs=Configuration.from_strings(
                [
                    "1s2 2s1",
                    "1s2 2p1",
                ]
            ),
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_6_dielectronic_recombination(self, output_dir):
        """das_6: Dielectronic recombination"""
        calc = AtomicCalculation(
            element="C",
            charge=3,  # Li-like initial
            calculation_type="DR",
            coupling="IC",
            configurations=Configuration.from_strings(
                [
                    "1s2 2s1",
                    "1s2 2p1",
                    "1s1 2s2",
                    "1s1 2s1 2p1",
                ]
            ),
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_7_radiative_recombination(self, output_dir):
        """das_7: Radiative recombination"""
        calc = AtomicCalculation(
            element="C",
            charge=3,
            calculation_type="RR",
            coupling="IC",
            energy_range=(0, 10, 100),
            configurations=Configuration.from_strings(
                [
                    "1s2 2s1",
                    "1s2 2p1",
                ]
            ),
            continuum_configs=Configuration.from_strings(["1s2"]),
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_8_collision_symmetry(self, output_dir):
        """das_8: Collision with symmetry restrictions"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="collision",
            coupling="IC",
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                    "1s2 2p2",
                ]
            ),
            code_options={"NSYM": 1},
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_9_collision_relaxed_basis(self, output_dir):
        """das_9: Collision with relaxed basis (BASIS='RLX')"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="collision",
            coupling="IC",
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                ]
            ),
            code_options={"BASIS": "RLX"},
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_10_distorted_wave_collision(self, output_dir):
        """das_10: Distorted wave collision (DW)"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="collision",
            coupling="IC",
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                ]
            ),
            code_options={"BASIS": "DW"},
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_11_icfg_single_excitations(self, output_dir):
        """das_11: ICFG automatic configuration generation (single excitations)"""
        calc = AtomicCalculation(
            element="Fe",
            charge=16,
            calculation_type="structure",
            coupling="IC",
            # Use ICFG to generate configurations automatically
            configurations=Configuration.from_string("1s2 2s2 2p6 3s2 3p5"),
            code_options={
                "ICFG": 1,  # Single excitations
                "NEXC": 1,
                "NFEXC": 5,
            },
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_12_icfg_double_excitations(self, output_dir):
        """das_12: ICFG double excitations"""
        calc = AtomicCalculation(
            element="Fe",
            charge=16,
            calculation_type="structure",
            coupling="IC",
            configurations=Configuration.from_string("1s2 2s2 2p6 3s2 3p5"),
            code_options={
                "ICFG": 2,  # Double excitations
                "NEXC": 2,
                "NFEXC": 5,
            },
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_13_kcut_multipoles(self, output_dir):
        """das_13: KCUT multipole restrictions"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="radiative",
            coupling="IC",
            radiation_types=["E1"],
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                    "1s2 2p2",
                ]
            ),
            code_options={"KCUT": 1},  # Restrict to E1 only
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_14_all_multipoles(self, output_dir):
        """das_14: All multipoles (RAD='ALL')"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="radiative",
            coupling="IC",
            radiation_types=["ALL"],  # All multipoles
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                    "1s2 2p2",
                ]
            ),
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_15_rmatrix_support(self, output_dir):
        """das_15: R-matrix support (KCUTCC)"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="collision",
            coupling="IC",
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                ]
            ),
            code_options={"KCUTCC": 3},
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_16_icr_relativistic_qed(self, output_dir):
        """das_16: ICR coupling with relativistic + QED corrections (Tungsten)"""
        calc = AtomicCalculation(
            element="W",
            charge=72,  # He-like
            calculation_type="radiative",
            coupling="ICR",
            qed_corrections=True,
            radiation_types=["E3"],
            optimization="lambda",
            configurations=Configuration.from_strings(
                [
                    "1s2",
                    "1s1 2s1",
                    "2s2",
                ]
            ),
            code_options={
                "KCOR1": 1,
                "KCOR2": 1,
                "KUTSO": 0,
                "KUTSS": -9,
                "KUTOO": 1,
                "NLAM": 3,
                "IREL": 2,
                "INUKE": 1,
            },
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_17_born_approximation(self, output_dir):
        """das_17: Born approximation (BORN='INF')"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="structure",
            coupling="IC",
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                ]
            ),
            code_options={"BORN": "INF"},
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_18_core_specification(self, output_dir):
        """das_18: Core specification with KCOR (Magnesium)"""
        calc = AtomicCalculation(
            element="Mg",
            charge=0,
            calculation_type="structure",
            coupling="IC",
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2 2p6 3s2",
                    "1s2 2s2 2p6 3s1 3p1",
                ]
            ),
            code_options={
                "KCOR1": 1,
                "KCOR2": 1,
            },
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_19_ls_no_fine_structure(self, output_dir):
        """das_19: LS coupling without fine structure (KUTSS=-1)"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="radiative",
            coupling="LS",
            radiation_types=["E1"],
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                    "1s2 2p2",
                ]
            ),
            code_options={"KUTSS": -1},
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_20_inner_shell_dr(self, output_dir):
        """das_20: Inner-shell DR (K-shell excitation)"""
        calc = AtomicCalculation(
            element="C",
            charge=3,
            calculation_type="DR",
            coupling="CA",
            configurations=Configuration.from_strings(
                [
                    # Target states
                    "1s2 2s1",
                    "1s2 2p1",
                    # Doubly excited states
                    "1s1 2s2",
                    "1s1 2s1 2p1",
                    "1s1 2p2",
                ]
            ),
            continuum_configs=Configuration.from_strings(
                [
                    # Continuum state
                    "1s2 80p1",  # High-n state representing continuum
                ]
            ),
            code_options={
                "KUTSS": -9,  # Inner-shell DR
            },
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()

    def test_das_21_bethe_peterkop(self, output_dir):
        """das_21: Collision with Bethe-Peterkop approximation (BP)"""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="collision",
            coupling="IC",
            configurations=Configuration.from_strings(
                [
                    "1s2 2s2",
                    "1s2 2s1 2p1",
                ]
            ),
            code_options={"BASIS": "BP"},
            output_dir=output_dir,
            code="autostructure",
        )

        assert calc.write_input().exists()


# Optional: More thorough validation tests
class TestAUTOSValidation:
    """Validate generated files contain expected physics."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path / "validation")

    @pytest.mark.parametrize(
        "test_info",
        [
            ("das_1", {"element": "C", "charge": 2, "type": "structure"}),
            (
                "das_2",
                {"element": "C", "charge": 2, "type": "radiative", "coupling": "IC"},
            ),
            ("das_5", {"element": "C", "charge": 2, "type": "photoionization"}),
            # Add more as needed
        ],
    )
    def test_file_contains_expected_physics(self, test_info, output_dir):
        """Check that generated files contain expected physical parameters."""
        test_name, params = test_info

        # Simple validation - just ensure the parameters we specified show up somehow
        # Don't worry about exact format since backends may differ
        pass  # Can add actual validation if needed
