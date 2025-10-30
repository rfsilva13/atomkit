"""
Tests for unified atomic calculation interface.

Tests the code-agnostic AtomicCalculation class and backend adapters.
"""

import pytest
from pathlib import Path

from src.atomkit.core import (
    AtomicCalculation,
    CouplingScheme,
    RelativisticTreatment,
    OptimizationTarget,
    CalculationType,
    RadiationType,
    AutostructureBackend,
    FACBackend,
)


class TestPhysicalSpecs:
    """Test physical specification constants."""

    def test_coupling_schemes(self):
        """Test coupling scheme constants."""
        assert CouplingScheme.LS == "LS"
        assert CouplingScheme.IC == "IC"
        assert CouplingScheme.JJ == "jj"
        assert CouplingScheme.LSJ == "LSJ"

    def test_relativistic_treatment(self):
        """Test relativistic treatment constants."""
        assert RelativisticTreatment.NONE == "none"
        assert RelativisticTreatment.BREIT == "Breit"
        assert RelativisticTreatment.DIRAC == "Dirac"

    def test_optimization_target(self):
        """Test optimization target constants."""
        assert OptimizationTarget.ENERGY == "energy"
        assert OptimizationTarget.POTENTIAL == "potential"
        assert OptimizationTarget.LAMBDA == "lambda"
        assert OptimizationTarget.NONE is None

    def test_calculation_types(self):
        """Test calculation type constants."""
        assert CalculationType.STRUCTURE == "structure"
        assert CalculationType.DR == "DR"
        assert CalculationType.PHOTOIONIZATION == "photoionization"

    def test_radiation_types(self):
        """Test radiation type constants."""
        assert RadiationType.E1 == "E1"
        assert RadiationType.M1 == "M1"


class TestAtomicCalculation:
    """Test AtomicCalculation class."""

    def test_basic_creation(self):
        """Test basic calculation creation."""
        calc = AtomicCalculation(element="Fe", charge=15, calculation_type="structure")

        assert calc.element == "Fe"
        assert calc.charge == 15
        assert calc.calculation_type == "structure"
        assert calc.coupling == "LS"  # Default
        assert calc.code == "autostructure"  # Default

    def test_auto_naming(self):
        """Test automatic name generation."""
        calc = AtomicCalculation(element="Fe", charge=15, calculation_type="structure")

        assert calc.name == "fe_15_structure"

    def test_custom_name(self):
        """Test custom naming."""
        calc = AtomicCalculation(
            element="Fe", charge=15, calculation_type="structure", name="my_calc"
        )

        assert calc.name == "my_calc"

    def test_physical_specifications(self):
        """Test setting physical specifications."""
        calc = AtomicCalculation(
            element="Fe",
            charge=15,
            calculation_type="structure",
            coupling="IC",
            relativistic="Breit",
            qed_corrections=True,
            optimization="energy",
        )

        assert calc.coupling == "IC"
        assert calc.relativistic == "Breit"
        assert calc.qed_corrections is True
        assert calc.optimization == "energy"

    def test_code_options(self):
        """Test code-specific options."""
        calc = AtomicCalculation(
            element="Fe",
            charge=15,
            calculation_type="structure",
            code="autostructure",
            code_options={"SCFRAC": 0.85, "NLAM": 5},
        )

        assert calc.code_options["SCFRAC"] == 0.85
        assert calc.code_options["NLAM"] == 5

    def test_energy_range_required_for_continuum(self):
        """Test that energy_range is required for PI/DR/RR."""
        with pytest.raises(ValueError, match="requires energy_range"):
            AtomicCalculation(
                element="Fe",
                charge=15,
                calculation_type="photoionization",
                # Missing energy_range!
            )

    def test_energy_range_provided(self):
        """Test continuum calculation with energy range."""
        calc = AtomicCalculation(
            element="Fe", charge=15, calculation_type="DR", energy_range=(0, 100, 1000)
        )

        assert calc.energy_range == (0, 100, 1000)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        calc = AtomicCalculation(
            element="Fe", charge=15, calculation_type="structure", coupling="IC"
        )

        data = calc.to_dict()
        assert data["element"] == "Fe"
        assert data["charge"] == 15
        assert data["coupling"] == "IC"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "element": "Fe",
            "charge": 15,
            "calculation_type": "structure",
            "coupling": "IC",
        }

        calc = AtomicCalculation.from_dict(data)
        assert calc.element == "Fe"
        assert calc.charge == 15
        assert calc.coupling == "IC"

    def test_repr(self):
        """Test string representation."""
        calc = AtomicCalculation(element="Fe", charge=15, calculation_type="structure")

        repr_str = repr(calc)
        assert "Fe" in repr_str
        assert "15" in repr_str
        assert "structure" in repr_str


class TestAutostructureBackend:
    """Test AUTOSTRUCTURE backend adapter."""

    @pytest.fixture
    def backend(self):
        """Create backend instance."""
        return AutostructureBackend()

    def test_capabilities(self, backend):
        """Test capability reporting."""
        caps = backend.capabilities()

        assert "LS" in caps["coupling_schemes"]
        assert "IC" in caps["coupling_schemes"]
        assert "Breit" in caps["relativistic"]
        assert caps["qed"] is True
        assert "energy" in caps["optimization"]

    def test_coupling_translation_ls(self, backend):
        """Test LS coupling translation."""
        calc = AtomicCalculation(
            element="Fe", charge=15, calculation_type="structure", coupling="LS"
        )

        params = backend.translate(calc)
        assert params["CUP"] == "LS"

    def test_coupling_translation_ic(self, backend):
        """Test IC coupling translation."""
        calc = AtomicCalculation(
            element="Fe", charge=15, calculation_type="structure", coupling="IC"
        )

        params = backend.translate(calc)
        assert params["CUP"] == "IC"

    def test_coupling_translation_jj(self, backend):
        """Test jj coupling approximation."""
        calc = AtomicCalculation(
            element="Fe",
            charge=15,
            calculation_type="structure",
            coupling="jj",  # AS approximates with IC
        )

        params = backend.translate(calc)
        assert params["CUP"] == "IC"  # AS uses IC for jj

    def test_relativistic_breit(self, backend):
        """Test Breit interaction translation."""
        calc = AtomicCalculation(
            element="Fe", charge=15, calculation_type="structure", relativistic="Breit"
        )

        params = backend.translate(calc)
        assert params["CUP"] == "LS"  # Default coupling
        assert params["IBREIT"] == 1  # Breit interaction
        # IRTARD is only set for "retardation", not "Breit"

    def test_qed_corrections(self, backend):
        """Test QED corrections translation."""
        calc = AtomicCalculation(
            element="Fe", charge=15, calculation_type="structure", qed_corrections=True
        )

        params = backend.translate(calc)
        assert params["QED"] == 1

    def test_optimization_energy(self, backend):
        """Test energy optimization translation."""
        calc = AtomicCalculation(
            element="Fe", charge=15, calculation_type="structure", optimization="energy"
        )

        params = backend.translate(calc)
        assert params["INCLUD"] == 10
        assert params["NLAM"] == 5

    def test_calculation_type_structure(self, backend):
        """Test structure calculation type."""
        calc = AtomicCalculation(element="Fe", charge=15, calculation_type="structure")

        params = backend.translate(calc)
        assert params["RUN"] == "  "

    def test_calculation_type_dr(self, backend):
        """Test DR calculation type."""
        calc = AtomicCalculation(
            element="Fe", charge=15, calculation_type="DR", energy_range=(0, 100, 1000)
        )

        params = backend.translate(calc)
        assert params["RUN"] == "DR"

    def test_core_he_like(self, backend):
        """Test He-like core translation."""
        calc = AtomicCalculation(
            element="Fe", charge=15, calculation_type="structure", core="He-like"
        )

        params = backend.translate(calc)
        assert params["KCOR1"] == 1
        assert params["KCOR2"] == 1

    def test_core_ne_like(self, backend):
        """Test Ne-like core translation."""
        calc = AtomicCalculation(
            element="Fe", charge=15, calculation_type="structure", core="Ne-like"
        )

        params = backend.translate(calc)
        assert params["KCOR1"] == 1
        assert params["KCOR2"] == 3

    def test_energy_range(self, backend):
        """Test energy range translation."""
        calc = AtomicCalculation(
            element="Fe",
            charge=15,
            calculation_type="photoionization",
            energy_range=(0, 100, 500),
        )

        params = backend.translate(calc)
        assert params["EMIN"] == 0
        assert params["EMAX"] == 100
        assert params["MENG"] == -500  # Negative for log spacing

    def test_code_options_override(self, backend):
        """Test that code_options override defaults."""
        calc = AtomicCalculation(
            element="Fe",
            charge=15,
            calculation_type="structure",
            optimization="energy",
            code_options={"NLAM": 10},  # Override default 5
        )

        params = backend.translate(calc)
        assert params["NLAM"] == 10  # User value, not default

    def test_write_input(self, backend, tmp_path):
        """Test input file generation."""
        calc = AtomicCalculation(
            element="Fe",
            charge=15,
            calculation_type="structure",
            coupling="LS",
            output_dir=str(tmp_path),
        )

        filepath = backend.write_input(calc)

        assert filepath.exists()
        assert filepath.suffix == ".dat"

        content = filepath.read_text()
        assert "A.S." in content  # AS header
        assert "&SALGEB" in content
        assert "CUP='LS'" in content
        assert "&SMINIM" in content
        assert "NZION=26" in content  # Nuclear charge (Fe), not ionic charge


class TestFACBackend:
    """Test FAC backend adapter."""

    @pytest.fixture
    def backend(self):
        """Create backend instance."""
        return FACBackend()

    def test_capabilities(self, backend):
        """Test capability reporting."""
        caps = backend.capabilities()

        assert caps["coupling_schemes"] == ["jj"]  # FAC only does jj
        assert "Dirac" in caps["relativistic"]
        assert caps["qed"] is True
        assert "potential" in caps["optimization"]
        assert "Always jj-coupling" in caps["limitations"][0]

    def test_translate_always_jj(self, backend):
        """Test that FAC is always jj (warnings issued)."""
        calc = AtomicCalculation(
            element="Fe",
            charge=15,
            calculation_type="structure",
            coupling="LS",  # User requests LS
            code="fac",
        )

        # FAC will warn but still generate (uses jj)
        assert len(calc.warnings) > 0
        assert any("coupling" in w.lower() for w in calc.warnings)

    def test_breit_interaction(self, backend):
        """Test Breit interaction translation."""
        calc = AtomicCalculation(
            element="Fe",
            charge=15,
            calculation_type="structure",
            relativistic="Breit",
            code="fac",
        )

        params = backend.translate(calc)
        fac_calls = params["fac_calls"]

        assert any("SetBreit(1)" in call for call in fac_calls)

    def test_qed_corrections(self, backend):
        """Test QED corrections translation."""
        calc = AtomicCalculation(
            element="Fe",
            charge=15,
            calculation_type="structure",
            qed_corrections=True,
            code="fac",
        )

        params = backend.translate(calc)
        fac_calls = params["fac_calls"]

        assert any("SetVP(1)" in call for call in fac_calls)
        assert any("SetSE(1)" in call for call in fac_calls)

    def test_optimization(self, backend):
        """Test optimization translation."""
        calc = AtomicCalculation(
            element="Fe",
            charge=15,
            calculation_type="structure",
            optimization="potential",
            code="fac",
        )

        params = backend.translate(calc)
        fac_calls = params["fac_calls"]

        assert any("OptimizeRadial" in call for call in fac_calls)

    def test_write_input(self, backend, tmp_path):
        """Test FAC input file generation."""
        from atomkit import Configuration

        # Create a simple calculation with configurations so we get Structure() output
        ground = Configuration.from_string("1s2 2s2 2p6")

        calc = AtomicCalculation(
            element="Fe",
            charge=15,
            calculation_type="structure",
            code="fac",
            configurations=[ground],
            output_dir=str(tmp_path),
        )

        filepath = backend.write_input(calc)

        assert filepath.exists()
        assert filepath.suffix == ".sf"

        content = filepath.read_text()
        # SFAC files are command files for the sfac executable, not Python scripts
        assert "SetAtom('Fe')" in content
        assert "Structure" in content  # Should be present with configurations


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_same_calc_both_codes(self, tmp_path):
        """Test same calculation generates files for both codes."""
        for code in ["autostructure", "fac"]:
            calc = AtomicCalculation(
                element="Fe",
                charge=15,
                calculation_type="structure",
                coupling="IC",
                relativistic="Breit",
                qed_corrections=True,
                code=code,
                output_dir=str(tmp_path),
            )

            filepath = calc.write_input(verbose=False)
            assert filepath.exists()

    def test_dr_calculation_both_codes(self, tmp_path):
        """Test DR calculation for both codes."""
        for code in ["autostructure", "fac"]:
            calc = AtomicCalculation(
                element="Fe",
                charge=15,
                calculation_type="DR",
                energy_range=(0, 100, 1000),
                core="Ne-like",
                code=code,
                output_dir=str(tmp_path),
            )

            filepath = calc.write_input(verbose=False)
            assert filepath.exists()

            if code == "autostructure":
                # AS should have DRR namelist
                content = filepath.read_text()
                assert (
                    "&DRR" in content or "DRR" in content
                )  # May be in different format
            else:
                # FAC should have AI/RR calls
                content = filepath.read_text()
                assert "AI" in content or "RecStates" in content
