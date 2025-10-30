"""
Minimal essential tests for unified AtomicCalculation interface.
"""

import pytest
from pathlib import Path
from atomkit.core import AtomicCalculation
from atomkit import Configuration


class TestBasicCalculation:
    """Test basic calculation creation."""

    def test_simple_structure_calculation(self, tmp_path):
        """Create a simple structure calculation."""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="structure",
            coupling="LS",
            configurations=[Configuration.from_string("1s2 2s2")],
            output_dir=tmp_path,
            code="autostructure",
        )
        assert calc.element == "C"
        assert calc.charge == 2
        assert calc.calculation_type == "structure"

    def test_radiative_calculation(self, tmp_path):
        """Create a radiative calculation."""
        calc = AtomicCalculation(
            element="Fe",
            charge=16,
            calculation_type="radiative",
            coupling="IC",
            radiation_types=["E1"],
            configurations=[
                Configuration.from_string("1s2 2s2 2p6"),
                Configuration.from_string("1s2 2s2 2p5 3s1"),
            ],
            output_dir=tmp_path,
            code="autostructure",
        )
        assert calc.calculation_type == "radiative"
        assert "E1" in calc.radiation_types


class TestCouplingSchemes:
    """Test different coupling schemes."""

    def test_ls_coupling(self, tmp_path):
        """Test LS coupling."""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="structure",
            coupling="LS",
            configurations=[Configuration.from_string("1s2 2s2")],
            output_dir=tmp_path,
            code="autostructure",
        )
        assert calc.coupling == "LS"

    def test_ic_coupling(self, tmp_path):
        """Test IC (intermediate coupling)."""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="structure",
            coupling="IC",
            configurations=[Configuration.from_string("1s2 2s2")],
            output_dir=tmp_path,
            code="autostructure",
        )
        assert calc.coupling == "IC"

    def test_icr_coupling(self, tmp_path):
        """Test ICR (intermediate coupling relativistic)."""
        calc = AtomicCalculation(
            element="Fe",
            charge=16,
            calculation_type="structure",
            coupling="ICR",
            configurations=[Configuration.from_string("1s2 2s2 2p6")],
            output_dir=tmp_path,
            code="autostructure",
        )
        assert calc.coupling == "ICR"


class TestInputGeneration:
    """Test input file generation."""

    def test_autostructure_input_created(self, tmp_path):
        """Test that AUTOSTRUCTURE input file is created."""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="structure",
            coupling="IC",
            configurations=[Configuration.from_string("1s2 2s2")],
            output_dir=tmp_path,
            code="autostructure",
        )

        output = calc.write_input()
        assert Path(output).exists()
        assert Path(output).is_file()

    def test_input_contains_element(self, tmp_path):
        """Test that input file contains element information."""
        calc = AtomicCalculation(
            element="Fe",
            charge=16,
            calculation_type="structure",
            coupling="IC",
            configurations=[Configuration.from_string("1s2 2s2 2p6")],
            output_dir=tmp_path,
            code="autostructure",
        )

        output = calc.write_input()
        content = Path(output).read_text()
        # Should contain nuclear charge (26 for Fe)
        assert "26" in content


class TestCodeOptions:
    """Test code-specific options."""

    def test_code_options_autostructure(self, tmp_path):
        """Test passing code-specific options to AUTOSTRUCTURE."""
        calc = AtomicCalculation(
            element="C",
            charge=2,
            calculation_type="structure",
            coupling="IC",
            configurations=[Configuration.from_string("1s2 2s2")],
            code_options={
                "NLAM": 5,
                "KCOR1": 1,
            },
            output_dir=tmp_path,
            code="autostructure",
        )

        output = calc.write_input()
        content = Path(output).read_text()
        assert "NLAM" in content or "5" in content


class TestRelativisticOptions:
    """Test relativistic corrections."""

    def test_breit_interaction(self, tmp_path):
        """Test Breit interaction."""
        calc = AtomicCalculation(
            element="Fe",
            charge=16,
            calculation_type="structure",
            coupling="IC",
            relativistic="Breit",
            configurations=[Configuration.from_string("1s2 2s2 2p6")],
            output_dir=tmp_path,
            code="autostructure",
        )
        assert calc.relativistic == "Breit"

    def test_qed_corrections(self, tmp_path):
        """Test QED corrections."""
        calc = AtomicCalculation(
            element="Fe",
            charge=16,
            calculation_type="structure",
            coupling="ICR",
            qed_corrections=True,
            configurations=[Configuration.from_string("1s2 2s2 2p6")],
            output_dir=tmp_path,
            code="autostructure",
        )
        assert calc.qed_corrections is True
