"""
Tests for AUTOSTRUCTURE configuration generator module.

Tests the functionality of generating electronic configurations and excitations
for AUTOSTRUCTURE calculations.
"""

import pytest
from pathlib import Path
import tempfile

from atomkit import Configuration, Shell, get_element_info, parse_ion_notation
from atomkit.converters.as_generator import (
    format_as_input,
    generate_as_configurations,
    print_as_format,
    write_as_format,
)


class TestElementInfo:
    """Test element information retrieval."""

    def test_get_element_by_symbol(self):
        """Test getting element info by symbol."""
        info = get_element_info("Fe")
        assert info["symbol"] == "Fe"
        assert info["Z"] == 26
        assert info["name"] == "Iron"

    def test_get_element_by_atomic_number(self):
        """Test getting element info by atomic number."""
        info = get_element_info(79)
        assert info["symbol"] == "Au"
        assert info["Z"] == 79
        assert info["name"] == "Gold"

    def test_get_element_various_symbols(self):
        """Test multiple elements."""
        test_cases = [
            ("H", 1, "Hydrogen"),
            ("C", 6, "Carbon"),
            ("Nd", 60, "Neodymium"),
            ("U", 92, "Uranium"),
        ]
        for symbol, z, name in test_cases:
            info = get_element_info(symbol)
            assert info["Z"] == z
            assert info["name"] == name


class TestIonNotation:
    """Test ion notation parsing."""

    def test_parse_neutral_atom(self):
        """Test parsing neutral atom (I = 0 charge)."""
        element, charge, electrons = parse_ion_notation("Fe I")
        assert element == "Fe"
        assert charge == 0
        assert electrons == 26

    def test_parse_singly_ionized(self):
        """Test parsing singly ionized atom (II = +1 charge)."""
        element, charge, electrons = parse_ion_notation("Fe II")
        assert element == "Fe"
        assert charge == 1
        assert electrons == 25

    def test_parse_doubly_ionized(self):
        """Test parsing doubly ionized atom (III = +2 charge)."""
        element, charge, electrons = parse_ion_notation("Fe III")
        assert element == "Fe"
        assert charge == 2
        assert electrons == 24

    def test_parse_various_ions(self):
        """Test various ion notations."""
        test_cases = [
            ("H I", "H", 0, 1),
            ("He II", "He", 1, 1),
            ("C IV", "C", 3, 3),
            ("Au I", "Au", 0, 79),
            ("Nd II", "Nd", 1, 59),
        ]
        for notation, expected_elem, expected_charge, expected_electrons in test_cases:
            element, charge, electrons = parse_ion_notation(notation)
            assert element == expected_elem
            assert charge == expected_charge
            assert electrons == expected_electrons

    def test_parse_invalid_notation(self):
        """Test that invalid notation raises error."""
        with pytest.raises(ValueError, match="Invalid ion notation"):
            parse_ion_notation("Fe")  # Missing roman numeral

        with pytest.raises(ValueError, match="Invalid ion notation"):
            parse_ion_notation("Fe I II")  # Too many parts


# Note: Tests for orbital parsing, configuration validation, configuration building,
# and single excitation generation have been removed as these functions are now redundant
# with Configuration/Shell classes. Use test_configuration.py to test Configuration and
# Shell functionality directly, including Configuration.generate_autostructure_configurations().


class TestASInputFormatting:
    """Test AUTOSTRUCTURE input formatting."""

    def test_format_simple_configs(self):
        """Test formatting simple configurations."""
        configs = ["3d10 4s1", "3d9 4s2"]
        result = format_as_input(configs, "3d")

        assert result["kcor1"] == 1
        assert result["mxconf"] == 2
        assert result["mxvorb"] == 2
        assert (3, 2) in result["orbitals"]  # 3d
        assert (4, 0) in result["orbitals"]  # 4s
        assert len(result["occupation_matrix"]) == 2

    def test_format_with_excitations(self):
        """Test formatting with excited configurations."""
        configs = ["3d10 4s2", "3d9 4s2 4p1", "3d10 4s1 4p1"]
        result = format_as_input(configs, "3d")

        assert result["mxconf"] == 3
        assert result["mxvorb"] == 3
        assert (3, 2) in result["orbitals"]  # 3d
        assert (4, 0) in result["orbitals"]  # 4s
        assert (4, 1) in result["orbitals"]  # 4p

    def test_format_occupation_matrix(self):
        """Test occupation matrix is correct."""
        configs = ["3d10 4s1", "3d9 4s2"]
        result = format_as_input(configs, "3d")

        # First config: 3d10 4s1
        assert result["occupation_matrix"][0] == [10, 1]
        # Second config: 3d9 4s2
        assert result["occupation_matrix"][1] == [9, 2]

    def test_format_invalid_core_orbital(self):
        """Test error when core orbital not found."""
        configs = ["4s2 4p1"]
        with pytest.raises(ValueError, match="Core orbital"):
            format_as_input(configs, "3d")


class TestFullWorkflow:
    """Test complete workflow."""

    def test_generate_fe_configurations(self):
        """Test generating configurations for Fe I."""
        result = generate_as_configurations(
            "Fe I", "1s2 2s2 2p6 3s2 3p6 3d6 4s2", "3d 4s", max_n=5, max_l_symbol="d"
        )

        assert result["mxconf"] > 1  # Ground state + excitations
        assert result["mxvorb"] >= 2  # At least 3d and 4s
        assert len(result["configurations"]) == result["mxconf"]
        assert len(result["occupation_matrix"]) == result["mxconf"]

    def test_generate_with_file_output(self):
        """Test generating with file output."""
        output_file = Path(tempfile.mktemp(suffix=".txt"))

        try:
            result = generate_as_configurations(
                "Fe I",
                "1s2 2s2 2p6 3s2 3p6 3d6 4s2",
                "3d 4s",
                max_n=5,
                max_l_symbol="d",
                output_file=output_file,
            )

            assert output_file.exists()

            with open(output_file, "r") as f:
                content = f.read()

            assert "KCOR1=1" in content
            assert "MXCONF=" in content
            assert "MXVORB=" in content

        finally:
            if output_file.exists():
                output_file.unlink()

    def test_generate_invalid_ground_state(self):
        """Test error with invalid ground state."""
        with pytest.raises(ValueError, match="Invalid ground state"):
            generate_as_configurations(
                "Fe I",
                "1s3",  # Invalid: too many electrons in 1s
                "1s",
                max_n=2,
                max_l_symbol="s",
            )

    def test_generate_electron_count_mismatch(self):
        """Test error when electron count doesn't match."""
        with pytest.raises(ValueError, match="Invalid ground state"):
            generate_as_configurations(
                "Fe I",  # 26 electrons
                "1s2 2s2",  # Only 4 electrons
                "1s 2s",
                max_n=3,
                max_l_symbol="s",
            )


class TestOutputFormatting:
    """Test output formatting functions."""

    def test_write_as_format(self):
        """Test writing AS format to file."""
        result = {
            "kcor1": 1,
            "kcor2": 5,
            "mxconf": 2,
            "mxvorb": 2,
            "orbitals": [(3, 2), (4, 0)],
            "occupation_matrix": [[10, 1], [9, 2]],
            "configurations": ["3d10 4s1", "3d9 4s2"],
        }

        output_file = Path(tempfile.mktemp(suffix=".txt"))

        try:
            write_as_format(result, output_file)

            with open(output_file, "r") as f:
                lines = f.readlines()

            assert "KCOR1=1\n" in lines
            assert "KCOR2=5\n" in lines
            assert "MXCONF=2\n" in lines
            assert "MXVORB=2\n" in lines

        finally:
            if output_file.exists():
                output_file.unlink()


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_cu(self):
        """Test end-to-end for Cu I (d10 s1 system)."""
        result = generate_as_configurations(
            "Cu I", "1s2 2s2 2p6 3s2 3p6 3d10 4s1", "3d 4s", max_n=5, max_l_symbol="d"
        )

        # Check that ground state is included (should contain valence part)
        assert "3d10" in result["configurations"][0]
        assert "4s1" in result["configurations"][0]

        # Check excitations exist
        assert result["mxconf"] > 1

        # Verify all configurations have 29 electrons using Configuration class
        for config_str in result["configurations"]:
            config = Configuration.from_string(config_str)
            assert config.total_electrons() == 29

    def test_end_to_end_nd(self):
        """Test end-to-end for Nd I (f4 s2 system)."""
        ground_state = "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 4f4 5s2 5p6 6s2"

        result = generate_as_configurations(
            "Nd I", ground_state, "4f 6s", max_n=7, max_l_symbol="d"
        )

        assert result["mxconf"] > 1

        # Verify all configurations have 60 electrons using Configuration class
        for config_str in result["configurations"]:
            config = Configuration.from_string(config_str)
            assert config.total_electrons() == 60


class TestConfigurationsToAutostructure:
    """Test the new configurations_to_autostructure function."""

    def test_with_configuration_objects(self):
        """Test with Configuration objects (recommended usage)."""
        from atomkit.converters import configurations_to_autostructure

        # Generate configurations
        config = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d6 4s2")
        all_configs = config.generate_autostructure_configurations(["3d", "4s"], 5, 2)

        # Format for AUTOSTRUCTURE
        result = configurations_to_autostructure(all_configs, last_core_orbital="3p")

        assert result["mxconf"] > 0
        assert result["mxvorb"] > 0
        assert "configurations" in result
        assert "orbitals" in result

        # Verify all configurations have same electron count
        for config_str in result["configurations"]:
            test_config = Configuration.from_string(config_str)
            assert test_config.total_electrons() == 26

    def test_with_strings(self):
        """Test with configuration strings (backward compatible)."""
        from atomkit.converters import configurations_to_autostructure

        config_strings = [
            "1s2 2s2 2p6 3s2 3p6 4s2 3d6",
            "1s2 2s2 2p6 3s2 3p6 4s1 3d6 4d1",
            "1s2 2s2 2p6 3s2 3p6 4s1 3d6 4p1",
        ]

        result = configurations_to_autostructure(config_strings, last_core_orbital="3p")

        assert result["mxconf"] == 3
        assert len(result["configurations"]) == 3

    def test_with_file_output(self):
        """Test file writing integration."""
        from atomkit.converters import configurations_to_autostructure

        config = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d10 4s1")
        all_configs = config.generate_autostructure_configurations(["3d", "4s"], 5, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.txt"

            result = configurations_to_autostructure(
                all_configs, last_core_orbital="3p", output_file=output_file
            )

            # Verify file was created
            assert output_file.exists()

            # Verify file contents
            content = output_file.read_text()
            assert f"MXCONF={result['mxconf']}" in content
            assert f"MXVORB={result['mxvorb']}" in content

    def test_filtering_workflow(self):
        """Test the power of filtering configurations before formatting."""
        from atomkit.converters import configurations_to_autostructure

        # Generate configurations
        config = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d10 4s1")
        all_configs = config.generate_autostructure_configurations(["3d", "4s"], 5, 2)

        # Filter to only include 5s excitations
        filtered = [c for c in all_configs if "5s" in c.to_string(separator=" ")]

        assert len(filtered) < len(all_configs), "Filtering should reduce count"

        # Format the filtered configurations
        result = configurations_to_autostructure(filtered, last_core_orbital="3p")

        assert result["mxconf"] == len(filtered)

        # Verify all have 5s
        for config_str in result["configurations"]:
            assert "5s" in config_str

    def test_combining_strategies(self):
        """Test combining configurations from different generation strategies."""
        from atomkit.converters import configurations_to_autostructure

        config = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d6 4s2")

        # Strategy 1: AUTOSTRUCTURE single excitations
        configs_1 = config.generate_autostructure_configurations(["3d", "4s"], 4, 1)

        # Strategy 2: Custom double excitations
        configs_2 = config.generate_excitations(
            target_shells=["3d"], excitation_level=2, source_shells=["2p"]
        )

        # Combine
        all_configs = list(configs_1) + list(configs_2)

        # Format
        result = configurations_to_autostructure(all_configs, last_core_orbital="3p")

        assert result["mxconf"] == len(all_configs)

    def test_empty_list(self):
        """Test with empty configuration list."""
        from atomkit.converters import configurations_to_autostructure

        with pytest.raises((ValueError, IndexError)):
            configurations_to_autostructure([], last_core_orbital="3p")

    def test_mixed_types_error(self):
        """Test that mixing Configuration objects and strings in same list works."""
        from atomkit.converters import configurations_to_autostructure

        config = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d6 4s2")

        # This should work - the function checks the first element
        mixed = [config]  # First is Configuration, function will convert all
        result = configurations_to_autostructure(mixed, last_core_orbital="3p")

        assert result["mxconf"] == 1

    def test_core_plus_valence_configs(self):
        """Test core + valence format (new feature)."""
        from atomkit.converters import configurations_to_autostructure

        # Define core and valence separately
        core = Configuration.from_string("1s2 2s2 2p6 3s2 3p6")
        valence1 = Configuration.from_string("3d6 4s2")
        valence2 = Configuration.from_string("3d6 4s1 4p1")

        result = configurations_to_autostructure(
            [valence1, valence2], core=core, last_core_orbital="3p"
        )

        assert result["mxconf"] == 2
        # Verify core was prepended
        assert "1s2 2s2 2p6 3s2 3p6" in result["configurations"][0]
        assert "1s2 2s2 2p6 3s2 3p6" in result["configurations"][1]

    def test_core_plus_valence_strings(self):
        """Test core + valence with strings."""
        from atomkit.converters import configurations_to_autostructure

        result = configurations_to_autostructure(
            ["3d6 4s2", "3d6 4s1 4p1"],
            core="1s2 2s2 2p6 3s2 3p6",
            last_core_orbital="3p",
        )

        assert result["mxconf"] == 2
        # Verify core was prepended (order may vary)
        assert "1s2 2s2 2p6 3s2 3p6" in result["configurations"][0]
        assert "3d6" in result["configurations"][0]
        assert "4s2" in result["configurations"][0]
        assert "4p1" in result["configurations"][1]
