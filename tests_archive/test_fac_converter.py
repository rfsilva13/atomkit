"""
Tests for FAC to AUTOSTRUCTURE converter module.

Tests the functionality of converting FAC configuration files to AUTOSTRUCTURE
input format, including orbital parsing, configuration extraction, and
occupation matrix generation.
"""

import pytest
from pathlib import Path
import tempfile

from atomkit.converters.fac_to_as import (
    parse_orbital,
    extract_closed_shells,
    extract_configurations,
    get_unique_orbitals,
    build_occupation_matrix,
    convert_fac_to_as,
    write_as_format,
    ORBITAL_MAP,
)


class TestOrbitalParsing:
    """Test orbital string parsing functionality."""

    def test_parse_s_orbital(self):
        """Test parsing s orbital."""
        n, l, occ = parse_orbital("4s2")
        assert n == 4
        assert l == 0
        assert occ == 2

    def test_parse_p_orbital(self):
        """Test parsing p orbital."""
        n, l, occ = parse_orbital("3p6")
        assert n == 3
        assert l == 1
        assert occ == 6

    def test_parse_d_orbital(self):
        """Test parsing d orbital."""
        n, l, occ = parse_orbital("3d10")
        assert n == 3
        assert l == 2
        assert occ == 10

    def test_parse_f_orbital(self):
        """Test parsing f orbital."""
        n, l, occ = parse_orbital("4f14")
        assert n == 4
        assert l == 3
        assert occ == 14

    def test_parse_single_electron(self):
        """Test parsing orbital with single electron."""
        n, l, occ = parse_orbital("5s1")
        assert n == 5
        assert l == 0
        assert occ == 1

    def test_orbital_map_coverage(self):
        """Test that orbital map contains expected symbols."""
        expected_orbitals = ["s", "p", "d", "f", "g", "h", "i", "k"]
        for orb in expected_orbitals:
            assert orb in ORBITAL_MAP


class TestClosedShellExtraction:
    """Test extraction of closed shells from FAC lines."""

    def test_extract_single_closed(self):
        """Test extracting single closed shell."""
        lines = ["Closed('1s')"]
        kcor2 = extract_closed_shells(lines)
        assert kcor2 == 1

    def test_extract_multiple_closed(self):
        """Test extracting multiple closed shells."""
        lines = ["Closed('1s 2s 2p')"]
        kcor2 = extract_closed_shells(lines)
        assert kcor2 == 3

    def test_no_closed_shells(self):
        """Test file with no closed shells."""
        lines = ["Config('MR', '3s2')"]
        kcor2 = extract_closed_shells(lines)
        assert kcor2 == 0


class TestConfigurationExtraction:
    """Test extraction of configurations from FAC lines."""

    def test_extract_single_config(self):
        """Test extracting single configuration."""
        lines = ["Config('MR', '3d10 4s1')"]
        configs = extract_configurations(lines, "MR")
        assert len(configs) == 1
        assert configs[0] == "3d10 4s1"

    def test_extract_multiple_configs(self):
        """Test extracting multiple configurations."""
        lines = [
            "Config('MR', '3d10 4s1')",
            "Config('MR', '3d9 4s2')",
            "Config('MR', '3d8 4s2 4p1')",
        ]
        configs = extract_configurations(lines, "MR")
        assert len(configs) == 3
        assert configs[0] == "3d10 4s1"
        assert configs[1] == "3d9 4s2"
        assert configs[2] == "3d8 4s2 4p1"

    def test_extract_wrong_label(self):
        """Test extracting with non-existent label."""
        lines = ["Config('MR', '3d10 4s1')"]
        configs = extract_configurations(lines, "OTHER")
        assert len(configs) == 0

    def test_mixed_labels(self):
        """Test extracting specific label from mixed labels."""
        lines = [
            "Config('MR', '3d10 4s1')",
            "Config('OTHER', '3d9 4s2')",
            "Config('MR', '3d8 4s2 4p1')",
        ]
        configs = extract_configurations(lines, "MR")
        assert len(configs) == 2
        assert configs[0] == "3d10 4s1"
        assert configs[1] == "3d8 4s2 4p1"


class TestUniqueOrbitals:
    """Test extraction of unique orbitals."""

    def test_single_config_single_orbital(self):
        """Test single configuration with one orbital."""
        configs = ["4s2"]
        orbitals = get_unique_orbitals(configs)
        assert len(orbitals) == 1
        assert (4, 0) in orbitals

    def test_single_config_multiple_orbitals(self):
        """Test single configuration with multiple orbitals."""
        configs = ["3d10 4s2"]
        orbitals = get_unique_orbitals(configs)
        assert len(orbitals) == 2
        assert (3, 2) in orbitals
        assert (4, 0) in orbitals

    def test_multiple_configs_same_orbitals(self):
        """Test multiple configurations sharing orbitals."""
        configs = ["3d10 4s1", "3d9 4s2"]
        orbitals = get_unique_orbitals(configs)
        assert len(orbitals) == 2
        assert (3, 2) in orbitals
        assert (4, 0) in orbitals

    def test_multiple_configs_different_orbitals(self):
        """Test multiple configurations with different orbitals."""
        configs = ["3d10 4s1", "3d9 4s2", "3d8 4s2 4p1"]
        orbitals = get_unique_orbitals(configs)
        assert len(orbitals) == 3
        assert (3, 2) in orbitals
        assert (4, 0) in orbitals
        assert (4, 1) in orbitals

    def test_orbital_sorting(self):
        """Test that orbitals are sorted by n, then l."""
        configs = ["4p1 3d10 4s2"]
        orbitals = get_unique_orbitals(configs)
        # Should be sorted: (3,2), (4,0), (4,1)
        assert orbitals == [(3, 2), (4, 0), (4, 1)]


class TestOccupationMatrix:
    """Test occupation matrix building."""

    def test_single_config(self):
        """Test matrix for single configuration."""
        configs = ["3d10 4s1"]
        orbitals = [(3, 2), (4, 0)]
        matrix = build_occupation_matrix(configs, orbitals)
        assert len(matrix) == 1
        assert matrix[0] == [10, 1]

    def test_multiple_configs(self):
        """Test matrix for multiple configurations."""
        configs = ["3d10 4s1", "3d9 4s2"]
        orbitals = [(3, 2), (4, 0)]
        matrix = build_occupation_matrix(configs, orbitals)
        assert len(matrix) == 2
        assert matrix[0] == [10, 1]
        assert matrix[1] == [9, 2]

    def test_sparse_occupations(self):
        """Test matrix with some zero occupations."""
        configs = ["3d10 4s1", "3d10 4p1"]
        orbitals = [(3, 2), (4, 0), (4, 1)]
        matrix = build_occupation_matrix(configs, orbitals)
        assert len(matrix) == 2
        assert matrix[0] == [10, 1, 0]
        assert matrix[1] == [10, 0, 1]

    def test_complex_case(self):
        """Test complex case with multiple orbitals and configs."""
        configs = ["3d10 4s2", "3d9 4s2 4p1", "3d10 4s1 4p1", "3d8 4s2 4p2"]
        orbitals = [(3, 2), (4, 0), (4, 1)]
        matrix = build_occupation_matrix(configs, orbitals)
        assert len(matrix) == 4
        assert matrix[0] == [10, 2, 0]
        assert matrix[1] == [9, 2, 1]
        assert matrix[2] == [10, 1, 1]
        assert matrix[3] == [8, 2, 2]


class TestFullConversion:
    """Test full FAC to AS conversion with mock files."""

    def create_fac_file(self, content: str) -> Path:
        """Helper to create temporary FAC file."""
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sf")
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_simple_conversion(self):
        """Test simple conversion with minimal FAC file."""
        content = """Closed('1s 2s 2p')
Config('MR', '3d10 4s1')
Config('MR', '3d9 4s2')
"""
        fac_file = self.create_fac_file(content)

        try:
            result = convert_fac_to_as(fac_file, "MR")

            assert result["icfg"] == 0
            assert result["kcor2"] == 3
            assert result["mxconf"] == 2
            assert result["mxvorb"] == 2
            assert len(result["orbitals"]) == 2
            assert len(result["occupation_matrix"]) == 2
            assert result["orbitals"] == [(3, 2), (4, 0)]
            assert result["occupation_matrix"][0] == [10, 1]
            assert result["occupation_matrix"][1] == [9, 2]
        finally:
            fac_file.unlink()

    def test_conversion_with_output_file(self):
        """Test conversion with output file writing."""
        content = """Closed('1s 2s 2p')
Config('MR', '3d10 4s1')
Config('MR', '3d9 4s2')
"""
        fac_file = self.create_fac_file(content)
        output_file = Path(tempfile.mktemp(suffix=".txt"))

        try:
            result = convert_fac_to_as(fac_file, "MR", output_file=output_file)

            # Check output file was created
            assert output_file.exists()

            # Read and verify content
            with open(output_file, "r") as f:
                lines = f.readlines()

            assert "ICFG=0" in lines[0]
            assert "KCOR2=3" in lines[1]
            assert "MXCONF=2" in lines[2]
            assert "MXVORB=2" in lines[3]

        finally:
            fac_file.unlink()
            if output_file.exists():
                output_file.unlink()

    def test_conversion_no_configs(self):
        """Test conversion with no matching configurations."""
        content = """Closed('1s 2s 2p')
Config('OTHER', '3d10 4s1')
"""
        fac_file = self.create_fac_file(content)

        try:
            with pytest.raises(ValueError, match="No configurations found"):
                convert_fac_to_as(fac_file, "MR")
        finally:
            fac_file.unlink()

    def test_conversion_nonexistent_file(self):
        """Test conversion with non-existent file."""
        with pytest.raises(FileNotFoundError):
            convert_fac_to_as("nonexistent_file.sf", "MR")


class TestOutputFormatting:
    """Test output formatting functions."""

    def test_write_as_format(self):
        """Test writing AS format to file."""
        result = {
            "icfg": 0,
            "kcor2": 3,
            "mxconf": 2,
            "mxvorb": 2,
            "orbitals": [(3, 2), (4, 0)],
            "occupation_matrix": [[10, 1], [9, 2]],
        }

        output_file = Path(tempfile.mktemp(suffix=".txt"))

        try:
            write_as_format(result, output_file)

            # Verify file content
            with open(output_file, "r") as f:
                content = f.read()

            assert "ICFG=0" in content
            assert "KCOR2=3" in content
            assert "MXCONF=2" in content
            assert "MXVORB=2" in content
            assert "3 2  4 0" in content

        finally:
            if output_file.exists():
                output_file.unlink()
