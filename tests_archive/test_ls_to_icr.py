"""
Tests for LS to ICR coupling converter module.

Tests the functionality of converting AUTOSTRUCTURE LS coupling input files
to ICR coupling, including lambda extraction, file modification, and
optional subprocess execution.
"""

import pytest
from pathlib import Path
import tempfile
import numpy as np

from atomkit.converters.ls_to_icr import (
    convert_ls_to_icr,
    create_icr_input,
    run_autostructure_ls,
    run_autostructure_icr,
    ls_to_icr_full_workflow,
)


class TestICRInputCreation:
    """Test ICR input file creation from LS files."""

    def create_ls_file(self, content: str) -> Path:
        """Helper to create temporary LS input file."""
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_LS.inp")
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_create_icr_basic(self):
        """Test basic ICR file creation with lambda substitution."""
        ls_content = """&SYST
 CUP="LS"
&END
&SMINIM INCLUD=1 NVAR=5 &END
 1.0  1.0  1.0  1.0  1.0
"""
        ls_file = self.create_ls_file(ls_content)
        icr_file = Path(tempfile.mktemp(suffix="_ICR.inp"))
        lambdas = np.array([0.95, 0.98, 1.02, 0.99, 1.01])

        try:
            create_icr_input(ls_file, icr_file, lambdas)

            # Verify ICR file was created
            assert icr_file.exists()

            # Read and verify content
            with open(icr_file, "r") as f:
                content = f.read()

            # Check CUP changed to ICR
            assert 'CUP="ICR"' in content
            assert 'CUP="LS"' not in content

            # Check INCLUD and NVAR removed
            assert "INCLUD" not in content
            assert "NVAR" not in content

            # Check lambdas are present
            for lam in lambdas:
                assert f"{lam:.10f}" in content

        finally:
            ls_file.unlink()
            if icr_file.exists():
                icr_file.unlink()

    def test_create_icr_cup_lowercase(self):
        """Test that lowercase CUP='ls' is also converted."""
        ls_content = """&SYST
 cup='ls'
&END
&SMINIM INCLUD=1 NVAR=3 &END
 1.0  1.0  1.0
"""
        ls_file = self.create_ls_file(ls_content)
        icr_file = Path(tempfile.mktemp(suffix="_ICR.inp"))
        lambdas = np.array([0.95, 0.98, 1.02])

        try:
            create_icr_input(ls_file, icr_file, lambdas)

            with open(icr_file, "r") as f:
                content = f.read()

            # Check conversion happened (case-insensitive)
            assert 'CUP="ICR"' in content
            assert "ls" not in content.lower() or 'CUP="ICR"' in content

        finally:
            ls_file.unlink()
            if icr_file.exists():
                icr_file.unlink()

    def test_create_icr_with_spaces_in_cup(self):
        """Test CUP with various spacing."""
        ls_content = """&SYST
 CUP  =  "LS"
&END
&SMINIM INCLUD=1 NVAR=2 &END
 1.0  1.0
"""
        ls_file = self.create_ls_file(ls_content)
        icr_file = Path(tempfile.mktemp(suffix="_ICR.inp"))
        lambdas = np.array([0.95, 0.98])

        try:
            create_icr_input(ls_file, icr_file, lambdas)

            with open(icr_file, "r") as f:
                content = f.read()

            assert 'CUP="ICR"' in content

        finally:
            ls_file.unlink()
            if icr_file.exists():
                icr_file.unlink()

    def test_create_icr_empty_lambdas(self):
        """Test that empty lambda array raises error."""
        ls_content = """&SYST CUP="LS" &END"""
        ls_file = self.create_ls_file(ls_content)
        icr_file = Path(tempfile.mktemp(suffix="_ICR.inp"))

        try:
            with pytest.raises(ValueError, match="Empty lambda array"):
                create_icr_input(ls_file, icr_file, np.array([]))
        finally:
            ls_file.unlink()
            if icr_file.exists():
                icr_file.unlink()

    def test_create_icr_nonexistent_ls_file(self):
        """Test error when LS file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            create_icr_input("nonexistent.inp", "output.inp", np.array([1.0]))

    def test_create_icr_multiple_sminim_sections(self):
        """Test that only first &SMINIM is modified."""
        ls_content = """&SYST CUP="LS" &END
&SMINIM INCLUD=1 NVAR=2 &END
 1.0  1.0
Some other content
&SMINIM INCLUD=2 NVAR=3 &END
 2.0  2.0  2.0
"""
        ls_file = self.create_ls_file(ls_content)
        icr_file = Path(tempfile.mktemp(suffix="_ICR.inp"))
        lambdas = np.array([0.95, 0.98])

        try:
            create_icr_input(ls_file, icr_file, lambdas)

            with open(icr_file, "r") as f:
                lines = f.readlines()

            # Check that first SMINIM was modified
            sminim_count = sum(1 for line in lines if "&SMINIM" in line)
            assert sminim_count == 2

        finally:
            ls_file.unlink()
            if icr_file.exists():
                icr_file.unlink()

    def test_create_icr_complex_input(self):
        """Test with more realistic AUTOSTRUCTURE input."""
        ls_content = """&SYST
 NZ=26
 NE=24
 NELC=24
 CUP="LS"
 PKEY=2
&END

&SMINIM INCLUD=1 NVAR=5 &END
 1.0000  1.0000  1.0000  1.0000  1.0000

&CSFLS
...
&END
"""
        ls_file = self.create_ls_file(ls_content)
        icr_file = Path(tempfile.mktemp(suffix="_ICR.inp"))
        lambdas = np.array(
            [0.9234567890, 0.9876543210, 1.0123456789, 0.9988776655, 1.0011223344]
        )

        try:
            create_icr_input(ls_file, icr_file, lambdas)

            with open(icr_file, "r") as f:
                content = f.read()

            assert 'CUP="ICR"' in content
            assert "INCLUD" not in content
            assert "NVAR" not in content

            # Verify lambda precision
            assert "0.9234567890" in content
            assert "0.9876543210" in content

        finally:
            ls_file.unlink()
            if icr_file.exists():
                icr_file.unlink()


class TestConvertLSToICR:
    """Test full LS to ICR conversion (without running AUTOSTRUCTURE)."""

    def create_ls_file(self, content: str) -> Path:
        """Helper to create temporary LS input file."""
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_LS.inp")
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_convert_with_provided_lambdas(self):
        """Test conversion with pre-computed lambdas."""
        ls_content = """&SYST CUP="LS" &END
&SMINIM INCLUD=1 NVAR=3 &END
 1.0  1.0  1.0
"""
        ls_file = self.create_ls_file(ls_content)
        lambdas = np.array([0.95, 0.98, 1.02])

        try:
            icr_file, result_lambdas = convert_ls_to_icr(
                ls_file, run_ls_calculation=False, lambdas=lambdas
            )

            # Check ICR file was created
            assert icr_file.exists()
            assert icr_file.stem.endswith("_ICR")

            # Check lambdas match
            np.testing.assert_array_almost_equal(result_lambdas, lambdas)

        finally:
            ls_file.unlink()
            if icr_file.exists():
                icr_file.unlink()

    def test_convert_auto_icr_filename(self):
        """Test that ICR filename is auto-generated correctly."""
        ls_content = """&SYST CUP="LS" &END
&SMINIM INCLUD=1 NVAR=2 &END
 1.0  1.0
"""
        ls_file = self.create_ls_file(ls_content)
        lambdas = np.array([0.95, 0.98])

        try:
            icr_file, _ = convert_ls_to_icr(
                ls_file, run_ls_calculation=False, lambdas=lambdas
            )

            # Check filename pattern
            assert "_ICR" in icr_file.stem
            assert icr_file.suffix == ls_file.suffix

        finally:
            ls_file.unlink()
            if icr_file.exists():
                icr_file.unlink()

    def test_convert_custom_icr_filename(self):
        """Test conversion with custom ICR filename."""
        ls_content = """&SYST CUP="LS" &END
&SMINIM INCLUD=1 NVAR=2 &END
 1.0  1.0
"""
        ls_file = self.create_ls_file(ls_content)
        icr_file = Path(tempfile.mktemp(suffix="_custom_ICR.inp"))
        lambdas = np.array([0.95, 0.98])

        try:
            result_icr, _ = convert_ls_to_icr(
                ls_file,
                icr_output_file=icr_file,
                run_ls_calculation=False,
                lambdas=lambdas,
            )

            assert result_icr == icr_file
            assert icr_file.exists()

        finally:
            ls_file.unlink()
            if icr_file.exists():
                icr_file.unlink()

    def test_convert_no_lambdas_no_calculation(self):
        """Test that error is raised when no lambdas and no calculation."""
        ls_content = """&SYST CUP="LS" &END"""
        ls_file = self.create_ls_file(ls_content)

        try:
            with pytest.raises(FileNotFoundError, match="olg file not found"):
                convert_ls_to_icr(ls_file, run_ls_calculation=False, lambdas=None)
        finally:
            ls_file.unlink()

    def test_convert_nonexistent_ls_file(self):
        """Test error when LS file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="LS input file not found"):
            convert_ls_to_icr(
                "nonexistent.inp", run_ls_calculation=False, lambdas=np.array([1.0])
            )


class TestAutostructureExecution:
    """Test AUTOSTRUCTURE execution functions (these will fail without AS executable)."""

    def test_run_ls_no_executable(self):
        """Test that missing executable raises appropriate error."""
        ls_file = Path(tempfile.mktemp(suffix=".inp"))
        ls_file.write_text("dummy content")

        try:
            with pytest.raises(
                FileNotFoundError, match="AUTOSTRUCTURE executable not found"
            ):
                run_autostructure_ls(ls_file, as_executable="./nonexistent_as.x")
        finally:
            if ls_file.exists():
                ls_file.unlink()

    def test_run_ls_no_input_file(self):
        """Test that missing input file raises error."""
        with pytest.raises(FileNotFoundError, match="LS input file not found"):
            run_autostructure_ls("nonexistent.inp")

    def test_run_icr_no_executable(self):
        """Test that missing executable raises appropriate error."""
        icr_file = Path(tempfile.mktemp(suffix=".inp"))
        icr_file.write_text("dummy content")

        try:
            with pytest.raises(
                FileNotFoundError, match="AUTOSTRUCTURE executable not found"
            ):
                run_autostructure_icr(icr_file, as_executable="./nonexistent_as.x")
        finally:
            if icr_file.exists():
                icr_file.unlink()

    def test_run_icr_no_input_file(self):
        """Test that missing input file raises error."""
        with pytest.raises(FileNotFoundError, match="ICR input file not found"):
            run_autostructure_icr("nonexistent.inp")


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_create_icr_preserves_other_content(self):
        """Test that ICR creation preserves non-modified sections."""
        ls_content = """# Comment line
&SYST
 NZ=26
 CUP="LS"
 PKEY=2
&END

&SMINIM INCLUD=1 NVAR=2 &END
 1.0  1.0

&CSFLS
 Some configuration data
&END

# End of file
"""
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".inp")
        tmp.write(ls_content)
        tmp.close()
        ls_file = Path(tmp.name)

        icr_file = Path(tempfile.mktemp(suffix=".inp"))
        lambdas = np.array([0.95, 0.98])

        try:
            create_icr_input(ls_file, icr_file, lambdas)

            with open(icr_file, "r") as f:
                content = f.read()

            # Check that other sections are preserved
            assert "# Comment line" in content
            assert "NZ=26" in content
            assert "PKEY=2" in content
            assert "&CSFLS" in content
            assert "Some configuration data" in content
            assert "# End of file" in content

            # Check modifications
            assert 'CUP="ICR"' in content
            assert "INCLUD" not in content
            assert "NVAR" not in content

        finally:
            ls_file.unlink()
            if icr_file.exists():
                icr_file.unlink()

    def test_lambda_array_types(self):
        """Test that various numpy array types work."""
        ls_content = """&SYST CUP="LS" &END
&SMINIM INCLUD=1 NVAR=3 &END
 1.0  1.0  1.0
"""
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".inp")
        tmp.write(ls_content)
        tmp.close()
        ls_file = Path(tmp.name)

        icr_file = Path(tempfile.mktemp(suffix=".inp"))

        # Test with list
        lambdas_list = [0.95, 0.98, 1.02]

        try:
            create_icr_input(ls_file, icr_file, lambdas_list)
            assert icr_file.exists()

            # Test with numpy array
            icr_file.unlink()
            lambdas_array = np.array([0.95, 0.98, 1.02], dtype=np.float32)
            create_icr_input(ls_file, icr_file, lambdas_array)
            assert icr_file.exists()

            # Test with numpy float64
            icr_file.unlink()
            lambdas_float64 = np.array([0.95, 0.98, 1.02], dtype=np.float64)
            create_icr_input(ls_file, icr_file, lambdas_float64)
            assert icr_file.exists()

        finally:
            ls_file.unlink()
            if icr_file.exists():
                icr_file.unlink()
