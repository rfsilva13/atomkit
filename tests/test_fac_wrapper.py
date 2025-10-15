"""
Unit tests for the FAC (Flexible Atomic Code) wrapper module.

Tests cover:
- SFACWriter basic functionality
- Syntax generation
- Context manager behavior
- Atomkit integration
- All major FAC functions
- Error handling
"""

import pytest
import tempfile
from pathlib import Path
from io import StringIO
import sys

from atomkit.fac import SFACWriter
from atomkit import Configuration


class TestSFACWriterBasics:
    """Test basic SFACWriter functionality."""
    
    def test_initialization(self, tmp_path):
        """Test SFACWriter can be initialized."""
        test_file = tmp_path / "test.sf"
        writer = SFACWriter(test_file)
        assert writer.filename == test_file
        assert writer._commands == []
        assert writer._file is None
        
    def test_context_manager(self, tmp_path):
        """Test context manager properly creates and closes file."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.SetAtom("Fe")
        
        # File should exist and be closed
        assert test_file.exists()
        content = test_file.read_text()
        assert "SetAtom('Fe')" in content
        assert "# FAC SFAC Input File" in content
    
    def test_manual_close(self, tmp_path):
        """Test manual open and close."""
        test_file = tmp_path / "test.sf"
        
        fac = SFACWriter(test_file)
        fac.__enter__()
        fac.SetAtom("Fe")
        fac.Config("2*8", group="n2")
        fac.__exit__(None, None, None)
        
        assert test_file.exists()
        content = test_file.read_text()
        assert "SetAtom('Fe')" in content
        assert "Config('2*8', group = 'n2')" in content
    
    def test_auto_write_mode(self, tmp_path):
        """Test auto-write mode writes immediately."""
        test_file = tmp_path / "test.sf"
        
        fac = SFACWriter(test_file, auto_write=True)
        fac.SetAtom("Fe")
        fac.close()
        
        assert test_file.exists()
        content = test_file.read_text()
        assert "SetAtom('Fe')" in content


class TestSyntaxGeneration:
    """Test SFAC syntax generation."""
    
    def test_format_string(self, tmp_path):
        """Test string formatting."""
        test_file = tmp_path / "test.sf"
        fac = SFACWriter(test_file)
        
        # String values should be quoted
        result = fac._format_value("Fe")
        assert result == "'Fe'"
        
        # Already quoted strings should not be double-quoted
        result = fac._format_value("'Fe'")
        assert result == "'Fe'"
    
    def test_format_list(self, tmp_path):
        """Test list formatting."""
        test_file = tmp_path / "test.sf"
        fac = SFACWriter(test_file)
        
        result = fac._format_value(["n2", "n3"])
        assert result == "['n2', 'n3']"
        
        result = fac._format_value([1, 2, 3])
        assert result == "[1, 2, 3]"
    
    def test_format_numbers(self, tmp_path):
        """Test number formatting."""
        test_file = tmp_path / "test.sf"
        fac = SFACWriter(test_file)
        
        assert fac._format_value(42) == "42"
        assert fac._format_value(3.14) == "3.14"
        assert fac._format_value(0) == "0"
        assert fac._format_value(-1) == "-1"
    
    def test_format_boolean(self, tmp_path):
        """Test boolean formatting."""
        test_file = tmp_path / "test.sf"
        fac = SFACWriter(test_file)
        
        # Booleans should become 0 or 1
        assert fac._format_value(True) == "1"
        assert fac._format_value(False) == "0"
    
    def test_format_none(self, tmp_path):
        """Test None formatting."""
        test_file = tmp_path / "test.sf"
        fac = SFACWriter(test_file)
        
        assert fac._format_value(None) == "None"
    
    def test_keyword_arguments(self, tmp_path):
        """Test keyword argument formatting."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.Config("2*8", group="n2")
        
        content = test_file.read_text()
        assert "Config('2*8', group = 'n2')" in content
    
    def test_multiple_arguments(self, tmp_path):
        """Test multiple positional and keyword arguments."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.TransitionTable("output.tr.b", ["n2"], ["n3"], -1)
        
        content = test_file.read_text()
        assert "TransitionTable('output.tr.b', ['n2'], ['n3'], -1)" in content


class TestAtomicStructureFunctions:
    """Test atomic structure calculation functions."""
    
    def test_set_atom(self, tmp_path):
        """Test SetAtom function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.SetAtom("Fe")
        
        content = test_file.read_text()
        assert "SetAtom('Fe')" in content
    
    def test_closed(self, tmp_path):
        """Test Closed function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.Closed("1s")
        
        content = test_file.read_text()
        assert "Closed('1s')" in content
    
    def test_config(self, tmp_path):
        """Test Config function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.Config("2*8", group="n2")
            fac.Config("2*7 3*1", group="n3")
        
        content = test_file.read_text()
        assert "Config('2*8', group = 'n2')" in content
        assert "Config('2*7 3*1', group = 'n3')" in content
    
    def test_config_energy(self, tmp_path):
        """Test ConfigEnergy function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.ConfigEnergy(0)
            fac.ConfigEnergy(1)
        
        content = test_file.read_text()
        assert "ConfigEnergy(0)" in content
        assert "ConfigEnergy(1)" in content
    
    def test_optimize_radial(self, tmp_path):
        """Test OptimizeRadial function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.OptimizeRadial(["n2"])
        
        content = test_file.read_text()
        assert "OptimizeRadial(['n2'])" in content
    
    def test_structure(self, tmp_path):
        """Test Structure function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.Structure("output.lev.b", ["n2", "n3"])
        
        content = test_file.read_text()
        assert "Structure('output.lev.b', ['n2', 'n3'])" in content
    
    def test_mem_en_table(self, tmp_path):
        """Test MemENTable function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.MemENTable("output.lev.b")
        
        content = test_file.read_text()
        assert "MemENTable('output.lev.b')" in content
    
    def test_print_table(self, tmp_path):
        """Test PrintTable function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.PrintTable("output.lev.b", "output.lev", 1)
        
        content = test_file.read_text()
        assert "PrintTable('output.lev.b', 'output.lev', 1)" in content


class TestTransitionFunctions:
    """Test radiative transition functions."""
    
    def test_transition_table_default(self, tmp_path):
        """Test TransitionTable with default multipole."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.TransitionTable("output.tr.b", ["n2"], ["n3"])
        
        content = test_file.read_text()
        assert "TransitionTable('output.tr.b', ['n2'], ['n3'])" in content
    
    def test_transition_table_multipole(self, tmp_path):
        """Test TransitionTable with specific multipole."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.TransitionTable("output.tr.b", ["n2"], ["n3"], multipole=-1)
        
        content = test_file.read_text()
        assert "TransitionTable('output.tr.b', ['n2'], ['n3'], -1)" in content
    
    def test_tr_table_alias(self, tmp_path):
        """Test TRTable alias."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.TRTable("output.tr.b", ["n2"], ["n3"])
        
        content = test_file.read_text()
        assert "TransitionTable('output.tr.b', ['n2'], ['n3'])" in content


class TestCollisionalFunctions:
    """Test collisional process functions."""
    
    def test_ce_table(self, tmp_path):
        """Test CETable function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.CETable("output.ce.b", ["n2"], ["n3"])
        
        content = test_file.read_text()
        assert "CETable('output.ce.b', ['n2'], ['n3'])" in content
    
    def test_ci_table(self, tmp_path):
        """Test CITable function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.CITable("output.ci.b", ["bound"], ["free"])
        
        content = test_file.read_text()
        assert "CITable('output.ci.b', ['bound'], ['free'])" in content


class TestPhotoionizationFunctions:
    """Test photoionization and recombination functions."""
    
    def test_rr_table(self, tmp_path):
        """Test RRTable function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.RRTable("output.rr.b", ["bound"], ["free"])
        
        content = test_file.read_text()
        assert "RRTable('output.rr.b', ['bound'], ['free'])" in content


class TestAutoionizationFunctions:
    """Test autoionization functions."""
    
    def test_ai_table(self, tmp_path):
        """Test AITable function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.AITable("output.ai.b", ["recombined"], ["target"])
        
        content = test_file.read_text()
        assert "AITable('output.ai.b', ['recombined'], ['target'])" in content
    
    def test_ai_table_msub(self, tmp_path):
        """Test AITableMSub function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.AITableMSub("output.aim.b", ["recombined"], ["target"])
        
        content = test_file.read_text()
        assert "AITableMSub('output.aim.b', ['recombined'], ['target'])" in content


class TestSettingsFunctions:
    """Test calculation settings functions."""
    
    def test_set_breit(self, tmp_path):
        """Test SetBreit function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.SetBreit(-1)
        
        content = test_file.read_text()
        assert "SetBreit(-1)" in content
    
    def test_set_se(self, tmp_path):
        """Test SetSE function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.SetSE(-1)
        
        content = test_file.read_text()
        assert "SetSE(-1)" in content
    
    def test_set_vp(self, tmp_path):
        """Test SetVP function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.SetVP(-1)
        
        content = test_file.read_text()
        assert "SetVP(-1)" in content
    
    def test_set_ms(self, tmp_path):
        """Test SetMS function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.SetMS(56, 0)
        
        content = test_file.read_text()
        assert "SetMS(56, 0)" in content
    
    def test_set_uta(self, tmp_path):
        """Test SetUTA function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.SetUTA(1)
        
        content = test_file.read_text()
        assert "SetUTA(1)" in content


class TestEnergyGridFunctions:
    """Test energy grid functions."""
    
    def test_set_usr_ce_grid(self, tmp_path):
        """Test SetUsrCEGrid function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.SetUsrCEGrid([0.1, 0.5, 1.0, 2.0])
        
        content = test_file.read_text()
        assert "SetUsrCEGrid([0.1, 0.5, 1.0, 2.0], 1)" in content
    
    def test_set_usr_ci_grid(self, tmp_path):
        """Test SetUsrCIGrid function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.SetUsrCIGrid([0.1, 0.5, 1.0, 2.0])
        
        content = test_file.read_text()
        assert "SetUsrCIGrid([0.1, 0.5, 1.0, 2.0], 0)" in content
    
    def test_set_usr_pe_grid(self, tmp_path):
        """Test SetUsrPEGrid function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.SetUsrPEGrid([0.1, 0.5, 1.0, 2.0])
        
        content = test_file.read_text()
        assert "SetUsrPEGrid([0.1, 0.5, 1.0, 2.0], 0)" in content


class TestMPIFunctions:
    """Test MPI parallel computing functions."""
    
    def test_initialize_mpi(self, tmp_path):
        """Test InitializeMPI function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.InitializeMPI(24)
        
        content = test_file.read_text()
        assert "InitializeMPI(24)" in content
    
    def test_finalize_mpi(self, tmp_path):
        """Test FinalizeMPI function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.FinalizeMPI()
        
        content = test_file.read_text()
        assert "FinalizeMPI()" in content
    
    def test_mpi_rank(self, tmp_path):
        """Test MPIRank function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.MPIRank(0)
        
        content = test_file.read_text()
        assert "MPIRank(0)" in content


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_add_comment(self, tmp_path):
        """Test add_comment function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.add_comment("This is a test comment")
            fac.add_comment("Multi-line\ncomment")
        
        content = test_file.read_text()
        assert "# This is a test comment" in content
        assert "# Multi-line" in content
        assert "# comment" in content
    
    def test_add_blank_line(self, tmp_path):
        """Test add_blank_line function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.SetAtom("Fe")
            fac.add_blank_line()
            fac.Closed("1s")
        
        content = test_file.read_text()
        lines = content.split('\n')
        # Should have a blank line between SetAtom and Closed
        assert any(line == "" for line in lines)
    
    def test_convert_to_sfac(self, tmp_path):
        """Test ConvertToSFAC function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.ConvertToSFAC("output.sf")
        
        content = test_file.read_text()
        assert "ConvertToSFAC('output.sf')" in content
    
    def test_close_sfac(self, tmp_path):
        """Test CloseSFAC function."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.CloseSFAC()
        
        content = test_file.read_text()
        assert "CloseSFAC()" in content


class TestAtomkitIntegration:
    """Test integration with atomkit Configuration objects."""
    
    def test_config_from_atomkit(self, tmp_path):
        """Test config_from_atomkit function."""
        test_file = tmp_path / "test.sf"
        
        # Create a configuration
        config = Configuration.from_string("1s2 2s2 2p1")
        
        with SFACWriter(test_file) as fac:
            fac.SetAtom("N")
            fac.config_from_atomkit(config, "ground")
        
        content = test_file.read_text()
        assert "SetAtom('N')" in content
        assert "Config('1s2 2s2 2p1', group = 'ground')" in content
    
    def test_configs_from_atomkit(self, tmp_path):
        """Test configs_from_atomkit function."""
        test_file = tmp_path / "test.sf"
        
        # Create multiple configurations
        ground = Configuration.from_string("1s2 2s2 2p1")
        excited1 = Configuration.from_string("1s2 2s1 2p2")
        excited2 = Configuration.from_string("1s2 2s2 3s1")
        
        with SFACWriter(test_file) as fac:
            fac.SetAtom("N")
            fac.configs_from_atomkit([ground, excited1, excited2], "level")
        
        content = test_file.read_text()
        assert "Config('1s2 2s2 2p1', group = 'level0')" in content
        assert "Config('1s2 2s1 2p2', group = 'level1')" in content
        assert "Config('1s2 2s2 3s1', group = 'level2')" in content
    
    def test_full_atomkit_workflow(self, tmp_path):
        """Test complete workflow with atomkit."""
        test_file = tmp_path / "test.sf"
        
        # Create configurations using atomkit
        ground = Configuration.from_element("Fe", 23)  # Fe XXIV (Li-like)
        excited = ground.generate_excitations(["2s", "2p"], 1, source_shells=["1s"])
        
        with SFACWriter(test_file) as fac:
            fac.add_comment("Fe XXIV calculation")
            fac.SetAtom("Fe")
            fac.SetBreit(-1)
            
            # Add configurations
            fac.config_from_atomkit(ground, "ground")
            for i, state in enumerate(excited[:3]):
                fac.config_from_atomkit(state, f"excited{i}")
            
            # Calculation
            fac.ConfigEnergy(0)
            fac.OptimizeRadial(["ground"])
            fac.ConfigEnergy(1)
            
            groups = ["ground"] + [f"excited{i}" for i in range(3)]
            fac.Structure("output.lev.b", groups)
        
        content = test_file.read_text()
        assert "# Fe XXIV calculation" in content
        assert "SetAtom('Fe')" in content
        assert "SetBreit(-1)" in content
        assert "Config('1s2 2s1', group = 'ground')" in content
        assert "OptimizeRadial(['ground'])" in content
        assert "Structure('output.lev.b'," in content


class TestGetContent:
    """Test get_content method."""
    
    def test_get_content_basic(self, tmp_path):
        """Test get_content returns accumulated commands."""
        test_file = tmp_path / "test.sf"
        
        fac = SFACWriter(test_file)
        fac.SetAtom("Fe")
        fac.Closed("1s")
        fac.Config("2*8", group="n2")
        
        content = fac.get_content()
        
        # get_content() returns just the commands, not the header
        # Header is added only when file is closed/written
        assert "SetAtom('Fe')" in content
        assert "Closed('1s')" in content
        assert "Config('2*8', group = 'n2')" in content
    
    def test_get_content_auto_write_error(self, tmp_path):
        """Test get_content raises error in auto_write mode."""
        test_file = tmp_path / "test.sf"
        
        fac = SFACWriter(test_file, auto_write=True)
        
        with pytest.raises(RuntimeError, match="Cannot get content in auto_write mode"):
            fac.get_content()
        
        fac.close()


class TestCompleteCalculations:
    """Test complete calculation workflows."""
    
    def test_ne_like_iron(self, tmp_path):
        """Test Ne-like iron example from FAC manual."""
        test_file = tmp_path / "fe17.sf"
        
        with SFACWriter(test_file) as fac:
            fac.add_comment("Ne-like Iron (Fe XVII)")
            fac.SetAtom("Fe")
            fac.Closed("1s")
            fac.Config("2*8", group="n2")
            fac.Config("2*7 3*1", group="n3")
            
            fac.ConfigEnergy(0)
            fac.OptimizeRadial(["n2"])
            fac.ConfigEnergy(1)
            
            fac.Structure("ne.lev.b", ["n2", "n3"])
            fac.MemENTable("ne.lev.b")
            fac.PrintTable("ne.lev.b", "ne.lev", 1)
            
            fac.TransitionTable("ne.tr.b", ["n2"], ["n3"])
            fac.PrintTable("ne.tr.b", "ne.tr", 1)
        
        content = test_file.read_text()
        
        # Verify structure
        assert "# Ne-like Iron (Fe XVII)" in content
        assert "SetAtom('Fe')" in content
        assert "Closed('1s')" in content
        assert "Config('2*8', group = 'n2')" in content
        assert "ConfigEnergy(0)" in content
        assert "OptimizeRadial(['n2'])" in content
        assert "Structure('ne.lev.b', ['n2', 'n3'])" in content
        assert "TransitionTable('ne.tr.b', ['n2'], ['n3'])" in content
    
    def test_autoionization_calculation(self, tmp_path):
        """Test autoionization calculation workflow."""
        test_file = tmp_path / "auto.sf"
        
        with SFACWriter(test_file) as fac:
            fac.add_comment("Autoionization calculation")
            fac.SetAtom("Fe")
            fac.SetBreit(-1)
            fac.SetSE(-1)
            
            fac.Config("1s2 2s1", group="target0")
            fac.Config("1s1 2s2", group="target1")
            
            fac.Config("1s1 2s1 2p1", group="auto0")
            fac.Config("1s1 2s1 3s1", group="auto1")
            
            fac.ConfigEnergy(0)
            fac.OptimizeRadial(["target0"])
            fac.ConfigEnergy(1)
            
            fac.Structure("output.lev.b", ["target0", "target1", "auto0", "auto1"])
            fac.PrintTable("output.lev.b", "output.lev.asc", 1)
            
            fac.AITable("output.ai.b", ["auto0", "auto1"], ["target0", "target1"])
            fac.PrintTable("output.ai.b", "output.ai.asc", 1)
        
        content = test_file.read_text()
        
        assert "# Autoionization calculation" in content
        assert "SetBreit(-1)" in content
        assert "SetSE(-1)" in content
        assert "AITable('output.ai.b', ['auto0', 'auto1'], ['target0', 'target1'])" in content
    
    def test_parallel_calculation(self, tmp_path):
        """Test MPI parallel calculation."""
        test_file = tmp_path / "parallel.sf"
        
        with SFACWriter(test_file) as fac:
            fac.add_comment("Parallel calculation with MPI")
            fac.InitializeMPI(24)
            
            fac.SetAtom("Cu")
            fac.Closed("1s 2s 2p 3s 3p")
            fac.Config("3d10 4s1", group="ground")
            fac.Config("3d10 4p1", group="excited")
            
            fac.OptimizeRadial(["ground"])
            fac.Structure("cu.lev.b", ["ground", "excited"])
            
            fac.FinalizeMPI()
        
        content = test_file.read_text()
        
        assert "# Parallel calculation with MPI" in content
        assert "InitializeMPI(24)" in content
        assert "FinalizeMPI()" in content


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_file(self, tmp_path):
        """Test creating file with no commands."""
        test_file = tmp_path / "empty.sf"
        
        with SFACWriter(test_file) as fac:
            pass
        
        # File should still have header
        content = test_file.read_text()
        assert "# FAC SFAC Input File" in content
    
    def test_string_with_special_characters(self, tmp_path):
        """Test handling strings with special characters."""
        test_file = tmp_path / "test.sf"
        
        with SFACWriter(test_file) as fac:
            fac.add_comment("Test with special chars: !@#$%")
        
        content = test_file.read_text()
        assert "# Test with special chars: !@#$%" in content
    
    def test_nested_lists(self, tmp_path):
        """Test handling nested lists."""
        test_file = tmp_path / "test.sf"
        fac = SFACWriter(test_file)
        
        # Nested lists should be formatted correctly
        result = fac._format_value([[1, 2], [3, 4]])
        assert result == "[[1, 2], [3, 4]]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
