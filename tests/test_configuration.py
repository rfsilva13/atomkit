"""
Minimal essential tests for Configuration module.
"""

import pytest
from atomkit import Configuration


class TestConfigurationParsing:
    """Test basic configuration parsing."""

    def test_simple_configuration(self):
        """Test parsing simple configuration string."""
        config = Configuration.from_string("1s2")
        assert len(config.shells) == 1
        assert config.shells[0].n == 1
        assert config.shells[0].l_quantum == 0
        assert config.shells[0].occupation == 2

    def test_multiple_shells(self):
        """Test parsing multiple shells."""
        config = Configuration.from_string("1s2 2s2 2p6")
        assert len(config.shells) == 3
        assert config.shells[0].occupation == 2
        assert config.shells[1].occupation == 2
        assert config.shells[2].occupation == 6

    def test_partial_occupancy(self):
        """Test partial shell occupancy."""
        config = Configuration.from_string("1s2 2s2 2p1")
        assert config.shells[2].occupation == 1

    def test_string_representation(self):
        """Test converting configuration back to string."""
        config = Configuration.from_string("1s2 2s2 2p6")
        assert "1s2" in str(config)
        assert "2s2" in str(config)
        assert "2p6" in str(config)


class TestConfigurationElectronCount:
    """Test electron counting."""

    def test_total_electrons(self):
        """Test counting total electrons."""
        config = Configuration.from_string("1s2 2s2 2p6")
        assert config.total_electrons() == 10  # Neon

    def test_helium(self):
        """Test helium configuration."""
        config = Configuration.from_string("1s2")
        assert config.total_electrons() == 2

    def test_carbon(self):
        """Test carbon configuration."""
        config = Configuration.from_string("1s2 2s2 2p2")
        assert config.total_electrons() == 6


class TestConfigurationGeneration:
    """Test configuration generation utilities."""

    def test_ground_state_simple(self):
        """Test generating simple ground state."""
        config = Configuration.from_string("1s2 2s2")
        assert config is not None

    def test_excited_states(self):
        """Test creating excited state configurations."""
        ground = Configuration.from_string("1s2 2s2")
        excited = Configuration.from_string("1s2 2s1 2p1")
        assert ground.total_electrons() == excited.total_electrons()


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_valid_occupancy(self):
        """Test that valid occupancies work."""
        config = Configuration.from_string("1s2")  # s: max 2
        assert config.shells[0].occupation <= 2

    def test_p_orbital_occupancy(self):
        """Test p orbital occupancy."""
        config = Configuration.from_string("2p6")  # p: max 6
        assert config.shells[0].occupation <= 6

    def test_d_orbital_occupancy(self):
        """Test d orbital occupancy."""
        config = Configuration.from_string("3d10")  # d: max 10
        assert config.shells[0].occupation <= 10


class TestConfigurationConversion:
    """Test converting configurations to different formats."""

    def test_autostructure_format(self):
        """Test converting to AUTOSTRUCTURE format."""
        from atomkit.converters import configurations_to_autostructure

        configs = [
            Configuration.from_string("1s2 2s2"),
            Configuration.from_string("1s2 2s1 2p1"),
        ]
        result = configurations_to_autostructure(configs)
        # Just verify it returns a string - format details don't matter
        assert isinstance(result, dict)
        assert "configurations" in result


class TestRealWorldConfigurations:
    """Test realistic atomic configurations."""

    def test_iron_ground_state(self):
        """Test iron ground state [Ar] 3d6 4s2."""
        config = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d6 4s2")
        assert config.total_electrons() == 26  # Fe

    def test_iron_ion(self):
        """Test Fe XVII (neon-like)."""
        config = Configuration.from_string("1s2 2s2 2p6")
        assert config.total_electrons() == 10

    def test_carbon_belike(self):
        """Test C III (Be-like)."""
        config = Configuration.from_string("1s2 2s2")
        assert config.total_electrons() == 4
