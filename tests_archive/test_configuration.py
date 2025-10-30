#!/usr/bin/env python3
"""
Unit tests for the Configuration class.

This test suite covers all public methods and edge cases of the Configuration class
including initialization, shell manipulation, string parsing, and electron calculations.
"""

import unittest
import sys
import os
from typing import List, Dict, Set

# Add the src directory to the path to import atomkit modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atomkit.configuration import Configuration
from atomkit.shell import Shell


class TestConfiguration(unittest.TestCase):
    """Test suite for the Configuration class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create some test shells for reuse
        self.shell_1s2 = Shell(1, 0, 2)  # 1s2
        self.shell_2s2 = Shell(2, 0, 2)  # 2s2
        self.shell_2p6 = Shell(2, 1, 6)  # 2p6
        self.shell_3s1 = Shell(3, 0, 1)  # 3s1
        self.shell_3d5 = Shell(3, 2, 5)  # 3d5

        # Test configurations
        self.empty_config = Configuration()
        self.he_config = Configuration([self.shell_1s2])  # Helium
        self.ne_config = Configuration(
            [self.shell_1s2, self.shell_2s2, self.shell_2p6]
        )  # Neon
        self.na_config = Configuration(
            [self.shell_1s2, self.shell_2s2, self.shell_2p6, self.shell_3s1]
        )  # Sodium

    def test_init_empty(self):
        """Test initialization with no shells."""
        config = Configuration()
        self.assertEqual(len(config), 0)
        self.assertEqual(config.total_electrons(), 0)
        self.assertEqual(str(config), "")

    def test_init_with_shells(self):
        """Test initialization with shell list."""
        config = Configuration([self.shell_1s2, self.shell_2s2])
        self.assertEqual(len(config), 2)
        self.assertEqual(config.total_electrons(), 4)
        self.assertIn(self.shell_1s2, config)

    def test_init_with_duplicate_shells(self):
        """Test initialization with shells having same structure - should combine."""
        shell1 = Shell(1, 0, 1)  # 1s1
        shell2 = Shell(1, 0, 1)  # another 1s1
        config = Configuration([shell1, shell2])
        self.assertEqual(len(config), 1)
        self.assertEqual(config.total_electrons(), 2)
        # Should contain 1s2
        result_shell = config.get_shell(1, 0)
        self.assertIsNotNone(result_shell)
        self.assertEqual(result_shell.occupation, 2)

    def test_init_with_zero_occupation(self):
        """Test initialization filters out zero occupation shells."""
        shell_zero = Shell(1, 0, 0)
        config = Configuration([self.shell_1s2, shell_zero])
        # Zero occupation shells are filtered out during Configuration initialization
        # Since shell_zero has 0 occupation, it should be ignored,
        # but shell_1s2 should also be ignored if it has same structure
        # Actually, let's use a different structure for testing
        shell_2s_zero = Shell(2, 0, 0)  # 2s0 (different structure)
        config = Configuration([self.shell_1s2, shell_2s_zero])
        self.assertEqual(len(config), 1)  # Only 1s2 should remain
        self.assertNotIn((2, 0, None), config)  # 2s0 structure should not be present

    def test_add_shell_basic(self):
        """Test adding shells to configuration."""
        config = Configuration()
        config.add_shell(self.shell_1s2)
        self.assertEqual(len(config), 1)
        self.assertEqual(config.total_electrons(), 2)

    def test_add_shell_combine_occupation(self):
        """Test adding shell with combine_occupation=True."""
        config = Configuration([Shell(1, 0, 1)])  # 1s1
        config.add_shell(Shell(1, 0, 1), combine_occupation=True)  # Add another 1s1
        result_shell = config.get_shell(1, 0)
        self.assertEqual(result_shell.occupation, 2)

    def test_add_shell_replace(self):
        """Test adding shell with combine_occupation=False (default) replaces."""
        config = Configuration([Shell(1, 0, 1)])  # 1s1
        config.add_shell(Shell(1, 0, 2), combine_occupation=False)  # Replace with 1s2
        result_shell = config.get_shell(1, 0)
        self.assertEqual(result_shell.occupation, 2)

    def test_add_shell_zero_occupation_removes(self):
        """Test adding shell with zero occupation removes it."""
        config = Configuration([self.shell_1s2])
        config.add_shell(Shell(1, 0, 0))  # Add 1s0
        self.assertEqual(len(config), 0)

    def test_add_shell_invalid_type(self):
        """Test adding non-Shell object raises TypeError."""
        config = Configuration()
        with self.assertRaises(TypeError):
            config.add_shell("not a shell")

    def test_add_shell_exceed_max_occupation(self):
        """Test combining shells that exceed max occupation raises ValueError."""
        config = Configuration([Shell(1, 0, 2)])  # 1s2 (full)
        with self.assertRaises(ValueError):
            config.add_shell(Shell(1, 0, 1), combine_occupation=True)  # Would make 1s3

    def test_get_shell_existing(self):
        """Test retrieving existing shell."""
        shell = self.ne_config.get_shell(2, 1)  # 2p from neon config
        self.assertIsNotNone(shell)
        self.assertEqual(shell.occupation, 6)

    def test_get_shell_nonexistent(self):
        """Test retrieving non-existent shell returns None."""
        shell = self.he_config.get_shell(2, 0)  # 2s not in helium
        self.assertIsNone(shell)

    def test_get_shell_with_j_quantum(self):
        """Test retrieving shell with j quantum number."""
        # Create a configuration with j-quantum numbers
        shell_2p_half = Shell(2, 1, 2, 0.5)  # 2p-
        shell_2p_3half = Shell(2, 1, 4, 1.5)  # 2p+
        config = Configuration([shell_2p_half, shell_2p_3half])

        retrieved_half = config.get_shell(2, 1, 0.5)
        retrieved_3half = config.get_shell(2, 1, 1.5)

        self.assertIsNotNone(retrieved_half)
        self.assertIsNotNone(retrieved_3half)
        self.assertEqual(retrieved_half.occupation, 2)
        self.assertEqual(retrieved_3half.occupation, 4)

    def test_shells_property_sorted(self):
        """Test that shells property returns sorted shells."""
        # Create unsorted configuration
        config = Configuration([self.shell_3s1, self.shell_1s2, self.shell_2s2])
        shells = config.shells

        # Should be sorted by energy (Madelung rule): 1s, 2s, 3s
        self.assertEqual(shells[0].n, 1)
        self.assertEqual(shells[1].n, 2)
        self.assertEqual(shells[2].n, 3)

    def test_total_electrons(self):
        """Test total electron calculation."""
        self.assertEqual(self.empty_config.total_electrons(), 0)
        self.assertEqual(self.he_config.total_electrons(), 2)
        self.assertEqual(self.ne_config.total_electrons(), 10)
        self.assertEqual(self.na_config.total_electrons(), 11)

    def test_copy(self):
        """Test deep copy functionality."""
        copy_config = self.ne_config.copy()

        # Should be equal but not the same object
        self.assertEqual(copy_config, self.ne_config)
        self.assertIsNot(copy_config, self.ne_config)

        # Modifying copy shouldn't affect original
        copy_config.add_shell(Shell(3, 0, 1))
        self.assertNotEqual(copy_config, self.ne_config)

    def test_get_ionstage_neutral(self):
        """Test ion stage calculation for neutral atoms."""
        # This test requires mendeleev - skip if not available
        try:
            import mendeleev

            # Test with Helium (Z=2, 2 electrons)
            ion_stage = self.he_config.get_ionstage(2)
            self.assertEqual(ion_stage, 0)  # Neutral

            # Test with Sodium (Z=11, 11 electrons)
            ion_stage = self.na_config.get_ionstage(11)
            self.assertEqual(ion_stage, 0)  # Neutral
        except ImportError:
            self.skipTest("mendeleev package not available")

    def test_get_ionstage_ions(self):
        """Test ion stage calculation for ions."""
        try:
            import mendeleev

            # Test with He+ (Z=2, 1 electron)
            he_plus = Configuration([Shell(1, 0, 1)])  # 1s1
            ion_stage = he_plus.get_ionstage(2)
            self.assertEqual(ion_stage, 1)  # +1 ion

            # Test with Na+ (Z=11, 10 electrons - same as Ne)
            na_plus = self.ne_config.copy()  # 10 electrons
            ion_stage = na_plus.get_ionstage(11)
            self.assertEqual(ion_stage, 1)  # +1 ion
        except ImportError:
            self.skipTest("mendeleev package not available")

    def test_get_ionstage_invalid_element(self):
        """Test ion stage with invalid element raises ValueError."""
        try:
            import mendeleev

            with self.assertRaises(ValueError):
                self.he_config.get_ionstage("invalid_element")
        except ImportError:
            self.skipTest("mendeleev package not available")

    def test_str_representation(self):
        """Test string representation of configurations."""
        self.assertEqual(str(self.empty_config), "")
        self.assertEqual(str(self.he_config), "1s2")

        # Neon should be properly sorted
        ne_str = str(self.ne_config)
        self.assertTrue(ne_str.startswith("1s2"))
        self.assertIn("2s2", ne_str)
        self.assertIn("2p6", ne_str)

    def test_to_string_default_separator(self):
        """Test to_string with default dot separator."""
        # Default should match str()
        self.assertEqual(self.ne_config.to_string(), str(self.ne_config))
        self.assertEqual(self.ne_config.to_string(), "1s2.2s2.2p6")

        # Empty config
        self.assertEqual(self.empty_config.to_string(), "")

    def test_to_string_space_separator(self):
        """Test to_string with space separator."""
        # Space-separated format (common in some programs)
        self.assertEqual(self.ne_config.to_string(separator=" "), "1s2 2s2 2p6")
        self.assertEqual(self.he_config.to_string(separator=" "), "1s2")

        # Sodium config
        self.assertEqual(self.na_config.to_string(separator=" "), "1s2 2s2 2p6 3s1")

    def test_to_string_compact_format(self):
        """Test to_string with empty separator (compact format)."""
        self.assertEqual(self.ne_config.to_string(separator=""), "1s22s22p6")
        self.assertEqual(self.he_config.to_string(separator=""), "1s2")

    def test_to_string_custom_separator(self):
        """Test to_string with custom separators."""
        # Comma separator
        self.assertEqual(self.ne_config.to_string(separator=", "), "1s2, 2s2, 2p6")

        # Hyphen separator
        self.assertEqual(self.he_config.to_string(separator="-"), "1s2")
        self.assertEqual(self.na_config.to_string(separator="-"), "1s2-2s2-2p6-3s1")

        # Underscore separator
        self.assertEqual(self.ne_config.to_string(separator="_"), "1s2_2s2_2p6")

    def test_to_string_multichar_separator(self):
        """Test to_string with multi-character separator."""
        self.assertEqual(self.ne_config.to_string(separator=" + "), "1s2 + 2s2 + 2p6")
        self.assertEqual(self.ne_config.to_string(separator=" | "), "1s2 | 2s2 | 2p6")

    def test_repr_representation(self):
        """Test repr representation."""
        repr_str = repr(self.he_config)
        self.assertIn("Configuration", repr_str)
        self.assertIn("Shell", repr_str)

    def test_equality(self):
        """Test configuration equality."""
        # Same configurations should be equal
        he_config2 = Configuration([Shell(1, 0, 2)])
        self.assertEqual(self.he_config, he_config2)

        # Different configurations should not be equal
        self.assertNotEqual(self.he_config, self.ne_config)

        # Empty configs should be equal
        empty2 = Configuration()
        self.assertEqual(self.empty_config, empty2)

    def test_equality_with_non_configuration(self):
        """Test equality with non-Configuration objects."""
        self.assertNotEqual(self.he_config, "not a configuration")
        self.assertNotEqual(self.he_config, [self.shell_1s2])

    def test_len(self):
        """Test length (number of distinct shell structures)."""
        self.assertEqual(len(self.empty_config), 0)
        self.assertEqual(len(self.he_config), 1)
        self.assertEqual(len(self.ne_config), 3)

    def test_iter(self):
        """Test iteration over shells."""
        shells = list(self.ne_config)
        self.assertEqual(len(shells), 3)
        # Should be sorted
        self.assertEqual(shells[0].n, 1)  # 1s
        self.assertEqual(shells[1].n, 2)  # 2s
        self.assertEqual(shells[2].n, 2)  # 2p

    def test_hash(self):
        """Test that Configuration objects are hashable."""
        # Should be able to put in a set
        config_set = {self.he_config, self.ne_config}
        self.assertEqual(len(config_set), 2)

        # Equal configurations should have same hash
        he_config2 = Configuration([Shell(1, 0, 2)])
        self.assertEqual(hash(self.he_config), hash(he_config2))

    def test_contains_shell_object(self):
        """Test __contains__ with Shell objects."""
        # Exact shell should be found
        self.assertIn(self.shell_1s2, self.he_config)

        # Shell with different occupation should not be found
        shell_1s1 = Shell(1, 0, 1)
        self.assertNotIn(shell_1s1, self.he_config)

        # Shell not in config should not be found
        self.assertNotIn(self.shell_3s1, self.he_config)

    def test_contains_tuple(self):
        """Test __contains__ with (n, l, j) tuples."""
        # Structure present should be found
        self.assertIn((1, 0, None), self.he_config)

        # Structure not present should not be found
        self.assertNotIn((2, 0, None), self.he_config)

    def test_contains_invalid_type(self):
        """Test __contains__ with invalid types raises TypeError."""
        with self.assertRaises(TypeError):
            "invalid" in self.he_config

        with self.assertRaises(TypeError):
            (1, 0) in self.he_config  # Wrong tuple length

    def test_from_string_basic(self):
        """Test parsing basic configuration strings."""
        # Single shell
        config = Configuration.from_string("1s2")
        self.assertEqual(len(config), 1)
        self.assertEqual(config.total_electrons(), 2)

        # Multiple shells with dots
        config = Configuration.from_string("1s2.2s2.2p6")
        self.assertEqual(len(config), 3)
        self.assertEqual(config.total_electrons(), 10)

        # Multiple shells with spaces
        config = Configuration.from_string("1s2 2s2 2p6")
        self.assertEqual(len(config), 3)
        self.assertEqual(config.total_electrons(), 10)

    def test_from_string_empty(self):
        """Test parsing empty string."""
        config = Configuration.from_string("")
        self.assertEqual(len(config), 0)

        config = Configuration.from_string("   ")  # Whitespace only
        self.assertEqual(len(config), 0)

    def test_from_string_with_j_quantum(self):
        """Test parsing strings with j quantum numbers."""
        config = Configuration.from_string("2p-2.2p+4")
        self.assertEqual(len(config), 2)
        self.assertEqual(config.total_electrons(), 6)

        # Check j quantum numbers
        p_minus = config.get_shell(2, 1, 0.5)
        p_plus = config.get_shell(2, 1, 1.5)
        self.assertIsNotNone(p_minus)
        self.assertIsNotNone(p_plus)
        self.assertEqual(p_minus.occupation, 2)
        self.assertEqual(p_plus.occupation, 4)

    def test_from_string_invalid(self):
        """Test parsing invalid configuration strings."""
        with self.assertRaises(ValueError):
            Configuration.from_string("invalid_shell")

        with self.assertRaises(ValueError):
            Configuration.from_string("1s2.invalid.2p6")

    def test_from_element_neutral(self):
        """Test creating configuration from neutral elements."""
        try:
            import mendeleev

            # Test Hydrogen (Z=1)
            h_config = Configuration.from_element(1)
            self.assertEqual(h_config.total_electrons(), 1)

            # Test Helium (Z=2)
            he_config = Configuration.from_element("He")
            self.assertEqual(he_config.total_electrons(), 2)

            # Test Lithium by symbol (names need to be properly capitalized)
            li_config = Configuration.from_element("Li")
            self.assertEqual(li_config.total_electrons(), 3)

        except ImportError:
            self.skipTest("mendeleev package not available")

    def test_from_element_ions(self):
        """Test creating configuration from ions."""
        try:
            import mendeleev

            # Test He+ (should have 1 electron)
            he_plus = Configuration.from_element(2, ion_charge=1)
            self.assertEqual(he_plus.total_electrons(), 1)

            # Test Li+ (should have 2 electrons, same as He)
            li_plus = Configuration.from_element(3, ion_charge=1)
            self.assertEqual(li_plus.total_electrons(), 2)

        except ImportError:
            self.skipTest("mendeleev package not available")

    def test_from_element_fully_ionized(self):
        """Test creating configuration from fully ionized atoms."""
        try:
            import mendeleev

            # H+ should result in empty configuration (mendeleev returns empty string)
            h_plus = Configuration.from_element(1, ion_charge=1)
            self.assertEqual(h_plus.total_electrons(), 0)
            self.assertEqual(len(h_plus), 0)

        except ImportError:
            self.skipTest("mendeleev package not available")

    def test_from_element_invalid(self):
        """Test from_element with invalid inputs."""
        try:
            import mendeleev

            # Invalid element
            with self.assertRaises(ValueError):
                Configuration.from_element("invalid")

            # Negative ion charge (not supported)
            with self.assertRaises(ValueError):
                Configuration.from_element(1, ion_charge=-1)

            # Ion charge too high
            with self.assertRaises(ValueError):
                Configuration.from_element(1, ion_charge=2)  # Can't remove 2e from H

        except ImportError:
            self.skipTest("mendeleev package not available")

    def test_from_compact_string_basic(self):
        """Test parsing compact configuration strings."""
        # Single shell
        config = Configuration.from_compact_string("1*2")
        self.assertEqual(config.total_electrons(), 2)

        # Multiple shells
        config = Configuration.from_compact_string("1*2.2*8")
        self.assertEqual(config.total_electrons(), 10)

        # Should have proper subshell distribution
        # 1*2 -> 1s2, 2*8 -> 2s2.2p6
        self.assertEqual(len(config), 3)  # 1s, 2s, 2p

    def test_from_compact_string_permutations(self):
        """Test compact string with permutation generation."""
        configs = Configuration.from_compact_string("2*7", generate_permutations=True)
        self.assertIsInstance(configs, list)
        self.assertGreater(len(configs), 1)  # Should generate multiple permutations

        # All should have 7 electrons in n=2 shells
        for config in configs:
            n2_electrons = sum(shell.occupation for shell in config if shell.n == 2)
            self.assertEqual(n2_electrons, 7)

    def test_from_compact_string_empty(self):
        """Test compact string parsing with empty input."""
        config = Configuration.from_compact_string("")
        self.assertEqual(len(config), 0)

        configs = Configuration.from_compact_string("", generate_permutations=True)
        self.assertEqual(configs, [])

    def test_from_compact_string_zero_electrons(self):
        """Test compact string with zero electrons."""
        config = Configuration.from_compact_string("1*0.2*8")
        self.assertEqual(config.total_electrons(), 8)
        # Should not have any n=1 shells
        self.assertIsNone(config.get_shell(1, 0))

    def test_from_compact_string_invalid(self):
        """Test compact string parsing with invalid input."""
        with self.assertRaises(ValueError):
            Configuration.from_compact_string("invalid")

        with self.assertRaises(ValueError):
            Configuration.from_compact_string("1*100")  # Too many electrons for n=1

        with self.assertRaises(ValueError):
            Configuration.from_compact_string("0*2")  # n must be >= 1

    def test_remove_filled_shells(self):
        """Test removing fully filled shells."""
        # Create config with mixed filled/partial shells
        filled_shell = Shell(1, 0, 2)  # 1s2 (filled)
        partial_shell = Shell(2, 0, 1)  # 2s1 (partial)
        config = Configuration([filled_shell, partial_shell])

        partial_config = config.remove_filled_shells()

        # Should only contain partial shells
        self.assertEqual(len(partial_config), 1)
        self.assertIn(partial_shell, partial_config)
        self.assertNotIn((1, 0, None), partial_config)  # 1s should be removed

    def test_get_holes(self):
        """Test hole calculation."""
        # Create config with holes
        partial_shell = Shell(2, 1, 4)  # 2p4 (2 holes from 2p6)
        config = Configuration([self.shell_1s2, partial_shell])

        holes = config.get_holes()

        # Should have holes in 2p
        self.assertIn("2p", holes)
        self.assertEqual(holes["2p"], 2)

        # Filled shells should not appear
        self.assertNotIn("1s", holes)

    def test_get_holes_with_j_quantum(self):
        """Test hole calculation with j quantum numbers."""
        # Create shells with j quantum numbers and holes
        p_minus_partial = Shell(2, 1, 1, 0.5)  # 2p- with 1 electron (1 hole)
        p_plus_full = Shell(2, 1, 4, 1.5)  # 2p+ filled
        config = Configuration([p_minus_partial, p_plus_full])

        holes = config.get_holes()

        # Should show hole in 2p-
        self.assertIn("2p-", holes)
        self.assertEqual(holes["2p-"], 1)

        # No holes in 2p+
        self.assertNotIn("2p+", holes)

    def test_compare_configurations(self):
        """Test comparing two configurations."""
        # Create two different configs
        config1 = Configuration([Shell(1, 0, 2), Shell(2, 0, 1)])  # 1s2.2s1
        config2 = Configuration([Shell(1, 0, 1), Shell(2, 0, 2)])  # 1s1.2s2

        differences = config1.compare(config2)

        # Should show differences in both shells
        self.assertIn("1s", differences)
        self.assertIn("2s", differences)
        self.assertEqual(differences["1s"], 1)  # |2-1| = 1
        self.assertEqual(differences["2s"], 1)  # |1-2| = 1

    def test_compare_identical_configurations(self):
        """Test comparing identical configurations."""
        differences = self.he_config.compare(self.he_config)
        self.assertEqual(len(differences), 0)

    def test_compare_with_missing_shells(self):
        """Test comparing configs where one has shells the other doesn't."""
        differences = self.he_config.compare(self.ne_config)

        # Should show differences for shells present in only one config
        self.assertIn("2s", differences)
        self.assertIn("2p", differences)
        self.assertEqual(differences["2s"], 2)  # |0-2| = 2
        self.assertEqual(differences["2p"], 6)  # |0-6| = 6

    def test_compare_invalid_type(self):
        """Test comparing with non-Configuration object."""
        with self.assertRaises(TypeError):
            self.he_config.compare("not a config")

    def test_split_core_valence_string(self):
        """Test splitting configuration into core and valence using strings."""
        # Define core as 1s
        core_shells = ["1s"]
        core_config, valence_config = self.ne_config.split_core_valence(core_shells)

        # Core should contain 1s2
        self.assertEqual(len(core_config), 1)
        self.assertEqual(core_config.total_electrons(), 2)
        self.assertIn((1, 0, None), core_config)

        # Valence should contain 2s2.2p6
        self.assertEqual(len(valence_config), 2)
        self.assertEqual(valence_config.total_electrons(), 8)
        self.assertIn((2, 0, None), valence_config)
        self.assertIn((2, 1, None), valence_config)

    def test_split_core_valence_tuples(self):
        """Test splitting using structure tuples."""
        core_shells = [(1, 0, None), (2, 0, None)]  # 1s and 2s
        core_config, valence_config = self.ne_config.split_core_valence(core_shells)

        # Core should contain 1s2.2s2
        self.assertEqual(len(core_config), 2)
        self.assertEqual(core_config.total_electrons(), 4)

        # Valence should contain 2p6
        self.assertEqual(len(valence_config), 1)
        self.assertEqual(valence_config.total_electrons(), 6)

    def test_split_core_valence_empty_core(self):
        """Test splitting with empty core definition."""
        core_config, valence_config = self.ne_config.split_core_valence([])

        # Core should be empty
        self.assertEqual(len(core_config), 0)

        # Valence should contain everything
        self.assertEqual(valence_config, self.ne_config)

    def test_split_core_valence_invalid_string(self):
        """Test splitting with invalid core string."""
        with self.assertRaises(ValueError):
            self.ne_config.split_core_valence(["invalid_shell"])

    def test_split_core_valence_invalid_type(self):
        """Test splitting with invalid core definition type."""
        with self.assertRaises(TypeError):
            self.ne_config.split_core_valence([123])  # Invalid type

    def test_generate_hole_configurations_single(self):
        """Test generating single hole configurations."""
        configs = self.he_config.generate_hole_configurations(num_holes=1)

        # Should generate one config with 1 electron
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].total_electrons(), 1)

    def test_generate_hole_configurations_multiple(self):
        """Test generating multiple hole configurations."""
        configs = self.ne_config.generate_hole_configurations(num_holes=2)

        # Should generate multiple unique configs
        self.assertGreater(len(configs), 1)

        # All should have 8 electrons (10 - 2 holes)
        for config in configs:
            self.assertEqual(config.total_electrons(), 8)

    def test_generate_hole_configurations_invalid(self):
        """Test generating holes with invalid parameters."""
        # Too many holes
        with self.assertRaises(ValueError):
            self.he_config.generate_hole_configurations(num_holes=3)  # Only 2 electrons

        # Invalid hole count
        with self.assertRaises(ValueError):
            self.he_config.generate_hole_configurations(num_holes=0)

        with self.assertRaises(ValueError):
            self.he_config.generate_hole_configurations(num_holes=-1)

    def test_generate_excitations_basic(self):
        """Test generating basic excitations."""
        # Excite to 3s from He ground state
        target_shells = ["3s"]
        configs = self.he_config.generate_excitations(target_shells, excitation_level=1)

        # Should generate excited configurations
        self.assertGreater(len(configs), 0)

        # All should still have 2 electrons
        for config in configs:
            self.assertEqual(config.total_electrons(), 2)

    def test_generate_excitations_multiple_targets(self):
        """Test excitations with multiple target shells."""
        target_shells = ["3s", "3p", "3d"]
        configs = self.he_config.generate_excitations(target_shells, excitation_level=1)

        # Should generate multiple different excited states
        self.assertGreater(len(configs), 1)

    def test_generate_excitations_with_source_restriction(self):
        """Test excitations with restricted source shells."""
        target_shells = ["3s"]
        source_shells = ["1s"]  # Only allow excitation from 1s
        configs = self.he_config.generate_excitations(
            target_shells, excitation_level=1, source_shells=source_shells
        )

        # Should still generate configurations
        self.assertGreater(len(configs), 0)

    def test_generate_excitations_invalid_target(self):
        """Test excitations with invalid target shells."""
        with self.assertRaises(ValueError):
            self.he_config.generate_excitations(["invalid_shell"], excitation_level=1)

    def test_generate_excitations_invalid_source(self):
        """Test excitations with invalid source shells."""
        with self.assertRaises(ValueError):
            self.he_config.generate_excitations(
                ["3s"], excitation_level=1, source_shells=["invalid_shell"]
            )

    def test_generate_excitations_invalid_level(self):
        """Test excitations with invalid excitation level."""
        with self.assertRaises(ValueError):
            self.he_config.generate_excitations(["3s"], excitation_level=0)

    def test_generate_recombined_configurations_basic(self):
        """Test basic recombined configuration generation."""
        # Start with Be-like configuration (1s2.2s2)
        be_config = Configuration.from_string("1s2.2s2")

        # Generate recombined configs with max_n=3, max_l=1 (s and p only)
        recombined = be_config.generate_recombined_configurations(max_n=3, max_l=1)

        # Should generate configurations
        self.assertGreater(len(recombined), 0)

        # All should have one more electron than original
        for config in recombined:
            self.assertEqual(config.total_electrons(), 5)

        # Should be unique
        self.assertEqual(len(recombined), len(set(recombined)))

    def test_generate_recombined_configurations_default_max_l(self):
        """Test recombined configuration generation with default max_l."""
        he_config = Configuration.from_string("1s2")

        # max_l=None should default to n-1 for each n
        recombined = he_config.generate_recombined_configurations(max_n=3)

        # Should generate multiple configs
        self.assertGreater(len(recombined), 0)

        # Should include s, p, and d orbitals for n=3
        config_strings = [str(c) for c in recombined]
        self.assertTrue(any("3d" in s for s in config_strings))
        self.assertTrue(any("3p" in s for s in config_strings))
        self.assertTrue(any("3s" in s for s in config_strings))

    def test_generate_recombined_configurations_full_shells(self):
        """Test that recombination skips full shells."""
        # 1s is full with 2 electrons
        he_config = Configuration.from_string("1s2")

        # Generate with max_n=1 (only 1s available)
        recombined = he_config.generate_recombined_configurations(max_n=1, max_l=0)

        # Should generate nothing since 1s is full
        self.assertEqual(len(recombined), 0)

        # With max_n=2, should generate 2s and 2p configs
        recombined = he_config.generate_recombined_configurations(max_n=2, max_l=1)
        self.assertGreater(len(recombined), 0)
        # None should have more than 2 electrons in 1s
        for config in recombined:
            shell_1s = config.get_shell(1, 0)
            if shell_1s:
                self.assertEqual(shell_1s.occupation, 2)

    def test_generate_recombined_configurations_uniqueness(self):
        """Test that recombined configurations are unique."""
        config = Configuration.from_string("1s2.2s1")
        recombined = config.generate_recombined_configurations(max_n=3, max_l=2)

        # Convert to strings for comparison
        config_strings = [str(c) for c in recombined]

        # Should have no duplicates
        self.assertEqual(len(config_strings), len(set(config_strings)))

    def test_generate_recombined_configurations_sorted(self):
        """Test that recombined configurations are sorted."""
        config = Configuration.from_string("1s2")
        recombined = config.generate_recombined_configurations(max_n=3, max_l=1)

        # Convert to strings
        config_strings = [str(c) for c in recombined]

        # Should be sorted
        self.assertEqual(config_strings, sorted(config_strings))

    def test_generate_recombined_configurations_invalid_max_n(self):
        """Test recombination with invalid max_n."""
        config = Configuration.from_string("1s2")

        # max_n must be >= 1
        with self.assertRaises(ValueError):
            config.generate_recombined_configurations(max_n=0)

        with self.assertRaises(ValueError):
            config.generate_recombined_configurations(max_n=-1)

    def test_generate_recombined_configurations_invalid_max_l(self):
        """Test recombination with invalid max_l."""
        config = Configuration.from_string("1s2")

        # max_l must be non-negative
        with self.assertRaises(ValueError):
            config.generate_recombined_configurations(max_n=3, max_l=-1)

    def test_generate_recombined_configurations_large_n(self):
        """Test recombination with larger n values."""
        h_config = Configuration.from_string("1s1")

        # Generate up to n=5
        recombined = h_config.generate_recombined_configurations(max_n=5, max_l=3)

        # Should generate many configurations
        self.assertGreater(len(recombined), 10)

        # All should have 2 electrons
        for config in recombined:
            self.assertEqual(config.total_electrons(), 2)

    # --- Tests for generate_recombined_configurations_batch ---

    def test_generate_recombined_configurations_batch_basic(self):
        """Test batch generation with multiple configurations."""
        # Create two simple configurations
        config1 = Configuration.from_string("1s2")  # He-like
        config2 = Configuration.from_string("1s2.2s1")  # Li-like

        configs = [config1, config2]
        recombined = Configuration.generate_recombined_configurations_batch(
            configs, max_n=3, max_l=1
        )

        # Should have results from both
        self.assertGreater(len(recombined), 0)

        # Check electron counts are correct
        for config in recombined:
            # Should be either 3 electrons (from config1) or 4 electrons (from config2)
            self.assertIn(config.total_electrons(), [3, 4])

        # All should be unique
        self.assertEqual(len(recombined), len(set(recombined)))

    def test_generate_recombined_configurations_batch_uniqueness(self):
        """Test that batch method removes duplicates across different inputs."""
        # Create two configs that will generate some overlapping results
        config1 = Configuration.from_string("1s2")
        config2 = Configuration.from_string("1s2")  # Same as config1

        configs = [config1, config2]
        recombined = Configuration.generate_recombined_configurations_batch(
            configs, max_n=2, max_l=1
        )

        # Generate from single config for comparison
        single_result = config1.generate_recombined_configurations(max_n=2, max_l=1)

        # Should be the same as single config (no duplicates)
        self.assertEqual(len(recombined), len(single_result))
        self.assertEqual(set(recombined), set(single_result))

    def test_generate_recombined_configurations_batch_sorted(self):
        """Test that batch results are sorted."""
        configs = [
            Configuration.from_string("1s2"),
            Configuration.from_string("1s2.2s1"),
        ]

        recombined = Configuration.generate_recombined_configurations_batch(
            configs, max_n=2, max_l=1
        )

        # Check sorting by converting to strings
        string_list = [str(c) for c in recombined]
        self.assertEqual(string_list, sorted(string_list))

    def test_generate_recombined_configurations_batch_empty_list(self):
        """Test batch method with empty configuration list."""
        with self.assertRaises(ValueError) as context:
            Configuration.generate_recombined_configurations_batch([], max_n=3)
        self.assertIn("empty", str(context.exception).lower())

    def test_generate_recombined_configurations_batch_invalid_type(self):
        """Test batch method with non-Configuration object in list."""
        configs = [
            Configuration.from_string("1s2"),
            "not a config",  # Invalid type
        ]

        with self.assertRaises(TypeError) as context:
            Configuration.generate_recombined_configurations_batch(configs, max_n=3)
        self.assertIn("Configuration", str(context.exception))

    def test_generate_recombined_configurations_batch_single_config(self):
        """Test batch method with single configuration (should work)."""
        config = Configuration.from_string("1s2")
        recombined = Configuration.generate_recombined_configurations_batch(
            [config], max_n=3, max_l=1
        )

        # Should be same as calling method on single config
        single_result = config.generate_recombined_configurations(max_n=3, max_l=1)
        self.assertEqual(set(recombined), set(single_result))

    def test_generate_recombined_configurations_batch_practical_example(self):
        """Test batch method with a practical use case: Li-like ions."""
        # Create ground state Li-like
        li_ground = Configuration.from_string("1s2.2s1")

        # Create a few excited Li-like states
        li_excited = li_ground.generate_excitations(["2p", "3s", "3p", "3d"], 1, ["2s"])

        # Combine ground + excited states
        all_li_like = [li_ground] + li_excited

        # Generate all recombined configs at once
        recombined = Configuration.generate_recombined_configurations_batch(
            all_li_like, max_n=4, max_l=2
        )

        # Should have many unique configurations
        self.assertGreater(len(recombined), 10)

        # All should have 4 electrons (Li-like has 3, recombined has 4)
        for config in recombined:
            self.assertEqual(config.total_electrons(), 4)

        # All should be unique
        self.assertEqual(len(recombined), len(set(recombined)))

    def test_generate_recombined_configurations_batch_invalid_params(self):
        """Test batch method propagates parameter validation errors."""
        configs = [Configuration.from_string("1s2")]

        # Invalid max_n
        with self.assertRaises(ValueError):
            Configuration.generate_recombined_configurations_batch(
                configs, max_n=0, max_l=1
            )

        # Invalid max_l
        with self.assertRaises(ValueError):
            Configuration.generate_recombined_configurations_batch(
                configs, max_n=3, max_l=-1
            )

    # --- Tests for generate_doubly_excited_autoionizing ---

    def test_generate_doubly_excited_autoionizing_basic(self):
        """Test basic doubly-excited autoionizing generation."""
        # Li-like configurations (3 electrons)
        li_configs = [
            Configuration.from_string("1s2.2p1"),
            Configuration.from_string("1s2.3s1"),
        ]

        # Generate doubly-excited (4 electrons with core hole)
        autoionizing = Configuration.generate_doubly_excited_autoionizing(
            li_configs, max_n=3, max_l=1
        )

        # Should have results
        self.assertGreater(len(autoionizing), 0)

        # All should have 4 electrons
        for config in autoionizing:
            self.assertEqual(config.total_electrons(), 4)

        # All should be unique
        self.assertEqual(len(autoionizing), len(set(autoionizing)))

    def test_generate_doubly_excited_autoionizing_specific_examples(self):
        """Test that specific doubly-excited configs are generated."""
        li_configs = [
            Configuration.from_string("1s2.2p1"),
            Configuration.from_string("1s2.3s1"),
        ]

        autoionizing = Configuration.generate_doubly_excited_autoionizing(
            li_configs, max_n=3, max_l=2
        )

        # Check for specific configurations
        target1 = Configuration.from_string("1s1.2s2.2p1")
        target2 = Configuration.from_string("1s1.2s1.2p2")
        target3 = Configuration.from_string("1s1.2s2.3s1")

        self.assertIn(target1, autoionizing)
        self.assertIn(target2, autoionizing)
        self.assertIn(target3, autoionizing)

    def test_generate_doubly_excited_autoionizing_empty_list(self):
        """Test with empty configuration list."""
        with self.assertRaises(ValueError) as context:
            Configuration.generate_doubly_excited_autoionizing([], max_n=3)
        self.assertIn("empty", str(context.exception).lower())

    def test_generate_doubly_excited_autoionizing_invalid_type(self):
        """Test with non-Configuration object in list."""
        configs = [
            Configuration.from_string("1s2.2s1"),
            "not a config",
        ]

        with self.assertRaises(TypeError) as context:
            Configuration.generate_doubly_excited_autoionizing(configs, max_n=3)
        self.assertIn("Configuration", str(context.exception))

    def test_generate_doubly_excited_autoionizing_invalid_num_holes(self):
        """Test with invalid num_holes parameter."""
        configs = [Configuration.from_string("1s2.2s1")]

        # Zero holes
        with self.assertRaises(ValueError):
            Configuration.generate_doubly_excited_autoionizing(
                configs, max_n=3, num_holes=0
            )

        # Negative holes
        with self.assertRaises(ValueError):
            Configuration.generate_doubly_excited_autoionizing(
                configs, max_n=3, num_holes=-1
            )

    def test_generate_doubly_excited_autoionizing_sorted(self):
        """Test that results are sorted."""
        li_configs = [Configuration.from_string("1s2.2s1")]

        autoionizing = Configuration.generate_doubly_excited_autoionizing(
            li_configs, max_n=3, max_l=1
        )

        # Check sorting
        string_list = [str(c) for c in autoionizing]
        self.assertEqual(string_list, sorted(string_list))

    def test_generate_doubly_excited_autoionizing_multiple_holes(self):
        """Test with multiple holes."""
        # Start with more electrons
        configs = [Configuration.from_string("1s2.2s2.2p2")]  # 6 electrons

        # Remove 2 electrons, add 3 back → 7 electrons
        autoionizing = Configuration.generate_doubly_excited_autoionizing(
            configs, max_n=3, max_l=1, num_holes=2
        )

        # Should have results
        self.assertGreater(len(autoionizing), 0)

        # All should have 7 electrons
        for config in autoionizing:
            self.assertEqual(config.total_electrons(), 7)

    def test_generate_doubly_excited_autoionizing_insufficient_electrons(self):
        """Test with configs that don't have enough electrons for holes."""
        # Only 1 electron, can't create 2 holes
        configs = [Configuration.from_string("1s1")]

        autoionizing = Configuration.generate_doubly_excited_autoionizing(
            configs, max_n=3, max_l=1, num_holes=2
        )

        # Should return empty list (can't create holes)
        self.assertEqual(len(autoionizing), 0)

    def test_generate_doubly_excited_autoionizing_uniqueness(self):
        """Test that duplicates from different inputs are removed."""
        # Create configs that will produce overlapping results
        li_configs = [
            Configuration.from_string("1s2.2s1"),
            Configuration.from_string("1s2.2s1"),  # Duplicate
        ]

        autoionizing = Configuration.generate_doubly_excited_autoionizing(
            li_configs, max_n=3, max_l=1
        )

        # Should not have duplicates
        self.assertEqual(len(autoionizing), len(set(autoionizing)))

    # --- Tests for configurations_to_string ---

    def test_configurations_to_string_basic(self):
        """Test basic conversion of configuration list to string."""
        configs = [
            Configuration.from_string("1s2"),
            Configuration.from_string("1s2.2s1"),
            Configuration.from_string("1s2.2p1"),
        ]

        result = Configuration.configurations_to_string(configs, list=False)
        expected = "1s2\n1s2.2s1\n1s2.2p1"
        self.assertEqual(result, expected)

    def test_configurations_to_string_numbered(self):
        """Test numbered output."""
        configs = [
            Configuration.from_string("1s2"),
            Configuration.from_string("1s2.2s1"),
            Configuration.from_string("1s2.2p1"),
        ]

        result = Configuration.configurations_to_string(
            configs, numbered=True, list=False
        )
        expected = "1. 1s2\n2. 1s2.2s1\n3. 1s2.2p1"
        self.assertEqual(result, expected)

    def test_configurations_to_string_custom_start_index(self):
        """Test numbered output with custom start index."""
        configs = [
            Configuration.from_string("1s2"),
            Configuration.from_string("1s2.2s1"),
        ]

        result = Configuration.configurations_to_string(
            configs, numbered=True, start_index=0, list=False
        )
        expected = "0. 1s2\n1. 1s2.2s1"
        self.assertEqual(result, expected)

    def test_configurations_to_string_space_separator(self):
        """Test with space separator (FAC format)."""
        configs = [
            Configuration.from_string("1s2.2s2"),
            Configuration.from_string("1s2.2s1.2p1"),
        ]

        result = Configuration.configurations_to_string(
            configs, separator=" ", list=False
        )
        expected = "1s2 2s2\n1s2 2s1 2p1"
        self.assertEqual(result, expected)

    def test_configurations_to_string_compact_format(self):
        """Test with compact format (no separator)."""
        configs = [
            Configuration.from_string("1s2.2s2"),
            Configuration.from_string("1s2.2p2"),
        ]

        result = Configuration.configurations_to_string(
            configs, separator="", list=False
        )
        expected = "1s22s2\n1s22p2"
        self.assertEqual(result, expected)

    def test_configurations_to_string_custom_line_separator(self):
        """Test with custom line separator."""
        configs = [
            Configuration.from_string("1s2"),
            Configuration.from_string("1s2.2s1"),
            Configuration.from_string("1s2.2p1"),
        ]

        # Comma-separated on one line
        result = Configuration.configurations_to_string(
            configs, line_separator=", ", list=False
        )
        expected = "1s2, 1s2.2s1, 1s2.2p1"
        self.assertEqual(result, expected)

    def test_configurations_to_string_semicolon_separator(self):
        """Test with semicolon line separator."""
        configs = [
            Configuration.from_string("1s2"),
            Configuration.from_string("1s2.2s1"),
        ]

        result = Configuration.configurations_to_string(
            configs, line_separator="; ", list=False
        )
        expected = "1s2; 1s2.2s1"
        self.assertEqual(result, expected)

    def test_configurations_to_string_numbered_with_spaces(self):
        """Test numbered output with space separator."""
        configs = [
            Configuration.from_string("1s2.2s2"),
            Configuration.from_string("1s2.2s1.2p1"),
        ]

        result = Configuration.configurations_to_string(
            configs, separator=" ", numbered=True, list=False
        )
        expected = "1. 1s2 2s2\n2. 1s2 2s1 2p1"
        self.assertEqual(result, expected)

    def test_configurations_to_string_empty_list(self):
        """Test with empty configuration list."""
        with self.assertRaises(ValueError) as context:
            Configuration.configurations_to_string([])
        self.assertIn("empty", str(context.exception).lower())

    def test_configurations_to_string_invalid_type(self):
        """Test with non-Configuration object in list."""
        configs = [
            Configuration.from_string("1s2"),
            "not a config",  # Invalid type
        ]

        with self.assertRaises(TypeError) as context:
            Configuration.configurations_to_string(configs)
        self.assertIn("Configuration", str(context.exception))

    def test_configurations_to_string_single_config(self):
        """Test with single configuration (should work)."""
        configs = [Configuration.from_string("1s2.2s2.2p6")]

        result = Configuration.configurations_to_string(configs, list=False)
        expected = "1s2.2s2.2p6"
        self.assertEqual(result, expected)

    def test_configurations_to_string_large_start_index(self):
        """Test numbered with large start index."""
        configs = [
            Configuration.from_string("1s2"),
            Configuration.from_string("1s2.2s1"),
        ]

        result = Configuration.configurations_to_string(
            configs, numbered=True, start_index=100, list=False
        )
        expected = "100. 1s2\n101. 1s2.2s1"
        self.assertEqual(result, expected)

    def test_configurations_to_string_combined_formats(self):
        """Test combination of all formatting options."""
        configs = [
            Configuration.from_string("1s2.2s2"),
            Configuration.from_string("1s2.2s1.2p1"),
            Configuration.from_string("1s2.2p2"),
        ]

        # Numbered, space separator, comma line separator
        result = Configuration.configurations_to_string(
            configs,
            separator=" ",
            numbered=True,
            start_index=10,
            line_separator=", ",
            list=False,
        )
        expected = "10. 1s2 2s2, 11. 1s2 2s1 2p1, 12. 1s2 2p2"
        self.assertEqual(result, expected)

    def test_calculate_xray_label_ground_state(self):
        """Test X-ray labeling for ground state (no holes)."""
        labels = self.he_config.calculate_xray_label(self.he_config)
        self.assertEqual(labels, ["Ground"])

    def test_calculate_xray_label_with_holes(self):
        """Test X-ray labeling with holes."""
        # Create He+ (one hole in 1s)
        he_plus = Configuration([Shell(1, 0, 1)])  # 1s1
        labels = he_plus.calculate_xray_label(self.he_config)

        # Should contain K shell hole label
        self.assertIn("K", labels)

    def test_calculate_xray_label_excited_state(self):
        """Test X-ray labeling for excited states."""
        # Create excited He with electron in 2s (not a simple hole state)
        he_excited = Configuration([Shell(1, 0, 1), Shell(2, 0, 1)])  # 1s1.2s1
        labels = he_excited.calculate_xray_label(self.he_config)

        # Should be labeled as unknown/excited since it's not just holes
        self.assertEqual(labels, ["Unknown/Excited"])

    def test_calculate_xray_label_invalid_reference(self):
        """Test X-ray labeling with invalid reference."""
        with self.assertRaises(TypeError):
            self.he_config.calculate_xray_label("not a config")


if __name__ == "__main__":
    unittest.main()
