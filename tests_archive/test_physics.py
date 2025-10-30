"""
Unit tests for the atomkit.physics module.

Tests cover:
- Energy unit conversions
- Cross section to collision strength conversions
- Resonant excitation calculations
- Plotting utilities
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atomkit.physics.units import (
    EnergyConverter,
    cross_section_to_collision_strength,
    collision_strength_to_cross_section,
    energy_converter,
    RY_TO_EV,
    EV_TO_CM_INV,
)

from atomkit.physics.cross_sections import (
    LorentzianProfile,
    ResonantExcitationCalculator,
)


class TestEnergyConverter(unittest.TestCase):
    """Test energy unit conversions."""

    def setUp(self):
        self.converter = EnergyConverter()
        self.tolerance = 1e-10

    def test_ev_to_rydberg(self):
        """Test eV to Rydberg conversion."""
        # 1 Ry = 13.605693... eV
        self.assertAlmostEqual(
            self.converter.ev_to_rydberg(RY_TO_EV), 1.0, delta=self.tolerance
        )
        self.assertAlmostEqual(
            self.converter.ev_to_rydberg(27.21138624598894), 2.0, delta=self.tolerance
        )

    def test_rydberg_to_ev(self):
        """Test Rydberg to eV conversion."""
        self.assertAlmostEqual(
            self.converter.rydberg_to_ev(1.0), RY_TO_EV, delta=self.tolerance
        )
        self.assertAlmostEqual(
            self.converter.rydberg_to_ev(2.0), 2 * RY_TO_EV, delta=self.tolerance
        )

    def test_ev_to_wavenumber(self):
        """Test eV to cm^-1 conversion."""
        result = self.converter.ev_to_wavenumber(1.0)
        self.assertAlmostEqual(result, EV_TO_CM_INV, delta=1e-5)

    def test_wavenumber_to_ev(self):
        """Test cm^-1 to eV conversion."""
        wavenumber = 8065.54  # approximately 1 eV
        result = self.converter.wavenumber_to_ev(wavenumber)
        self.assertAlmostEqual(result, 1.0, delta=0.01)

    def test_convert_general(self):
        """Test general convert method."""
        # eV to Ry
        self.assertAlmostEqual(
            self.converter.convert(13.6, "eV", "Ry"), 1.0, delta=0.01
        )

        # Ry to eV
        self.assertAlmostEqual(
            self.converter.convert(1.0, "Ry", "eV"), RY_TO_EV, delta=self.tolerance
        )

        # Identity conversion
        self.assertEqual(self.converter.convert(100, "eV", "eV"), 100)

    def test_array_conversion(self):
        """Test conversion with numpy arrays."""
        energies = np.array([1.0, 10.0, 100.0])
        result = self.converter.ev_to_rydberg(energies)

        self.assertIsInstance(result, np.ndarray)
        assert isinstance(result, np.ndarray)  # Type narrowing for type checker
        self.assertEqual(len(result), 3)
        np.testing.assert_array_almost_equal(result, energies / RY_TO_EV, decimal=10)

    def test_invalid_unit(self):
        """Test that invalid units raise ValueError."""
        with self.assertRaises(ValueError):
            self.converter.convert(1.0, "eV", "invalid_unit")

        with self.assertRaises(ValueError):
            self.converter.convert(1.0, "invalid_unit", "eV")

    def test_hartree_conversion(self):
        """Test Hartree energy conversions."""
        # 1 Hartree = 27.2114 eV
        result = self.converter.ev_to_hartree(27.2114)
        self.assertAlmostEqual(result, 1.0, delta=0.001)

        result = self.converter.hartree_to_ev(1.0)
        self.assertAlmostEqual(result, 27.2114, delta=0.001)


class TestCollisionStrengthConversion(unittest.TestCase):
    """Test collision strength conversions."""

    def test_cross_section_to_collision_strength(self):
        """Test cross section to collision strength conversion."""
        sigma = 1e-16  # cm²
        energy = 100  # eV
        g_i = 2

        omega = cross_section_to_collision_strength(sigma, energy, g_i)

        # Collision strength should be dimensionless and positive
        self.assertGreater(omega, 0)
        self.assertIsInstance(omega, (float, np.floating))

    def test_collision_strength_to_cross_section(self):
        """Test collision strength to cross section conversion."""
        omega = 0.5
        energy = 100  # eV
        g_i = 2

        sigma = collision_strength_to_cross_section(omega, energy, g_i)

        # Cross section should be positive
        self.assertGreater(sigma, 0)
        self.assertIsInstance(sigma, (float, np.floating))

    def test_round_trip_conversion(self):
        """Test that conversions are inverses of each other."""
        sigma_original = 1e-17  # cm²
        energy = 50  # eV
        g_i = 4

        # Convert to collision strength and back
        omega = cross_section_to_collision_strength(sigma_original, energy, g_i)
        sigma_recovered = collision_strength_to_cross_section(omega, energy, g_i)

        self.assertAlmostEqual(
            sigma_original, sigma_recovered, delta=sigma_original * 1e-10
        )

    def test_array_conversion(self):
        """Test conversion with arrays."""
        sigma = np.array([1e-16, 1e-17, 1e-18])
        energy = np.array([50, 100, 200])
        g_i = 2

        omega = cross_section_to_collision_strength(sigma, energy, g_i)

        self.assertIsInstance(omega, np.ndarray)
        assert isinstance(omega, np.ndarray)  # Type narrowing for type checker
        self.assertEqual(len(omega), 3)
        self.assertTrue(np.all(omega > 0))

    def test_energy_dependence(self):
        """Test that collision strength scales correctly with energy."""
        sigma = 1e-16  # fixed cross section
        energies = np.array([10, 100, 1000])
        g_i = 2

        omegas = cross_section_to_collision_strength(sigma, energies, g_i)

        # For fixed sigma, omega should increase with energy (k² scaling)
        self.assertTrue(np.all(np.diff(omegas) > 0))


class TestLorentzianProfile(unittest.TestCase):
    """Test Lorentzian profile calculations."""

    def test_initialization(self):
        """Test Lorentzian profile initialization."""
        profile = LorentzianProfile(energy_center=100, gamma=10)

        self.assertEqual(profile.energy_center, 100)
        self.assertEqual(profile.gamma, 10)

    def test_evaluation_at_center(self):
        """Test that maximum is at center energy."""
        profile = LorentzianProfile(energy_center=100, gamma=10)

        # Evaluate at several energies
        energies = np.linspace(90, 110, 100)
        values = profile(energies)

        # Maximum should be at center
        max_idx = np.argmax(values)
        self.assertAlmostEqual(energies[max_idx], 100, delta=0.5)

    def test_normalization(self):
        """Test that Lorentzian has correct normalization."""
        profile = LorentzianProfile(energy_center=100, gamma=5)

        # Value at center
        center_value = profile(100)

        # Value at FWHM points (E0 ± Γ/2)
        hwhm_value = profile(100 + 2.5)

        # At FWHM, value should be half of maximum
        self.assertAlmostEqual(hwhm_value, center_value / 2, delta=center_value * 0.01)

    def test_array_evaluation(self):
        """Test evaluation with numpy array."""
        profile = LorentzianProfile(energy_center=50, gamma=2)
        energies = np.array([48, 49, 50, 51, 52])

        values = profile(energies)

        self.assertIsInstance(values, np.ndarray)
        assert isinstance(values, np.ndarray)  # Type narrowing for type checker
        self.assertEqual(len(values), 5)
        self.assertTrue(np.all(values > 0))


class TestResonantExcitationCalculator(unittest.TestCase):
    """Test ResonantExcitationCalculator class."""

    def setUp(self):
        """Create mock data for testing."""
        # Create mock levels dataframe
        self.levels = pd.DataFrame(
            {
                "level_index": [0, 1, 10, 11, 12],
                "energy": [0.0, 100.0, 200.0, 205.0, 210.0],
                "2j": [1, 3, 5, 3, 1],
                "configuration": [
                    "1s2",
                    "1s2 2p1",
                    "1s1 2s2",
                    "1s1 2s1 2p1",
                    "1s1 2p2",
                ],
                "ion_charge": [1, 1, 0, 0, 0],
            }
        )

        # Create mock autoionization data
        self.autoionization = pd.DataFrame(
            {
                "level_index_upper": [10, 10, 11, 11, 12],
                "level_index_lower": [0, 1, 0, 1, 1],
                "ai_rate": [1e12, 5e11, 8e11, 6e11, 4e11],
                "energy": [200.0, 100.0, 205.0, 105.0, 110.0],
            }
        )

        # Create mock transitions data
        self.transitions = pd.DataFrame(
            {
                "level_index_upper": [10, 11, 12],
                "level_index_lower": [0, 0, 1],
                "A": [1e10, 5e9, 3e9],
            }
        )

        self.calculator = ResonantExcitationCalculator(
            levels=self.levels,
            autoionization=self.autoionization,
            transitions=self.transitions,
        )

    def test_initialization(self):
        """Test calculator initialization."""
        self.assertIsNotNone(self.calculator.levels)
        self.assertIsNotNone(self.calculator.autoionization)
        self.assertIsNotNone(self.calculator.transitions)

    def test_get_level_by_index(self):
        """Test retrieving level by index."""
        level = self.calculator.get_level_by_index(0)

        self.assertEqual(level["level_index"], 0)
        self.assertEqual(level["energy"], 0.0)
        self.assertEqual(level["configuration"], "1s2")

    def test_get_level_by_index_not_found(self):
        """Test that invalid index raises error."""
        with self.assertRaises(ValueError):
            self.calculator.get_level_by_index(999)

    def test_get_level_by_config(self):
        """Test retrieving level by configuration string."""
        level = self.calculator.get_level_by_config("1s2 2p1", ion_charge=1)

        self.assertEqual(level["configuration"], "1s2 2p1")
        self.assertEqual(level["ion_charge"], 1)

    def test_calculate_total_decay_rate(self):
        """Test total decay rate calculation."""
        Aa_total, Ar_total = self.calculator.calculate_total_decay_rate(10)

        # Should sum autoionization rates for level 10
        expected_Aa = 1e12 + 5e11
        self.assertAlmostEqual(Aa_total, expected_Aa, delta=1e9)

        # Should get radiative rate for level 10
        expected_Ar = 1e10
        self.assertAlmostEqual(Ar_total, expected_Ar, delta=1e7)

    def test_calculate_branching_ratio(self):
        """Test branching ratio calculation."""
        # Branching to level 1 from resonance 10
        branching = self.calculator.calculate_branching_ratio(10, 1)

        # Should be Aa(10->1) / (Aa_total + Ar_total)
        Aa_to_1 = 5e11
        Aa_total = 1e12 + 5e11
        Ar_total = 1e10
        expected = Aa_to_1 / (Aa_total + Ar_total)

        self.assertAlmostEqual(branching, expected, delta=0.001)

    def test_calculate_capture_cross_section(self):
        """Test capture cross section calculation."""
        energy_grid = np.linspace(100, 300, 100)

        capture_cs, res_energy, gamma = self.calculator.calculate_capture_cross_section(
            initial_level=0,
            resonant_level_index=10,
            energy_grid=energy_grid,
        )

        # Should return arrays/values
        self.assertIsInstance(capture_cs, np.ndarray)
        self.assertEqual(len(capture_cs), 100)
        self.assertGreater(res_energy, 0)
        self.assertGreater(gamma, 0)

        # Maximum should be near resonance energy
        max_idx = np.argmax(capture_cs)
        self.assertAlmostEqual(energy_grid[max_idx], res_energy, delta=10)

    def test_calculate_resonant_excitation(self):
        """Test full resonant excitation calculation."""
        energy_grid = np.linspace(50, 250, 200)

        cs, info = self.calculator.calculate_resonant_excitation(
            initial_level=0,
            final_level=1,
            energy_grid=energy_grid,
        )

        # Check outputs
        self.assertIsInstance(cs, np.ndarray)
        self.assertEqual(len(cs), 200)
        self.assertIn("level_index", info)
        self.assertIn("energies", info)
        self.assertIn("widths", info)
        self.assertIn("contributions", info)

        # Should find resonances
        self.assertGreater(len(info["level_index"]), 0)

    def test_calculate_resonant_excitation_with_config_strings(self):
        """Test calculation using configuration strings."""
        energy_grid = np.linspace(50, 250, 100)

        cs, info = self.calculator.calculate_resonant_excitation(
            initial_level="1s2",
            final_level="1s2 2p1",
            energy_grid=energy_grid,
            ion_charge=1,
        )

        self.assertIsInstance(cs, np.ndarray)
        self.assertEqual(len(cs), 100)

    def test_no_resonances_found(self):
        """Test behavior when no resonances connect initial and final states."""
        # Create data with no connecting resonances
        levels = pd.DataFrame(
            {
                "level_index": [0, 1, 10],
                "energy": [0.0, 100.0, 200.0],
                "2j": [1, 3, 5],
                "configuration": ["1s2", "1s2 2p1", "1s1 2s2"],
                "ion_charge": [1, 1, 0],
            }
        )

        autoionization = pd.DataFrame(
            {
                "level_index_upper": [10],
                "level_index_lower": [0],
                "ai_rate": [1e12],
                "energy": [200.0],
            }
        )

        transitions = pd.DataFrame(
            {
                "level_index_upper": [10],
                "level_index_lower": [0],
                "A": [1e10],
            }
        )

        calc = ResonantExcitationCalculator(levels, autoionization, transitions)
        energy_grid = np.linspace(50, 250, 100)

        # Should work but give zero or very small cross section
        cs, info = calc.calculate_resonant_excitation(
            initial_level=0,
            final_level=1,
            energy_grid=energy_grid,
        )

        # Should return zeros when no resonances found
        self.assertTrue(np.all(cs == 0))


class TestPhysicsModule(unittest.TestCase):
    """Integration tests for the physics module."""

    def test_module_imports(self):
        """Test that all main components can be imported."""
        from atomkit.physics import (
            ResonantExcitationCalculator,
            calculate_resonant_excitation_cross_section,
            LorentzianProfile,
        )

        # Check they are callable/instantiable
        self.assertTrue(callable(ResonantExcitationCalculator))
        self.assertTrue(callable(calculate_resonant_excitation_cross_section))
        self.assertTrue(callable(LorentzianProfile))

    def test_energy_converter_instance(self):
        """Test that module-level energy_converter works."""
        self.assertIsInstance(energy_converter, EnergyConverter)

        # Test a conversion
        result = energy_converter.ev_to_rydberg(13.6)
        self.assertAlmostEqual(result, 1.0, delta=0.01)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
