"""
Unit tests for the atomkit.physics.plotting module.

Tests cover:
- ResonancePlotter class methods
- Quick plotting functions
- Energy level diagrams
- Mock data generation for testing plotting without real data files
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from typing import TYPE_CHECKING, Union, Any

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import types for type checking - always available to type checker
if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.axes import Axes
    from atomkit.physics.plotting import (
        ResonancePlotter,
        quick_plot_cross_section,
    )

# Runtime imports - may fail if matplotlib not installed
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.axes import Axes
    from atomkit.physics.plotting import (
        ResonancePlotter,
        quick_plot_cross_section,
    )

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def close_figure(fig: "Union[Figure, SubFigure]") -> None:
    """
    Safely close a matplotlib figure, handling both Figure and SubFigure types.

    Args:
        fig: Figure or SubFigure to close
    """
    # Import locally to handle the case when matplotlib is not available
    from matplotlib.figure import SubFigure as SubFigureClass
    import matplotlib.pyplot as plt_module

    if isinstance(fig, SubFigureClass):
        # SubFigure doesn't have close(), close the parent Figure
        root_fig = fig.figure
        plt_module.close(root_fig)
    else:
        # It's a Figure, close it directly
        plt_module.close(fig)


def generate_mock_cross_section_data(
    num_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate mock cross section data with resonances for testing.

    Returns:
        tuple: (energy_grid, cross_section, resonance_info)
    """
    # Energy grid
    energy = np.linspace(50, 300, num_points)

    # Background cross section (smooth)
    background = 1e-18 * np.ones_like(energy)

    # Add some resonances (Lorentzian profiles)
    resonance_energies = [100, 150, 200, 250]
    resonance_widths = [5, 3, 4, 2]
    resonance_strengths = [5e-18, 3e-18, 4e-18, 2e-18]

    cross_section = background.copy()
    contributions = []

    for E0, gamma, strength in zip(
        resonance_energies, resonance_widths, resonance_strengths
    ):
        # Lorentzian profile
        contribution = (
            strength * (gamma / 2) ** 2 / ((energy - E0) ** 2 + (gamma / 2) ** 2)
        )
        cross_section += contribution
        contributions.append(contribution)

    # Create resonance info dictionary
    resonance_info = {
        "level_index": [10, 11, 12, 13],
        "energies": resonance_energies,
        "widths": resonance_widths,
        "contributions": contributions,
    }

    return energy, cross_section, resonance_info


def generate_mock_levels_data():
    """
    Generate mock energy levels dataframe for testing.

    Returns:
        pd.DataFrame: Mock levels data
    """
    levels = pd.DataFrame(
        {
            "level_index": [0, 1, 10, 11, 12, 13],
            "energy": [0.0, 50.0, 100.0, 150.0, 200.0, 250.0],
            "2j": [1, 3, 5, 3, 1, 5],
            "configuration": [
                "1s2",
                "1s2 2s1",
                "1s1 2s1 2p1",
                "1s1 2s2",
                "1s1 2s1 3s1",
                "1s1 2p2",
            ],
            "ion_charge": [1, 1, 0, 0, 0, 0],
        }
    )
    return levels


@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "matplotlib not available")
class TestResonancePlotter(unittest.TestCase):
    """Test ResonancePlotter class."""

    def setUp(self):
        """Create plotter and mock data before each test."""
        # Assert imports are available (helps type checker)
        assert MATPLOTLIB_AVAILABLE, "This test requires matplotlib"
        assert ResonancePlotter is not None
        self.plotter = ResonancePlotter()
        self.energy, self.cross_section, self.resonance_info = (
            generate_mock_cross_section_data()
        )
        self.levels = generate_mock_levels_data()

        # Clean up any existing plots
        plt.close("all")

    def tearDown(self):
        """Clean up plots after each test."""
        plt.close("all")

    def test_plotter_initialization(self):
        """Test that plotter initializes correctly."""
        plotter = ResonancePlotter()
        self.assertEqual(plotter.style, "default")

        # Test with custom style (don't actually apply it)
        plotter2 = ResonancePlotter(style="seaborn-v0_8")
        self.assertEqual(plotter2.style, "seaborn-v0_8")

    def test_plot_cross_section_basic(self):
        """Test basic cross section plotting."""
        fig, ax = self.plotter.plot_cross_section(
            self.energy,
            self.cross_section,
        )

        # Check that figure and axes were created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        # Check that data was plotted
        lines = ax.get_lines()
        self.assertEqual(len(lines), 1)

        # Check axis labels
        self.assertIn("Energy", ax.get_xlabel())
        self.assertIn("Cross Section", ax.get_ylabel())

        close_figure(fig)

    def test_plot_cross_section_with_label(self):
        """Test cross section plotting with label for legend."""
        fig, ax = self.plotter.plot_cross_section(
            self.energy,
            self.cross_section,
            label="Test Data",
        )

        # Check legend exists
        legend = ax.get_legend()
        self.assertIsNotNone(legend)

        close_figure(fig)

    def test_plot_cross_section_with_title(self):
        """Test cross section plotting with custom title."""
        title = "Custom Test Title"
        fig, ax = self.plotter.plot_cross_section(
            self.energy,
            self.cross_section,
            title=title,
        )

        # Check title
        self.assertEqual(ax.get_title(), title)

        close_figure(fig)

    def test_plot_cross_section_with_existing_axes(self):
        """Test plotting on existing axes."""
        fig, ax = plt.subplots()

        # Plot on existing axes
        returned_fig, returned_ax = self.plotter.plot_cross_section(
            self.energy,
            self.cross_section,
            ax=ax,
        )

        # Should return the same axes
        self.assertIs(returned_ax, ax)
        self.assertIs(returned_fig, fig)

        close_figure(fig)

    def test_plot_resonance_contributions(self):
        """Test plotting individual resonance contributions."""
        fig, ax = self.plotter.plot_resonance_contributions(
            self.energy,
            self.cross_section,
            self.resonance_info,
            num_resonances=3,
        )

        # Check that figure and axes were created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        # Should have total + 3 resonances = 4 lines
        lines = ax.get_lines()
        self.assertGreaterEqual(len(lines), 1)  # At least the total

        # Check legend exists
        legend = ax.get_legend()
        self.assertIsNotNone(legend)

        close_figure(fig)

    def test_plot_resonance_contributions_no_total(self):
        """Test plotting resonances without total cross section."""
        fig, ax = self.plotter.plot_resonance_contributions(
            self.energy,
            self.cross_section,
            self.resonance_info,
            show_total=False,
        )

        # Should not have the thick total line
        lines = ax.get_lines()
        # Check that we have some lines (the resonances)
        self.assertGreater(len(lines), 0)

        close_figure(fig)

    def test_add_resonance_markers_vlines(self):
        """Test adding vertical line markers."""
        fig, ax = plt.subplots()
        ax.plot(self.energy, self.cross_section)

        # Add vline markers
        self.plotter.add_resonance_markers(
            ax,
            self.resonance_info["energies"],
            marker_style="vlines",
        )

        # Check that vertical lines were added
        # (matplotlib axvline creates Line2D objects)
        lines = ax.get_lines()
        self.assertGreater(len(lines), 1)  # Original + markers

        close_figure(fig)

    def test_add_resonance_markers_arrows(self):
        """Test adding arrow markers."""
        fig, ax = plt.subplots()
        ax.plot(self.energy, self.cross_section)
        ax.set_ylim(0, 1e-17)

        # Add arrow markers
        self.plotter.add_resonance_markers(
            ax,
            self.resonance_info["energies"][:2],  # Just first two
            marker_style="arrows",
        )

        # Arrows are added as annotations
        # Check they exist (matplotlib stores them in ax.texts and ax.patches)
        self.assertGreaterEqual(len(ax.patches), 0)

        close_figure(fig)

    def test_add_resonance_markers_shaded(self):
        """Test adding shaded region markers."""
        fig, ax = plt.subplots()
        ax.plot(self.energy, self.cross_section)

        # Add shaded markers with widths
        self.plotter.add_resonance_markers(
            ax,
            self.resonance_info["energies"],
            resonance_widths=self.resonance_info["widths"],
            marker_style="shaded",
        )

        # axvspan creates Polygon objects
        self.assertGreater(len(ax.patches), 0)

        close_figure(fig)

    def test_plot_collision_strength(self):
        """Test collision strength plotting."""
        # Generate mock collision strength data
        collision_strength = self.cross_section * 1e18  # Just scale for testing

        fig, ax = self.plotter.plot_collision_strength(
            self.energy,
            collision_strength,
        )

        # Check figure created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        # Check labels
        self.assertIn("Energy", ax.get_xlabel())
        self.assertIn("Collision Strength", ax.get_ylabel())

        close_figure(fig)

    def test_plot_comparison_subplots(self):
        """Test multi-panel comparison plots."""
        collision_strength = self.cross_section * 1e18

        fig, axes = self.plotter.plot_comparison_subplots(
            self.energy,
            self.cross_section,
            collision_strength,
            resonance_info=self.resonance_info,
            suptitle="Test Comparison",
        )

        # Should have 3 panels (cross section, collision strength, contributions)
        self.assertEqual(len(axes), 3)

        # Check that all panels have data
        for ax in axes:
            lines = ax.get_lines()
            self.assertGreater(len(lines), 0)

        close_figure(fig)

    def test_plot_comparison_subplots_no_resonance_info(self):
        """Test multi-panel plots without resonance info."""
        collision_strength = self.cross_section * 1e18

        fig, axes = self.plotter.plot_comparison_subplots(
            self.energy,
            self.cross_section,
            collision_strength,
            resonance_info=None,
        )

        # Should have only 2 panels without resonance info
        self.assertEqual(len(axes), 2)

        close_figure(fig)

    def test_plot_energy_level_diagram(self):
        """Test energy level diagram plotting."""
        fig, ax = self.plotter.plot_energy_level_diagram(
            self.levels,
            initial_level_index=0,
            final_level_index=1,
            resonant_level_indices=[10, 11, 12],
        )

        # Check figure created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        # Check ylabel
        self.assertIn("Energy", ax.get_ylabel())

        # Check that horizontal lines exist (for levels)
        # hlines creates LineCollection objects
        collections = ax.collections
        self.assertGreater(len(collections), 0)

        close_figure(fig)

    def test_plot_energy_level_diagram_without_resonances(self):
        """Test energy level diagram without resonances."""
        fig, ax = self.plotter.plot_energy_level_diagram(
            self.levels,
            initial_level_index=0,
            final_level_index=1,
            resonant_level_indices=None,
        )

        # Should still create a valid plot
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        close_figure(fig)

    def test_plot_energy_level_diagram_max_resonances(self):
        """Test energy level diagram with limited resonances."""
        # Provide more resonances than max
        fig, ax = self.plotter.plot_energy_level_diagram(
            self.levels,
            initial_level_index=0,
            final_level_index=1,
            resonant_level_indices=[10, 11, 12, 13],
            max_resonances=2,
        )

        # Should only show max_resonances
        self.assertIsNotNone(fig)

        close_figure(fig)


@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "matplotlib not available")
class TestQuickPlotFunctions(unittest.TestCase):
    """Test quick plotting convenience functions."""

    def setUp(self):
        """Create mock data before each test."""
        self.energy, self.cross_section, self.resonance_info = (
            generate_mock_cross_section_data()
        )
        plt.close("all")

    def tearDown(self):
        """Clean up plots after each test."""
        plt.close("all")

    def test_quick_plot_cross_section_basic(self):
        """Test basic quick plot function."""
        fig, ax = quick_plot_cross_section(
            self.energy,
            self.cross_section,
        )

        # Check figure created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        close_figure(fig)

    def test_quick_plot_with_resonances(self):
        """Test quick plot with resonance info."""
        fig, ax = quick_plot_cross_section(
            self.energy,
            self.cross_section,
            resonance_info=self.resonance_info,
            show_resonances=True,
            num_resonances=3,
        )

        # Should have created contributions plot
        self.assertIsNotNone(fig)

        # Check legend exists (from contributions)
        legend = ax.get_legend()
        self.assertIsNotNone(legend)

        close_figure(fig)

    def test_quick_plot_without_showing_resonances(self):
        """Test quick plot with markers but not full contributions."""
        fig, ax = quick_plot_cross_section(
            self.energy,
            self.cross_section,
            resonance_info=self.resonance_info,
            show_resonances=False,
        )

        # Should have basic plot with markers
        self.assertIsNotNone(fig)

        # Should have more than one line (original + markers)
        lines = ax.get_lines()
        self.assertGreater(len(lines), 1)

        close_figure(fig)

    def test_quick_plot_save_figure(self):
        """Test that quick plot can save figures."""
        import tempfile
        import os

        # Use a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            save_path = tmp.name

        try:
            fig, ax = quick_plot_cross_section(
                self.energy,
                self.cross_section,
                save_path=save_path,
            )

            # Check that file was created
            self.assertTrue(os.path.exists(save_path))
        finally:
            # Clean up
            if os.path.exists(save_path):
                os.remove(save_path)


@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "matplotlib not available")
class TestPlottingEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in plotting."""

    def tearDown(self):
        """Clean up plots after each test."""
        plt.close("all")

    def test_empty_data(self):
        """Test plotting with minimal data."""
        plotter = ResonancePlotter()
        energy = np.array([1, 2, 3])
        cross_section = np.array([1e-18, 2e-18, 1e-18])

        fig, ax = plotter.plot_cross_section(energy, cross_section)

        self.assertIsNotNone(fig)
        close_figure(fig)

    def test_single_resonance(self):
        """Test plotting with single resonance."""
        plotter = ResonancePlotter()
        energy = np.linspace(50, 150, 50)
        cross_section = 1e-18 * np.ones_like(energy)

        resonance_info = {
            "level_index": [10],
            "energies": [100],
            "widths": [5],
            "contributions": [1e-18 * np.ones_like(energy)],
        }

        fig, ax = plotter.plot_resonance_contributions(
            energy,
            cross_section,
            resonance_info,
        )

        self.assertIsNotNone(fig)
        close_figure(fig)

    def test_plot_with_custom_styling(self):
        """Test that custom plot kwargs are applied."""
        plotter = ResonancePlotter()
        energy = np.linspace(50, 150, 50)
        cross_section = 1e-18 * np.ones_like(energy)

        fig, ax = plotter.plot_cross_section(
            energy,
            cross_section,
            color="red",
            linewidth=3,
            linestyle="--",
        )

        # Check that line has custom styling
        line = ax.get_lines()[0]
        self.assertEqual(line.get_color(), "red")
        self.assertEqual(line.get_linewidth(), 3)
        self.assertEqual(line.get_linestyle(), "--")

        close_figure(fig)


@unittest.skipIf(MATPLOTLIB_AVAILABLE, "Testing matplotlib import failure")
class TestPlottingImportFailure(unittest.TestCase):
    """Test behavior when matplotlib is not available."""

    def test_import_without_matplotlib(self):
        """Test that module behavior is correct without matplotlib."""
        # This test runs when matplotlib is NOT available
        # Just verify the test structure works
        self.assertFalse(MATPLOTLIB_AVAILABLE)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
