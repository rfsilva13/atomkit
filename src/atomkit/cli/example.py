#!/usr/bin/env python3
"""
Example script showing how to use the interactive plotter programmatically.

This demonstrates how to call the interactive plotter functions directly
from Python code instead of using the command-line interface.
"""

import sys
from pathlib import Path

# Add atomkit to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from atomkit.analysis import (apply_linewidths_to_transitions,
                              calculate_diagram_intensities,
                              calculate_fluorescence_yield,
                              calculate_satellite_intensities,
                              calculate_widths,
                              create_interactive_energy_shifter,
                              create_lorentzian_spectrum, get_shake_off_data,
                              label_hole_states, load_data)


def create_interactive_plot(input_file: str,
                          shell: str = "1s",
                          include_satellites: bool = True,
                          include_shakeoff: bool = True,
                          include_lorentzian: bool = True,
                          energy_min: float = 20500,
                          energy_max: float = 21800,
                          energy_grid_points: int = 5000):
    """
    Create an interactive energy shifter plot programmatically.

    Parameters
    ----------
    input_file : str
        Base filename for input data
    shell : str
        Shell to analyze
    include_satellites : bool
        Whether to include satellite lines
    include_shakeoff : bool
        Whether to apply shake-off corrections
    include_lorentzian : bool
        Whether to create Lorentzian-broadened spectrum
    energy_min, energy_max : float
        Energy range for spectrum
    energy_grid_points : int
        Number of points in energy grid
    """

    print(f"Loading data from: {input_file}")

    # Load data
    levels, transitions, auger = load_data(input_file)
    print(f"Loaded {len(levels)} levels, {len(transitions)} transitions, {len(auger)} auger transitions")

    # Calculate fluorescence yield
    w, radiative_sum, auger_sum = calculate_fluorescence_yield(transitions, auger)
    print(f"Fluorescence yield: {w:.6f}")

    # Label hole states
    levels_labeled = label_hole_states(levels, hole_shell=shell)
    hole_count = levels_labeled['is_hole_state'].sum()
    print(f"Found {hole_count} levels with {shell} holes")

    # Calculate natural linewidths
    if include_lorentzian:
        print("Calculating natural linewidths...")
        levels_with_widths = calculate_widths(levels.copy(), transitions, auger)
        print(f"Average linewidth: {levels_with_widths['width_total'].mean():.2e} eV")

    # Get shake-off data
    shake_off_probability = 0.0
    if include_shakeoff:
        shake_off = get_shake_off_data()
        shake_off_probability = shake_off[f'Q{shell}'].sum()
        print(f"Total shake-off probability: {shake_off_probability:.6f}")

    # Calculate diagram intensities
    print("Calculating diagram line intensities...")
    diagram = calculate_diagram_intensities(
        transitions,
        levels_labeled,
        auger,
        hole_shell=shell,
        shake_off_probability=shake_off_probability if include_shakeoff else 0.0
    )
    print(f"Found {len(diagram)} diagram transitions")

    # Calculate satellite intensities
    satellite = None
    if include_satellites:
        print("Calculating satellite line intensities...")

        # Prepare shake-off data
        shake_off_for_satellites = shake_off if include_shakeoff else pd.DataFrame({
            'shell': ['1s', '2s', '2p', '3s', '3p', '3d', '4s', '4p', '4d', '4f', '5s', '5p', '5d'],
            'Q1s': [1.0] * 13  # No shake-off correction when disabled
        })

        satellite = calculate_satellite_intensities(
            transitions,
            levels_labeled,
            auger,
            shake_off_data=shake_off_for_satellites,
            hole_shell=shell
        )
        print(f"Found {len(satellite)} satellite transitions")

    # Apply linewidths
    diagram_with_widths = diagram.copy()
    satellite_with_widths = None

    if include_lorentzian:
        print("Applying linewidths to transitions...")
        diagram_with_widths = apply_linewidths_to_transitions(diagram.copy(), levels_with_widths)

        if satellite is not None and len(satellite) > 0:
            satellite_with_widths = apply_linewidths_to_transitions(satellite.copy(), levels_with_widths)

    # Create energy grid and Lorentzian spectrum
    energy_grid = np.linspace(energy_min, energy_max, energy_grid_points)
    lorentzian_spectrum = None

    if include_lorentzian:
        print("Generating Lorentzian-broadened spectrum...")
        lorentzian_spectrum = np.zeros_like(energy_grid)

        # Add diagram lines
        if len(diagram_with_widths) > 0:
            lorentzian_spectrum += create_lorentzian_spectrum(diagram_with_widths, energy_grid)

        # Add satellite lines
        if satellite_with_widths is not None and len(satellite_with_widths) > 0:
            lorentzian_spectrum += create_lorentzian_spectrum(satellite_with_widths, energy_grid)

        print(f"Generated spectrum with {len(energy_grid)} energy points")

    # Create interactive plotter
    print("Creating interactive plotter...")

    plotter = create_interactive_energy_shifter(
        diagram_lines=diagram_with_widths,
        satellite_lines=satellite_with_widths,
        lorentzian_spectrum=lorentzian_spectrum,
        energy_grid=energy_grid,
        shell=shell,
        energy_min=energy_min,
        energy_max=energy_max
    )

    return plotter


def main():
    """Example usage of the interactive plotter."""

    # Example 1: Basic K-shell analysis
    print("="*60)
    print("Example 1: Basic K-shell analysis")
    print("="*60)

    plotter1 = create_interactive_plot(
        input_file="input_files/Pd",
        shell="1s",
        include_satellites=True,
        include_shakeoff=True,
        include_lorentzian=True
    )

    print("\nInteractive plot created! Use the sliders to adjust energy shifts.")
    print("Displaying plot...")
    from IPython.display import display
    display(plotter1)

    # Example 2: L-shell without satellites (faster)
    print("\n" + "="*60)
    print("Example 2: L-shell analysis without satellites")
    print("="*60)

    plotter2 = create_interactive_plot(
        input_file="input_files/Pd",
        shell="2s",
        include_satellites=False,  # Faster without satellites
        include_shakeoff=True,
        include_lorentzian=True,
        energy_min=18000,
        energy_max=20000
    )

    print("\nSecond interactive plot created!")
    display(plotter2)


if __name__ == "__main__":
    main()