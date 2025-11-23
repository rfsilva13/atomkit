#!/usr/bin/env python3
"""
Command-line interface for interactive energy shifting of spectral lines.

This tool provides an interactive Plotly-based visualization for adjusting
energy shifts of diagram and satellite lines in atomic spectra.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import display
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

# Import atomkit functions
from atomkit.analysis import (apply_linewidths_to_transitions,
                              calculate_diagram_intensities,
                              calculate_fluorescence_yield,
                              calculate_satellite_intensities,
                              calculate_widths,
                              create_interactive_energy_shifter,
                              create_lorentzian_spectrum, get_shake_off_data,
                              label_hole_states, load_data)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive energy shifting for atomic spectral lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input_files/Pd --shell 1s
  %(prog)s input_files/Pd --shell 2s --no-satellites
  %(prog)s input_files/Pd --shell 1s --energy-min 20000 --energy-max 22000
  %(prog)s input_files/Pd --shell 1s --no-shakeoff --no-lorentzian
        """
    )

    parser.add_argument(
        "input_file",
        help="Base filename for input data (without extensions)"
    )

    parser.add_argument(
        "--shell",
        default="1s",
        help="Shell to analyze (default: 1s)"
    )

    parser.add_argument(
        "--no-satellites",
        action="store_true",
        help="Exclude satellite lines from analysis"
    )

    parser.add_argument(
        "--no-shakeoff",
        action="store_true",
        help="Disable shake-off corrections for diagram lines"
    )

    parser.add_argument(
        "--no-lorentzian",
        action="store_true",
        help="Disable Lorentzian broadening (show delta functions only)"
    )

    parser.add_argument(
        "--energy-min",
        type=float,
        default=20500,
        help="Minimum energy for spectrum (default: 20500)"
    )

    parser.add_argument(
        "--energy-max",
        type=float,
        default=21800,
        help="Maximum energy for spectrum (default: 21800)"
    )

    parser.add_argument(
        "--energy-grid-points",
        type=int,
        default=5000,
        help="Number of points in energy grid for Lorentzian spectrum (default: 5000)"
    )

    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.5,
        help="Bin width for binned spectrum in eV (default: 0.5)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save output files (optional)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def load_and_process_data(args):
    """Load and process atomic data according to command line options."""

    if args.verbose:
        print(f"Loading data from: {args.input_file}")

    # Load data
    levels, transitions, auger = load_data(args.input_file)

    if args.verbose:
        print(f"Loaded {len(levels)} levels, {len(transitions)} transitions, {len(auger)} auger transitions")

    # Calculate fluorescence yield
    w, radiative_sum, auger_sum = calculate_fluorescence_yield(transitions, auger)
    if args.verbose:
        print(f"Fluorescence yield: {w:.6f}")

    # Label hole states
    levels_labeled = label_hole_states(levels, hole_shell=args.shell)
    hole_count = levels_labeled['is_hole_state'].sum()
    if args.verbose:
        print(f"Found {hole_count} levels with {args.shell} holes")

    # Calculate natural linewidths
    if not args.no_lorentzian:
        if args.verbose:
            print("Calculating natural linewidths...")
        levels_with_widths = calculate_widths(levels.copy(), transitions, auger)
        if args.verbose:
            print(f"Average linewidth: {levels_with_widths['width_total'].mean():.2e} eV")

    # Get shake-off data
    shake_off_probability = 0.0
    if not args.no_shakeoff:
        shake_off = get_shake_off_data()
        shake_off_probability = shake_off[f'Q{args.shell}'].sum()
        if args.verbose:
            print(f"Total shake-off probability: {shake_off_probability:.6f}")

    # Calculate diagram intensities
    if args.verbose:
        print("Calculating diagram line intensities...")
    diagram = calculate_diagram_intensities(
        transitions,
        levels_labeled,
        auger,
        hole_shell=args.shell,
        shake_off_probability=shake_off_probability if not args.no_shakeoff else 0.0
    )
    if args.verbose:
        print(f"Found {len(diagram)} diagram transitions")

    # Calculate satellite intensities
    satellite = None
    if not args.no_satellites:
        if args.verbose:
            print("Calculating satellite line intensities...")

        # Prepare shake-off data (use default if disabled)
        shake_off_for_satellites = shake_off if not args.no_shakeoff else pd.DataFrame({
            'shell': ['1s', '2s', '2p', '3s', '3p', '3d', '4s', '4p', '4d', '4f', '5s', '5p', '5d'],
            'Q1s': [1.0] * 13  # No shake-off correction when disabled
        })

        satellite = calculate_satellite_intensities(
            transitions,
            levels_labeled,
            auger,
            shake_off_data=shake_off_for_satellites,
            hole_shell=args.shell
        )
        if args.verbose:
            print(f"Found {len(satellite)} satellite transitions")

    # Apply linewidths
    diagram_with_widths = diagram.copy()
    satellite_with_widths = None

    if not args.no_lorentzian:
        if args.verbose:
            print("Applying linewidths to transitions...")
        diagram_with_widths = apply_linewidths_to_transitions(diagram.copy(), levels_with_widths)

        if satellite is not None and len(satellite) > 0:
            satellite_with_widths = apply_linewidths_to_transitions(satellite.copy(), levels_with_widths)

    # Create energy grid and Lorentzian spectrum
    energy_grid = np.linspace(args.energy_min, args.energy_max, args.energy_grid_points)
    lorentzian_spectrum = None

    if not args.no_lorentzian:
        if args.verbose:
            print("Generating Lorentzian-broadened spectrum...")
        lorentzian_spectrum = np.zeros_like(energy_grid)

        # Add diagram lines
        if len(diagram_with_widths) > 0:
            lorentzian_spectrum += create_lorentzian_spectrum(diagram_with_widths, energy_grid)

        # Add satellite lines
        if satellite_with_widths is not None and len(satellite_with_widths) > 0:
            lorentzian_spectrum += create_lorentzian_spectrum(satellite_with_widths, energy_grid)

        if args.verbose:
            print(f"Generated spectrum with {len(energy_grid)} energy points")

    return {
        'diagram_lines': diagram_with_widths,
        'satellite_lines': satellite_with_widths,
        'lorentzian_spectrum': lorentzian_spectrum,
        'energy_grid': energy_grid,
        'shell': args.shell,
        'energy_min': args.energy_min,
        'energy_max': args.energy_max,
        'diagram': diagram,  # Original diagram for saving
        'satellite': satellite,  # Original satellite for saving
    }


def save_outputs(data, output_dir):
    """Save analysis results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save diagram lines
    data['diagram'].to_csv(output_path / 'diagram_lines.csv', index=False)
    print(f"Saved diagram lines to {output_path / 'diagram_lines.csv'}")

    # Save satellite lines
    if data['satellite'] is not None and len(data['satellite']) > 0:
        data['satellite'].to_csv(output_path / 'satellite_lines.csv', index=False)
        print(f"Saved satellite lines to {output_path / 'satellite_lines.csv'}")

    # Save Lorentzian spectrum
    if data['lorentzian_spectrum'] is not None:
        lorentzian_df = pd.DataFrame({
            'energy': data['energy_grid'],
            'intensity': data['lorentzian_spectrum']
        })
        lorentzian_df.to_csv(output_path / 'spectrum_lorentzian.csv', index=False)
        print(f"Saved Lorentzian spectrum to {output_path / 'spectrum_lorentzian.csv'}")


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Check if ipywidgets is available
    if not IPYWIDGETS_AVAILABLE:
        print("Error: ipywidgets is required for interactive plotting.")
        print("Install with: pip install ipywidgets plotly")
        sys.exit(1)

    try:
        # Load and process data
        data = load_and_process_data(args)

        # Save outputs if requested
        if args.output_dir:
            save_outputs(data, args.output_dir)

        # Create interactive plotter
        if args.verbose:
            print("Creating interactive plotter...")

        plotter = create_interactive_energy_shifter(
            diagram_lines=data['diagram_lines'],
            satellite_lines=data['satellite_lines'],
            lorentzian_spectrum=data['lorentzian_spectrum'],
            energy_grid=data['energy_grid'],
            shell=data['shell'],
            energy_min=data['energy_min'],
            energy_max=data['energy_max']
        )

        # Display the plotter
        print("\n" + "="*60)
        print("Interactive Energy Shifter")
        print("="*60)
        print("Use the sliders below to adjust energy shifts:")
        print("- Diagram Shift: Adjusts all diagram line energies")
        print("- Satellite Shift: Adjusts all satellite line energies")
        print("- X Min/Max: Control the energy range displayed")
        print("- Reset button: Return all shifts to zero")
        print("="*60 + "\n")

        display(plotter)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()