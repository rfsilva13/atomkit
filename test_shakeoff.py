#!/usr/bin/env python3
"""Test shake-off corrections in universal analysis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from atomkit.analysis import (
    load_data,
    label_hole_states,
    calculate_diagram_intensities,
    calculate_satellite_intensities,
    get_shake_off_data,
)

base_filename = "/home/rfsilva/Programs/fac-analysis/input_files/Pd"
levels, transitions, auger = load_data(base_filename)

# Get shake-off data
shake_off = get_shake_off_data()
total_shake_off = shake_off['Q1s'].sum()

print(f"Total shake-off probability: {total_shake_off:.6f}\n")

# Label hole states
levels_labeled = label_hole_states(levels, hole_shell="1s")

# Calculate diagram WITH shake-off correction
diagram = calculate_diagram_intensities(
    transitions,
    levels_labeled,
    auger,
    hole_shell="1s",
    shake_off_probability=total_shake_off
)

print(f"Diagram lines: {len(diagram)}")
print(f"  Total intensity (raw): {diagram['intensity'].sum():.6e}")
print(f"  Total intensity (final, with shake-off): {diagram['intensity_final'].sum():.6e}")
print(f"  Reduction factor: {diagram['intensity_final'].sum() / diagram['intensity'].sum():.6f}")
print(f"  Expected: {1 - total_shake_off:.6f}\n")

# Calculate satellites WITH shell-specific shake-off
satellite = calculate_satellite_intensities(
    transitions,
    levels_labeled,
    auger,
    shake_off_data=shake_off,
    hole_shell="1s"
)

print(f"Satellite lines: {len(satellite)}")
if len(satellite) > 0:
    print(f"  Total intensity (raw): {satellite['intensity'].sum():.6e}")
    print(f"  Total intensity (final, weighted by Q): {satellite['intensity_final'].sum():.6e}")
    print(f"\nTop 5 satellites:")
    top5 = satellite.nlargest(5, 'intensity_final')
    print(top5[['energy', 'spectator_shell', 'Q1s', 'intensity', 'intensity_final']])
