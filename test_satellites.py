#!/usr/bin/env python3
"""Test satellite count."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from atomkit.analysis import (
    load_data,
    calculate_satellite_intensities,
    label_hole_states,
    calculate_fluorescence_yield,
    get_shake_off_data,
)

base_filename = "/home/rfsilva/Programs/fac-analysis/input_files/Pd"
levels, transitions, auger = load_data(base_filename)

# Calculate creation rate
shake_off = get_shake_off_data()
w, rad_sum, auger_sum = calculate_fluorescence_yield(transitions, auger)
creation_rate = shake_off['Q1s'].sum() * rad_sum

# Calculate satellites
satellite = calculate_satellite_intensities(
    transitions, levels, auger, creation_rate, hole_shell="1s"
)

print(f"Satellite transitions: {len(satellite)}")
print(f"Total satellite intensity: {satellite['intensity'].sum():.6e}")
print(f"\nTop 5 most intense satellites:")
print(satellite.nlargest(5, 'intensity')[['energy', 'rate', 'intensity']])
