#!/usr/bin/env python3
"""Check K-shell hole configuration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from atomkit.readers import read_levels

base_filename = "/home/rfsilva/Programs/fac-analysis/input_files/Pd"
levels = read_levels(base_filename)

# Find K-shell hole (1s)
print("Looking for K-shell (1s) holes:")
k_holes = levels[levels['configuration'].str.contains('1s', na=False)]
print(f"\nFound {len(k_holes)} levels with '1s' in configuration:")
print(k_holes[['level_index', 'configuration', 'ion_charge', 'energy']].to_string(index=False))

# Check what the universal reader put in the levels
print("\n\nAll column names in levels DataFrame:")
print(levels.columns.tolist())
