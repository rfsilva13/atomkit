#!/usr/bin/env python3
"""Debug diagram vs satellite classification."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from atomkit.analysis import load_data, label_hole_states

base_filename = "/home/rfsilva/Programs/fac-analysis/input_files/Pd"
levels, transitions, auger = load_data(base_filename)

# Label hole states
levels_labeled = label_hole_states(levels, hole_shell="1s")

print("=== HOLE STATE ANALYSIS ===\n")

# Check hole states
hole_states = levels_labeled[levels_labeled['is_hole_state']]
print(f"Total K-shell (1s) hole states: {len(hole_states)}")
print(f"\nConfigurations:")
print(hole_states['configuration'].value_counts())

# Check which are single-hole (diagram) vs multi-hole (satellite)
print("\n\n=== DIAGRAM vs SATELLITE ===\n")

# Single 1s hole = diagram (e.g., "1s+1(1)1" alone)
# Multiple holes including 1s = satellite (e.g., "1s+1(1)1.4d+5(5)6")

single_hole = hole_states[~hole_states['configuration'].str.contains(r'\.', regex=True, na=False)]
multi_hole = hole_states[hole_states['configuration'].str.contains(r'\.', regex=True, na=False)]

print(f"Single-hole states (diagram): {len(single_hole)}")
print(f"  Example: {single_hole['configuration'].iloc[0] if len(single_hole) > 0 else 'None'}")

print(f"\nMulti-hole states (satellite): {len(multi_hole)}")
if len(multi_hole) > 0:
    print("  Examples:")
    for config in multi_hole['configuration'].head(5):
        print(f"    - {config}")

# Now check transitions
print("\n\n=== TRANSITION ANALYSIS ===\n")

# Merge with levels to get upper/lower configurations
trans_with_config = transitions.merge(
    levels_labeled[['level_index', 'configuration', 'is_hole_state']],
    left_on='upper_level',
    right_on='level_index',
    suffixes=('', '_upper')
).merge(
    levels_labeled[['level_index', 'configuration', 'is_hole_state']],
    left_on='lower_level',
    right_on='level_index',
    suffixes=('_upper', '_lower')
)

# Diagram: upper has single 1s hole, lower has no 1s hole
diagram_trans = trans_with_config[
    trans_with_config['is_hole_state_upper'] & 
    ~trans_with_config['configuration_upper'].str.contains(r'\.', regex=True, na=False)
]

# Satellite: upper has multiple holes including 1s
satellite_trans = trans_with_config[
    trans_with_config['is_hole_state_upper'] & 
    trans_with_config['configuration_upper'].str.contains(r'\.', regex=True, na=False)
]

print(f"Diagram transitions (from single 1s hole): {len(diagram_trans)}")
print(f"Satellite transitions (from multi-hole with 1s): {len(satellite_trans)}")

print("\n\nExample diagram upper configs:")
for config in diagram_trans['configuration_upper'].unique()[:3]:
    print(f"  - {config}")

print("\nExample satellite upper configs:")
for config in satellite_trans['configuration_upper'].unique()[:5]:
    print(f"  - {config}")
