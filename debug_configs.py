#!/usr/bin/env python3
"""Debug FAC configuration format."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from atomkit.readers import read_levels

base_filename = "/home/rfsilva/Programs/fac-analysis/input_files/Pd"
levels = read_levels(base_filename)

print("Sample configurations:")
print(levels[['level_index', 'configuration', 'ion_charge']].head(20).to_string(index=False))

print("\n\nUnique configurations:")
print(levels['configuration'].unique()[:10])
