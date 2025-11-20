"""
Compare Siegbahn labeling: notebook XRlabel vs robust backend fac_label_to_siegbahn for all labels in the FAC .lev.asc file.
Report any mismatches.
"""
import pandas as pd

from atomkit.readers.siegbahn import XRlabel, fac_label_to_siegbahn

# Load raw FAC labels from a sample .lev.asc file
lev_path = "examples/inspiration_private/auger_test/Pd_shake/Pd.lev.asc"
raw_labels = []
with open(lev_path) as f:
    for line in f:
        parts = line.strip().split()
        # FAC label is usually the last column, skip header lines
        if len(parts) > 0 and parts[-1].count('(') > 0:
            raw_labels.append(parts[-1])

# Apply both methods
atomic_number = 46  # Pd
conf1 = ''
prev_siegbahn = [fac_label_to_siegbahn(label) for label in raw_labels]
notebook_siegbahn = [XRlabel(atomic_number, conf1, label, label) for label in raw_labels]

# Compare results
comparison = pd.DataFrame({
    'FAC label': raw_labels,
    'Previous (fac_label_to_siegbahn)': prev_siegbahn,
    'Notebook (XRlabel)': notebook_siegbahn
})
comparison['Match'] = comparison['Previous (fac_label_to_siegbahn)'] == comparison['Notebook (XRlabel)']

mismatches = comparison[~comparison['Match']]

print(f"Total labels checked: {len(comparison)}")
print(f"Number of mismatches: {len(mismatches)}")

if not mismatches.empty:
    print('\nMismatches found:')
    print(mismatches)
else:
    print('\nAll labels match between both methods for all data in the lev.asc file!')
