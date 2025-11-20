import pandas as pd

from atomkit.readers.siegbahn import (XRlabel, config_to_siegbahn,
                                      fac_label_to_siegbahn)

# Read Pd.lev.asc and extract the FAC label column (last column)
lev_path = "examples/inspiration_private/auger_test/Pd_shake/Pd.lev.asc"
labels = []
with open(lev_path) as f:
    for line in f:
        parts = line.strip().split()
        # FAC label is usually the last column, skip header lines
        if len(parts) > 0 and parts[-1].count('(') > 0:
            labels.append(parts[-1])

atomic_number = 46
conf1 = ''
label = ''

results = []
for fac_label in labels:
    # For this test, treat the FAC label as a configuration string
    xr = XRlabel(atomic_number, conf1, fac_label, label)
    backend = config_to_siegbahn(fac_label, holes_only=True, compact=True)
    fac_siegbahn = fac_label_to_siegbahn(fac_label)
    results.append({
        'fac_label': fac_label,
        'XRlabel': xr,
        'config_to_siegbahn': backend,
        'fac_label_to_siegbahn': fac_siegbahn,
        'XRlabel_vs_fac': xr == fac_siegbahn,
        'backend_vs_fac': backend == fac_siegbahn
    })

df = pd.DataFrame(results)
print(df)

mismatches = df[~df['match']]
if not mismatches.empty:
    mismatches = df[~df['XRlabel_vs_fac'] | ~df['backend_vs_fac']]
    print('\nMismatches:')
    print(mismatches)
else:
    print('\nAll labels match!')
    print('\nAll labels match the FAC parser!')
