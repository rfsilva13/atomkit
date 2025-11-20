import pandas as pd

from atomkit.readers.siegbahn import (XRlabel, config_to_siegbahn,
                                      shell_to_siegbahn)

# Example FAC-like configuration strings (single and multi-hole)
test_configs = [
    '1s1 2s2 2p-2 2p+2',  # K hole
    '2s1 2p-2 2p+2',      # L1 hole
    '2p-1 2s2 2p+2',      # L2 hole
    '2p+1 2s2 2p-2',      # L3 hole
    '2p-1 3d-1 2s2 2p+2 3d+2',  # L2 M4 double hole
    '2p+1 3d+1 2s2 2p-2 3d-2',  # L3 M5 double hole
    '3d-1 3d+1 2s2 2p-2 2p+2',  # M4 M5 double hole
    '2p-1 3d+1 2s2 2p+2 3d-2',  # L2 M5 double hole
    '2p+1 3d-1 2s2 2p-2 3d+2',  # L3 M4 double hole
]

atomic_number = 46  # Pd
conf1 = ''  # Not used in backend
label = ''  # Not used in backend

results = []
for config in test_configs:
    label_xr = XRlabel(atomic_number, conf1, config, label)
    label_backend = config_to_siegbahn(config, holes_only=True, compact=True)
    results.append({
        'config': config,
        'XRlabel': label_xr,
        'config_to_siegbahn': label_backend,
        'match': label_xr == label_backend
    })

df = pd.DataFrame(results)
print(df)

# Optionally, print mismatches only
mismatches = df[~df['match']]
if not mismatches.empty:
    print('\nMismatches:')
    print(mismatches)
else:
    print('\nAll labels match!')
