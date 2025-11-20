"""
Compare Siegbahn labeling: notebook XRlabel vs robust backend fac_label_to_siegbahn
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

print("Sample comparison:")
print(comparison.head(10))

mismatches = comparison[~comparison['Match']]
if not mismatches.empty:
    print('\nMismatches found:')
    print(mismatches)
else:
    print('\nAll labels match between both methods!')

# Edge cases and noisy data
def test_edge_cases():
    edge_labels = [
        '',  # empty
        None,  # None
        '1s',  # minimal
        '4d+6(0)0',  # normal
        '2p-1(1)1.3d+5(5)6',  # multi-shell
        'badlabel',  # invalid
        '3d+5(5)6',  # missing dot
        '2p-1(1)1.',  # trailing dot
        '4d+6(0)0 extra',  # extra text
        '4d+6(0)0.2p-1(1)1',  # valid multi
    ]
    prev_edge = [fac_label_to_siegbahn(l) for l in edge_labels]
    notebook_edge = [XRlabel(atomic_number, conf1, l, l) for l in edge_labels]
    edge_df = pd.DataFrame({
        'FAC label': edge_labels,
        'Previous (fac_label_to_siegbahn)': prev_edge,
        'Notebook (XRlabel)': notebook_edge
    })
    edge_df['Match'] = edge_df['Previous (fac_label_to_siegbahn)'] == edge_df['Notebook (XRlabel)']
    print("\nEdge case comparison:")
    print(edge_df)
    print('\nSummary:')
    print(edge_df.groupby('Match').size())

test_edge_cases()
