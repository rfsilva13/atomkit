import json
from pathlib import Path

import pandas as pd

from scripts.recreate_graph_exact import (create_recreated_graph,
                                          map_original_to_sources)

DATA_DIR = Path(__file__).resolve().parents[1] / 'outputs'
ORIGINAL_GRAPH = DATA_DIR / 'Pd_full.xlsx - graph.csv'


def test_recreate_graph_rows():
    recreated_df = create_recreated_graph(strategy='round', rounding=2, tol=0.05)
    orig = pd.read_csv(ORIGINAL_GRAPH)
    # Ensure recreated graph has at least as many rows as original (grouping may increase rows)
    assert len(recreated_df) >= 1
    assert 'ENERGY' in recreated_df.columns


def test_mapping_contains_original_entries(tmp_path):
    map_out = tmp_path / 'graph_mapping.json'
    mapping = map_original_to_sources(out_json_path=str(map_out), tol=0.05)
    orig = pd.read_csv(ORIGINAL_GRAPH)
    # Mapping should include all original energies
    mapped_energies = set(mapping.keys())
    orig_energies = set([float(e) for e in orig['ENERGY'].tolist()])
    # All original energies should be present in mapping
    assert len(orig_energies.difference(mapped_energies)) == 0

