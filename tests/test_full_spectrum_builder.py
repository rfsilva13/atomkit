from pathlib import Path

import pandas as pd

from scripts.full_spectrum_builder import (build_spectrum, find_file,
                                           load_csv_guess)

DATA_DIR = Path(__file__).resolve().parents[1] / 'analysis_k_alpha'


def test_build_spectrum_runs():
    # Check that build_spectrum produces a DataFrame and metadata when given available files
    # More robust: try several candidate names found in analysis_k_alpha
    diag_path = find_file(DATA_DIR, ['Diagram_copy_rfsilva.csv', 'Pd_full.xlsx - Diagram_copy_rfsilva.csv', 'Pd_full*Diag*.csv'])
    sat_path = find_file(DATA_DIR, ['Satelites_copy_rfsilva.csv', 'Pd_full.xlsx - Satelites_copy_rfsilva.csv', 'Pd_full*Satellite*.csv'])
    aug_path = find_file(DATA_DIR, ['JM auger.csv', 'Pd_full.xlsx - JM auger.csv'])
    shake_path = find_file(DATA_DIR, ['K Shake (1).csv', 'Pd_full.xlsx - K Shake (1).csv', 'K Shake.csv'])

    assert diag_path is not None
    assert sat_path is not None
    assert aug_path is not None
    assert shake_path is not None

    diag = load_csv_guess(diag_path)
    sat = load_csv_guess(sat_path)
    aug = load_csv_guess(aug_path, header=1)
    shake = load_csv_guess(shake_path, header=2)

    df, meta = build_spectrum(diag, sat, aug, shake, bin_width=1.0, emin=20000, emax=25000)
    assert isinstance(df, pd.DataFrame)
    assert 'Energy_eV' in df.columns and 'Intensity' in df.columns
    assert 'wK_univ' in meta and meta['diag_count'] > 0

