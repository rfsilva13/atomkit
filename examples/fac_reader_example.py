# examples/fac_reader_example.py

"""
AtomKit FAC Reader: Practical Examples

This script demonstrates a continuous workflow using the atomkit.readers module:
1. Reading FAC energy level data (.lev.asc).
2. Reading FAC transition data (.tr.asc).
3. Adding level information (ion charge, labels, configurations) to the
   transitions data using `add_level_info_to_transitions`.
4. Demonstrating customization options like unit conversion, column selection,
   and renaming for both levels and the enriched transitions.
5. Showing how to work with the DataFrame index (MultiIndex or RangeIndex).

**Important:** For the transition labeling step (Step 3) to work effectively,
ensure the level and transition files correspond to the same element and
contain overlapping ion stages. This example uses "Pd" for both, assuming
Pd.lev.asc and Pd.tr.asc exist in the test_files directory.

To run this example, ensure you have the 'atomkit' package installed
(e.g., via 'poetry install' or 'pip install .') from the project root,
and that the 'test_files' directory with 'Pd.lev.asc' and 'Pd.tr.asc'
is present in the project root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- AtomKit Import ---
# If atomkit is installed (e.g., with poetry install), direct imports should work.
try:
    from atomkit.definitions import EV_TO_CM1  # Import the constant
    from atomkit.readers import (
        add_level_info_to_transitions,
        read_fac,
        read_fac_transitions,
    )
except ImportError as e:
    print(f"\nError importing from atomkit: {e}")
    print("Please ensure 'atomkit' is installed in your Python environment.")
    print("If running from the project root, try commands like:")
    print("  'poetry install' (if using Poetry)")
    print("  'pip install -e .' (for an editable install)")
    print(
        "Then run: 'python examples/fac_reader_example.py' or 'poetry run python examples/fac_reader_example.py'"
    )
    sys.exit(1)

# --- Configuration: File Paths ---
# Determine project root assuming this script is in atomkit/examples/
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback if __file__ is not defined (e.g., in some interactive environments)
    project_root = Path(".").resolve()


# --- !! IMPORTANT !! ---
# Set these to the base names of your corresponding level and transition files.
LEVEL_FILE_BASE_NAME = "Pd"
TRANSITION_FILE_BASE_NAME = "Pd"
# --- !! IMPORTANT !! ---

test_files_dir = project_root / "test_files"
level_file_base_path = test_files_dir / LEVEL_FILE_BASE_NAME
transition_file_base_path = test_files_dir / TRANSITION_FILE_BASE_NAME

level_file_full_path = level_file_base_path.with_suffix(".lev.asc")
transition_file_full_path = transition_file_base_path.with_suffix(".tr.asc")


# --- Pandas Display Settings ---
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.precision", 5)


# --- Helper Function to Print DataFrame Info ---
def print_df_info(df: pd.DataFrame, title: str, head_rows: int = 4):
    print(f"\n--- {title} ---")
    if df is None or df.empty:
        print(" -> DataFrame is empty or None.")
        return
    print(f"Shape: {df.shape}")
    with pd.option_context("display.max_columns", None):
        print(f"Columns: {df.columns.tolist()}")
    if df.index.name or (isinstance(df.index, pd.MultiIndex) and any(df.index.names)):
        print(f"Index Names: {df.index.names}")
        if isinstance(df.index, pd.MultiIndex):
            print(f"Index Levels: {df.index.nlevels}")
    else:
        print("Index: Default RangeIndex")
    print("Sample Data:")
    with pd.option_context("display.max_colwidth", 35, "display.width", 250):
        print(df.head(head_rows).replace({np.nan: "-NA-", "<N/A>": "<N/A>"}))
    print("-" * (len(title) + 6))


# =============================================================================
# --- Main Workflow ---
# =============================================================================
print("\n" + "=" * 70)
print("--- AtomKit Reader Workflow Example ---")
print(f"--- Using Base Name: '{TRANSITION_FILE_BASE_NAME}' ---")
print("=" * 70)

df_levels = pd.DataFrame()
df_transitions = pd.DataFrame()
df_levels_flat = pd.DataFrame()
df_transitions_labeled = pd.DataFrame()

print("\n>>> Step 1: Reading Level Data")
if not level_file_full_path.exists():
    print(f"ERROR: Level file not found: '{level_file_full_path}'. Cannot proceed.")
    print(
        f"Please ensure '{LEVEL_FILE_BASE_NAME}.lev.asc' exists in '{test_files_dir}'"
    )
    sys.exit(1)

df_levels = read_fac(str(level_file_base_path))
print_df_info(df_levels, "1a. Initial Level Data (Defaults)")

if not df_levels.empty:
    df_levels_flat = (
        df_levels.reset_index()
        if isinstance(df_levels.index, pd.MultiIndex)
        else df_levels.copy()
    )
    essential_level_cols = [
        "atomic_number",
        "ion_charge",
        "level_index",
        "label",
        "configuration",
    ]
    if not all(col in df_levels_flat.columns for col in essential_level_cols):
        print(
            f"WARNING: Level data missing essential columns needed for labeling: {essential_level_cols}"
        )
        df_levels_flat = pd.DataFrame()
    print_df_info(df_levels_flat, "1b. Levels with Index as Columns (Flattened)")

print("\n>>> Step 2: Reading Transition Data")
if not transition_file_full_path.exists():
    print(
        f"ERROR: Transition file not found: '{transition_file_full_path}'. Cannot proceed."
    )
    print(
        f"Please ensure '{TRANSITION_FILE_BASE_NAME}.tr.asc' exists in '{test_files_dir}'"
    )
    sys.exit(1)

df_transitions = read_fac_transitions(str(transition_file_base_path))
print_df_info(
    df_transitions, "2. Initial Transition Data (Defaults, No ion_charge yet)"
)

print("\n>>> Step 3: Adding Level Info to Transitions")
if not df_transitions.empty and not df_levels_flat.empty:
    print("   Adding 'ion_charge', 'label', and 'configuration' from levels data...")
    df_transitions_labeled = add_level_info_to_transitions(
        df_transitions,
        df_levels_flat,
        cols_to_add=[
            "ion_charge",
            "label",
            "configuration",
        ],
        missing_value="<Level N/A>",
    )
    cols_to_show = [
        "atomic_number",
        "ion_charge",
        "level_index_lower",
        "level_index_upper",
        "energy",
        "lambda",
        "A",
        "type",
        "label_lower",
        "configuration_lower",
        "label_upper",
        "configuration_upper",
    ]
    cols_to_show = [
        col for col in cols_to_show if col in df_transitions_labeled.columns
    ]
    print_df_info(
        df_transitions_labeled[cols_to_show], "3. Transitions with Added Level Info"
    )
else:
    print(" -> Skipping adding level info as Level or Transition DataFrame is empty.")

print("\n>>> Step 4: Customizing Labeled Transitions")
if not df_transitions_labeled.empty:
    trans_cols_to_keep = [
        "energy",
        "lambda",
        "A",
        "type",
        "label_lower",
        "label_upper",
        "ion_charge",  # Keep ion_charge
    ]
    trans_rename_map = {
        "energy": "E_trans(cm-1)",
        "lambda": "Wavelength(nm)",
        "A": "A-coeff",
        "type": "Multipole",
        "label_lower": "Lower",
        "label_upper": "Upper",
        # "ion_charge": "Q", # Removed renaming ion_charge to Q
    }
    df_trans_custom = df_transitions_labeled.copy()
    if "energy" in df_trans_custom.columns and "EV_TO_CM1" in globals():
        df_trans_custom["energy"] = pd.to_numeric(
            df_trans_custom["energy"], errors="coerce"
        )
        df_trans_custom["energy"] = df_trans_custom["energy"] * EV_TO_CM1
    elif "energy" in df_trans_custom.columns:
        print(
            "Warning: EV_TO_CM1 not defined or 'energy' column not numeric, cannot convert energy units for transitions."
        )
    if "lambda" in df_trans_custom.columns:
        df_trans_custom["lambda"] = pd.to_numeric(
            df_trans_custom["lambda"], errors="coerce"
        )
        df_trans_custom["lambda"] = df_trans_custom["lambda"] / 10.0
    df_trans_custom.rename(columns=trans_rename_map, inplace=True)

    # Ensure 'ion_charge' is selected if it exists, using its original or renamed name
    # (though we removed its renaming)
    final_cols = []
    for c in trans_cols_to_keep:
        final_name = trans_rename_map.get(
            c, c
        )  # Get final name if renamed, else original
        if final_name in df_trans_custom.columns:
            final_cols.append(final_name)
        elif (
            c in df_trans_custom.columns
        ):  # Check original name if final_name wasn't found (e.g. not in rename_map)
            final_cols.append(c)

    if "atomic_number" in df_trans_custom.columns and "atomic_number" not in final_cols:
        final_cols.insert(0, "atomic_number")

    # Ensure ion_charge is included if it exists and wasn't explicitly in trans_cols_to_keep with a new name
    ion_charge_final_name = trans_rename_map.get("ion_charge", "ion_charge")
    if (
        ion_charge_final_name in df_trans_custom.columns
        and ion_charge_final_name not in final_cols
    ):
        # Try to insert it after atomic_number if possible, or append
        try:
            idx = final_cols.index("atomic_number") + 1
            final_cols.insert(idx, ion_charge_final_name)
        except ValueError:
            final_cols.append(ion_charge_final_name)

    df_trans_custom = df_trans_custom[
        [col for col in final_cols if col in df_trans_custom.columns]
    ]
    print_df_info(
        df_trans_custom,
        "4. Customized Labeled Transitions (cm-1, nm, Selected, Renamed)",
    )

    # Example Analysis: Show strongest E1 transitions for a specific ion stage
    charge_col = "ion_charge"  # Use 'ion_charge' directly
    type_col = trans_rename_map.get("type", "type")  # Use final name for type
    a_coeff_col = trans_rename_map.get("A", "A")  # Use final name for A

    if charge_col in df_trans_custom.columns:
        df_trans_custom[charge_col] = pd.to_numeric(
            df_trans_custom[charge_col], errors="coerce"
        )
        available_ion_stages = sorted(df_trans_custom[charge_col].dropna().unique())

        target_ion_stage = None
        if available_ion_stages:
            target_ion_stage = available_ion_stages[
                0
            ]  # Select the first available ion stage
            print(
                f"\n   Example Analysis: Strongest E1 Transitions for Ion Stage {target_ion_stage}"
            )
        else:
            print(f"\n   Example Analysis: No ion stages found in the data to analyze.")

        if (
            target_ion_stage is not None
            and type_col in df_trans_custom.columns
            and a_coeff_col in df_trans_custom.columns
        ):

            strongest_e1 = df_trans_custom[
                (df_trans_custom[charge_col] == target_ion_stage)
                & (df_trans_custom[type_col] == "E1")
            ].sort_values(a_coeff_col, ascending=False)

            if strongest_e1.empty:
                print(
                    f"     -> No E1 transitions found for ion stage {target_ion_stage}."
                )
            else:
                cols_to_display_analysis = [
                    charge_col,
                    trans_rename_map.get("lambda", "lambda"),  # Wavelength(nm)
                    a_coeff_col,  # A-coeff
                    trans_rename_map.get("label_lower", "label_lower"),  # Lower
                    trans_rename_map.get("label_upper", "label_upper"),  # Upper
                ]
                cols_to_display_analysis = [
                    c for c in cols_to_display_analysis if c in strongest_e1.columns
                ]
                print_df_info(
                    strongest_e1[cols_to_display_analysis],
                    f"Strongest E1 Transitions (Ion Stage {target_ion_stage})",
                    head_rows=5,
                )
        elif target_ion_stage is not None:
            print(
                f"   Skipping E1 analysis for ion stage {target_ion_stage}: Missing one of required columns: '{type_col}', '{a_coeff_col}'"
            )
    else:
        print(
            f"   Skipping E1 analysis: Column '{charge_col}' not found in the customized DataFrame."
        )
else:
    print(" -> Skipping customization as labeled transition data is empty.")

print("\n" + "=" * 70)
print("--- AtomKit Reader Workflow Example Finished ---")
print("=" * 70)
