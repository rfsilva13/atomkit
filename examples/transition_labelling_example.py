# examples/transition_labeling_example.py

"""
AtomKit Example: Adding Level Information to Transitions

This script demonstrates how to:
1. Read energy level data using `read_fac`.
2. Read transition data using `read_fac_transitions`.
3. Check the ion stages present in both files.
4. Use the `add_level_info_to_transitions` function to merge level details
   (like configuration, label, or energy) directly into the transitions DataFrame,
   creating separate columns for lower and upper levels. It highlights that
   labeling requires corresponding data in both files.

**Important:** For meaningful labels, the level file (.lev.asc) and
transition file (.tr.asc) should correspond to the same element and
contain overlapping ion stages.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- AtomKit Import ---
try:
    # Import the necessary functions
    from atomkit.readers import (
        add_level_info_to_transitions,
        read_fac,
        read_fac_transitions,
    )
except ImportError as e:
    print(f"\nError importing reader functions from atomkit: {e}")
    print("Please ensure atomkit is installed and accessible in your PYTHONPATH.")
    print("If running from the project root, try: poetry install")
    sys.exit(1)

# --- Configuration: File Paths ---
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path(".").resolve()

# --- !! IMPORTANT !! ---
# Set these to the base names of your corresponding level and transition files.
# Using "Pd" for both assumes you have Pd.lev.asc and Pd.tr.asc.
LEVEL_FILE_BASE_NAME = "Pd"
TRANSITION_FILE_BASE_NAME = "Pd"
# --- !! IMPORTANT !! ---

level_file_base_path = project_root / "test_files" / LEVEL_FILE_BASE_NAME
transition_file_base_path = project_root / "test_files" / TRANSITION_FILE_BASE_NAME

level_file_full_path = level_file_base_path.with_suffix(".lev.asc")
transition_file_full_path = transition_file_base_path.with_suffix(".tr.asc")

# --- Pandas Display Settings ---
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.precision", 5)


# --- Helper Function to Print DataFrame Info ---
def print_df_info(df: pd.DataFrame, title: str, head_rows: int = 4):
    print(f"\n--- {title} ---")
    if df is None or df.empty:
        print(" -> DataFrame is empty or None.")
        return
    print(f"Shape: {df.shape}")
    # Ensure columns are displayed even if many exist
    with pd.option_context("display.max_columns", None):
        print(f"Columns: {df.columns.tolist()}")
    if df.index.name or (isinstance(df.index, pd.MultiIndex) and any(df.index.names)):
        print(f"Index Names: {df.index.names}")
        if isinstance(df.index, pd.MultiIndex):
            print(f"Index Levels: {df.index.nlevels}")
    else:
        print("Index: Default RangeIndex")
    print("Sample Data:")
    # Display with NaN representation changed for clarity if needed
    with pd.option_context(
        "display.max_colwidth", 40, "display.width", 250
    ):  # Limit column width
        print(
            df.head(head_rows).replace({np.nan: "-NA-", "<N/A>": "<N/A>"})
        )  # Keep <N/A> visible
    print("-" * (len(title) + 6))


# =============================================================================
# --- Main Workflow ---
# =============================================================================
print("\n" + "=" * 70)
print("--- Transition Labeling Workflow ---")
print(f"--- Using Levels: '{LEVEL_FILE_BASE_NAME}.lev.asc' ---")
print(f"--- Using Transitions: '{TRANSITION_FILE_BASE_NAME}.tr.asc' ---")
print("=" * 70)

# --- Step 1: Read Level Data ---
print("\n>>> Step 1: Reading Level Data")
df_levels = pd.DataFrame()  # Initialize
if not level_file_full_path.exists():
    print(
        f"ERROR: Level file not found at '{level_file_full_path}'. Cannot proceed with labeling."
    )
else:
    # Read levels, keeping columns needed for labeling + some context
    df_levels = read_fac(
        str(level_file_base_path),
        # Explicitly keep columns needed for add_level_info_to_transitions
        columns_to_keep=[
            "level_index",
            "label",
            "configuration",
            "energy",
            "p",
            "2j",
            "atomic_number",
            "ion_charge",
        ],
    )
    print_df_info(df_levels, "1. Loaded Level Data")

# --- Step 2: Read Transition Data ---
print("\n>>> Step 2: Reading Transition Data")
df_transitions = pd.DataFrame()  # Initialize
if not transition_file_full_path.exists():
    print(
        f"ERROR: Transition file not found at '{transition_file_full_path}'. Cannot proceed."
    )
else:
    # Read transitions with default columns. Note: ion_charge is NOT read from header anymore.
    df_transitions = read_fac_transitions(str(transition_file_base_path))
    print_df_info(df_transitions, "2. Loaded Transition Data (No ion_charge yet)")

    # Check if essential columns are present after read
    required_trans_cols = ["atomic_number", "level_index_lower", "level_index_upper"]
    if not all(col in df_transitions.columns for col in required_trans_cols):
        print(
            f"ERROR: Transition DataFrame missing required columns for labeling: {required_trans_cols}. Cannot proceed."
        )
        df_transitions = pd.DataFrame()  # Mark as empty

# --- Step 3: Check Ion Stage Overlap (Informational) ---
print("\n>>> Step 3: Checking Ion Stage Overlap (Informational)")
if not df_levels.empty and not df_transitions.empty:
    try:
        # Ensure 'ion_charge' is accessible as a column for unique() check
        levels_ions = (
            df_levels.reset_index()["ion_charge"].unique()
            if "ion_charge" in df_levels.index.names
            else df_levels["ion_charge"].unique()
        )
        # Cannot check transitions ion_charge here, as it's derived from levels
        print(
            f"   Ion stages found in Levels file ('{LEVEL_FILE_BASE_NAME}.lev.asc'): {np.sort(levels_ions)}"
        )
        print(f"   (Ion stage for transitions will be determined from level match)")

    except KeyError as e:
        print(f"   Could not check ion stage overlap due to missing column: {e}")
else:
    print(" -> Skipping ion stage check as one or both DataFrames are empty.")

# --- Step 4: Add Level Info (including ion_charge) to Transitions ---
print("\n>>> Step 4: Adding Level Info to Transitions")
df_transitions_labeled = pd.DataFrame()  # Initialize
if not df_transitions.empty and not df_levels.empty:
    print("   Adding 'ion_charge', 'label', and 'configuration' from levels data...")

    # Ensure levels DataFrame is suitable for merging (index as columns)
    levels_for_lookup = (
        df_levels.reset_index()
        if isinstance(df_levels.index, pd.MultiIndex)
        else df_levels.copy()
    )

    # Check if levels_for_lookup has the necessary columns after potential reset
    # 'ion_charge' is now crucial here
    required_level_lookup_cols = [
        "atomic_number",
        "ion_charge",
        "level_index",
        "label",
        "configuration",
    ]
    if not all(col in levels_for_lookup.columns for col in required_level_lookup_cols):
        print(
            f"   ERROR: Levels DataFrame is missing required columns for lookup: {required_level_lookup_cols}. Cannot add level info."
        )
    else:
        # Add the level info using the new function
        # Explicitly request 'ion_charge' to be added
        df_transitions_labeled = add_level_info_to_transitions(
            df_transitions,
            levels_for_lookup,
            cols_to_add=[
                "ion_charge",
                "label",
                "configuration",
            ],  # Add ion_charge first
            missing_value="<N/A>",
        )

    # Display relevant columns including the new ones
    cols_to_show = [
        "atomic_number",
        "ion_charge",  # Now added from levels
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
    # Ensure only existing columns are selected
    cols_to_show = [
        col for col in cols_to_show if col in df_transitions_labeled.columns
    ]
    print_df_info(
        df_transitions_labeled[cols_to_show],
        "4. Transitions with Added Level Info Columns",
    )

    print(
        "\n   Note: '<N/A>' appears if the corresponding (atomic_number, level_index)"
    )
    print("         was not found in the loaded levels data for that specific level.")

elif df_levels.empty:
    print(" -> Cannot add level info because level data is missing or empty.")
else:
    print(" -> Cannot add level info because transition data is missing or empty.")


print("\n" + "=" * 70)
print("--- Transition Labeling Example Finished ---")
print("=" * 70)
