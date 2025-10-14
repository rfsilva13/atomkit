# atomkit/src/atomkit/readers/labeling.py

"""
Function for adding level information (labels, configurations, etc.)
to transition data.
"""

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

# No direct dependency on base reader or definitions needed here, only pandas/numpy

logger = logging.getLogger(__name__)


def add_level_info_to_transitions(
    transitions_df: pd.DataFrame,
    levels_df: pd.DataFrame,
    cols_to_add: list[str] = ["label", "configuration", "ion_charge"],
    missing_value: Any = "<N/A>",
) -> pd.DataFrame:
    """
    Adds specified level information (e.g., ion_charge, configuration, label) for
    both the lower and upper levels to a transitions DataFrame by merging.

    This function performs a left merge, keeping all original transitions and
    adding information where a matching level is found in `levels_df`.

    Args:
        transitions_df: DataFrame containing transition data. Must include
                        'atomic_number', 'level_index_lower', 'level_index_upper'.
                        These columns will be coerced to integer types for merging.
        levels_df: DataFrame containing level data. Must include 'atomic_number',
                   'level_index', and the columns specified in `cols_to_add`.
                   'atomic_number' and 'level_index' will be coerced to integer.
                   If `levels_df` has a MultiIndex, it will be reset.
        cols_to_add: A list of column names from `levels_df` to add to the
                     `transitions_df` for both lower and upper levels.
                     'ion_charge' is strongly recommended to be included if available
                     in `levels_df`, as it's crucial for identifying transitions.
        missing_value: The value to use in the added columns if a corresponding
                       level (based on atomic_number and level_index) is not
                       found in `levels_df`. Defaults to "<N/A>".

    Returns:
        A new DataFrame based on `transitions_df` with added columns (e.g.,
        'ion_charge_lower', 'label_lower', 'ion_charge_upper', 'label_upper').
        If 'ion_charge' was successfully added for both levels and they match,
        a single 'ion_charge' column is created.
        Returns a copy of the original `transitions_df` if required columns
        are missing in either input DataFrame or if an error occurs during merge.
    """
    # --- Input Validation ---
    if transitions_df.empty:
        logger.warning("Input transitions DataFrame is empty. Cannot add level info.")
        return transitions_df.copy()
    if levels_df.empty:
        logger.warning("Input levels DataFrame is empty. Cannot add level info.")
        return transitions_df.copy()

    # --- Prepare levels_df for merging ---
    # Make a copy to avoid modifying the original DataFrame
    levels_lookup = levels_df.copy()
    # Reset index if it's a MultiIndex to make index columns available for merging
    if isinstance(levels_lookup.index, pd.MultiIndex):
        levels_lookup = levels_lookup.reset_index()

    # --- Check for required columns ---
    required_trans_cols = {"atomic_number", "level_index_lower", "level_index_upper"}
    # Base required level columns for merging + user-requested columns
    required_level_cols = {"atomic_number", "level_index"}.union(set(cols_to_add))

    missing_trans_cols = required_trans_cols - set(transitions_df.columns)
    if missing_trans_cols:
        logger.error(
            f"Transitions DataFrame missing required columns for merge: {missing_trans_cols}. Cannot add level info."
        )
        return transitions_df.copy()

    missing_level_cols = required_level_cols - set(levels_lookup.columns)
    if missing_level_cols:
        logger.error(
            f"Levels DataFrame missing required columns for merge: {missing_level_cols}. Cannot add level info."
        )
        return transitions_df.copy()

    # --- Prepare DataFrames for Merge ---
    transitions_merged = transitions_df.copy()
    merge_key_cols_levels = ["atomic_number", "level_index"]
    trans_lower_key_cols = ["atomic_number", "level_index_lower"]
    trans_upper_key_cols = ["atomic_number", "level_index_upper"]

    try:
        # Convert merge key columns to nullable integer type for robust merging
        for col in merge_key_cols_levels:
            levels_lookup[col] = pd.to_numeric(
                levels_lookup[col], errors="coerce"
            ).astype("Int64")
        for (
            col
        ) in required_trans_cols:  # Includes atomic_number, level_index_lower/upper
            transitions_merged[col] = pd.to_numeric(
                transitions_merged[col], errors="coerce"
            ).astype("Int64")

        # Drop rows where merge keys became NA after conversion
        levels_lookup.dropna(subset=merge_key_cols_levels, inplace=True)
        transitions_merged.dropna(subset=list(required_trans_cols), inplace=True)

        # Check if DataFrames became empty after cleaning keys
        if levels_lookup.empty:
            logger.warning(
                "Levels DataFrame became empty after cleaning key columns. Cannot add level info."
            )
            return transitions_df.copy()
        if transitions_merged.empty:
            logger.warning(
                "Transitions DataFrame became empty after cleaning key columns. Cannot add level info."
            )
            return transitions_df.copy()

    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error preparing merge key columns: {e}. Cannot add level info.")
        return transitions_df.copy()

    # Select only necessary columns from levels for merge efficiency
    # Ensure only existing columns are selected
    cols_for_level_subset = [
        col for col in required_level_cols if col in levels_lookup.columns
    ]
    levels_subset = levels_lookup[cols_for_level_subset]

    # --- Perform Merges ---
    logger.debug(
        f"Merging for lower level info. Transitions shape: {transitions_merged.shape}, Levels subset shape: {levels_subset.shape}"
    )

    # Merge for lower level
    transitions_merged = pd.merge(
        transitions_merged,
        levels_subset,
        left_on=trans_lower_key_cols,
        right_on=merge_key_cols_levels,
        how="left",  # Keep all transitions
        suffixes=("", "_drop_lower"),  # Suffix for duplicated columns from levels
    )
    # Rename the newly added columns from levels
    rename_lower = {col: f"{col}_lower" for col in cols_to_add}
    transitions_merged.rename(columns=rename_lower, inplace=True)
    # Drop the original merge key ('level_index') and any duplicated 'atomic_number'
    cols_to_drop_lower = ["level_index"] + [
        f"{c}_drop_lower"
        for c in ["atomic_number"]
        if f"{c}_drop_lower" in transitions_merged.columns
    ]
    transitions_merged.drop(columns=cols_to_drop_lower, inplace=True, errors="ignore")
    logger.debug(
        f"After lower merge. Shape: {transitions_merged.shape}. Columns added: {list(rename_lower.values())}"
    )

    # Merge for upper level
    logger.debug(
        f"Merging for upper level info. Transitions shape: {transitions_merged.shape}, Levels subset shape: {levels_subset.shape}"
    )
    transitions_merged = pd.merge(
        transitions_merged,
        levels_subset,
        left_on=trans_upper_key_cols,
        right_on=merge_key_cols_levels,
        how="left",  # Keep all transitions
        suffixes=("", "_drop_upper"),
    )
    # Rename the newly added columns
    rename_upper = {col: f"{col}_upper" for col in cols_to_add}
    transitions_merged.rename(columns=rename_upper, inplace=True)
    # Drop the original merge key and duplicated 'atomic_number'
    cols_to_drop_upper = ["level_index"] + [
        f"{c}_drop_upper"
        for c in ["atomic_number"]
        if f"{c}_drop_upper" in transitions_merged.columns
    ]
    transitions_merged.drop(columns=cols_to_drop_upper, inplace=True, errors="ignore")
    logger.debug(
        f"After upper merge. Shape: {transitions_merged.shape}. Columns added: {list(rename_upper.values())}"
    )

    # --- Handle Ion Charge Consistency ---
    # If ion_charge was requested and added for both levels, check consistency
    # and create a single 'ion_charge' column.
    ion_charge_lower_col = "ion_charge_lower" if "ion_charge" in cols_to_add else None
    ion_charge_upper_col = "ion_charge_upper" if "ion_charge" in cols_to_add else None

    if (
        ion_charge_lower_col
        and ion_charge_upper_col
        and ion_charge_lower_col in transitions_merged.columns
        and ion_charge_upper_col in transitions_merged.columns
    ):

        # Ensure charges are numeric before comparison
        lower_charge_num = pd.to_numeric(
            transitions_merged[ion_charge_lower_col], errors="coerce"
        )
        upper_charge_num = pd.to_numeric(
            transitions_merged[ion_charge_upper_col], errors="coerce"
        )

        # Identify mismatches where both are numeric but different
        mismatch = (
            lower_charge_num.notna()
            & upper_charge_num.notna()
            & (lower_charge_num != upper_charge_num)
        )
        if mismatch.any():
            n_mismatch = mismatch.sum()
            logger.warning(
                f"Found {n_mismatch} transitions where lower and upper level ion charges differ. Using lower level charge for the 'ion_charge' column."
            )
            # Example: Log the first few mismatches for debugging
            # logger.debug(f"Mismatched transitions (first 5):\n{transitions_merged[mismatch].head()}")

        # Create the final 'ion_charge' column, prioritizing lower level charge
        # Use .loc to avoid SettingWithCopyWarning
        transitions_merged.loc[:, "ion_charge"] = lower_charge_num.astype(
            "Int64"
        )  # Use nullable integer

        # Drop the individual lower/upper ion charge columns
        transitions_merged.drop(
            columns=[ion_charge_lower_col, ion_charge_upper_col], inplace=True
        )
        logger.info("Added unified 'ion_charge' column based on level data.")

    elif ion_charge_lower_col and ion_charge_lower_col in transitions_merged.columns:
        # Only lower was requested/merged, use it as the main ion_charge
        transitions_merged["ion_charge"] = pd.to_numeric(
            transitions_merged[ion_charge_lower_col], errors="coerce"
        ).astype("Int64")
        transitions_merged.drop(columns=[ion_charge_lower_col], inplace=True)
        logger.info("Added 'ion_charge' column based on lower level data.")
    elif ion_charge_upper_col and ion_charge_upper_col in transitions_merged.columns:
        # Only upper was requested/merged, use it as the main ion_charge
        transitions_merged["ion_charge"] = pd.to_numeric(
            transitions_merged[ion_charge_upper_col], errors="coerce"
        ).astype("Int64")
        transitions_merged.drop(columns=[ion_charge_upper_col], inplace=True)
        logger.info("Added 'ion_charge' column based on upper level data.")
    elif "ion_charge" in cols_to_add:
        # ion_charge was requested but couldn't be added (e.g., missing from levels_df)
        logger.warning(
            "Could not add 'ion_charge' column as it was missing or failed during merge."
        )
        # Optionally add an empty column if needed downstream, otherwise do nothing
        # transitions_merged['ion_charge'] = pd.NA

    # --- Fill Missing Values ---
    # Fill NaNs in the newly added columns resulting from failed lookups
    n_filled_total = 0
    added_cols_final = list(rename_lower.values()) + list(rename_upper.values())
    # Also include the unified 'ion_charge' if it was created
    if (
        "ion_charge" in transitions_merged.columns
        and "ion_charge" not in added_cols_final
    ):
        added_cols_final.append("ion_charge")

    for col_name in added_cols_final:
        if col_name in transitions_merged.columns:
            n_filled = transitions_merged[col_name].isna().sum()
            if n_filled > 0:
                n_filled_total += n_filled
                transitions_merged[col_name].fillna(missing_value, inplace=True)

    if n_filled_total > 0:
        logger.info(
            f"Filled {n_filled_total} missing values across added level info columns with '{missing_value}'."
        )

    logger.info(f"Successfully added level information columns.")

    return transitions_merged
