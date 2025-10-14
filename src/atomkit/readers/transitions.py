# atomkit/src/atomkit/readers/transitions.py

"""
Reader class and function specifically for FAC transition files (.tr.asc).
"""

import logging
import os
import re
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.constants as const  # Keep for constants if used directly here

# Use absolute import based on package structure
from ..definitions import *  # Import constants and allowed units
from .base import _split_fac_file  # <--- Added import for _split_fac_file
from .base import (
    _BaseFacReader,  # Import from the new base module
    _extract_header_info,
    data_optimize,
)

logger = logging.getLogger(__name__)


class FacTransitionReader(_BaseFacReader):
    """
    Reads and processes FAC transition files (.tr.asc).

    Inherits common functionality from _BaseFacReader and implements
    transition-specific parsing and processing logic.
    """

    # Default concise column names expected/produced by this reader
    # Note: 'ion_charge' is NOT included here by default; it's added during labeling.
    _DEFAULT_OUTPUT_COLUMNS = [
        "atomic_number",
        "level_index_lower",
        "level_index_upper",
        "2j_lower",
        "2j_upper",
        "energy",  # Delta E for the transition
        "lambda",  # Calculated wavelength
        "gf",  # Oscillator strength * degeneracy
        "A",  # Transition probability (Einstein A)
        "S",  # Line strength
        "multipole",  # Original multipole value from file (if present)
        "type",  # Derived multipole type (e.g., 'E1', 'M2')
    ]
    # Columns potentially used for indexing *after* labeling adds ion_charge
    _INDEX_COLUMNS = [
        "atomic_number",
        "ion_charge",
        "level_index_upper",
        "level_index_lower",
    ]

    def __init__(
        self,
        energy_unit: str = "ev",
        wavelength_unit: str = "a",
        columns_to_keep: Optional[list[str]] = None,
        rename_columns: Optional[dict[str, str]] = None,
        output_prefix: str = "temp_fac_tr_block",  # Default prefix for temp files
        verbose: int = 1,
        include_method: bool = False,
    ):
        """
        Initializes the FAC transition file reader.

        Args:
            energy_unit (str): Target unit for the transition 'energy' (Delta E).
                               Allowed: 'ev', 'cm-1', 'ry', 'hz', 'j', 'ha'. Defaults to 'ev'.
            wavelength_unit (str): Target unit for the calculated 'lambda'.
                                   Allowed: 'a' (Angstrom), 'nm', 'm', 'cm'. Defaults to 'a'.
            columns_to_keep (Optional[list[str]]): List of concise column names to keep.
                                                   If None, defaults are used.
            rename_columns (Optional[dict[str, str]]): Dictionary mapping concise column names
                                                       to desired final names.
            output_prefix (str): Prefix for temporary files if splitting is needed.
            verbose (int): Logging verbosity (0: Warnings/Errors, 1: Info, 2: Debug). Defaults to 1.
            include_method (bool): If True, include a 'method' column (value 'FAC'). Defaults to False.
        """
        super().__init__(verbose=verbose, output_prefix=output_prefix)
        # Validate and store energy unit
        self.energy_unit = energy_unit.lower().replace("[", "").replace("]", "")
        if self.energy_unit not in ALLOWED_ENERGY_UNITS:
            raise ValueError(f"Invalid energy_unit. Allowed: {ALLOWED_ENERGY_UNITS}")
        # Validate and store wavelength unit
        self.wavelength_unit = wavelength_unit.lower().replace("[", "").replace("]", "")
        if self.wavelength_unit not in ALLOWED_WAVELENGTH_UNITS:
            raise ValueError(
                f"Invalid wavelength_unit. Allowed: {ALLOWED_WAVELENGTH_UNITS}"
            )
        self.columns_to_keep = columns_to_keep
        self.rename_columns = rename_columns
        self.include_method = include_method

    def read(
        self,
        input_filename: str,
        block_files: list[str] = [],
        block_starts: list[int] = [],
    ) -> pd.DataFrame:
        """
        Reads transition data from the specified file or pre-split block files.
        This method is typically called by the public `read_fac_transitions` function.

        Args:
            input_filename: Path to the original FAC transition file.
            block_files: List of paths to temporary block files (if pre-split).
            block_starts: List of original starting line numbers for each block file.

        Returns:
            A pandas DataFrame containing the processed transition data.
            Note: Ion charge is NOT included at this stage; it's added later via labeling.
        """
        all_transitions = pd.DataFrame()

        try:
            if block_files:  # Process pre-split files
                logger.info(
                    f"Reading {len(block_files)} pre-split transition block files..."
                )
                transitions_list = []
                for i, block_filename in enumerate(block_files):
                    start_line = block_starts[i] if i < len(block_starts) else None
                    start_line_display = (
                        start_line if start_line is not None else "Unknown"
                    )
                    logger.info(
                        f"--- Reading Transition Block {i+1} (Original Line: {start_line_display}) from {block_filename} ---"
                    )
                    transitions_temp = self._read_fac_transition_block_data(
                        block_filename, start_line
                    )
                    if not transitions_temp.empty:
                        transitions_list.append(transitions_temp)
                    else:
                        logger.info(
                            f"Skipping transition block {i+1} (from {block_filename}) due to errors or no data."
                        )
                if transitions_list:
                    logger.info(
                        f"--- Successfully read {len(transitions_list)} transition blocks. Concatenating... ---"
                    )
                    # Concatenate directly, post-processing handles sorting
                    all_transitions = pd.concat(transitions_list, ignore_index=True)
                    all_transitions = self._post_process_concatenated(
                        all_transitions
                    )  # Sorts
                else:
                    logger.error(
                        f"No transition blocks could be successfully read from provided block files."
                    )

            else:  # Process single file directly
                logger.info(f"Reading single transition file: {input_filename}")
                all_transitions = self._read_fac_transition_block_data(
                    input_filename, original_start_line=1
                )

        except Exception as e:
            logger.error(
                f"Unexpected error in FacTransitionReader.read for {input_filename}: {e}",
                exc_info=True,
            )
            all_transitions = pd.DataFrame()  # Ensure empty DF on error
        finally:
            # Log level restoration is handled by the main read_fac_transitions function
            # Cleanup is also handled by the main function
            pass

        # --- Apply Final Manipulations (Selection/Renaming) ---
        # Note: optional_col_map is None here as transitions don't have 'new_config'
        try:
            all_transitions = self._apply_final_manipulations(
                all_transitions,
                self._DEFAULT_OUTPUT_COLUMNS,  # Default columns for transitions
                self.columns_to_keep,
                self.rename_columns,
                self.include_method,
                optional_col_map=None,  # No extra optional columns specific to transitions reader
            )
        except Exception as e:
            logger.error(
                f"Error during final column manipulation for transitions from {input_filename}: {e}",
                exc_info=True,
            )
            all_transitions = pd.DataFrame()  # Return empty on error here too

        # Note: Index setting based on _INDEX_COLUMNS cannot reliably happen here
        # because 'ion_charge' is typically added during a later labeling step.

        # Final log message before returning
        if logger.isEnabledFor(logging.INFO):
            if not all_transitions.empty:
                logger.info(
                    f"Finished transition processing for {os.path.basename(input_filename)} (before labeling). Returning DataFrame with {len(all_transitions)} rows. Columns: {all_transitions.columns.tolist()}. Index: {all_transitions.index.names}"
                )
            else:
                logger.info(
                    f"Finished transition processing for {os.path.basename(input_filename)}, but resulted in an empty DataFrame."
                )
        return all_transitions

    def _read_fac_transition_block_data(
        self, file_transitions: str, original_start_line: Optional[int] = None
    ) -> pd.DataFrame:
        """Reads and processes data from a single FAC transition block file."""
        # Extract header info (Z, MULTIP) - Nele/ion_charge are not read from transition header
        atomic_number, _, multipole_header_value = _extract_header_info(
            file_transitions, reader_type="transition"
        )
        if atomic_number is None:
            logger.warning(
                f"Failed to extract Z from {file_transitions} (Original Line: {original_start_line or '?'}). Skipping block."
            )
            return pd.DataFrame()

        # Process the actual transition data within the block
        transitions_df = self._process_fac_transitions(
            atomic_number, multipole_header_value, file_transitions, original_start_line
        )

        if transitions_df.empty:
            logger.warning(
                f"Processing resulted in empty DataFrame for transition block (Original Line: {original_start_line or '?'})."
            )
        return transitions_df

    def _find_transition_data_start_row(self, lines: list[str]) -> tuple[int, str]:
        """Heuristically finds the starting row of the data table in transition files."""
        # Specific header pattern for transition files
        header_pattern = re.compile(r"\s*ILEV_UP\s+2J_UP\s+ILEV_LO\s+2J_LO\s+")
        for i, line in enumerate(lines):
            if header_pattern.match(line):
                # Data starts on the next line
                return (
                    i + 1,
                    f"Found specific transition header. Determined skiprows={i+1}.",
                )

        # Fallback: search for the first line starting with two numbers separated by space
        message = "Specific transition header not found. Fallback: searching for first line starting with two numbers."
        for i, line in enumerate(lines):
            if re.match(r"^\s*\d+\s+\d+", line):
                # Check if previous line did *not* match (to avoid header numbers) or if it's the first line
                if i > 0 and not re.match(r"^\s*\d+\s+\d+", lines[i - 1]):
                    return i, message + f" Found data at line {i+1}. skiprows={i}."
                elif i == 0:
                    return i, message + f" Found data at line {i+1}. skiprows={i}."

        # Default guess if patterns fail
        message = "Could not determine transition data start. Default guess."
        # Find MULTIP line index, default guess is 4 lines after that, or fallback
        multip_idx = next(
            (i for i, l in enumerate(lines) if l.strip().startswith("MULTIP")), -1
        )
        nele_idx = next(
            (i for i, l in enumerate(lines) if l.strip().startswith("NELE")), -1
        )  # Check NELE too
        skip_rows = (
            multip_idx + 4
            if multip_idx != -1
            else (nele_idx + 3 if nele_idx != -1 else 13)
        )  # Heuristic default
        return skip_rows, message + f" Determined skiprows={skip_rows}."

    def _process_fac_transitions(
        self,
        atomic_number: int,
        multipole_header_value: Optional[int],
        file_transitions: str,
        original_start_line: Optional[int] = None,
    ) -> pd.DataFrame:
        """Parses and processes the data within a single transition block."""
        file_context = f"(Original Line: {original_start_line or '?'}, File: {os.path.basename(file_transitions)})"
        try:
            with open(file_transitions, "r") as file:
                lines = file.readlines()
            if not lines:
                logger.warning(
                    f"Transition file {file_transitions} {file_context} is empty."
                )
                return pd.DataFrame()
        except Exception as e:
            logger.error(
                f"Error reading transition file {file_transitions} {file_context}: {e}",
                exc_info=True,
            )
            return pd.DataFrame()

        # Determine where data starts
        skiprows, skip_message = self._find_transition_data_start_row(lines)
        logger.info(f"{skip_message} {file_context}")

        # Define expected column names based on typical FAC .tr.asc format
        col_names = [
            "level_index_upper",
            "2j_upper",
            "level_index_lower",
            "2j_lower",
            "E",  # Transition energy (Delta E)
            "gf",  # Oscillator strength * degeneracy
            "A",  # Einstein A coefficient
            "multipole",  # Original multipole value from file (can be float like 0.007989845)
        ]
        # Essential columns needed for basic functionality and calculations
        essential_cols = [
            "level_index_upper",
            "2j_upper",
            "level_index_lower",
            "2j_lower",
            "E",
            "gf",
            "A",
        ]
        transitions = pd.DataFrame()

        # --- Attempt Reading Data ---
        try:
            # Try reading with space separation
            transitions = pd.read_csv(
                file_transitions,
                sep=r"\s+",
                names=col_names,
                skiprows=skiprows,
                on_bad_lines="warn",
                engine="python",
                comment="=",
                skipinitialspace=True,
            )
            logger.debug(f"CSV read successful for transitions {file_context}.")

            # --- Basic Cleaning and Validation ---
            if "E" in transitions.columns:
                transitions.rename(columns={"E": "energy"}, inplace=True)
            else:
                transitions["energy"] = np.nan  # Add energy column if missing

            essential_cols_renamed = [
                c if c != "E" else "energy" for c in essential_cols
            ]

            # Check if essential columns are present
            if not all(col in transitions.columns for col in essential_cols_renamed):
                # Fallback attempt if standard format failed (e.g., missing 2j columns)
                logger.warning(
                    f"Essential transition columns missing after standard read {file_context}. Trying fallback format."
                )
                col_names_alt = [
                    "level_index_upper",
                    "level_index_lower",
                    "E",
                    "gf",
                    "A",
                    "multipole",
                ]
                # Assuming the file structure for fallback is: ILEV_UP, (skip 2J_UP), ILEV_LO, (skip 2J_LO), E, GF, A, MULTIPOLE
                # This means we'd usecols like [0, 2, 4, 5, 6, 7] from the original 8-column expectation
                # If the file truly has fewer columns, this needs careful adjustment or a different parsing strategy.
                # For now, let's assume the original 8 columns are there but some might be empty or unparseable.
                # A more robust fallback might try to infer column positions.
                # This simplified fallback assumes the *structure* is there but 2j values might be missing/problematic.
                # If the file has physically fewer columns, read_csv would fail differently or misalign.
                # A common case is that 2j columns are present but perhaps empty or non-numeric.
                # The initial read_csv attempts to get all 8. If some are entirely missing, it might lead to fewer columns.
                # The check `all(col in transitions.columns for col in essential_cols_renamed)` handles this.

                # If 2j columns were truly missing from the read (e.g. file only had 6 columns), add them as NaN
                if "2j_upper" not in transitions.columns:
                    transitions["2j_upper"] = np.nan
                if "2j_lower" not in transitions.columns:
                    transitions["2j_lower"] = np.nan

                # Re-check essential columns after attempting to add missing 2j
                if not all(
                    col in transitions.columns for col in essential_cols_renamed
                ):
                    raise ValueError(
                        "Essential transition columns still missing after attempting to add 2j columns."
                    )

            # Convert essential columns to numeric
            for col in essential_cols_renamed:
                if col in transitions.columns:
                    transitions[col] = pd.to_numeric(transitions[col], errors="coerce")

            # Drop rows with NaN in essential columns
            initial_rows = len(transitions)
            transitions.dropna(
                subset=[c for c in essential_cols_renamed if c in transitions.columns],
                inplace=True,
            )
            if len(transitions) < initial_rows:
                logger.info(
                    f"Dropped {initial_rows-len(transitions)} rows (non-numeric essentials) {file_context}."
                )

            if transitions.empty:
                logger.warning(
                    f"No valid transition data after cleaning essentials {file_context}."
                )
                return pd.DataFrame()

            # Ensure multipole column exists
            if "multipole" not in transitions.columns:
                transitions["multipole"] = (
                    "-"  # Use string for potential non-numeric values
                )

        except Exception as e_read:
            logger.error(
                f"Error reading transition data {file_context}: {e_read}", exc_info=True
            )
            return pd.DataFrame()

        # --- Post-processing (Applied after successful read) ---
        try:
            # Convert index/J columns to nullable integers
            for col in ["level_index_upper", "level_index_lower"]:
                transitions[col] = pd.to_numeric(
                    transitions[col], errors="coerce"
                ).astype("Int64")
            for col in ["2j_upper", "2j_lower"]:
                if col in transitions.columns:
                    transitions[col] = pd.to_numeric(
                        transitions[col], errors="coerce"
                    ).astype("Int64")

            # Clean multipole string column
            if "multipole" in transitions.columns:
                transitions["multipole"] = (
                    transitions["multipole"].astype(str).str.strip().fillna("-")
                )

            # Calculate wavelength (lambda) in Angstrom (internal standard)
            if "energy" in transitions.columns and pd.api.types.is_numeric_dtype(
                transitions["energy"]
            ):
                # Use only positive energy transitions for wavelength calculation
                valid_delta_e = (
                    transitions["energy"] > 1e-9
                )  # Avoid division by zero or small numbers
                transitions["lambda"] = np.nan  # Initialize column
                if valid_delta_e.any():
                    transitions.loc[valid_delta_e, "lambda"] = (
                        HC_EV_ANGSTROM / transitions.loc[valid_delta_e, "energy"]
                    )
                logger.debug(f"Calculated internal 'lambda' (Angstrom) {file_context}.")
            else:
                transitions["lambda"] = np.nan

            # Determine multipole type (E1, M1, E2, etc.)
            # Use header value if available, otherwise try parsing the 'multipole' column
            if multipole_header_value is not None:
                # Determine type based on sign and value from header
                transitions["type"] = (
                    f"{'M' if multipole_header_value > 0 else 'E'}{abs(multipole_header_value)}"
                )
            else:
                # Try to infer from the 'multipole' column if it looks like an integer +/- value
                def infer_type(mp_val):
                    try:
                        # Attempt to convert to float first, then to int, to handle cases like "7.989845E-03" for M1 etc.
                        # The sign of the *integer part* of the multipole value from the file often indicates E/M.
                        # For FAC, the last column is often a small mixing coefficient or other value,
                        # not always the integer multipole order directly.
                        # The *header* MULTIP is more reliable for E/M type.
                        # If header is missing, this inference is a best guess.
                        # A common convention is that the integer part of the last column (if it's the multipole order)
                        # would be used. E.g., -1 for E1, +1 for M1.
                        mp_float = float(mp_val)
                        # Check if it's close to an integer
                        if (
                            abs(mp_float - round(mp_float)) < 1e-3
                        ):  # If it's essentially an integer
                            mp_int = int(round(mp_float))
                            if mp_int != 0:
                                return f"{'M' if mp_int > 0 else 'E'}{abs(mp_int)}"
                    except (ValueError, TypeError):
                        pass  # Cannot convert or invalid value
                    return "Unknown"  # Default if cannot infer

                if "multipole" in transitions.columns:
                    transitions["type"] = transitions["multipole"].apply(infer_type)
                else:
                    transitions["type"] = "Unknown"
            logger.debug(f"Determined 'type' (multipole type) {file_context}.")

            # Calculate line strength (S) in atomic units
            if "gf" in transitions.columns and "lambda" in transitions.columns:
                gf_num = pd.to_numeric(transitions["gf"], errors="coerce")
                lambda_num = pd.to_numeric(
                    transitions["lambda"], errors="coerce"
                )  # Use calculated lambda in Angstrom
                valid_S = gf_num.notna() & lambda_num.notna() & (lambda_num > 1e-9)
                transitions["S"] = np.nan  # Initialize column
                if valid_S.any():
                    # Formula: S(au) = gf * lambda(A)^2 / (const * E(eV)) -> simplified S = gf * lambda(A) / 303.756
                    transitions.loc[valid_S, "S"] = (
                        gf_num[valid_S] * lambda_num[valid_S]
                    ) / LINE_STRENGTH_CONST
                logger.debug(f"Calculated 'S' (a.u.) {file_context}.")
            else:
                transitions["S"] = np.nan

            # Convert energy units (based on internal eV)
            if "energy" in transitions.columns and pd.api.types.is_numeric_dtype(
                transitions["energy"]
            ):
                energy_ev = transitions["energy"].copy()
                if self.energy_unit == "cm-1":
                    transitions["energy"] = energy_ev * EV_TO_CM1
                elif self.energy_unit == "ry":
                    transitions["energy"] = energy_ev / RYD_EV
                elif self.energy_unit == "hz":
                    transitions["energy"] = energy_ev * const.e / const.h
                elif self.energy_unit == "j":
                    transitions["energy"] = energy_ev * const.e
                elif self.energy_unit == "ha":
                    transitions["energy"] = energy_ev / HARTREE_EV
                if self.energy_unit != "ev":
                    logger.debug(
                        f"Converted 'energy' to {self.energy_unit} {file_context}."
                    )

            # Convert wavelength units (based on internal Angstrom)
            if "lambda" in transitions.columns and pd.api.types.is_numeric_dtype(
                transitions["lambda"]
            ):
                lambda_A = transitions["lambda"].copy()
                if self.wavelength_unit == "nm":
                    transitions["lambda"] = lambda_A / 10.0
                elif self.wavelength_unit == "m":
                    transitions["lambda"] = lambda_A * 1e-10
                elif self.wavelength_unit == "cm":
                    transitions["lambda"] = lambda_A * 1e-8
                if self.wavelength_unit != "a":
                    logger.debug(
                        f"Converted 'lambda' to {self.wavelength_unit} {file_context}."
                    )

            # Add metadata columns (Z and method)
            transitions["atomic_number"] = atomic_number
            transitions["method"] = "FAC"

            # Optimize data types
            transitions = data_optimize(transitions)

            # Select and order default columns before user manipulation
            current_default_cols = self._DEFAULT_OUTPUT_COLUMNS[:]  # Make a copy
            if self.include_method:
                current_default_cols.append("method")

            # Keep only columns that actually exist
            transitions = transitions[
                [col for col in current_default_cols if col in transitions.columns]
            ]

            # Final optimization and sorting
            transitions = data_optimize(transitions)
            transitions = transitions.sort_values(
                by=["level_index_upper", "level_index_lower"]
            )

            logger.debug(f"Transition processing for block complete (index not set).")

        except Exception as e:
            logger.error(
                f"Error in transition post-processing {file_context}: {e}",
                exc_info=True,
            )
            return pd.DataFrame()

        return transitions

    def _post_process_concatenated(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-processes the combined DataFrame after reading all transition blocks."""
        # Sorts but does NOT attempt to set index here, as ion_charge is missing.
        if df.empty:
            return df
        df = data_optimize(df)
        # Sort primarily by atomic number, then level indices
        sort_cols = [
            col
            for col in ["atomic_number", "level_index_upper", "level_index_lower"]
            if col in df.columns
        ]
        if sort_cols:
            df.sort_values(by=sort_cols, inplace=True)
        logger.debug("Concatenated transitions sorted.")
        return df


# --- Main Public Function (using the class) ---
def read_fac_transitions(
    base_filename: str, file_extension: str = ".tr.asc", verbose: int = 1, **kwargs
) -> pd.DataFrame:
    """
    Reads FAC transition data (.tr.asc), handling multi-block files.

    Args:
        base_filename (str): Base path and name of the FAC file (without extension).
        file_extension (str): File extension (default: '.tr.asc').
        verbose (int): Logging verbosity (0: Warnings/Errors, 1: Info, 2: Debug).
        **kwargs: Additional keyword arguments passed to the FacTransitionReader constructor,
                  e.g., `energy_unit`, `wavelength_unit`, `columns_to_keep`,
                  `rename_columns`, `include_method`.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed transition data.
                          Returns an empty DataFrame on error or if no data found.
    """
    input_filename = f"{base_filename}{file_extension}"
    # Create a unique prefix for temp files based on the input filename
    output_prefix = (
        f"temp_fac_tr_block_{os.path.splitext(os.path.basename(input_filename))[0]}"
    )
    reader = None  # Initialize reader to None
    all_transitions = pd.DataFrame()

    # Check file existence early
    if not os.path.exists(input_filename):
        logger.error(f"Input file not found: {input_filename}")
        return pd.DataFrame()

    try:
        # Instantiate the reader (handles log level setup)
        reader = FacTransitionReader(
            verbose=verbose, output_prefix=output_prefix, **kwargs
        )

        # Split the file if necessary (using function from base)
        num_blocks_written, block_start_lines = (
            _split_fac_file(  # Ensure this is called correctly
                input_filename, reader.output_prefix  # Use reader's output_prefix
            )
        )

        # Prepare list of block files if splitting occurred
        block_files = []
        if (
            num_blocks_written > 1
        ):  # Only if multiple blocks were actually written as temp files
            block_files = [
                f"{reader.output_prefix}_{i+1}.txt" for i in range(num_blocks_written)
            ]
            reader._temp_files_created = block_files  # Register temp files for cleanup

        # Read the data (either single file or blocks)
        if num_blocks_written == 0:
            logger.error(
                f"Error during file splitting or no data blocks found for {input_filename}."
            )
            # No need to call read() if splitting failed this way
        elif (
            num_blocks_written == 1 and not block_files
        ):  # Single block detected by splitter (no temp files written)
            all_transitions = reader.read(
                input_filename, block_files=[], block_starts=block_start_lines
            )
        else:  # Multiple blocks written or single block file needs reading
            all_transitions = reader.read(
                input_filename, block_files=block_files, block_starts=block_start_lines
            )

    except Exception as e:
        logger.error(
            f"Error during read_fac_transitions execution for {input_filename}: {e}",
            exc_info=True,
        )
        all_transitions = pd.DataFrame()  # Ensure empty DF on error
    finally:
        # Cleanup temp files and restore log level using the reader instance if it was created
        if reader:
            reader._cleanup_temp_files()
            reader._restore_log_level()  # Restore log level

    return all_transitions
