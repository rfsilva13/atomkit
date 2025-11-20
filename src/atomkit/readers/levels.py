# atomkit/src/atomkit/readers/levels.py

"""
Reader class and function specifically for FAC energy level files (.lev.asc).
"""

import logging
import os
import re
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.constants as const  # Keep for EV_TO_CM1 etc. if used directly here in future

from ..constants import *  # Import constants and allowed units
# Use absolute import based on package structure
from ..structure import Configuration
from .base import J_PI_indexing  # Import from the new base module
from .base import _split_fac_file  # <--- Added import for _split_fac_file
from .base import _BaseFacReader, _extract_header_info, data_optimize

logger = logging.getLogger(__name__)


class FacReader(_BaseFacReader):
    """
    Reads and processes FAC energy level files (.lev.asc).

    Inherits common functionality from _BaseFacReader and implements
    level-specific parsing and processing logic.
    """

    # Default concise column names expected/produced by this reader
    _DEFAULT_OUTPUT_COLUMNS = [
        "atomic_number",
        "ion_charge",
        "level_index",
        "energy",
        "p",
        "2j",
        "J_PI_index",
        "conf_detail",
        "configuration",
        "rel_config",
        "term",
        "label",
    ]
    # Columns used for setting the multi-index
    _INDEX_COLUMNS = ["atomic_number", "ion_charge", "2j", "p", "J_PI_index"]

    def __init__(
        self,
        conf1: bool = False,  # Kept for potential future use, currently unused
        energy_unit: str = "ev",
        columns_to_keep: Optional[list[str]] = None,
        rename_columns: Optional[dict[str, str]] = None,
        output_prefix: str = "temp_fac_lev_block",  # Default prefix for temp files
        verbose: int = 1,
        include_method: bool = False,
        include_new_config: bool = False,
    ):
        """
        Initializes the FAC level file reader.

        Args:
            conf1 (bool): Placeholder, currently unused. Defaults to False.
            energy_unit (str): Target unit for the 'energy' column.
                               Allowed: 'ev', 'cm-1', 'ry', 'hz', 'j', 'ha'. Defaults to 'ev'.
            columns_to_keep (Optional[list[str]]): List of concise column names to keep.
                                                   If None, defaults are used.
            rename_columns (Optional[dict[str, str]]): Dictionary mapping concise column names
                                                       to desired final names.
            output_prefix (str): Prefix for temporary files if splitting is needed.
            verbose (int): Logging verbosity (0: Warnings/Errors, 1: Info, 2: Debug). Defaults to 1.
            include_method (bool): If True, include a 'method' column (value 'FAC'). Defaults to False.
            include_new_config (bool): If True, attempt to parse 'conf_detail' into
                                       Configuration objects in a 'new_config' column. Defaults to False.
        """
        super().__init__(verbose=verbose, output_prefix=output_prefix)
        self.conf1 = conf1  # Store even if unused for now
        # Validate and store energy unit
        self.energy_unit = energy_unit.lower().replace("[", "").replace("]", "")
        if self.energy_unit not in ALLOWED_ENERGY_UNITS:
            raise ValueError(
                f"Invalid energy_unit '{energy_unit}'. Allowed: {ALLOWED_ENERGY_UNITS}"
            )
        self.columns_to_keep = columns_to_keep
        self.rename_columns = rename_columns
        self.include_method = include_method
        self.include_new_config = include_new_config

    def read(
        self,
        input_filename: str,
        block_files: list[str] = [],
        block_starts: list[int] = [],
    ) -> pd.DataFrame:
        """
        Reads level data from the specified file or pre-split block files.
        This method is typically called by the public `read_fac` function.

        Args:
            input_filename: Path to the original FAC level file.
            block_files: List of paths to temporary block files (if pre-split).
            block_starts: List of original starting line numbers for each block file.

        Returns:
            A pandas DataFrame containing the processed level data.
        """
        all_levels = pd.DataFrame()

        try:
            if block_files:  # Process pre-split files
                logger.info(
                    f"Reading {len(block_files)} pre-split level block files..."
                )
                levels_list = []
                for i, block_filename in enumerate(block_files):
                    start_line = block_starts[i] if i < len(block_starts) else None
                    start_line_display = (
                        start_line if start_line is not None else "Unknown"
                    )
                    logger.info(
                        f"--- Reading Level Block {i+1} (Original Line: {start_line_display}) from {block_filename} ---"
                    )
                    levels_temp = self._read_fac_block_data(block_filename, start_line)
                    if not levels_temp.empty:
                        levels_list.append(levels_temp)
                    else:
                        logger.info(
                            f"Skipping level block {i+1} (from {block_filename}) due to errors or no data."
                        )
                if levels_list:
                    logger.info(
                        f"--- Successfully read {len(levels_list)} level blocks. Concatenating... ---"
                    )
                    # Reset index before concat if multi-index exists to avoid issues
                    processed_blocks = []
                    for df_block in levels_list:
                        if df_block.index.nlevels > 1 or any(df_block.index.names):
                            processed_blocks.append(df_block.reset_index(drop=False))
                        else:
                            processed_blocks.append(df_block)
                    all_levels = pd.concat(processed_blocks, ignore_index=True)
                    all_levels = self._post_process_concatenated(
                        all_levels
                    )  # Apply sorting and indexing
                else:
                    logger.error(
                        f"No level blocks could be successfully read from provided block files."
                    )

            else:  # Process single file directly
                logger.info(f"Reading single level file: {input_filename}")
                all_levels = self._read_fac_block_data(
                    input_filename, original_start_line=1
                )  # Assume start line 1 for single file

        except Exception as e:
            logger.error(
                f"Unexpected error in FacReader.read for {input_filename}: {e}",
                exc_info=True,
            )
            all_levels = pd.DataFrame()  # Ensure empty DF on error
        finally:
            # Log level restoration is handled by the main read_fac function
            # Cleanup is also handled by the main function
            pass

        # --- Apply Final Manipulations (Selection/Renaming) ---
        # Define which optional columns might exist based on flags
        optional_col_map = {"new_config": self.include_new_config}
        try:
            all_levels = self._apply_final_manipulations(
                all_levels,
                self._DEFAULT_OUTPUT_COLUMNS,
                self.columns_to_keep,
                self.rename_columns,
                self.include_method,
                optional_col_map,
            )
        except Exception as e:
            logger.error(
                f"Error during final column manipulation for levels from {input_filename}: {e}",
                exc_info=True,
            )
            all_levels = pd.DataFrame()  # Return empty on error here too

        # Final log message before returning
        if logger.isEnabledFor(logging.INFO):
            if not all_levels.empty:
                logger.info(
                    f"Finished level processing for {os.path.basename(input_filename)}. Returning DataFrame with {len(all_levels)} rows. Columns: {all_levels.columns.tolist()}. Index: {all_levels.index.names}"
                )
            else:
                logger.info(
                    f"Finished level processing for {os.path.basename(input_filename)}, but resulted in an empty DataFrame."
                )
        return all_levels

    def _read_fac_block_data(
        self, file_levels: str, original_start_line: Optional[int] = None
    ) -> pd.DataFrame:
        """Reads and processes data from a single FAC level block file."""
        # Extract header info (Z, Nele) - crucial for levels
        atomic_number, nele, _ = _extract_header_info(file_levels, reader_type="level")
        if atomic_number is None or nele is None:
            logger.warning(
                f"Failed to extract Z/Nele from {file_levels} (Original Line: {original_start_line or '?'}). Skipping block."
            )
            return pd.DataFrame()

        # Calculate ion charge
        ion_charge = atomic_number - nele

        # Process the actual level data within the block
        levels_df = self._process_fac_levels(
            atomic_number, ion_charge, file_levels, original_start_line
        )

        if levels_df.empty:
            logger.warning(
                f"Processing resulted in empty DataFrame for level block (Original Line: {original_start_line or '?'})."
            )
        return levels_df

    def _find_data_start_row(self, lines: list[str]) -> tuple[int, str]:
        """Heuristically finds the starting row of the data table in level files."""
        # Specific header pattern for level files
        header_pattern = re.compile(r"\s*ILEV\s+IBASE\s+ENERGY\s+P\s+VNL\s+2J")
        for i, line in enumerate(lines):
            if header_pattern.match(line):
                # Data starts on the next line after this header
                return i + 1, f"Found specific level header. Determined skiprows={i+1}."

        # Fallback: search for the first line starting with a digit
        message = "Specific level header not found. Fallback: searching for first line starting with a digit."
        for i, line in enumerate(lines):
            if re.match(r"^\s*\d+", line):
                # Check if previous line did *not* start with a digit (to avoid header numbers)
                # or if it's the very first line
                if i > 0 and not re.match(r"^\s*\d+", lines[i - 1]):
                    return i, message + f" Found data at line {i+1}. skiprows={i}."
                elif i == 0:
                    return i, message + f" Found data at line {i+1}. skiprows={i}."

        # Default guess if patterns fail
        message = "Could not determine level data start. Default guess."
        # Find NELE line index, default guess is 3 lines after that, or 12 if NELE not found
        nele_idx = next(
            (i for i, l in enumerate(lines) if l.strip().startswith("NELE")), -1
        )
        skip_rows = nele_idx + 3 if nele_idx != -1 else 12
        return skip_rows, message + f" Determined skiprows={skip_rows}."

    def _process_fac_levels(
        self,
        atomic_number: int,
        ion_charge: int,
        file_levels: str,
        original_start_line: Optional[int] = None,
    ) -> pd.DataFrame:
        """Parses and processes the data within a single level block."""
        file_context = f"(Original Line: {original_start_line or '?'}, File: {os.path.basename(file_levels)})"
        try:
            with open(file_levels, "r") as file:
                lines = file.readlines()
            if not lines:
                logger.warning(f"Level file {file_levels} {file_context} is empty.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(
                f"Error reading level file {file_levels} {file_context}: {e}",
                exc_info=True,
            )
            return pd.DataFrame()

        # Determine where data starts
        skiprows, skip_message = self._find_data_start_row(lines)
        logger.info(f"{skip_message} {file_context}")

        # Define expected column names based on typical FAC .lev.asc format
        observed_col_names = [
            "level_index",
            "IBASE",  # Often present but unused
            "E",  # Energy (will be renamed to 'energy')
            "p",  # Parity
            "VNL",  # Often present but unused
            "2j",  # Total angular momentum * 2
            "conf_detail",  # Compact config or mixing info
            "configuration",  # Full configuration string
            "label",  # Term label / relativistic label
        ]
        levels = pd.DataFrame()

        # --- Attempt Reading Data ---
        try:
            # Try reading with space separation first (more flexible)
            levels = pd.read_csv(
                file_levels,
                sep=r"\s+",  # Regex for one or more spaces
                names=observed_col_names,
                skiprows=skiprows,
                on_bad_lines="warn",  # Report problematic lines
                engine="python",  # More robust for regex separators
                comment="=",  # Skip comment lines
                skipinitialspace=True,
            )
            logger.debug(f"CSV read successful for levels {file_context}.")

            # --- Basic Cleaning and Validation (Common to both read methods) ---
            if "E" in levels.columns:
                levels.rename(columns={"E": "energy"}, inplace=True)
            else:
                levels["energy"] = np.nan  # Add energy column if missing

            # Check for essential numeric columns
            essential_cols = ["level_index", "energy", "p", "2j"]
            if not all(col in levels.columns for col in essential_cols):
                # If essential columns are missing after CSV read, try FWF immediately
                raise ValueError("Essential level columns missing after CSV read.")

            # Convert essential columns to numeric, coercing errors
            for col in essential_cols:
                levels[col] = pd.to_numeric(levels[col], errors="coerce")

            initial_rows = len(levels)
            levels.dropna(
                subset=essential_cols, inplace=True
            )  # Drop rows with NaN in essentials
            if len(levels) < initial_rows:
                logger.info(
                    f"Dropped {initial_rows - len(levels)} rows due to non-numeric essentials {file_context}."
                )

            if levels.empty:
                logger.warning(
                    f"No valid level data after cleaning essentials {file_context}."
                )
                return pd.DataFrame()

            # Ensure string columns exist, filling with '-' if needed
            for col in [
                "conf_detail",
                "configuration",
                "label",
                "conf1",
            ]:  # conf1 might appear sometimes
                if col not in levels.columns:
                    levels[col] = "-"

            # Handle potential merged configuration/label column
            # If 'label' is empty/'-' and 'configuration' contains spaces (likely term appended)
            if (
                levels["label"].isin(["-", np.nan]).all()
                and levels["configuration"].astype(str).str.contains(r"\s\S+$").any()
            ):
                logger.debug(
                    f"Attempting to split merged 'configuration' and 'label' {file_context}."
                )
                # Split 'configuration' on the last space
                split_conf = (
                    levels["configuration"].astype(str).str.rsplit(n=1, expand=True)
                )
                if split_conf.shape[1] == 2:  # Check if split was successful
                    levels["configuration"], levels["label"] = (
                        split_conf[0].str.strip(),
                        split_conf[1].str.strip(),
                    )
                else:
                    logger.warning(
                        f"Splitting configuration/label failed {file_context}."
                    )

        except Exception as e_csv:
            logger.warning(
                f"Flexible CSV read failed for levels {file_context}: {e_csv}. Trying FWF."
            )
            # --- Fallback to Fixed-Width Reading ---
            try:
                # Define fixed-width specifications
                widths = [
                    (0, 7),
                    (7, 14),
                    (14, 30),
                    (30, 31),
                    (32, 38),
                    (38, 43),
                    (43, 70),
                    (70, 100),
                    (100, 150),  # Adjust widths as needed
                ]
                names_fwf = observed_col_names  # Use same names
                levels = pd.read_fwf(
                    file_levels, colspecs=widths, names=names_fwf, skiprows=skiprows
                )

                # Repeat essential cleaning steps for FWF data
                if "E" in levels.columns:
                    levels.rename(columns={"E": "energy"}, inplace=True)
                else:
                    levels["energy"] = np.nan

                essential_cols = ["level_index", "energy", "p", "2j"]
                for col in essential_cols:
                    if col in levels.columns:
                        levels[col] = pd.to_numeric(levels[col], errors="coerce")

                levels.dropna(
                    subset=[c for c in essential_cols if c in levels.columns],
                    inplace=True,
                )

                if levels.empty:
                    raise ValueError("No valid data after FWF cleaning.")

                # Ensure string columns exist after FWF read
                for col in ["conf_detail", "configuration", "label", "conf1"]:
                    if col not in levels.columns:
                        levels[col] = "-"

            except Exception as e_fwf:
                logger.error(
                    f"FWF read also failed for levels {file_context}: {e_fwf}",
                    exc_info=True,
                )
                return pd.DataFrame()  # Return empty if both methods fail

        # --- Post-processing (Applied after successful read) ---
        try:
            # Convert types
            for col in ["level_index", "p", "2j"]:
                # Use nullable integer type Int64
                levels[col] = pd.to_numeric(levels[col], errors="coerce").astype(
                    "Int64"
                )
            for col in ["conf_detail", "configuration", "label", "conf1"]:
                if col in levels.columns:
                    levels[col] = levels[col].astype(str).str.strip().fillna("-")

            # Extract relativistic config from label if possible
            if "label" in levels.columns:
                # Regex to find patterns like '3d+5', '1s2', '4f-1' etc. within the label string
                orbital_pattern = re.compile(r"(\d+[a-zA-Z][+\-]?\d*)")
                levels["rel_config"] = levels["label"].apply(
                    lambda x: (
                        ".".join(orbital_pattern.findall(x))
                        if isinstance(x, str) and orbital_pattern.findall(x)
                        else ("-" if x == "-" else x)
                    )  # Keep original label if no pattern found
                )
            else:
                levels["rel_config"] = "-"

            # Create Configuration objects if requested
            levels["new_config"] = None  # Initialize column
            if self.include_new_config and "conf_detail" in levels.columns:

                def safe_from_compact(x: Any) -> Any:
                    """Convert compact string to Configuration object, returns None on failure."""
                    if isinstance(x, str) and x.strip() and x != "-":
                        try:
                            # Assuming from_compact_string returns a single config or raises error
                            result = Configuration.from_compact_string(
                                x, generate_permutations=False
                            )
                            return result
                        except (ValueError, RuntimeError) as e:
                            logger.debug(f"Could not parse compact string '{x}': {e}")
                            return None  # Return None if parsing fails
                    return None

                # Apply function - using Any return type to avoid pandas type signature issues
                # This is safe because pandas .apply() with object dtype works correctly at runtime
                levels["new_config"] = levels["conf_detail"].apply(safe_from_compact)

            levels["term"] = "-"  # Placeholder for term symbol

            # Optimize data types
            levels = data_optimize(levels)

            # Convert energy units
            if "energy" in levels.columns and pd.api.types.is_numeric_dtype(
                levels["energy"]
            ):
                energy_ev = levels["energy"].copy()  # Assume input is eV
                if self.energy_unit == "cm-1":
                    levels["energy"] = energy_ev * EV_TO_CM1
                elif self.energy_unit == "ry":
                    levels["energy"] = energy_ev / RYD_EV
                elif self.energy_unit == "hz":
                    levels["energy"] = energy_ev * const.e / const.h
                elif self.energy_unit == "j":
                    levels["energy"] = energy_ev * const.e
                elif self.energy_unit == "ha":
                    levels["energy"] = energy_ev / HARTREE_EV
                # else: keep as eV (default)
                if self.energy_unit != "ev":
                    logger.debug(
                        f"Converted 'energy' to {self.energy_unit} {file_context}."
                    )

            # Add metadata columns
            levels["atomic_number"] = atomic_number
            levels["ion_charge"] = ion_charge
            levels["method"] = "FAC"  # Add method column

            # Calculate J_PI index
            levels = J_PI_indexing(levels)

            # Clean up configuration string (remove occupation 1)
            if "configuration" in levels.columns:
                levels["configuration"] = levels["configuration"].apply(
                    lambda x: (
                        re.sub(r"(?<=[a-zA-Z])1(?!\d)", "", x)
                        if isinstance(x, str)
                        else x
                    )
                )

            # Select and order default columns before user manipulation
            # This ensures essential calculated columns are present before filtering
            current_default_cols = self._DEFAULT_OUTPUT_COLUMNS[:]  # Make a copy
            if self.include_method:
                current_default_cols.append("method")
            if self.include_new_config:
                current_default_cols.append("new_config")

            # Keep only columns that actually exist in the DataFrame
            levels = levels[
                [col for col in current_default_cols if col in levels.columns]
            ]

            # Final optimization and sorting
            levels = data_optimize(levels)
            if "energy" in levels.columns:
                # Ensure energy is numeric before sorting
                levels["energy"] = pd.to_numeric(levels["energy"], errors="coerce")
                levels = levels.sort_values(by=["energy"])

            # Set multi-index if possible
            index_cols_present = [
                col for col in self._INDEX_COLUMNS if col in levels.columns
            ]
            if len(index_cols_present) == len(self._INDEX_COLUMNS):
                # Check for duplicates before setting index
                if not levels.duplicated(subset=index_cols_present).any():
                    levels.set_index(index_cols_present, inplace=True)
                    logger.debug(f"Set multi-index for level block {file_context}.")
                else:
                    logger.warning(
                        f"Duplicate index values found in level block {file_context}. Index not set for this block."
                    )
            else:
                logger.warning(
                    f"One or more index columns ({self._INDEX_COLUMNS}) missing in level block {file_context}. Not setting index."
                )

        except Exception as e:
            logger.error(
                f"Error in level post-processing {file_context}: {e}", exc_info=True
            )
            return pd.DataFrame()

        return levels

    def _post_process_concatenated(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-processes the combined DataFrame after reading all level blocks."""
        if df.empty:
            return df

        # Optimize data types first
        df = data_optimize(df)

        # Sort by Z, ion_charge, then energy
        sort_cols = [
            col
            for col in ["atomic_number", "ion_charge", "energy"]
            if col in df.columns
        ]
        if sort_cols:
            # Ensure energy is numeric before sorting
            if "energy" in df.columns:
                df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
            df.sort_values(by=sort_cols, inplace=True, na_position="last")

        # Attempt to set the multi-index
        index_cols_present = [col for col in self._INDEX_COLUMNS if col in df.columns]
        if len(index_cols_present) == len(self._INDEX_COLUMNS):
            try:
                # Check for duplicates *across the concatenated dataframe*
                if not df.duplicated(subset=index_cols_present).any():
                    df.set_index(index_cols_present, inplace=True)
                    logger.info(
                        "Successfully set multi-index after concatenating level blocks."
                    )
                else:
                    logger.warning(
                        f"Duplicate index values found after concatenating level blocks. Index not set."
                    )
            except Exception as e:
                logger.warning(
                    f"Could not set multi-index after concatenating level blocks: {e}. Keeping sorted but unindexed."
                )
        else:
            logger.warning(
                "One or more index columns missing after level concatenation. Keeping sorted but unindexed."
            )
        return df


# --- Main Public Function (using the class) ---
def read_fac(
    base_filename: str, file_extension: str = ".lev.asc", verbose: int = 1, **kwargs
) -> pd.DataFrame:
    """
    Reads FAC energy level data (.lev.asc), handling multi-block files.

    Args:
        base_filename (str): Base path and name of the FAC file (without extension).
        file_extension (str): File extension (default: '.lev.asc').
        verbose (int): Logging verbosity (0: Warnings/Errors, 1: Info, 2: Debug).
        **kwargs: Additional keyword arguments passed to the FacReader constructor,
                  e.g., `energy_unit`, `columns_to_keep`, `rename_columns`,
                  `include_method`, `include_new_config`.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed level data.
                          Returns an empty DataFrame on error or if no data found.
    """
    input_filename = f"{base_filename}{file_extension}"
    # Create a unique prefix for temp files based on the input filename
    output_prefix = (
        f"temp_fac_lev_block_{os.path.splitext(os.path.basename(input_filename))[0]}"
    )
    reader = None  # Initialize reader to None
    all_levels = pd.DataFrame()

    # Check file existence early
    if not os.path.exists(input_filename):
        logger.error(f"Input file not found: {input_filename}")
        return pd.DataFrame()

    try:
        # Instantiate the reader (handles log level setup)
        reader = FacReader(verbose=verbose, output_prefix=output_prefix, **kwargs)

        # Split the file if necessary. _split_fac_file is now imported from .base
        num_blocks_written, block_start_lines = _split_fac_file(
            input_filename, reader.output_prefix  # Use reader's output_prefix
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
            all_levels = reader.read(
                input_filename, block_files=[], block_starts=block_start_lines
            )
        else:  # Multiple blocks written or single block file needs reading (if num_blocks_written was 1 but block_files was populated somehow)
            all_levels = reader.read(
                input_filename, block_files=block_files, block_starts=block_start_lines
            )

    except Exception as e:
        logger.error(
            f"Error during read_fac execution for {input_filename}: {e}", exc_info=True
        )
        all_levels = pd.DataFrame()  # Ensure empty DF on error
    finally:
        # Cleanup temp files and restore log level using the reader instance if it was created
        if reader:
            reader._cleanup_temp_files()
            reader._restore_log_level()  # Restore log level

    return all_levels
