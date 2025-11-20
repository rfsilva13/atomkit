# atomkit/src/atomkit/readers/base.py

"""
Base class and utility functions for reading FAC output files.
"""

import logging
import os
import re
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.constants as const

from ..constants import *  # Import constants and allowed units
# Use absolute import based on package structure
# Assuming definitions and configuration are one level up
from ..structure import Configuration

# --- Setup Logging ---
# Use colorlog if available for nicer terminal output
try:
    import colorlog  # type: ignore[import-not-found]

    # Check if a handler already exists to prevent duplicates
    root_logger = logging.getLogger()
    if not any(isinstance(h, colorlog.StreamHandler) for h in root_logger.handlers):
        handler = colorlog.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(name)-15s %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        logger = colorlog.getLogger(__name__)
    else:
        # Logger already configured (e.g., by a parent application)
        logger = colorlog.getLogger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    # Basic configuration if no handlers are set up at all
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s")

# --- Utility Functions (Moved from original readers.py) ---


def data_optimize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes DataFrame memory usage by converting object columns to numeric
    where possible, excluding known string columns. Converts numeric columns
    to the smallest possible subtype (integer or float).

    Args:
        df: The pandas DataFrame to optimize.

    Returns:
        The optimized pandas DataFrame.
    """
    # Columns known to contain non-numeric string data
    known_string_cols = {
        "configuration",
        "label",
        "conf1",
        "term",
        "method",
        "conf_detail",
        "rel_config",
        "multipole",
        "multipole_type",
        "type",
        "lower_level_label",
        "upper_level_label",
        "configuration_lower",
        "configuration_upper",
    }
    # Iterate through columns with 'object' dtype
    for col in df.select_dtypes(include=["object"]).columns:
        if col in known_string_cols:
            continue  # Skip known string columns
        try:
            # Attempt conversion to numeric, coercing errors to NaN
            converted = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
            # Check if conversion resulted in any non-NaN values
            if not converted.isna().all():
                # Downcast integers or floats to smallest possible subtype
                if converted.dtype.kind in "iu":
                    df[col] = pd.to_numeric(converted, downcast="integer")
                elif converted.dtype.kind == "f":
                    df[col] = pd.to_numeric(converted, downcast="float")
                else:
                    df[col] = converted  # Keep as is if not int/float (e.g., boolean)
        except (ValueError, TypeError):
            # Ignore columns that cannot be converted
            pass
    return df


def J_PI_indexing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds the 'J_PI_index' column to the DataFrame.
    This index represents the order of levels within a group defined by
    the same total angular momentum (2J) and parity (P).

    Args:
        df: The pandas DataFrame containing level data. Must have '2j' and 'p' columns.

    Returns:
        The DataFrame with the 'J_PI_index' column added.
    """
    # Check if required columns exist
    if "2j" in df.columns and "p" in df.columns:
        # Attempt to convert columns to numeric, coercing errors
        j_numeric = pd.to_numeric(df["2j"], errors="coerce")
        p_numeric = pd.to_numeric(df["p"], errors="coerce")

        # Proceed only if both columns could be converted (at least partially)
        if j_numeric.notna().any() and p_numeric.notna().any():
            # Create a mask for rows where both J and P are valid numbers
            mask = j_numeric.notna() & p_numeric.notna()
            # Fill NaN values temporarily for grouping (use values outside typical range)
            j_filled = j_numeric.fillna(-1)
            p_filled = p_numeric.fillna(-1)

            if mask.any():  # If there are any valid rows to group
                # Calculate cumulative count within each (J, P) group for valid rows
                cumcount_result = (
                    df.loc[mask]
                    .groupby([j_filled[mask], p_filled[mask]], sort=False)
                    .cumcount()
                    + 1  # Add 1 to start indexing from 1
                )
                # Initialize the J_PI_index column with Pandas nullable integer type
                df["J_PI_index"] = pd.Series(dtype="Int64")
                # Assign the calculated index only to the valid rows
                df.loc[mask, "J_PI_index"] = cumcount_result
            else:
                # No valid (J, P) pairs found
                df["J_PI_index"] = pd.NA
        else:
            logger.warning(
                "Cannot calculate J_PI_index: '2j' or 'p' columns not convertible to numeric."
            )
            df["J_PI_index"] = pd.NA
    else:
        logger.warning("Cannot calculate J_PI_index: '2j' or 'p' columns not found.")
        df["J_PI_index"] = pd.NA
    return df


def _extract_header_info(
    file_path: str, reader_type: str = "level"
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extracts Z, Nele (for levels), and optionally Multipole (for transitions)
    from the header lines of a FAC output file.

    Args:
        file_path: Path to the FAC output file.
        reader_type: Type of reader ('level' or 'transition') to adjust
                     which information is essential.

    Returns:
        A tuple (atomic_number, nele, multipole).
        `nele` will be None if `reader_type` is 'transition'.
        `multipole` will be None if `reader_type` is 'level'.
        Values can be None if not found in the header.
    """
    atomic_number, nele, multipole = None, None, None
    try:
        with open(file_path, "r") as file:
            # Read a reasonable number of lines assuming header is near the top
            header_lines = [file.readline() for _ in range(50)]
            # Check if file was empty or unreadable
            if not any(header_lines):
                logger.error(f"File {file_path} is empty or unreadable.")
                return None, None, None

            # Process non-empty lines read
            for line in filter(None, header_lines):
                # Match Z = number (allows for float like 46.0)
                z_match = re.match(
                    r"^\s*[A-Za-z]{1,2}\s+Z\s*=\s*(\d+(\.\d+)?)", line.strip()
                )
                if z_match:
                    # Convert Z to integer
                    atomic_number = int(float(z_match.group(1)))

                # Only extract Nele for levels reader
                if reader_type == "level":
                    nele_match = re.match(r"^\s*NELE\s*=\s*(\d+)", line.strip())
                    if nele_match:
                        nele = int(nele_match.group(1))

                # Only extract Multipole for transitions reader
                if reader_type == "transition":
                    multip_match = re.match(r"^\s*MULTIP\s*=\s*(-?\d+)", line.strip())
                    if multip_match:
                        multipole = int(multip_match.group(1))

                # Check termination conditions based on reader type
                if atomic_number is not None:
                    if reader_type == "level" and nele is not None:
                        break  # Found both needed for levels
                    if reader_type == "transition" and multipole is not None:
                        # Found Z and MULTIP needed for transitions
                        break
                    if reader_type == "transition" and multipole is None:
                        # Found Z but still need MULTIP for transitions
                        continue

        # Log errors/warnings if essential info wasn't found
        if atomic_number is None:
            logger.error(f"Could not extract Z from header of {file_path}")
        if reader_type == "level" and nele is None:
            logger.error(f"Could not extract NELE from level header of {file_path}")
        if reader_type == "transition" and multipole is None:
            logger.warning(
                f"Could not extract MULTIP from transition header of {file_path}. Multipole type will be unknown."
            )

        # Return None for nele if it's a transition file or wasn't found for levels
        nele_to_return = nele if reader_type == "level" else None

        return atomic_number, nele_to_return, multipole

    except FileNotFoundError:
        logger.error(f"File not found while extracting header: {file_path}")
    except Exception as e:
        logger.error(f"Error processing header for {file_path}: {e}", exc_info=True)
    # Return None for all if any error occurs
    return None, None, None


def _split_fac_file(input_filename: str, output_prefix: str) -> tuple[int, list[int]]:
    """
    Splits a multi-block FAC file (where blocks start with 'NELE') into
    separate temporary files, preserving the original header in each.

    Args:
        input_filename: Path to the original FAC file.
        output_prefix: Prefix for the temporary output files (e.g., 'temp_block').
                       Files will be named like 'temp_block_1.txt', 'temp_block_2.txt'.

    Returns:
        A tuple: (number_of_blocks_written, list_of_original_start_lines).
        Returns (0, []) on error or if no data blocks found.
        Returns (1, [start_line]) if only one block is found (no files written).
    """
    header, blocks_content, current_block_lines, block_start_lines = [], [], [], []
    line_counter = 0
    first_nele_found = False
    try:
        with open(input_filename, "r") as infile:
            # --- Read Header ---
            # Read lines until the first 'NELE' line is found or data starts
            while True:
                line = infile.readline()
                line_counter += 1
                if not line:
                    logger.warning(
                        f"End of file reached before finding NELE or data in {input_filename}. Treating as single block (potentially empty)."
                    )
                    # If some lines were read as header, assume it's one block starting at line 1
                    return (1, [1]) if header else (0, [])

                # Check if the line marks the start of a new block
                if line.strip().startswith("NELE"):
                    first_nele_line = line
                    first_nele_found = True
                    break  # Found the first NELE line, header reading is done

                # Check for potential data lines before NELE (heuristic)
                # If a line starts with digits and we've read past the first line
                if re.match(r"^\s*\d", line) and line_counter > 1:
                    logger.warning(
                        f"Data line encountered before NELE in {input_filename}. Treating as single block starting around line {line_counter}."
                    )
                    # Treat as a single block starting conceptually at line 1
                    # The actual data start will be determined by the reader later
                    return (1, [1])

                # If not NELE or data, assume it's part of the header
                header.append(line)

            if not first_nele_found:  # Safety check (should be caught above)
                logger.error(
                    f"Logic error: NELE line not found but header loop exited in {input_filename}."
                )
                return (1, [1])  # Treat as single block

            # --- Process Blocks ---
            # Start the first block (includes header and the first NELE line)
            block_start_lines.append(line_counter)
            current_block_lines.extend(header + [first_nele_line])

            # Read the rest of the file line by line
            while True:
                line = infile.readline()
                if not line:
                    break  # End of file
                line_counter += 1

                # Check if a new block starts
                if line.strip().startswith("NELE"):
                    # Save the completed previous block if it contains data
                    # (more than just header + NELE line)
                    if len(current_block_lines) > len(header) + 1:
                        blocks_content.append(list(current_block_lines))
                    else:
                        logger.debug(
                            f"Skipping empty block ending before line {line_counter}."
                        )
                    # Start the new block
                    block_start_lines.append(line_counter)
                    current_block_lines = header.copy() + [line]
                elif line.strip():  # Add non-empty lines to the current block
                    current_block_lines.append(line)

            # Add the last block if it contains data
            if len(current_block_lines) > len(header) + 1:
                blocks_content.append(list(current_block_lines))
            else:
                logger.debug("Skipping empty last block ending at EOF.")

    except FileNotFoundError:
        logger.error(f"Input file not found during split: {input_filename}")
        return 0, []
    except Exception as e:
        logger.error(
            f"Error reading/processing {input_filename} during split: {e}",
            exc_info=True,
        )
        return 0, []

    # --- Final Check and File Writing ---
    num_data_blocks = len(blocks_content)

    if num_data_blocks == 0:
        if block_start_lines:  # NELE lines were found, but no data followed
            logger.warning(
                f"{len(block_start_lines)} NELE blocks detected in {input_filename}, but none contained data rows."
            )
        # If header exists, it's one block (header only), otherwise zero blocks
        return (1, [1]) if header else (0, [])

    if num_data_blocks == 1:
        # Only one block found, no need to write temp files
        logger.info(
            f"Only one data block found in {input_filename}. Reading original file directly."
        )
        # Return 1 block found, and the starting line number of that block
        return 1, block_start_lines[:1]

    # --- Write Multiple Blocks to Temporary Files ---
    num_blocks_written, written_block_indices = 0, []
    written_files = []  # Keep track of files actually written
    output_filename = ""  # Initialize to prevent "possibly unbound" error
    try:
        for i, block_lines_data in enumerate(blocks_content):
            output_filename = f"{output_prefix}_{i+1}.txt"
            with open(output_filename, "w") as outfile:
                outfile.writelines(block_lines_data)
            num_blocks_written += 1
            written_block_indices.append(i)
            written_files.append(output_filename)  # Add to list of written files
    except Exception as e:
        logger.error(f"Error writing block file {output_filename}: {e}", exc_info=True)
        # Attempt cleanup of any files already written
        for fname_to_remove in written_files:
            if os.path.exists(fname_to_remove):
                try:
                    os.remove(fname_to_remove)
                except Exception as cleanup_e:
                    logger.warning(
                        f"Failed to cleanup temp file {fname_to_remove}: {cleanup_e}"
                    )
        return 0, []  # Indicate failure

    # Return the count of blocks successfully written and their original start lines
    return num_blocks_written, [block_start_lines[idx] for idx in written_block_indices]


# --- Base Reader Class ---
class _BaseFacReader:
    """
    Base class containing common logic for FAC file readers.
    Handles logging setup, temporary file cleanup, and final column manipulations.
    """

    # Columns that should generally not be renamed by the user map
    _PROTECTED_RENAME_COLS = {"atomic_number", "ion_charge"}

    def __init__(self, verbose: int, output_prefix: str):
        """
        Initializes the base reader.

        Args:
            verbose: Logging verbosity level (0, 1, 2).
            output_prefix: Prefix for temporary files if splitting is needed.
        """
        self.verbose = verbose
        self.output_prefix = output_prefix
        self._temp_files_created: list[str] = []  # Track temp files

        # Set logger level based on verbosity
        current_logger = logging.getLogger(__name__)
        self._original_log_level = current_logger.level  # Store original level
        if self.verbose == 0:
            log_level = logging.WARNING
        elif self.verbose >= 2:
            log_level = logging.DEBUG
        else:  # verbose == 1
            log_level = logging.INFO
        current_logger.setLevel(log_level)
        # Ensure basic config if no handlers exist
        if not current_logger.hasHandlers() and not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=log_level, format="%(levelname)s:%(name)s: %(message)s"
            )

    def _cleanup_temp_files(self):
        """Removes any temporary files created during processing."""
        if self._temp_files_created:
            logger.info(
                f"Cleaning up {len(self._temp_files_created)} temporary block files..."
            )
            for temp_file in self._temp_files_created:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {temp_file}: {e}")
        self._temp_files_created = []  # Reset list

    def _restore_log_level(self):
        """Restores the original log level."""
        logging.getLogger(__name__).setLevel(self._original_log_level)

    def _apply_final_manipulations(
        self,
        df: pd.DataFrame,
        default_cols: list[str],
        user_cols_to_keep: Optional[list[str]],
        user_rename_map: Optional[dict[str, str]],
        include_method_flag: bool,
        optional_col_map: Optional[dict[str, bool]] = None,
    ) -> pd.DataFrame:
        """
        Applies final column renaming and selection based on user preferences
        and defaults.

        Args:
            df: The DataFrame to process.
            default_cols: List of default concise column names for this reader.
            user_cols_to_keep: Optional list of concise column names requested by user.
            user_rename_map: Optional dict mapping concise names to desired final names.
            include_method_flag: Whether the 'method' column should be included by default.
            optional_col_map: Optional dict mapping other optional concise column names
                              to boolean flags indicating if they should be included by default.

        Returns:
            The processed DataFrame with selected and renamed columns.
        """
        if df.empty:
            return df

        # --- 1. Prepare Rename Map ---
        # Start with user's map, remove protected columns with warnings
        final_rename_map = {}
        if user_rename_map:
            final_rename_map = user_rename_map.copy()
            for protected in self._PROTECTED_RENAME_COLS:
                if protected in final_rename_map:
                    target_name = final_rename_map.pop(protected)  # Remove from map
                    logger.warning(
                        f"Attempt to rename protected column '{protected}' to '{target_name}' was ignored."
                    )

        # --- 2. Apply Renaming ---
        df_renamed = df.copy()
        # Get current column and index names before renaming
        current_index_names_before_rename = (
            list(df_renamed.index.names)
            if df_renamed.index.nlevels > 1 or any(df_renamed.index.names)
            else []
        )
        current_columns_before_rename = list(df_renamed.columns)

        # Determine which renames apply to columns vs index
        actual_col_rename_map = {
            k: v
            for k, v in final_rename_map.items()
            if k in current_columns_before_rename
        }
        actual_index_rename_map = {
            k: v
            for k, v in final_rename_map.items()
            if k in current_index_names_before_rename
        }

        # Log warnings for requested renames that didn't match any existing column/index
        if user_rename_map:  # Log only if user provided a map
            for k_orig, k_new in user_rename_map.items():
                is_protected_and_attempted = (
                    k_orig in self._PROTECTED_RENAME_COLS and k_orig in user_rename_map
                )
                if (
                    not is_protected_and_attempted
                    and k_orig not in current_columns_before_rename
                    and k_orig not in current_index_names_before_rename
                ):
                    logger.warning(
                        f"Column/Index '{k_orig}' not found for renaming to '{k_new}'."
                    )

        # Perform renaming
        if actual_index_rename_map:
            # Cast to satisfy pandas type signature which expects dict[str | int, str]
            # Create a new dict with the union type to satisfy pandas strict type checking
            index_map: dict[str | int, str] = {
                k: v for k, v in actual_index_rename_map.items()
            }
            df_renamed.rename_axis(index=index_map, inplace=True)
        if actual_col_rename_map:
            df_renamed.rename(columns=actual_col_rename_map, inplace=True)

        # Get column/index names *after* renaming for selection step
        current_cols_after_rename = list(df_renamed.columns)
        current_index_after_rename = (
            list(df_renamed.index.names)
            if df_renamed.index.nlevels > 1 or any(df_renamed.index.names)
            else []
        )

        # --- 3. Determine Columns to Keep (using final names) ---
        final_cols_to_select_set = set()

        if user_cols_to_keep is not None:
            # User specified columns - prioritize this list
            for concise_name in user_cols_to_keep:
                final_name = final_rename_map.get(concise_name, concise_name)
                if final_name in current_cols_after_rename:
                    final_cols_to_select_set.add(final_name)
                elif final_name in current_index_after_rename:
                    pass  # Index columns are handled by index itself
                else:
                    logger.warning(
                        f"Requested column '{concise_name}' (final name '{final_name}') not found in DataFrame."
                    )
            # Ensure protected columns are always kept if they exist as columns
            for protected_col_orig_name in self._PROTECTED_RENAME_COLS:
                # Check using original name as it wouldn't be in final_rename_map if protected
                if protected_col_orig_name in current_cols_after_rename:
                    final_cols_to_select_set.add(protected_col_orig_name)

        else:
            # No specific columns requested by user - use defaults + optional flags
            # Add default columns if they exist (using their final names)
            for concise_name in default_cols:
                final_name = final_rename_map.get(concise_name, concise_name)
                if final_name in current_cols_after_rename:
                    final_cols_to_select_set.add(final_name)
                # Don't need to check index here, as defaults usually refer to data columns

            # Add 'method' if flagged and exists
            if include_method_flag:
                method_final_name = final_rename_map.get("method", "method")
                if method_final_name in current_cols_after_rename:
                    final_cols_to_select_set.add(method_final_name)

            # Add other optional columns if flagged and exist
            if optional_col_map:
                for col_concise, include_flag in optional_col_map.items():
                    if include_flag:
                        col_final = final_rename_map.get(col_concise, col_concise)
                        if col_final in current_cols_after_rename:
                            final_cols_to_select_set.add(col_final)

        # --- 4. Perform Column Selection ---
        # Filter the list of current columns based on the selection set
        cols_for_final_df = [
            col for col in current_cols_after_rename if col in final_cols_to_select_set
        ]

        # Try to preserve order based on user request or defaults
        if user_cols_to_keep is not None:
            # Base order on user's list (using final names)
            ordered_basis = [final_rename_map.get(c, c) for c in user_cols_to_keep]
            # Add protected columns if they weren't in user list but exist
            for protected_col in self._PROTECTED_RENAME_COLS:
                if (
                    protected_col in current_cols_after_rename
                    and protected_col not in ordered_basis
                ):
                    # Add protected columns (using original name as key)
                    ordered_basis.append(protected_col)
        else:
            # Base order on default list + included optional columns
            ordered_basis = [final_rename_map.get(c, c) for c in default_cols]
            if include_method_flag:
                method_final_name = final_rename_map.get("method", "method")
                if (
                    method_final_name not in ordered_basis
                    and method_final_name in cols_for_final_df
                ):
                    ordered_basis.append(method_final_name)
            if optional_col_map:
                for col_concise, include_flag in optional_col_map.items():
                    if include_flag:
                        col_final = final_rename_map.get(col_concise, col_concise)
                        if (
                            col_final not in ordered_basis
                            and col_final in cols_for_final_df
                        ):
                            ordered_basis.append(col_final)

        # Filter the ordered list to include only columns actually present in cols_for_final_df
        final_ordered_cols = [col for col in ordered_basis if col in cols_for_final_df]
        # Add any remaining selected columns that weren't in the ordered basis (preserves all selected)
        final_ordered_cols.extend(
            [col for col in cols_for_final_df if col not in final_ordered_cols]
        )

        # Select columns in the desired order
        final_df = df_renamed[final_ordered_cols]

        return final_df
