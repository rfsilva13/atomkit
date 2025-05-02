# atomkit/src/atomkit/readers.py

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.constants as const

# Use absolute import based on package structure
from .configuration import Configuration

# --- Constants ---
# Conversion factor from eV to cm⁻¹ (CODATA 2018)
EV_TO_CM1 = 8065.544004795713
# Planck's constant * speed of light (for eV to Angstrom conversion)
HC_EV_ANGSTROM = 12398.419843320027  # eV·Å
# Constant for Line Strength calculation S(a.u.) = (gf * lambda(A)) / LINE_STRENGTH_CONST
LINE_STRENGTH_CONST = 303.756
# Hartree energy in eV
HARTREE_EV = const.physical_constants["Hartree energy in eV"][0]
# Rydberg constant times hc in eV
RYD_EV = const.physical_constants["Rydberg constant times hc in eV"][0]

# --- Allowed Units ---
ALLOWED_ENERGY_UNITS = ["ev", "cm-1", "ry", "hz", "j", "ha"]
ALLOWED_WAVELENGTH_UNITS = ["a", "nm", "m", "cm"]

# --- Setup Logging ---
# (Logging setup remains the same)
try:
    import colorlog

    root_logger = logging.getLogger()
    # Prevent adding duplicate handlers
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
        # Default level set in __init__ based on verbose flag
        logger = colorlog.getLogger(__name__)
    else:
        logger = colorlog.getLogger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    if not logging.getLogger().hasHandlers():
        # Default level set in __init__ based on verbose flag
        logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s")
        # logger.info("colorlog library not found, using standard logging.")


# --- Utility Functions ---


def data_optimize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes DataFrame memory usage by converting object columns to numeric
    where possible, excluding known string columns.
    """
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
        "type",  # Added type
    }

    for col in df.select_dtypes(include=["object"]).columns:
        if col in known_string_cols:
            continue
        try:
            converted = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
            if not converted.isna().all():
                if converted.dtype.kind in "iu":
                    df[col] = pd.to_numeric(converted, downcast="integer")
                elif converted.dtype.kind == "f":
                    df[col] = pd.to_numeric(converted, downcast="float")
                else:
                    df[col] = converted
        except (ValueError, TypeError):
            pass
    return df


def J_PI_indexing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds the 'J_PI_index' column based on '2j' and 'p' columns.
    """
    if "2j" in df.columns and "p" in df.columns:
        j_numeric = pd.to_numeric(df["2j"], errors="coerce")
        p_numeric = pd.to_numeric(df["p"], errors="coerce")
        if j_numeric.notna().any() and p_numeric.notna().any():
            mask = j_numeric.notna() & p_numeric.notna()
            j_filled = j_numeric.fillna(-1)
            p_filled = p_numeric.fillna(-1)
            if mask.any():
                cumcount_result = (
                    df.loc[mask]
                    .groupby([j_filled[mask], p_filled[mask]], sort=False)
                    .cumcount()
                    + 1
                )
                df["J_PI_index"] = pd.Series(dtype="Int64")
                df.loc[mask, "J_PI_index"] = cumcount_result
            else:
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
    file_path: str, reader_type: str = "level"  # 'level' or 'transition'
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extracts Z, Nele, and optionally Multipole from the header.
    """
    atomic_number = None
    nele = None
    multipole = None
    try:
        with open(file_path, "r") as file:
            lines_read = 0
            max_header_lines = 50
            header_lines = []
            for _ in range(max_header_lines):
                line = file.readline()
                if not line:
                    break
                header_lines.append(line)
                lines_read += 1

            if lines_read == 0:
                logger.error(f"File {file_path} is empty.")
                return None, None, None

            for line in header_lines:
                z_match = re.match(
                    r"^\s*[A-Za-z]{1,2}\s+Z\s*=\s*(\d+(\.\d+)?)", line.strip()
                )
                if z_match:
                    atomic_number = int(float(z_match.group(1)))

                nele_match = re.match(r"^\s*NELE\s*=\s*(\d+)", line.strip())
                if nele_match:
                    nele = int(nele_match.group(1))

                # --- Only look for MULTIP if it's a transition file ---
                if reader_type == "transition":
                    multip_match = re.match(r"^\s*MULTIP\s*=\s*(-?\d+)", line.strip())
                    if multip_match:
                        multipole = int(multip_match.group(1))
                # --- End MULTIP check ---

                if atomic_number is not None and nele is not None:
                    # If it's a transition file, keep looking for MULTIP
                    if reader_type == "transition" and multipole is None:
                        continue
                    # Otherwise (level file or MULTIP found for transition), break
                    break

        if atomic_number is None:
            logger.error(f"Could not extract Z from header of {file_path}")
        if nele is None:
            logger.error(f"Could not extract NELE from header of {file_path}")
        if reader_type == "transition" and multipole is None:
            logger.warning(
                f"Could not extract MULTIP from header of {file_path}. Multipole type will be unknown."
            )

        if atomic_number is None or nele is None:
            return None, None, multipole
        else:
            return atomic_number, nele, multipole

    except FileNotFoundError:
        logger.error(f"File not found while extracting header: {file_path}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error processing header for {file_path}: {e}", exc_info=True)
        return None, None, None


def _split_fac_file(input_filename: str, output_prefix: str) -> Tuple[int, List[int]]:
    """
    Splits a FAC file into multiple block files based on 'NELE' lines.
    Internal helper function.
    """
    # (Function remains the same as previous version)
    header = []
    blocks_content = []
    current_block_lines = []
    block_start_lines = []
    temp_files_created = []  # Track files created by this specific call
    line_counter = 0

    try:
        with open(input_filename, "r") as infile:
            # 1. Read header lines
            while True:
                line = infile.readline()
                line_counter += 1
                if not line:
                    logger.warning(
                        f"No 'NELE' line found in {input_filename}. Treating as single block."
                    )
                    return 1, []
                if line.startswith("NELE"):
                    break
                header.append(line)

            # 2. Process first block
            first_nele_line = line
            block_start_lines.append(line_counter)
            current_block_lines.extend(header)
            current_block_lines.append(first_nele_line)

            # 3. Process rest of the file
            while True:
                line = infile.readline()
                if not line:
                    break
                line_counter += 1

                if line.startswith("NELE"):
                    # Only append if the block has more than just header+NELE line
                    if len(current_block_lines) > len(header) + 1:
                        blocks_content.append(list(current_block_lines))  # Append copy
                    else:
                        logger.debug(
                            f"Skipping empty block ending before line {line_counter}."
                        )

                    block_start_lines.append(line_counter)
                    current_block_lines = header.copy()
                    current_block_lines.append(line)
                else:
                    # Only add non-empty data lines
                    if line.strip():
                        current_block_lines.append(line)

            # 4. Append the last block if it contains data
            if len(current_block_lines) > len(header) + 1:
                blocks_content.append(list(current_block_lines))
            else:
                logger.debug(f"Skipping empty last block ending at EOF.")

    except FileNotFoundError:
        logger.error(f"Input file not found during split: {input_filename}")
        return 0, []
    except Exception as e:
        logger.error(
            f"Error reading or processing {input_filename} during split: {e}",
            exc_info=True,
        )
        return 0, []

    num_blocks_found = len(block_start_lines)
    num_data_blocks = len(blocks_content)

    if num_data_blocks == 0:
        if num_blocks_found > 0:
            logger.warning(
                f"{num_blocks_found} NELE blocks detected, but none contained data rows."
            )
            return 1, block_start_lines
        else:
            return 1, []

    if num_data_blocks == 1:
        logger.info("Only one data block found. Reading original file directly.")
        return 1, block_start_lines

    # 5. Write blocks with data to temporary files
    num_blocks_written = 0
    written_block_indices = []
    for i, block_lines in enumerate(blocks_content):
        output_filename = f"{output_prefix}_{i+1}.txt"
        try:
            with open(output_filename, "w") as outfile:
                outfile.writelines(block_lines)
            temp_files_created.append(output_filename)
            num_blocks_written += 1
            written_block_indices.append(i)
        except Exception as e:
            logger.error(
                f"Error writing block file {output_filename}: {e}", exc_info=True
            )
            for f in temp_files_created:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            return 0, []

    written_start_lines = [block_start_lines[idx] for idx in written_block_indices]
    return num_blocks_written, written_start_lines


# --- FAC Level File Reader Class ---


class FacReader:
    """
    Reads and processes FAC energy level files (.lev.asc), handling multi-block structures.
    """

    # --- UPDATED Default Columns (Concise) ---
    _DEFAULT_OUTPUT_COLUMNS = [
        "atomic_number",
        "ion_charge",
        "level_index",
        "E",
        "p",
        "2j",
        "J_PI_index",
        "conf_detail",
        "configuration",
        "rel_config",
        "term",
        "label",
        "method",  # new_config removed from default
    ]
    _INDEX_COLUMNS = ["atomic_number", "ion_charge", "2j", "p", "J_PI_index"]

    def __init__(
        self,
        conf1: bool = False,  # Placeholder for potential future use
        energy_unit: str = "ev",  # Target unit for energy output
        columns_to_keep: Optional[List[str]] = None,
        rename_columns: Optional[Dict[str, str]] = None,
        output_prefix: str = "temp_fac_lev_block",
        verbose: int = 1,  # Default verbose=1 (INFO)
        include_method: bool = False,
        include_new_config: bool = False,  # Default False
    ):
        """Initializes the FacReader."""
        self.conf1 = conf1
        self.energy_unit = energy_unit.lower().replace("[", "").replace("]", "")
        if self.energy_unit not in ALLOWED_ENERGY_UNITS:
            raise ValueError(
                f"Invalid energy_unit '{energy_unit}'. Allowed: {ALLOWED_ENERGY_UNITS}"
            )
        self.columns_to_keep = columns_to_keep
        self.rename_columns = rename_columns
        self.output_prefix = output_prefix
        self.verbose = verbose
        self.include_method = include_method
        self.include_new_config = include_new_config  # Store this option
        self._temp_files_created = []

        # Set logger level based on verbosity
        current_logger = logging.getLogger(__name__)
        self._original_log_level = current_logger.level
        if self.verbose == 1:
            log_level = logging.INFO
        elif self.verbose >= 2:
            log_level = logging.DEBUG
        else:
            log_level = logging.WARNING  # verbose=0

        current_logger.setLevel(log_level)
        # Ensure logger has a handler if none exists
        if not current_logger.hasHandlers() and not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=log_level, format="%(levelname)s:%(name)s: %(message)s"
            )

    def read(self, input_filename: str) -> pd.DataFrame:
        """Reads the specified FAC level file."""
        all_levels = pd.DataFrame()
        self._temp_files_created = []

        current_logger = logging.getLogger(__name__)
        original_level = current_logger.level
        if self.verbose == 1:
            log_level = logging.INFO
        elif self.verbose >= 2:
            log_level = logging.DEBUG
        else:
            log_level = logging.WARNING
        current_logger.setLevel(log_level)

        if not os.path.exists(input_filename):
            logger.error(f"Input file not found: {input_filename}")
            current_logger.setLevel(original_level)
            return pd.DataFrame()

        try:
            logger.info(f"Processing level file: {input_filename}")
            num_blocks_written, block_start_lines = _split_fac_file(
                input_filename, self.output_prefix
            )
            if num_blocks_written > 1:
                self._temp_files_created = [
                    f"{self.output_prefix}_{i+1}.txt" for i in range(num_blocks_written)
                ]

            if num_blocks_written == 0:
                logger.error(f"Error during file splitting for {input_filename}.")
                all_levels = pd.DataFrame()
            elif num_blocks_written == 1:
                logger.info(
                    f"File {input_filename} contains a single block. Reading directly."
                )
                start_line_for_single = block_start_lines[0] if block_start_lines else 1
                all_levels = self._read_fac_block_data(
                    input_filename, original_start_line=start_line_for_single
                )
            else:
                logger.info(
                    f"Detected {num_blocks_written} data blocks starting at lines: {block_start_lines}. Reading split files..."
                )
                levels_list = []
                for i in range(num_blocks_written):
                    block_index = i + 1
                    block_filename = f"{self.output_prefix}_{block_index}.txt"
                    original_start_line = (
                        block_start_lines[i]
                        if i < len(block_start_lines)
                        else "Unknown"
                    )
                    logger.info(
                        f"--- Reading Level Block {block_index} (Original Line: {original_start_line}) ---"
                    )
                    levels_temp = self._read_fac_block_data(
                        block_filename, original_start_line
                    )
                    if not levels_temp.empty:
                        levels_list.append(levels_temp)
                    else:
                        logger.info(
                            f"Skipping level block {block_index} due to read/processing errors."
                        )

                if levels_list:
                    logger.info(
                        f"--- Successfully read {len(levels_list)} level blocks. Concatenating... ---"
                    )
                    all_levels = pd.concat(
                        [
                            (
                                df.reset_index()
                                if df.index.nlevels > 1
                                or any(name is not None for name in df.index.names)
                                else df
                            )
                            for df in levels_list
                        ],
                        ignore_index=True,
                    )
                    all_levels = self._post_process_concatenated(all_levels)
                else:
                    logger.error(
                        f"No level blocks could be successfully read from {input_filename} after splitting."
                    )
                    all_levels = pd.DataFrame()
        except Exception as e:
            logger.error(
                f"An unexpected error occurred in FacReader.read: {e}", exc_info=True
            )
            all_levels = pd.DataFrame()
        finally:
            self._cleanup_temp_files()
            current_logger.setLevel(original_level)

        all_levels = self._apply_final_manipulations(all_levels)

        if logger.isEnabledFor(logging.INFO):
            if not all_levels.empty:
                cols_display = list(all_levels.columns)
                idx_display = list(all_levels.index.names)
                logger.info(
                    f"Finished level processing. Returning DataFrame with {len(all_levels)} rows."
                )
                if not all(name is None for name in idx_display):
                    logger.info(f"Index: {idx_display}")
                logger.info(f"Columns: {cols_display}")
            elif os.path.exists(input_filename):
                logger.info(
                    f"Finished level processing {input_filename}, but resulted in an empty DataFrame."
                )

        return all_levels

    def _read_fac_block_data(
        self, file_levels: str, original_start_line: Optional[int] = None
    ) -> pd.DataFrame:
        """Reads a single FAC level file/block."""
        atomic_number, nele, _ = _extract_header_info(
            file_levels, reader_type="level"
        )  # Specify type

        if atomic_number is None or nele is None:
            logger.warning(
                f"Failed to extract header info from {file_levels}. Skipping block."
            )
            return pd.DataFrame()

        ion_charge = atomic_number - nele
        levels_df = self._process_fac_levels(
            atomic_number, ion_charge, file_levels, original_start_line
        )

        if levels_df.empty:
            logger.warning(
                f"Processing resulted in empty DataFrame for level block from original line {original_start_line or '?'}."
            )

        return levels_df

    def _find_data_start_row(self, lines: List[str]) -> Tuple[int, str]:
        """Finds the start row for level data."""
        # (Function remains the same)
        header_pattern = re.compile(r"\s*ILEV\s+IBASE\s+ENERGY\s+P\s+VNL\s+2J")
        for i, line in enumerate(lines):
            if header_pattern.match(line):
                skip_rows_count = i + 1
                message = f"Found specific level header line ('ILEV IBASE ENERGY...'). Determined skiprows={skip_rows_count}."
                return skip_rows_count, message

        message = "Specific level header line not found. Using fallback: searching for first line starting with a digit."
        for i, line in enumerate(lines):
            if re.match(r"^\s*\d+", line):
                if i > 0 and not re.match(r"^\s*\d+", lines[i - 1]):
                    skip_rows_count = i
                    message += f" Found potential data start at line {i+1}. Determined skiprows={skip_rows_count}."
                    return skip_rows_count, message
                elif i == 0:
                    skip_rows_count = i
                    message += f" Found potential data start at line {i+1}. Determined skiprows={skip_rows_count}."
                    return skip_rows_count, message

        message = "Could not determine level data start row. Using default guess."
        nele_index = next((i for i, l in enumerate(lines) if l.startswith("NELE")), -1)
        skip_rows_count = nele_index + 3 if nele_index != -1 else 12
        message += f" Determined skiprows={skip_rows_count}."
        return skip_rows_count, message

    def _process_fac_levels(
        self,
        atomic_number: int,
        ion_charge: int,
        file_levels: str,
        original_start_line: Optional[int] = None,
    ) -> pd.DataFrame:
        """Processes data rows from a single FAC level file/block."""
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

        skiprows, skip_message = self._find_data_start_row(lines)
        logger.info(f"{skip_message} {file_context}")

        levels = pd.DataFrame()
        # Use concise names for reading
        observed_col_names = [
            "level_index",
            "IBASE",
            "E",
            "p",
            "VNL",
            "2j",
            "conf_detail",
            "configuration",
            "label",
        ]
        try:
            levels = pd.read_csv(
                file_levels,
                sep=r"\s+",
                names=observed_col_names,
                skiprows=skiprows,
                on_bad_lines="warn",
                engine="python",
                comment="=",
                skipinitialspace=True,
            )

            essential_cols = ["level_index", "E", "p", "2j"]
            if not all(col in levels.columns for col in essential_cols):
                raise ValueError(
                    f"Essential level columns missing after CSV read: {essential_cols}"
                )

            for col in essential_cols:
                levels[col] = pd.to_numeric(levels[col], errors="coerce")
            initial_rows = len(levels)
            levels.dropna(subset=essential_cols, inplace=True)
            if len(levels) < initial_rows:
                logger.info(
                    f"Dropped {initial_rows - len(levels)} rows due to non-numeric essential level data {file_context}."
                )
            if levels.empty:
                logger.warning(
                    f"No valid level data rows found after cleaning essential columns {file_context}."
                )
                return pd.DataFrame()

            for col in ["conf_detail", "configuration", "label", "conf1"]:
                if col not in levels.columns:
                    levels[col] = "-"

            if "configuration" in levels.columns and "label" in levels.columns:
                if (
                    levels["label"].isin(["-", np.nan]).all()
                    and levels["configuration"]
                    .astype(str)
                    .str.contains(r"\s\S+$")
                    .any()
                ):
                    logger.debug(
                        f"Attempting to split potentially merged level 'configuration' and 'label' {file_context}."
                    )
                    split_conf = (
                        levels["configuration"].astype(str).str.rsplit(n=1, expand=True)
                    )
                    if split_conf.shape[1] == 2:
                        levels["configuration"] = split_conf[0].str.strip()
                        levels["label"] = split_conf[1].str.strip()
                    else:
                        logger.warning(
                            f"Splitting level configuration/label failed {file_context}."
                        )

        except Exception as e_csv:
            logger.warning(
                f"Flexible CSV reading failed for levels {file_context}: {e_csv}. Trying fixed-width."
            )
            # (Fixed-width fallback remains the same)
            try:
                widths = [
                    (0, 7),
                    (7, 14),
                    (14, 30),
                    (30, 31),
                    (32, 38),
                    (38, 43),
                    (43, 70),
                    (70, 100),
                    (100, 150),
                ]
                names = [
                    "level_index",
                    "IBASE",
                    "E",
                    "p",
                    "VNL",
                    "2j",
                    "conf_detail",
                    "configuration",
                    "label",
                ]
                levels = pd.read_fwf(
                    file_levels, colspecs=widths, names=names, skiprows=skiprows
                )
                levels.dropna(subset=["level_index", "E", "p", "2j"], inplace=True)
                if levels.empty:
                    raise ValueError("No valid data after FWF cleaning.")
                for col in ["conf_detail", "configuration", "label"]:
                    if col not in levels.columns:
                        levels[col] = "-"
                levels["conf1"] = "-"
            except Exception as e_fwf:
                logger.error(
                    f"Fixed-width reading also failed for levels {file_context}: {e_fwf}",
                    exc_info=True,
                )
                return pd.DataFrame()

        try:
            # Convert types
            levels["level_index"] = levels["level_index"].astype(int)
            levels["p"] = levels["p"].astype(int)
            levels["2j"] = levels["2j"].astype(int)
            levels["E"] = pd.to_numeric(levels["E"], errors="coerce")  # Read as eV

            for col in ["conf_detail", "configuration", "label", "conf1"]:
                if col in levels.columns:
                    levels[col] = levels[col].astype(str).str.strip().fillna("-")

            # Parse rel_config
            if "label" in levels.columns:
                orbital_pattern = re.compile(r"(\d+[a-zA-Z][+-]?\d*)")

                def extract_rel_config_findall(label_str):
                    if not isinstance(label_str, str):
                        return "-"
                    matches = orbital_pattern.findall(label_str)
                    return (
                        ".".join(matches)
                        if matches
                        else (label_str if label_str != "-" else "-")
                    )

                levels["rel_config"] = levels["label"].apply(extract_rel_config_findall)
                logger.debug(f"Parsed 'rel_config' column from 'label' {file_context}.")
            else:
                levels["rel_config"] = "-"

            # Create Configuration objects
            if "conf_detail" in levels.columns:

                def safe_from_compact(compact_str):
                    try:
                        return Configuration.from_compact_string(compact_str)
                    except Exception as e:
                        logger.debug(
                            f"Could not parse compact string '{compact_str}' into Configuration object: {e}"
                        )
                        return None

                levels["new_config"] = levels["conf_detail"].apply(safe_from_compact)
                logger.debug(
                    f"Created 'new_config' column with Configuration objects {file_context}."
                )
            else:
                levels["new_config"] = None

            levels["term"] = "-"
            levels = data_optimize(levels)

            # --- Unit Conversion for Energy ---
            if "E" in levels.columns and pd.api.types.is_numeric_dtype(levels["E"]):
                energy_ev = levels["E"].copy()  # Keep original eV
                target_unit = self.energy_unit
                if target_unit == "cm-1":
                    levels["E"] = energy_ev * EV_TO_CM1
                elif target_unit == "ry":
                    levels["E"] = energy_ev / RYD_EV
                elif target_unit == "hz":
                    levels["E"] = energy_ev * const.e / const.h
                elif target_unit == "j":
                    levels["E"] = energy_ev * const.e
                elif target_unit == "ha":
                    levels["E"] = energy_ev / HARTREE_EV
                elif target_unit != "ev":
                    logger.warning(
                        f"Energy unit '{target_unit}' not recognized, keeping eV."
                    )
            # --- End Unit Conversion ---

            levels["atomic_number"] = atomic_number
            levels["ion_charge"] = ion_charge
            levels["method"] = "FAC"
            # levels["priority"] = 10 # Removed

            levels = J_PI_indexing(levels)

            if "configuration" in levels.columns:
                levels["configuration"] = levels["configuration"].apply(
                    lambda x: (
                        re.sub(r"(?<=[a-zA-Z])1(?!\d)", "", x)
                        if isinstance(x, str)
                        else x
                    )
                )

            # --- Column Selection/Reordering (using concise names) ---
            # Define default columns to show
            default_display_cols = [
                "atomic_number",
                "ion_charge",
                "level_index",
                "E",
                "p",
                "2j",
                "J_PI_index",
                "conf_detail",
                "configuration",
                "rel_config",
                "term",
                "label",
            ]
            # Add optional columns
            if self.include_method:
                default_display_cols.append("method")
            if self.include_new_config:
                default_display_cols.append(
                    "new_config"
                )  # Conditionally add new_config
            if (
                self.conf1
                and "conf1" in levels.columns
                and levels["conf1"].ne("-").any()
            ):
                try:
                    insert_before_col = (
                        "configuration"
                        if "configuration" in default_display_cols
                        else "rel_config"
                    )
                    conf_index = default_display_cols.index(insert_before_col)
                    default_display_cols.insert(conf_index, "conf1")
                except ValueError:
                    default_display_cols.append("conf1")

            # Filter to columns actually present in the DataFrame
            final_columns_present = [
                col for col in default_display_cols if col in levels.columns
            ]

            levels = levels[final_columns_present]
            # --- End Column Selection ---

            levels = data_optimize(levels)
            if "E" in levels.columns:
                levels = levels.sort_values(by=["E"])

            # --- Set Index ---
            index_cols_present = [
                col for col in self._INDEX_COLUMNS if col in levels.columns
            ]
            if len(index_cols_present) == len(self._INDEX_COLUMNS):
                try:
                    levels.set_index(index_cols_present, inplace=True)
                except Exception as e:
                    logger.error(
                        f"Error setting level index {file_context}: {e}. Keeping unindexed.",
                        exc_info=True,
                    )
            else:
                missing_idx_cols = [
                    col for col in self._INDEX_COLUMNS if col not in levels.columns
                ]
                logger.warning(
                    f"Could not set level multi-index {file_context} due to missing columns: {missing_idx_cols}. Keeping unindexed."
                )

        except Exception as e:
            logger.error(
                f"Error during level post-processing {file_context}: {e}", exc_info=True
            )
            return pd.DataFrame()

        return levels

    def _post_process_concatenated(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-processes concatenated level DataFrame."""
        if df.empty:
            return df
        df = data_optimize(df)

        sort_cols = [
            col for col in ["atomic_number", "ion_charge", "E"] if col in df.columns
        ]
        if sort_cols:
            if "E" in df.columns:
                df["E"] = pd.to_numeric(df["E"], errors="coerce")
            df.sort_values(by=sort_cols, inplace=True, na_position="last")

        index_cols_present = [col for col in self._INDEX_COLUMNS if col in df.columns]
        if len(index_cols_present) == len(self._INDEX_COLUMNS):
            try:
                df.set_index(index_cols_present, inplace=True)
            except Exception as e:
                logger.warning(
                    f"Could not set multi-index after concatenating level blocks: {e}. Keeping sorted but unindexed."
                )
        else:
            logger.warning(
                "Index columns missing after level concatenation. Keeping sorted but unindexed."
            )

        return df

    def _apply_final_manipulations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies final column keeping and renaming for levels."""
        if df.empty:
            return df

        # --- Handle Renaming First (using concise internal names) ---
        rename_map_internal = {}
        # Determine current columns (check if indexed)
        was_indexed = df.index.nlevels > 1 or any(
            name is not None for name in df.index.names
        )
        current_columns = list(df.columns)
        current_index_names = list(df.index.names) if was_indexed else []
        all_current_names = current_columns + current_index_names

        if self.rename_columns:
            for concise_key, new_name in self.rename_columns.items():
                if concise_key in all_current_names:
                    rename_map_internal[concise_key] = new_name
                else:
                    logger.warning(
                        f"Column/Index '{concise_key}' not found for renaming in level data."
                    )

        # Apply renaming
        df_renamed = df.copy()  # Avoid modifying original df passed to function
        if rename_map_internal:
            index_rename_map = {
                k: v for k, v in rename_map_internal.items() if k in current_index_names
            }
            col_rename_map = {
                k: v for k, v in rename_map_internal.items() if k in current_columns
            }
            if index_rename_map:
                df_renamed.rename_axis(index=index_rename_map, inplace=True)
            if col_rename_map:
                df_renamed.rename(columns=col_rename_map, inplace=True)

        # --- Handle Column Keeping (using potentially renamed concise names) ---
        # Get current columns and index names after potential rename
        current_cols_after_rename = list(df_renamed.columns)
        current_index_after_rename = list(df_renamed.index.names)
        final_df = df_renamed  # Start with potentially renamed df

        if self.columns_to_keep is not None:
            # User provides concise names in columns_to_keep
            requested_concise_keep = set(self.columns_to_keep)

            # Map requested concise names to their potentially renamed versions
            target_names_to_keep = set()
            for concise_name in requested_concise_keep:
                target_names_to_keep.add(
                    rename_map_internal.get(concise_name, concise_name)
                )

            # Always keep index columns
            target_names_to_keep.update(current_index_after_rename)

            # Filter current columns based on the target names
            cols_to_actually_keep = [
                col for col in current_cols_after_rename if col in target_names_to_keep
            ]
            index_levels_to_keep = [
                name
                for name in current_index_after_rename
                if name in target_names_to_keep
            ]

            missing_cols = (
                target_names_to_keep
                - set(cols_to_actually_keep)
                - set(index_levels_to_keep)
            )
            # Only warn if the missing column wasn't an index level (which are handled separately)
            missing_cols_non_index = [
                c for c in missing_cols if c not in current_index_after_rename
            ]
            if missing_cols_non_index:
                logger.warning(
                    f"Requested level columns to keep not found or not available after renaming: {missing_cols_non_index}"
                )

            if not cols_to_actually_keep and not index_levels_to_keep:
                logger.warning(
                    "columns_to_keep resulted in no columns or index levels being selected. Returning original."
                )
                final_df = df_renamed  # Revert to df after renaming
            else:
                # Select columns, keeping the potentially modified index
                if df_renamed.index.nlevels > 1 or any(
                    name is not None for name in df_renamed.index.names
                ):
                    # Check if all requested index levels are still present
                    if set(index_levels_to_keep) == set(current_index_after_rename):
                        final_df = df_renamed[
                            cols_to_actually_keep
                        ]  # Select columns, index remains
                    else:
                        # If index levels were removed, reset and select
                        logger.debug(
                            "Index levels modified by columns_to_keep. Resetting index."
                        )
                        final_df = df_renamed.reset_index()
                        # Combine kept columns and kept index levels for final selection
                        final_selection = cols_to_actually_keep + index_levels_to_keep
                        # Ensure no duplicates and columns exist
                        final_selection = [
                            c for c in final_selection if c in final_df.columns
                        ]
                        final_df = final_df[
                            list(dict.fromkeys(final_selection))
                        ]  # Keep order, remove duplicates
                else:
                    # If not indexed, just select columns
                    final_df = df_renamed[cols_to_actually_keep]

        # Remove internal 'new_config' unless explicitly kept OR renamed and kept
        new_config_final_name = rename_map_internal.get("new_config", "new_config")
        # Check if 'new_config' was requested *by its original name*
        should_keep_new_config = (
            self.columns_to_keep is not None and "new_config" in self.columns_to_keep
        )

        if (
            new_config_final_name in final_df.columns
            and not self.include_new_config
            and not should_keep_new_config
        ):
            final_df = final_df.drop(columns=[new_config_final_name])

        return final_df

    def _cleanup_temp_files(self):
        """Removes temporary files created by this instance."""
        if self._temp_files_created:
            logger.info(
                f"Cleaning up {len(self._temp_files_created)} temporary level block files..."
            )
            for temp_file in self._temp_files_created:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.warning(
                            f"Could not remove temporary file {temp_file}: {e}"
                        )
        self._temp_files_created = []


# --- FAC Transition File Reader Class ---


class FacTransitionReader:
    """
    Reads and processes FAC transition files (.tr.asc), handling multi-block structures.
    """

    # --- UPDATED Default Columns (Concise) ---
    _DEFAULT_OUTPUT_COLUMNS = [
        "atomic_number",
        "ion_charge",
        "level_index_lower",
        "level_index_upper",
        "2j_lower",
        "2j_upper",
        "E",
        "lambda",
        "gf",
        "A",  # Concise names
        "S",
        "multipole",
        "type",
        "method",  # Concise names
    ]
    _INDEX_COLUMNS = [
        "atomic_number",
        "ion_charge",
        "level_index_upper",
        "level_index_lower",
    ]

    def __init__(
        self,
        energy_unit: str = "ev",  # Target unit for energy output
        wavelength_unit: str = "a",  # Target unit for wavelength output
        columns_to_keep: Optional[List[str]] = None,
        rename_columns: Optional[Dict[str, str]] = None,
        output_prefix: str = "temp_fac_tr_block",
        verbose: int = 1,  # Default verbose=1 (INFO)
        include_method: bool = False,
    ):
        """Initializes the FacTransitionReader."""
        self.energy_unit = energy_unit.lower().replace("[", "").replace("]", "")
        self.wavelength_unit = wavelength_unit.lower().replace("[", "").replace("]", "")
        if self.energy_unit not in ALLOWED_ENERGY_UNITS:
            raise ValueError(
                f"Invalid energy_unit '{energy_unit}'. Allowed: {ALLOWED_ENERGY_UNITS}"
            )
        if self.wavelength_unit not in ALLOWED_WAVELENGTH_UNITS:
            raise ValueError(
                f"Invalid wavelength_unit '{wavelength_unit}'. Allowed: {ALLOWED_WAVELENGTH_UNITS}"
            )

        self.columns_to_keep = columns_to_keep
        self.rename_columns = rename_columns
        self.output_prefix = output_prefix
        self.verbose = verbose
        self.include_method = include_method
        self._temp_files_created = []

        # Set logger level based on verbosity
        current_logger = logging.getLogger(__name__)
        self._original_log_level = current_logger.level
        if self.verbose == 1:
            log_level = logging.INFO
        elif self.verbose >= 2:
            log_level = logging.DEBUG
        else:
            log_level = logging.WARNING  # verbose=0

        current_logger.setLevel(log_level)
        # Ensure logger has a handler if none exists
        if not current_logger.hasHandlers() and not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=log_level, format="%(levelname)s:%(name)s: %(message)s"
            )

    def read(self, input_filename: str) -> pd.DataFrame:
        """Reads the specified FAC transition file."""
        all_transitions = pd.DataFrame()
        self._temp_files_created = []

        current_logger = logging.getLogger(__name__)
        original_level = current_logger.level
        if self.verbose == 1:
            log_level = logging.INFO
        elif self.verbose >= 2:
            log_level = logging.DEBUG
        else:
            log_level = logging.WARNING
        current_logger.setLevel(log_level)

        if not os.path.exists(input_filename):
            logger.error(f"Input file not found: {input_filename}")
            current_logger.setLevel(original_level)
            return pd.DataFrame()

        try:
            logger.info(f"Processing transition file: {input_filename}")
            num_blocks_written, block_start_lines = _split_fac_file(
                input_filename, self.output_prefix
            )
            if num_blocks_written > 1:
                self._temp_files_created = [
                    f"{self.output_prefix}_{i+1}.txt" for i in range(num_blocks_written)
                ]

            if num_blocks_written == 0:
                logger.error(f"Error during file splitting for {input_filename}.")
                all_transitions = pd.DataFrame()
            elif num_blocks_written == 1:
                logger.info(
                    f"File {input_filename} contains a single block. Reading directly."
                )
                start_line_for_single = block_start_lines[0] if block_start_lines else 1
                all_transitions = self._read_fac_transition_block_data(
                    input_filename, original_start_line=start_line_for_single
                )
            else:
                logger.info(
                    f"Detected {num_blocks_written} data blocks starting at lines: {block_start_lines}. Reading split files..."
                )
                transitions_list = []
                for i in range(num_blocks_written):
                    block_index = i + 1
                    block_filename = f"{self.output_prefix}_{block_index}.txt"
                    original_start_line = (
                        block_start_lines[i]
                        if i < len(block_start_lines)
                        else "Unknown"
                    )
                    logger.info(
                        f"--- Reading Transition Block {block_index} (Original Line: {original_start_line}) ---"
                    )
                    transitions_temp = self._read_fac_transition_block_data(
                        block_filename, original_start_line
                    )
                    if not transitions_temp.empty:
                        transitions_list.append(transitions_temp)
                    else:
                        logger.info(
                            f"Skipping transition block {block_index} due to read/processing errors."
                        )

                if transitions_list:
                    logger.info(
                        f"--- Successfully read {len(transitions_list)} transition blocks. Concatenating... ---"
                    )
                    all_transitions = pd.concat(
                        [
                            (
                                df.reset_index()
                                if df.index.nlevels > 1
                                or any(name is not None for name in df.index.names)
                                else df
                            )
                            for df in transitions_list
                        ],
                        ignore_index=True,
                    )
                    all_transitions = self._post_process_concatenated(all_transitions)
                else:
                    logger.error(
                        f"No transition blocks could be successfully read from {input_filename} after splitting."
                    )
                    all_transitions = pd.DataFrame()
        except Exception as e:
            logger.error(
                f"An unexpected error occurred in FacTransitionReader.read: {e}",
                exc_info=True,
            )
            all_transitions = pd.DataFrame()
        finally:
            self._cleanup_temp_files()
            current_logger.setLevel(original_level)

        all_transitions = self._apply_final_manipulations(all_transitions)

        if logger.isEnabledFor(logging.INFO):
            if not all_transitions.empty:
                cols_display = list(all_transitions.columns)
                idx_display = list(all_transitions.index.names)
                logger.info(
                    f"Finished transition processing. Returning DataFrame with {len(all_transitions)} rows."
                )
                if not all(name is None for name in idx_display):
                    logger.info(f"Index: {idx_display}")
                logger.info(f"Columns: {cols_display}")
            elif os.path.exists(input_filename):
                logger.info(
                    f"Finished transition processing {input_filename}, but resulted in an empty DataFrame."
                )

        return all_transitions

    def _read_fac_transition_block_data(
        self, file_transitions: str, original_start_line: Optional[int] = None
    ) -> pd.DataFrame:
        """Reads a single FAC transition file/block."""
        atomic_number, nele, multipole_header_value = _extract_header_info(
            file_transitions, reader_type="transition"
        )  # Specify type

        if atomic_number is None or nele is None:
            logger.warning(
                f"Failed to extract header info (Z/Nele) from {file_transitions}. Skipping block."
            )
            return pd.DataFrame()

        ion_charge = atomic_number - nele
        transitions_df = self._process_fac_transitions(
            atomic_number,
            ion_charge,
            file_transitions,
            multipole_header_value,
            original_start_line,
        )

        if transitions_df.empty:
            logger.warning(
                f"Processing resulted in empty DataFrame for transition block from original line {original_start_line or '?'}."
            )

        return transitions_df

    def _find_transition_data_start_row(self, lines: List[str]) -> Tuple[int, str]:
        """Finds the start row for transition data."""
        # (Function remains the same)
        header_pattern = re.compile(r"\s*ILEV_UP\s+2J_UP\s+ILEV_LO\s+2J_LO\s+")
        for i, line in enumerate(lines):
            if header_pattern.match(line):
                skip_rows_count = i + 1
                message = f"Found specific transition header line ('ILEV_UP 2J_UP...'). Determined skiprows={skip_rows_count}."
                return skip_rows_count, message

        message = "Specific transition header line not found. Using fallback: searching for first line starting with two numbers."
        for i, line in enumerate(lines):
            if re.match(r"^\s*\d+\s+\d+", line):
                if i > 0 and not re.match(r"^\s*\d+\s+\d+", lines[i - 1]):
                    skip_rows_count = i
                    message += f" Found potential data start at line {i+1}. Determined skiprows={skip_rows_count}."
                    return skip_rows_count, message
                # --- Corrected Indentation ---
                elif i == 0:
                    skip_rows_count = i
                    message += f" Found potential data start at line {i+1}. Determined skiprows={skip_rows_count}."
                    return skip_rows_count, message
                # --- End Correction ---

        message = "Could not determine transition data start row. Using default guess."
        nele_index = next((i for i, l in enumerate(lines) if l.startswith("NELE")), -1)
        multip_index = next(
            (i for i, l in enumerate(lines) if l.strip().startswith("MULTIP")), -1
        )
        if multip_index != -1:
            skip_rows_count = multip_index + 4
        elif nele_index != -1:
            skip_rows_count = nele_index + 3
        else:
            skip_rows_count = 13
        message += f" Determined skiprows={skip_rows_count}."
        return skip_rows_count, message

    def _process_fac_transitions(
        self,
        atomic_number: int,
        ion_charge: int,
        file_transitions: str,
        multipole_header_value: Optional[int],
        original_start_line: Optional[int] = None,
    ) -> pd.DataFrame:
        """Processes data rows from a single FAC transition file/block."""
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

        skiprows, skip_message = self._find_transition_data_start_row(lines)
        logger.info(f"{skip_message} {file_context}")

        transitions = pd.DataFrame()
        # --- UPDATED Concise internal names ---
        col_names = [
            "level_index_upper",
            "2j_upper",
            "level_index_lower",
            "2j_lower",
            "E",
            "gf",
            "A",
            "multipole",  # Use E, A
        ]
        essential_cols = [
            "level_index_upper",
            "2j_upper",
            "level_index_lower",
            "2j_lower",
            "E",
            "gf",
            "A",
        ]

        try:
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

            if not all(col in transitions.columns for col in essential_cols):
                logger.error(
                    f"Essential transition columns missing after CSV read {file_context}. Expected: {essential_cols}, Found: {transitions.columns.tolist()}"
                )
                logger.warning(
                    f"Attempting fallback read using specific columns based on original function: [0, 2, 4, 5, 6, 7]"
                )
                col_names_alt = [
                    "level_index_upper",
                    "level_index_lower",
                    "E",
                    "gf",
                    "A",
                    "multipole",
                ]
                usecols_alt = [0, 2, 4, 5, 6, 7]
                transitions = pd.read_csv(
                    file_transitions,
                    sep=r"\s+",
                    names=col_names_alt,
                    skiprows=skiprows,
                    usecols=usecols_alt,
                    on_bad_lines="warn",
                    engine="python",
                    comment="=",
                    skipinitialspace=True,
                )
                if "2j_upper" not in transitions.columns:
                    transitions["2j_upper"] = np.nan
                if "2j_lower" not in transitions.columns:
                    transitions["2j_lower"] = np.nan
                if not all(col in transitions.columns for col in essential_cols):
                    logger.error(
                        f"Essential transition columns still missing after fallback read {file_context}. Found: {transitions.columns.tolist()}"
                    )
                    raise ValueError(
                        f"Essential transition columns missing even after trying alternative format: {essential_cols}"
                    )

            for col in essential_cols:
                if col in transitions.columns:
                    transitions[col] = pd.to_numeric(transitions[col], errors="coerce")

            initial_rows = len(transitions)
            transitions.dropna(
                subset=[c for c in essential_cols if c in transitions.columns],
                inplace=True,
            )
            if len(transitions) < initial_rows:
                logger.info(
                    f"Dropped {initial_rows - len(transitions)} rows due to non-numeric essential transition data {file_context}."
                )

            if transitions.empty:
                logger.warning(
                    f"No valid transition data rows found after cleaning essential columns {file_context}."
                )
                return pd.DataFrame()

            for col in ["multipole"]:
                if col not in transitions.columns:
                    transitions[col] = "-"

        except Exception as e_csv:
            logger.error(
                f"Error reading transition data {file_context}: {e_csv}", exc_info=True
            )
            return pd.DataFrame()

        try:
            # Convert types
            transitions["level_index_upper"] = transitions["level_index_upper"].astype(
                int
            )
            transitions["level_index_lower"] = transitions["level_index_lower"].astype(
                int
            )
            if "2j_upper" in transitions.columns:
                transitions["2j_upper"] = pd.to_numeric(
                    transitions["2j_upper"], errors="coerce"
                ).astype("Int64")
            if "2j_lower" in transitions.columns:
                transitions["2j_lower"] = pd.to_numeric(
                    transitions["2j_lower"], errors="coerce"
                ).astype("Int64")
            if "multipole" in transitions.columns:
                transitions["multipole"] = (
                    transitions["multipole"].astype(str).str.strip().fillna("-")
                )
            if "E" in transitions.columns:
                transitions["E"] = pd.to_numeric(
                    transitions["E"], errors="coerce"
                )  # Read as eV
            else:
                transitions["E"] = np.nan

            # Calculate wavelength (internal, in Angstrom)
            wavelength_A = np.nan
            if "E" in transitions.columns and pd.api.types.is_numeric_dtype(
                transitions["E"]
            ):
                valid_delta_e = transitions["E"] > 1e-9
                if valid_delta_e.any():
                    wavelength_A = HC_EV_ANGSTROM / transitions.loc[valid_delta_e, "E"]
            transitions["lambda"] = (
                wavelength_A  # Store internal Angstrom value temporarily
            )

            # Add multipole_type column (concise: 'type')
            if multipole_header_value is not None:
                m_abs = abs(multipole_header_value)
                m_type = "M" if multipole_header_value > 0 else "E"
                transitions["type"] = f"{m_type}{m_abs}"
            else:
                transitions["type"] = "Unknown"

            # Calculate Line Strength S (a.u.) using internal wavelength_A
            if "gf" in transitions.columns and "lambda" in transitions.columns:
                gf_numeric = pd.to_numeric(transitions["gf"], errors="coerce")
                lambda_a_numeric = pd.to_numeric(
                    transitions["lambda"], errors="coerce"
                )  # Use internal Angstrom value
                valid_for_S = (
                    gf_numeric.notna()
                    & lambda_a_numeric.notna()
                    & (lambda_a_numeric > 1e-9)
                )
                transitions["S"] = np.nan  # Concise name
                if valid_for_S.any():
                    transitions.loc[valid_for_S, "S"] = (
                        gf_numeric[valid_for_S]
                        * lambda_a_numeric[valid_for_S]
                        / LINE_STRENGTH_CONST
                    )
                logger.debug(f"Calculated 'S' column (atomic units) {file_context}.")
            else:
                logger.warning(
                    f"Cannot calculate S as 'gf' or 'lambda' are missing/non-numeric {file_context}."
                )
                transitions["S"] = np.nan

            # --- Unit Conversion ---
            # Energy
            if "E" in transitions.columns and pd.api.types.is_numeric_dtype(
                transitions["E"]
            ):
                target_unit_e = self.energy_unit
                if target_unit_e == "cm-1":
                    transitions["E"] *= EV_TO_CM1
                elif target_unit_e == "ry":
                    transitions["E"] /= RYD_EV
                elif target_unit_e == "hz":
                    transitions["E"] *= const.e / const.h
                elif target_unit_e == "j":
                    transitions["E"] *= const.e
                elif target_unit_e == "ha":
                    transitions["E"] /= HARTREE_EV
                elif target_unit_e != "ev":
                    logger.warning(
                        f"Energy unit '{target_unit_e}' not recognized, keeping eV."
                    )
            # Wavelength
            if "lambda" in transitions.columns and pd.api.types.is_numeric_dtype(
                transitions["lambda"]
            ):
                target_unit_w = self.wavelength_unit
                if target_unit_w == "nm":
                    transitions["lambda"] /= 10.0
                elif target_unit_w == "m":
                    transitions["lambda"] *= 1e-10
                elif target_unit_w == "cm":
                    transitions["lambda"] *= 1e-8
                elif target_unit_w != "a":
                    logger.warning(
                        f"Wavelength unit '{target_unit_w}' not recognized, keeping Angstrom."
                    )
            # --- End Unit Conversion ---

            # Add metadata
            transitions["atomic_number"] = atomic_number
            transitions["ion_charge"] = ion_charge
            transitions["method"] = "FAC"
            # transitions["priority"] = 10 # Removed

            transitions = data_optimize(transitions)

            # --- Column Selection/Reordering ---
            final_column_list = [
                "atomic_number",
                "ion_charge",
                "level_index_lower",
                "level_index_upper",
                "2j_lower",
                "2j_upper",
                "E",
                "lambda",
                "gf",
                "A",
                "S",
                "multipole",
                "type",  # Use concise names
            ]
            if self.include_method:
                final_column_list.append("method")

            final_columns_present = [
                col for col in final_column_list if col in transitions.columns
            ]
            transitions = transitions[final_columns_present]
            # --- End Column Selection ---

            transitions = data_optimize(transitions)
            transitions = transitions.sort_values(
                by=["level_index_upper", "level_index_lower"]
            )

            # --- Set Index ---
            index_cols_present = [
                col for col in self._INDEX_COLUMNS if col in transitions.columns
            ]
            if len(index_cols_present) == len(self._INDEX_COLUMNS):
                try:
                    transitions.set_index(index_cols_present, inplace=True)
                except Exception as e:
                    logger.error(
                        f"Error setting transition index {file_context}: {e}. Keeping unindexed.",
                        exc_info=True,
                    )
            else:
                missing_idx_cols = [
                    col for col in self._INDEX_COLUMNS if col not in transitions.columns
                ]
                logger.warning(
                    f"Could not set transition multi-index {file_context} due to missing columns: {missing_idx_cols}. Keeping unindexed."
                )

        except Exception as e:
            logger.error(
                f"Error during transition post-processing {file_context}: {e}",
                exc_info=True,
            )
            return pd.DataFrame()

        return transitions

    def _post_process_concatenated(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-processes concatenated transition DataFrame."""
        if df.empty:
            return df
        df = data_optimize(df)

        sort_cols = [
            col
            for col in [
                "atomic_number",
                "ion_charge",
                "level_index_upper",
                "level_index_lower",
            ]
            if col in df.columns
        ]
        if sort_cols:
            df.sort_values(by=sort_cols, inplace=True)

        index_cols_present = [col for col in self._INDEX_COLUMNS if col in df.columns]
        if len(index_cols_present) == len(self._INDEX_COLUMNS):
            try:
                df.set_index(index_cols_present, inplace=True)
            except Exception as e:
                logger.warning(
                    f"Could not set multi-index after concatenating transition blocks: {e}. Keeping sorted but unindexed."
                )
        else:
            logger.warning(
                "Index columns missing after transition concatenation. Keeping sorted but unindexed."
            )

        return df

    def _apply_final_manipulations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies final column keeping and renaming for transitions."""
        if df.empty:
            return df

        # --- Handle Renaming First (using concise internal names) ---
        rename_map_internal = {}
        # Determine current columns (check if indexed)
        was_indexed = df.index.nlevels > 1 or any(
            name is not None for name in df.index.names
        )
        current_columns = list(df.columns)
        current_index_names = list(df.index.names) if was_indexed else []
        all_current_names = current_columns + current_index_names

        if self.rename_columns:
            for concise_key, new_name in self.rename_columns.items():
                if concise_key in all_current_names:
                    rename_map_internal[concise_key] = new_name
                else:
                    logger.warning(
                        f"Column/Index '{concise_key}' not found for renaming in transition data."
                    )

        # Apply renaming
        df_renamed = df.copy()  # Avoid modifying original df passed to function
        if rename_map_internal:
            index_rename_map = {
                k: v for k, v in rename_map_internal.items() if k in current_index_names
            }
            col_rename_map = {
                k: v for k, v in rename_map_internal.items() if k in current_columns
            }
            if index_rename_map:
                df_renamed.rename_axis(index=index_rename_map, inplace=True)
            if col_rename_map:
                df_renamed.rename(columns=col_rename_map, inplace=True)

        # --- Handle Column Keeping (using potentially renamed concise names) ---
        # Get current columns and index names after potential rename
        current_cols_after_rename = list(df_renamed.columns)
        current_index_after_rename = list(df_renamed.index.names)
        final_df = df_renamed  # Start with potentially renamed df

        if self.columns_to_keep is not None:
            # User provides concise names in columns_to_keep
            requested_concise_keep = set(self.columns_to_keep)

            # Map requested concise names to their potentially renamed versions
            target_names_to_keep = set()
            for concise_name in requested_concise_keep:
                target_names_to_keep.add(
                    rename_map_internal.get(concise_name, concise_name)
                )

            # Always keep index columns
            target_names_to_keep.update(current_index_after_rename)

            # Filter current columns based on the target names
            cols_to_actually_keep = [
                col for col in current_cols_after_rename if col in target_names_to_keep
            ]
            index_levels_to_keep = [
                name
                for name in current_index_after_rename
                if name in target_names_to_keep
            ]

            missing_cols = (
                target_names_to_keep
                - set(cols_to_actually_keep)
                - set(index_levels_to_keep)
            )
            # Only warn if the missing column wasn't an index level (which are handled separately)
            missing_cols_non_index = [
                c for c in missing_cols if c not in current_index_after_rename
            ]
            if missing_cols_non_index:
                logger.warning(
                    f"Requested transition columns/index levels to keep not found or not available after renaming: {missing_cols_non_index}"
                )

            if not cols_to_actually_keep and not index_levels_to_keep:
                logger.warning(
                    "columns_to_keep resulted in no columns or index levels being selected. Returning original."
                )
                final_df = df_renamed  # Revert to df after renaming
            else:
                # Select columns, keeping the potentially modified index
                if df_renamed.index.nlevels > 1 or any(
                    name is not None for name in df_renamed.index.names
                ):
                    # Check if all requested index levels are still present
                    if set(index_levels_to_keep) == set(current_index_after_rename):
                        final_df = df_renamed[
                            cols_to_actually_keep
                        ]  # Select columns, index remains
                    else:
                        # If index levels were removed, reset and select
                        logger.debug(
                            "Index levels modified by columns_to_keep. Resetting index."
                        )
                        final_df = df_renamed.reset_index()
                        # Combine kept columns and kept index levels for final selection
                        final_selection = cols_to_actually_keep + index_levels_to_keep
                        # Ensure no duplicates and columns exist
                        final_selection = [
                            c for c in final_selection if c in final_df.columns
                        ]
                        final_df = final_df[
                            list(dict.fromkeys(final_selection))
                        ]  # Keep order, remove duplicates
                else:
                    # If not indexed, just select columns
                    final_df = df_renamed[cols_to_actually_keep]

        return final_df

    def _cleanup_temp_files(self):
        """Removes temporary files created by this instance."""
        if self._temp_files_created:
            logger.info(
                f"Cleaning up {len(self._temp_files_created)} temporary transition block files..."
            )
            for temp_file in self._temp_files_created:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.warning(
                            f"Could not remove temporary file {temp_file}: {e}"
                        )
        self._temp_files_created = []


# --- Main Public Functions (using the classes) ---


def read_fac(
    base_filename: str,
    file_extension: str = ".lev.asc",
    verbose: int = 1,  # Default verbose=1 (INFO)
    **kwargs,
) -> pd.DataFrame:
    """
    Reads FAC energy level data (.lev.asc) using the FacReader class.

    Args:
        base_filename (str): Base name of the FAC file (without extension).
        file_extension (str): Extension of the FAC file (default: '.lev.asc').
        verbose (int): Logging level: 0=Warning, 1=Info (default), 2=Debug.
        **kwargs: Additional arguments passed to FacReader constructor
                  (e.g., energy_unit='cm-1', include_method=True, include_new_config=True,
                   columns_to_keep=['E'], rename_columns={'p':'Parity'}).

    Returns:
        pandas.DataFrame: Processed energy levels. Empty on error.
    """
    input_filename = f"{base_filename}{file_extension}"
    reader = FacReader(verbose=verbose, **kwargs)
    df = reader.read(input_filename)
    return df


def read_fac_transitions(
    base_filename: str,
    file_extension: str = ".tr.asc",  # Default extension for transitions
    verbose: int = 1,  # Default verbose=1 (INFO)
    **kwargs,
) -> pd.DataFrame:
    """
    Reads FAC transition data (.tr.asc) using the FacTransitionReader class.

    Args:
        base_filename (str): Base name of the FAC file (without extension).
        file_extension (str): Extension of the FAC file (default: '.tr.asc').
        verbose (int): Logging level: 0=Warning, 1=Info (default), 2=Debug.
        **kwargs: Additional arguments passed to FacTransitionReader constructor
                  (e.g., energy_unit='ry', wavelength_unit='nm', include_method=True,
                   columns_to_keep=['lambda'], rename_columns={'A':'A_value'}).

    Returns:
        pandas.DataFrame: Processed transitions data. Empty on error.
    """
    input_filename = f"{base_filename}{file_extension}"
    reader = FacTransitionReader(verbose=verbose, **kwargs)  # Use the new reader
    df = reader.read(input_filename)
    return df
