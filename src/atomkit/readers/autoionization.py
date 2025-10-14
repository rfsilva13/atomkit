# atomkit/src/atomkit/readers/autoionization.py

"""
Reader class and function specifically for FAC autoionization files (.ai.asc).
"""

import logging
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..definitions import *
from .base import (
    _BaseFacReader,
    _extract_header_info,
    _split_fac_file,
    data_optimize,
)

logger = logging.getLogger(__name__)


class FacAutoionizationReader(_BaseFacReader):
    """
    Reads and processes FAC autoionization files (.ai.asc).
    """

    _DEFAULT_OUTPUT_COLUMNS = [
        "atomic_number",
        "level_index_upper",
        "2j_upper",
        "level_index_lower",
        "2j_lower",
        "energy",
        "ai_rate",
        "dc_rate",
    ]
    _INDEX_COLUMNS = [
        "atomic_number",
        "ion_charge",
        "level_index_upper",
        "level_index_lower",
    ]

    def __init__(
        self,
        energy_unit: str = "ev",
        columns_to_keep: Optional[list[str]] = None,
        rename_columns: Optional[dict[str, str]] = None,
        output_prefix: str = "temp_fac_ai_block",
        verbose: int = 1,
        include_method: bool = False,
    ):
        """
        Initializes the FAC autoionization file reader.
        """
        super().__init__(verbose=verbose, output_prefix=output_prefix)
        self.energy_unit = energy_unit.lower().replace("[", "").replace("]", "")
        if self.energy_unit not in ALLOWED_ENERGY_UNITS:
            raise ValueError(f"Invalid energy_unit. Allowed: {ALLOWED_ENERGY_UNITS}")
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
        Reads autoionization data from the specified file or pre-split block files.
        """
        all_ai_data = pd.DataFrame()

        try:
            if block_files:
                ai_list = []
                for i, block_filename in enumerate(block_files):
                    start_line = block_starts[i] if i < len(block_starts) else None
                    ai_temp = self._read_fac_autoionization_block_data(
                        block_filename, start_line
                    )
                    if not ai_temp.empty:
                        ai_list.append(ai_temp)
                if ai_list:
                    all_ai_data = pd.concat(ai_list, ignore_index=True)
                    all_ai_data = self._post_process_concatenated(all_ai_data)
            else:
                all_ai_data = self._read_fac_autoionization_block_data(
                    input_filename, original_start_line=1
                )

        except Exception as e:
            logger.error(
                f"Unexpected error in FacAutoionizationReader.read for {input_filename}: {e}",
                exc_info=True,
            )
            all_ai_data = pd.DataFrame()

        try:
            all_ai_data = self._apply_final_manipulations(
                all_ai_data,
                self._DEFAULT_OUTPUT_COLUMNS,
                self.columns_to_keep,
                self.rename_columns,
                self.include_method,
            )
        except Exception as e:
            logger.error(
                f"Error during final column manipulation for autoionization data from {input_filename}: {e}",
                exc_info=True,
            )
            all_ai_data = pd.DataFrame()

        return all_ai_data

    def _read_fac_autoionization_block_data(
        self, file_autoionization: str, original_start_line: Optional[int] = None
    ) -> pd.DataFrame:
        atomic_number, nele, _ = _extract_header_info(
            file_autoionization, reader_type="level"
        )
        if atomic_number is None or nele is None:
            return pd.DataFrame()

        ion_charge = atomic_number - nele

        ai_df = self._process_fac_autoionization(
            atomic_number,
            ion_charge,
            file_autoionization,
            original_start_line,
        )

        return ai_df

    def _find_data_start_row(self, lines: list[str]) -> tuple[int, str]:
        negrid = 0
        negrid_line_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("NEGRID"):
                try:
                    negrid = int(line.strip().split("=")[1])
                    negrid_line_index = i
                    break
                except (ValueError, IndexError):
                    pass
        if negrid_line_index != -1:
            return (
                negrid_line_index + negrid + 1,
                f"Found NEGRID={negrid}. Data starts after energy grid.",
            )
        # Default: assume data starts at line 0
        return (0, "No NEGRID found, assuming data starts at beginning.")

    def _process_fac_autoionization(
        self,
        atomic_number: int,
        ion_charge: int,
        file_autoionization: str,
        original_start_line: Optional[int] = None,
    ) -> pd.DataFrame:
        file_context = f"(Original Line: {original_start_line or '?'}, File: {os.path.basename(file_autoionization)})"
        try:
            with open(file_autoionization, "r") as file:
                lines = file.readlines()
        except Exception as e:
            logger.error(
                f"Error reading file {file_autoionization}: {e}", exc_info=True
            )
            return pd.DataFrame()

        skiprows, skip_message = self._find_data_start_row(lines)
        logger.info(f"{skip_message} {file_context}")

        col_names = [
            "level_index_upper",
            "2j_upper",
            "level_index_lower",
            "2j_lower",
            "energy",
            "ai_rate",
            "dc_rate",
        ]
        try:
            ai_data = pd.read_csv(
                file_autoionization,
                sep=r"\s+",
                names=col_names,
                skiprows=skiprows,
                engine="python",
            )
        except Exception as e:
            logger.error(
                f"Error parsing data in {file_autoionization}: {e}", exc_info=True
            )
            return pd.DataFrame()

        ai_data["atomic_number"] = atomic_number
        ai_data["ion_charge"] = ion_charge

        return data_optimize(ai_data)

    def _post_process_concatenated(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = data_optimize(df)
        sort_cols = [
            "atomic_number",
            "ion_charge",
            "level_index_upper",
            "level_index_lower",
        ]
        df.sort_values(by=[col for col in sort_cols if col in df.columns], inplace=True)
        return df


def read_fac_autoionization(
    base_filename: str, file_extension: str = ".ai.asc", verbose: int = 1, **kwargs
) -> pd.DataFrame:
    """
    Reads FAC autoionization data (.ai.asc), handling multi-block files.
    """
    input_filename = f"{base_filename}{file_extension}"
    output_prefix = (
        f"temp_fac_ai_block_{os.path.splitext(os.path.basename(input_filename))[0]}"
    )

    if not os.path.exists(input_filename):
        logger.error(f"Input file not found: {input_filename}")
        return pd.DataFrame()

    reader = FacAutoionizationReader(
        verbose=verbose, output_prefix=output_prefix, **kwargs
    )
    try:
        num_blocks_written, block_start_lines = _split_fac_file(
            input_filename, reader.output_prefix
        )
        block_files = []
        if num_blocks_written > 1:
            block_files = [
                f"{reader.output_prefix}_{i+1}.txt" for i in range(num_blocks_written)
            ]
            reader._temp_files_created = block_files

        if num_blocks_written == 0:
            return pd.DataFrame()
        elif num_blocks_written == 1 and not block_files:
            return reader.read(
                input_filename, block_files=[], block_starts=block_start_lines
            )
        else:
            return reader.read(
                input_filename, block_files=block_files, block_starts=block_start_lines
            )
    finally:
        reader._cleanup_temp_files()
        reader._restore_log_level()
