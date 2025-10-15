"""
AUTOSTRUCTURE output file reader module.

Provides functions to read and parse AUTOSTRUCTURE 'olg' output files,
extracting energy levels, transitions, and configuration information.

Compatible with NumPy 2.x and Pandas 2.x.

Original Author: Tomás Campante (October 2025)
Adapted by: Ricardo Silva (rfsilva@lip.pt)
Date: October 2025
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import logging

import pandas as pd
import numpy as np
import re

logger = logging.getLogger(__name__)


def read_as_levels(
    filename: str | Path, output_file: Optional[str | Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read atomic energy levels from AUTOSTRUCTURE 'olg' output file.

    Parses the fine structure energy levels section of an AUTOSTRUCTURE
    output file and returns level information including quantum numbers,
    energies, and electronic configurations.

    Parameters
    ----------
    filename : str or Path
        Path to the AUTOSTRUCTURE 'olg' output file
    output_file : str or Path, optional
        If provided, saves the parsed data to this file with metadata header

    Returns
    -------
    df_levels : pd.DataFrame
        DataFrame with columns:
        - K : Level index
        - P : Parity (0 for even, 1 for odd)
        - 2J : Twice the total angular momentum
        - 2*S+1 : Spin multiplicity
        - L : Orbital angular momentum quantum number
        - Level (Ry) : Energy level relative to ground state in Rydbergs
        - CF : Electronic configuration string
    metadata : dict
        Dictionary containing atomic structure metadata

    Examples
    --------
    >>> df, meta = read_as_levels('output.olg')
    >>> print(meta['Atomic number'])
    26
    >>> print(df[['K', '2J', 'Level (Ry)', 'CF']].head())

    Notes
    -----
    Preserves original parsing logic from Tomás Campante's olg_reader_levels+trans.py,
    modernized for NumPy 2.x and type safety.
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"AUTOSTRUCTURE output file not found: {filepath}")

    logger.info(f"Reading AUTOSTRUCTURE levels from {filepath}")

    # Initialize variables
    configs_dictionary: Dict[str, str] = {}
    Z, E = 0, 0
    cput, tcput = "N/A", "N/A"
    gs = "N/A"

    # Create DataFrame with desired columns
    df_levels = pd.DataFrame(columns=["K", "P", "2J", "2*S+1", "L", "Level (Ry)", "CF"])

    # Parsing state variables
    levels_index_found = False
    config_index_found = 0
    previous_line = ""
    config_header_line = ""
    columns: Dict[str, int] = {}

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):

                # Build configuration dictionary
                if "BASIC CONFIGURATION" in line:
                    config_index_found = line_number

                if config_index_found > 0 and ": " in line:
                    try:
                        if config_header_line == "":
                            config_header_line = previous_line.strip()
                        line_stripped = line.strip()
                        if line_stripped and line_stripped[0].isdigit():
                            config_info = _parse_configuration(config_header_line, line)
                            configs_dictionary[config_info[0]] = config_info[1]
                    except (ValueError, IndexError):
                        config_index_found = 0

                previous_line = line

                # Read fine structure energies
                if "2J" in line and "(EK-E1)/RY" in line:
                    levels_index_found = True
                    header_tokens = line.split()
                    columns = {value: idx for idx, value in enumerate(header_tokens)}
                    gs = header_tokens[-1] if header_tokens else "N/A"
                    continue

                if levels_index_found and len(line.strip()) == 0:
                    levels_index_found = False

                if levels_index_found and columns:
                    parts = [p for p in line.split() if p]
                    try:
                        data_row: Dict[str, Any] = {}
                        data_row["K"] = (
                            parts[columns["K"]]
                            if "K" in columns and len(parts) > columns["K"]
                            else None
                        )
                        cf_key = (
                            parts[columns["CF"]]
                            if "CF" in columns and len(parts) > columns["CF"]
                            else None
                        )
                        data_row["CF"] = (
                            configs_dictionary.get(cf_key, cf_key) if cf_key else None
                        )
                        data_row["2J"] = (
                            parts[columns["2J"]]
                            if "2J" in columns and len(parts) > columns["2J"]
                            else None
                        )
                        data_row["2*S+1"] = (
                            int(parts[columns["2*S+1"]])
                            if "2*S+1" in columns and len(parts) > columns["2*S+1"]
                            else None
                        )
                        data_row["L"] = (
                            int(parts[columns["L"]])
                            if "L" in columns and len(parts) > columns["L"]
                            else None
                        )
                        data_row["Level (Ry)"] = (
                            float(parts[columns["(EK-E1)/RY"]])
                            if "(EK-E1)/RY" in columns
                            and len(parts) > columns["(EK-E1)/RY"]
                            else None
                        )
                        s2 = data_row["2*S+1"]
                        data_row["P"] = 0 if s2 and s2 >= 0 else 1
                        df_levels = pd.concat(
                            [df_levels, pd.DataFrame([data_row])], ignore_index=True
                        )
                    except (ValueError, KeyError, IndexError) as e:
                        logger.warning(
                            f"Skipped invalid level line {line_number}: {line.strip()} ({e})"
                        )
                        continue

                # Retrieve metadata
                if Z == 0 and "ATOMIC NUMBER" in line:
                    try:
                        start_idx = line.find("ATOMIC NUMBER") + len("ATOMIC NUMBER")
                        end_idx = line.find(",   NUMBER OF ELECTRONS")
                        if end_idx > start_idx:
                            Z = int(line[start_idx:end_idx].strip())
                    except ValueError:
                        logger.warning(
                            f"Could not parse atomic number from line {line_number}"
                        )

                if E == 0 and "NUMBER OF ELECTRONS" in line:
                    try:
                        start_idx = line.find("NUMBER OF ELECTRONS") + len(
                            "NUMBER OF ELECTRONS"
                        )
                        E = int(line[start_idx:].strip().rstrip("\n"))
                    except ValueError:
                        logger.warning(
                            f"Could not parse electron number from line {line_number}"
                        )

                if "CPU TIME" in line:
                    line_parts = line.split(" ")
                    cleaned = [x.strip() for x in line_parts if x.strip() != ""]
                    if len(cleaned) > 7:
                        if not (cleaned[2] == "0.000" and cleaned[7] == "0.000"):
                            cput = f"{cleaned[2]} {cleaned[3]}"
                            tcput = f"{cleaned[7]} {cleaned[8]}"

    except Exception as e:
        logger.error(f"Error reading AUTOSTRUCTURE file: {e}")
        raise

    # Build metadata dictionary
    closed_config = _get_closed_shells(configs_dictionary.get("1", ""))
    metadata: Dict[str, Any] = {
        "Atomic number": Z,
        "Number of electrons": E,
        "Closed": closed_config[0] if closed_config else "N/A",
        "Ground state energy (Ry)": gs,
        "CPU time": cput,
        "Total CPU time": tcput,
    }

    logger.info(f"Successfully read {len(df_levels)} levels for Z={Z}, E={E}")

    # Optionally save to file
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("########\n")
            df_levels.to_csv(f, sep="\t", index=False)
        logger.info(f"Saved levels data to {output_path}")

    return df_levels, metadata


def read_as_transitions(
    filename: str | Path, output_file: Optional[str | Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read radiative transitions from AUTOSTRUCTURE 'olg' output file.

    Parameters
    ----------
    filename : str or Path
        Path to the AUTOSTRUCTURE 'olg' output file
    output_file : str or Path, optional
        If provided, saves the parsed data to this file

    Returns
    -------
    df_transitions : pd.DataFrame
        DataFrame with transition data
    metadata : dict
        Atomic structure metadata

    Examples
    --------
    >>> df_trans, meta = read_as_transitions('output.olg')
    >>> print(df_trans[['K', 'Klower', 'WAVEL/AE', 'A(K)*SEC']].head())

    Notes
    -----
    Preserves original parsing logic, modernized for NumPy 2.x.
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"AUTOSTRUCTURE output file not found: {filepath}")

    logger.info(f"Reading AUTOSTRUCTURE transitions from {filepath}")

    # Initialize variables
    configs_dictionary: Dict[str, str] = {}
    Z, E = 0, 0
    cput, tcput = "N/A", "N/A"
    gs = "N/A"

    # Create DataFrame
    df_transitions = pd.DataFrame(
        columns=["index", "K", "Klower", "WAVEL/AE", "A(K)*SEC", "F(ABS)", "log(gf)"]
    )

    # Parsing state variables
    transitions_index_found = False
    config_index_found = 0
    previous_line = ""
    config_header_line = ""
    columns: Dict[str, int] = {}

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):

                # Build configuration dictionary
                if "BASIC CONFIGURATION" in line:
                    config_index_found = line_number

                if config_index_found > 0 and ": " in line:
                    try:
                        if config_header_line == "":
                            config_header_line = previous_line.strip()
                        line_stripped = line.strip()
                        if line_stripped and line_stripped[0].isdigit():
                            config_info = _parse_configuration(config_header_line, line)
                            configs_dictionary[config_info[0]] = config_info[1]
                    except (ValueError, IndexError):
                        config_index_found = 0

                previous_line = line

                # Read transitions
                if "G*F" in line and "WAVEL/AE" in line and "V(GFL*GFV)" not in line:
                    transitions_index_found = True
                    header_tokens = line.split()
                    columns = {value: idx for idx, value in enumerate(header_tokens)}
                    continue

                if transitions_index_found and len([p for p in line.split() if p]) == 0:
                    transitions_index_found = False

                if transitions_index_found and columns:
                    parts = [p for p in line.split() if p]
                    try:
                        data_row: Dict[str, Any] = {}
                        data_row["index"] = (
                            int(parts[columns["E1-DATA"]])
                            if "E1-DATA" in columns and len(parts) > columns["E1-DATA"]
                            else None
                        )
                        data_row["K"] = (
                            int(parts[columns["K"]])
                            if "K" in columns and len(parts) > columns["K"]
                            else None
                        )
                        data_row["Klower"] = (
                            int(parts[columns["KP"]])
                            if "KP" in columns and len(parts) > columns["KP"]
                            else None
                        )
                        data_row["WAVEL/AE"] = (
                            float(parts[columns["WAVEL/AE"]])
                            if "WAVEL/AE" in columns
                            and len(parts) > columns["WAVEL/AE"]
                            else None
                        )
                        data_row["A(K)*SEC"] = (
                            f"{float(parts[columns['A(EK)*SEC']]):.3E}"
                            if "A(EK)*SEC" in columns
                            and len(parts) > columns["A(EK)*SEC"]
                            else None
                        )
                        data_row["F(ABS)"] = (
                            float(parts[columns["F(ABS)"]])
                            if "F(ABS)" in columns and len(parts) > columns["F(ABS)"]
                            else None
                        )
                        gf_val = (
                            float(parts[columns["G*F"]])
                            if "G*F" in columns and len(parts) > columns["G*F"]
                            else None
                        )
                        data_row["log(gf)"] = (
                            f"{np.log(abs(gf_val)):.6f}" if gf_val is not None else None
                        )
                        df_transitions = pd.concat(
                            [df_transitions, pd.DataFrame([data_row])],
                            ignore_index=True,
                        )
                    except (ValueError, KeyError, IndexError) as e:
                        logger.warning(
                            f"Skipped invalid transition line {line_number}: {line.strip()} ({e})"
                        )
                        continue

                # Retrieve metadata
                if Z == 0 and "ATOMIC NUMBER" in line:
                    try:
                        start_idx = line.find("ATOMIC NUMBER") + len("ATOMIC NUMBER")
                        end_idx = line.find(",   NUMBER OF ELECTRONS")
                        if end_idx > start_idx:
                            Z = int(line[start_idx:end_idx].strip())
                    except ValueError:
                        logger.warning(
                            f"Could not parse atomic number from line {line_number}"
                        )

                if E == 0 and "NUMBER OF ELECTRONS" in line:
                    try:
                        start_idx = line.find("NUMBER OF ELECTRONS") + len(
                            "NUMBER OF ELECTRONS"
                        )
                        E = int(line[start_idx:].strip().rstrip("\n"))
                    except ValueError:
                        logger.warning(
                            f"Could not parse electron number from line {line_number}"
                        )

                if "CPU TIME" in line:
                    line_parts = line.split(" ")
                    cleaned = [x.strip() for x in line_parts if x.strip() != ""]
                    if len(cleaned) > 7:
                        if not (cleaned[2] == "0.000" and cleaned[7] == "0.000"):
                            cput = f"{cleaned[2]} {cleaned[3]}"
                            tcput = f"{cleaned[7]} {cleaned[8]}"

    except Exception as e:
        logger.error(f"Error reading AUTOSTRUCTURE file: {e}")
        raise

    # Build metadata
    closed_config = _get_closed_shells(configs_dictionary.get("1", ""))
    metadata: Dict[str, Any] = {
        "Atomic number": Z,
        "Number of electrons": E,
        "Closed": closed_config[0] if closed_config else "N/A",
        "Ground state energy (Ry)": gs,
        "CPU time": cput,
        "Total CPU time": tcput,
    }

    logger.info(f"Successfully read {len(df_transitions)} transitions for Z={Z}, E={E}")

    # Optionally save to file
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("########\n")
            df_transitions.to_csv(f, sep="\t", index=False)
        logger.info(f"Saved transitions data to {output_path}")

    return df_transitions, metadata


def read_as_lambdas(filename: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract lambda (scaling) parameters from AUTOSTRUCTURE 'olg' output file.

    Parameters
    ----------
    filename : str or Path
        Path to the AUTOSTRUCTURE 'olg' output file

    Returns
    -------
    nl_array : np.ndarray
        Array of (n, l) orbital identifiers, shape (N, 2)
    lambda_array : np.ndarray
        Array of lambda scaling parameters, shape (N,)

    Examples
    --------
    >>> nl, lambdas = read_as_lambdas('output.olg')
    >>> print(f"Orbital (n={nl[0,0]}, l={nl[0,1]}): lambda={lambdas[0]:.6f}")

    Notes
    -----
    Uses NumPy 2.x compatible array creation. Adapted from read_adjust_from_olg.py
    by Tomás Campante.
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"AUTOSTRUCTURE output file not found: {filepath}")

    logger.info(f"Reading lambda parameters from {filepath}")

    nl_list: List[List[int]] = []
    lambda_list: List[float] = []

    section_found = False

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                if "(ADJUST, REND, 3 LAST P)" in line:
                    section_found = True
                    continue

                if section_found:
                    if line.strip() == "" or "MATRIX" in line:
                        break

                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            n = int(parts[0])
                            l = int(parts[1])
                            lambda_val = float(parts[2])
                            nl_list.append([n, l])
                            lambda_list.append(lambda_val)
                        except ValueError:
                            continue

    except Exception as e:
        logger.error(f"Error reading lambda parameters: {e}")
        raise

    if not nl_list:
        logger.warning("No lambda parameters found in file")
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    # Use NumPy 2.x compatible array creation with explicit dtypes
    nl_array = np.array(nl_list, dtype=np.int32)
    lambda_array = np.array(lambda_list, dtype=np.float64)

    logger.info(f"Successfully read {len(lambda_array)} lambda parameters")

    return nl_array, lambda_array


# ============================================================================
# Private helper functions
# ============================================================================


def _parse_configuration(header: str, line: str) -> Tuple[str, str]:
    """Parse AUTOSTRUCTURE configuration format into standard notation."""
    header_parts = header.split()
    data_parts = line.split(":")[1].split()

    orbitals = [
        (int(header_parts[i]), int(header_parts[i + 1]))
        for i in range(0, len(header_parts), 2)
    ]

    l_to_orbital = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}

    configuration = []
    for (n, l), electrons in zip(orbitals, data_parts):
        electrons_int = int(electrons)
        if electrons_int > 0:
            orbital_label = f"{n}{l_to_orbital.get(l, '?')}"
            configuration.append(f"{orbital_label}{electrons_int}")

    level_number = line.split(":")[0].strip()

    return level_number, " ".join(configuration)


def _get_closed_shells(target_orbital: str) -> List[str]:
    """Generate closed shell configurations up to target orbital."""
    if not target_orbital:
        return []

    l_order = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

    match = re.match(r"(\d+)([spdfg])(\d+)", target_orbital)
    if not match:
        logger.warning(f"Invalid orbital format: {target_orbital}")
        return []

    n_target = int(match.group(1))
    l_target = match.group(2)

    orbitals = []
    for n in range(1, n_target + 1):
        for l, l_val in sorted(l_order.items(), key=lambda x: x[1]):
            if l_val >= n:
                continue
            if n == n_target and l_val >= l_order[l_target]:
                break
            max_electrons = 2 * (2 * l_val + 1)
            orbitals.append(f"{n}{l}{max_electrons}")

    return orbitals
