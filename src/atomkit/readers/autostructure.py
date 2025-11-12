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

                # Read fine structure energies - only parse jj coupling format with complete quantum numbers
                if (
                    "K" in line
                    and "LV" in line
                    and "2J" in line
                    and "(EK-E1)/RY" in line
                    and "E1/RY" in line
                ):
                    levels_index_found = True
                    header_tokens = line.split()
                    columns = {value: idx for idx, value in enumerate(header_tokens)}
                    # Extract ground state energy from the header line
                    gs = header_tokens[-1] if header_tokens else "N/A"
                    continue

                if levels_index_found and len(line.strip()) == 0:
                    levels_index_found = False

                # Stop parsing levels when we hit CORE CONTRIB or certain other markers
                if levels_index_found and (
                    "CORE CONTRIB" in line or "FUNCTIONAL" in line or "ZETA(" in line
                ):
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
                            int(parts[columns["2J"]])
                            if "2J" in columns
                            and columns["2J"] is not None
                            and len(parts) > columns["2J"]
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
                        # Convert excitation energies to absolute energies for jj coupling format
                        if data_row["Level (Ry)"] is not None and gs != "N/A":
                            try:
                                gs_energy = float(gs)
                                data_row["Level (Ry)"] = (
                                    gs_energy + data_row["Level (Ry)"]
                                )
                            except (ValueError, TypeError):
                                pass
                        # Only include levels with negative absolute energies (bound states)
                        if (
                            data_row["Level (Ry)"] is not None
                            and data_row["Level (Ry)"] >= 0
                        ):
                            continue
                        data_row["P"] = (
                            int(parts[columns["P"]])
                            if "P" in columns and len(parts) > columns["P"]
                            else (
                                0 if data_row["2*S+1"] and data_row["2*S+1"] >= 0 else 1
                            )
                        )
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
                    # Skip '1-DATA' token which is not a real data column
                    header_tokens = [
                        token for token in header_tokens if token != "1-DATA"
                    ]
                    columns = {value: idx for idx, value in enumerate(header_tokens)}
                    continue

                if transitions_index_found and len([p for p in line.split() if p]) == 0:
                    transitions_index_found = False

                if transitions_index_found and columns:
                    parts = [p for p in line.split() if p]
                    try:
                        data_row: Dict[str, Any] = {}
                        data_row["index"] = (
                            int(parts[columns["E"]])
                            if "E" in columns and len(parts) > columns["E"]
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
                            float(parts[columns["A(EK)*SEC"]])
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
                            float(np.log10(abs(gf_val)))
                            if gf_val is not None and gf_val != 0
                            else None
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


def read_as_terms(
    filename: str | Path, output_file: Optional[str | Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read term energies from AUTOSTRUCTURE TERMS file or olg output file.

    Parameters
    ----------
    filename : str or Path
        Path to the AUTOSTRUCTURE TERMS file or olg output file
    output_file : str or Path, optional
        If provided, saves the parsed data to this file

    Returns
    -------
    df_terms : pd.DataFrame
        DataFrame with term data containing columns:
        - '2*S+1': total spin multiplicity
        - 'L': orbital angular momentum quantum number
        - 'P': parity (0 or 1)
        - 'CF': configuration index
        - 'NI': number of levels in term
        - 'Energy (Ry)': term energy in Rydbergs
        - 'Configuration': configuration string (if available)
    metadata : dict
        Atomic structure metadata

    Examples
    --------
    >>> df_terms, meta = read_as_terms('TERMS')
    >>> print(df_terms[['2*S+1', 'L', 'P', 'Energy (Ry)']].head())

    Notes
    -----
    Reads from dedicated TERMS files when available, or attempts to extract
    term data from TERMS sections in olg files. TERMS files contain term-averaged
    energies with quantum numbers (2S+1, L, P) rather than individual level data.
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"AUTOSTRUCTURE file not found: {filepath}")

    logger.info(f"Reading AUTOSTRUCTURE terms from {filepath}")

    # Initialize variables
    configs_dictionary: Dict[str, str] = {}
    Z, E = 0, 0
    cput, tcput = "N/A", "N/A"
    gs = "N/A"

    # Create DataFrame
    df_terms = pd.DataFrame(
        columns=["2*S+1", "L", "P", "CF", "NI", "Energy (Ry)", "Configuration"]
    )

    # Check if this is a dedicated TERMS file
    is_terms_file = (
        filepath.name.upper() == "TERMS" or filepath.suffix.upper() == ".TERMS"
    )

    # Parsing state variables
    config_index_found = 0
    terms_found = False
    previous_line = ""
    config_header_line = ""

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):

                # Build configuration dictionary (for olg files)
                if not is_terms_file:
                    if "BASIC CONFIGURATION" in line:
                        config_index_found = line_number

                    if config_index_found > 0 and ": " in line:
                        try:
                            if config_header_line == "":
                                config_header_line = previous_line.strip()
                            line_stripped = line.strip()
                            if line_stripped and line_stripped[0].isdigit():
                                config_info = _parse_configuration(
                                    config_header_line, line
                                )
                                configs_dictionary[config_info[0]] = config_info[1]
                        except (ValueError, IndexError):
                            config_index_found = 0

                    previous_line = line

                # Read terms data
                if is_terms_file:
                    # Dedicated TERMS file format
                    if "S L P" in line and "ENERGY(RYD)" in line:
                        terms_found = True
                        continue

                    if terms_found and line.strip():
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                data_row: Dict[str, Any] = {}
                                data_row["2*S+1"] = int(parts[0])
                                data_row["L"] = int(parts[1])
                                data_row["P"] = int(parts[2])
                                data_row["CF"] = int(parts[3])
                                data_row["NI"] = int(parts[4])
                                data_row["Energy (Ry)"] = float(parts[5])
                                data_row["Configuration"] = (
                                    None  # Will be filled if configs available
                                )

                                df_terms = pd.concat(
                                    [df_terms, pd.DataFrame([data_row])],
                                    ignore_index=True,
                                )
                            except (ValueError, KeyError, IndexError) as e:
                                logger.warning(
                                    f"Skipped invalid term line {line_number}: {line.strip()} ({e})"
                                )
                                continue

                else:
                    # Look for TERMS section in olg file
                    if line.strip() == "TERMS":
                        terms_found = True
                        continue

                    if terms_found and line.strip():
                        # Try to parse as term data (same format as TERMS file)
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                # Check if it looks like term data (reasonable ranges)
                                s, l, p = int(parts[0]), int(parts[1]), int(parts[2])
                                if (
                                    0 <= s <= 10
                                    and 0 <= l <= 10
                                    and 0 <= p <= 1
                                    and len(parts) >= 6
                                ):
                                    data_row: Dict[str, Any] = {}
                                    data_row["2*S+1"] = s
                                    data_row["L"] = l
                                    data_row["P"] = p
                                    data_row["CF"] = int(parts[3])
                                    data_row["NI"] = int(parts[4])
                                    data_row["Energy (Ry)"] = float(parts[5])
                                    data_row["Configuration"] = configs_dictionary.get(
                                        str(data_row["CF"]), None
                                    )

                                    df_terms = pd.concat(
                                        [df_terms, pd.DataFrame([data_row])],
                                        ignore_index=True,
                                    )
                            except (ValueError, KeyError, IndexError):
                                # Not term data, continue
                                continue

                        # Stop at section boundaries
                        if any(
                            x in line for x in ["SUMMARY", "CPU TIME", "END", "****"]
                        ):
                            terms_found = False

                # Retrieve metadata (for olg files)
                if not is_terms_file:
                    if Z == 0 and "ATOMIC NUMBER" in line:
                        try:
                            start_idx = line.find("ATOMIC NUMBER") + len(
                                "ATOMIC NUMBER"
                            )
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
        logger.error(f"Error reading AUTOSTRUCTURE terms: {e}")
        raise

    # Note: Configuration strings could be filled from associated olg file if needed

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

    logger.info(f"Successfully read {len(df_terms)} terms for Z={Z}, E={E}")

    # Optionally save to file
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("########\n")
            df_terms.to_csv(f, sep="\t", index=False)
        logger.info(f"Saved terms data to {output_path}")

    return df_terms, metadata


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


def detect_file_format(filename: str | Path) -> str:
    """
    Detect the format of an atomic structure file.

    Parameters
    ----------
    filename : str or Path
        Path to the atomic structure file

    Returns
    -------
    str
        Format identifier: 'autostructure', 'fac', or 'unknown'

    Examples
    --------
    >>> detect_file_format('output.olg')
    'autostructure'
    >>> detect_file_format('levels.sf')
    'fac'
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Check file extension first
    if filepath.suffix.lower() in [".olg", ".ols"]:
        return "autostructure"
    elif filepath.suffix.lower() in [".sf", ".dat"]:
        return "fac"

    # Check file name
    if filepath.name.upper() in ["OLG", "OLS", "TERMS"]:
        return "autostructure"
    elif filepath.name.upper() in ["SF", "DAT"]:
        return "fac"

    # Check file content
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            # Read first few lines
            lines = []
            for i, line in enumerate(file):
                lines.append(line)
                if i >= 10:  # Check first 10 lines
                    break

            content = "\n".join(lines)

            # AUTOSTRUCTURE indicators
            if any(
                keyword in content
                for keyword in [
                    "AUTOSTRUCTURE",
                    "LEVEL TABLE",
                    "T,2S+1L",
                    "EIGEN-H",
                    "BASIC CONFIGURATION",
                    "RADIAL FUNCTIONS",
                ]
            ):
                return "autostructure"

            # FAC indicators
            if any(
                keyword in content
                for keyword in [
                    "FAC",
                    "Flexible Atomic Code",
                    "LEVELS",
                    "TRANSITIONS",
                    "AUTOIONIZATION",
                    "CONFIGURATION LIST",
                ]
            ):
                return "fac"

            # Check for specific FAC file patterns
            if "CONFIGURATION" in content and "LEVEL" in content:
                # Could be either, but let's check more specifically
                if "2S+1" in content and "L" in content and "ENERGY" in content:
                    return "fac"  # FAC level format

    except Exception:
        pass

    return "unknown"


def get_levels(
    filename: str | Path,
    output_file: Optional[str | Path] = None,
    coupling: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified interface to extract level data from atomic structure files.

    Automatically detects file format (AUTOSTRUCTURE or FAC) and extracts
    level information with appropriate quantum numbers.

    Parameters
    ----------
    filename : str or Path
        Path to the atomic structure file
    output_file : str or Path, optional
        If provided, saves the parsed data to this file
    coupling : str, optional
        Coupling scheme preference: 'jj', 'ls', or None (auto-detect)

    Returns
    -------
    df_levels : pd.DataFrame
        DataFrame with level data. Columns depend on source format:
        - AUTOSTRUCTURE JJ: ['K', 'CF', 'Level (Ry)', '2J', '2*S+1', 'L', 'P']
        - AUTOSTRUCTURE LS: ['K', 'CF', 'Level (Ry)', '2*S+1', 'L', 'P']
        - FAC: ['index', 'configuration', 'term', 'J', 'energy', 'g']
    metadata : dict
        Atomic structure metadata

    Examples
    --------
    >>> df_levels, meta = get_levels('output.olg')
    >>> print(f"Found {len(df_levels)} levels for Z={meta['Atomic number']}")

    >>> df_levels, meta = get_levels('levels.sf', coupling='jj')
    >>> print(df_levels[['configuration', 'term', 'J', 'energy']].head())
    """
    format_type = detect_file_format(filename)

    if format_type == "autostructure":
        return read_as_levels(filename, output_file)
    elif format_type == "fac":
        # Import here to avoid circular imports
        from .levels import read_fac

        # Convert Path to str and split into base + extension
        filename_str = str(filename)
        base_filename = filename_str
        file_extension = ""

        # Handle different FAC file extensions
        if filename_str.endswith(".lev.asc"):
            base_filename = filename_str[:-8]  # Remove '.lev.asc'
            file_extension = ".lev.asc"
        elif filename_str.endswith(".sf"):
            base_filename = filename_str[:-3]  # Remove '.sf'
            file_extension = ".lev.asc"  # FAC levels are typically .lev.asc
        else:
            # Assume it's a base filename without extension
            file_extension = ".lev.asc"

        # Call FAC reader
        df = read_fac(base_filename, file_extension)

        # Create metadata dict for consistency
        metadata = {
            "Atomic number": None,
            "Number of electrons": None,
            "Ground state energy (Ry)": None,
            "CPU time": "N/A",
            "Total CPU time": "N/A",
            "Method": "FAC",
        }

        # Try to extract atomic number from the DataFrame
        if not df.empty and "atomic_number" in df.columns:
            metadata["Atomic number"] = (
                df["atomic_number"].iloc[0]
                if not df["atomic_number"].isna().all()
                else None
            )

        return df, metadata
    else:
        raise ValueError(f"Unknown or unsupported file format for {filename}")


def get_transitions(
    filename: str | Path, output_file: Optional[str | Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified interface to extract transition data from atomic structure files.

    Automatically detects file format (AUTOSTRUCTURE or FAC) and extracts
    radiative transition information.

    Parameters
    ----------
    filename : str or Path
        Path to the atomic structure file
    output_file : str or Path, optional
        If provided, saves the parsed data to this file

    Returns
    -------
    df_transitions : pd.DataFrame
        DataFrame with transition data. Columns depend on source format:
        - AUTOSTRUCTURE: ['index', 'K', 'Klower', 'WAVEL/AE', 'A(K)*SEC', 'F(ABS)', 'log(gf)']
        - FAC: ['lower_level', 'upper_level', 'wavelength', 'A', 'gf', 'S']
    metadata : dict
        Atomic structure metadata

    Examples
    --------
    >>> df_trans, meta = get_transitions('output.olg')
    >>> print(f"Found {len(df_trans)} transitions")

    >>> df_trans, meta = get_transitions('transitions.sf')
    >>> print(df_trans[['wavelength', 'A', 'gf']].head())
    """
    format_type = detect_file_format(filename)

    if format_type == "autostructure":
        return read_as_transitions(filename, output_file)
    elif format_type == "fac":
        # Import here to avoid circular imports
        from .transitions import read_fac_transitions

        # Convert Path to str and split into base + extension
        filename_str = str(filename)
        base_filename = filename_str
        file_extension = ""

        # Handle different FAC file extensions
        if filename_str.endswith(".tr.asc"):
            base_filename = filename_str[:-7]  # Remove '.tr.asc'
            file_extension = ".tr.asc"
        elif filename_str.endswith(".sf"):
            base_filename = filename_str[:-3]  # Remove '.sf'
            file_extension = ".tr.asc"  # FAC transitions are typically .tr.asc
        else:
            # Assume it's a base filename without extension
            file_extension = ".tr.asc"

        # Call FAC reader
        df = read_fac_transitions(base_filename, file_extension)

        # Create metadata dict for consistency
        metadata = {
            "Atomic number": None,
            "Number of electrons": None,
            "Ground state energy (Ry)": None,
            "CPU time": "N/A",
            "Total CPU time": "N/A",
            "Method": "FAC",
        }

        # Try to extract atomic number from the DataFrame
        if not df.empty and "atomic_number" in df.columns:
            metadata["Atomic number"] = (
                df["atomic_number"].iloc[0]
                if not df["atomic_number"].isna().all()
                else None
            )

        return df, metadata
    else:
        raise ValueError(f"Unknown or unsupported file format for {filename}")


def get_terms(
    filename: str | Path, output_file: Optional[str | Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified interface to extract term data from atomic structure files.

    Automatically detects file format and extracts term-averaged energies.
    Currently supports AUTOSTRUCTURE TERMS files.

    Parameters
    ----------
    filename : str or Path
        Path to the atomic structure file (TERMS file for AUTOSTRUCTURE)
    output_file : str or Path, optional
        If provided, saves the parsed data to this file

    Returns
    -------
    df_terms : pd.DataFrame
        DataFrame with term data containing:
        - AUTOSTRUCTURE: ['2*S+1', 'L', 'P', 'CF', 'NI', 'Energy (Ry)', 'Configuration']
    metadata : dict
        Atomic structure metadata

    Examples
    --------
    >>> df_terms, meta = get_terms('TERMS')
    >>> print(f"Found {len(df_terms)} terms")
    >>> print(df_terms[['2*S+1', 'L', 'P', 'Energy (Ry)']].head())
    """
    format_type = detect_file_format(filename)

    if format_type == "autostructure":
        return read_as_terms(filename, output_file)
    else:
        raise ValueError(f"Term extraction not supported for format: {format_type}")


def get_autoionization(
    filename: str | Path, output_file: Optional[str | Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified interface to extract autoionization data from atomic structure files.

    Currently supports FAC autoionization files.

    Parameters
    ----------
    filename : str or Path
        Path to the autoionization file
    output_file : str or Path, optional
        If provided, saves the parsed data to this file

    Returns
    -------
    df_auto : pd.DataFrame
        DataFrame with autoionization data
    metadata : dict
        Atomic structure metadata

    Examples
    --------
    >>> df_auto, meta = get_autoionization('autoionization.sf')
    >>> print(f"Found {len(df_auto)} autoionization transitions")
    """
    format_type = detect_file_format(filename)

    if format_type == "fac":
        # Import here to avoid circular imports
        from .autoionization import read_fac_autoionization

        # Convert Path to str and split into base + extension
        filename_str = str(filename)
        base_filename = filename_str
        file_extension = ""

        # Handle different FAC file extensions
        if filename_str.endswith(".ai.asc"):
            base_filename = filename_str[:-7]  # Remove '.ai.asc'
            file_extension = ".ai.asc"
        elif filename_str.endswith(".sf"):
            base_filename = filename_str[:-3]  # Remove '.sf'
            file_extension = ".ai.asc"  # FAC autoionization are typically .ai.asc
        else:
            # Assume it's a base filename without extension
            file_extension = ".ai.asc"

        # Call FAC reader
        df = read_fac_autoionization(base_filename, file_extension)

        # Create metadata dict for consistency
        metadata = {
            "Atomic number": None,
            "Number of electrons": None,
            "Ground state energy (Ry)": None,
            "CPU time": "N/A",
            "Total CPU time": "N/A",
            "Method": "FAC",
        }

        # Try to extract atomic number from the DataFrame
        if not df.empty and "atomic_number" in df.columns:
            metadata["Atomic number"] = (
                df["atomic_number"].iloc[0]
                if not df["atomic_number"].isna().all()
                else None
            )

        return df, metadata
    else:
        raise ValueError(
            f"Autoionization extraction not supported for format: {format_type}"
        )
