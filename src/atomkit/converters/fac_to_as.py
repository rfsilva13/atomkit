"""
FAC to AUTOSTRUCTURE converter module.

Provides functions to convert FAC (Flexible Atomic Code) configuration files
to AUTOSTRUCTURE input format. Extracts configuration information and generates
the necessary flags and orbital occupation matrices for AUTOSTRUCTURE calculations.

Compatible with NumPy 2.x and modern Python 3.13+.

Original Author: Tomás Campante (October 2025)
Adapted by: Ricardo Silva (rfsilva@lip.pt)
Date: October 2025
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..structure import Configuration, Shell

logger = logging.getLogger(__name__)


# Orbital angular momentum mapping (kept for backward compatibility, but Shell class is preferred)
ORBITAL_MAP: Dict[str, int] = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
    "h": 5,
    "i": 6,
    "k": 7,
}
"""Mapping from orbital symbol (s, p, d, f, ...) to quantum number l."""


def parse_orbital(orbital_string: str) -> Tuple[int, int, int]:
    """
    Parse a single orbital string into quantum numbers and occupation.

    REFACTORED: Now uses atomkit's Shell class for parsing to ensure
    consistency with the rest of the package and avoid code duplication.

    Extracts the principal quantum number (n), orbital angular momentum (l),
    and electron occupation from orbital strings like '3d10', '4s2', etc.

    Parameters
    ----------
    orbital_string : str
        Orbital notation string (e.g., '3d10', '4s2', '5p6')

    Returns
    -------
    tuple of (int, int, int)
        Tuple containing:
        - n : Principal quantum number
        - l : Orbital angular momentum quantum number
        - occupation : Number of electrons in the orbital

    Examples
    --------
    >>> parse_orbital('3d10')
    (3, 2, 10)
    >>> parse_orbital('4s2')
    (4, 0, 2)
    >>> parse_orbital('5p6')
    (5, 1, 6)

    Notes
    -----
    Preserves original parsing logic from Tomás Campante's FAC2AS.py
    but now delegates to Shell.from_string() for consistency.
    """
    try:
        shell = Shell.from_string(orbital_string)
        return (shell.n, shell.l_quantum, shell.occupation)
    except (ValueError, KeyError) as e:
        raise ValueError(f"Cannot parse orbital string: {orbital_string}") from e


def read_fac_file(filepath: str | Path) -> List[str]:
    """
    Read a FAC input file and return lines as a list.

    Parameters
    ----------
    filepath : str or Path
        Path to the FAC input file

    Returns
    -------
    list of str
        List of lines from the file

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"FAC file not found: {filepath}")

    logger.info(f"Reading FAC file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    return lines


def extract_closed_shells(fac_lines: List[str]) -> int:
    """
    Extract the number of closed shells from FAC file lines.

    Searches for lines containing 'Closed(' and counts the number of
    closed subshells. This corresponds to the KCOR2 flag in AUTOSTRUCTURE.

    Parameters
    ----------
    fac_lines : list of str
        Lines from a FAC input file

    Returns
    -------
    int
        Number of closed subshells (KCOR2 value)

    Examples
    --------
    >>> lines = ["Closed('1s 2s 2p')", "Config('MR', '3s 3p')"]
    >>> extract_closed_shells(lines)
    3
    """
    closed_lines = [line for line in fac_lines if "Closed(" in line]

    if not closed_lines:
        logger.warning("No closed shells found in FAC file")
        return 0

    # Count number of orbitals in the first Closed() line
    # Assuming format: Closed('1s 2s 2p') or similar
    closed_orbitals = closed_lines[0].split(" ")
    kcor2 = len(closed_orbitals)

    logger.info(f"Found {kcor2} closed subshells")
    return kcor2


def extract_configurations(fac_lines: List[str], label: str = "MR") -> List[str]:
    """
    Extract configuration strings from FAC file for a specific label.

    Finds all configuration lines with the specified label and extracts
    the orbital configuration strings.

    Parameters
    ----------
    fac_lines : list of str
        Lines from a FAC input file
    label : str, optional
        Configuration label to search for (default: 'MR')

    Returns
    -------
    list of str
        List of configuration strings (e.g., ['3d10 4s1', '3d9 4s2'])

    Examples
    --------
    >>> lines = ["Config('MR', '3d10 4s1')", "Config('MR', '3d9 4s2')"]
    >>> extract_configurations(lines, 'MR')
    ['3d10 4s1', '3d9 4s2']
    """
    # Find lines starting with Config('label
    config_prefix = f"Config('{label}"
    config_lines = [
        line for line in fac_lines if line[: len(config_prefix)] == config_prefix
    ]

    if not config_lines:
        logger.warning(f"No configurations found with label '{label}'")
        return []

    # Extract configuration strings between ', ' and final ')
    configurations = []
    for line in config_lines:
        # Find the configuration string after ', '
        start_idx = line.find("', '") + len("', '")
        config_str = line[start_idx:].rstrip("')")
        configurations.append(config_str)

    logger.info(f"Extracted {len(configurations)} configurations with label '{label}'")
    return configurations


def get_unique_orbitals(configurations: List[str]) -> List[Tuple[int, int]]:
    """
    Extract unique (n, l) orbital pairs from configurations.

    REFACTORED: Now uses atomkit's Configuration class for parsing to ensure
    consistency and proper validation of configurations.

    Parses all orbital strings in the configurations and returns a sorted
    list of unique (n, l) pairs.

    Parameters
    ----------
    configurations : list of str
        List of configuration strings

    Returns
    -------
    list of tuple
        Sorted list of (n, l) tuples representing unique orbitals

    Examples
    --------
    >>> configs = ['3d10 4s1', '3d9 4s2']
    >>> get_unique_orbitals(configs)
    [(3, 2), (4, 0)]
    """
    unique_orbitals: Set[Tuple[int, int]] = set()

    for config_str in configurations:
        try:
            # Use Configuration class to parse and validate
            config = Configuration.from_string(config_str)

            # Extract (n, l) from each shell
            for shell in config.shells:
                unique_orbitals.add((shell.n, shell.l_quantum))
        except ValueError as e:
            logger.warning(f"Could not parse configuration '{config_str}': {e}")
            continue

    # Sort by n, then by l
    sorted_orbitals = sorted(unique_orbitals)

    logger.info(f"Found {len(sorted_orbitals)} unique valence orbitals")
    return sorted_orbitals


def build_occupation_matrix(
    configurations: List[str], orbitals: List[Tuple[int, int]]
) -> List[List[int]]:
    """
    Build occupation matrix for AUTOSTRUCTURE input.

    REFACTORED: Now uses atomkit's Configuration class for parsing to ensure
    consistency and proper validation of configurations.

    Creates a matrix where each row represents a configuration and each
    column represents an orbital. Matrix entries are electron occupations.

    Parameters
    ----------
    configurations : list of str
        List of configuration strings
    orbitals : list of tuple
        Sorted list of (n, l) tuples defining column order

    Returns
    -------
    list of list of int
        Occupation matrix where matrix[i][j] is the occupation of
        orbital j in configuration i

    Examples
    --------
    >>> configs = ['3d10 4s1', '3d9 4s2']
    >>> orbs = [(3, 2), (4, 0)]
    >>> build_occupation_matrix(configs, orbs)
    [[10, 1], [9, 2]]
    """
    matrix = []

    for config_str in configurations:
        # Initialize row with zeros
        row = [0] * len(orbitals)

        try:
            # Use Configuration class to parse and validate
            config = Configuration.from_string(config_str)

            # Fill in occupations for each shell
            for shell in config.shells:
                orbital_tuple = (shell.n, shell.l_quantum)

                # Find index of this orbital
                try:
                    idx = orbitals.index(orbital_tuple)
                    row[idx] = shell.occupation
                except ValueError:
                    logger.warning(f"Orbital {orbital_tuple} not found in orbital list")
                    continue

        except ValueError as e:
            logger.warning(f"Could not parse configuration '{config_str}': {e}")
            # Keep row as zeros for invalid configuration
            pass

        matrix.append(row)

    return matrix


def convert_fac_to_as(
    fac_filepath: str | Path,
    config_label: str = "MR",
    output_file: Optional[str | Path] = None,
) -> Dict[str, any]:
    """
    Convert FAC configuration file to AUTOSTRUCTURE input format.

    Main conversion function that reads a FAC file and extracts all information
    needed for AUTOSTRUCTURE input, including flags (ICFG, KCOR2, MXCONF, MXVORB)
    and the orbital occupation matrix.

    Parameters
    ----------
    fac_filepath : str or Path
        Path to the FAC input file
    config_label : str, optional
        Configuration label to extract (default: 'MR')
    output_file : str or Path, optional
        If provided, writes formatted output to this file

    Returns
    -------
    dict
        Dictionary containing:
        - 'icfg' : int, configuration flag (always 0)
        - 'kcor2' : int, number of closed subshells
        - 'mxconf' : int, number of configurations
        - 'mxvorb' : int, number of valence orbitals
        - 'orbitals' : list of (n, l) tuples
        - 'occupation_matrix' : list of lists with occupations
        - 'configurations' : list of original configuration strings

    Examples
    --------
    >>> result = convert_fac_to_as('Fe_I.sf', 'MR')
    >>> print(f"KCOR2={result['kcor2']}")
    >>> print(f"MXCONF={result['mxconf']}")

    Notes
    -----
    Preserves original logic from Tomás Campante's FAC2AS.py script.
    """
    # Read FAC file
    fac_lines = read_fac_file(fac_filepath)

    # Extract closed shells
    kcor2 = extract_closed_shells(fac_lines)

    # Extract configurations
    configurations = extract_configurations(fac_lines, config_label)

    if not configurations:
        raise ValueError(f"No configurations found with label '{config_label}'")

    # Get unique orbitals
    orbitals = get_unique_orbitals(configurations)

    # Build occupation matrix
    matrix = build_occupation_matrix(configurations, orbitals)

    # Prepare result dictionary
    result = {
        "icfg": 0,  # Always 0 for this type of conversion
        "kcor2": kcor2,
        "mxconf": len(configurations),
        "mxvorb": len(orbitals),
        "orbitals": orbitals,
        "occupation_matrix": matrix,
        "configurations": configurations,
    }

    # Write output file if requested
    if output_file:
        write_as_format(result, output_file)

    logger.info(
        f"Conversion complete: {len(configurations)} configurations, "
        f"{len(orbitals)} valence orbitals"
    )

    return result


def write_as_format(result: Dict[str, any], output_file: str | Path) -> None:
    """
    Write AUTOSTRUCTURE format output to file.

    Formats the conversion result in the style expected for AUTOSTRUCTURE
    input files, with flags and occupation matrix.

    Parameters
    ----------
    result : dict
        Result dictionary from convert_fac_to_as()
    output_file : str or Path
        Path to output file
    """
    output_path = Path(output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        # Write flags
        f.write(f"ICFG={result['icfg']}\n")
        f.write(f"KCOR2={result['kcor2']}\n")
        f.write(f"MXCONF={result['mxconf']}\n")
        f.write(f"MXVORB={result['mxvorb']}\n")
        f.write("\n")

        # Write orbital header
        header = " " + "  ".join(f"{n} {l}" for n, l in result["orbitals"])
        f.write(header + "\n")

        # Write occupation matrix
        for row in result["occupation_matrix"]:
            row_str = " " + "   ".join(f"{x:2}" for x in row)
            f.write(row_str + "\n")

    logger.info(f"AUTOSTRUCTURE format written to {output_path}")


def print_as_format(result: Dict[str, any]) -> None:
    """
    Print AUTOSTRUCTURE format output to console.

    Displays the conversion result in the style expected for AUTOSTRUCTURE
    input files, with flags and occupation matrix.

    Parameters
    ----------
    result : dict
        Result dictionary from convert_fac_to_as()
    """
    # Print flags
    print(f"ICFG={result['icfg']}")
    print(f"KCOR2={result['kcor2']}")
    print(f"MXCONF={result['mxconf']}")
    print(f"MXVORB={result['mxvorb']}")
    print()

    # Print orbital header
    header = " " + "  ".join(f"{n} {l}" for n, l in result["orbitals"])
    print(header)

    # Print occupation matrix
    for row in result["occupation_matrix"]:
        row_str = " " + "   ".join(f"{x:2}" for x in row)
        print(row_str)
