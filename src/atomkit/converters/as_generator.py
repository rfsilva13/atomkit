"""
AUTOSTRUCTURE configuration generator module.

Provides functions to generate electronic configurations and excitations for
AUTOSTRUCTURE calculations. This module helps create valence electron
configurations with systematic excitations for atomic structure computations.

Compatible with modern Python 3.13+ and integrates with atomkit definitions.

Original Author: Tomás Campante (October 2025)
Adapted by: Ricardo Silva (rfsilva@lip.pt)
Date: October 2025

Note:
    The original author recommends using AUTOSTRUCTURE's built-in ICFG=1
    compact configuration selection instead of this generator when possible.
    Use this module when you need explicit control over configurations.

Refactored: Now uses atomkit's Configuration class for parsing, validation,
and excitation generation to avoid code duplication and ensure consistency
across the package.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from ..constants import ANGULAR_MOMENTUM_MAP, L_QUANTUM_MAP, L_SYMBOLS
from ..structure import Configuration, Shell
from ..utils import get_element_info, parse_ion_notation

logger = logging.getLogger(__name__)


def configurations_to_autostructure(
    configurations: List[Configuration] | List[str],
    core: Optional[Union[Configuration, str]] = None,
    last_core_orbital: str = None,
    output_file: Optional[str | Path] = None,
) -> Dict[str, any]:
    """
    Convert configurations to AUTOSTRUCTURE input format.

    This function is completely code-agnostic until the final formatting step.
    It accepts configurations in two flexible formats:

    1. **Full configurations** (core + valence together)
    2. **Core + valence** separately (more flexible)

    Parameters
    ----------
    configurations : list of Configuration or list of str
        EITHER:
        - Full configuration objects/strings (e.g., "1s2 2s2 2p6 3d10 4s2")
        - Valence-only configuration objects/strings (e.g., "3d10 4s2") when `core` is provided
    core : Configuration or str, optional
        Core configuration to prepend to all valence configurations.
        If provided, `configurations` are treated as valence-only.
        If None, `configurations` must be full configurations.
    last_core_orbital : str, optional
        Last core orbital prefix (e.g., '3p', '4p').
        Required to separate core from valence in AUTOSTRUCTURE format.
        If None and full configs provided, will auto-detect from first configuration.
    output_file : str or Path, optional
        If provided, write formatted output to this file

    Returns
    -------
    dict
        Dictionary containing:
        - 'kcor1': Core flag (always 1)
        - 'kcor2': Number of core orbitals
        - 'mxconf': Number of configurations
        - 'mxvorb': Number of valence orbitals
        - 'orbitals': List of (n, l) tuples for valence orbitals
        - 'occupation_matrix': Occupation matrix for valence electrons
        - 'configurations': Configuration strings

    Examples
    --------
    **Example 1: Full configurations (code-agnostic approach)**

    >>> from atomkit import Configuration
    >>> # Generate using general excitation methods
    >>> ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d6 4s2")
    >>> excited = ground.generate_excitations(
    ...     target_shells=["4p", "3d", "5s"],
    ...     excitation_level=1,
    ...     source_shells=["3d", "4s"]
    ... )
    >>> all_configs = [ground] + excited
    >>>
    >>> # Convert to AUTOSTRUCTURE format
    >>> result = configurations_to_autostructure(
    ...     all_configs,
    ...     last_core_orbital='3p'
    ... )

    **Example 2: Core + valence (maximum flexibility)**

    >>> # Define core once
    >>> core = Configuration.from_string("1s2 2s2 2p6 3s2 3p6")
    >>>
    >>> # Generate valence configurations independently
    >>> valence1 = Configuration.from_string("3d6 4s2")  # ground
    >>> valence2 = Configuration.from_string("3d6 4s1 4p1")  # excited
    >>> valence3 = Configuration.from_string("3d5 4s2 4p1")  # excited
    >>>
    >>> # Convert to AUTOSTRUCTURE - core is prepended automatically
    >>> result = configurations_to_autostructure(
    ...     [valence1, valence2, valence3],
    ...     core=core,
    ...     last_core_orbital='3p'
    ... )

    **Example 3: Filter and combine (code-agnostic)**

    >>> # Generate different excitation types using general methods
    >>> single_exc = ground.generate_excitations(["4p", "5s"], excitation_level=1)
    >>> double_exc = ground.generate_excitations(["3d"], excitation_level=2, source_shells=["2p"])
    >>>
    >>> # Filter, combine, modify as needed (physics is done!)
    >>> filtered = [c for c in single_exc if "4p" in c.to_string()]
    >>> all_configs = [ground] + filtered + double_exc
    >>>
    >>> # Only now convert to AUTOSTRUCTURE format (I/O)
    >>> result = configurations_to_autostructure(all_configs, last_core_orbital='3p')

    Notes
    -----
    **Philosophy - Complete Code Agnosticism:**

    This function embodies the atomkit philosophy:
    1. **Physics (Configuration class)**: Generate configurations using general methods
       - `generate_excitations()` - any type of excitation
       - `generate_hole_configurations()` - holes
       - `generate_recombined_configurations()` - autoionization
       - Filter, modify, combine as needed
    2. **I/O (this converter)**: Only format for AUTOSTRUCTURE at the final step

    The physics is completely independent of AUTOSTRUCTURE until this formatting step.
    You can use the same configurations for FAC, GRASP, or any other code.

    See Also
    --------
    Configuration.generate_excitations : General excitation generation
    Configuration.generate_hole_configurations : Generate hole states
    Configuration.generate_recombined_configurations : Autoionization states
    """
    # Handle core + valence format
    if core is not None:
        # Convert core to string
        if isinstance(core, Configuration):
            core_str = core.to_string(separator=" ")
        else:
            core_str = core

        # Prepend core to all valence configurations
        full_config_strings = []
        for val_config in configurations:
            if isinstance(val_config, Configuration):
                val_str = val_config.to_string(separator=" ")
            else:
                val_str = val_config
            # Combine core + valence as strings
            full_config_str = f"{core_str} {val_str}"
            full_config_strings.append(full_config_str)

        config_strings = full_config_strings
        logger.info(f"Combined core with {len(config_strings)} valence configurations")
    else:
        # Convert Configuration objects to strings if needed
        if configurations and isinstance(configurations[0], Configuration):
            config_strings = [c.to_string(separator=" ") for c in configurations]
        else:
            config_strings = list(configurations)

    if not config_strings:
        raise ValueError("No configurations provided")

    # Auto-detect last_core_orbital if not provided
    if last_core_orbital is None:
        # Use the first orbital in the first configuration as a heuristic
        # This assumes the configurations start with the last core orbital
        first_config = config_strings[0]
        first_orbital = first_config.split()[0]
        # Extract just the orbital notation (e.g., "3p6" -> "3p", "4s2" -> "4s")
        # Keep the principal quantum number (n) and angular momentum letter (l)
        import re

        match = re.match(r"(\d+[spdfgh])", first_orbital)
        if match:
            last_core_orbital = match.group(1)
        else:
            raise ValueError(
                f"Could not auto-detect last_core_orbital from '{first_orbital}'. "
                "Please specify explicitly."
            )
        logger.info(f"Auto-detected last_core_orbital: {last_core_orbital}")

    # Format using existing function
    result = format_as_input(config_strings, last_core_orbital)

    # Write to file if requested
    if output_file:
        write_as_format(result, output_file)
        logger.info(f"Wrote {result['mxconf']} configurations to {output_file}")

    return result


def format_as_input(
    configurations: List[str], last_core_orbital: str
) -> Dict[str, any]:
    """
    Format configurations for AUTOSTRUCTURE input.

    Extracts valence orbitals and creates the occupation matrix needed
    for AUTOSTRUCTURE input files.

    Parameters
    ----------
    configurations : list of str
        List of configuration strings
    last_core_orbital : str
        Last core orbital prefix (e.g., '3p', '4p')

    Returns
    -------
    dict
        Dictionary containing:
        - 'kcor1': Core flag (always 1)
        - 'kcor2': Number of core orbitals
        - 'mxconf': Number of configurations
        - 'mxvorb': Number of valence orbitals
        - 'orbitals': List of (n, l) tuples for valence orbitals
        - 'occupation_matrix': Occupation matrix for valence electrons
        - 'configurations': Original configuration strings

    Examples
    --------
    >>> configs = ['3d10 4s2', '3d9 4s2 4p1']
    >>> result = format_as_input(configs, '3p')
    >>> print(result['mxconf'])
    2
    >>> print(result['mxvorb'])
    3
    """
    valence_orbital_set: Set[Tuple[int, int]] = set()
    parsed_valences = []

    # Find last core orbital index
    first_config = configurations[0]
    first_orbitals = first_config.split()
    last_core_idx = -1

    for i, orb_str in enumerate(first_orbitals):
        if orb_str.startswith(last_core_orbital):
            last_core_idx = i

    if last_core_idx == -1:
        raise ValueError(
            f"Core orbital '{last_core_orbital}' not found in configuration: {first_config}"
        )

    # Extract valence orbitals from all configurations
    for config_str in configurations:
        orbitals = config_str.split()

        # Find last core orbital in this configuration
        this_last_core_idx = -1
        for i, orb_str in enumerate(orbitals):
            if orb_str.startswith(last_core_orbital):
                this_last_core_idx = i

        if this_last_core_idx == -1:
            raise ValueError(
                f"Core orbital '{last_core_orbital}' not found in configuration: {config_str}"
            )

        # Extract valence orbitals (from last_core_idx onward)
        valence_orbs = orbitals[this_last_core_idx:]

        # Parse using Shell class
        parsed = []
        for orb_str in valence_orbs:
            shell = Shell.from_string(orb_str)
            parsed.append((shell.n, shell.l_quantum, shell.occupation))
            valence_orbital_set.add((shell.n, shell.l_quantum))

        parsed_valences.append(parsed)

    # Sort orbitals by n, then l
    sorted_orbitals = sorted(valence_orbital_set)

    # Build occupation matrix
    occupation_matrix = []
    for parsed_valence in parsed_valences:
        row = [0] * len(sorted_orbitals)
        for n, l, occupation in parsed_valence:
            # n and l are already integers from Shell
            idx = sorted_orbitals.index((n, l))
            row[idx] = occupation
        occupation_matrix.append(row)

    result = {
        "kcor1": 1,
        "kcor2": last_core_idx,
        "mxconf": len(configurations),
        "mxvorb": len(sorted_orbitals),
        "orbitals": sorted_orbitals,
        "occupation_matrix": occupation_matrix,
        "configurations": configurations,
    }

    logger.info(
        f"Formatted {len(configurations)} configurations with {len(sorted_orbitals)} valence orbitals"
    )

    return result


def print_as_format(result: Dict[str, any]) -> None:
    """
    Print AUTOSTRUCTURE format output to console.

    Parameters
    ----------
    result : dict
        Result dictionary from format_as_input()
    """
    print(f"KCOR1={result['kcor1']}")
    print(f"KCOR2={result['kcor2']}")
    print(f"MXCONF={result['mxconf']}")
    print(f"MXVORB={result['mxvorb']}")
    print()

    # Print orbital header
    header_parts = []
    for n, l in result["orbitals"]:
        header_parts.append(f"{n:<2}{l:<2}  ")
    print("".join(header_parts).strip())

    # Print occupation matrix
    for row in result["occupation_matrix"]:
        row_str = " " + "  ".join(f"{occ:<4}" for occ in row)
        print(row_str)


def write_as_format(result: Dict[str, any], output_file: str | Path) -> None:
    """
    Write AUTOSTRUCTURE format output to file.

    Parameters
    ----------
    result : dict
        Result dictionary from format_as_input()
    output_file : str or Path
        Output file path
    """
    output_path = Path(output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"KCOR1={result['kcor1']}\n")
        f.write(f"KCOR2={result['kcor2']}\n")
        f.write(f"MXCONF={result['mxconf']}\n")
        f.write(f"MXVORB={result['mxvorb']}\n")
        f.write("\n")

        # Write orbital header
        header_parts = []
        for n, l in result["orbitals"]:
            header_parts.append(f"{n:<2}{l:<2}  ")
        f.write("".join(header_parts).strip() + "\n")

        # Write occupation matrix
        for row in result["occupation_matrix"]:
            row_str = " " + "  ".join(f"{occ:<4}" for occ in row)
            f.write(row_str + "\n")

    logger.info(f"Wrote AUTOSTRUCTURE format to {output_path}")


def generate_as_configurations(
    ion_notation: str,
    ground_config: str,
    valence_orbitals: str,
    max_n: int,
    max_l_symbol: str,
    output_file: Optional[str | Path] = None,
) -> Dict[str, any]:
    """
    Complete workflow to generate AUTOSTRUCTURE configurations.

    .. deprecated:: 1.0
        This convenience function mixes configuration generation (physics) with
        format conversion (I/O). For better separation of concerns, use:

        1. Configuration.generate_autostructure_configurations() to generate configs
        2. configurations_to_autostructure() to format for AUTOSTRUCTURE

        This provides more flexibility to manipulate configurations before formatting.

    Parameters
    ----------
    ion_notation : str
        Ion notation (e.g., 'Fe I', 'Nd II')
    ground_config : str
        Ground state configuration string
    valence_orbitals : str
        Space-separated valence orbital prefixes
    max_n : int
        Maximum n for excitations
    max_l_symbol : str
        Maximum l symbol for excitations
    output_file : str or Path, optional
        If provided, write output to this file

    Returns
    -------
    dict
        Formatted result dictionary with configurations and occupation matrix

    Examples
    --------
    **Old way (this function):**

    >>> result = generate_as_configurations(
    ...     'Fe I',
    ...     '1s2 2s2 2p6 3s2 3p6 3d6 4s2',
    ...     '3d 4s',
    ...     max_n=5,
    ...     max_l_symbol='d'
    ... )

    **Recommended new way:**

    >>> from atomkit import Configuration
    >>> from atomkit.converters import configurations_to_autostructure
    >>>
    >>> # Step 1: Generate configurations (physics)
    >>> config = Configuration.from_string('1s2 2s2 2p6 3s2 3p6 3d6 4s2')
    >>> all_configs = config.generate_autostructure_configurations(['3d', '4s'], 5, 2)
    >>>
    >>> # Step 2: Format for AUTOSTRUCTURE (I/O)
    >>> result = configurations_to_autostructure(all_configs, last_core_orbital='3p')

    See Also
    --------
    configurations_to_autostructure : Recommended replacement (format only)
    Configuration.generate_autostructure_configurations : Generate configurations

    Notes
    -----
    This function will be removed in version 2.0. Please migrate to the
    two-step workflow for better flexibility and cleaner architecture.
    """
    import warnings

    warnings.warn(
        "generate_as_configurations() is deprecated. "
        "Use Configuration.generate_autostructure_configurations() followed by "
        "configurations_to_autostructure() for better separation of concerns. "
        "This function will be removed in version 2.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Parse ion notation
    element, charge, num_electrons = parse_ion_notation(ion_notation)
    logger.info(
        f"Generating configurations for {ion_notation} ({num_electrons} electrons)"
    )

    # Validate ground state using Configuration class
    try:
        config = Configuration.from_string(ground_config)
        if config.total_electrons() != num_electrons:
            raise ValueError(
                f"Ground state has {config.total_electrons()} electrons, "
                f"expected {num_electrons} for {ion_notation}"
            )
    except ValueError as e:
        raise ValueError(f"Invalid ground state configuration: {ground_config}") from e

    # Determine last core orbital
    valence_list = valence_orbitals.split()
    last_core_orbital = valence_list[0]

    logger.info(f"Ground state: {ground_config}")
    logger.info(f"Valence orbitals: {valence_orbitals}")
    logger.info(f"Excitation limits: n <= {max_n}, l <= {max_l_symbol}")

    # Generate excitations using Configuration method
    max_l = ANGULAR_MOMENTUM_MAP[max_l_symbol]
    all_config_objs = config.generate_autostructure_configurations(
        valence_shells=valence_list, max_n=max_n, max_l=max_l
    )

    # Use new function for formatting
    result = configurations_to_autostructure(
        all_config_objs, last_core_orbital=last_core_orbital, output_file=output_file
    )

    return result
