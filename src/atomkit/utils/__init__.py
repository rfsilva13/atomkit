"""
AtomKit utility functions.

General-purpose utilities for element information and ion notation parsing.
These utilities are useful across the entire atomkit package, not specific
to any particular file format or calculation type.
"""

import logging
from typing import Dict, Tuple

import mendeleev
import roman

logger = logging.getLogger(__name__)


def get_element_info(element_symbol_or_z: str | int) -> Dict[str, any]:
    """
    Get element information from symbol or atomic number.

    This is a convenient wrapper around the mendeleev package that accepts
    both element symbols and atomic numbers and returns standardized
    element information.

    Parameters
    ----------
    element_symbol_or_z : str or int
        Element symbol (e.g., 'Fe', 'Au') or atomic number (e.g., 26, 79)

    Returns
    -------
    dict
        Dictionary containing:
        - 'symbol': Element symbol
        - 'Z': Atomic number
        - 'name': Element name

    Examples
    --------
    >>> info = get_element_info('Fe')
    >>> print(info['Z'])
    26
    >>> info = get_element_info(79)
    >>> print(info['symbol'])
    Au

    Notes
    -----
    Requires the mendeleev package to be installed.
    """
    if isinstance(element_symbol_or_z, int):
        elem = mendeleev.element(element_symbol_or_z)
    else:
        elem = mendeleev.element(element_symbol_or_z)

    return {"symbol": elem.symbol, "Z": elem.atomic_number, "name": elem.name}


def parse_ion_notation(ion_notation: str) -> Tuple[str, int, int]:
    """
    Parse spectroscopic ion notation into element, charge, and electron count.

    Parses standard spectroscopic/astronomical notation where Roman numerals
    indicate ionization state: I = neutral, II = +1, III = +2, etc.

    Parameters
    ----------
    ion_notation : str
        Ion notation like 'Fe I', 'Fe II', 'Au III', etc.
        Format: 'Element RomanNumeral' with a space separator.

    Returns
    -------
    tuple of (str, int, int)
        - element_symbol: Element symbol (e.g., 'Fe', 'Au')
        - charge: Integer charge (0 for neutral, 1 for +1, etc.)
        - num_electrons: Number of electrons in the ion

    Raises
    ------
    ValueError
        If the ion notation format is invalid or the element is unknown.

    Examples
    --------
    >>> element, charge, electrons = parse_ion_notation('Fe I')
    >>> print(f"{element}: charge={charge}, electrons={electrons}")
    Fe: charge=0, electrons=26

    >>> element, charge, electrons = parse_ion_notation('Fe II')
    >>> print(f"{element}: charge={charge}, electrons={electrons}")
    Fe: charge=1, electrons=25

    >>> element, charge, electrons = parse_ion_notation('Au III')
    >>> print(f"{element}: charge={charge}, electrons={electrons}")
    Au: charge=2, electrons=77

    Notes
    -----
    This follows the spectroscopic notation convention where:
    - I = neutral atom (charge 0)
    - II = singly ionized (+1)
    - III = doubly ionized (+2)
    - etc.

    Requires the mendeleev package for element data and the roman package
    for Roman numeral conversion.
    """
    parts = ion_notation.strip().split()
    if len(parts) != 2:
        raise ValueError(
            f"Invalid ion notation: {ion_notation}. Expected format: 'Element RomanNumeral'"
        )

    element_symbol = parts[0]
    roman_numeral = parts[1]

    # Get atomic number
    info = get_element_info(element_symbol)
    Z = info["Z"]

    # Convert roman numeral to charge
    # I = neutral (0), II = +1, III = +2, etc.
    charge = roman.fromRoman(roman_numeral) - 1

    # Calculate number of electrons
    num_electrons = Z - charge

    logger.info(
        f"Parsed {ion_notation}: {element_symbol} (Z={Z}), charge={charge}, electrons={num_electrons}"
    )

    return element_symbol, charge, num_electrons


def config_obj_to_nonrel(conf) -> str:
    """Convert a Configuration object to a non-relativistic configuration string.

    Sums occupations across any j-split subshells and returns a dot-separated
    configuration string (e.g., '1s2.2s2.2p6'). Designed to be used by
    examples and writers that need FAC-style non-relativistic input.
    """
    from pathlib import Path

    non_rel = {}
    # Accept both Configuration objects and objects with .shells attribute
    shells = getattr(conf, "shells", conf)
    for shell in shells:
        n = shell.n
        l = shell.l_quantum
        occ = shell.occupation
        key = (n, l)
        non_rel[key] = non_rel.get(key, 0) + occ

    sorted_keys = sorted(non_rel.keys())
    l_letters = 'spdfghik'
    parts = [f"{n}{l_letters[l]}{non_rel[(n,l)]}" for n, l in sorted_keys]
    return '.'.join(parts)


def postprocess_sfac_groups(sf_path: str, counts: dict) -> None:
    """Post-process a generated FAC SF file to replace group ids with friendly names.

    Parameters
    ----------
    sf_path : str
        Path to the FAC SF file to edit (will be overwritten).
    counts : dict
        Dictionary describing group counts. Expected keys:
        - 'one_hole': int number of one-hole groups
        - 'two_hole': int number of two-hole groups
        Optional keys:
        - 'one_hole_label': if present (str), use this base label for all one-hole
            groups (e.g., 'one_hole') instead of numbering them.
        - 'two_hole_label': similarly for two-hole groups.

    The function replaces tokens like g0, 'g1', "g2" with readable labels while
    preserving quoting.
    """
    import re
    from pathlib import Path

    mapping = {}
    idx = 0
    mapping[f"g{idx}"] = 'gc'
    idx += 1

    # Determine whether caller requested simple (non-numbered) labels
    one_hole_count = counts.get('one_hole', 0)
    two_hole_count = counts.get('two_hole', 0)
    one_hole_base = counts.get('one_hole_label')
    two_hole_base = counts.get('two_hole_label')

    if one_hole_base:
        # Use the same label for all one-hole groups
        for _ in range(one_hole_count):
            mapping[f"g{idx}"] = one_hole_base
            idx += 1
    else:
        for i in range(1, one_hole_count + 1):
            mapping[f"g{idx}"] = f"one_hole_{i}"
            idx += 1

    if two_hole_base:
        for _ in range(two_hole_count):
            mapping[f"g{idx}"] = two_hole_base
            idx += 1
    else:
        for i in range(1, two_hole_count + 1):
            mapping[f"g{idx}"] = f"two_hole_{i}"
            idx += 1

    text = Path(sf_path).read_text()

    def repl(match):
        key = match.group(0)
        stripped = key.strip("'\"")
        new = mapping.get(stripped, stripped)
        if key.startswith("'") or key.startswith('"'):
            quote = key[0]
            return quote + new + quote
        return new

    pattern = re.compile(r"'g\d+'|\"g\d+\"|\bg\d+\b")
    new_text = pattern.sub(repl, text)
    # Additionally, collapse duplicate entries inside Python-style lists
    # produced in SF files, e.g. "['gc', 'one_hole', 'one_hole']" -> "['gc','one_hole']"
    list_pattern = re.compile(r"\[([^\]]+)\]")

    def dedupe_list(match):
        content = match.group(1)
        parts = [p.strip() for p in content.split(',') if p.strip()]
        seen = set()
        out = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return '[' + ', '.join(out) + ']'

    final_text = list_pattern.sub(dedupe_list, new_text)
    Path(sf_path).write_text(final_text)
