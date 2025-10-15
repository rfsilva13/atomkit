"""
AtomKit utility functions.

General-purpose utilities for element information and ion notation parsing.
These utilities are useful across the entire atomkit package, not specific
to any particular file format or calculation type.
"""

from typing import Dict, Tuple
import logging

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
