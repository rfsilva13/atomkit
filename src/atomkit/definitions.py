# atomkit/src/atomkit/definitions.py

"""
Defines fundamental constants, mappings, and validation functions
related to atomic structure used throughout the atomkit library.
"""

from typing import Optional, Union

import scipy.constants as const  # Import scipy constants

# --- Physical Constants ---
# Conversion factor from eV to cm⁻¹ (CODATA 2018)
EV_TO_CM1 = 8065.544004795713
# Planck's constant * speed of light (for eV to Angstrom conversion)
HC_EV_ANGSTROM = 12398.419843320027  # eV·Å
# Constant for Line Strength calculation S(a.u.) = (gf * lambda(A)) / LINE_STRENGTH_CONST
LINE_STRENGTH_CONST = 303.756
# Hartree energy in eV (CODATA 2018)
HARTREE_EV = const.physical_constants["Hartree energy in eV"][0]
# Rydberg constant times hc in eV (CODATA 2018)
RYD_EV = const.physical_constants["Rydberg constant times hc in eV"][0]
# Fine structure constant (CODATA 2018)
ALPHA = const.fine_structure
# Speed of light in m/s (CODATA 2018)
C_SI = const.c
# Electron mass in kg (CODATA 2018)
M_E_SI = const.m_e
# Elementary charge in C (CODATA 2018)
E_SI = const.e
# Planck constant in J*s (CODATA 2018)
H_SI = const.h
# Reduced Planck constant in J*s (CODATA 2018)
HBAR_SI = const.hbar

# --- Unit Definitions ---
# Allowed units for user input/output requests
ALLOWED_ENERGY_UNITS = ["ev", "cm-1", "ry", "hz", "j", "ha"]
ALLOWED_WAVELENGTH_UNITS = ["a", "nm", "m", "cm"]  # 'a' for Angstrom

# --- Atomic Structure Definitions ---

# Spectroscopic symbols for orbital angular momentum (l)
L_SYMBOLS: list[str] = [
    "s",
    "p",
    "d",
    "f",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "q",
    "r",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
"""List of standard spectroscopic symbols for orbital angular momentum (l)."""

# Mapping from l symbol (e.g., 'p') to l quantum number (e.g., 1)
ANGULAR_MOMENTUM_MAP: dict[str, int] = {
    symbol: l_val for l_val, symbol in enumerate(L_SYMBOLS)
}
"""Mapping from l symbol (e.g., 'p') to l quantum number (e.g., 1)."""

# Mapping from l quantum number (e.g., 1) to l symbol (e.g., 'p')
# Note: This map is limited by L_SYMBOLS. Higher l values will need direct handling.
L_QUANTUM_MAP: dict[int, str] = {
    l_val: symbol for symbol, l_val in ANGULAR_MOMENTUM_MAP.items()
}
"""Mapping from l quantum number (e.g., 1) to l symbol (e.g., 'p'). Limited by L_SYMBOLS."""


# --- Function to generate SHELL_LABEL_MAP ---
def _build_shell_label_map(max_n: int = 7) -> dict[str, str]:
    """
    Generates the mapping from shell structure strings to X-ray labels.

    Args:
        max_n: The maximum principal quantum number to generate labels for.

    Returns:
        A dictionary mapping shell strings (e.g., "1s", "2p-") to labels ("K", "L2").
    """
    shell_map: dict[str, str] = {}
    n_to_letter: dict[int, str] = {
        1: "K",
        2: "L",
        3: "M",
        4: "N",
        5: "O",
        6: "P",
        7: "Q",
    }

    for n in range(1, max_n + 1):
        principal_letter = n_to_letter.get(n)
        if not principal_letter:
            break

        if n == 1:
            shell_map["1s"] = "K"
            continue

        current_subshell_index = 1
        for l_quantum in range(n):
            l_symbol = L_QUANTUM_MAP.get(l_quantum)
            if l_symbol is None:
                break

            if l_quantum == 0:
                shell_key = f"{n}{l_symbol}"
                shell_map[shell_key] = f"{principal_letter}{current_subshell_index}"
                current_subshell_index += 1
            else:
                shell_key_minus = f"{n}{l_symbol}-"
                shell_map[shell_key_minus] = (
                    f"{principal_letter}{current_subshell_index}"
                )
                index_minus = current_subshell_index
                current_subshell_index += 1

                shell_key_plus = f"{n}{l_symbol}+"
                shell_map[shell_key_plus] = (
                    f"{principal_letter}{current_subshell_index}"
                )
                index_plus = current_subshell_index
                current_subshell_index += 1

                shell_key_combined = f"{n}{l_symbol}"
                shell_map[shell_key_combined] = (
                    f"{principal_letter}{index_minus}{index_plus}"
                )

    return shell_map


# Generate the map up to n=7 by default
SHELL_LABEL_MAP: dict[str, str] = _build_shell_label_map(max_n=7)
"""
Mapping from simple shell structure string (e.g., '1s', '2p-') to X-ray notation label.
Generated automatically. Includes relativistic ('nl+', 'nl-') and combined ('nl') labels.
"""
# --- End Generated Map ---


def validate_j_quantum(l_quantum: int, j_quantum: Union[float, int]) -> float:
    """
    Validates the total angular momentum quantum number (j) against the
    orbital angular momentum quantum number (l).

    Args:
        l_quantum: The orbital angular momentum quantum number (integer >= 0).
        j_quantum: The total angular momentum quantum number to validate.

    Returns:
        The validated j_quantum as a float.

    Raises:
        ValueError: If j_quantum is invalid or inconsistent with l_quantum.
        TypeError: If j_quantum is not float or int.
    """
    if not isinstance(j_quantum, (float, int)):
        raise TypeError(
            f"j quantum number must be float or int, got type {type(j_quantum)}"
        )
    j_float = float(j_quantum)
    if j_float < 0.5:
        raise ValueError(f"j quantum number must be >= 0.5, got {j_float}")

    two_j = j_float * 2
    if abs(two_j - round(two_j)) > 1e-6:
        raise ValueError(
            f"Expected 2*j to be an integer, but got {two_j} for j={j_float}"
        )

    if l_quantum == 0:
        if abs(j_float - 0.5) > 1e-6:
            raise ValueError(f"j must be 0.5 for s shell (l=0), got j={j_float}")
    else:
        j_minus = l_quantum - 0.5
        j_plus = l_quantum + 0.5
        is_j_minus = abs(j_float - j_minus) < 1e-6
        is_j_plus = abs(j_float - j_plus) < 1e-6
        if not (is_j_minus or is_j_plus):
            raise ValueError(
                f"Invalid j={j_float} for l={l_quantum}. Must be approximately {j_minus} or {j_plus}."
            )

    return j_float


def get_max_shell_occupation(
    l_quantum: int, j_quantum: Optional[Union[float, int]] = None
) -> int:
    """
    Calculates the maximum electron occupancy for a given shell or subshell.

    Args:
        l_quantum: The orbital angular momentum quantum number (integer >= 0).
        j_quantum: The total angular momentum quantum number (optional).

    Returns:
        Maximum number of electrons allowed in the shell/subshell.

    Raises:
        ValueError: If l_quantum is negative, or if j_quantum is invalid/inconsistent.
        TypeError: If j_quantum type is invalid.
    """
    if not isinstance(l_quantum, int) or l_quantum < 0:
        raise ValueError(
            f"l quantum number must be a non-negative integer, got {l_quantum}."
        )

    if j_quantum is not None:
        valid_j_float = validate_j_quantum(l_quantum, j_quantum)
        two_j = valid_j_float * 2
        return round(two_j) + 1
    else:
        return 2 * (2 * l_quantum + 1)


def get_min_n_for_l(l_quantum: int) -> int:
    """
    Gets the minimum allowed principal quantum number (n) for a given
    orbital angular momentum quantum number (l). Rule: n >= l + 1.

    Args:
        l_quantum: The orbital angular momentum quantum number (integer >= 0).

    Returns:
        The minimum valid n value (integer).

    Raises:
        ValueError: If l_quantum is negative.
        TypeError: If l_quantum is not an integer.
    """
    if not isinstance(l_quantum, int):
        raise TypeError(
            f"l quantum number must be an integer, got type {type(l_quantum)}."
        )
    if l_quantum < 0:
        raise ValueError(f"l quantum number must be non-negative, got {l_quantum}.")
    return l_quantum + 1
