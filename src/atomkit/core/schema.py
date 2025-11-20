"""
Universal data schema for atomic structure calculations.

Defines standard column names that all atomic codes (FAC, AUTOSTRUCTURE, etc.) 
convert to, enabling code-agnostic analysis functions.
"""

from typing import Final

# Universal column names for energy levels
LEVELS_SCHEMA: Final[dict] = {
    'level_index': 'Unique identifier for the level',
    'energy': 'Level energy in eV',
    'J': 'Total angular momentum quantum number',
    'parity': 'Parity (+1 or -1)',
    'g': 'Statistical weight (2J+1)',
    'configuration': 'Electronic configuration string',
    'term': 'Term symbol (optional)',
    'label': 'Full level label/description',
    'atomic_number': 'Atomic number Z',
    'ion_charge': 'Ion charge state',
}

# Universal column names for radiative transitions
TRANSITIONS_SCHEMA: Final[dict] = {
    'upper_level': 'Upper level index',
    'lower_level': 'Lower level index',
    'energy': 'Transition energy in eV',
    'wavelength': 'Wavelength in Angstrom',
    'rate': 'Radiative decay rate (Einstein A) in s^-1',
    'gf': 'Oscillator strength (weighted)',
    'S': 'Line strength',
    'multipolarity': 'Transition type (E1, M1, E2, etc.)',
    'atomic_number': 'Atomic number Z',
    'ion_charge': 'Ion charge state',
}

# Universal column names for autoionization/Auger transitions
AUTOIONIZATION_SCHEMA: Final[dict] = {
    'upper_level': 'Upper (excited) level index',
    'lower_level': 'Lower (final) level index',
    'energy': 'Transition energy in eV',
    'rate': 'Autoionization rate in s^-1',
    'atomic_number': 'Atomic number Z',
    'ion_charge': 'Ion charge state',
}


def get_required_levels_columns() -> list[str]:
    """Return required column names for levels DataFrame."""
    return ['level_index', 'energy', 'J', 'g', 'configuration', 'atomic_number']


def get_required_transitions_columns() -> list[str]:
    """Return required column names for transitions DataFrame."""
    return ['upper_level', 'lower_level', 'energy', 'rate']


def get_required_autoionization_columns() -> list[str]:
    """Return required column names for autoionization DataFrame."""
    return ['upper_level', 'lower_level', 'rate']
