"""
Unified reader interface for atomic structure data.

Auto-detects file format and returns data in universal schema.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .autoionization import read_fac_autoionization as _read_fac_autoionization
from .autostructure import detect_file_format
from .autostructure import read_as_levels as _read_as_levels
from .autostructure import read_as_transitions as _read_as_transitions
from .levels import read_fac as _read_fac_levels
from .transitions import read_fac_transitions as _read_fac_transitions

logger = logging.getLogger(__name__)


def _normalize_fac_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert FAC levels DataFrame to universal schema."""
    normalized = pd.DataFrame()
    
    # Required columns
    normalized['level_index'] = df['level_index']
    normalized['energy'] = df['energy']  # Already in eV
    
    # Calculate J from 2j
    if '2j' in df.columns:
        normalized['J'] = df['2j'] / 2.0
    elif 'J' in df.columns:
        normalized['J'] = df['J']
    
    # Calculate g from 2j
    if '2j' in df.columns:
        normalized['g'] = df['2j'] + 1
    elif 'g' in df.columns:
        normalized['g'] = df['g']
    
    normalized['configuration'] = df['label']  # Use full label as configuration
    normalized['atomic_number'] = df.get('atomic_number', None)
    normalized['ion_charge'] = df.get('ion_charge', None)
    
    # Optional columns
    if 'term' in df.columns:
        normalized['term'] = df['term']
    if 'p' in df.columns:
        normalized['parity'] = df['p']
    elif 'P' in df.columns:
        normalized['parity'] = df['P']
    normalized['label'] = df['label']
    
    return normalized


def _normalize_fac_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Convert FAC transitions DataFrame to universal schema."""
    normalized = pd.DataFrame()
    
    # Required columns
    normalized['upper_level'] = df['level_index_upper']
    normalized['lower_level'] = df['level_index_lower']
    normalized['energy'] = df['energy']  # Already in eV
    normalized['rate'] = df['A']  # Einstein A coefficient
    
    # Optional columns
    if 'gf' in df.columns:
        normalized['gf'] = df['gf']
    if 'S' in df.columns:
        normalized['S'] = df['S']
    if 'lambda' in df.columns:
        normalized['wavelength'] = df['lambda']
    if 'type' in df.columns:
        normalized['multipolarity'] = df['type']
    elif 'multipole' in df.columns:
        normalized['multipolarity'] = df['multipole']
    if 'atomic_number' in df.columns:
        normalized['atomic_number'] = df['atomic_number']
    
    return normalized


def _normalize_fac_autoionization(df: pd.DataFrame) -> pd.DataFrame:
    """Convert FAC autoionization DataFrame to universal schema."""
    normalized = pd.DataFrame()
    
    # Required columns
    normalized['upper_level'] = df['level_index_upper']
    normalized['lower_level'] = df['level_index_lower']
    normalized['rate'] = df['ai_rate']  # Autoionization rate
    
    # Optional columns
    if 'energy' in df.columns:
        normalized['energy'] = df['energy']
    if 'atomic_number' in df.columns:
        normalized['atomic_number'] = df['atomic_number']
    if 'ion_charge' in df.columns:
        normalized['ion_charge'] = df['ion_charge']
    
    return normalized


def _normalize_as_levels(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Convert AUTOSTRUCTURE levels DataFrame to universal schema."""
    import scipy.constants as const
    RY_TO_EV = const.physical_constants['Rydberg constant times hc in eV'][0]
    
    normalized = pd.DataFrame()
    
    # Required columns
    normalized['level_index'] = df['K']
    normalized['energy'] = df['Level (Ry)'] * RY_TO_EV  # Convert Ry to eV
    normalized['J'] = df['2J'] / 2.0
    normalized['g'] = df['2J'] + 1  # g = 2J + 1
    normalized['configuration'] = df['CF']
    normalized['atomic_number'] = metadata.get('Atomic number', None)
    
    # Optional columns
    if 'P' in df.columns:
        normalized['parity'] = df['P']
    if '2*S+1' in df.columns and 'L' in df.columns:
        # Build term symbol from L and S
        S = (df['2*S+1'] - 1) / 2
        normalized['term'] = df['2*S+1'].astype(str) + df['L'].astype(str)
    
    normalized['label'] = df['CF']
    
    return normalized


def _normalize_as_transitions(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Convert AUTOSTRUCTURE transitions DataFrame to universal schema."""
    normalized = pd.DataFrame()
    
    # Required columns
    normalized['upper_level'] = df['K']
    normalized['lower_level'] = df['Klower']
    
    # Calculate energy from wavelength if available
    if 'WAVEL/AE' in df.columns:
        # E(eV) = 12398.419 / wavelength(Angstrom)
        normalized['wavelength'] = df['WAVEL/AE']
        normalized['energy'] = 12398.419 / df['WAVEL/AE']
    
    if 'A(K)*SEC' in df.columns:
        normalized['rate'] = df['A(K)*SEC']
    elif 'A(EK)*SEC' in df.columns:
        normalized['rate'] = df['A(EK)*SEC']
    
    # Optional columns
    if 'log(gf)' in df.columns:
        normalized['gf'] = 10 ** df['log(gf)']
    elif 'G*F' in df.columns:
        normalized['gf'] = df['G*F']
    
    normalized['atomic_number'] = metadata.get('Atomic number', None)
    
    return normalized


def read_levels(
    filename: str | Path,
    file_extension: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read atomic level data from any supported format.
    
    Auto-detects file format (FAC, AUTOSTRUCTURE, etc.) and returns
    data in universal schema with standard column names.
    
    Parameters
    ----------
    filename : str or Path
        Path to file or base filename
    file_extension : str, optional
        File extension to use (for FAC files)
    **kwargs
        Additional arguments passed to format-specific readers
        
    Returns
    -------
    pd.DataFrame
        Levels data with universal column names:
        - level_index: Level identifier
        - energy: Energy in eV
        - J: Total angular momentum
        - g: Statistical weight
        - configuration: Electronic configuration
        - atomic_number: Atomic number
        - ion_charge: Ion charge (if available)
        - parity: Parity (if available)
        - term: Term symbol (if available)
        - label: Full level label
        
    Examples
    --------
    >>> levels = read_levels('Pd.lev.asc')  # FAC file
    >>> levels = read_levels('output.olg')  # AUTOSTRUCTURE file
    >>> print(levels[['level_index', 'energy', 'J', 'configuration']].head())
    """
    filepath = Path(filename)
    
    # Detect format
    if file_extension or not filepath.exists():
        # Assume FAC if extension provided or base filename
        format_type = 'fac'
        base_filename = str(filename)
        if file_extension is None:
            file_extension = '.lev.asc'
    else:
        format_type = detect_file_format(filepath)
    
    # Read with format-specific reader
    if format_type == 'fac':
        ext = file_extension if file_extension is not None else '.lev.asc'
        df = _read_fac_levels(base_filename, ext, **kwargs)
        return _normalize_fac_levels(df)
    elif format_type == 'autostructure':
        df, metadata = _read_as_levels(filepath, **kwargs)
        return _normalize_as_levels(df, metadata)
    else:
        raise ValueError(f"Unsupported file format: {format_type}")


def read_transitions(
    filename: str | Path,
    file_extension: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read radiative transition data from any supported format.
    
    Auto-detects file format and returns data in universal schema.
    
    Parameters
    ----------
    filename : str or Path
        Path to file or base filename
    file_extension : str, optional
        File extension to use (for FAC files)
    **kwargs
        Additional arguments passed to format-specific readers
        
    Returns
    -------
    pd.DataFrame
        Transitions data with universal column names:
        - upper_level: Upper level index
        - lower_level: Lower level index
        - energy: Transition energy in eV
        - rate: Einstein A coefficient in s^-1
        - wavelength: Wavelength in Angstrom (if available)
        - gf: Oscillator strength (if available)
        - S: Line strength (if available)
        - multipolarity: E1, M1, etc. (if available)
        
    Examples
    --------
    >>> transitions = read_transitions('Pd.tr.asc')  # FAC
    >>> transitions = read_transitions('output.olg')  # AUTOSTRUCTURE
    """
    filepath = Path(filename)
    
    # Detect format
    if file_extension or not filepath.exists():
        format_type = 'fac'
        base_filename = str(filename)
        if file_extension is None:
            file_extension = '.tr.asc'
    else:
        format_type = detect_file_format(filepath)
    
    # Read with format-specific reader
    if format_type == 'fac':
        ext = file_extension if file_extension is not None else '.tr.asc'
        df = _read_fac_transitions(base_filename, ext, **kwargs)
        return _normalize_fac_transitions(df)
    elif format_type == 'autostructure':
        df, metadata = _read_as_transitions(filepath, **kwargs)
        return _normalize_as_transitions(df, metadata)
    else:
        raise ValueError(f"Unsupported file format: {format_type}")


def read_autoionization(
    filename: str | Path,
    file_extension: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read autoionization/Auger data from any supported format.
    
    Auto-detects file format and returns data in universal schema.
    
    Parameters
    ----------
    filename : str or Path
        Path to file or base filename
    file_extension : str, optional
        File extension to use (for FAC files)
    **kwargs
        Additional arguments passed to format-specific readers
        
    Returns
    -------
    pd.DataFrame
        Autoionization data with universal column names:
        - upper_level: Upper level index
        - lower_level: Lower level index
        - rate: Autoionization rate in s^-1
        - energy: Transition energy in eV (if available)
        
    Examples
    --------
    >>> auger = read_autoionization('Pd.ai.asc')  # FAC
    """
    filepath = Path(filename)
    
    # Detect format
    if file_extension or not filepath.exists():
        format_type = 'fac'
        base_filename = str(filename)
        if file_extension is None:
            file_extension = '.ai.asc'
    else:
        format_type = detect_file_format(filepath)
    
    # Read with format-specific reader
    if format_type == 'fac':
        ext = file_extension if file_extension is not None else '.ai.asc'
        df = _read_fac_autoionization(base_filename, ext, **kwargs)
        return _normalize_fac_autoionization(df)
    else:
        # AUTOSTRUCTURE doesn't have autoionization files yet
        raise ValueError(f"Autoionization not supported for format: {format_type}")
