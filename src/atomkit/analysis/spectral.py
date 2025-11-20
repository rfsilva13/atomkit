"""
Universal spectral analysis functions.

Works with data in universal schema from any atomic code.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def load_data(base_filename: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load atomic data from any supported format using universal readers.
    
    Auto-detects file format and returns data in universal schema.
    
    Parameters
    ----------
    base_filename : str
        Base path to data files (without extension)
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        levels, transitions, autoionization DataFrames with universal column names
        
    Examples
    --------
    >>> levels, trans, auger = load_data('/path/to/Pd')
    >>> print(levels[['level_index', 'energy', 'J']].head())
    """
    from ..readers import read_autoionization, read_levels, read_transitions
    
    levels = read_levels(base_filename)
    transitions = read_transitions(base_filename)
    autoionization = read_autoionization(base_filename)
    
    return levels, transitions, autoionization


def add_transition_energies(transitions: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """
    Add upper and lower level energies to transitions DataFrame.
    
    Parameters
    ----------
    transitions : pd.DataFrame
        Transitions with 'upper_level' and 'lower_level' columns
    levels : pd.DataFrame
        Levels with 'level_index' and 'energy' columns
        
    Returns
    -------
    pd.DataFrame
        Transitions with added 'energy_upper' and 'energy_lower' columns
    """
    result = transitions.copy()
    
    # Merge upper level energies
    result = result.merge(
        levels[['level_index', 'energy']].rename(columns={'energy': 'energy_upper'}),
        left_on='upper_level',
        right_on='level_index',
        how='left'
    ).drop(columns=['level_index'])
    
    # Merge lower level energies
    result = result.merge(
        levels[['level_index', 'energy']].rename(columns={'energy': 'energy_lower'}),
        left_on='lower_level',
        right_on='level_index',
        how='left'
    ).drop(columns=['level_index'])
    
    return result


def calculate_fluorescence_yield(
    radiative_transitions: pd.DataFrame,
    auger_transitions: pd.DataFrame,
    shell_filter: str | None = None
) -> Tuple[float, float, float]:
    """
    Calculate fluorescence yield for a shell.
    
    Parameters
    ----------
    radiative_transitions : pd.DataFrame
        Radiative transitions with 'rate' column
    auger_transitions : pd.DataFrame
        Auger transitions with 'rate' column
    shell_filter : str, optional
        Shell to filter for (not yet implemented)
        
    Returns
    -------
    Tuple[float, float, float]
        (fluorescence_yield, radiative_rate_sum, auger_rate_sum)
        
    Examples
    --------
    >>> w, r_sum, a_sum = calculate_fluorescence_yield(diagram, auger)
    >>> print(f"Fluorescence yield: {w:.4f}")
    """
    radiative_sum = radiative_transitions['rate'].sum()
    auger_sum = auger_transitions['rate'].sum()
    
    if radiative_sum + auger_sum == 0:
        return 0.0, radiative_sum, auger_sum
    
    w = radiative_sum / (radiative_sum + auger_sum)
    return w, radiative_sum, auger_sum


def label_hole_states(levels: pd.DataFrame, hole_shell: str = "1s1") -> pd.DataFrame:
    """
    Identify and label levels representing hole states in a specific shell.
    
    Parameters
    ----------
    levels : pd.DataFrame
        Level data with 'configuration' column
    hole_shell : str
        Shell pattern to identify holes (e.g., "1s1" for K-shell holes)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'is_hole_state' boolean column and 'hole_label' column
    """
    levels = levels.copy()
    levels['is_hole_state'] = levels['configuration'].str.contains(hole_shell, na=False)
    levels['hole_label'] = levels['configuration'].where(levels['is_hole_state'], "")
    return levels


def filter_transitions_by_shell(
    transitions: pd.DataFrame,
    levels: pd.DataFrame,
    upper_shell: str | None = None,
    lower_shell: str | None = None
) -> pd.DataFrame:
    """
    Filter transitions based on shell configurations of upper/lower levels.
    
    Parameters
    ----------
    transitions : pd.DataFrame
        Transition data with 'upper_level' and 'lower_level' columns
    levels : pd.DataFrame
        Level data with 'level_index' and 'configuration' columns
    upper_shell : str, optional
        Shell pattern for upper level (e.g., "1s1")
    lower_shell : str, optional
        Shell pattern for lower level
        
    Returns
    -------
    pd.DataFrame
        Filtered transitions DataFrame
    """
    # Merge level configurations
    trans_with_configs = transitions.merge(
        levels[['level_index', 'configuration']],
        left_on='upper_level',
        right_on='level_index',
        how='left',
        suffixes=('', '_upper')
    ).merge(
        levels[['level_index', 'configuration']],
        left_on='lower_level',
        right_on='level_index',
        how='left',
        suffixes=('_upper', '_lower')
    )
    
    # Apply filters
    mask = pd.Series(True, index=trans_with_configs.index)
    if upper_shell:
        mask &= trans_with_configs['configuration_upper'].str.contains(upper_shell, na=False)
    if lower_shell:
        mask &= trans_with_configs['configuration_lower'].str.contains(lower_shell, na=False)
    
    return transitions[mask]


def calculate_diagram_intensities(
    transitions: pd.DataFrame,
    levels: pd.DataFrame,
    auger_transitions: pd.DataFrame,
    hole_shell: str = "1s1"
) -> pd.DataFrame:
    """
    Calculate diagram line intensities: I = w * rate * branching_ratio
    
    Diagram lines come from SINGLE-HOLE states (e.g., just "1s+1(1)1").
    
    Parameters
    ----------
    transitions : pd.DataFrame
        Radiative transitions with 'rate' column
    levels : pd.DataFrame
        Level data with configurations
    auger_transitions : pd.DataFrame
        Auger transitions for fluorescence yield
    hole_shell : str
        Shell defining the hole state (default: K-shell "1s1")
        
    Returns
    -------
    pd.DataFrame
        Transitions with added 'intensity', 'fluorescence_yield', 'branching_ratio' columns
    """
    # Label hole states
    levels = label_hole_states(levels, hole_shell)
    
    # Identify SINGLE-hole states (no dot in configuration means single hole)
    # e.g., "1s+1(1)1" is single, "1s+1(1)1.4d+5(5)6" is multi
    levels = levels.copy()
    levels['is_single_hole'] = (
        levels['is_hole_state'] & 
        ~levels['configuration'].str.contains(r'\.', regex=True, na=False)
    )
    
    # Add level info to transitions
    trans_with_levels = transitions.merge(
        levels[['level_index', 'is_single_hole']],
        left_on='upper_level',
        right_on='level_index',
        how='left'
    ).drop(columns=['level_index'])
    
    trans_with_levels = add_transition_energies(trans_with_levels, levels)
    
    # Filter transitions from SINGLE-HOLE states only
    diagram_transitions = trans_with_levels[
        trans_with_levels['is_single_hole'] == True
    ].copy()
    
    # Calculate fluorescence yield
    w, rad_rate, auger_rate = calculate_fluorescence_yield(
        transitions, auger_transitions, hole_shell
    )
    diagram_transitions['fluorescence_yield'] = w
    
    # Calculate branching ratios for each upper level
    total_rates = diagram_transitions.groupby('upper_level')['rate'].transform('sum')
    diagram_transitions['branching_ratio'] = diagram_transitions['rate'] / total_rates
    
    # Calculate intensity
    diagram_transitions['intensity'] = (
        diagram_transitions['fluorescence_yield'] *
        diagram_transitions['rate'] *
        diagram_transitions['branching_ratio']
    )
    
    return diagram_transitions


def calculate_satellite_intensities(
    transitions: pd.DataFrame,
    levels: pd.DataFrame,
    auger_transitions: pd.DataFrame,
    creation_rate: float,
    hole_shell: str = "1s1"
) -> pd.DataFrame:
    """
    Calculate satellite line intensities from spectator-hole states.
    
    Satellite lines come from MULTI-HOLE states (e.g., "1s+1(1)1.4d+5(5)6").
    
    Parameters
    ----------
    transitions : pd.DataFrame
        Radiative transitions
    levels : pd.DataFrame
        Level data with configurations
    auger_transitions : pd.DataFrame
        Auger transitions
    creation_rate : float
        Rate of creating spectator-hole states (e.g., from shake-off)
    hole_shell : str
        Primary hole shell (default: "1s1")
        
    Returns
    -------
    pd.DataFrame
        Satellite transitions with 'intensity' column
    """
    # Label hole states first
    levels = label_hole_states(levels, hole_shell)
    
    # Identify satellite states (MULTI-hole: has primary hole AND other holes)
    # Presence of dot indicates multiple subshells, e.g., "1s+1(1)1.4d+5(5)6"
    levels = levels.copy()
    levels['is_satellite'] = (
        levels['is_hole_state'] &
        levels['configuration'].str.contains(r'\.', regex=True, na=False)
    )
    
    # Filter transitions from satellite states
    satellite_levels = levels[levels['is_satellite']]['level_index'].values
    satellite_transitions = transitions[
        transitions['upper_level'].isin(satellite_levels)
    ].copy()
    
    # Calculate fluorescence yield for satellite states
    w_sat, _, _ = calculate_fluorescence_yield(
        satellite_transitions, auger_transitions
    )
    
    # Calculate branching ratios
    total_rates = satellite_transitions.groupby('upper_level')['rate'].transform('sum')
    satellite_transitions['branching_ratio'] = satellite_transitions['rate'] / total_rates
    
    # Intensity = creation_rate * w * branching_ratio
    satellite_transitions['intensity'] = (
        creation_rate * w_sat * satellite_transitions['branching_ratio']
    )
    satellite_transitions['fluorescence_yield'] = w_sat
    
    return satellite_transitions


def calculate_spectrum(
    diagram_transitions: pd.DataFrame,
    satellite_transitions: pd.DataFrame | None = None,
    energy_min: float | None = None,
    energy_max: float | None = None,
    bin_width: float = 0.1
) -> pd.DataFrame:
    """
    Combine diagram and satellite lines into a binned spectrum.
    
    Parameters
    ----------
    diagram_transitions : pd.DataFrame
        Diagram lines with 'energy' and 'intensity'
    satellite_transitions : pd.DataFrame, optional
        Satellite lines with 'energy' and 'intensity'
    energy_min : float, optional
        Minimum energy for spectrum (default: auto from data)
    energy_max : float, optional
        Maximum energy for spectrum (default: auto from data)
    bin_width : float
        Energy bin width in eV
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'energy' and 'intensity' columns representing the spectrum
    """
    # Combine all transitions
    all_transitions = [diagram_transitions]
    if satellite_transitions is not None:
        all_transitions.append(satellite_transitions)
    combined = pd.concat(all_transitions, ignore_index=True)
    
    # Determine energy range
    e_min = energy_min if energy_min is not None else combined['energy'].min() - bin_width
    e_max = energy_max if energy_max is not None else combined['energy'].max() + bin_width
    
    # Create energy bins
    bins = np.arange(e_min, e_max + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Bin the intensities
    binned_intensity, _ = np.histogram(
        combined['energy'],
        bins=bins,
        weights=combined['intensity']
    )
    
    spectrum = pd.DataFrame({
        'energy': bin_centers,
        'intensity': binned_intensity
    })
    
    return spectrum
