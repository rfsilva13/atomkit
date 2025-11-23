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
    hole_shell: str = "1s1",
    shake_off_probability: float | None = None
) -> pd.DataFrame:
    """
    Calculate diagram line intensities: I = w * rate * branching_ratio
    
    Diagram lines come from SINGLE-HOLE states (e.g., just "1s+1(1)1").
    Final intensity includes shake-off correction: I_final = I * (1 - total_shake_off)
    
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
    shake_off_probability : float, optional
        Total shake-off probability (sum of all Q values).
        If None, no shake-off correction is applied.
        
    Returns
    -------
    pd.DataFrame
        Transitions with added 'intensity', 'intensity_final', 'fluorescence_yield', 
        'branching_ratio' columns
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
    
    # Calculate raw intensity (before shake-off correction)
    # I = w * Branching_Ratio (NOT w * rate * BR, since BR = rate/sum_rates!)
    diagram_transitions['intensity'] = (
        diagram_transitions['fluorescence_yield'] *
        diagram_transitions['branching_ratio']
    )
    
    # Apply shake-off correction if provided
    if shake_off_probability is not None:
        diagram_transitions['intensity_final'] = (
            diagram_transitions['intensity'] * (1 - shake_off_probability)
        )
    else:
        diagram_transitions['intensity_final'] = diagram_transitions['intensity']
    
    return diagram_transitions


def calculate_satellite_intensities(
    transitions: pd.DataFrame,
    levels: pd.DataFrame,
    auger_transitions: pd.DataFrame,
    shake_off_data: pd.DataFrame,
    hole_shell: str = "1s1"
) -> pd.DataFrame:
    """
    Calculate satellite line intensities from spectator-hole states.
    
    Satellite lines come from MULTI-HOLE states (e.g., "1s+1(1)1.4d+5(5)6").
    Intensity is weighted by shell-specific shake-off probability.
    
    Parameters
    ----------
    transitions : pd.DataFrame
        Radiative transitions
    levels : pd.DataFrame
        Level data with configurations
    auger_transitions : pd.DataFrame
        Auger transitions
    shake_off_data : pd.DataFrame
        Shake-off probabilities with 'shell' and 'Q1s' columns.
        Get from atomkit.analysis.get_shake_off_data()
    hole_shell : str
        Primary hole shell (default: "1s1")
        
    Returns
    -------
    pd.DataFrame
        Satellite transitions with 'intensity', 'intensity_final' columns
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
    
    if satellite_transitions.empty:
        return satellite_transitions
    
    # Add level configurations to transitions
    satellite_transitions = satellite_transitions.merge(
        levels[['level_index', 'configuration']],
        left_on='upper_level',
        right_on='level_index',
        how='left',
        suffixes=('', '_upper')
    ).drop(columns=['level_index'])
    
    # Extract spectator hole from configuration
    # For FAC: "1s+1(1)1.4d+5(5)6" means 1s and 4d holes
    # The spectator is the additional hole beyond the primary
    def extract_spectator_shell(config, primary_shell):
        """Extract spectator hole from multi-hole configuration."""
        if pd.isna(config) or '.' not in config:
            return None
        # Split by dot and find non-primary shells
        parts = config.split('.')
        for part in parts:
            if primary_shell not in part:
                # Extract shell notation (e.g., "4d+5" -> "4d5/2", "4d-3" -> "4d3/2")
                # Simplified: just get the shell letter and number
                import re
                match = re.match(r'(\d+)([spdf])[+-]?', part)
                if match:
                    n, l = match.groups()
                    # For shake-off data, use simplified notation
                    # Map to shake-off table format (e.g., "4d3/2", "4d5/2")
                    if '+' in part:
                        return f"{n}{l}5/2" if l in ['p', 'd', 'f'] else f"{n}{l}"
                    elif '-' in part:
                        return f"{n}{l}3/2" if l in ['p', 'd', 'f'] else f"{n}{l}"
                    else:
                        return f"{n}{l}"
        return None
    
    satellite_transitions['spectator_shell'] = satellite_transitions['configuration'].apply(
        lambda c: extract_spectator_shell(c, hole_shell)
    )
    
    # Calculate fluorescence yield for satellite states
    w_sat, _, _ = calculate_fluorescence_yield(
        satellite_transitions, auger_transitions
    )
    
    # Calculate branching ratios
    total_rates = satellite_transitions.groupby('upper_level')['rate'].transform('sum')
    satellite_transitions['branching_ratio'] = satellite_transitions['rate'] / total_rates.replace(0, 1)
    
    satellite_transitions['fluorescence_yield'] = w_sat
    
    # Merge with shake-off data to get shell-specific probabilities
    satellite_transitions = satellite_transitions.merge(
        shake_off_data[['shell', 'Q1s']],
        left_on='spectator_shell',
        right_on='shell',
        how='left'
    )
    
    # Calculate raw intensity
    satellite_transitions['intensity'] = (
        w_sat * satellite_transitions['branching_ratio']
    )
    
    # Final intensity weighted by shake-off probability
    satellite_transitions['intensity_final'] = (
        satellite_transitions['intensity'] * satellite_transitions['Q1s'].fillna(0)
    )
    
    return satellite_transitions


def calculate_spectrum(
    diagram_transitions: pd.DataFrame,
    satellite_transitions: pd.DataFrame | None = None,
    energy_min: float | None = None,
    energy_max: float | None = None,
    bin_width: float = 0.1,
    use_final_intensity: bool = True
) -> pd.DataFrame:
    """
    Combine diagram and satellite lines into a binned spectrum.
    
    Parameters
    ----------
    diagram_transitions : pd.DataFrame
        Diagram lines with 'energy' and 'intensity' or 'intensity_final'
    satellite_transitions : pd.DataFrame, optional
        Satellite lines with 'energy' and 'intensity' or 'intensity_final'
    energy_min : float, optional
        Minimum energy for spectrum (default: auto from data)
    energy_max : float, optional
        Maximum energy for spectrum (default: auto from data)
    bin_width : float
        Energy bin width in eV
    use_final_intensity : bool
        If True, use 'intensity_final' (with shake-off correction).
        If False, use 'intensity' (raw). Default: True
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'energy' and 'intensity' columns representing the spectrum
    """
    # Determine which intensity column to use
    intensity_col = 'intensity_final' if use_final_intensity else 'intensity'
    
    # Combine all transitions
    all_transitions = []
    for df in [diagram_transitions, satellite_transitions]:
        if df is not None and not df.empty:
            # Check if the intensity column exists, fallback to 'intensity'
            if intensity_col not in df.columns and 'intensity' in df.columns:
                df_copy = df.copy()
                df_copy[intensity_col] = df_copy['intensity']
                all_transitions.append(df_copy[['energy', intensity_col]])
            elif intensity_col in df.columns:
                all_transitions.append(df[['energy', intensity_col]])
    
    if not all_transitions:
        # Return empty spectrum
        return pd.DataFrame({'energy': [], 'intensity': []})
    
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
        weights=combined[intensity_col]
    )
    
    spectrum = pd.DataFrame({
        'energy': bin_centers,
        'intensity': binned_intensity
    })
    
    return spectrum


def calculate_widths(levels: pd.DataFrame, transitions: pd.DataFrame, auger: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate natural linewidths for atomic levels based on radiative and Auger decay rates.
    
    Uses the formula: Width = ħ * Σ(Rates) where ħ is the reduced Planck constant.
    
    Parameters
    ----------
    levels : pd.DataFrame
        Level data with 'level_index' column
    transitions : pd.DataFrame
        Radiative transitions with 'upper_level', 'rate' columns
    auger : pd.DataFrame
        Auger transitions with 'upper_level', 'rate' columns
        
    Returns
    -------
    pd.DataFrame
        Levels DataFrame with added 'width_radiative', 'width_auger', 'width_total' columns
        
    Notes
    -----
    Based on "Line Intensity and Width2024.pdf" reference.
    """
    # Physical constant: Planck constant in eV*s
    HBAR_EV = 6.582119569e-16

    # Calculate radiative widths (sum of A_r for transitions starting from each level)
    rad_widths = transitions.groupby("upper_level")["rate"].sum() * HBAR_EV

    # Calculate Auger widths (sum of A_nr for Auger transitions starting from each level)
    auger_widths = auger.groupby("upper_level")["rate"].sum() * HBAR_EV

    # Map back to levels DataFrame
    levels = levels.copy()
    levels["width_radiative"] = levels["level_index"].map(rad_widths).fillna(0.0)
    levels["width_auger"] = levels["level_index"].map(auger_widths).fillna(0.0)
    levels["width_total"] = levels["width_radiative"] + levels["width_auger"]

    return levels


def apply_linewidths_to_transitions(transitions: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """
    Apply natural linewidths to transitions.
    
    The linewidth for each transition is Gamma_ij = Gamma_i + Gamma_j,
    where Gamma_i and Gamma_j are the total widths of the upper and lower levels.
    
    Parameters
    ----------
    transitions : pd.DataFrame
        Transitions with 'upper_level', 'lower_level' columns
    levels : pd.DataFrame
        Levels with 'level_index', 'width_total' columns
        
    Returns
    -------
    pd.DataFrame
        Transitions with added 'width_upper', 'width_lower', 'line_width' columns
    """
    # Create mapping for fast lookup
    width_map = levels.set_index("level_index")["width_total"].to_dict()

    transitions = transitions.copy()
    transitions["width_upper"] = transitions["upper_level"].map(width_map).fillna(0.0)
    transitions["width_lower"] = transitions["lower_level"].map(width_map).fillna(0.0)
    transitions["line_width"] = transitions["width_upper"] + transitions["width_lower"]

    return transitions


def create_lorentzian_spectrum(
    transitions_df: pd.DataFrame, 
    energy_grid: np.ndarray, 
    intensity_col: str = 'intensity_final'
) -> np.ndarray:
    """
    Create a Lorentzian-broadened spectrum from transition lines.
    
    Each line is broadened with a Lorentzian profile: 
    I(ω) = (I_0 * Γ/2) / ((ω - ω_0)^2 + (Γ/2)^2) * (2/π) / (Γ/2)
    where Γ is the natural linewidth.
    
    Parameters
    ----------
    transitions_df : pd.DataFrame
        DataFrame with 'energy', intensity_col, and 'line_width' columns
    energy_grid : np.ndarray
        Energy values for the spectrum in eV
    intensity_col : str
        Column name for intensity values (default: 'intensity_final')
        
    Returns
    -------
    np.ndarray
        Lorentzian-broadened intensity values at each energy grid point
        
    Notes
    -----
    For lines with zero width, adds intensity as a delta function at the nearest grid point.
    """
    spectrum = np.zeros_like(energy_grid)
    
    for _, line in transitions_df.iterrows():
        energy = line['energy']
        intensity = line[intensity_col]
        width = line['line_width']
        
        if width > 0:
            # Lorentzian profile normalized so integral = intensity
            # I(ω) = I_0 * (Γ/2)^2 / ((ω - ω_0)^2 + (Γ/2)^2) / π / (Γ/2)
            lorentzian = intensity * (width/2)**2 / ((energy_grid - energy)**2 + (width/2)**2) / np.pi / (width/2)
            spectrum += lorentzian
        else:
            # Delta function for zero-width lines
            idx = np.argmin(np.abs(energy_grid - energy))
            spectrum[idx] += intensity
    
    return spectrum
