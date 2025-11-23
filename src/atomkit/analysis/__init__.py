"""
AtomKit analysis module.

Functions for processing and analyzing atomic spectral data, including
diagram and satellite line intensities, shake-off corrections, and plotting.

Universal analysis functions work with data from any atomic code.
Legacy FAC-specific functions are also available for backward compatibility.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .plotting import create_interactive_energy_shifter
# Export universal analysis functions
from .spectral import (add_transition_energies,
                       apply_linewidths_to_transitions,
                       calculate_diagram_intensities,
                       calculate_fluorescence_yield,
                       calculate_satellite_intensities, calculate_spectrum,
                       calculate_widths, create_lorentzian_spectrum,
                       filter_transitions_by_shell, label_hole_states,
                       load_data)

__all__ = [
    # Universal analysis functions (recommended)
    'load_data',
    'add_transition_energies',
    'calculate_fluorescence_yield',
    'label_hole_states',
    'filter_transitions_by_shell',
    'calculate_diagram_intensities',
    'calculate_satellite_intensities',
    'calculate_spectrum',
    'calculate_widths',
    'apply_linewidths_to_transitions',
    'create_lorentzian_spectrum',
    'create_interactive_energy_shifter',
    # Legacy FAC-specific functions (for backward compatibility)
    'calculate_g',
    'get_spectator_hole',
    'load_spectral_data',
    'get_shake_off_data',
    'clean_spectral_data',
    'process_diagram_intensities',
    'process_satellite_intensities',
    'filter_satellite_data',
    'plot_k_alpha_spectrum',
    'parse_config_occupations',
    'get_full_occupation',
    'identify_fac_holes',
    'get_fac_hole_to_shell_map',
    'map_fac_holes_to_shell',
    'calculate_g_fac',
    'load_fac_data',
    'process_fac_diagram_intensities',
    'process_fac_satellite_intensities',
    'add_fac_transition_energies_and_holes',
    'filter_fac_k_alpha_transitions',
    'calculate_fac_wk',
    'plot_fac_k_alpha_spectrum',
]

# Maximum electrons for each shell type
MAX_ELECTRONS = {'s': 2, 'p': 6, 'd': 10, 'f': 14}

# Full occupations for relativistic subshells
FULL_OCCUPATIONS = {
    's': 2,
    'p-': 2,
    'p+': 4,
    'd-': 4,
    'd+': 6,
    'f-': 6,
    'f+': 8,
}


def calculate_g(shell_string: str) -> int:
    """
    Calculate the total statistical weight (g) for a configuration string.

    Parameters
    ----------
    shell_string : str
        String representing holes, e.g., '1s_2p_2p'.

    Returns
    -------
    int
        Total statistical weight.

    Examples
    --------
    >>> calculate_g('1s_2p')
    12
    """
    if not isinstance(shell_string, str) or not shell_string:
        return 1

    holes = [h.strip() for h in shell_string.split('_')]
    hole_counts = Counter(holes)
    total_g = 1

    for shell, x in hole_counts.items():
        match = re.search(r'[spdf]', shell.lower())
        if not match:
            continue
        shell_type = match.group(0)
        y = MAX_ELECTRONS[shell_type]
        sub_g = math.comb(y, x)
        total_g *= sub_g

    return total_g


def get_spectator_hole(initial_state_str: str, final_state_str: str) -> Optional[str]:
    """
    Identify spectator holes by comparing initial and final state strings.

    Parameters
    ----------
    initial_state_str : str
        Initial state hole string.
    final_state_str : str
        Final state hole string.

    Returns
    -------
    str or None
        Spectator hole string or None.

    Examples
    --------
    >>> get_spectator_hole('1s_1s', '1s_2p')
    '1s'
    """
    if not initial_state_str or not final_state_str:
        return None

    holes_is = [h.strip() for h in initial_state_str.split('_')]
    holes_fs = [h.strip() for h in final_state_str.split('_')]
    count_is = Counter(holes_is)
    count_fs = Counter(holes_fs)
    spectators_count = count_is & count_fs
    spectator_list = list(spectators_count.elements())

    if not spectator_list:
        return None

    return '_'.join(sorted(spectator_list))


def load_spectral_data(file_paths: Dict[str, str]) -> Tuple[pd.DataFrame, ...]:
    """
    Load spectral data from CSV files.

    Parameters
    ----------
    file_paths : dict
        Dictionary with keys: 'diagram', 'satellite', 'satellite_complete', 'auger', 'holes'

    Returns
    -------
    tuple of pd.DataFrame
        Loaded DataFrames: diagram, satellite, satellite_complete, auger, holes
    """
    diagram_data = pd.read_csv(file_paths['diagram'])
    satellite_data = pd.read_csv(file_paths['satellite'])
    satellite_data_complete = pd.read_csv(file_paths['satellite_complete'])
    auger_data = pd.read_csv(file_paths['auger'])
    holes_data = pd.read_csv(file_paths['holes'])

    for df in [diagram_data, satellite_data, auger_data, satellite_data_complete, holes_data]:
        df.columns = df.columns.str.strip()

    return diagram_data, satellite_data, satellite_data_complete, auger_data, holes_data


def get_shake_off_data(element: str | int | None = None, use_carlson_for_pd: bool = False) -> pd.DataFrame:
    """
    Get shake-off probabilities DataFrame.

    Parameters
    ----------
    element : str, int, or None, optional
        Element symbol (str), atomic number (int), or None for Pd (default).
        If None, returns the original Pd data for backward compatibility.
    use_carlson_for_pd : bool, optional
        If True, use Carlson data for Pd instead of original photoionization data.
        Default is False (use original Pd data).

    Returns
    -------
    pd.DataFrame
        Shake-off data with columns 'shell' and 'Q1s'.
    """
    import csv
    import os

    # Define conversion function
    def estimate_k_shakeoff(Z, shell_name, carlson_prob_pct):
        """Convert Carlson beta decay % to photoionization %"""
        import re
        match = re.match(r'(\d+)[spdf]', shell_name)
        if not match:
            raise ValueError(f'Invalid shell format: {shell_name}')
        
        n_target = int(match.group(1))
        
        if Z <= 2: n_max = 1
        elif Z <= 10: n_max = 2
        elif Z <= 18: n_max = 3
        elif Z <= 36: n_max = 4
        elif Z <= 54: n_max = 5
        elif Z <= 86: n_max = 6
        else: n_max = 7

        # Empirically-derived correction factors from Carlson-to-photoionization validation
        if shell_name.startswith('1s'):
            factor = 0.15
        elif n_target == n_max:
            factor = 1.1  # valence
        elif n_target == n_max - 1:
            factor = 0.9  # sub-valence
        else:
            factor = 0.3  # inner

        return carlson_prob_pct * factor / 100  # Convert % to fractional

    # Basic element symbol to atomic number mapping
    element_symbols = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
        'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
        'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
        'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
        'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
        'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100
    }

    # If no element specified, use original Pd data for backward compatibility
    if element is None:
        shake_off_data = {
            'shell': ['1s', '2s', '3s', '4s', '2p1/2', '2p3/2', '3p1/2', '3p3/2', '3d3/2', '3d5/2', '4p1/2', '4p3/2', '4d3/2', '4d5/2'],
            'Q1s': [6.069821E-05, 3.090648E-04, 1.243763E-03, 4.907061E-03, 5.219590E-04, 9.372514E-04, 3.200017E-03, 3.991437E-03, 4.447591E-03, 6.591329E-03, 8.118569E-03, 1.580881E-02, 6.859390E-02, 1.098453E-01]
        }
        shake_off_df = pd.DataFrame(shake_off_data)
        return shake_off_df

    # Try to load the comprehensive shake-off table
    shake_table_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shake_table.csv')

    if not os.path.exists(shake_table_path):
        # Fallback to Pd data if table not found
        print(f"Warning: shake_table.csv not found at {shake_table_path}, using Pd data")
        return get_shake_off_data(None)

    try:
        # Load the shake-off table using csv reader (pandas has parsing issues)
        data = []
        with open(shake_table_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                data.append(row)

        # Find the element row
        element_row = None
        target_z = None
        
        if isinstance(element, str):
            # Convert element symbol to atomic number if needed
            if element in element_symbols:
                target_z = element_symbols[element]
                # Special handling for Pd - use original data by default, Carlson if requested
                if target_z == 46 and not use_carlson_for_pd:
                    print("Using original photoionization data for Pd")
                    return get_shake_off_data(None)
                # Find by atomic number
                for row in data:
                    if len(row) > 0 and row[0] == str(target_z):
                        element_row = row
                        break
            else:
                # Try to find by element symbol directly
                for row in data:
                    if len(row) > 1 and row[1].strip() == element:
                        element_row = row
                        target_z = int(row[0])
                        break
        elif isinstance(element, int):
            target_z = element  # Set target_z regardless of whether found in table
            # Special handling for Pd - use original data by default, Carlson if requested
            if target_z == 46 and not use_carlson_for_pd:
                print("Using original photoionization data for Pd")
                return get_shake_off_data(None)
            # Find by atomic number
            for row in data:
                if len(row) > 0 and row[0] == str(element):
                    element_row = row
                    break
        
        if element_row is None:
            # Try local interpolation for missing elements (nearest neighbors)
            print(f"Warning: Element '{element}' not found in shake_table.csv, trying local interpolation")
            
            # Get all available Z values
            available_z = []
            for row in data:
                if len(row) > 0 and row[0].isdigit():
                    available_z.append(int(row[0]))
            
            available_z.sort()
            
            if len(available_z) < 2:
                print(f"Warning: Not enough data for interpolation, using Pd data")
                return get_shake_off_data(None)
            
            # Find nearest elements for local interpolation
            # target_z should already be set from symbol conversion above
            if target_z is None:
                if isinstance(element, str) and element not in element_symbols:
                    print(f"Warning: Unknown element symbol '{element}', using Pd data")
                    return get_shake_off_data(None)
                else:
                    print(f"Warning: Cannot interpolate for element '{element}', using Pd data")
                    return get_shake_off_data(None)
            
            # Find lower and upper bounds (nearest neighbors)
            lower_z = None
            upper_z = None
            
            for z in available_z:
                if z < target_z:
                    lower_z = z
                elif z > target_z and upper_z is None:
                    upper_z = z
                    break
            
            if lower_z is None or upper_z is None:
                # Extrapolate using nearest element
                nearest_z = min(available_z, key=lambda x: abs(x - target_z))
                print(f"Warning: Cannot interpolate Z={target_z}, extrapolating from Z={nearest_z}")
                return get_shake_off_data(nearest_z)
            
            # Get data for lower and upper elements
            lower_row = None
            upper_row = None
            
            for row in data:
                if len(row) > 0 and row[0] == str(lower_z):
                    lower_row = row
                elif len(row) > 0 and row[0] == str(upper_z):
                    upper_row = row
            
            if lower_row is None or upper_row is None:
                print(f"Warning: Could not find data for interpolation bounds, using Pd data")
                return get_shake_off_data(None)
            
            # Interpolate shake-off values for each shell using local linear interpolation
            interpolated_data = {}
            
            for i, col_name in enumerate(headers):
                if i >= 2 and i < len(headers) - 1:  # Skip Z, Element, and Total Shake-off columns
                    lower_val = lower_row[i] if i < len(lower_row) and lower_row[i] and lower_row[i].strip() else 0
                    upper_val = upper_row[i] if i < len(upper_row) and upper_row[i] and upper_row[i].strip() else 0
                    
                    try:
                        lower_pct = float(lower_val) if lower_val else 0.0
                        upper_pct = float(upper_val) if upper_val else 0.0
                        
                        if lower_pct > 0 or upper_pct > 0:
                            # Local linear interpolation between nearest neighbors
                            if lower_pct > 0 and upper_pct > 0:
                                # Both values available - interpolate
                                ratio = (target_z - lower_z) / (upper_z - lower_z)
                                interpolated_pct = lower_pct + ratio * (upper_pct - lower_pct)
                            elif lower_pct > 0:
                                # Only lower value - use it
                                interpolated_pct = lower_pct
                            else:
                                # Only upper value - use it
                                interpolated_pct = upper_pct
                            
                            interpolated_data[col_name] = interpolated_pct
                    except (ValueError, IndexError):
                        continue
            
            if not interpolated_data:
                print(f"Warning: No valid interpolated data found for Z={target_z}, using Pd data")
                return get_shake_off_data(None)
            
            # Convert interpolated Carlson data to photoionization probabilities
            shells = []
            q1s_values = []
            
            for shell_name, carlson_pct in interpolated_data.items():
                photoionization_fractional = estimate_k_shakeoff(target_z, shell_name, carlson_pct)
                shells.append(shell_name)
                q1s_values.append(photoionization_fractional)
            
            print(f"Successfully interpolated shake-off data for Z={target_z} using local interpolation between Z={lower_z} and Z={upper_z}")
            
            shake_off_data = {
                'shell': shells,
                'Q1s': q1s_values
            }
            shake_off_df = pd.DataFrame(shake_off_data)
            return shake_off_df

        # Extract Carlson shake-off data (in %)
        carlson_data = {}
        for i, col_name in enumerate(headers):
            if i >= 2 and i < len(headers) - 1:  # Skip Z, Element, and Total Shake-off columns
                value = element_row[i]
                if value and value.strip() and float(value) > 0:
                    shell_name = col_name
                    carlson_data[shell_name] = float(value)

        if not carlson_data:
            print(f"Warning: No valid shake-off data found for {element}, using Pd data")
            return get_shake_off_data(None)

        # Convert Carlson beta decay % to photoionization fractional probabilities
        shells = []
        q1s_values = []
        
        for shell_name, carlson_pct in carlson_data.items():
            photoionization_fractional = estimate_k_shakeoff(target_z, shell_name, carlson_pct)
            shells.append(shell_name)
            q1s_values.append(photoionization_fractional)

        shake_off_data = {
            'shell': shells,
            'Q1s': q1s_values
        }
        shake_off_df = pd.DataFrame(shake_off_data)
        return shake_off_df

    except Exception as e:
        print(f"Warning: Error loading shake_table.csv: {e}, using Pd data")
        return get_shake_off_data(None)


def clean_spectral_data(diagram_data: pd.DataFrame, satellite_data: pd.DataFrame,
                        satellite_data_complete: pd.DataFrame, auger_data: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Clean spectral data by dropping unnecessary columns.

    Parameters
    ----------
    diagram_data, satellite_data, satellite_data_complete, auger_data : pd.DataFrame

    Returns
    -------
    tuple of pd.DataFrame
        Cleaned DataFrames.
    """
    drop_cols = ['IS higher configuration', 'IS percentage', 'FS higher configuration', 'FS percentage', 'IS Configuration', 'FS Configuration']
    diagram_data_cleaned = diagram_data.drop(columns=drop_cols + ['number multipoles'])
    satellite_data_cleaned = satellite_data.drop(columns=drop_cols)
    auger_data_cleaned = auger_data.drop(columns=drop_cols)
    satellite_data_complete_cleaned = satellite_data_complete.drop(columns=drop_cols)
    return diagram_data_cleaned, satellite_data_cleaned, satellite_data_complete_cleaned, auger_data_cleaned


def process_diagram_intensities(diagram_data_cleaned: pd.DataFrame, auger_data_cleaned: pd.DataFrame,
                                shake_off_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Process diagram data to calculate intensities.

    Parameters
    ----------
    diagram_data_cleaned, auger_data_cleaned, shake_off_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame, float
        Processed diagram data and wK.
    """
    sum_diagram_rates = diagram_data_cleaned['rate [s-1]'].sum()
    sum_auger_rates = auger_data_cleaned['rate [s-1]'].sum()
    sum_shake_off = shake_off_df['Q1s'].sum()

    wK = sum_diagram_rates / (sum_diagram_rates + sum_auger_rates)

    diagram_data_cleaned = diagram_data_cleaned.copy()
    diagram_data_cleaned['gi'] = diagram_data_cleaned['IS 2JJ'] + 1
    diagram_data_cleaned['gf'] = diagram_data_cleaned.apply(lambda row: calculate_g(row['Shell IS']), axis=1)
    diagram_data_cleaned['Branching Ratio'] = diagram_data_cleaned['rate [s-1]'] / sum_diagram_rates
    diagram_data_cleaned['TYif'] = diagram_data_cleaned['Branching Ratio'] * wK
    diagram_data_cleaned['I'] = diagram_data_cleaned['TYif'] * (diagram_data_cleaned['gi'] / diagram_data_cleaned['gf'])
    diagram_data_cleaned['I_final'] = diagram_data_cleaned['I'] * (1 - sum_shake_off)

    return diagram_data_cleaned, wK


def process_satellite_intensities(satellite_data_complete_cleaned: pd.DataFrame, wk: float,
                                  shake_off_df: pd.DataFrame, holes_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process satellite data to calculate intensities.

    Parameters
    ----------
    satellite_data_complete_cleaned, shake_off_df, holes_data : pd.DataFrame
    wk : float

    Returns
    -------
    pd.DataFrame
        Processed satellite data.
    """
    satellite_data_complete_cleaned = satellite_data_complete_cleaned.copy()

    # Group
    shellgroup = satellite_data_complete_cleaned.groupby(['Shell IS', 'IS 2JJ', 'IS eigenvalue'])['rate [s-1]'].sum().reset_index()
    shellgroup.columns = ['Shell IS', 'IS 2JJ', 'IS eigenvalue', 'Sum Rate [s-1]']
    satellite_data_complete_cleaned = pd.merge(satellite_data_complete_cleaned, shellgroup, on=['Shell IS', 'IS 2JJ', 'IS eigenvalue'], how='left')

    # Calculate
    satellite_data_complete_cleaned['branching ratio'] = satellite_data_complete_cleaned['rate [s-1]'] / satellite_data_complete_cleaned['Sum Rate [s-1]']
    satellite_data_complete_cleaned['gi'] = satellite_data_complete_cleaned['IS 2JJ'] + 1
    satellite_data_complete_cleaned['gf'] = satellite_data_complete_cleaned.apply(lambda row: calculate_g(row['Shell IS']), axis=1)
    satellite_data_complete_cleaned['TYif'] = satellite_data_complete_cleaned['branching ratio'] * wk
    satellite_data_complete_cleaned['I'] = satellite_data_complete_cleaned['TYif'] * (satellite_data_complete_cleaned['gi'] / satellite_data_complete_cleaned['gf'])
    satellite_data_complete_cleaned['hole'] = satellite_data_complete_cleaned.apply(lambda row: get_spectator_hole(row['Shell IS'], row['Shell FS']), axis=1)

    # Strip
    for col in satellite_data_complete_cleaned.select_dtypes(include=['object']).columns:
        satellite_data_complete_cleaned[col] = satellite_data_complete_cleaned[col].str.strip()
    for col in holes_data.select_dtypes(include=['object']).columns:
        holes_data[col] = holes_data[col].str.strip()
    holes_data = holes_data.drop_duplicates()

    # Merge
    merged = pd.merge(satellite_data_complete_cleaned, holes_data[['Shell IS', 'IS 2JJ', 'IS eigenvalue', 'hole']],
                      on=['Shell IS', 'IS 2JJ', 'IS eigenvalue'], how='left', suffixes=('', '_new'))
    merged2 = pd.merge(merged, shake_off_df[['shell', 'Q1s']], left_on='hole', right_on='shell', how='left')
    merged2['I_final'] = merged2['I'] * merged2['Q1s']

    return merged2


def filter_satellite_data(satellite_data: pd.DataFrame, energy_min: float = 20891, energy_max: float = 21720) -> pd.DataFrame:
    """
    Filter satellite data by energy range.

    Parameters
    ----------
    satellite_data : pd.DataFrame
    energy_min, energy_max : float

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    filtered = satellite_data[
        (satellite_data['transition energy [eV]'] >= energy_min) &
        (satellite_data['transition energy [eV]'] <= energy_max)
    ].sort_values(by='transition energy [eV]')
    return filtered


def plot_k_alpha_spectrum(diagram_data: pd.DataFrame, satellite_data_filtered: pd.DataFrame,
                          title: str = "K-alpha Satellite Lines Intensity vs Transition Energy",
                          save_path: str = 'k_alpha_spectrum.png') -> None:
    """
    Plot the K-alpha spectrum.

    Parameters
    ----------
    diagram_data, satellite_data_filtered : pd.DataFrame
    title : str
    save_path : str
    """
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(satellite_data_filtered['transition energy [eV]'], satellite_data_filtered['I_final'],
           width=1.0, color='skyblue', alpha=0.8, label='Satellite Lines', edgecolor='black', linewidth=0.5)
    ax.bar(diagram_data['transition energy [eV]'], diagram_data['I_final'],
           width=2.0, color='coral', alpha=0.8, label='Diagram Lines', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Transition Energy [eV]', fontsize=14)
    ax.set_ylabel('Intensity (I_final)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def parse_config_occupations(config: str) -> Dict[str, int]:
    """
    Parse FAC relativistic configuration label to extract subshell occupations.

    Parameters
    ----------
    config : str
        FAC relativistic label, e.g., '4d+6(0)0' or '4d-3(3)3.4d+5(5)6'

    Returns
    -------
    Dict[str, int]
        Dictionary of subshell occupations

    Examples
    --------
    >>> parse_config_occupations('4d+6(0)0')
    {'4d+': 6}
    >>> parse_config_occupations('4d-3(3)3.4d+5(5)6')
    {'4d-': 3, '4d+': 5}
    """
    parts = config.split('.')
    occupations = {}
    for part in parts:
        # Match subshell and occupation
        match = re.match(r'(\d+[spdf][+-]?)(\d+)', part)
        if match:
            subshell = match.group(1)
            occ = int(match.group(2))
            # For s shells, remove + since s doesn't split
            if 's' in subshell and '+' in subshell:
                subshell = subshell.replace('+', '')
            occupations[subshell] = occ
        else:
            # For s shells without +
            match = re.match(r'(\d+s)(\d+)', part)
            if match:
                subshell = match.group(1)
                occ = int(match.group(2))
                occupations[subshell] = occ
    return occupations


def get_full_occupation(subshell: str) -> int:
    """
    Get the full occupation for a relativistic subshell.

    Parameters
    ----------
    subshell : str
        Subshell label, e.g., '4d-', '3p+', '2s'

    Returns
    -------
    int
        Full occupation
    """
    match = re.search(r'[spdf]', subshell)
    if not match:
        return 0
    l = match.group(0)
    if l == 's':
        return FULL_OCCUPATIONS['s']
    elif l == 'p':
        if '-' in subshell:
            return FULL_OCCUPATIONS['p-']
        elif '+' in subshell:
            return FULL_OCCUPATIONS['p+']
        else:
            return MAX_ELECTRONS['p']
    elif l == 'd':
        if '-' in subshell:
            return FULL_OCCUPATIONS['d-']
        elif '+' in subshell:
            return FULL_OCCUPATIONS['d+']
        else:
            return MAX_ELECTRONS['d']
    elif l == 'f':
        if '-' in subshell:
            return FULL_OCCUPATIONS['f-']
        elif '+' in subshell:
            return FULL_OCCUPATIONS['f+']
        else:
            return MAX_ELECTRONS['f']
    else:
        return 0


def identify_fac_holes(label: str, ground_occupations: Dict[str, int]) -> str:
    """
    Identify X-ray hole labels from FAC relativistic configuration label.

    Parameters
    ----------
    label : str
        FAC relativistic label
    ground_occupations : dict
        Ground state subshell occupations

    Returns
    -------
    str
        Comma-separated X-ray hole labels

    Examples
    --------
    >>> ground_occ = {'4d+': 6, '4d-': 4}
    >>> identify_fac_holes('4d+6(0)0', ground_occ)
    'Ground'
    >>> identify_fac_holes('4d+4(8)8', ground_occ)
    'N5, N5'
    >>> identify_fac_holes('4d-3(3)3.4d+5(5)6', ground_occ)
    'N4, N5'
    """
    from ..constants import SHELL_LABEL_MAP
    
    excited_occupations = parse_config_occupations(label)
    
    # Check if it's ground state
    if all(occ == ground_occupations.get(subshell, 0) for subshell, occ in excited_occupations.items()):
        return 'Ground'
    
    labels = []
    for subshell, occ in excited_occupations.items():
        ground_occ = ground_occupations.get(subshell, 0)
        holes = ground_occ - occ
        if holes > 0:
            xray_label = SHELL_LABEL_MAP.get(subshell)
            if xray_label:
                labels.extend([xray_label] * holes)
            else:
                labels.extend([f"{subshell}"] * holes)
    
    if not labels:
        return 'Unknown'
    
    return ', '.join(sorted(labels))


def get_fac_hole_to_shell_map() -> Dict[str, str]:
    """
    Get mapping from X-ray hole labels to relativistic shell notation for FAC data.

    Returns
    -------
    Dict[str, str]
        Mapping from X-ray labels to shell strings
    """
    from ..constants import SHELL_LABEL_MAP
    
    label_to_shell = {v: k for k, v in SHELL_LABEL_MAP.items()}
    label_to_shell.update({
        'L3': '2p3/2', 'M4': '3d3/2', 'M5': '3d5/2', 'N4': '4d3/2', 'N5': '4d5/2', 
        'N6': '4f3/2', 'N7': '4f5/2', 'O4': '5f3/2', 'O5': '5d3/2', 'O6': '5p3/2', 
        'O7': '5s', 'P4': '6f3/2', 'P5': '6d3/2', 'P6': '6p3/2', 'P7': '6s'
    })
    return label_to_shell


def map_fac_holes_to_shell(hole_str: str) -> str:
    """
    Map FAC X-ray hole labels to shell notation.

    Parameters
    ----------
    hole_str : str
        Comma-separated X-ray hole labels (e.g., 'K, N5')

    Returns
    -------
    str
        Underscore-separated shell strings (e.g., '1s_4d5/2')
    """
    label_to_shell = get_fac_hole_to_shell_map()
    holes = [h.strip() for h in hole_str.split(', ')]
    shells = [label_to_shell[h] if h in label_to_shell else h for h in holes]
    return '_'.join(shells)


def calculate_g_fac(hole_str: str) -> int:
    """
    Calculate statistical weight g for FAC hole labels.

    Parameters
    ----------
    hole_str : str
        Comma-separated X-ray hole labels

    Returns
    -------
    int
        Statistical weight
    """
    shell_str = map_fac_holes_to_shell(hole_str)
    return calculate_g(shell_str)


def load_fac_data(base_filename: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load FAC data files and add hole labels.

    Parameters
    ----------
    base_filename : str
        Base path to FAC files (without extension)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        levels_df, transitions_df, autoionization_df with hole labels added
    """
    from ..constants import SHELL_LABEL_MAP
    from ..readers import (read_fac, read_fac_autoionization,
                           read_fac_transitions)
    
    levels_df = read_fac(base_filename)
    transitions_df = read_fac_transitions(base_filename)
    autoionization_df = read_fac_autoionization(base_filename)
    
    # Build reference configuration from the CLOSED-SHELL CORE
    # FAC uses compact notation: only shows valence/excited electrons, not the closed core.
    # Strategy: 
    # 1. Identify ALL subshells that appear in ANY neutral ground state level
    #    These are the valence shells (partially filled) -> IGNORE them
    # 2. All other subshells that appear in excited states are either:
    #    a) Fully occupied core shells (not shown in ground) -> count holes relative to full
    #    b) Excited electrons beyond ground -> handled automatically
    
    # Get ALL neutral ground state configurations (ion_charge == 0)
    neutral_levels = levels_df[levels_df['ion_charge'] == 0]
    
    # Collect all subshells that appear in ANY neutral ground state
    # These are the VALENCE subshells (partially filled in ground state)
    valence_subshells = set()
    for label in neutral_levels['label']:
        occ = parse_config_occupations(label)
        valence_subshells.update(occ.keys())
    
    # Collect all subshells present in ANY configuration
    all_subshells = set()
    for label in levels_df['label']:
        all_subshells.update(parse_config_occupations(label).keys())
    
    # Build reference: 
    # - Subshells that appear in ground state (valence) -> IGNORE
    # - Subshells that DON'T appear in ground state (core) -> fully occupied
    ground_occupations = {}
    for subshell in all_subshells:
        if subshell not in valence_subshells:
            # This is a core subshell (doesn't appear in any ground state level)
            # It's implicitly fully occupied in ground state
            ground_occupations[subshell] = get_full_occupation(subshell)
    
    if len(ground_occupations) > 0:
        
        # Add hole labels
        levels_df['hole_labels'] = levels_df['label'].apply(
            lambda x: identify_fac_holes(x, ground_occupations)
        )
        
        # Fix hole labels for ionized ground states
        def fix_hole_labels(row):
            if row['hole_labels'] == 'Ground' and row['ion_charge'] > 0:
                conf_detail = str(row['conf_detail'])
                counts = {}
                for part in conf_detail.split('.'):
                    if '*' in part:
                        try:
                            n, count = part.split('*')
                            counts[int(n)] = int(count)
                        except ValueError:
                            continue
                hole_labels = []
                neutral_counts = {1: 2, 2: 8, 3: 18, 4: 18}
                for n in neutral_counts:
                    holes_n = neutral_counts[n] - counts.get(n, 0)
                    if holes_n > 0:
                        subshell = f"{n}s"
                        xray_label = SHELL_LABEL_MAP.get(subshell)
                        if xray_label:
                            hole_labels.extend([xray_label] * holes_n)
                if hole_labels:
                    return ', '.join(hole_labels)
            return row['hole_labels']
        
        levels_df['hole_labels'] = levels_df.apply(fix_hole_labels, axis=1)
        levels_df['holes'] = levels_df['hole_labels'].apply(
            lambda hl: 0 if hl == 'Ground' else len(hl.split(', ')) if hl not in ['Unknown'] else 0
        )
    else:
        levels_df['hole_labels'] = 'Unknown'
        levels_df['holes'] = 0
    
    return levels_df, transitions_df, autoionization_df


def process_fac_diagram_intensities(diagram_lines: pd.DataFrame, levels_df: pd.DataFrame, 
                                    wK: float, sum_shake_off: float) -> pd.DataFrame:
    """
    Process FAC diagram lines to calculate intensities.

    Parameters
    ----------
    diagram_lines : pd.DataFrame
        Diagram transitions
    levels_df : pd.DataFrame
        Levels data with 2j values
    wK : float
        Fluorescence yield
    sum_shake_off : float
        Total shake-off probability

    Returns
    -------
    pd.DataFrame
        Processed diagram data with intensities
    """
    diagram_rate_sum = diagram_lines['A'].sum()
    level_2j_map = levels_df.set_index('level_index')['2j'].to_dict()
    
    def get_deg(level_index: int) -> int:
        two_j = level_2j_map.get(level_index, 0)
        return int(two_j) + 1
    
    diagram_lines = diagram_lines.copy()
    diagram_lines['Branching Ratio'] = diagram_lines['A'] / diagram_rate_sum if diagram_rate_sum > 0 else 0.0
    diagram_lines['gi'] = diagram_lines['level_index_upper'].apply(get_deg)
    diagram_lines['gf'] = diagram_lines['hole_labels_upper'].apply(calculate_g_fac)
    diagram_lines['TYif'] = diagram_lines['Branching Ratio'] * wK
    diagram_lines['I'] = diagram_lines['TYif'] * (diagram_lines['gi'] / diagram_lines['gf'])
    diagram_lines['I_final'] = diagram_lines['I'] * (1 - sum_shake_off)
    
    return diagram_lines


def process_fac_satellite_intensities(satellite_lines: pd.DataFrame, levels_df: pd.DataFrame,
                                      wK: float, shake_off_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process FAC satellite lines to calculate intensities.

    Parameters
    ----------
    satellite_lines : pd.DataFrame
        Satellite transitions
    levels_df : pd.DataFrame
        Levels data with 2j values
    wK : float
        Fluorescence yield
    shake_off_df : pd.DataFrame
        Shake-off probabilities

    Returns
    -------
    pd.DataFrame
        Processed satellite data with intensities
    """
    level_2j_map = levels_df.set_index('level_index')['2j'].to_dict()
    
    def get_deg(level_index: int) -> int:
        two_j = level_2j_map.get(level_index, 0)
        return int(two_j) + 1
    
    satellite_lines = satellite_lines.copy()
    satellite_lines['gi'] = satellite_lines['level_index_upper'].apply(get_deg)
    satellite_lines['gf'] = satellite_lines['hole_labels_upper'].apply(calculate_g_fac)
    
    # Sum A per upper level
    upper_group_sum = satellite_lines.groupby('level_index_upper')['A'].transform('sum')
    satellite_lines['Branching Ratio'] = satellite_lines['A'] / upper_group_sum.replace(0, 1)
    satellite_lines['TYif'] = satellite_lines['Branching Ratio'] * wK
    satellite_lines['I_raw'] = satellite_lines['TYif'] * (satellite_lines['gi'] / satellite_lines['gf'])
    
    # Get spectator hole
    satellite_lines['spectator'] = satellite_lines.apply(
        lambda row: ', '.join(set(row['hole_labels_upper'].split(', ')) & 
                            set(row['hole_labels_lower'].split(', '))), 
        axis=1
    )
    satellite_lines['spectator_shell'] = satellite_lines['spectator'].apply(
        lambda h: map_fac_holes_to_shell(h) if h else None
    )
    
    # Merge with shake_off_df
    satellite_lines = satellite_lines.merge(
        shake_off_df[['shell', 'Q1s']], 
        left_on='spectator_shell', 
        right_on='shell', 
        how='left'
    )
    satellite_lines['I_final'] = satellite_lines['I_raw'] * satellite_lines['Q1s']
    
    return satellite_lines


def add_fac_transition_energies_and_holes(transitions_df: pd.DataFrame, 
                                          levels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transition energies and hole labels to FAC transitions.
    
    Parameters
    ----------
    transitions_df : pd.DataFrame
        FAC transitions data
    levels_df : pd.DataFrame
        FAC levels data with hole labels
        
    Returns
    -------
    pd.DataFrame
        Transitions with energy, hole labels, and ion charges added
    """
    trans_with_energy = transitions_df.merge(
        levels_df[['level_index', 'energy', 'hole_labels', 'holes', 'ion_charge']].rename(
            columns={'energy': 'energy_lower', 'hole_labels': 'hole_labels_lower', 
                    'holes': 'holes_lower', 'ion_charge': 'ion_charge_lower'}
        ),
        left_on='level_index_lower', right_on='level_index', how='left'
    ).merge(
        levels_df[['level_index', 'energy', 'hole_labels', 'holes', 'ion_charge']].rename(
            columns={'energy': 'energy_upper', 'hole_labels': 'hole_labels_upper', 
                    'holes': 'holes_upper', 'ion_charge': 'ion_charge_upper'}
        ),
        left_on='level_index_upper', right_on='level_index', how='left'
    ).drop(['level_index_x', 'level_index_y'], axis=1)
    
    trans_with_energy['delta_energy'] = trans_with_energy['energy_upper'] - trans_with_energy['energy_lower']
    
    return trans_with_energy


def filter_fac_k_alpha_transitions(trans_with_energy: pd.DataFrame, shell: str = 'K') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter FAC transitions to diagram and satellite lines for a given shell.
    
    Parameters
    ----------
    trans_with_energy : pd.DataFrame
        Transitions with energies and hole labels
    shell : str, optional
        Shell to analyze (e.g., 'K', 'L1', 'L2', 'L3'), default 'K'
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (diagram_lines, satellite_lines)
    """
    # Diagram: single-hole to single-hole, from specified shell
    diagram_mask = (
        (trans_with_energy['holes_upper'] == 1) & 
        (trans_with_energy['holes_lower'] == 1) & 
        (trans_with_energy['hole_labels_upper'] == shell)
    )
    diagram_lines = trans_with_energy[diagram_mask].copy()
    
    # Satellites: multiple holes with specified shell involved
    satellite_mask = (
        (trans_with_energy['holes_upper'] > 1) & 
        trans_with_energy['hole_labels_upper'].str.contains(shell, na=False)
    )
    satellite_lines = trans_with_energy[satellite_mask].copy()
    
    return diagram_lines, satellite_lines


def calculate_fac_wk(diagram_lines: pd.DataFrame, 
                     autoionization_df: pd.DataFrame,
                     levels_df: pd.DataFrame,
                     shell: str = 'K') -> Tuple[float, float, float]:
    """
    Calculate fluorescence yield for FAC data for a given shell.
    
    Parameters
    ----------
    diagram_lines : pd.DataFrame
        Diagram transitions for the shell
    autoionization_df : pd.DataFrame
        FAC autoionization data
    levels_df : pd.DataFrame
        Levels data with hole labels
    shell : str, optional
        Shell to analyze (e.g., 'K', 'L1', 'L2', 'L3'), default 'K'
        
    Returns
    -------
    Tuple[float, float, float]
        (fluorescence_yield, diagram_rate_sum, auger_rate_sum)
    """
    # Add hole labels to autoionization data
    hole_labels_map = levels_df.set_index('level_index')['hole_labels'].to_dict()
    auto_df = autoionization_df.copy()
    auto_df['hole_labels_upper'] = auto_df['level_index_upper'].map(hole_labels_map)
    
    # Sum rates for specified shell
    diagram_rate_sum = diagram_lines[diagram_lines['hole_labels_upper'] == shell]['A'].sum()
    auger_shell = auto_df[auto_df['hole_labels_upper'] == shell]
    auger_rate_sum = auger_shell['ai_rate'].sum()
    
    # Calculate fluorescence yield
    w_shell = diagram_rate_sum / (diagram_rate_sum + auger_rate_sum) if (diagram_rate_sum + auger_rate_sum) > 0 else 0.0
    
    return w_shell, diagram_rate_sum, auger_rate_sum


def plot_fac_k_alpha_spectrum(diagram_df: pd.DataFrame, 
                               satellite_df: pd.DataFrame,
                               energy_range: Tuple[float, float] = (20800, 21300),
                               output_file: str = 'fac_k_alpha_spectrum.png',
                               title: str = 'FAC K-alpha Spectrum') -> None:
    """
    Plot FAC spectrum with diagram and satellite lines.
    
    Parameters
    ----------
    diagram_df : pd.DataFrame
        Processed diagram lines with I_final column
    satellite_df : pd.DataFrame
        Processed satellite lines with I_final column
    energy_range : Tuple[float, float], optional
        Energy range for plot in eV
    output_file : str, optional
        Output filename
    title : str, optional
        Plot title
    """
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter and plot satellites if provided
    if satellite_df is not None and not satellite_df.empty:
        satellite_filtered = satellite_df[
            (satellite_df['transition energy [eV]'] >= energy_range[0]) & 
            (satellite_df['transition energy [eV]'] <= energy_range[1])
        ]
        if not satellite_filtered.empty:
            ax.bar(satellite_filtered['transition energy [eV]'], satellite_filtered['I_final'],
                   width=0.5, color='skyblue', label='Satellite Lines', linewidth=0.1)
    
    # Plot diagram lines if provided
    if diagram_df is not None and not diagram_df.empty:
        ax.bar(diagram_df['transition energy [eV]'], diagram_df['I_final'],
               width=0.5, color='coral', label='Diagram Lines', linewidth=0.1)
    
    ax.set_xlabel('Transition Energy [eV]', fontsize=14)
    ax.set_ylabel('Intensity (I_final)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlim(energy_range)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)