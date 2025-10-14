"""
Cross section calculations for electron-ion interactions.

This module provides tools for calculating electron impact excitation and
ionization cross sections, including resonant contributions via autoionization.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy.constants import hbar, m_e, e, c


class LorentzianProfile:
    """
    Lorentzian (Breit-Wigner) resonance profile.

    Used for modeling resonant features in cross sections.
    """

    def __init__(self, energy_center: float, gamma: float):
        """
        Initialize a Lorentzian profile.

        Parameters
        ----------
        energy_center : float
            Center energy of the resonance in eV.
        gamma : float
            Full width at half maximum (FWHM) of the resonance in eV.
        """
        self.energy_center = energy_center
        self.gamma = gamma

    def __call__(self, energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the Lorentzian profile at given energy/energies.

        Parameters
        ----------
        energy : float or array-like
            Energy or energies at which to evaluate the profile (eV).

        Returns
        -------
        float or ndarray
            Normalized Lorentzian profile value(s).
        """
        return (self.gamma / (2 * np.pi)) / (
            (energy - self.energy_center) ** 2 + (self.gamma / 2) ** 2
        )


class ResonantExcitationCalculator:
    """
    Calculator for resonant electron impact excitation cross sections.

    This class implements the calculation of resonant excitation cross sections
    via dielectronic capture followed by autoionization. The process is:

    1. Electron capture into a doubly-excited autoionizing state (resonance)
    2. Autoionization back to the target ion in an excited state

    The cross section includes:
    - Capture cross section (energy-dependent, Lorentzian profile)
    - Branching ratio (probability of decay to final state)

    Attributes
    ----------
    hbar_eV_s : float
        Reduced Planck constant in eV·s
    m_e_eV_c2 : float
        Electron rest mass in eV/c²
    c_cm_s : float
        Speed of light in cm/s
    prefactor : float
        Prefactor for cross section calculation in eV²·cm²
    """

    # Physical constants
    hbar_eV_s = hbar / e  # ℏ in eV·s
    m_e_eV_c2 = 510998.9461  # Electron mass in eV/c²
    c_cm_s = c * 100  # Speed of light in cm/s

    # Prefactor: π(ℏc)²/(2m_e c²)
    prefactor = np.pi * ((hbar_eV_s * c_cm_s) ** 2) / (2 * m_e_eV_c2)

    def __init__(
        self,
        levels: pd.DataFrame,
        autoionization: pd.DataFrame,
        transitions: pd.DataFrame,
    ):
        """
        Initialize the resonant excitation calculator.

        Parameters
        ----------
        levels : pd.DataFrame
            Energy levels data with columns: level_index, energy, 2j, ion_charge, etc.
        autoionization : pd.DataFrame
            Autoionization rates with columns: level_index_upper, level_index_lower,
            ai_rate, energy, etc.
        transitions : pd.DataFrame
            Radiative transition rates with columns: level_index_upper,
            level_index_lower, A (Einstein A coefficient), etc.
        """
        self.levels = levels
        self.autoionization = autoionization
        self.transitions = transitions

    def get_level_by_index(self, level_index: int) -> pd.Series:
        """
        Get level data by level index.

        Parameters
        ----------
        level_index : int
            The level index to retrieve.

        Returns
        -------
        pd.Series
            Level data for the specified index.

        Raises
        ------
        ValueError
            If level index is not found.
        """
        level = self.levels[self.levels["level_index"] == level_index]
        if len(level) == 0:
            raise ValueError(f"Level index {level_index} not found in levels data")
        return level.iloc[0]

    def get_level_by_config(
        self, configuration: str, ion_charge: Optional[int] = None
    ) -> pd.Series:
        """
        Get level data by electronic configuration string.

        Parameters
        ----------
        configuration : str
            Electronic configuration (e.g., "1s2 2s1").
        ion_charge : int, optional
            Ion charge state to filter by. If None, uses the first match.

        Returns
        -------
        pd.Series
            Level data for the specified configuration.

        Raises
        ------
        ValueError
            If configuration is not found.
        """
        mask = self.levels["configuration"] == configuration
        if ion_charge is not None:
            mask &= self.levels["ion_charge"] == ion_charge

        level = self.levels[mask]
        if len(level) == 0:
            msg = f"Configuration '{configuration}' not found"
            if ion_charge is not None:
                msg += f" for ion charge {ion_charge}"
            raise ValueError(msg)
        return level.iloc[0]

    def calculate_total_decay_rate(
        self, resonant_level_index: int
    ) -> tuple[float, float]:
        """
        Calculate total decay rate of a resonant level.

        Parameters
        ----------
        resonant_level_index : int
            Level index of the resonant (autoionizing) state.

        Returns
        -------
        Aa_total : float
            Total autoionization rate (sum of all autoionization channels) in s⁻¹.
        Ar_total : float
            Total radiative decay rate (sum of all radiative transitions) in s⁻¹.
        """
        # Total autoionization rate
        Aa_total = self.autoionization[
            self.autoionization["level_index_upper"] == resonant_level_index
        ]["ai_rate"].sum()

        # Total radiative decay rate
        Ar_total = self.transitions[
            self.transitions["level_index_upper"] == resonant_level_index
        ]["A"].sum()

        return Aa_total, Ar_total

    def calculate_capture_cross_section(
        self,
        initial_level: Union[int, str, pd.Series],
        resonant_level_index: int,
        energy_grid: np.ndarray,
        ion_charge: Optional[int] = None,
    ) -> tuple[np.ndarray, float, float]:
        """
        Calculate electron capture cross section to a resonant state.

        Parameters
        ----------
        initial_level : int, str, or pd.Series
            Initial level specified by index, configuration string, or level data.
        resonant_level_index : int
            Level index of the resonant (autoionizing) state.
        energy_grid : np.ndarray
            Energy grid for cross section calculation (eV).
        ion_charge : int, optional
            Ion charge state (used when initial_level is a string).

        Returns
        -------
        capture_cs : np.ndarray
            Capture cross section on the energy grid (cm²).
        resonance_energy : float
            Resonance energy (eV).
        gamma_total : float
            Total width of the resonance (eV).
        """
        # Get initial level data
        if isinstance(initial_level, (int, np.integer)):
            initial_state = self.get_level_by_index(initial_level)
        elif isinstance(initial_level, str):
            initial_state = self.get_level_by_config(initial_level, ion_charge)
        else:
            initial_state = initial_level

        # Get resonant level data
        resonant_state = self.get_level_by_index(resonant_level_index)

        # Get decay rates
        Aa_total, Ar_total = self.calculate_total_decay_rate(resonant_level_index)

        # Get specific autoionization rate from initial state
        ai_data = self.autoionization[
            (self.autoionization["level_index_upper"] == resonant_level_index)
            & (self.autoionization["level_index_lower"] == initial_state["level_index"])
        ]

        if len(ai_data) == 0:
            # No direct capture channel from this initial state
            return np.zeros_like(energy_grid), 0.0, 0.0

        Aa_initial = ai_data["ai_rate"].iloc[0]
        resonance_energy = ai_data["energy"].iloc[0]

        # Statistical weights
        g_res = resonant_state["2j"] + 1
        g_i = initial_state["2j"] + 1

        # Total width (in eV)
        Gamma_total = (Aa_total + Ar_total) * self.hbar_eV_s

        # Lorentzian profile
        lorentzian = (Gamma_total / (2 * np.pi)) / (
            (energy_grid - resonance_energy) ** 2 + (Gamma_total / 2) ** 2
        )

        unnormalizing_factor_eV = 2 * np.pi * self.hbar_eV_s * Aa_initial

        # Capture cross section
        capture_cs = (
            (self.prefactor / energy_grid)
            * (g_res / g_i)
            * (unnormalizing_factor_eV * lorentzian)
        )

        return capture_cs, resonance_energy, Gamma_total

    def calculate_branching_ratio(
        self,
        resonant_level_index: int,
        final_level: Union[int, str, pd.Series],
        ion_charge: Optional[int] = None,
    ) -> float:
        """
        Calculate branching ratio for decay to a specific final state.

        Parameters
        ----------
        resonant_level_index : int
            Level index of the resonant (autoionizing) state.
        final_level : int, str, or pd.Series
            Final level specified by index, configuration string, or level data.
        ion_charge : int, optional
            Ion charge state (used when final_level is a string).

        Returns
        -------
        float
            Branching ratio (probability of decay to the final state).
        """
        # Get final level data
        if isinstance(final_level, (int, np.integer)):
            final_state = self.get_level_by_index(final_level)
        elif isinstance(final_level, str):
            final_state = self.get_level_by_config(final_level, ion_charge)
        else:
            final_state = final_level

        # Get decay rates
        Aa_total, Ar_total = self.calculate_total_decay_rate(resonant_level_index)
        Gamma_total = Aa_total + Ar_total

        if Gamma_total == 0:
            return 0.0

        # Get specific autoionization rate to final state
        ai_data = self.autoionization[
            (self.autoionization["level_index_upper"] == resonant_level_index)
            & (self.autoionization["level_index_lower"] == final_state["level_index"])
        ]

        if len(ai_data) == 0:
            return 0.0

        Aa_final = ai_data["ai_rate"].iloc[0]

        return Aa_final / Gamma_total

    def calculate_resonant_excitation(
        self,
        initial_level: Union[int, str, pd.Series],
        final_level: Union[int, str, pd.Series],
        energy_grid: np.ndarray,
        ion_charge: Optional[int] = None,
        resonant_levels: Optional[list] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Calculate total resonant excitation cross section.

        This calculates the cross section for electron impact excitation from
        an initial state to a final state via resonant capture and autoionization.

        Parameters
        ----------
        initial_level : int, str, or pd.Series
            Initial level (before electron impact).
        final_level : int, str, or pd.Series
            Final level (after electron impact excitation).
        energy_grid : np.ndarray
            Energy grid for cross section calculation (eV).
        ion_charge : int, optional
            Ion charge state (used when levels are specified as strings).
        resonant_levels : list, optional
            List of resonant level indices to include. If None, includes all
            resonant levels that connect the initial and final states.

        Returns
        -------
        total_cs : np.ndarray
            Total resonant excitation cross section on the energy grid (cm²).
        resonance_info : dict
            Dictionary containing information about each resonance contribution:
            - 'level_index': resonant level indices
            - 'energies': resonance energies (eV)
            - 'widths': resonance widths (eV)
            - 'contributions': individual cross section contributions
        """
        # Get level data
        if isinstance(initial_level, (int, np.integer)):
            initial_state = self.get_level_by_index(initial_level)
        elif isinstance(initial_level, str):
            initial_state = self.get_level_by_config(initial_level, ion_charge)
        else:
            initial_state = initial_level

        if isinstance(final_level, (int, np.integer)):
            final_state = self.get_level_by_index(final_level)
        elif isinstance(final_level, str):
            final_state = self.get_level_by_config(final_level, ion_charge)
        else:
            final_state = final_level

        # Determine which resonant levels to include
        if resonant_levels is None:
            # Find all resonant levels that connect initial and final states
            has_capture = self.autoionization[
                self.autoionization["level_index_lower"] == initial_state["level_index"]
            ]["level_index_upper"].unique()

            has_decay = self.autoionization[
                self.autoionization["level_index_lower"] == final_state["level_index"]
            ]["level_index_upper"].unique()

            resonant_levels = list(set(has_capture) & set(has_decay))

        # Initialize arrays
        total_cs = np.zeros_like(energy_grid, dtype=float)

        resonance_info = {
            "level_index": [],
            "energies": [],
            "widths": [],
            "contributions": [],
        }

        # Sum over all resonant levels
        for res_level in resonant_levels:
            # Calculate capture cross section
            capture_cs, res_energy, gamma = self.calculate_capture_cross_section(
                initial_state, res_level, energy_grid
            )

            # Calculate branching ratio to final state
            branching = self.calculate_branching_ratio(res_level, final_state)

            # Resonant contribution
            resonant_cs = capture_cs * branching
            total_cs += resonant_cs

            # Store resonance information
            resonance_info["level_index"].append(res_level)
            resonance_info["energies"].append(res_energy)
            resonance_info["widths"].append(gamma)
            resonance_info["contributions"].append(resonant_cs)

        return total_cs, resonance_info

    def plot_cross_section(
        self, energy_grid: np.ndarray, cross_section: np.ndarray, ax=None, **kwargs
    ):
        """
        Plot cross section vs energy.

        Parameters
        ----------
        energy_grid : np.ndarray
            Energy grid (eV).
        cross_section : np.ndarray
            Cross section values (cm²).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        **kwargs
            Additional keyword arguments passed to plt.plot().

        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        ax.plot(energy_grid, cross_section, **kwargs)
        ax.set_xlabel("Electron Energy (eV)")
        ax.set_ylabel("Cross Section (cm²)")
        ax.set_title("Resonant Excitation Cross Section")
        ax.grid(True, alpha=0.3)

        return fig, ax


def calculate_resonant_excitation_cross_section(
    initial_level: Union[int, str],
    final_level: Union[int, str],
    levels: pd.DataFrame,
    autoionization: pd.DataFrame,
    transitions: pd.DataFrame,
    energy_grid: np.ndarray,
    ion_charge: Optional[int] = None,
    resonant_levels: Optional[list] = None,
) -> tuple[np.ndarray, dict]:
    """
    Convenience function to calculate resonant excitation cross section.

    This is a simpler interface to ResonantExcitationCalculator for one-off calculations.

    Parameters
    ----------
    initial_level : int or str
        Initial level (level index or configuration string).
    final_level : int or str
        Final level (level index or configuration string).
    levels : pd.DataFrame
        Energy levels data.
    autoionization : pd.DataFrame
        Autoionization rates data.
    transitions : pd.DataFrame
        Radiative transition rates data.
    energy_grid : np.ndarray
        Energy grid for cross section calculation (eV).
    ion_charge : int, optional
        Ion charge state (required if levels are specified as strings).
    resonant_levels : list, optional
        List of resonant level indices to include.

    Returns
    -------
    cross_section : np.ndarray
        Total resonant excitation cross section (cm²).
    resonance_info : dict
        Information about resonance contributions.

    Examples
    --------
    >>> # Using level indices
    >>> energy = np.linspace(6000, 7000, 1000)
    >>> cs, info = calculate_resonant_excitation_cross_section(
    ...     initial_level=0,
    ...     final_level=1,
    ...     levels=levels_df,
    ...     autoionization=ai_df,
    ...     transitions=tr_df,
    ...     energy_grid=energy,
    ... )

    >>> # Using configuration strings
    >>> cs, info = calculate_resonant_excitation_cross_section(
    ...     initial_level="1s2 2s1",
    ...     final_level="1s2 2p1",
    ...     levels=levels_df,
    ...     autoionization=ai_df,
    ...     transitions=tr_df,
    ...     energy_grid=energy,
    ...     ion_charge=23,
    ... )
    """
    calculator = ResonantExcitationCalculator(levels, autoionization, transitions)

    return calculator.calculate_resonant_excitation(
        initial_level=initial_level,
        final_level=final_level,
        energy_grid=energy_grid,
        ion_charge=ion_charge,
        resonant_levels=resonant_levels,
    )
