"""
Unit conversion utilities for atomic physics calculations.

This module provides tools for converting between different energy units
and between cross sections and collision strengths.
"""

import numpy as np
from typing import Union
from scipy.constants import physical_constants, h, c, e

# Physical constants
RY_TO_EV = physical_constants["Rydberg constant times hc in eV"][
    0
]  # 13.605693122994 eV
EV_TO_RY = 1.0 / RY_TO_EV
EV_TO_CM_INV = e / (h * c * 100)  # eV to cm^-1 conversion
CM_INV_TO_EV = 1.0 / EV_TO_CM_INV
EV_TO_JOULE = e
JOULE_TO_EV = 1.0 / e
EV_TO_HZ = e / h
HZ_TO_EV = h / e
BOHR_RADIUS_CM = physical_constants["Bohr radius"][0] * 100  # in cm
PI_A0_SQUARED = np.pi * (BOHR_RADIUS_CM**2)  # π a₀² in cm²


class EnergyConverter:
    """
    Utility class for converting between different energy units.

    Supports: eV, Rydberg, cm⁻¹, Joule, Hz, Kelvin, Hartree

    Examples
    --------
    >>> converter = EnergyConverter()
    >>> energy_ry = converter.ev_to_rydberg(13.6)
    >>> energy_cm = converter.ev_to_wavenumber(1.0)
    >>> # Or use the convert method
    >>> energy_ry = converter.convert(13.6, from_unit='eV', to_unit='Ry')
    """

    # Conversion factors to eV
    _to_ev = {
        "eV": 1.0,
        "Ry": RY_TO_EV,
        "Rydberg": RY_TO_EV,
        "cm-1": CM_INV_TO_EV,
        "cm^-1": CM_INV_TO_EV,
        "wavenumber": CM_INV_TO_EV,
        "J": JOULE_TO_EV,
        "Joule": JOULE_TO_EV,
        "Hz": HZ_TO_EV,
        "Hertz": HZ_TO_EV,
        "K": physical_constants["Boltzmann constant in eV/K"][0],
        "Kelvin": physical_constants["Boltzmann constant in eV/K"][0],
        "Ha": physical_constants["Hartree energy in eV"][0],
        "Hartree": physical_constants["Hartree energy in eV"][0],
    }

    # Conversion factors from eV
    _from_ev = {k: 1.0 / v for k, v in _to_ev.items() if k != "eV"}
    _from_ev["eV"] = 1.0

    def convert(
        self,
        energy: Union[float, np.ndarray],
        from_unit: str = "eV",
        to_unit: str = "eV",
    ) -> Union[float, np.ndarray]:
        """
        Convert energy from one unit to another.

        Parameters
        ----------
        energy : float or array-like
            Energy value(s) to convert.
        from_unit : str
            Input energy unit. Options: 'eV', 'Ry', 'cm-1', 'J', 'Hz', 'K', 'Ha'
        to_unit : str
            Output energy unit. Same options as from_unit.

        Returns
        -------
        float or ndarray
            Converted energy value(s).

        Examples
        --------
        >>> converter = EnergyConverter()
        >>> converter.convert(13.6, 'eV', 'Ry')
        1.0
        >>> converter.convert(1.0, 'Ry', 'cm-1')
        109737.31568160...
        """
        if from_unit not in self._to_ev:
            raise ValueError(
                f"Unknown unit '{from_unit}'. "
                f"Supported units: {list(self._to_ev.keys())}"
            )
        if to_unit not in self._from_ev:
            raise ValueError(
                f"Unknown unit '{to_unit}'. "
                f"Supported units: {list(self._from_ev.keys())}"
            )

        # Convert to eV first, then to target unit
        energy_ev = energy * self._to_ev[from_unit]
        return energy_ev * self._from_ev[to_unit]

    # Convenience methods for common conversions

    def ev_to_rydberg(
        self, energy_ev: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert eV to Rydberg."""
        return energy_ev * EV_TO_RY

    def rydberg_to_ev(
        self, energy_ry: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert Rydberg to eV."""
        return energy_ry * RY_TO_EV

    def ev_to_wavenumber(
        self, energy_ev: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert eV to cm⁻¹."""
        return energy_ev * EV_TO_CM_INV

    def wavenumber_to_ev(
        self, wavenumber: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert cm⁻¹ to eV."""
        return wavenumber * CM_INV_TO_EV

    def ev_to_joule(
        self, energy_ev: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert eV to Joule."""
        return energy_ev * EV_TO_JOULE

    def joule_to_ev(
        self, energy_j: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert Joule to eV."""
        return energy_j * JOULE_TO_EV

    def ev_to_hz(self, energy_ev: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert eV to Hz."""
        return energy_ev * EV_TO_HZ

    def hz_to_ev(self, frequency: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert Hz to eV."""
        return frequency * HZ_TO_EV

    def ev_to_kelvin(
        self, energy_ev: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert eV to Kelvin."""
        return self.convert(energy_ev, "eV", "K")

    def kelvin_to_ev(
        self, temp_k: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert Kelvin to eV."""
        return self.convert(temp_k, "K", "eV")

    def ev_to_hartree(
        self, energy_ev: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert eV to Hartree."""
        return self.convert(energy_ev, "eV", "Ha")

    def hartree_to_ev(
        self, energy_ha: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert Hartree to eV."""
        return self.convert(energy_ha, "Ha", "eV")


def cross_section_to_collision_strength(
    cross_section: Union[float, np.ndarray],
    energy: Union[float, np.ndarray],
    g_initial: int,
) -> Union[float, np.ndarray]:
    """
    Convert cross section to collision strength (Omega).

    The collision strength is a dimensionless quantity defined as:

        Ω = (k²/πa₀²) σ = (2mE/ħ²πa₀²) σ

    where k is the wave vector, a₀ is the Bohr radius, and σ is the cross section.

    For electron impact excitation:
        Ω = (g_i / π a₀²) × k² × σ

    where g_i is the statistical weight of the initial level.

    Parameters
    ----------
    cross_section : float or array-like
        Cross section in cm².
    energy : float or array-like
        Electron energy in eV.
    g_initial : int
        Statistical weight of initial level (2J+1).

    Returns
    -------
    float or ndarray
        Collision strength (dimensionless).

    Notes
    -----
    The collision strength is related to the cross section by:
        σ = (π a₀²/k²) Ω/g_i

    where k² = 2mE/ħ² (in atomic units: k² = 2E in Rydberg).

    References
    ----------
    Van Regemorter, H. (1962). ApJ, 136, 906.

    Examples
    --------
    >>> sigma = 1e-18  # cm²
    >>> E = 100  # eV
    >>> g_i = 2
    >>> omega = cross_section_to_collision_strength(sigma, E, g_i)
    """
    # Convert energy to Rydberg (atomic units)
    energy_ry = energy * EV_TO_RY

    # Wave vector squared: k² = 2E (in Rydberg units)
    k_squared = 2.0 * energy_ry

    # Collision strength: Ω = (k²/πa₀²) × g_i × σ
    omega = (k_squared / PI_A0_SQUARED) * g_initial * cross_section

    return omega


def collision_strength_to_cross_section(
    collision_strength: Union[float, np.ndarray],
    energy: Union[float, np.ndarray],
    g_initial: int,
) -> Union[float, np.ndarray]:
    """
    Convert collision strength (Omega) to cross section.

    Parameters
    ----------
    collision_strength : float or array-like
        Collision strength (dimensionless).
    energy : float or array-like
        Electron energy in eV.
    g_initial : int
        Statistical weight of initial level (2J+1).

    Returns
    -------
    float or ndarray
        Cross section in cm².

    See Also
    --------
    cross_section_to_collision_strength : Inverse conversion

    Examples
    --------
    >>> omega = 0.5
    >>> E = 100  # eV
    >>> g_i = 2
    >>> sigma = collision_strength_to_cross_section(omega, E, g_i)
    """
    # Convert energy to Rydberg (atomic units)
    energy_ry = energy * EV_TO_RY

    # Wave vector squared: k² = 2E (in Rydberg units)
    k_squared = 2.0 * energy_ry

    # Cross section: σ = (πa₀²/k²) × Ω/g_i
    cross_section = (PI_A0_SQUARED / k_squared) * (collision_strength / g_initial)

    return cross_section


def effective_collision_strength_maxwellian(
    collision_strength_func,
    temperature: float,
    energy_range: tuple = (0.1, 1000),
    num_points: int = 1000,
) -> Union[float, np.ndarray]:
    """
    Calculate Maxwell-averaged effective collision strength.

    The effective collision strength is:

        Υ(T) = ∫₀^∞ Ω(E) exp(-E/kT) d(E/kT)

    Parameters
    ----------
    collision_strength_func : callable
        Function that takes energy (eV) and returns collision strength.
    temperature : float
        Temperature in Kelvin.
    energy_range : tuple
        (min_energy, max_energy) in eV for integration.
    num_points : int
        Number of integration points.

    Returns
    -------
    float
        Effective collision strength.

    Examples
    --------
    >>> def omega_func(E):
    ...     return 0.5 * np.ones_like(E)  # Constant collision strength
    >>> T = 1e6  # K
    >>> upsilon = effective_collision_strength_maxwellian(omega_func, T)
    """
    from scipy.integrate import simpson

    # Convert temperature to eV
    kT = temperature * physical_constants["Boltzmann constant in eV/K"][0]

    # Create energy grid
    energies = np.linspace(energy_range[0], energy_range[1], num_points)

    # Calculate collision strength at each energy
    omega = collision_strength_func(energies)

    # Integrand: Ω(E) exp(-E/kT)
    integrand = omega * np.exp(-energies / kT)

    # Integrate using Simpson's rule
    # Note: d(E/kT) = dE/kT
    upsilon = simpson(integrand, x=energies) / kT

    return upsilon


# Convenience instances
energy_converter = EnergyConverter()
