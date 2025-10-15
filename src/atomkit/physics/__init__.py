"""
Physics calculations module for atomkit.

This module provides tools for calculating various atomic physics quantities
such as cross sections, rates, and resonance profiles.
"""

from .cross_sections import (
    ResonantExcitationCalculator,
    calculate_resonant_excitation_cross_section,
    LorentzianProfile,
)

from .units import (
    EnergyConverter,
    energy_converter,
    cross_section_to_collision_strength,
    collision_strength_to_cross_section,
    effective_collision_strength_maxwellian,
    RY_TO_EV,
    EV_TO_RY,
    EV_TO_CM_INV,
    CM_INV_TO_EV,
)

from .plotting import (
    ResonancePlotter,
    quick_plot_cross_section,
)

from .potentials import (
    EffectivePotentialCalculator,
)

__all__ = [
    # Cross section calculations
    "ResonantExcitationCalculator",
    "calculate_resonant_excitation_cross_section",
    "LorentzianProfile",
    # Unit conversions
    "EnergyConverter",
    "energy_converter",
    "cross_section_to_collision_strength",
    "collision_strength_to_cross_section",
    "effective_collision_strength_maxwellian",
    "RY_TO_EV",
    "EV_TO_RY",
    "EV_TO_CM_INV",
    "CM_INV_TO_EV",
    # Plotting
    "ResonancePlotter",
    "quick_plot_cross_section",
    # Effective potentials
    "EffectivePotentialCalculator",
]
