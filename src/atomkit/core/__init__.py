"""
Unified atomic calculation interface.

This module provides a code-agnostic interface for atomic structure calculations,
automatically translating physical specifications to backend-specific implementations.
"""

from .specs import (
    CouplingScheme,
    RelativisticTreatment,
    OptimizationTarget,
    CalculationType,
    RadiationType,
)
from .calculation import AtomicCalculation
from .backends import AutostructureBackend, FACBackend

__all__ = [
    "AtomicCalculation",
    "CouplingScheme",
    "RelativisticTreatment",
    "OptimizationTarget",
    "CalculationType",
    "RadiationType",
    "AutostructureBackend",
    "FACBackend",
]
