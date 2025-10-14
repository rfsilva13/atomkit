# src/atomkit/__init__.py

"""
AtomKit: A Python toolkit for atomic structure and spectra calculations.
"""

from .shell import Shell
from .configuration import Configuration
from .generators import generate_recombined_configs

# Import physics submodule for cross section calculations
from . import physics

__all__ = [
    "Shell",
    "Configuration",
    "generate_recombined_configs",
    "physics",
]
