# src/atomkit/__init__.py

"""
AtomKit: A Python toolkit for atomic structure and spectra calculations.
"""

from .shell import Shell
from .configuration import Configuration
from .utils import get_element_info, parse_ion_notation

# Import physics submodule for cross section calculations
from . import physics

__all__ = [
    "Shell",
    "Configuration",
    "get_element_info",
    "parse_ion_notation",
    "physics",
]
