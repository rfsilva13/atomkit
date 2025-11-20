# src/atomkit/__init__.py

"""
AtomKit: A Python toolkit for atomic structure and spectra calculations.

Package Structure
-----------------
- structure: Fundamental atomic structure classes (Shell, Configuration)
- definitions: Physical constants and atomic structure definitions
- utils: General utility functions
- analysis: Spectral analysis and data processing
- readers: File format readers for various codes
- converters: Format conversion utilities
- physics: Cross-sections, potentials, and physical calculations
- fac: FAC (Flexible Atomic Code) interface
- autostructure: AUTOSTRUCTURE interface
- core: Unified calculation interface
"""

__version__ = "0.1.0"

# Import submodules
from . import (analysis, autostructure, constants, converters, core, fac,
               physics, readers, structure, utils)
# Import main unified class
from .core import AtomicCalculation
from .structure import Configuration, Shell
from .utils import get_element_info, parse_ion_notation

__all__ = [
    # Core classes
    "Shell",
    "Configuration",
    # Utility functions
    "get_element_info",
    "parse_ion_notation",
    # Submodules
    "structure",
    "constants",
    "utils",
    "analysis",
    "physics",
    "fac",
    "autostructure",
    "core",
    "readers",
    "converters",
    # Main interface
    "AtomicCalculation",
]
