"""
AUTOSTRUCTURE wrapper module for atomkit.

This module provides a Python interface for generating AUTOSTRUCTURE input files
(.dat format) without requiring manual NAMELIST formatting.

Key Features
------------
- Pythonic interface to AUTOSTRUCTURE namelists
- Automatic conversion from atomkit Configuration objects
- Context manager support for clean file handling
- Support for all major AUTOSTRUCTURE calculation types
- Helper classes for common configurations (Phase 6 UX enhancements)
- High-level presets for typical calculation types
- Fluent interface for method chaining
- Validation methods to catch errors early

Example Usage
-------------
Basic usage:
>>> from atomkit.autostructure import ASWriter
>>> from atomkit import Configuration
>>>
>>> ground = Configuration.from_element("C", charge=2)  # Be-like C
>>> with ASWriter("c_belike.dat") as asw:
...     asw.write_header("Be-like Carbon structure")
...     asw.add_salgeb(CUP='IC', RAD='E1')
...     asw.configs_from_atomkit([ground], last_core_orbital='1s')
...     asw.add_sminim(NZION=6)

Using presets (Phase 6):
>>> from atomkit.autostructure import ASWriter, CoreSpecification
>>> asw = ASWriter.for_structure_calculation(
...     "ne_structure.dat",
...     nzion=10,
...     coupling="IC",
...     core=CoreSpecification.helium_like()
... )

Classes
-------
ASWriter : Main class for generating AUTOSTRUCTURE input files

Helper Classes (Phase 6)
-------------------------
CoreSpecification : Simplified core specification (He-like, Ne-like, etc.)
SymmetryRestriction : Helper for symmetry restrictions (terms/levels)
EnergyShifts : Energy shift specifications
CollisionParams : Collision calculation parameters
OptimizationParams : Orbital optimization parameters
RydbergSeries : Rydberg series specification for DR/RR
"""

from .as_writer import (
    ASWriter,
    CoreSpecification,
    SymmetryRestriction,
    EnergyShifts,
    CollisionParams,
    OptimizationParams,
    RydbergSeries,
)

__all__ = [
    "ASWriter",
    "CoreSpecification",
    "SymmetryRestriction",
    "EnergyShifts",
    "CollisionParams",
    "OptimizationParams",
    "RydbergSeries",
]
