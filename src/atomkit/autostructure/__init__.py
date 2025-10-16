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

Example Usage
-------------
>>> from atomkit.autostructure import ASWriter
>>> from atomkit import Configuration
>>>
>>> ground = Configuration.from_element("C", charge=2)  # Be-like C
>>> with ASWriter("c_belike.dat") as asw:
...     asw.write_header("Be-like Carbon structure")
...     asw.add_salgeb(CUP='IC', RAD='E1')
...     asw.configs_from_atomkit([ground], last_core_orbital='1s')
...     asw.add_sminim(NZION=6)

Classes
-------
ASWriter : Main class for generating AUTOSTRUCTURE input files
"""

from .as_writer import ASWriter

__all__ = ["ASWriter"]
