"""
FAC (Flexible Atomic Code) Integration Module.

This module provides a Python wrapper for generating FAC input files (.sf format).
Instead of requiring the pfac Python bindings, this module allows you to generate
SFAC (Simple FAC) input files programmatically using modern Python syntax.

The wrapper translates Python method calls into SFAC commands that can be written
to .sf files and executed by the FAC command-line tools (sfac, scrm, spol).

Example:
    >>> from atomkit.fac import SFACWriter
    >>>
    >>> with SFACWriter("fe_calculation.sf") as fac:
    ...     fac.SetAtom("Fe")
    ...     fac.Closed("1s")
    ...     fac.Config("2*8", group="n2")
    ...     fac.Config("2*7 3*1", group="n3")
    ...     fac.ConfigEnergy(0)
    ...     fac.OptimizeRadial(["n2"])
    ...     fac.ConfigEnergy(1)
    ...     fac.Structure("ne.lev.b", ["n2", "n3"])
    ...     fac.PrintTable("ne.lev.b", "ne.lev", 1)
"""

from .sfac_writer import SFACWriter

__all__ = ["SFACWriter"]
