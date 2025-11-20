"""
AtomKit structure module.

Fundamental atomic structure classes for shells and configurations.
"""

from .configuration import Configuration
from .shell import Shell

__all__ = [
    "Shell",
    "Configuration",
]
