# atomkit/src/atomkit/readers/__init__.py

"""
Initializes the readers sub-package and makes public functions available.
"""

from .autoionization import read_fac_autoionization
from .autostructure import read_as_levels, read_as_transitions, read_as_lambdas
from .labeling import add_level_info_to_transitions
from .levels import read_fac
from .transitions import read_fac_transitions

# Define what gets imported when someone does 'from atomkit.readers import *'
# This makes the functions directly accessible.
__all__ = [
    "read_fac",
    "read_fac_transitions",
    "add_level_info_to_transitions",
    "read_fac_autoionization",
    "read_as_levels",
    "read_as_transitions",
    "read_as_lambdas",
]
