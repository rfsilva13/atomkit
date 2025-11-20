# atomkit/src/atomkit/readers/__init__.py

"""
Initializes the readers sub-package and makes public functions available.
"""

from .autoionization import read_fac_autoionization
from .autostructure import (detect_file_format, get_autoionization, get_levels,
                            get_terms, get_transitions, read_as_lambdas,
                            read_as_levels, read_as_terms, read_as_transitions)
from .labeling import add_level_info_to_transitions
from .levels import read_fac
from .siegbahn import (add_siegbahn_labels, classify_transitions,
                       compact_siegbahn_label, config_to_siegbahn,
                       fac_label_to_siegbahn, shell_to_siegbahn)
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
    "read_as_terms",
    "detect_file_format",
    "get_levels",
    "get_transitions",
    "get_terms",
    "get_autoionization",
    "shell_to_siegbahn",
    "config_to_siegbahn",
    "fac_label_to_siegbahn",
    "compact_siegbahn_label",
    "classify_transitions",
    "add_siegbahn_labels",
]
