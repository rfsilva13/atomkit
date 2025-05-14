# atomkit/src/atomkit/readers/__init__.py

"""
Initializes the readers sub-package and makes public functions available.
"""

from .labeling import add_level_info_to_transitions

# Import the public reader function(s) from their respective modules
from .levels import read_fac
from .transitions import read_fac_transitions

# Define what gets imported when someone does 'from atomkit.readers import *'
# This makes the functions directly accessible.
__all__ = ["read_fac", "read_fac_transitions", "add_level_info_to_transitions"]
