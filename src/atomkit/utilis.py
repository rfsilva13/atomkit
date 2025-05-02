# atomkit/src/atomkit/utils.py

"""
General utility functions for the atomkit package.
"""

from typing import Union

# Use absolute import based on package structure
from .configuration import Configuration

try:
    import mendeleev

    MENDELEEV_AVAILABLE = True
except ImportError:
    MENDELEEV_AVAILABLE = False


def get_ionstage(config: Configuration, element_identifier: Union[int, str]) -> int:
    """
    Calculates the ion stage (charge) of a given configuration relative
    to the neutral atom of the specified element.

    Ion Stage = Z - Ne
    where Z is the atomic number and Ne is the number of electrons in the config.

    Args:
        config: The Configuration object.
        element_identifier: The element's atomic number (int), symbol (str),
                            or name (str). Used to determine Z.

    Returns:
        The calculated ion stage (integer).

    Raises:
        ValueError: If the element identifier is invalid.
        ImportError: If the mendeleev package is not installed.
    """
    if not MENDELEEV_AVAILABLE:
        raise ImportError(
            "The 'mendeleev' package is required for get_ionstage function."
        )

    try:
        element = mendeleev.element(element_identifier)
        atomic_number = element.atomic_number
    except Exception as e:
        raise ValueError(f"Invalid element identifier '{element_identifier}': {e}")

    num_electrons = config.total_electrons()
    ion_stage = atomic_number - num_electrons
    return ion_stage


# You could also move the data_optimize function from auxiliary.py here later if desired.
# import numpy as np
# import pandas as pd
# def data_optimize(df, object_option=False): ...
