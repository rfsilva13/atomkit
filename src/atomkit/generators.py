# atomkit/src/atomkit/generators.py

"""
Functions for programmatically generating new atomic configurations.
"""

import re
import itertools
from typing import List, Optional, Set

from .configuration import Configuration
from .shell import Shell
from .definitions import L_SYMBOLS


def _expand_fac_shorthand(config_str: str) -> List[str]:
    """
    Expands a single FAC configuration string containing '*' shorthand into a list
    of fully specified configuration strings.

    Example: "1s2 3*1" -> ["1s2 3s1", "1s2 3p1", "1s2 3d1"]
    """
    match = re.search(r"(\d+)\*(\d+)", config_str)

    # Base case: no shorthand found, return the original string in a list
    if not match:
        return [config_str]

    n_str, e_str = match.groups()
    n, e = int(n_str), int(e_str)

    # This simple expansion assumes e=1, which is the most common use case.
    # A more complex version could handle e>1 with combinations.
    if e != 1:
        raise NotImplementedError(
            f"Shorthand expansion for {e} electrons is not implemented. Only 'n*1' is supported."
        )

    base_config = re.sub(r"\s*\d+\*\d+", "", config_str).strip()

    expanded_configs = []
    # Iterate l from 0 (s) up to n-1
    for l_quantum in range(n):
        l_symbol = L_SYMBOLS[l_quantum]
        new_shell_str = f"{n}{l_symbol}{e}"

        # Combine base and new shell, ensuring no leading/trailing dots
        if base_config:
            new_config = f"{base_config}.{new_shell_str}"
        else:
            new_config = new_shell_str
        expanded_configs.append(new_config)

    return expanded_configs


def generate_recombined_configs(
    target_configs: List[str], max_n: int, max_l: int
) -> List[str]:
    """
    Generates a list of (N+1)-electron autoionizing configurations by adding
    one electron to a list of N-electron target configurations.

    This function understands and expands FAC shorthand notation (e.g., "3*1").
    It's a convenience wrapper around Configuration.generate_recombined_configurations()
    that handles multiple input configurations and FAC shorthand notation.

    Args:
        target_configs: A list of N-electron configuration strings, which can
                        include FAC shorthand like "1s2 3*1".
        max_n: The maximum principal quantum number of the shell to add the
               electron to.
        max_l: The maximum orbital angular momentum of the shell to add the
               electron to.

    Returns:
        A sorted list of unique, (N+1)-electron configuration strings.

    Example:
        >>> configs = ["1s2.2s2", "1s2.2s1.2p1"]
        >>> recombined = generate_recombined_configs(configs, max_n=3, max_l=2)
        >>> # Returns all unique configurations with one electron added
    """

    # First, fully expand all shorthand notations from the input list
    expanded_target_configs = []
    for conf_str in target_configs:
        expanded_target_configs.extend(_expand_fac_shorthand(conf_str))

    # Use a set to store unique final configurations
    final_configs_set: Set[Configuration] = set()

    # Process each fully specified target configuration
    for conf_str in expanded_target_configs:
        base_config = Configuration.from_string(conf_str)

        # Use the Configuration class method to generate recombined configs
        recombined_list = base_config.generate_recombined_configurations(
            max_n=max_n, max_l=max_l
        )

        # Add all generated configurations to the set
        final_configs_set.update(recombined_list)

    # Convert the set of Configuration objects back to sorted strings for the output
    final_config_strings = sorted([str(c) for c in final_configs_set])

    return final_config_strings
