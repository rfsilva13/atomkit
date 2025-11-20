import re


def fac_label_to_siegbahn(fac_label: str) -> str:
    """
    Convert a raw FAC label (e.g., '4d+6(0)0', '2p-1(1)1.3d+5(5)6') to Siegbahn notation (e.g., 'L2', 'L2 M5').
    Handles multi-shell (dot-separated) FAC labels.

    Args:
        fac_label: FAC shell label with j-splitting notation, may contain multiple shells separated by dots

    Returns:
        Siegbahn notation (e.g., 'K', 'L1', 'L2', 'L3', 'M1', etc.)
        For multi-shell labels, returns space-separated Siegbahn labels
    """
    if not fac_label or not isinstance(fac_label, str):
        return ""
    # Handle multi-shell labels (double-hole states) separated by dots
    if '.' in fac_label:
        return ' '.join(fac_label_to_siegbahn(part.strip()) for part in fac_label.split('.'))
    # Extract the shell part (e.g., '4d+' from '4d+6(0)0')
    shell_match = re.match(r'(\d+)([spdfghik]+)([+-]?)', fac_label.strip())
    if shell_match:
        n_str, l_symbol, sign = shell_match.groups()
        n = int(n_str)
        n_to_letter = {1: "K", 2: "L", 3: "M", 4: "N", 5: "O", 6: "P", 7: "Q"}
        if n not in n_to_letter:
            return ""
        principal_letter = n_to_letter[n]
        l_symbols = ["s", "p", "d", "f", "g", "h", "i", "k"]
        if l_symbol not in l_symbols:
            return ""
        l_quantum = l_symbols.index(l_symbol)
        if n == 1:
            if l_symbol == "s":
                return "K"
            return ""
        current_subshell_index = 1
        for curr_l in range(n):
            if curr_l == 0:
                if l_quantum == 0:
                    return f"{principal_letter}{current_subshell_index}"
                current_subshell_index += 1
            else:
                if curr_l == l_quantum:
                    if sign == "-":
                        return f"{principal_letter}{current_subshell_index}"
                    elif sign == "+":
                        return f"{principal_letter}{current_subshell_index + 1}"
                    else:
                        return f"{principal_letter}{current_subshell_index}{current_subshell_index + 1}"
                current_subshell_index += 2
    return ""
def XRlabel(atomic_number, conf1, configuration, label):
    """
    Wrapper for Siegbahn labeling to match the notebook's XRlabel signature.
    This function ignores atomic_number, conf1, and label for compatibility,
    and only uses the configuration string, as in the backend logic.

    Args:
        atomic_number: (int) Atomic number (ignored)
        conf1: (str) Not used, for compatibility
        configuration: (str) Configuration string (FAC/AS format)
        label: (str) Not used, for compatibility

    Returns:
        Siegbahn label string (e.g., 'L3', 'L2 M4', etc.)
    """
    # Try config_to_siegbahn first (for configuration strings)
    try:
        label_str = config_to_siegbahn(configuration, holes_only=True, compact=True)
        if label_str:
            return label_str
    except Exception:
        pass
    # Fallback: try fac_label_to_siegbahn on the label (raw FAC label)
    try:
        fac_label_str = fac_label_to_siegbahn(label)
        if fac_label_str:
            return fac_label_str
    except Exception:
        pass
    # If all else fails, return empty string
    return ""
# atomkit/src/atomkit/readers/siegbahn.py

"""
Utilities for Siegbahn notation (X-ray spectroscopy labels).

Siegbahn notation is the standard labeling system for X-ray transitions,
using symbols like K, L1, L2, L3, M1-M5, etc. to denote electronic shells
based on their quantum numbers.

This module provides functions to:
- Convert Configuration objects to Siegbahn labels
- Convert shell strings (e.g., "2p-") to Siegbahn notation (e.g., "L2")
- Classify X-ray transitions as diagram or satellite lines
- Generate compact transition labels (e.g., "L3-M5")
"""

import logging
from typing import Optional, Union

import pandas as pd

from ..constants import SHELL_LABEL_MAP
from ..structure import Configuration

logger = logging.getLogger(__name__)


def shell_to_siegbahn(shell_str: str) -> str:
    """
    Convert a shell string to Siegbahn notation.

    Args:
        shell_str: Shell string like "1s", "2p-", "2p+", "3d-", etc.

    Returns:
        Siegbahn label (e.g., "K", "L2", "L3", "M4", "M5").

    Raises:
        ValueError: If the shell string is not recognized.

    Examples:
        >>> shell_to_siegbahn("1s")
        'K'
        >>> shell_to_siegbahn("2p-")
        'L2'
        >>> shell_to_siegbahn("2p+")
        'L3'
        >>> shell_to_siegbahn("3d-")
        'M4'
    """
    if not isinstance(shell_str, str):
        raise TypeError(f"shell_str must be a string, got {type(shell_str)}")

    shell_str = shell_str.strip()
    if not shell_str:
        raise ValueError("shell_str cannot be empty")

    label = SHELL_LABEL_MAP.get(shell_str)
    if label is None:
        raise ValueError(
            f"Unrecognized shell string '{shell_str}'. "
            f"Must be in format like '1s', '2p-', '2p+', '3d-', etc."
        )

    return label


def config_to_siegbahn(
    config: Union[Configuration, str],
    holes_only: bool = True,
    compact: bool = True,
) -> str:
    """
    Convert a Configuration to Siegbahn notation, optionally showing only holes.

    This function is particularly useful for X-ray spectroscopy where we typically
    track core hole positions rather than full electronic configurations.

    Args:
        config: Configuration object or configuration string.
        holes_only: If True, only show subshells with holes (occupation < max).
                   If False, show all subshells. Default True.
        compact: If True, use compact notation (e.g., "L2" for single hole).
                If False, include occupation numbers (e.g., "L2[5]" for one hole in 2p-).
                Default True.

    Returns:
        Siegbahn label string. Multiple holes separated by spaces (e.g., "L2 M4").
        Returns empty string if no holes found (when holes_only=True).

    Examples:
        >>> from atomkit import Configuration
        >>> # Single core hole in 2p- (one electron missing)
        >>> config = Configuration("[Ne] 2p-[5] 2p+[6] 3s[2]")
        >>> config_to_siegbahn(config, holes_only=True)
        'L2'

        >>> # Double hole: one in 2p-, one in 3d-
        >>> config = Configuration("[Ne] 2p-[5] 2p+[6] 3s[2] 3p-[2] 3p+[4] 3d-[3] 3d+[4]")
        >>> config_to_siegbahn(config, holes_only=True)
        'L2 M4'

        >>> # Show with occupation numbers
        >>> config_to_siegbahn(config, holes_only=True, compact=False)
        'L2[5] M4[3]'
    """
    # Handle string input
    if isinstance(config, str):
        config = Configuration.from_string(config)
    elif not isinstance(config, Configuration):
        raise TypeError(f"config must be Configuration or str, got {type(config)}")

    siegbahn_parts = []

    for subshell in config.shells:
        n = subshell.n
        l = subshell.l_quantum
        j = subshell.j_quantum
        occ = subshell.occupation
        max_occ = subshell.max_occupation()

        # Construct shell string for lookup
        l_symbol = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i"}.get(l)
        if l_symbol is None:
            logger.warning(f"Unsupported l quantum number: {l}")
            continue

        # Build shell string with j-splitting if applicable
        if l == 0:
            shell_str = f"{n}{l_symbol}"
        else:
            # Determine j-splitting: j = l ± 1/2
            if j is None:
                logger.warning(f"Shell {n}{l_symbol} has no j quantum number, skipping")
                continue
            j_minus = l - 0.5
            j_plus = l + 0.5
            if abs(j - j_minus) < 1e-6:
                shell_str = f"{n}{l_symbol}-"
            elif abs(j - j_plus) < 1e-6:
                shell_str = f"{n}{l_symbol}+"
            else:
                logger.warning(f"Unexpected j value {j} for l={l}")
                continue

        # Check if we should include this subshell
        is_hole = occ < max_occ
        if holes_only and not is_hole:
            continue

        # Convert to Siegbahn notation
        try:
            siegbahn = shell_to_siegbahn(shell_str)
        except ValueError as e:
            logger.warning(f"Could not convert shell {shell_str}: {e}")
            continue

        # Format the label
        if compact:
            siegbahn_parts.append(siegbahn)
        else:
            siegbahn_parts.append(f"{siegbahn}[{occ}]")

    return " ".join(siegbahn_parts)


def compact_siegbahn_label(
    lower_label: str,
    upper_label: str,
    separator: str = "-",
) -> str:
    """
    Create compact transition notation from Siegbahn labels.

    Args:
        lower_label: Siegbahn label for lower level (e.g., "L3").
        upper_label: Siegbahn label for upper level (e.g., "M5").
        separator: String to separate labels. Default "-".

    Returns:
        Compact transition label (e.g., "L3-M5").

    Examples:
        >>> compact_siegbahn_label("L3", "M5")
        'L3-M5'
        >>> compact_siegbahn_label("L2 M4", "M4 M5")
        'L2 M4-M4 M5'
        >>> compact_siegbahn_label("K", "L1", separator="→")
        'K→L1'
    """
    return f"{lower_label}{separator}{upper_label}"


def classify_transitions(
    transitions_df: pd.DataFrame,
    primary_label: str,
    satellite_labels: list[str],
    label_column: str = "label",
    lower_suffix: str = "_lower",
    upper_suffix: str = "_upper",
) -> pd.DataFrame:
    """
    Classify X-ray transitions as diagram or satellite lines.

    This function categorizes transitions based on the number of spectator holes:
    - **Diagram lines**: Single-hole transitions (e.g., L3 → M5)
    - **Satellite lines**: Double-hole transitions with one spectator hole
      (e.g., L3 M5 → M4 M5, where M5 is the spectator)

    The classification helps distinguish between main X-ray lines and their
    satellites, which are important for plasma diagnostics and spectroscopy.

    Args:
        transitions_df: DataFrame containing transition data with Siegbahn labels.
        primary_label: The primary shell label to track (e.g., "L3" for L3-edge).
        satellite_labels: List of allowed satellite shell labels (e.g., ["M4", "M5"]).
        label_column: Base name for the label columns (default "label").
                     Will look for "{label_column}{lower_suffix}" and
                     "{label_column}{upper_suffix}" columns.
        lower_suffix: Suffix for lower level column (default "_lower").
        upper_suffix: Suffix for upper level column (default "_upper").

    Returns:
        Filtered DataFrame containing only classified transitions, with an added
        'line_type' column. Values are either:
        - "Diagram": Single-hole transition
        - "Satellite {X}": Double-hole transition with spectator in shell X

    Examples:
        >>> import pandas as pd
        >>> # Suppose we have transitions with Siegbahn labels
        >>> trans = pd.DataFrame({
        ...     'label_lower': ['L3', 'L3', 'L3 M5', 'L3 M4'],
        ...     'label_upper': ['M4', 'M5', 'M4 M5', 'M4 M5'],
        ...     'energy': [530.0, 532.0, 520.0, 525.0],
        ...     'rate': [1e13, 8e12, 5e11, 3e11]
        ... })
        >>> classified = classify_transitions(
        ...     trans,
        ...     primary_label='L3',
        ...     satellite_labels=['M4', 'M5']
        ... )
        >>> classified['line_type'].tolist()
        ['Diagram', 'Diagram', 'Satellite M5', 'Satellite M5']

    Notes:
        - Transitions with shells beyond the maximum satellite shell are filtered out
        - Only transitions involving the primary_label are kept
        - Unclassified transitions are dropped from the output
    """
    # Input validation
    if not isinstance(satellite_labels, (list, tuple)) or len(satellite_labels) == 0:
        raise ValueError("satellite_labels must be a non-empty list or tuple")
    if not isinstance(primary_label, str) or not primary_label:
        raise ValueError("primary_label must be a non-empty string")

    # Construct column names
    lower_col = f"{label_column}{lower_suffix}"
    upper_col = f"{label_column}{upper_suffix}"

    # Check required columns exist
    if lower_col not in transitions_df.columns:
        raise ValueError(f"Column '{lower_col}' not found in transitions_df")
    if upper_col not in transitions_df.columns:
        raise ValueError(f"Column '{upper_col}' not found in transitions_df")

    # Determine max shell from satellite labels (K < L < M < N < O < P < Q)
    shell_order = ["K", "L", "M", "N", "O", "P", "Q"]
    max_shell = max(label[0] for label in satellite_labels)
    max_shell_idx = shell_order.index(max_shell)
    allowed_shells = set(shell_order[: max_shell_idx + 1])

    # Create working copy
    df = transitions_df.copy()
    df["line_type"] = pd.NA

    def check_shells(label):
        """Check if all shells in a label are allowed."""
        if pd.isna(label) or label == "":
            return False
        for part in str(label).split():
            if part[0] not in allowed_shells:
                return False
        return True

    # Filter: only keep transitions with allowed shells
    mask_allowed = df[lower_col].apply(check_shells) & df[upper_col].apply(check_shells)
    df = df[mask_allowed].copy()

    if df.empty:
        logger.warning("No transitions remaining after shell filtering")
        return df

    # Pre-compute single-label checks
    lower_single = ~df[lower_col].astype(str).str.contains(" ")
    upper_single = ~df[upper_col].astype(str).str.contains(" ")

    # Pattern 1: "Diagram" - Single hole transitions
    # Either: primary_label (lower) → satellite (upper)
    # Or:     satellite (lower) → primary_label (upper)
    satellite_pattern = "|".join(satellite_labels)
    diagram_mask = (
        (df[lower_col].str.contains(primary_label, regex=False) & lower_single)
        & (df[upper_col].str.contains(satellite_pattern) & upper_single)
    ) | (
        (df[upper_col].str.contains(primary_label, regex=False) & upper_single)
        & (df[lower_col].str.contains(satellite_pattern) & lower_single)
    )
    df.loc[diagram_mask, "line_type"] = "Diagram"

    # Pattern 2: "Satellite X" - Double-hole transitions with spectator
    # Example: "L3 M5" → "M4 M5" is a satellite with M5 spectator
    unclassified_mask = df["line_type"].isna()

    if unclassified_mask.any():
        # Pre-compute ordered parts for double-hole labels
        df.loc[unclassified_mask, "lower_ordered"] = df.loc[
            unclassified_mask, lower_col
        ].apply(lambda x: " ".join(sorted(str(x).split())))
        df.loc[unclassified_mask, "upper_ordered"] = df.loc[
            unclassified_mask, upper_col
        ].apply(lambda x: " ".join(sorted(str(x).split())))

        # Check each satellite label as potential spectator
        for sat_label in satellite_labels:

            def is_satellite(row):
                if pd.notna(row["line_type"]):
                    return False

                lower_parts = set(row["lower_ordered"].split())
                upper_parts = set(row["upper_ordered"].split())

                # Check if transition involves primary and this satellite
                has_primary_satellite = (
                    primary_label in lower_parts and sat_label in upper_parts
                ) or (primary_label in upper_parts and sat_label in lower_parts)

                if not has_primary_satellite:
                    return False

                # Check if there's exactly one common spectator hole
                lower_set = lower_parts - {primary_label, sat_label}
                upper_set = upper_parts - {primary_label, sat_label}
                return len(lower_set) == 1 and lower_set == upper_set

            mask_satellite = df[unclassified_mask].apply(is_satellite, axis=1)
            df.loc[mask_satellite[mask_satellite].index, "line_type"] = (
                f"Satellite {sat_label}"
            )

        # Clean up temporary columns
        df = df.drop(["lower_ordered", "upper_ordered"], axis=1, errors="ignore")

    # Drop unclassified transitions
    df = df.dropna(subset=["line_type"])

    if df.empty:
        logger.warning(
            f"No transitions classified for primary={primary_label}, satellites={satellite_labels}"
        )

    return df


def add_siegbahn_labels(
    transitions_df: pd.DataFrame,
    levels_df: pd.DataFrame,
    config_column: str = "configuration",
    holes_only: bool = True,
    compact: bool = True,
) -> pd.DataFrame:
    """
    Add Siegbahn notation labels to transitions DataFrame.

    This is a convenience function that:
    1. Extracts configurations for lower and upper levels
    2. Converts them to Siegbahn notation
    3. Adds new columns with Siegbahn labels

    Args:
        transitions_df: DataFrame with transition data. Must have:
                       'atomic_number', 'level_index_lower', 'level_index_upper'.
        levels_df: DataFrame with level data. Must have:
                  'atomic_number', 'level_index', and config_column.
        config_column: Name of the configuration column in levels_df.
        holes_only: If True, only show holes in Siegbahn labels.
        compact: If True, use compact notation without occupation numbers.

    Returns:
        New DataFrame with added columns:
        - 'siegbahn_lower': Siegbahn label for lower level
        - 'siegbahn_upper': Siegbahn label for upper level
        - 'siegbahn_transition': Compact transition notation (e.g., "L3-M5")

    Raises:
        ValueError: If required columns are missing.

    Examples:
        >>> # After reading FAC data
        >>> levels = read_fac("fe.lev.asc")
        >>> transitions = read_fac_transitions("fe.tr.asc")
        >>> trans_labeled = add_siegbahn_labels(transitions, levels)
        >>> trans_labeled[['siegbahn_lower', 'siegbahn_upper', 'energy']].head()
    """
    from .labeling import add_level_info_to_transitions

    # First, add configuration info to transitions
    trans_with_config = add_level_info_to_transitions(
        transitions_df,
        levels_df,
        cols_to_add=[config_column],
        missing_value="<N/A>",
    )

    # Check if configurations were successfully added
    lower_config_col = f"{config_column}_lower"
    upper_config_col = f"{config_column}_upper"

    if lower_config_col not in trans_with_config.columns:
        raise ValueError(
            f"Could not add configuration info. Check that levels_df has '{config_column}' column."
        )

    # Convert configurations to Siegbahn notation

    def safe_siegbahn_label(config_str, fac_label):
        # Try config_to_siegbahn first (more reliable for hole detection)
        if pd.isna(config_str) or config_str == "<N/A>" or not config_str:
            config_str = None
        if config_str:
            try:
                label = config_to_siegbahn(config_str, holes_only=holes_only, compact=compact)
                if label:
                    return label
            except Exception as e:
                logger.warning(f"Could not convert config '{config_str}': {e}")
        # Fallback: try fac_label_to_siegbahn
        if fac_label and isinstance(fac_label, str):
            try:
                label = fac_label_to_siegbahn(fac_label)
                if label:
                    return label
            except Exception as e:
                logger.warning(f"Could not convert FAC label '{fac_label}': {e}")
        return ""

    # Try to get the label columns from levels_df
    lower_label_col = "label_lower" if "label_lower" in trans_with_config.columns else None
    upper_label_col = "label_upper" if "label_upper" in trans_with_config.columns else None

    trans_with_config["siegbahn_lower"] = trans_with_config.apply(
        lambda row: safe_siegbahn_label(
            row[lower_config_col], row[lower_label_col] if lower_label_col else None
        ), axis=1
    )
    trans_with_config["siegbahn_upper"] = trans_with_config.apply(
        lambda row: safe_siegbahn_label(
            row[upper_config_col], row[upper_label_col] if upper_label_col else None
        ), axis=1
    )

    # Create compact transition labels
    trans_with_config["siegbahn_transition"] = trans_with_config.apply(
        lambda row: compact_siegbahn_label(
            row["siegbahn_lower"], row["siegbahn_upper"]
        ),
        axis=1,
    )

    return trans_with_config
