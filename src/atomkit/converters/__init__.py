"""
Converters module for atomkit.

Provides conversion utilities between different atomic structure code formats,
including FAC (Flexible Atomic Code) and AUTOSTRUCTURE.
"""

from .fac_to_as import (
    convert_fac_to_as,
    parse_orbital,
    extract_configurations,
    get_unique_orbitals,
    build_occupation_matrix,
    print_as_format,
    write_as_format,
    ORBITAL_MAP,
)

from .ls_to_icr import (
    convert_ls_to_icr,
    create_icr_input,
    run_autostructure_ls,
    run_autostructure_icr,
    ls_to_icr_full_workflow,
)

from .as_generator import (
    configurations_to_autostructure,
    generate_as_configurations,
    format_as_input,
)

__all__ = [
    # FAC to AS converter
    "convert_fac_to_as",
    "parse_orbital",
    "extract_configurations",
    "get_unique_orbitals",
    "build_occupation_matrix",
    "print_as_format",
    "write_as_format",
    "ORBITAL_MAP",
    # LS to ICR converter
    "convert_ls_to_icr",
    "create_icr_input",
    "run_autostructure_ls",
    "run_autostructure_icr",
    "ls_to_icr_full_workflow",
    # AS configuration generator
    "configurations_to_autostructure",
    "generate_as_configurations",  # Deprecated, use configurations_to_autostructure
    "format_as_input",
]
