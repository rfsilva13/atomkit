#!/usr/bin/env python3
"""
AUTOSTRUCTURE Workflow - Code-Agnostic Examples

Demonstrates the atomkit philosophy:
- Generate configurations using general Configuration methods (physics)
- Format for AUTOSTRUCTURE only at the final I/O step

Configuration generation is completely code-agnostic.
"""

from pathlib import Path
from atomkit import Configuration
from atomkit.converters import configurations_to_autostructure


def example_1_basic():
    """Example 1: Basic code-agnostic workflow."""
    print("=" * 70)
    print("Example 1: Code-Agnostic Workflow")
    print("=" * 70)

    # Physics: Generate configurations
    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d6 4s2")
    excited = ground.generate_excitations(
        target_shells=["4p", "5s"], excitation_level=1, source_shells=["3d", "4s"]
    )

    # I/O: Format for AUTOSTRUCTURE
    result = configurations_to_autostructure(
        [ground] + excited[:5], last_core_orbital="3p"
    )
    print(f"Generated {result['mxconf']} configurations")


def example_2_core_valence():
    """Example 2: Core + valence format."""
    print("\n" + "=" * 70)
    print("Example 2: Core + Valence")
    print("=" * 70)

    core = Configuration.from_string("1s2 2s2 2p6 3s2 3p6")
    valence = [
        Configuration.from_string("3d6 4s2"),
        Configuration.from_string("3d6 4s1 4p1"),
    ]

    result = configurations_to_autostructure(valence, core=core, last_core_orbital="3p")
    print(f"Generated {result['mxconf']} configurations")
    for cfg in result["configurations"]:
        print(f"  {cfg}")


if __name__ == "__main__":
    print("AUTOSTRUCTURE Workflow - Code-Agnostic Examples")
    print("=" * 70)
    example_1_basic()
    example_2_core_valence()
    print("\n✅ Examples complete!")
