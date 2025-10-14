#!/usr/bin/env python3
"""
Basic Usage Example for AtomKit

This example demonstrates the fundamental features of AtomKit:
- Creating Shell objects
- Bui    # Convert to different separators
    config2 = Configuration.from_string("1s2.2s2.2p6.3s2.3p6.3d10")
    space_sep = config2.to_string(separator=" ")
    compact = config2.to_string(separator="")
    print(f"\nDifferent separators for {config2}:")
    print(f"  Space-separated: {space_sep}")
    print(f"  Compact: {compact}")
    print()Configuration objects
- Parsing configurations from strings
- Basic configuration properties
"""

from atomkit import Configuration, Shell


def shell_basics():
    """Demonstrate Shell class basics."""
    print("=" * 60)
    print("Shell Class Basics")
    print("=" * 60)

    # Create shells
    shell_1s = Shell(n=1, l_quantum=0, occupation=2)
    shell_2p_minus = Shell(
        n=2, l_quantum=1, j_quantum=0.5, occupation=2
    )  # 2p- has j=l-1/2
    shell_2p_plus = Shell(
        n=2, l_quantum=1, j_quantum=1.5, occupation=4
    )  # 2p+ has j=l+1/2

    print(f"1s shell: {shell_1s}")
    print(f"2p- shell: {shell_2p_minus}")
    print(f"2p+ shell: {shell_2p_plus}")

    # Shell properties
    print(f"\n1s shell properties:")
    print(f"  String representation: {str(shell_1s)}")
    print(f"  Max occupation: {shell_1s.max_occupation()}")
    print(f"  Is full: {shell_1s.is_full()}")
    print()


def configuration_from_shells():
    """Create configurations from Shell objects."""
    print("=" * 60)
    print("Configuration from Shells")
    print("=" * 60)

    # Create shells
    shells = [
        Shell(n=1, l_quantum=0, occupation=2),
        Shell(n=2, l_quantum=0, occupation=2),
        Shell(n=2, l_quantum=1, occupation=6),
    ]

    # Create configuration
    config = Configuration(shells)

    print(f"Configuration: {config}")
    print(f"Total electrons: {config.total_electrons()}")
    print(f"Number of shells: {len(config)}")
    print()


def configuration_from_string():
    """Parse configurations from string notation."""
    print("=" * 60)
    print("Configuration from String")
    print("=" * 60)

    # Standard notation
    config1 = Configuration.from_string("1s2.2s2.2p6")
    print(f"Neon ground state: {config1}")
    print(f"Total electrons: {config1.total_electrons()}")

    # With j-quantum numbers
    config2 = Configuration.from_string("1s2.2s2.2p-2.2p+4")
    print(f"\nWith j-quantum numbers: {config2}")

    # Excited state
    config3 = Configuration.from_string("1s2.2s1.2p6.3s1")
    print(f"\nExcited state: {config3}")
    print(f"Total electrons: {config3.total_electrons()}")
    print()


def configuration_from_element():
    """Create configurations from element symbols."""
    print("=" * 60)
    print("Configuration from Element")
    print("=" * 60)

    # Neutral atoms
    neon = Configuration.from_element("Ne")
    print(f"Neon (Z=10): {neon}")

    argon = Configuration.from_element("Ar")
    print(f"Argon (Z=18): {argon}")

    # Ions
    ne_plus = Configuration.from_element("Ne", ion_charge=1)
    print(f"\nNe+ ion: {ne_plus}")
    print(f"Total electrons: {ne_plus.total_electrons()}")

    fe_24 = Configuration.from_element("Fe", ion_charge=24)
    print(f"\nFe24+ ion: {fe_24}")
    print(f"Total electrons: {fe_24.total_electrons()}")
    print()


def configuration_comparison():
    """Compare configurations."""
    print("=" * 60)
    print("Configuration Comparison")
    print("=" * 60)

    config1 = Configuration.from_string("1s2.2s2.2p6")
    config2 = Configuration.from_string("1s2.2s2.2p5.3s1")

    print(f"Config 1: {config1}")
    print(f"Config 2: {config2}")

    # Compare configurations
    diff = config1.compare(config2)

    print(f"\nDifference (Config1 - Config2):")
    for shell, occupation_diff in diff.items():
        if occupation_diff != 0:
            print(f"  {shell}: {occupation_diff:+d}")
    print()


def configuration_iteration():
    """Iterate over shells in a configuration."""
    print("=" * 60)
    print("Configuration Iteration")
    print("=" * 60)

    config = Configuration.from_string("1s2.2s2.2p6.3s2.3p1")

    print(f"Configuration: {config}\n")
    print("Shells:")
    for shell in config:
        print(
            f"  {str(shell)}: " f"{shell.occupation}/{shell.max_occupation()} electrons"
        )
    print()


def compact_notation():
    """Use compact notation for configurations."""
    print("=" * 60)
    print("Compact Notation")
    print("=" * 60)

    # Create from compact notation
    config = Configuration.from_compact_string("1*2.2*8.3*2")
    print(f"From compact '1*2.2*8.3*2': {config}")

    # Convert to different separators
    config2 = Configuration.from_string("1s2.2s2.2p6.3s2.3p6.3d10")
    space_sep = config2.to_string(separator=" ")
    compact = config2.to_string(separator="")
    print(f"\nDifferent separators for {config2}:")
    print(f"  Space-separated: {space_sep}")
    print(f"  Compact: {compact}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("AtomKit Basic Usage Examples")
    print("=" * 60 + "\n")

    shell_basics()
    configuration_from_shells()
    configuration_from_string()
    configuration_from_element()
    configuration_comparison()
    configuration_iteration()
    compact_notation()

    print("=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
