#!/usr/bin/env python3
"""
AUTOSTRUCTURE Workflow - Code-Agnostic Examples

Demonstrates the atomkit philosophy with advanced physics:
- Generate configurations using general Configuration methods (physics)
- Single/double excitations, autoionization, hole states, etc.
- Format for AUTOSTRUCTURE only at the final I/O step

All configuration generation is completely code-agnostic.
"""

from pathlib import Path
from atomkit import Configuration
from atomkit.converters import configurations_to_autostructure


def example_1_basic():
    """Example 1: Basic single excitations (code-agnostic)."""
    print("=" * 70)
    print("Example 1: Single Excitations from Valence")
    print("=" * 70)

    # Physics: Generate single excitations
    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d6 4s2")
    excited = ground.generate_excitations(
        target_shells=["4p", "5s", "4d"],
        excitation_level=1,
        source_shells=["3d", "4s"]  # Excite from valence only
    )

    print(f"  Ground: {ground.to_string(separator=' ')}")
    print(f"  Generated {len(excited)} single excitations")
    print(f"  First 3: {[c.to_string(separator=' ') for c in excited[:3]]}")

    # I/O: Format for AUTOSTRUCTURE
    result = configurations_to_autostructure(
        [ground] + excited[:5], last_core_orbital="3p"
    )
    print(f"  Formatted {result['mxconf']} configs for AUTOSTRUCTURE")


def example_2_double_excitations():
    """Example 2: Double excitations for autoionization."""
    print("\n" + "=" * 70)
    print("Example 2: Double Excitations (Autoionization)")
    print("=" * 70)
    print("Useful for: Autoionization resonances, doubly excited states\n")

    # Ground state of neutral Ne
    ground = Configuration.from_string("1s2 2s2 2p6")

    # Double excitations: 2p^2 -> nl nl' (autoionizing states)
    double_exc = ground.generate_excitations(
        target_shells=["3s", "3p", "3d", "4s", "4p"],
        excitation_level=2,  # TWO electrons excited
        source_shells=["2p"]  # Both from 2p
    )

    print(f"  Ground: {ground.to_string(separator=' ')}")
    print(f"  Generated {len(double_exc)} doubly excited configurations")
    print(f"\n  Examples of autoionizing states:")
    for i, cfg in enumerate(double_exc[:5], 1):
        print(f"    {i}. {cfg.to_string(separator=' ')}")
        
    # Format for AUTOSTRUCTURE
    result = configurations_to_autostructure(
        [ground] + double_exc, last_core_orbital="2p"
    )
    print(f"\n  Formatted {result['mxconf']} configs for AUTOSTRUCTURE")


def example_3_core_excitations():
    """Example 3: Core excitations (inner-shell excitations)."""
    print("\n" + "=" * 70)
    print("Example 3: Core Excitations (Inner-Shell)")
    print("=" * 70)
    print("Useful for: X-ray absorption, core-hole states\n")

    # Fe I ground state
    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d6 4s2")

    # Excite from 2p core to 3d/4p (L-shell absorption)
    core_exc = ground.generate_excitations(
        target_shells=["3d", "4p", "4s"],
        excitation_level=1,
        source_shells=["2p"]  # Excite from core!
    )

    print(f"  Ground: {ground.to_string(separator=' ')}")
    print(f"  Generated {len(core_exc)} core excitations (2p → valence)")
    print(f"\n  Examples of L-shell excited states:")
    for i, cfg in enumerate(core_exc[:5], 1):
        print(f"    {i}. {cfg.to_string(separator=' ')}")

    # Format for AUTOSTRUCTURE
    result = configurations_to_autostructure(
        [ground] + core_exc, last_core_orbital="3p"
    )
    print(f"\n  Formatted {result['mxconf']} configs for AUTOSTRUCTURE")


def example_4_hole_configurations():
    """Example 4: Hole configurations (ionized states)."""
    print("\n" + "=" * 70)
    print("Example 4: Hole Configurations (Ionization)")
    print("=" * 70)
    print("Useful for: Photoionization, ionized states\n")

    # Neutral atom
    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d10 4s1")

    # Create hole configurations (remove electrons)
    holes = ground.generate_hole_configurations(num_holes=1)

    print(f"  Ground (neutral): {ground.to_string(separator=' ')}")
    print(f"  Generated {len(holes)} singly ionized configurations")
    print(f"\n  Examples of ionized states:")
    for i, cfg in enumerate(holes[:5], 1):
        print(f"    {i}. {cfg.to_string(separator=' ')}")

    # Format for AUTOSTRUCTURE
    result = configurations_to_autostructure(holes, last_core_orbital="3p")
    print(f"\n  Formatted {result['mxconf']} configs for AUTOSTRUCTURE")


def example_5_recombination():
    """Example 5: Recombination configurations (capture processes)."""
    print("\n" + "=" * 70)
    print("Example 5: Recombination Configurations")
    print("=" * 70)
    print("Useful for: Dielectronic recombination, capture processes\n")

    # Ion ground state
    ion = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d9")

    # Recombination: add electron to various shells
    recomb = ion.generate_recombined_configurations(
        max_n=5,
        max_l=2,
        num_electrons=1
    )

    print(f"  Ion (3d9): {ion.to_string(separator=' ')}")
    print(f"  Generated {len(recomb)} recombined configurations")
    print(f"\n  Examples of capture states (3d9 + e → 3d9 nl):")
    for i, cfg in enumerate(recomb[:8], 1):
        print(f"    {i}. {cfg.to_string(separator=' ')}")

    # Format for AUTOSTRUCTURE
    result = configurations_to_autostructure(recomb, last_core_orbital="3p")
    print(f"\n  Formatted {result['mxconf']} configs for AUTOSTRUCTURE")


def example_6_mixed_strategy():
    """Example 6: Combining multiple generation strategies."""
    print("\n" + "=" * 70)
    print("Example 6: Mixed Strategy (Complete Calculation)")
    print("=" * 70)
    print("Combine: ground + single exc + double exc + autoionization\n")

    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d6 4s2")

    # Strategy 1: Single excitations from valence
    single_exc = ground.generate_excitations(
        target_shells=["4p", "5s", "4d"],
        excitation_level=1,
        source_shells=["3d", "4s"]
    )
    print(f"  Strategy 1: {len(single_exc)} single excitations (3d,4s → 4p,5s,4d)")

    # Strategy 2: Double excitations from valence (autoionization)
    double_exc = ground.generate_excitations(
        target_shells=["4p", "5s"],
        excitation_level=2,
        source_shells=["3d", "4s"]
    )
    print(f"  Strategy 2: {len(double_exc)} double excitations (autoionizing)")

    # Strategy 3: Core excitations (2p → 3d)
    core_exc = ground.generate_excitations(
        target_shells=["3d"],
        excitation_level=1,
        source_shells=["2p"]
    )
    print(f"  Strategy 3: {len(core_exc)} core excitations (2p → 3d)")

    # Combine all
    all_configs = [ground] + single_exc + double_exc + core_exc
    print(f"\n  Total: {len(all_configs)} configurations")

    # Format for AUTOSTRUCTURE
    result = configurations_to_autostructure(all_configs, last_core_orbital="3p")
    print(f"  Formatted {result['mxconf']} configs for AUTOSTRUCTURE")


def example_7_core_valence_format():
    """Example 7: Core + valence format (maximum flexibility)."""
    print("\n" + "=" * 70)
    print("Example 7: Core + Valence Format")
    print("=" * 70)
    print("Define core once, work with valence only\n")

    # Define core once
    core = Configuration.from_string("1s2 2s2 2p6 3s2 3p6")
    print(f"  Core: {core.to_string(separator=' ')}")

    # Generate valence configurations only
    valence_ground = Configuration.from_string("3d6 4s2")
    valence_exc = valence_ground.generate_excitations(
        target_shells=["4p", "5s"],
        excitation_level=1
    )

    print(f"  Valence ground: {valence_ground.to_string(separator=' ')}")
    print(f"  Generated {len(valence_exc)} valence excitations")

    # Format with core prepended automatically
    result = configurations_to_autostructure(
        [valence_ground] + valence_exc,
        core=core,
        last_core_orbital="3p"
    )
    
    print(f"\n  Full configurations with core prepended:")
    for i, cfg in enumerate(result["configurations"][:3], 1):
        print(f"    {i}. {cfg}")
    print(f"  Formatted {result['mxconf']} configs for AUTOSTRUCTURE")


def example_8_file_output():
    """Example 8: Write to file."""
    print("\n" + "=" * 70)
    print("Example 8: Write to File")
    print("=" * 70)

    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d10 4s1")
    excited = ground.generate_excitations(
        target_shells=["4p", "5s"], excitation_level=1
    )

    output_file = Path("cu_autostructure.txt")
    result = configurations_to_autostructure(
        [ground] + excited,
        last_core_orbital="3p",
        output_file=output_file
    )

    print(f"  Generated {result['mxconf']} configurations")
    print(f"  Written to: {output_file}")
    
    if output_file.exists():
        print(f"\n  File preview (first 10 lines):")
        with open(output_file) as f:
            for i, line in enumerate(f, 1):
                if i <= 10:
                    print(f"    {line.rstrip()}")
                else:
                    break


if __name__ == "__main__":
    print("=" * 70)
    print("AUTOSTRUCTURE Workflow - Advanced Physics Examples")
    print("=" * 70)
    print("\nAll examples use code-agnostic methods!")
    print("Same configurations can be formatted for FAC, GRASP, etc.")
    print("=" * 70)

    example_1_basic()
    example_2_double_excitations()
    example_3_core_excitations()
    example_4_hole_configurations()
    example_5_recombination()
    example_6_mixed_strategy()
    example_7_core_valence_format()
    example_8_file_output()

    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)
