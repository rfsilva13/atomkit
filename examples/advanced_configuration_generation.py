#!/usr/bin/env python3
"""
Advanced Configuration Generation Examples

This module demonstrates the power and flexibility of atomkit's code-agnostic
configuration generation methods. All examples generate configurations that can
be formatted for ANY atomic structure code (AUTOSTRUCTURE, FAC, GRASP, etc.).

Topics covered:
- Single and multiple excitations
- Doubly excited states for autoionization
- Hole configurations
- Recombined configurations
- Complex filtering and combining strategies
"""

from atomkit import Configuration


def example_1_single_excitations():
    """Example 1: Single excitations with different strategies."""
    print("=" * 70)
    print("Example 1: Single Excitations")
    print("=" * 70)

    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d10 4s1")
    print(f"Ground state: {ground.to_string()}")

    # Simple: excite to any target shells
    print("\n1a. Excite 4s electron to 4p or 5s:")
    excited = ground.generate_excitations(
        target_shells=["4p", "5s"], excitation_level=1
    )
    print(f"  Generated {len(excited)} configurations:")
    for cfg in excited:
        print(f"    {cfg.to_string(separator=' ')}")

    # Restricted: only excite from specific shells
    print("\n1b. Excite only from 3d shell:")
    excited_3d = ground.generate_excitations(
        target_shells=["4p", "4d", "5s"], excitation_level=1, source_shells=["3d"]
    )
    print(f"  Generated {len(excited_3d)} configurations:")
    for cfg in excited_3d[:5]:
        print(f"    {cfg.to_string(separator=' ')}")
    if len(excited_3d) > 5:
        print(f"    ... and {len(excited_3d) - 5} more")


def example_2_double_excitations():
    """Example 2: Double excitations."""
    print("\n" + "=" * 70)
    print("Example 2: Double Excitations")
    print("=" * 70)

    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d6 4s2")
    print(f"Ground state: {ground.to_string()}")

    # Double excitations from valence
    print("\n2a. Double excitations from valence (3d, 4s) to 4p:")
    double_excited = ground.generate_excitations(
        target_shells=["4p"], excitation_level=2, source_shells=["3d", "4s"]
    )
    print(f"  Generated {len(double_excited)} configurations:")
    for cfg in double_excited:
        print(f"    {cfg.to_string(separator=' ')}")

    # Double excitations from core (useful for autoionization)
    print("\n2b. Double excitations from 2p core to 3d:")
    double_core = ground.generate_excitations(
        target_shells=["3d"], excitation_level=2, source_shells=["2p"]
    )
    print(f"  Generated {len(double_core)} configurations:")
    for cfg in double_core:
        print(f"    {cfg.to_string(separator=' ')}")


def example_3_autoionization_states():
    """Example 3: Doubly excited autoionizing configurations."""
    print("\n" + "=" * 70)
    print("Example 3: Autoionization States (Doubly Excited)")
    print("=" * 70)

    # For autoionization: inner-shell excitation + valence excitation
    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d10 4s2")
    print(f"Ground state: {ground.to_string()}")

    print("\n3a. Classic autoionization: 2p → 3d + 4s → 4p")
    print("  (Creates doubly excited state that can autoionize)")

    # Step 1: Excite from 2p to 3d
    inner_excited = ground.generate_excitations(
        target_shells=["3d"], excitation_level=1, source_shells=["2p"]
    )

    # Step 2: From each inner-excited state, excite valence
    autoionizing = []
    for cfg in inner_excited:
        valence_excited = cfg.generate_excitations(
            target_shells=["4p", "5s"], excitation_level=1, source_shells=["4s"]
        )
        autoionizing.extend(valence_excited)

    print(f"  Generated {len(autoionizing)} autoionizing configurations:")
    for cfg in autoionizing[:5]:
        print(f"    {cfg.to_string(separator=' ')}")
    if len(autoionizing) > 5:
        print(f"    ... and {len(autoionizing) - 5} more")

    # Using built-in method - creates (N+1)-electron states with holes
    print("\n3b. Using built-in autoionization generator:")
    print("  (Generates (N+1)-electron states with core holes)")

    # Start with a list of base configurations
    base_configs = [ground]
    auto_configs = Configuration.generate_doubly_excited_autoionizing(
        base_configs, max_n=5, max_l=3, num_holes=1
    )
    print(f"  Generated {len(auto_configs)} autoionizing configurations:")
    for cfg in auto_configs[:5]:
        print(f"    {cfg.to_string(separator=' ')}")
    if len(auto_configs) > 5:
        print(f"    ... and {len(auto_configs) - 5} more")


def example_4_hole_configurations():
    """Example 4: Hole configurations (inner-shell ionization)."""
    print("\n" + "=" * 70)
    print("Example 4: Hole Configurations")
    print("=" * 70)

    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d10 4s2")
    print(f"Ground state: {ground.to_string()}")

    # Single holes (X-ray transitions, Auger processes)
    print("\n4a. Single hole configurations:")
    single_holes = ground.generate_hole_configurations(num_holes=1)
    print(f"  Generated {len(single_holes)} configurations:")
    for cfg in single_holes[:8]:
        print(f"    {cfg.to_string(separator=' ')}")
    if len(single_holes) > 8:
        print(f"    ... and {len(single_holes) - 8} more")

    # Double holes (Auger cascades, double ionization)
    print("\n4b. Double hole configurations:")
    double_holes = ground.generate_hole_configurations(num_holes=2)
    print(f"  Generated {len(double_holes)} configurations:")
    for cfg in double_holes[:5]:
        print(f"    {cfg.to_string(separator=' ')}")
    if len(double_holes) > 5:
        print(f"    ... and {len(double_holes) - 5} more")


def example_5_recombined_configurations():
    """Example 5: Recombined configurations (dielectronic recombination)."""
    print("\n" + "=" * 70)
    print("Example 5: Recombined Configurations")
    print("=" * 70)

    # Start with ionized state
    ionized = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d10")
    print(f"Ionized state: {ionized.to_string()}")

    print("\n5a. Add electron to various shells (recombination):")
    recombined = ionized.generate_recombined_configurations(max_n=5, max_l=3)
    print(f"  Generated {len(recombined)} recombined configurations:")
    # Filter to show only 4s, 4p, 4d, 5s additions
    filtered = [
        c
        for c in recombined
        if any(shell in c.to_string() for shell in ["4s", "4p", "4d", "5s"])
    ]
    for cfg in filtered[:10]:
        print(f"    {cfg.to_string(separator=' ')}")
    if len(filtered) > 10:
        print(f"    ... and {len(filtered) - 10} more")


def example_6_complex_filtering():
    """Example 6: Complex filtering and selection strategies."""
    print("\n" + "=" * 70)
    print("Example 6: Complex Filtering Strategies")
    print("=" * 70)

    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d6 4s2")

    # Generate many excitations
    excited = ground.generate_excitations(
        target_shells=["4p", "4d", "5s", "5p", "5d"],
        excitation_level=1,
        source_shells=["3d", "4s"],
    )
    print(f"\nGenerated {len(excited)} total excitations")

    # Filter 1: Only configurations with 4p
    print("\n6a. Filter: Only configurations with 4p electron:")
    with_4p = [c for c in excited if "4p" in c.to_string()]
    print(f"  Found {len(with_4p)} configurations")
    for cfg in with_4p[:3]:
        print(f"    {cfg.to_string(separator=' ')}")

    # Filter 2: Only configurations with d orbitals in shell n=5
    print("\n6b. Filter: Only configurations with 5d electron:")
    with_5d = [c for c in excited if "5d" in c.to_string()]
    print(f"  Found {len(with_5d)} configurations")
    for cfg in with_5d:
        print(f"    {cfg.to_string(separator=' ')}")

    # Filter 3: Custom logic - configurations with even total electrons in n=4
    print("\n6c. Filter: Custom logic based on occupation:")
    custom_filter = []
    for cfg in excited:
        cfg_str = cfg.to_string(separator=" ")
        # Count electrons in n=4 shells
        n4_electrons = 0
        for shell_str in cfg_str.split():
            if shell_str.startswith("4"):
                # Extract occupation number
                occupation = int(
                    "".join(
                        c for c in shell_str if c.isdigit() and shell_str.index(c) > 0
                    )
                )
                n4_electrons += occupation
        if n4_electrons % 2 == 0:
            custom_filter.append(cfg)

    print(f"  Found {len(custom_filter)} configurations with even n=4 occupation")


def example_7_combining_strategies():
    """Example 7: Combining multiple generation strategies."""
    print("\n" + "=" * 70)
    print("Example 7: Combining Multiple Strategies")
    print("=" * 70)

    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d10 4s1")
    print(f"Ground state: {ground.to_string()}")

    # Strategy 1: Single excitations from 4s
    print("\n7a. Single excitations from 4s:")
    single_4s = ground.generate_excitations(
        target_shells=["4p", "5s"], excitation_level=1, source_shells=["4s"]
    )
    print(f"  Generated {len(single_4s)} configurations")

    # Strategy 2: Single excitations from 3d
    print("\n7b. Single excitations from 3d:")
    single_3d = ground.generate_excitations(
        target_shells=["4p", "4d"], excitation_level=1, source_shells=["3d"]
    )
    print(f"  Generated {len(single_3d)} configurations")

    # Strategy 3: Double excitations from 3d
    print("\n7c. Double excitations from 3d:")
    double_3d = ground.generate_excitations(
        target_shells=["4p"], excitation_level=2, source_shells=["3d"]
    )
    print(f"  Generated {len(double_3d)} configurations")

    # Combine all strategies
    print("\n7d. Combine all strategies:")
    all_configs = [ground] + single_4s + single_3d + double_3d

    # Remove duplicates
    unique_configs = []
    seen = set()
    for cfg in all_configs:
        cfg_str = cfg.to_string()
        if cfg_str not in seen:
            unique_configs.append(cfg)
            seen.add(cfg_str)

    print(f"  Total unique configurations: {len(unique_configs)}")
    print(f"  (Originally {len(all_configs)} before removing duplicates)")

    print("\n  First 10 configurations:")
    for i, cfg in enumerate(unique_configs[:10], 1):
        print(f"    {i:2d}. {cfg.to_string(separator=' ')}")


def example_8_systematic_generation():
    """Example 8: Systematic generation for complete configuration space."""
    print("\n" + "=" * 70)
    print("Example 8: Systematic Generation")
    print("=" * 70)

    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d2 4s2")
    print(f"Ground state: {ground.to_string()}")
    print("\n8. Generate all single excitations to n=5, l≤2 (s,p,d):")

    # Build target shell list systematically
    target_shells = []
    for n in range(4, 6):  # n=4,5
        for l_symbol in ["s", "p", "d"]:
            target_shells.append(f"{n}{l_symbol}")

    print(f"  Target shells: {target_shells}")

    # Generate excitations
    systematic = ground.generate_excitations(
        target_shells=target_shells, excitation_level=1, source_shells=["3d", "4s"]
    )

    print(f"  Generated {len(systematic)} configurations")
    print(f"\n  Sample (first 8):")
    for cfg in systematic[:8]:
        print(f"    {cfg.to_string(separator=' ')}")


def example_9_practical_use_case():
    """Example 9: Practical use case - preparing for atomic calculations."""
    print("\n" + "=" * 70)
    print("Example 9: Practical Use Case")
    print("=" * 70)
    print("Scenario: Cu I, want configs for photoionization cross-sections")

    ground = Configuration.from_string("1s2 2s2 2p6 3s2 3p6 3d10 4s1")
    print(f"\nGround state: {ground.to_string()}")

    print("\nStep 1: Generate low-lying excited states (for bound-bound)")
    low_lying = ground.generate_excitations(
        target_shells=["4p", "4d", "5s"], excitation_level=1
    )
    print(f"  Generated {len(low_lying)} low-lying states")

    print("\nStep 2: Generate autoionizing states (for resonances)")
    # Use built-in generator for (N+1)-electron states with core holes
    autoionizing = Configuration.generate_doubly_excited_autoionizing(
        [ground], max_n=5, max_l=2, num_holes=1
    )
    print(f"  Generated {len(autoionizing)} autoionizing states")

    print("\nStep 3: Filter to keep only physically important states")
    # Example: Keep states with 4p or 5s (strong transitions)
    important = [c for c in low_lying if "4p" in c.to_string() or "5s" in c.to_string()]
    print(f"  Filtered to {len(important)} important states")

    print("\nStep 4: Combine all for complete calculation")
    all_states = [ground] + important + autoionizing[:10]  # Limit autoionizing
    print(f"  Total configurations for calculation: {len(all_states)}")

    print("\n  Ready to format for ANY code:")
    print("    - AUTOSTRUCTURE: configurations_to_autostructure(all_states, ...)")
    print("    - FAC: configurations_to_fac(all_states, ...)  # Future")
    print("    - GRASP: configurations_to_grasp(all_states, ...)  # Future")

    print("\n  Sample configurations:")
    for i, cfg in enumerate(all_states[:5], 1):
        print(f"    {i}. {cfg.to_string(separator=' ')}")


if __name__ == "__main__":
    print("=" * 70)
    print("Advanced Configuration Generation Examples")
    print("=" * 70)
    print("\nDemonstrating the power of code-agnostic configuration generation")
    print("All configurations can be formatted for ANY atomic structure code!")
    print("=" * 70)

    example_1_single_excitations()
    example_2_double_excitations()
    example_3_autoionization_states()
    example_4_hole_configurations()
    example_5_recombined_configurations()
    example_6_complex_filtering()
    example_7_combining_strategies()
    example_8_systematic_generation()
    example_9_practical_use_case()

    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    print("\nKey takeaway: Generate configurations ONCE using these methods,")
    print("then format for AUTOSTRUCTURE, FAC, GRASP, or any other code!")
    print("=" * 70)
