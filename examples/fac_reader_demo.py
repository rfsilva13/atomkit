#!/usr/bin/env python3
"""
FAC Reader Demonstration
========================

This example demonstrates how to use AtomKit's FAC readers to load and analyze
output from FAC (Flexible Atomic Code) calculations.

We use the Fe XXIV autoionization calculation output as an example, which includes:
- Energy levels (.lev.asc)
- Radiative transitions (.tr.asc)
- Autoionization rates (.ai.asc)

The example shows:
1. Loading data from FAC output files
2. Basic data inspection and statistics
3. Simple analysis and filtering
4. Cross-referencing between different data types
"""

import numpy as np
import pandas as pd
from atomkit.readers import read_fac, read_fac_transitions, read_fac_autoionization


def main():
    print("=" * 70)
    print("AtomKit FAC Reader Demonstration")
    print("=" * 70)
    print()

    # ============================================================================
    # 1. Load FAC Energy Levels
    # ============================================================================
    print("1. Loading FAC Energy Levels")
    print("-" * 40)

    levels = read_fac("examples/fac_outputs/fe24")
    print(f"✓ Loaded {len(levels)} energy levels")
    print(f"✓ Columns: {list(levels.columns)}")
    print()

    # Show sample levels
    print("Sample energy levels:")
    sample_levels = levels.head(5)[["level_index", "energy", "configuration", "term"]]
    for _, level in sample_levels.iterrows():
        print(
            f"  Level {level['level_index']}: {level['energy']:.1f} eV, {level['configuration']}, {level['term']}"
        )
    print()

    # ============================================================================
    # 2. Load FAC Radiative Transitions
    # ============================================================================
    print("2. Loading FAC Radiative Transitions")
    print("-" * 40)

    transitions = read_fac_transitions("examples/fac_outputs/fe24_ai")
    print(f"✓ Loaded {len(transitions)} radiative transitions")
    print(f"✓ Columns: {list(transitions.columns)}")
    print()

    # Show sample transitions
    print("Sample radiative transitions:")
    sample_trans = transitions.head(5)[
        ["level_index_lower", "level_index_upper", "lambda", "gf", "A", "type"]
    ]
    for _, trans in sample_trans.iterrows():
        print(
            f"  {trans['level_index_lower']} → {trans['level_index_upper']}: λ={trans['lambda']:.1f} Å, gf={trans['gf']:.2e}, A={trans['A']:.2e} s⁻¹, {trans['type']}"
        )
    print()

    # ============================================================================
    # 3. Load FAC Autoionization Rates
    # ============================================================================
    print("3. Loading FAC Autoionization Rates")
    print("-" * 40)

    ai_rates = read_fac_autoionization("examples/fac_outputs/fe24_ai")
    print(f"✓ Loaded {len(ai_rates)} autoionization rates")
    print(f"✓ Columns: {list(ai_rates.columns)}")
    print()

    # Show sample AI rates
    print("Sample autoionization rates:")
    sample_ai = ai_rates.head(5)[["level_index_upper", "level_index_lower", "ai_rate"]]
    for _, ai in sample_ai.iterrows():
        print(
            f"  Level {ai['level_index_upper']} → {ai['level_index_lower']}: A_i = {ai['ai_rate']:.2e} s⁻¹"
        )
    print()

    # ============================================================================
    # 4. Basic Analysis
    # ============================================================================
    print("4. Basic Analysis")
    print("-" * 40)

    # Energy level statistics
    print("Energy Level Statistics:")
    print(f"  Total levels: {len(levels)}")
    print(
        f"  Energy range: {levels['energy'].min():.3f} - {levels['energy'].max():.3f} eV"
    )
    print(f"  Mean energy: {levels['energy'].mean():.3f} eV")
    print(f"  Median energy: {levels['energy'].median():.3f} eV")
    print()

    # Transition statistics
    print("Radiative Transition Statistics:")
    print(f"  Total transitions: {len(transitions)}")
    print(
        f"  Wavelength range: {transitions['lambda'].min():.2f} - {transitions['lambda'].max():.2f} Å"
    )
    print(f"  gf range: {transitions['gf'].min():.2e} - {transitions['gf'].max():.2e}")
    print(f"  A range: {transitions['A'].min():.2e} - {transitions['A'].max():.2e} s⁻¹")
    print()

    # Autoionization statistics
    print("Autoionization Statistics:")
    print(f"  Total AI rates: {len(ai_rates)}")
    print(
        f"  AI rate range: {ai_rates['ai_rate'].min():.2e} - {ai_rates['ai_rate'].max():.2e} s⁻¹"
    )
    print(f"  Mean AI rate: {ai_rates['ai_rate'].mean():.2e} s⁻¹")
    print(f"  Median AI rate: {ai_rates['ai_rate'].median():.2e} s⁻¹")
    print()

    # ============================================================================
    # 5. Advanced Analysis: Strongest Transitions
    # ============================================================================
    print("5. Strongest Radiative Transitions")
    print("-" * 40)

    # Find strongest transitions by oscillator strength
    strongest = transitions.nlargest(10, "gf")
    print("Top 10 strongest transitions (by oscillator strength):")
    print("Rank | Lower→Upper | Wavelength (Å) | gf | A (s⁻¹) | Type")
    print("-" * 60)

    for i, (_, trans) in enumerate(strongest.iterrows(), 1):
        lower = trans['level_index_lower']
        upper = trans['level_index_upper']
        print(
            f"{i:2d}   | {lower:3.0f}→{upper:3.0f}    | {trans['lambda']:8.1f}    | {trans['gf']:.2e} | {trans['A']:.2e} | {trans['type']}"
        )
    print()

    # ============================================================================
    # 6. Advanced Analysis: Fastest Autoionization
    # ============================================================================
    print("6. Fastest Autoionization Rates")
    print("-" * 40)

    # Find fastest autoionization rates
    fastest_ai = ai_rates.nlargest(10, "ai_rate")
    print("Top 10 fastest autoionization rates:")
    print("Rank | Upper→Lower | A_i (s⁻¹)")
    print("-" * 35)

    for i, (_, ai) in enumerate(fastest_ai.iterrows(), 1):
        upper = ai['level_index_upper']
        lower = ai['level_index_lower']
        print(
            f"{i:2d}   | {upper:3.0f}→{lower:3.0f}   | {ai['ai_rate']:.2e}"
        )
    print()

    # ============================================================================
    # 7. Cross-Analysis: Energy Level Ranges
    # ============================================================================
    print("7. Energy Level Distribution")
    print("-" * 40)

    # Group levels by energy ranges
    energy_bins = [0, 6900, 7000, 7100, 7200, float("inf")]
    labels = ["<6900 eV", "6900-7000 eV", "7000-7100 eV", "7100-7200 eV", ">7200 eV"]

    levels["energy_range"] = pd.cut(levels["energy"], bins=energy_bins, labels=labels)
    energy_dist = levels["energy_range"].value_counts().sort_index()

    print("Energy level distribution:")
    for range_name, count in energy_dist.items():
        print(f"  {range_name}: {count} levels")
    print()

    # ============================================================================
    # 8. Configuration Analysis
    # ============================================================================
    print("8. Configuration Analysis")
    print("-" * 40)

    # Most common configurations
    config_counts = levels["configuration"].value_counts().head(10)
    print("Most common electron configurations:")
    for config, count in config_counts.items():
        print(f"  {config}: {count} levels")
    print()

    # ============================================================================
    # 9. Summary
    # ============================================================================
    print("9. Summary")
    print("-" * 40)

    print("FAC Reader Demonstration Complete!")
    print()
    print("Key takeaways:")
    print(f"• Successfully loaded {len(levels)} energy levels from FAC calculation")
    print(f"• Successfully loaded {len(transitions)} radiative transitions")
    print(f"• Successfully loaded {len(ai_rates)} autoionization rates")
    print("• All data properly parsed with consistent column names")
    print("• Ready for further analysis, plotting, or export")
    print()
    print("Data is now available as pandas DataFrames for:")
    print("• Energy level analysis and Grotrian diagrams")
    print("• Spectral line identification and synthesis")
    print("• Autoionization rate calculations")
    print("• Plasma modeling and atomic database creation")
    print()

    return levels, transitions, ai_rates


if __name__ == "__main__":
    # Run the demonstration
    levels_df, transitions_df, ai_df = main()

    # The DataFrames are now available for further analysis
    print("DataFrames available for further analysis:")
    print("• levels_df: Energy levels")
    print("• transitions_df: Radiative transitions")
    print("• ai_df: Autoionization rates")
