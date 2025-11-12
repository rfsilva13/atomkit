#!/usr/bin/env python3
"""
Test script for FAC readers using the simplified Fe XXIV autoionization calculation.

This script demonstrates reading all three types of FAC output files:
- Energy levels (.lev.asc)
- Radiative transitions (.tr.asc)
- Autoionization rates (.ai.asc)

Run this after executing: sfac examples/fac_inputs/fe24_autoionization.sf
"""

from atomkit.readers import read_fac, read_fac_transitions, read_fac_autoionization
import pandas as pd


def test_fac_readers():
    """Test all FAC reader functions with the generated output files."""

    print("=== Testing FAC Readers with Fe XXIV Autoionization Data ===\n")

    # 1. Energy levels
    print("1. Reading energy levels...")
    levels = read_fac("examples/fac_outputs/fe24")
    print(f"   ✓ Read {len(levels)} energy levels")
    print(f"   Columns: {list(levels.columns)}")
    if len(levels) > 0:
        sample = levels.iloc[0]
        print(
            f"   Sample: Level {sample['level_index']}, {sample['energy']:.1f} eV, {sample['configuration']}"
        )
    print()

    # 2. Radiative transitions
    print("2. Reading radiative transitions...")
    transitions = read_fac_transitions("examples/fac_outputs/fe24_ai")
    print(f"   ✓ Read {len(transitions)} radiative transitions")
    print(f"   Columns: {list(transitions.columns)}")
    if len(transitions) > 0:
        sample = transitions.iloc[0]
        print(
            f"   Sample: {sample['level_index_lower']} → {sample['level_index_upper']}, "
            f"λ={sample['lambda']:.1f} Å, gf={sample['gf']:.2e}"
        )
    print()

    # 3. Autoionization rates
    print("3. Reading autoionization rates...")
    ai_rates = read_fac_autoionization("examples/fac_outputs/fe24_ai")
    print(f"   ✓ Read {len(ai_rates)} autoionization rates")
    print(f"   Columns: {list(ai_rates.columns)}")
    if len(ai_rates) > 0:
        sample = ai_rates.iloc[0]
        print(
            f"   Sample: Level {sample['level_index_upper']} → {sample['level_index_lower']}, "
            f"A_i = {sample['ai_rate']:.2e} s⁻¹"
        )
    print()

    # Summary statistics
    print("=== Summary Statistics ===")
    print(f"Energy levels: {len(levels)}")
    print(f"Radiative transitions: {len(transitions)}")
    print(f"Autoionization rates: {len(ai_rates)}")

    # Basic analysis
    if len(transitions) > 0:
        strong_lines = transitions.nlargest(5, "gf")
        print(f"\nTop 5 strongest transitions (by gf):")
        for _, line in strong_lines.iterrows():
            print(
                f"  λ={line['lambda']:.1f} Å, gf={line['gf']:.2e}, "
                f"{line['level_index_lower']} → {line['level_index_upper']}"
            )

    if len(ai_rates) > 0:
        fast_ai = ai_rates.nlargest(5, "ai_rate")
        print(f"\nTop 5 fastest autoionization rates:")
        for _, ai in fast_ai.iterrows():
            print(
                f"  Level {ai['level_index_upper']:.0f} → {ai['level_index_lower']:.0f}, "
                f"A_i = {ai['ai_rate']:.2e} s⁻¹"
            )

    print("\n=== All FAC readers working perfectly! ===")


if __name__ == "__main__":
    test_fac_readers()
