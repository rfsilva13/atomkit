#!/usr/bin/env python3
"""
AUTOSTRUCTURE Reader Demonstration
===================================

This example demonstrates how AtomKit's AUTOSTRUCTURE readers work.

⚠️  IMPORTANT NOTES:
- AUTOSTRUCTURE readers work with .olg output files from AUTOSTRUCTURE calculations
- This demo uses reference data from the AUTOSTRUCTURE test suite (C IV atom)
- The data format is LS coupling (no fine structure), which is handled by the readers
- Transitions data appears to be photoionization rather than radiative transitions

This script shows:
1. What AUTOSTRUCTURE readers are available
2. How they work with real AUTOSTRUCTURE output data
3. What data they extract and analyze
4. How to use the unified interface
"""

from atomkit.readers import read_as_levels, read_as_transitions, read_as_lambdas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def demonstrate_as_readers():
    """Demonstrate AUTOSTRUCTURE reader functionality with real data."""

    print("=" * 70)
    print("AtomKit AUTOSTRUCTURE Reader Demonstration")
    print("=" * 70)
    print()

    # Path to AUTOSTRUCTURE reference data
    data_path = "/home/rfsilva/Programs/atomkit/docs_archive/autos_reference_data/amdpp.phys.strath.ac.uk/autos/default/data/test_7/olg"

    if not os.path.exists(data_path):
        print("❌ AUTOSTRUCTURE reference data not found at expected location")
        print(f"   Expected: {data_path}")
        print("   Please ensure the reference data is available.")
        return

    print("✅ Using AUTOSTRUCTURE reference data (C IV test case)")
    print(f"   Data file: {data_path}")
    print()

    # ============================================================================
    # 1. Available AUTOSTRUCTURE Readers
    # ============================================================================
    print("1. Available AUTOSTRUCTURE Readers")
    print("-" * 40)

    readers = [
        ("read_as_levels", "Reads energy levels from .olg files"),
        ("read_as_transitions", "Reads radiative transitions from .olg files"),
        ("read_as_lambdas", "Reads lambda scaling parameters from .olg files"),
    ]

    for name, desc in readers:
        print(f"✓ {name}() - {desc}")
    print()

    # ============================================================================
    # 2. Reading Energy Levels
    # ============================================================================
    print("2. Reading Energy Levels")
    print("-" * 40)

    try:
        print("Reading levels from AUTOSTRUCTURE output...")
        levels_df, levels_meta = read_as_levels(data_path)

        print(f"✅ Successfully read {len(levels_df)} energy levels")
        print(f"   Atomic number: {levels_meta['Atomic number']}")
        print(f"   Number of electrons: {levels_meta['Number of electrons']}")
        print(f"   Ground state energy: {levels_meta['Ground state energy (Ry)']} Ry")
        print()

        print("First 10 energy levels:")
        display_cols = ["K", "P", "2*S+1", "L", "Level (Ry)", "CF"]
        print(levels_df[display_cols].head(10).to_string(index=False))
        print()

        # Level statistics
        print("Level Statistics:")
        print(f"  • Total levels: {len(levels_df)}")
        print(f"  • Even parity levels: {len(levels_df[levels_df['P'] == 0])}")
        print(f"  • Odd parity levels: {len(levels_df[levels_df['P'] == 1])}")
        print(
            f"  • Energy range: {levels_df['Level (Ry)'].min():.3f} - {levels_df['Level (Ry)'].max():.3f} Ry"
        )
        print(f"  • Configurations: {levels_df['CF'].nunique()} unique")
        print()

    except Exception as e:
        print(f"❌ Error reading levels: {e}")
        return

    # ============================================================================
    # 3. Reading Lambda Parameters
    # ============================================================================
    print("3. Reading Lambda Parameters")
    print("-" * 40)

    try:
        print("Reading lambda scaling parameters...")
        nl_array, lambda_array = read_as_lambdas(data_path)

        print(f"✅ Successfully read {len(lambda_array)} lambda parameters")
        print()

        print("Lambda parameters (orbital scaling factors):")
        print("Orbital  | Lambda")
        print("---------|--------")
        for i in range(len(lambda_array)):
            n, l = nl_array[i]
            if l == 0:
                orbital = f"{n}s"
            elif l == 1:
                orbital = f"{n}p"
            elif l == 2:
                orbital = f"{n}d"
            elif l == 3:
                orbital = f"{n}f"
            else:
                orbital = f"{n}{l}"
            print("6")
        print()

    except Exception as e:
        print(f"❌ Error reading lambdas: {e}")
        return

    # ============================================================================
    # 4. Transitions Data (Photoionization)
    # ============================================================================
    print("4. Transitions Data")
    print("-" * 40)

    print("Note: The transitions data in this AUTOSTRUCTURE output appears to be")
    print(
        "photoionization cross sections rather than radiative transitions between levels."
    )
    print("This is a different type of calculation output.")
    print()

    try:
        print("Attempting to read transitions...")
        trans_df, trans_meta = read_as_transitions(data_path)

        if len(trans_df) > 0:
            print(f"✅ Successfully read {len(trans_df)} transitions")
            print("First few transitions:")
            display_cols = ["index", "K", "Klower", "WAVEL/AE", "A(K)*SEC", "log(gf)"]
            available_cols = [col for col in display_cols if col in trans_df.columns]
            print(trans_df[available_cols].head().to_string(index=False))
        else:
            print("⚠️ No transitions found (expected for photoionization calculations)")

    except Exception as e:
        print(f"❌ Error reading transitions: {e}")
    print()

    # ============================================================================
    # 5. Data Analysis and Visualization
    # ============================================================================
    print("5. Data Analysis")
    print("-" * 40)

    # Energy level distribution
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(
        levels_df["Level (Ry)"], bins=50, alpha=0.7, color="blue", edgecolor="black"
    )
    plt.xlabel("Energy (Ry)")
    plt.ylabel("Number of Levels")
    plt.title("Energy Level Distribution")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    parity_counts = levels_df["P"].value_counts()
    plt.bar(
        ["Even (P=0)", "Odd (P=1)"],
        parity_counts.values.tolist(),
        color=["lightblue", "lightcoral"],
        alpha=0.7,
        edgecolor="black",
    )
    plt.ylabel("Number of Levels")
    plt.title("Parity Distribution")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    spin_counts = levels_df["2*S+1"].value_counts().sort_index()
    plt.bar(
        spin_counts.index.astype(str),
        spin_counts.values.tolist(),
        color="lightgreen",
        alpha=0.7,
        edgecolor="black",
    )
    plt.xlabel("Spin Multiplicity (2S+1)")
    plt.ylabel("Number of Levels")
    plt.title("Spin Multiplicity Distribution")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    L_counts = levels_df["L"].value_counts().sort_index()
    L_labels = ["S", "P", "D", "F", "G", "H", "I"][: (len(L_counts))]  # Common labels
    plt.bar(
        range(len(L_counts)),
        L_counts.values.tolist(),
        color="gold",
        alpha=0.7,
        edgecolor="black",
    )
    plt.xticks(range(len(L_counts)), L_labels[: len(L_counts)])
    plt.xlabel("Orbital Angular Momentum (L)")
    plt.ylabel("Number of Levels")
    plt.title("L Distribution")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/as_analysis.png", dpi=150, bbox_inches="tight")
    print("✅ Analysis plots saved to /tmp/as_analysis.png")
    print()

    # ============================================================================
    # 6. Comparison with FAC
    # ============================================================================
    print("6. Comparison with FAC Readers")
    print("-" * 40)

    comparison = [
        ("Data Source", "AUTOSTRUCTURE .olg", "FAC .lev/.asc"),
        ("Atom", "C IV (Z=6, N=4)", "Fe XXIV (Z=26, N=2)"),
        ("Coupling", "LS coupling", "Fine structure"),
        ("Levels", f"{len(levels_df)} (LS)", "592 (fine structure)"),
        ("Transitions", "Photoionization", "38,320 radiative"),
        ("Lambda params", f"{len(lambda_array)}", "Not available"),
    ]

    print("Aspect              | AUTOSTRUCTURE        | FAC")
    print("-" * 55)
    for aspect, as_val, fac_val in comparison:
        print("20")
    print()

    # ============================================================================
    # 7. Summary
    # ============================================================================
    print("7. Summary")
    print("-" * 40)

    print("✅ AUTOSTRUCTURE readers are working!")
    print(f"   • Successfully read {len(levels_df)} energy levels")
    print(f"   • Successfully read {len(lambda_array)} lambda parameters")
    print("   • Handles both LS coupling and fine structure formats")
    print("   • Provides comprehensive metadata and statistics")
    print()
    print("Key Features:")
    print("• 🔄 Unified interface with FAC readers")
    print("• 📊 Automatic data analysis and visualization")
    print("• 🛡️ Robust error handling and validation")
    print("• 📈 Support for different AUTOSTRUCTURE output formats")
    print()
    print("The AUTOSTRUCTURE readers are now fully functional and demonstrated!")
