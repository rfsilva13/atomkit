#!/usr/bin/env python3
"""Test universal analysis functions with FAC data."""

import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from atomkit.analysis import (
    load_data,
    calculate_fluorescence_yield,
    label_hole_states,
    calculate_diagram_intensities,
    calculate_spectrum,
)

def test_universal_analysis():
    """Test universal analysis functions with Pd FAC data."""
    
    # Path to test data
    base_filename = "/home/rfsilva/Programs/fac-analysis/input_files/Pd"
    
    print("=" * 70)
    print("Testing Universal Analysis Functions")
    print("=" * 70)
    
    # Test load_data
    print("\n1. Testing load_data()...")
    levels, transitions, auger = load_data(base_filename)
    print(f"   ✓ Loaded {len(levels)} levels")
    print(f"   ✓ Loaded {len(transitions)} transitions")
    print(f"   ✓ Loaded {len(auger)} auger transitions")
    
    # Test calculate_fluorescence_yield
    print("\n2. Testing calculate_fluorescence_yield()...")
    w, rad_sum, auger_sum = calculate_fluorescence_yield(transitions, auger)
    print(f"   ✓ Fluorescence yield: {w:.6f}")
    print(f"   ✓ Radiative rate sum: {rad_sum:.6e}")
    print(f"   ✓ Auger rate sum: {auger_sum:.6e}")
    
    # Test label_hole_states
    print("\n3. Testing label_hole_states()...")
    levels_labeled = label_hole_states(levels, hole_shell="1s")
    hole_count = levels_labeled['is_hole_state'].sum()
    print(f"   ✓ Found {hole_count} K-shell hole states")
    print(f"   ✓ Sample hole labels:")
    hole_samples = levels_labeled[levels_labeled['is_hole_state']]['configuration'].head(3)
    for config in hole_samples:
        print(f"      - {config}")
    
    # Test calculate_diagram_intensities
    print("\n4. Testing calculate_diagram_intensities()...")
    diagram = calculate_diagram_intensities(
        transitions, levels, auger, hole_shell="1s"
    )
    print(f"   ✓ Calculated {len(diagram)} diagram transitions")
    print(f"   ✓ Columns: {list(diagram.columns)}")
    print(f"   ✓ Total intensity: {diagram['intensity'].sum():.6e}")
    
    # Test calculate_spectrum
    print("\n5. Testing calculate_spectrum()...")
    spectrum = calculate_spectrum(
        diagram, 
        energy_min=20800, 
        energy_max=21300, 
        bin_width=1.0
    )
    print(f"   ✓ Generated spectrum with {len(spectrum)} bins")
    print(f"   ✓ Energy range: {spectrum['energy'].min():.1f} - {spectrum['energy'].max():.1f} eV")
    print(f"   ✓ Total spectrum intensity: {spectrum['intensity'].sum():.6e}")
    
    # Show sample spectrum data
    print("\n6. Sample spectrum data (first 5 bins):")
    print(spectrum.head().to_string(index=False))
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)

if __name__ == "__main__":
    test_universal_analysis()
