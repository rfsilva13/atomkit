"""
Unified Interface Example: Side-by-Side Comparison
===================================================

This example demonstrates the unified AtomicCalculation interface
by generating input files for BOTH AUTOSTRUCTURE and FAC from the
SAME physical specification.

This shows the power of code-agnostic language: express your
calculation ONCE in physical terms, generate inputs for MULTIPLE codes.
"""

from atomkit.core import AtomicCalculation
from atomkit import Configuration

print("=" * 80)
print("UNIFIED INTERFACE: Side-by-Side Comparison")
print("=" * 80)
print()

# =============================================================================
#                    Example 1: Simple Structure Calculation
# =============================================================================

print("Example 1: Simple Structure Calculation")
print("-" * 80)
print()

# Define configurations (code-agnostic)
ground = Configuration.from_string("1s2.2s2.2p6")
excited_3s = Configuration.from_string("1s2.2s2.2p5.3s1")
excited_3p = Configuration.from_string("1s2.2s2.2p5.3p1")

configs = [ground, excited_3s, excited_3p]

print("Physical specification:")
print(f"  Element: Fe (iron)")
print(f"  Charge: 16+ (Ne-like)")
print(f"  Coupling: LS")
print(f"  Relativistic: none")
print(f"  Configurations: {len(configs)}")
for cfg in configs:
    print(f"    - {cfg}")
print()

# Generate AUTOSTRUCTURE input
print("Generating AUTOSTRUCTURE input...")
calc_as = AtomicCalculation(
    element="Fe",
    charge=16,
    calculation_type="structure",
    coupling="LS",
    relativistic="none",
    configurations=configs,
    code="autostructure",
)

file_as = calc_as.write_input()
print(f"✓ Created: {file_as}")
print()

# Generate FAC input (SAME physical specification!)
print("Generating FAC input...")
calc_fac = AtomicCalculation(
    element="Fe",
    charge=16,
    calculation_type="structure",
    coupling="LS",  # FAC will warn: always jj-based
    relativistic="none",  # FAC will warn: always Dirac
    configurations=configs,
    code="fac",
)

file_fac = calc_fac.write_input()
print(f"✓ Created: {file_fac}")
print()

print("💡 Notice: FAC issued warnings because it's always jj-coupled and")
print("   always relativistic. The system knows each code's limitations!")
print()
print()

# =============================================================================
#                 Example 2: Intermediate Coupling with Breit
# =============================================================================

print("Example 2: Intermediate Coupling + Breit Interaction")
print("-" * 80)
print()

print("Physical specification:")
print(f"  Element: Fe")
print(f"  Charge: 15+ (F-like)")
print(f"  Coupling: IC (intermediate)")
print(f"  Relativistic: Breit interaction")
print(f"  QED: Enabled")
print(f"  Optimization: Energy")
print()

# Same specification for both codes
for code in ["autostructure", "fac"]:
    print(f"Generating {code.upper()} input...")

    calc = AtomicCalculation(
        element="Fe",
        charge=15,
        calculation_type="structure",
        coupling="IC",
        relativistic="Breit",
        qed_corrections=True,
        optimization="energy",
        configurations=[
            "1s2.2s2.2p5",
            "1s2.2s2.2p4.3s1",
        ],
        code=code,
    )

    filepath = calc.write_input()
    print(f"✓ Created: {filepath}")
    print()

print()

# =============================================================================
#              Example 3: DR Calculation with Code-Specific Tuning
# =============================================================================

print("Example 3: Dielectronic Recombination with Code-Specific Options")
print("-" * 80)
print()

print("Physical specification:")
print(f"  Element: Fe")
print(f"  Charge: 15+")
print(f"  Type: DR (dielectronic recombination)")
print(f"  Coupling: IC")
print(f"  Energy range: 0-100 eV, 1000 points")
print(f"  Core: Ne-like")
print()

# AUTOSTRUCTURE with code-specific tuning
print("AUTOSTRUCTURE with code-specific options:")
calc_as_dr = AtomicCalculation(
    element="Fe",
    charge=15,
    calculation_type="DR",
    coupling="IC",
    energy_range=(0, 100, 1000),
    core="Ne-like",
    code="autostructure",
    code_options={
        "SCFRAC": 0.85,  # AS-specific: SCF convergence
        "NLAM": 5,  # AS-specific: Lambda scaling points
        "DRR": {  # AS-specific: Rydberg series
            "NMIN": 3,
            "NMAX": 20,
            "LMAX": 10,
        },
    },
)

file_as_dr = calc_as_dr.write_input()
print(f"✓ Created: {file_as_dr}")
print()

# FAC with code-specific tuning
print("FAC with code-specific options:")
calc_fac_dr = AtomicCalculation(
    element="Fe",
    charge=15,
    calculation_type="DR",
    coupling="IC",  # FAC uses jj, but that's OK
    energy_range=(0, 100, 1000),
    core="Ne-like",
    code="fac",
    code_options={
        # FAC-specific options can go here
        # e.g., {"MaxLevels": 10000}
    },
)

file_fac_dr = calc_fac_dr.write_input(verbose=False)  # Suppress warnings
print(f"✓ Created: {file_fac_dr}")
print()

print("💡 code_options allows fine-tuning for each code while keeping")
print("   the main specification code-agnostic!")
print()
print()

# =============================================================================
#                          Example 4: Batch Comparison
# =============================================================================

print("Example 4: Batch Code Comparison")
print("-" * 80)
print()

print("Generating inputs for multiple coupling schemes across both codes...")
print()

coupling_schemes = ["LS", "IC"]
elements_charges = [
    ("O", 7),  # O VIII (H-like oxygen)
    ("Ne", 9),  # Ne X (H-like neon)
    ("Fe", 15),  # Fe XVI (F-like iron)
]

for element, charge in elements_charges:
    for coupling in coupling_schemes:
        for code in ["autostructure", "fac"]:
            calc = AtomicCalculation(
                element=element,
                charge=charge,
                calculation_type="structure",
                coupling=coupling,
                code=code,
                name=f"{element}_{charge}_{coupling}_{code}",
            )

            filepath = calc.write_input(verbose=False)
            print(f"✓ {element} {charge}+ {coupling:3s} {code:14s} → {filepath.name}")

print()
print("Generated 12 input files (2 elements × 2 couplings × 2 codes)!")
print()
print()

# =============================================================================
#                              Summary
# =============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print("Benefits of Unified Interface:")
print("  ✓ Express calculations in PHYSICAL terms")
print("  ✓ ONE specification works for MULTIPLE codes")
print("  ✓ Automatic translation to code-specific formats")
print("  ✓ Warnings for code limitations")
print("  ✓ code_options for fine-tuning when needed")
print("  ✓ Easy to compare codes (same input!)")
print()

print("Files Generated:")
print("  - AUTOSTRUCTURE: .dat files (NAMELIST format)")
print("  - FAC: .sf files (Python scripts)")
print()

print("Next Steps:")
print("  1. Run AUTOSTRUCTURE: $ as < fe_16_structure.dat")
print("  2. Run FAC: $ python fe_16_structure.sf")
print("  3. Compare results!")
print()

print("=" * 80)
print("🎉 Unified interface working! Compare the generated files to see")
print("   how the SAME physical specification translates to different codes.")
print("=" * 80)
