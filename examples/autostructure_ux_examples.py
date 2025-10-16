"""
AUTOSTRUCTURE UX Enhancements - Examples
=========================================

Phase 6 enhances the AUTOSTRUCTURE writer with:
1. Helper classes for common configurations
2. High-level presets for typical calculation types
3. Fluent interface for method chaining
4. Validation methods to catch errors early

This example demonstrates all these features.
"""

from atomkit.autostructure import (
    ASWriter,
    CoreSpecification,
    SymmetryRestriction,
    EnergyShifts,
    CollisionParams,
    OptimizationParams,
    RydbergSeries,
)
from atomkit import Configuration


# ============================================================================
#          Example 1: Helper Classes - Simplified Core Specification
# ============================================================================

print("=" * 70)
print("Example 1: Using Helper Classes")
print("=" * 70)

# Old way (verbose):
# asw.add_salgeb(CUP='IC', RAD='E1', KCOR1=1, KCOR2=3)

# New way (readable):
core = CoreSpecification.neon_like()  # 1s2 2s2 2p6
print(f"Neon-like core: KCOR1={core.kcor1}, KCOR2={core.kcor2}")

# Other predefined cores:
he_core = CoreSpecification.helium_like()  # 1s2
ar_core = CoreSpecification.argon_like()  # through 3p6
custom_core = CoreSpecification.from_orbitals(1, 5)  # custom range

print("✅ Helper classes make code more readable!\n")


# ============================================================================
#          Example 2: High-Level Presets - Quick Setup
# ============================================================================

print("=" * 70)
print("Example 2: High-Level Presets")
print("=" * 70)

# Example 2a: Structure Calculation
# ----------------------------------
print("\n2a. Structure calculation preset:")

asw = ASWriter.for_structure_calculation(
    "ne_structure.dat",
    nzion=10,
    coupling="IC",
    radiation="E1",
    core=CoreSpecification.helium_like(),
    comment="Ne-like iron structure"
)

# Just add your configurations and you're done!
# asw.configs_from_atomkit([ground, excited1, excited2])
# asw.close()

print(f"Created structure calculation input")
print(f"  - Coupling: IC")
print(f"  - Radiation: E1")
print(f"  - Core: He-like (1s2)")


# Example 2b: Photoionization
# ---------------------------
print("\n2b. Photoionization preset:")

asw_pi = ASWriter.for_photoionization(
    "ne_pi.dat",
    nzion=10,
    energy_min=0.0,
    energy_max=100.0,
    n_energies=20,
    coupling="IC"
)

print(f"Created photoionization input")
print(f"  - Energy range: 0-100 Ry")
print(f"  - Energy points: 20")
print(f"  - Automatically configured RUN='PI' and SRADCON")


# Example 2c: Dielectronic Recombination
# --------------------------------------
print("\n2c. Dielectronic recombination preset:")

rydberg = RydbergSeries(n_min=3, n_max=15, l_max=7)

asw_dr = ASWriter.for_dielectronic_recombination(
    "ne_dr.dat",
    nzion=10,
    rydberg=rydberg,
    energy_min=0.0,
    energy_max=150.0
)

print(f"Created DR calculation input")
print(f"  - Rydberg series: n={rydberg.n_min}-{rydberg.n_max}, l=0-{rydberg.l_max}")
print(f"  - Automatically configured RUN='DR', SRADCON, and DRR")


# Example 2d: Electron Impact Collision
# -------------------------------------
print("\n2d. Collision calculation preset:")

collision = CollisionParams(
    min_L=0, max_L=10,
    min_J=0, max_J=20,
    max_exchange_L=8,
    include_orbit_orbit=True
)

asw_coll = ASWriter.for_collision(
    "ne_collision.dat",
    nzion=10,
    collision=collision,
    coupling="IC"
)

print(f"Created collision input")
print(f"  - L range: {collision.min_L}-{collision.max_L}")
print(f"  - J range: {collision.min_J}-{collision.max_J}")
print(f"  - Exchange L limit: {collision.max_exchange_L}")
print(f"  - Orbit-orbit interaction: {collision.include_orbit_orbit}")

print("\n✅ Presets handle all the boilerplate for common calculation types!\n")


# ============================================================================
#          Example 3: Fluent Interface - Method Chaining
# ============================================================================

print("=" * 70)
print("Example 3: Fluent Interface (Method Chaining)")
print("=" * 70)

# Build a complex calculation with readable method chaining
asw_fluent = (ASWriter("ne_advanced.dat")
    .with_core(CoreSpecification.helium_like())
    .with_optimization(OptimizationParams(
        include_lowest=10,
        n_lambdas=5,
        weighting='statistical'
    ))
    .with_energy_shifts(EnergyShifts(
        ls_shift=0.5,
        ic_shift=0.3
    ))
)

print("Created writer with chained configuration:")
print("  ✓ He-like core")
print("  ✓ Optimization (10 lowest states, 5 λ parameters)")
print("  ✓ Energy shifts (LS: 0.5, IC: 0.3 Ry)")

print("\n✅ Fluent interface allows elegant, readable configuration!\n")


# ============================================================================
#          Example 4: Validation - Catch Errors Early
# ============================================================================

print("=" * 70)
print("Example 4: Validation Methods")
print("=" * 70)

# Example 4a: Incomplete configuration
print("\n4a. Validation catches missing required sections:")

asw_incomplete = ASWriter("incomplete.dat")
asw_incomplete.write_header("Test")
asw_incomplete.add_salgeb(CUP='LS', RAD='E1')

warnings = asw_incomplete.validate()
print(f"Validation warnings: {len(warnings)}")
for w in warnings:
    print(f"  ⚠️  {w}")


# Example 4b: Check completeness
print("\n4b. Check if input is ready:")

print(f"Is incomplete input ready? {asw_incomplete.check_completeness()}")

# Add missing SMINIM
asw_incomplete.add_sminim(NZION=10)
print(f"After adding SMINIM: {asw_incomplete.check_completeness()}")


# Example 4c: Validation catches specific errors
print("\n4c. Validation catches RUN-type specific requirements:")

asw_pi_missing = ASWriter("pi_missing.dat")
asw_pi_missing.write_header("PI without SRADCON")
asw_pi_missing.add_salgeb(CUP='IC', RUN='PI')  # PI requires SRADCON!
asw_pi_missing.add_sminim(NZION=10)

warnings_pi = asw_pi_missing.validate()
print(f"PI validation warnings:")
for w in warnings_pi:
    print(f"  ⚠️  {w}")


# Example 4d: Use validate_and_raise for critical validation
print("\n4d. Raise exception on validation errors:")

try:
    asw_incomplete.validate_and_raise()
except ValueError as e:
    print(f"❌ Validation failed (as expected):")
    print(f"   {str(e)[:100]}...")

print("\n✅ Validation helps catch configuration errors before running AUTOSTRUCTURE!\n")


# ============================================================================
#          Example 5: Complete Workflow with All Features
# ============================================================================

print("=" * 70)
print("Example 5: Complete Workflow")
print("=" * 70)

# Combine all UX features for a production calculation
print("\nBuilding a complete Ne-like structure calculation...")

# Define configurations
ground = Configuration.from_string("1s2.2s2.2p6")
excited = ground.generate_excitations(["3s", "3p", "3d"], excitation_level=1)

# Create writer with preset and fluent configuration
asw_complete = (
    ASWriter.for_structure_calculation(
        "ne_complete.dat",
        nzion=10,
        coupling="IC",
        radiation="E1",
        core=CoreSpecification.helium_like(),
        optimization=OptimizationParams(
            include_lowest=15,
            n_lambdas=5,
            weighting='statistical'
        ),
        comment="Ne-like Fe XVI structure with optimization"
    )
    .with_energy_shifts(EnergyShifts(ic_shift=0.2))
)

# Add configurations
asw_complete.configs_from_atomkit([ground] + excited, last_core_orbital='1s')

# Validate before writing
print("\nValidating configuration...")
warnings = asw_complete.validate()
if warnings:
    print("⚠️  Warnings found:")
    for w in warnings:
        print(f"   - {w}")
else:
    print("✅ Configuration is valid!")

# Check completeness
if asw_complete.check_completeness():
    print("✅ Input is complete and ready to write!")
    # asw_complete.close()  # Uncomment to actually write file
else:
    print("❌ Input is not complete yet")

print("\n✅ Complete workflow: Preset → Configure → Validate → Write!\n")


# ============================================================================
#          Summary: Before vs After
# ============================================================================

print("=" * 70)
print("Summary: Code Comparison")
print("=" * 70)

print("""
BEFORE (Verbose, error-prone):
-------------------------------
asw = ASWriter("pi.dat")
asw.write_header("Photoionization")
asw.add_salgeb(CUP='IC', RAD='E1', RUN='PI', KCOR1=1, KCOR2=1)
# ... add configurations ...
asw.add_sminim(NZION=10, INCLUD=10, NLAM=5)
asw.add_sradcon(MENG=-15, EMIN=0.0, EMAX=100.0)
asw.close()


AFTER (Clean, safe, readable):
-------------------------------
asw = ASWriter.for_photoionization(
    "pi.dat",
    nzion=10,
    energy_min=0.0,
    energy_max=100.0,
    core=CoreSpecification.helium_like()
)
# ... add configurations ...
asw.validate_and_raise()  # Catch errors early!
asw.close()


Benefits:
✅ Less boilerplate code
✅ More readable and maintainable
✅ Fewer chances for user error
✅ Early error detection via validation
✅ Method chaining for fluent API
✅ Reusable configuration objects
""")

print("=" * 70)
print("🎉 Phase 6 UX Enhancements Complete!")
print("=" * 70)
