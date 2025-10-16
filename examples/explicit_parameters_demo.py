"""
Example demonstrating new explicit AUTOSTRUCTURE parameters.

This example shows how to use the newly added explicit parameters
for core specification and collision control in AUTOSTRUCTURE calculations.
"""

from atomkit import Configuration
from atomkit.autostructure import ASWriter

print("=" * 80)
print("AUTOSTRUCTURE Explicit Parameters Demo")
print("=" * 80)
print()

# =============================================================================
# Example 1: Core Specification with KCOR1/KCOR2
# =============================================================================
print("Example 1: Ne-like Fe with Explicit Core Specification")
print("-" * 80)

# Create Ne-like Fe configurations (10 electrons)
ground = Configuration.from_string("1s2.2s2.2p6")
excited = ground.generate_excitations(["3s", "3p", "3d"], excitation_level=1)

with ASWriter("examples/as_inputs/fe_with_core.dat") as asw:
    asw.write_header("Fe XVII (Ne-like) with explicit Ne core")
    asw.add_comment("Using KCOR1=1, KCOR2=3 to specify 1s2.2s2.2p6 closed core")
    asw.add_blank_line()
    
    # KCOR1=1, KCOR2=3 means orbitals 1 (1s), 2 (2s), 3 (2p) form the core
    # This is important for:
    # - Defining the correlation in structure calculations
    # - Core model potential for R-matrix
    asw.add_salgeb(CUP="IC", RAD="E1", KCOR1=1, KCOR2=3)
    
    info = asw.configs_from_atomkit([ground] + excited[:3], last_core_orbital="2p")
    print(f"  Configurations: {info['n_configs']}")
    print(f"  Core orbitals: {info['core_orbitals']}")
    print(f"  Valence orbitals: {info['valence_orbitals']}")
    
    asw.add_sminim(NZION=26)

print(f"✓ Created: examples/as_inputs/fe_with_core.dat")
print()

# =============================================================================
# Example 2: Photoionization WITHOUT Autoionization
# =============================================================================
print("Example 2: Photoionization with AUGER='NO'")
print("-" * 80)

# Li-like to He-like photoionization
target = Configuration.from_string("1s2")
initial_2s = Configuration.from_string("1s2.2s1")
initial_2p = Configuration.from_string("1s2.2p1")

with ASWriter("examples/as_inputs/pi_no_auger.dat") as asw:
    asw.write_header("Li-like to He-like photoionization")
    asw.add_comment("Using AUGER='NO' to disable autoionization rates")
    asw.add_comment("Only photoionization cross sections will be calculated")
    asw.add_blank_line()
    
    # For pure photoionization, we don't want autoionization rates
    asw.add_salgeb(
        RUN="PI",
        CUP="LS",
        RAD="  ",  # PI implies radiation
        MXCCF=2,
        AUGER="NO",  # Explicitly disable autoionization
        KCOR1=1,
        KCOR2=1,  # 1s core
    )
    
    asw.configs_from_atomkit([target], last_core_orbital=None)
    asw.add_blank_line()
    asw.configs_from_atomkit([initial_2s, initial_2p], last_core_orbital="1s")
    
    asw.add_sminim(NZION=26)
    asw.add_sradcon(MENG=-15, EMIN=0.0, EMAX=1500.0)

print(f"✓ Created: examples/as_inputs/pi_no_auger.dat")
print()

# =============================================================================
# Example 3: Structure with Born Collision Strengths
# =============================================================================
print("Example 3: Structure Calculation with Born Collision Strengths")
print("-" * 80)

# Be-like C structure
ground = Configuration.from_string("1s2.2s2")
excited = ground.generate_excitations(["2p", "3s", "3p"], excitation_level=1)

with ASWriter("examples/as_inputs/structure_with_born.dat") as asw:
    asw.write_header("Be-like C with Born collision strengths")
    asw.add_comment("Using BORN='INF' for infinite energy limit Born strengths")
    asw.add_comment("This is useful for high-temperature plasma applications")
    asw.add_blank_line()
    
    # Born collision strengths require RAD='ALL'
    asw.add_salgeb(
        CUP="IC",
        RAD="ALL",
        BORN="INF",  # Infinite energy limit
        KCOR1=1,
        KCOR2=1,  # 1s core
    )
    
    info = asw.configs_from_atomkit([ground] + excited[:4], last_core_orbital="1s")
    print(f"  Configurations: {info['n_configs']}")
    
    asw.add_sminim(NZION=6)

print(f"✓ Created: examples/as_inputs/structure_with_born.dat")
print()

# =============================================================================
# Example 4: High-Precision with Full Fine-Structure
# =============================================================================
print("Example 4: High-Precision Fe with Full Fine-Structure")
print("-" * 80)

# Fe-like structure for high-precision calculation
ground = Configuration.from_string("1s2.2s2.2p6.3s2.3p6.3d6.4s2")
excited = ground.generate_excitations(["4p"], excitation_level=1)

with ASWriter("examples/as_inputs/fe_high_precision.dat") as asw:
    asw.write_header("Fe with full fine-structure interactions")
    asw.add_comment("KUTSS, KUTSO, KUTOO = 1 for high-precision heavy element")
    asw.add_comment("Important for accurate transition rates in Fe")
    asw.add_blank_line()
    
    # For heavy elements, include all fine-structure interactions
    asw.add_salgeb(
        CUP="IC",
        RAD="E1",
        KUTSS=1,  # spin-spin fully included
        KUTSO=1,  # spin-orbit fully included
        KUTOO=1,  # orbit-orbit fully included
        KCOR1=1,
        KCOR2=3,  # Ne-like core
    )
    
    info = asw.configs_from_atomkit([ground] + excited[:2], last_core_orbital="2p")
    print(f"  Configurations: {info['n_configs']}")
    print(f"  Fine-structure: spin-spin, spin-orbit, orbit-orbit all included")
    
    asw.add_sminim(NZION=26)

print(f"✓ Created: examples/as_inputs/fe_high_precision.dat")
print()

# =============================================================================
# Example 5: LS Coupling with No Fine-Structure
# =============================================================================
print("Example 5: LS Coupling with No Fine-Structure")
print("-" * 80)

# C-like ion in pure LS coupling
ground = Configuration.from_string("1s2.2s2.2p2")
excited = ground.generate_excitations(["3s", "3p"], excitation_level=1)

with ASWriter("examples/as_inputs/ls_no_finestructure.dat") as asw:
    asw.write_header("C-like ion in pure LS coupling")
    asw.add_comment("KUTSS=-1 explicitly disables fine-structure")
    asw.add_comment("Use for light elements or when LS is sufficient")
    asw.add_blank_line()
    
    # For pure LS coupling, explicitly disable fine-structure
    asw.add_salgeb(
        CUP="LS",
        RAD="E1",
        KUTSS=-1,  # No fine-structure in spin-spin
        KCOR1=1,
        KCOR2=1,  # 1s core
    )
    
    info = asw.configs_from_atomkit([ground] + excited[:3], last_core_orbital="1s")
    print(f"  Configurations: {info['n_configs']}")
    print(f"  Coupling: Pure LS (no fine-structure)")
    
    asw.add_sminim(NZION=6)

print(f"✓ Created: examples/as_inputs/ls_no_finestructure.dat")
print()

# =============================================================================
# Example 6: Alternative Core Specification with KORB
# =============================================================================
print("Example 4: Using KORB1/KORB2 Alternative")
print("-" * 80)

ground = Configuration.from_string("1s2.2s2.2p6.3s2")
excited = ground.generate_excitations(["3p", "3d"], excitation_level=1)

with ASWriter("examples/as_inputs/mg_with_korb.dat") as asw:
    asw.write_header("Mg-like ion with KORB specification")
    asw.add_comment("KORB1/KORB2 is alternative to KCOR1/KCOR2")
    asw.add_blank_line()
    
    # Using KORB instead of KCOR (functionally equivalent)
    asw.add_salgeb(CUP="LS", RAD="E1", KORB1=1, KORB2=3)
    
    info = asw.configs_from_atomkit([ground] + excited[:2], last_core_orbital="2p")
    print(f"  Core: {info['core_orbitals']}")
    print(f"  Valence: {info['valence_orbitals']}")
    
    asw.add_sminim(NZION=26)

print(f"✓ Created: examples/as_inputs/mg_with_korb.dat")
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 80)
print("Summary of New Explicit Parameters")
print("=" * 80)
print()
print("Core Specification:")
print("-" * 80)
print("✓ KCOR1/KCOR2: Specify closed core orbitals (orbital indices)")
print("  - Important for correlation and R-matrix calculations")
print("  - Example: KCOR1=1, KCOR2=3 for Ne-like core (1s, 2s, 2p)")
print()
print("✓ KORB1/KORB2: Alternative to KCOR1/KCOR2 (functionally equivalent)")
print()
print("Collision & Autoionization Control:")
print("-" * 80)
print("✓ AUGER: Control autoionization rate calculation")
print("  - 'YES': Calculate rates when continuum present (default)")
print("  - 'NO': Disable autoionization (useful for pure PI)")
print()
print("✓ BORN: Control Born collision strength calculation")
print("  - 'INF': Infinite energy limit (useful for high-T plasmas)")
print("  - 'YES': Finite energy Born strengths")
print("  - 'NO': Don't calculate (default)")
print()
print("Fine-Structure Interaction Control:")
print("-" * 80)
print("✓ KUTSS: Fine-structure in 2-body spin-spin interactions")
print("  - -1: No fine-structure (LS coupling only)")
print("  - 0: Perturbative correction (default)")
print("  - 1: Fully included in interaction matrix")
print("  - 2: Direct integration (most accurate)")
print()
print("✓ KUTSO: Fine-structure in 2-body spin-orbit interactions")
print("  - Same options as KUTSS")
print("  - Important for relativistic effects")
print()
print("✓ KUTOO: Fine-structure in 2-body orbit-orbit interactions")
print("  - Same options as KUTSS")
print("  - Critical for heavy elements and high precision")
print()
print("=" * 80)
print("All these parameters are now explicit in add_salgeb() with full")
print("documentation and type hints, making the code more readable,")
print("self-documenting, and easier to use correctly.")
print("=" * 80)
