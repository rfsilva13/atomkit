"""
Example: Using atomkit's AUTOSTRUCTURE wrapper to generate .dat input files.

This example demonstrates how to use the ASWriter class to generate
AUTOSTRUCTURE input files (.dat format) programmatically using Python,
with automatic conversion from atomkit Configuration objects.

The generated .dat files can be executed with AUTOSTRUCTURE:
    as < input.dat > output.log
"""

from atomkit.autostructure import ASWriter
from atomkit import Configuration

print("=" * 70)
print("Example 1: Basic Structure Calculation - Be-like Carbon")
print("=" * 70)

# Create Be-like Carbon (C2+) configurations
ground = Configuration.from_element("C", ion_charge=2)  # 1s2.2s2
print(f"Ground state: {ground}")

# Generate excited states
excited = ground.generate_excitations(
    target_shells=["3s", "3p", "3d"], excitation_level=1, source_shells=["2s"]
)
print(f"Generated {len(excited)} excited configurations")

# Create AUTOSTRUCTURE input file
with ASWriter("examples/examples/as_inputs/c_belike_structure.dat") as asw:
    asw.write_header("Be-like Carbon structure calculation")
    asw.add_comment("Calculates energy levels and E1 radiative transitions")
    asw.add_blank_line()

    # SALGEB: Use intermediate coupling, include E1 radiation
    asw.add_salgeb(CUP="IC", RAD="E1")

    # Add configurations - automatically converts from atomkit format
    info = asw.configs_from_atomkit(
        [ground] + excited, last_core_orbital="1s"  # Treat 1s as closed core
    )
    print(f"  Valence orbitals: {info['valence_orbitals']}")

    # SMINIM: Carbon has Z=6
    asw.add_sminim(NZION=6)

print(f"✓ Created: examples/as_inputs/c_belike_structure.dat")
print(f"  Execute with: as < examples/as_inputs/c_belike_structure.dat > output.log")
print()

print("=" * 70)
print("Example 2: With Optimization - Be-like Iron")
print("=" * 70)

# Create Be-like Fe configurations
ground_fe = Configuration.from_element(
    "Fe", ion_charge=22
)  # Fe XXII (Be-like: 1s².2s²)

# Generate excitations to 2p and n=3 shells
# This ensures 2p is included for lambda optimization
excited_2p = ground_fe.generate_excitations(
    target_shells=["2p"], excitation_level=1, source_shells=["2s"]
)
excited_n3 = ground_fe.generate_excitations(
    target_shells=["3s", "3p", "3d"], excitation_level=1, source_shells=["2s"]
)
all_excited = excited_2p + excited_n3

with ASWriter("examples/examples/as_inputs/fe_belike_optimized.dat") as asw:
    asw.write_header("Be-like Fe XXII with orbital optimization")
    asw.add_comment("Using optimize_from_orbital='2s' for lambda optimization")
    asw.add_comment("Orbitals from 2s onwards will be explicit; 1s will be core")

    asw.add_salgeb(CUP="LS", RAD="E1")

    # Use optimize_from_orbital='2s' - this automatically:
    # 1. Treats 1s as core (KCOR)
    # 2. Writes 2s, 2p, 3s, 3p, 3d explicitly (available for lambda optimization)
    configs = [ground_fe] + [excited_2p[0]] + excited_n3[:3]
    info = asw.configs_from_atomkit(configs, optimize_from_orbital="2s")

    n_orb = info["n_orbitals"]
    orbital_labels = info["valence_orbitals"]
    print(f"Valence orbitals for optimization: {orbital_labels}")
    print(f"Core orbitals: {info['core_orbitals']}")

    # Optimize: NLAM = number of valence orbitals (5: 2s, 2p, 3s, 3p, 3d)
    #           NVAR = vary all except first (2s used as reference)
    asw.add_sminim(NZION=26, INCLUD=4, NLAM=n_orb, NVAR=n_orb - 1)

    asw.add_blank_line()
    asw.add_comment(
        f"Initial lambda values ({n_orb} orbitals: {', '.join(orbital_labels)}):"
    )
    asw.lines.append("  ".join(["1.0"] * n_orb))
    asw.add_blank_line()
    asw.add_comment(
        f"Vary lambdas 2-{n_orb} ({', '.join(orbital_labels[1:])}, keep {orbital_labels[0]} as reference):"
    )
    asw.lines.append("  ".join(str(i) for i in range(2, n_orb + 1)))

print(f"✓ Created: examples/as_inputs/fe_belike_optimized.dat")
print()

print("=" * 70)
print("Example 3: Photoionization - Li-like to He-like")
print("=" * 70)

# Target: He-like core (after photoionization)
target = Configuration.from_element("Fe", ion_charge=24)  # 1s2 (He-like)
print(f"Photoionization target: {target}")

# Initial states: Li-like (before photoionization)
li_ground = Configuration.from_element("Fe", ion_charge=23)  # 1s2.2s1
li_excited = Configuration.from_string("1s2.2p1")  # Li-like 2p
print(f"Initial states: {li_ground}, {li_excited}")

with ASWriter("examples/examples/as_inputs/fe_photoionization.dat") as asw:
    asw.write_header("Photoionization of Fe XXIII (Li-like) to Fe XXIV (He-like)")
    asw.add_blank_line()

    # RUN='PI' for photoionization
    asw.add_salgeb(RUN="PI", CUP="LS", RAD="  ")

    # Target configuration (He-like core)
    # Initial configurations (Li-like states)
    asw.configs_from_atomkit([target], last_core_orbital="1s")
    asw.add_blank_line()
    asw.configs_from_atomkit([li_ground, li_excited], last_core_orbital="1s")

    asw.add_sminim(NZION=26)

    # Continuum energy grid
    asw.add_sradcon(MENG=-15, EMIN=0.0, EMAX=1500.0)

print(f"✓ Created: examples/as_inputs/fe_photoionization.dat")
print()

print("=" * 70)
print("Example 4: Dielectronic Recombination - Li-like Carbon")
print("=" * 70)

# DR of Li-like C: 1s2 + e- -> 1s.2s.2p (autoionizing) -> 1s2.2s + photon
# Target: He-like C (1s2)
target_c = Configuration.from_element("C", ion_charge=4)  # 1s2

# Autoionizing states: Li-like doubly excited
autoion1 = Configuration.from_string("1s1.2s2")
autoion2 = Configuration.from_string("1s1.2s1.2p1")
autoion3 = Configuration.from_string("1s1.2p2")

with ASWriter("examples/examples/as_inputs/c_dr_calculation.dat") as asw:
    asw.write_header("DR of Li-like Carbon")
    asw.add_comment("KLL resonances: 1s2 + e- -> 1s.2snl -> 1s2 + photon")
    asw.add_blank_line()

    # RUN='DR' with MXCCF for autoionizing configurations
    asw.add_salgeb(RUN="DR", CUP="IC", RAD="  ", MXCCF=3)

    # N-electron target (He-like)
    asw.configs_from_atomkit([target_c], last_core_orbital="1s")
    asw.add_blank_line()

    # (N+1)-electron autoionizing configurations
    asw.configs_from_atomkit([autoion1, autoion2, autoion3])

    # Rydberg series: n=3-15, l=0-7
    asw.add_drr(NMIN=3, NMAX=15, LMIN=0, LMAX=7)

    asw.add_sminim(NZION=6)

    # Continuum energies
    asw.add_sradcon(MENG=-15, EMIN=0.0, EMAX=25.0)

print(f"✓ Created: examples/as_inputs/c_dr_calculation.dat")
print()

print("=" * 70)
print("Example 5: Manual Configuration Input (Advanced)")
print("=" * 70)

# Sometimes you might want manual control
with ASWriter("examples/examples/as_inputs/manual_example.dat") as asw:
    asw.write_header("Manual configuration specification")
    asw.add_blank_line()

    # Specify everything manually
    asw.add_salgeb(CUP="LS", RAD="E1", MXCONF=3, MXVORB=3)

    # Manually add orbitals: 1s, 2s, 2p
    asw.add_orbitals([(1, 0), (2, 0), (2, 1)])

    # Manually add configurations as occupation numbers
    asw.add_configurations(
        [
            [2, 2, 0],  # 1s2.2s2
            [2, 1, 1],  # 1s2.2s1.2p1
            [2, 0, 2],  # 1s2.2p2
        ]
    )

    asw.add_sminim(NZION=6)

print(f"✓ Created: examples/as_inputs/manual_example.dat")
print()

print("=" * 70)
print("Example 6: Previewing Content")
print("=" * 70)

# You can preview content before writing
asw = ASWriter("examples/examples/as_inputs/preview.dat")
asw.write_header("Preview example")
asw.add_salgeb(CUP="LS", RAD="E1")
ground_o = Configuration.from_element("O")
asw.configs_from_atomkit([ground_o])
asw.add_sminim(NZION=8)

content = asw.get_content()
print("Preview of AUTOSTRUCTURE input:")
print("-" * 70)
print(content)
print("-" * 70)
# Don't write this one (or uncomment next line to write)
# asw.close()

print()
print("=" * 70)
print("Summary")
print("=" * 70)
print(
    """
The ASWriter class provides:
  ✓ Pythonic interface to AUTOSTRUCTURE namelists
  ✓ Automatic conversion from atomkit Configuration objects
  ✓ Context manager support for clean file handling
  ✓ Support for all major AUTOSTRUCTURE calculation types
  ✓ Manual mode for advanced users

Generated files can be executed with:
  as < filename.dat > output.log
  
The generated .dat files are plain text and can be:
  - Edited manually if needed
  - Version controlled with git
  - Shared with collaborators
  - Used as templates for similar calculations
  
AUTOSTRUCTURE vs FAC:
  - AS: Non-relativistic (default), NAMELIST format
  - FAC: Fully relativistic, Python-like syntax
  - Both supported by atomkit with similar interfaces!
"""
)
