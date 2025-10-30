"""
Example: Using atomkit's FAC wrapper to generate SFAC input files.

This example demonstrates how to use the SFACWriter class to generate
FAC input files (.sf format) programmatically using Python, without
requiring the pfac Python bindings.

The generated .sf files can be executed with FAC command-line tools:
    sfac fe_calculation.sf
    scrm spectrum_calculation.sf
"""

from atomkit.fac import SFACWriter
from atomkit import Configuration

print("=" * 70)
print("Example 1: Basic FAC Calculation - Ne-like Iron")
print("=" * 70)

# Create a FAC input file for Ne-like iron (Fe XVII)
# This reproduces the example from the FAC manual
with SFACWriter("examples/fac_inputs/fe17_structure.sf") as fac:
    fac.add_comment("Ne-like Iron (Fe XVII) structure calculation")
    fac.add_comment("This calculates energy levels and radiative transitions")
    fac.add_comment("between n=2 and n=3 complexes")
    fac.add_blank_line()

    # Set atomic element
    fac.SetAtom("Fe")

    # Define closed shells and configurations
    fac.Closed("1s")
    fac.Config("2*8", group="n2")  # n=2 complex with 8 electrons
    fac.Config("2*7 3*1", group="n3")  # One electron from n=2 to n=3

    # Self-consistent field optimization
    fac.add_blank_line()
    fac.add_comment("Self-consistent optimization")
    fac.ConfigEnergy(0)
    fac.OptimizeRadial(["n2"])
    fac.ConfigEnergy(1)

    # Calculate structure
    fac.add_blank_line()
    fac.add_comment("Calculate energy levels")
    fac.Structure("ne.lev.b", ["n2", "n3"])
    fac.MemENTable("ne.lev.b")  # IMPORTANT: Store in memory for later use
    fac.PrintTable("ne.lev.b", "ne.lev", 1)

    # Calculate transitions
    fac.add_blank_line()
    fac.add_comment("Calculate radiative transitions")
    fac.TransitionTable("ne.tr.b", ["n2"], ["n3"])
    fac.PrintTable("ne.tr.b", "ne.tr", 1)

print(f"✓ Created: examples/fac_inputs/fe17_structure.sf")
print(f"  Execute with: cd examples/fac_inputs && sfac fe17_structure.sf")
print()

print("=" * 70)
print("Example 2: Using atomkit Configurations with FAC")
print("=" * 70)

# Generate configurations using atomkit, then export to FAC
ground = Configuration.from_element("Fe", 23)  # Fe XXIV (Li-like)
print(f"Ground state: {ground}")

# Generate excited states
excited = ground.generate_excitations(
    target_shells=["2s", "2p", "3s", "3p", "3d"],
    excitation_level=1,
    source_shells=["1s", "2s"],
)
print(f"Generated {len(excited)} excited states")

# Generate autoionizing states
autoionizing = Configuration.generate_doubly_excited_autoionizing(
    [ground] + excited[:5], max_n=5, max_l=3
)
print(f"Generated {len(autoionizing)} autoionizing states")

# Create FAC input using these configurations
with SFACWriter("examples/fac_inputs/fe24_autoionization.sf") as fac:
    fac.add_comment("Fe XXIV (Li-like) autoionization calculation")
    fac.add_comment(f"Generated using atomkit Configuration class")
    fac.add_blank_line()

    # Enable QED corrections
    fac.SetAtom("Fe")
    fac.SetBreit(-1)
    fac.SetSE(-1)

    # Add ground state
    fac.add_blank_line()
    fac.add_comment("Target configurations (N-electron)")
    fac.config_from_atomkit(ground, "target0")

    # Add selected excited states
    for i, state in enumerate(excited[:10]):
        fac.config_from_atomkit(state, f"target{i+1}")

    # Add autoionizing configurations
    fac.add_blank_line()
    fac.add_comment("Autoionizing configurations (N+1-electron)")
    for i, state in enumerate(autoionizing[:20]):
        fac.config_from_atomkit(state, f"auto{i}")

    # Build list of target and autoionizing group names
    target_groups = ["target0"] + [f"target{i+1}" for i in range(10)]
    auto_groups = [f"auto{i}" for i in range(20)]

    # Optimization
    fac.add_blank_line()
    fac.add_comment("Optimization")
    fac.ConfigEnergy(0)
    fac.OptimizeRadial(["target0"])
    fac.ConfigEnergy(1)

    # Structure calculation
    fac.add_blank_line()
    fac.add_comment("Calculate energy levels")
    fac.Structure("fe24.lev.b", target_groups + auto_groups)
    fac.MemENTable("fe24.lev.b")  # IMPORTANT: Store in memory for later use
    fac.PrintTable("fe24.lev.b", "fe24.lev.asc", 1)

    # Transitions between autoionizing states
    fac.add_blank_line()
    fac.add_comment("Radiative transitions in autoionizing manifold")
    fac.TransitionTable("fe24_ai.tr.b", auto_groups, auto_groups)
    fac.PrintTable("fe24_ai.tr.b", "fe24_ai.tr.asc", 1)

    # Autoionization rates
    fac.add_blank_line()
    fac.add_comment("Autoionization rates")
    fac.AITable("fe24_ai.ai.b", auto_groups, target_groups)
    fac.PrintTable("fe24_ai.ai.b", "fe24_ai.ai.asc", 1)

print(f"✓ Created: examples/fac_inputs/fe24_autoionization.sf")
print(f"  Execute with: cd examples/fac_inputs && sfac fe24_autoionization.sf")
print()

print("=" * 70)
print("Example 3: Parallel Calculation with MPI")
print("=" * 70)

# Large-scale calculation with MPI parallelization
with SFACWriter("examples/fac_inputs/cu_photoionization.sf") as fac:
    fac.add_comment("Cu I photoionization - parallel calculation")
    fac.add_blank_line()

    # Initialize MPI
    fac.InitializeMPI(24)  # Use 24 cores

    fac.SetAtom("Cu")
    fac.Closed("1s 2s 2p 3s 3p")

    # Ground and low-lying states
    fac.Config("3d10 4s1", group="ground")
    fac.Config("3d10 4p1", group="4p")
    fac.Config("3d10 5s1", group="5s")
    fac.Config("3d10 4d1", group="4d")
    fac.Config("3d9 4s2", group="d9s2")

    # Continuum configurations
    fac.Config("3d10 k1", group="cont_s")  # 4s -> continuum
    fac.Config("3d9 4s1 k1", group="cont_d")  # 3d -> continuum

    # Optimization
    fac.ConfigEnergy(0)
    fac.OptimizeRadial(["ground"])
    fac.ConfigEnergy(1)

    # Structure
    bound_groups = ["ground", "4p", "5s", "4d", "d9s2"]
    cont_groups = ["cont_s", "cont_d"]

    fac.Structure("cu.lev.b", bound_groups + cont_groups)
    fac.MemENTable("cu.lev.b")  # IMPORTANT: Store in memory for later use
    fac.PrintTable("cu.lev.b", "cu.lev.asc", 0)

    # Set energy grid for photoionization
    fac.add_blank_line()
    fac.add_comment("Energy grid for photoionization (in Rydberg)")
    energy_grid = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    fac.SetUsrPEGrid(energy_grid)

    # Photoionization cross sections
    fac.add_blank_line()
    fac.add_comment("Photoionization cross sections")
    fac.RRTable("cu.rr.b", bound_groups, cont_groups)
    fac.PrintTable("cu.rr.b", "cu.rr.asc", 0)

    # Finalize MPI
    fac.FinalizeMPI()

print(f"✓ Created: examples/fac_inputs/cu_photoionization.sf")
print(f"  Execute with: cd examples/fac_inputs && mpirun -n 24 sfac cu_photoionization.sf")
print()

print("=" * 70)
print("Example 4: Previewing SFAC Content")
print("=" * 70)

# You can preview the content before writing to file
fac = SFACWriter("examples/fac_inputs/preview.sf")
fac.SetAtom("O")
fac.Closed("1s")
fac.Config("2s2 2p4", group="ground")
fac.Config("2s1 2p5", group="excited")

# Get the content as a string
content = fac.get_content()
print("Preview of SFAC file content:")
print("-" * 70)
print(content)
print("-" * 70)

# Close without writing (we just wanted to preview)
# Or uncomment the next line to actually write it
# fac.close()

print()
print("=" * 70)
print("Summary")
print("=" * 70)
print(
    """
The SFACWriter class provides:
  ✓ Pythonic interface to FAC functions
  ✓ Automatic SFAC syntax generation
  ✓ Integration with atomkit Configuration objects
  ✓ Context manager support for clean file handling
  ✓ Comments and formatting for readable output
  ✓ All major FAC functions implemented

Generated files can be executed with:
  - sfac filename.sf          (serial execution)
  - mpirun -n N sfac file.sf  (parallel execution)
  
The generated .sf files are plain text and can be:
  - Edited manually if needed
  - Version controlled with git
  - Shared with collaborators
  - Used as templates for similar calculations
"""
)
