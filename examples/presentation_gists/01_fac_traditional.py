"""
Traditional FAC Input for Fe XVII Complex CI Calculation
=========================================================
File: fe17_complex.sf

This shows the TRADITIONAL way to set up a multi-configuration
calculation in FAC, including:
- 25+ configurations manually listed
- Breit interaction enabled
- QED corrections (vacuum polarization, self-energy)

⚠️  NOTE: This file contains FAC syntax, not executable Python code!
   The functions below (SetAtom, SetBreit, etc.) are FAC commands,
   not Python functions. This is for demonstration purposes only.
"""

# FAC input file content (would be saved as .sf file):
fac_input_content = """
SetAtom("Fe")

# Enable relativistic corrections
SetBreit(-1)  # Breit interaction
SetVP(-1)  # Vacuum polarization (QED)
SetSE(-1)  # Self-energy correction (QED)

# Ground configuration
Config("1s2 2s2 2p6", group="ground")

# Single excitations: 2p -> n=3 (3 configs)
Config("1s2 2s2 2p5 3s1", group="n3_single")
Config("1s2 2s2 2p5 3p1", group="n3_single")
Config("1s2 2s2 2p5 3d1", group="n3_single")

# Single excitations: 2p -> n=4 (4 configs)
Config("1s2 2s2 2p5 4s1", group="n4_single")
Config("1s2 2s2 2p5 4p1", group="n4_single")
Config("1s2 2s2 2p5 4d1", group="n4_single")
Config("1s2 2s2 2p5 4f1", group="n4_single")

# Core excitations: 2s,2p -> 3s,3p,3d (6 configs)
Config("1s2 2s1 2p5 3s2", group="core_excited")
Config("1s2 2s1 2p5 3s1 3p1", group="core_excited")
Config("1s2 2s1 2p5 3s1 3d1", group="core_excited")
Config("1s2 2s1 2p5 3p2", group="core_excited")
Config("1s2 2s1 2p5 3p1 3d1", group="core_excited")
Config("1s2 2s1 2p5 3d2", group="core_excited")

# Correlation: 2p^2 -> nl^2 (3 configs)
Config("1s2 2s2 2p4 3s2", group="correlation")
Config("1s2 2s2 2p4 3p2", group="correlation")
Config("1s2 2s2 2p4 3d2", group="correlation")

# ... (would need 8+ more for complete CI)

# Total: ~25 configurations manually entered!

# Optimization
ConfigEnergy(0)
OptimizeRadial(["ground"])
ConfigEnergy(1)

# Structure calculation with all groups
groups = ["ground", "n3_single", "n4_single", "core_excited", "correlation"]
Structure("fe17.lev.b", groups)
MemENTable("fe17.lev.b")

# Transition calculations
TransitionTable(
    "fe17.tr.b", ["ground"], ["n3_single", "n4_single", "core_excited", "correlation"]
)

# Print ASCII output
PrintTable("fe17.lev.b", "fe17.lev", 1)
PrintTable("fe17.tr.b", "fe17.tr", 1)
"""

print("Traditional FAC Input (.sf file):")
print("=" * 50)
print(fac_input_content.strip())
print("=" * 50)

print(
    """
❌ PROBLEMS:
  • 60+ lines for 25 configurations
  • Must manually list EVERY configuration
  • Easy to miss configurations
  • Tedious to modify (want n=5? add 10+ more lines!)
  • No validation (typos won't be caught)
"""
)
