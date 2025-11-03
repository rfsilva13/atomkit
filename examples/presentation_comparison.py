"""
Presentation Example: Traditional vs AtomKit Approach
======================================================

This example demonstrates the key advantage of AtomKit:
Write physics ONCE, run on multiple codes.

Perfect for presentations showing:
1. Traditional FAC input (code-specific syntax)
2. Traditional AUTOSTRUCTURE input (different code-specific syntax)
3. AtomKit unified approach (physics-first, code-agnostic)

Example Problem: Fe XVII radiative transitions
- Ground configuration: 1s² 2s² 2p⁶ (Ne-like)
- Excited configuration: 1s² 2s² 2p⁵ 3s¹ (2p → 3s excitation)
- Calculate E1 transitions with intermediate coupling
"""

# ==============================================================================
# TRADITIONAL APPROACH: FAC (Flexible Atomic Code)
# ==============================================================================
print("=" * 80)
print("TRADITIONAL FAC INPUT (.sf file)")
print("=" * 80)

traditional_fac_input = '''
# Fe XVII calculation using FAC
# Different syntax, different concepts, different workflow

SetAtom('Fe')
# FAC uses charge state
Config('1s2 2s2 2p6', group='ground')
Config('1s2 2s2 2p5 3s1', group='excited')

# FAC-specific optimization and structure calculation
ConfigEnergy(0)
OptimizeRadial(['ground'])
ConfigEnergy(1)

# FAC output format
Structure('fe17.lev.b', ['ground', 'excited'])
MemENTable('fe17.lev.b')
TransitionTable('fe17.tr.b', ['ground'], ['excited'])
PrintTable('fe17.lev.b', 'fe17.lev', 1)
PrintTable('fe17.tr.b', 'fe17.tr', 1)
'''

print(traditional_fac_input)
print("\n⚠️  FAC-specific syntax and concepts")
print("⚠️  Must learn FAC functions and workflow")
print("⚠️  Output in FAC binary format (.b files)")

# ==============================================================================
# TRADITIONAL APPROACH: AUTOSTRUCTURE
# ==============================================================================
print("\n" + "=" * 80)
print("TRADITIONAL AUTOSTRUCTURE INPUT (das file)")
print("=" * 80)

traditional_as_input = '''
Fe XVII radiative calculation
 &SALGEB CUP='IC' RAD='E1' MXVORB=4 MXCONF=2 &END
 1 0  2 0  2 1  3 0  3 1
  2    2    6    0    0
  2    2    5    1    0
 &SMINIM NZION=26 &END
'''

print(traditional_as_input)
print("\n⚠️  AUTOSTRUCTURE-specific format")
print("⚠️  Cryptic notation: orbital occupation numbers")
print("⚠️  Format: nl pairs (1 0 = 1s, 2 0 = 2s, 2 1 = 2p, etc.)")
print("⚠️  Then occupation per config: '2 2 6 0 0' = 1s² 2s² 2p⁶")
print("⚠️  Must learn namelist structure (&SALGEB, &SMINIM)")
print("⚠️  Completely different from FAC!")

# ==============================================================================
# ATOMKIT APPROACH: Unified, Code-Agnostic
# ==============================================================================
print("\n" + "=" * 80)
print("ATOMKIT UNIFIED APPROACH (Python)")
print("=" * 80)

from atomkit.core import AtomicCalculation
from atomkit import Configuration

# Define physics ONCE in clear, physical terms
calc = AtomicCalculation(
    element="Fe",
    charge=16,                          # Fe XVII (Ne-like)
    
    # Physics definition (code-agnostic!)
    calculation_type="radiative",       # What we want to compute
    coupling="IC",                       # Intermediate coupling
    relativistic="Breit",                # Breit-Pauli Hamiltonian
    radiation_types=["E1"],              # Electric dipole transitions
    
    # Configurations in standard notation
    configurations=[
        Configuration.from_string("1s2 2s2 2p6"),        # Ground
        Configuration.from_string("1s2 2s2 2p5 3s1"),    # Excited
    ],
    
    # Code selection (ONLY code-specific choice!)
    code="autostructure"  # or "fac" - just change this!
)

print("\n# STEP 1: Define physics once (code-agnostic)")
print("-" * 80)
print("calc = AtomicCalculation(")
print("    element='Fe',")
print("    charge=16,                          # Fe XVII")
print("    calculation_type='radiative',       # What physics?")
print("    coupling='IC',                       # IC/LS/jj coupling")
print("    relativistic='Breit',                # Breit-Pauli")
print("    radiation_types=['E1'],              # E1 transitions")
print("    configurations=[")
print("        Configuration.from_string('1s2 2s2 2p6'),    # Ground")
print("        Configuration.from_string('1s2 2s2 2p5 3s1'),# Excited")
print("    ],")
print("    code='autostructure'  # ← Only code-specific parameter!")
print(")")

# Generate AUTOSTRUCTURE input
print("\n# STEP 2a: Generate AUTOSTRUCTURE input")
print("-" * 80)
calc.code = "autostructure"
as_file = calc.write_input()
print(f"✅ Generated AUTOSTRUCTURE input: {as_file}")
print("   → Same physics, translated to AS namelist format")

# Generate FAC input by changing ONE parameter!
print("\n# STEP 2b: Generate FAC input (change ONE parameter!)")
print("-" * 80)
calc.code = "fac"
fac_file = calc.write_input()
print(f"✅ Generated FAC input: {fac_file}")
print("   → Same physics, translated to FAC function calls")

# ==============================================================================
# THE ATOMKIT ADVANTAGE
# ==============================================================================
print("\n" + "=" * 80)
print("THE ATOMKIT ADVANTAGE")
print("=" * 80)

print("""
✨ DEFINE PHYSICS ONCE, RUN ANYWHERE ✨

Traditional Approach:
──────────────────────
❌ Learn FAC syntax and concepts
❌ Write FAC input file (fe17.sf)
❌ Learn AUTOSTRUCTURE syntax (completely different!)
❌ Rewrite entire input in AS format (fe17.d)
❌ Write custom parsers for each code's output
❌ Cannot easily compare results

AtomKit Approach:
─────────────────
✅ Define physics in clear Python (one time!)
✅ Generate FAC input automatically
✅ Generate AS input automatically (same code!)
✅ Change calc.code = "grasp" for GRASP
✅ Unified output parsing (same DataFrame!)
✅ Direct comparison and validation

PRODUCTIVITY MULTIPLIER:
────────────────────────
→ Switch codes by changing ONE parameter
→ Same analysis pipeline for all codes
→ Easy validation and benchmarking
→ Focus on physics, not syntax!

REAL-WORLD WORKFLOW:
───────────────────
""")

# Show complete workflow
print("# Complete atomkit workflow example:")
print("-" * 80)

workflow_example = """
from atomkit.core import AtomicCalculation
from atomkit.readers import read_levels, read_transitions
from atomkit.physics.plotting import plot_energy_levels, plot_grotrian_diagram
from atomkit.export import to_chianti_format, to_latex_table

# 1. Define physics (code-agnostic)
calc = AtomicCalculation(
    element="Fe", charge=16,
    calculation_type="radiative",
    coupling="IC",
    configurations=[...],
    code="autostructure"
)

# 2. Run on multiple codes
for code in ["autostructure", "fac"]:
    calc.code = code
    input_file = calc.write_input(f"fe17_{code}")
    # Run the code (external)
    # $ autostructure < fe17_autostructure.d
    # $ sfac fe17_fac.sf
    
# 3. Analyze results (unified!)
as_levels = read_levels("autostructure_output.lev")
fac_levels = read_levels("fac_output.lev.asc")

# Both are pandas DataFrames with same columns!
print("AUTOSTRUCTURE:", len(as_levels), "levels")
print("FAC:", len(fac_levels), "levels")

# 4. Compare and validate
comparison = compare_energy_levels(as_levels, fac_levels)
plot_comparison(comparison, save="as_vs_fac.pdf")

# 5. Export for other platforms
to_chianti_format(as_levels, transitions, "fe_17.elvlc")
to_latex_table(as_levels, "table1.tex")
"""

print(workflow_example)

# ==============================================================================
# SUMMARY TABLE FOR SLIDES
# ==============================================================================
print("\n" + "=" * 80)
print("COMPARISON SUMMARY (Perfect for presentation slides)")
print("=" * 80)

summary = """
┌─────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ Aspect              │ Traditional FAC      │ Traditional AS       │ AtomKit              │
├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Syntax              │ FAC-specific         │ Namelist format      │ Python (physics)     │
│                     │ SetAtom(), Config()  │ &RUN, &CFG, &STG     │ AtomicCalculation    │
├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Learning Curve      │ Must learn FAC       │ Must learn AS        │ Learn once, use all  │
│                     │ manual & functions   │ namelists & keywords │ codes                │
├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Switching Codes     │ Rewrite entire input │ Rewrite entire input │ Change 1 parameter   │
│                     │ in AS format         │ in FAC format        │ code="fac"/"as"      │
├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Output Parsing      │ Custom FAC parser    │ Custom AS parser     │ Unified DataFrame    │
│                     │ (.lev.asc format)    │ (.j file format)     │ (all codes)          │
├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Code Comparison     │ Difficult - need to  │ Difficult - need to  │ Easy - same format   │
│                     │ match different fmt  │ match different fmt  │ direct comparison    │
├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Reproducibility     │ FAC input only       │ AS input only        │ Physics definition   │
│                     │                      │                      │ works for all codes  │
├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Export Formats      │ Manual conversion    │ Manual conversion    │ Built-in: CHIANTI,   │
│                     │                      │                      │ ADAS, LaTeX, etc.    │
└─────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘

KEY TAKEAWAY FOR PRESENTATION:
══════════════════════════════
"AtomKit lets you write the physics once in clear Python,
 then automatically generates input for any atomic code.
 Switch codes by changing ONE parameter!"
"""

print(summary)

# ==============================================================================
# VISUAL COMPARISON FOR SLIDES
# ==============================================================================
print("\n" + "=" * 80)
print("VISUAL WORKFLOW (for slides)")
print("=" * 80)

print("""
TRADITIONAL WORKFLOW:
═════════════════════

    [Your Physics Problem: Fe XVII transitions]
              │
              ├─────────────────────┬─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  Learn FAC       │  │  Learn AS        │  │  Learn GRASP     │
    │  Write .sf file  │  │  Write .d file   │  │  Write .in file  │
    │  (SetAtom, etc.) │  │  (&RUN, &CFG)    │  │  (different!)    │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                     │
             ▼                     ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  Run FAC         │  │  Run AS          │  │  Run GRASP       │
    │  sfac fe17.sf    │  │  autos < fe17.d  │  │  grasp fe17.in   │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                     │
             ▼                     ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │ Parse FAC output │  │ Parse AS output  │  │ Parse GRASP out  │
    │ .lev.asc format  │  │ .j file format   │  │ .mixing format   │
    │ (custom parser)  │  │ (custom parser)  │  │ (custom parser)  │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                     │
             └─────────────────────┴─────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────┐
                    │  Manually combine results │
                    │  (different formats!)     │
                    └───────────────────────────┘


ATOMKIT WORKFLOW:
═════════════════

    [Your Physics Problem: Fe XVII transitions]
              │
              ▼
    ┌──────────────────────────────────────────┐
    │  Define ONCE in AtomKit (Python)         │
    │  AtomicCalculation(                      │
    │      element="Fe", charge=16,            │
    │      coupling="IC", radiation=["E1"],    │
    │      configurations=[...],               │
    │      code="autostructure"  ← CHANGE ME!  │
    │  )                                       │
    └────────┬─────────────────────────────────┘
             │
             ├─────────────────────┬─────────────────────┐
             │                     │                     │
             ▼                     ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  Generate        │  │  Generate        │  │  Generate        │
    │  FAC input       │  │  AS input        │  │  GRASP input     │
    │  (automatic!)    │  │  (automatic!)    │  │  (automatic!)    │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                     │
             ▼                     ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  Run FAC         │  │  Run AS          │  │  Run GRASP       │
    │  sfac fe17.sf    │  │  autos < fe17.d  │  │  grasp fe17.in   │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                     │
             ▼                     ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │ read_levels()    │  │ read_levels()    │  │ read_levels()    │
    │ (same function!) │  │ (same function!) │  │ (same function!) │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                     │
             └─────────────────────┴─────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │  All results in same format   │
                    │  (pandas DataFrame)           │
                    │  → Direct comparison!         │
                    │  → Unified analysis!          │
                    │  → Easy export (CHIANTI/ADAS) │
                    └───────────────────────────────┘

RESULT: 10x more productive, 100% reproducible!
""")

print("\n" + "=" * 80)
print("END OF PRESENTATION EXAMPLE")
print("=" * 80)
print("\nThis file demonstrates the core AtomKit value proposition:")
print("  • Write physics definitions once")
print("  • Run on multiple atomic codes")
print("  • Unified analysis pipeline")
print("  • Easy validation and comparison")
print("\nPerfect for presentations, papers, and teaching!")
