"""
Advanced Example: Lambda Optimization in AUTOSTRUCTURE

This example demonstrates how to use orbital scaling parameter (lambda) optimization
in AUTOSTRUCTURE calculations for improved accuracy. Lambda parameters scale the
radial wavefunctions to better represent correlation effects.

For details, see AUTOSTRUCTURE manual sections on NLAM and NVAR parameters.
"""

from atomkit.autostructure import ASWriter
from atomkit import Configuration

print("=" * 80)
print("AUTOSTRUCTURE Lambda Optimization Examples")
print("=" * 80)
print()

# =============================================================================
# Example 1: Basic Lambda Optimization - Fe XXII (Be-like Iron)
# =============================================================================
print("Example 1: Basic Lambda Optimization - Fe XXII")
print("-" * 80)

# Create Be-like Fe (Fe22+) configurations
ground_fe = Configuration.from_element("Fe", ion_charge=22)  # 1s2.2s2
print(f"Ground state: {ground_fe}")

# Generate excited states for optimization
excited_fe = ground_fe.generate_excitations(
    target_shells=["3s", "3p", "3d"], excitation_level=1, source_shells=["2s"]
)
print(f"Generated {len(excited_fe)} excited configurations")

with ASWriter("examples/examples/as_inputs/fe22_lambda_basic.dat") as asw:
    asw.write_header("Fe XXII with lambda optimization")
    asw.add_comment("Be-like iron (Fe22+) with orbital scaling parameter optimization")
    asw.add_comment("")
    asw.add_comment("Lambda parameters scale orbital wavefunctions:")
    asw.add_comment("  Pnl(r) -> Pnl(lambda_nl * r)")
    asw.add_blank_line()

    # SALGEB: LS coupling, E1 radiation
    asw.add_salgeb(CUP="LS", RAD="E1")

    # Add configurations (ground + first 2 excited)
    configs = [ground_fe] + excited_fe[:2]
    # Use optimize_from_orbital to automatically handle core/valence split
    # 1s -> core (implicit), 2s onwards -> explicit valence for optimization
    info = asw.configs_from_atomkit(configs, optimize_from_orbital="2s")
    print(f"Valence orbitals: {info['valence_orbitals']}")
    print(f"Core orbitals: {info['core_orbitals']}")

    # SMINIM with lambda optimization:
    # INCLUD=3: Include lowest 3 terms in energy minimization
    n_val = info["n_orbitals"]
    asw.add_sminim(NZION=26, INCLUD=3, NLAM=n_val, NVAR=n_val - 1)

    asw.add_blank_line()
    asw.add_comment(f"Initial lambda values ({n_val} valence orbitals):")
    # Initial lambdas: start near 1.0 for each orbital
    asw.lines.append("  ".join(["1.0"] * n_val))
    asw.add_blank_line()
    asw.add_comment(f"Vary lambdas 2-{n_val} (keep 1st fixed as reference):")
    vary_indices = "  ".join(str(i) for i in range(2, n_val + 1))
    asw.lines.append(vary_indices)
    orb_labels = ", ".join(info["valence_orbitals"])
    asw.add_comment(f"Valence orbitals: {orb_labels}")

print(f"✓ Created: examples/as_inputs/fe22_lambda_basic.dat")
print()

# =============================================================================
# Example 2: Advanced Lambda Optimization - Multiple Excitations
# =============================================================================
print("Example 2: Advanced Lambda Optimization with Multiple Excitations")
print("-" * 80)

# Create more complex configuration set
ground_c = Configuration.from_element("C", ion_charge=2)  # 1s2.2s2 (Be-like)
print(f"Ground state: {ground_c}")

# Generate excitations to n=3,4 shells
excited_c = ground_c.generate_excitations(
    target_shells=["3s", "3p", "3d", "4s", "4p"],
    excitation_level=1,
    source_shells=["2s"],
)
print(f"Generated {len(excited_c)} excited configurations")

with ASWriter("examples/examples/as_inputs/c_belike_lambda_advanced.dat") as asw:
    asw.write_header("C II with extended lambda optimization")
    asw.add_comment("Be-like carbon with optimization over n=2,3,4 orbitals")
    asw.add_blank_line()

    # Intermediate coupling for better accuracy
    asw.add_salgeb(CUP="IC", RAD="E1")

    # Use ground + first 10 excited states
    configs = [ground_c] + excited_c[:10]
    info = asw.configs_from_atomkit(configs, optimize_from_orbital="2s")
    n_valence = info["n_orbitals"]
    print(f"Number of valence orbitals: {n_valence}")
    print(f"Valence orbitals: {info['valence_orbitals']}")
    print(f"Core orbitals: {info['core_orbitals']}")

    # More aggressive optimization:
    # INCLUD=5: Use 5 lowest terms for fitting
    # NLAM=n_valence: One lambda per valence orbital
    # NVAR=n_valence-1: Vary all except one (usually keep 2s fixed as reference)
    asw.add_sminim(NZION=6, INCLUD=5, NLAM=n_valence, NVAR=n_valence - 1)

    asw.add_blank_line()
    asw.add_comment(f"Initial lambda values ({n_valence} orbitals):")
    # Start all lambdas at 1.0
    initial_lambdas = "  ".join(["1.0"] * n_valence)
    asw.lines.append(initial_lambdas)

    asw.add_blank_line()
    asw.add_comment(f"Vary lambdas 2 through {n_valence} (keep first fixed):")
    # Vary all except the first (2s orbital)
    vary_indices = "  ".join(str(i) for i in range(2, n_valence + 1))
    asw.lines.append(vary_indices)

print(f"✓ Created: examples/as_inputs/c_belike_lambda_advanced.dat")
print()

# =============================================================================
# Example 3: Lambda Optimization for Heavy Ions
# =============================================================================
print("Example 3: Lambda Optimization for Heavy Ions (Ni-like W)")
print("-" * 80)

# Ni-like W (W46+) - important for fusion plasmas
ground_w = Configuration.from_element("W", ion_charge=46)  # [Ar].3d10
print(f"Ground state: {ground_w}")

# Single excitation from 3d to 4s, 4p, 4d, 4f
excited_w = ground_w.generate_excitations(
    target_shells=["4s", "4p", "4d", "4f"], excitation_level=1, source_shells=["3d"]
)
print(f"Generated {len(excited_w)} excited configurations")

with ASWriter("examples/examples/as_inputs/w46_nilike_lambda.dat") as asw:
    asw.write_header("Ni-like W with lambda optimization")
    asw.add_comment("W XLVI (Ni-like tungsten) - 3d-4l transitions")
    asw.add_comment("Important for tokamak diagnostics")
    asw.add_blank_line()

    # For heavy ions, IC or jj coupling often better
    # But here we use LS as example
    asw.add_salgeb(CUP="LS", RAD="E1")

    configs = [ground_w] + excited_w[:5]
    info = asw.configs_from_atomkit(configs, optimize_from_orbital="3d")
    print(f"Valence orbitals: {info['valence_orbitals']}")
    print(f"Core orbitals: {info['core_orbitals']}")

    # For heavy ions, optimization is crucial for accuracy
    # Include more terms for better fitting
    asw.add_sminim(
        NZION=74,  # Z=74 for W
        INCLUD=10,  # Include 10 lowest terms
        NLAM=info["n_orbitals"],
        NVAR=info["n_orbitals"] - 1,
        NPRN=1,  # Print extra information
    )

    asw.add_blank_line()
    asw.add_comment("Initial lambdas (starting values):")
    n_orb = info["n_orbitals"]
    asw.lines.append("  ".join(["1.0"] * n_orb))

    asw.add_blank_line()
    asw.add_comment("Vary all lambdas except 3d (keep as reference):")
    # Vary all lambdas except the first (3d) - keep 3d as reference
    asw.lines.append("  ".join(str(i) for i in range(2, n_orb + 1)))

print(f"✓ Created: examples/as_inputs/w46_nilike_lambda.dat")
print()

# =============================================================================
# Example 4: No Optimization (Comparison Case)
# =============================================================================
print("Example 4: Without Lambda Optimization (for comparison)")
print("-" * 80)

# Same Fe XXII system but without optimization
with ASWriter("examples/examples/as_inputs/fe22_no_optimization.dat") as asw:
    asw.write_header("Fe XXII without lambda optimization")
    asw.add_comment("Reference calculation (Be-like Fe) - all lambdas = 1.0")
    asw.add_blank_line()

    asw.add_salgeb(CUP="LS", RAD="E1")

    configs = [ground_fe] + excited_fe[:2]
    asw.configs_from_atomkit(configs, optimize_from_orbital="2s")

    # No NLAM/NVAR specified - all lambdas default to 1.0
    asw.add_sminim(NZION=26)

print(f"✓ Created: examples/as_inputs/fe22_no_optimization.dat")
print()

# =============================================================================
# Example 5: Term-Dependent Optimization
# =============================================================================
print("Example 5: Term-Dependent Lambda Optimization")
print("-" * 80)

# More sophisticated: different lambdas for different terms
ground_o = Configuration.from_element("O")  # 1s2.2s2.2p4
excited_o = ground_o.generate_excitations(
    target_shells=["3s", "3p", "3d"], excitation_level=1
)

with ASWriter("examples/examples/as_inputs/o_term_dependent.dat") as asw:
    asw.write_header("O I with term-dependent lambda optimization")
    asw.add_comment("Optimize lambdas for specific terms of interest")
    asw.add_blank_line()

    asw.add_salgeb(CUP="LS", RAD="E1")

    configs = [ground_o] + excited_o[:8]
    info = asw.configs_from_atomkit(configs, optimize_from_orbital="2s")

    # Optimize for specific terms (e.g., lowest 7 terms)
    # This is useful when you want accurate energies for particular states
    asw.add_sminim(
        NZION=8,
        INCLUD=7,  # Use 7 terms for optimization
        NLAM=info["n_orbitals"],
        NVAR=info["n_orbitals"] - 2,  # Keep 2 lambdas fixed
        NPRN=2,  # Print detailed information
    )

    asw.add_blank_line()
    asw.add_comment("Initial lambda values:")
    asw.lines.append("  ".join(["1.0"] * info["n_orbitals"]))

    asw.add_blank_line()
    asw.add_comment("Vary middle orbitals, fix innermost and outermost:")
    # Vary only middle orbitals (e.g., indices 2 to n-1)
    if info["n_orbitals"] > 3:
        vary = "  ".join(str(i) for i in range(2, info["n_orbitals"]))
        asw.lines.append(vary)

    asw.add_blank_line()
    asw.add_comment("After optimization, AUTOSTRUCTURE will output:")
    asw.add_comment("  - Optimized lambda values")
    asw.add_comment("  - Energy improvement compared to lambda=1.0")
    asw.add_comment("  - Term energies with optimized wavefunctions")

print(f"✓ Created: examples/as_inputs/o_term_dependent.dat")
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 80)
print("Summary: Lambda Optimization in AUTOSTRUCTURE")
print("=" * 80)
print(
    """
Lambda Parameters:
  - Scale radial wavefunctions: Pnl(r) -> Pnl(lambda_nl * r)
  - Account for correlation effects beyond central field approximation
  - Typically one lambda per valence orbital (n,l)
  - Starting values usually near 1.0

Key Parameters:
  NLAM  - Number of lambda parameters (= number of valence orbitals)
  NVAR  - Number of lambdas to vary in optimization (typically NLAM-1)
  INCLUD - Number of lowest terms to include in fitting

Workflow:
  1. Set NLAM = number of valence orbitals
  2. Set NVAR = NLAM-1 (keep one lambda fixed as reference)
  3. Provide initial lambda values (start with 1.0 for all)
  4. Specify which lambdas to vary (by index)
  5. Run AUTOSTRUCTURE - it will optimize lambdas to minimize energy
  6. Use optimized lambdas for production calculations

When to Use Lambda Optimization:
  ✓ High-accuracy calculations (spectroscopy, transition rates)
  ✓ Comparing with experimental data
  ✓ Heavy ions where correlation is important
  ✓ Systems with strong configuration interaction

When NOT to Use:
  ✗ Quick exploratory calculations
  ✗ Very large configuration spaces (too slow)
  ✗ Highly ionized systems (hydrogenic - no correlation)

Files generated:
  - fe22_lambda_basic.dat          : Basic Fe XXII (Be-like) optimization
  - c_belike_lambda_advanced.dat   : Extended optimization with n=3,4
  - w46_nilike_lambda.dat          : Heavy ion optimization
  - fe22_no_optimization.dat       : Reference (no optimization)
  - o_term_dependent.dat           : Optimizing for specific terms

Execute with: as < filename.dat > output.log

Check output for:
  - Optimized lambda values
  - Energy convergence
  - Comparison with unoptimized energies
"""
)
