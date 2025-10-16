# AUTOSTRUCTURE Lambda Optimization Guide

## Overview

Lambda optimization in AUTOSTRUCTURE is a powerful technique for improving the accuracy of atomic structure calculations by allowing the code to variationally optimize radial wavefunction scaling parameters. This guide explains how to use `atomkit`'s `ASWriter` class to set up lambda optimization calculations.

## What are Lambda Parameters?

Lambda parameters scale the radial wavefunctions:

```
Pnl(r) → Pnl(λnl · r)
```

Each valence orbital (n,l) gets its own lambda parameter. By optimizing these parameters, AUTOSTRUCTURE can better represent electron correlation effects beyond the central field approximation.

## The `optimize_from_orbital` Parameter

The key to seamless lambda optimization in `atomkit` is the `optimize_from_orbital` parameter in the `configs_from_atomkit()` method.

### Basic Usage

```python
from atomkit.autostructure import ASWriter
from atomkit import Configuration

# Create Be-like Fe (Fe XXII = Fe22+)
ground = Configuration.from_element("Fe", ion_charge=22)  # 1s².2s²
excited = ground.generate_excitations(["3s", "3p", "3d"], 1)

with ASWriter("fe22_optimized.dat") as asw:
    asw.write_header("Fe XXII with lambda optimization")
    asw.add_salgeb(CUP="LS", RAD="E1")
    
    # This is the key line:
    info = asw.configs_from_atomkit([ground] + excited, optimize_from_orbital="2s")
    
    # Set up optimization
    n_val = info["n_orbitals"]  # Number of valence orbitals
    asw.add_sminim(NZION=26, NLAM=n_val, NVAR=n_val-1, INCLUD=3)
    
    # Provide initial lambda values (all 1.0)
    asw.lines.append("  ".join(["1.0"] * n_val))
    
    # Specify which lambdas to vary (all except the first)
    asw.lines.append("  ".join(str(i) for i in range(2, n_val+1)))
```

### How It Works

When you specify `optimize_from_orbital="2s"`:

1. **Core orbitals**: All orbitals *before* 2s (i.e., 1s) go to the core
2. **Valence orbitals**: All orbitals *from* 2s onwards are written explicitly

**Example**: For Fe XXII with configurations involving 1s, 2s, 3s, 3p, 3d:
- `optimize_from_orbital="2s"` → Core: [1s], Valence: [2s, 3s, 3p, 3d]
- `optimize_from_orbital="3s"` → Core: [1s, 2s, 2p], Valence: [3s, 3p, 3d]

### Why This Matters

AUTOSTRUCTURE's lambda optimization only works on **explicitly listed orbitals**. Core orbitals are not written in the orbital list, so they cannot be optimized. The `optimize_from_orbital` parameter automatically ensures that all orbitals you want to optimize are written explicitly.

**Before** (the old way - manual):
```python
# ❌ Hard to manage - need to manually track which orbitals appear
info = asw.configs_from_atomkit(configs, auto_detect_core=False)
# Then manually figure out which orbitals are in the list
```

**After** (the new way - automatic):
```python
# ✅ Simple and clear - specify where optimization starts
info = asw.configs_from_atomkit(configs, optimize_from_orbital="2s")
```

## Complete Example with Dynamic Parameters

The best practice is to use the returned `info` dictionary to dynamically set up optimization:

```python
from atomkit.autostructure import ASWriter
from atomkit import Configuration

# Create atom/ion
ground = Configuration.from_element("C", ion_charge=2)  # Be-like C
excited = ground.generate_excitations(["3s", "3p", "3d", "4s", "4p"], 1)

with ASWriter("carbon_optimized.dat") as asw:
    asw.write_header("C II with lambda optimization")
    asw.add_salgeb(CUP="IC", RAD="E1")
    
    # Use optimize_from_orbital
    configs = [ground] + excited[:10]
    info = asw.configs_from_atomkit(configs, optimize_from_orbital="2s")
    
    # Extract info
    n_val = info["n_orbitals"]
    val_orbs = info["valence_orbitals"]
    core_orbs = info["core_orbitals"]
    
    print(f"Valence orbitals: {val_orbs}")
    print(f"Core orbitals: {core_orbs}")
    
    # Set up optimization - all parameters computed dynamically
    asw.add_sminim(
        NZION=6,
        NLAM=n_val,           # One lambda per valence orbital
        NVAR=n_val - 1,       # Vary all except one (keep first as reference)
        INCLUD=5              # Use 5 lowest terms for fitting
    )
    
    asw.add_blank_line()
    asw.add_comment(f"Initial lambdas ({n_val} valence orbitals: {', '.join(val_orbs)}):")
    asw.lines.append("  ".join(["1.0"] * n_val))
    
    asw.add_blank_line()
    asw.add_comment(f"Vary lambdas 2-{n_val} (keep first as reference):")
    asw.lines.append("  ".join(str(i) for i in range(2, n_val + 1)))
```

## Key Parameters

### NLAM (Number of Lambdas)
- Must equal the number of explicitly listed valence orbitals
- Use `info["n_orbitals"]` to set this automatically

### NVAR (Number to Vary)
- Typically `NLAM - 1` (keep one lambda fixed as reference, usually the deepest/first orbital)
- Can vary fewer if desired (e.g., only optimize highest orbitals)

### INCLUD (Terms for Fitting)
- Number of lowest-energy terms to include in the energy minimization
- Higher values → better fit but slower
- Typical values: 3-10 depending on configuration space size

## Workflow Summary

1. **Create configurations** including ground and excited states
2. **Use `optimize_from_orbital`** to specify where optimization begins
3. **Extract orbital info** from the returned dictionary
4. **Set NLAM = n_orbitals** (one lambda per valence orbital)
5. **Set NVAR = NLAM - 1** (or fewer if desired)
6. **Provide initial lambdas** (all 1.0 is standard)
7. **Specify which to vary** (typically 2, 3, ..., NLAM)

## When to Use Lambda Optimization

### ✓ Use Lambda Optimization For:
- High-accuracy spectroscopy calculations
- Comparing with experimental energy levels
- Transition rate calculations requiring precise wavefunctions
- Heavy ions where correlation is significant
- Systems with strong configuration interaction

### ✗ Don't Use Lambda Optimization For:
- Quick exploratory calculations (adds computational cost)
- Very large configuration spaces (optimization becomes slow)
- Highly ionized hydrogenic systems (minimal correlation effects)
- Preliminary structural studies

## Advanced: Term-Dependent Optimization

You can optimize for specific terms of interest by adjusting `INCLUD`:

```python
# Optimize specifically for the lowest 7 terms
asw.add_sminim(NZION=8, NLAM=n_val, NVAR=n_val-1, INCLUD=7)
```

This is useful when you need accurate energies for particular atomic states (e.g., metastable levels, specific transitions).

## Comparison with Other Approaches

### Old Approach: `last_core_orbital`
```python
# Specify what goes in core (everything else is valence)
info = asw.configs_from_atomkit(configs, last_core_orbital="1s")
```
- Good for: Normal calculations without lambda optimization
- Problem: May not write all needed orbitals explicitly if they don't appear in configs

### New Approach: `optimize_from_orbital`
```python
# Specify where optimization starts (before = core, from onwards = explicit)
info = asw.configs_from_atomkit(configs, optimize_from_orbital="2s")
```
- Good for: Lambda optimization workflows
- Advantage: Guarantees all optimization orbitals are explicit

## Examples in atomkit

See these example files for complete working examples:

1. **`examples/lambda_optimization_demo.py`**: Comprehensive demonstration with 5 examples
   - Basic Fe XXII optimization
   - Extended C II with multiple shells
   - Heavy ion (Ni-like W)
   - Comparison without optimization
   - Term-dependent optimization

2. **`examples/autostructure_wrapper_demo.py`**: Basic usage including lambda optimization

3. **Generated files**: Check `as_inputs/*.dat` for properly formatted AUTOSTRUCTURE input files

## References

- AUTOSTRUCTURE manual: Sections on NLAM, NVAR, INCLUD parameters
- Badnell, N. R. (2011). "A Breit-Pauli distorted wave implementation for AUTOSTRUCTURE", *Computer Physics Communications*, 182, 1528-1535
- Eissner, W., Jones, M., & Nussbaumer, H. (1974). "Techniques for the calculation of atomic structures and radiative data including relativistic corrections", *Computer Physics Communications*, 8, 270-306

## Support

For issues or questions:
- Check `tests/test_as_writer.py` for unit test examples
- Review `src/atomkit/autostructure/as_writer.py` for implementation details
- See `docs/quick_start.md` for general atomkit usage
