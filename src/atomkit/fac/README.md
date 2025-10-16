# FAC Integration

This module provides a Python wrapper for generating FAC (Flexible Atomic Code) input files.

## Overview

Instead of requiring the `pfac` Python bindings, atomkit provides the `SFACWriter` class that generates SFAC (Simple FAC) format input files (`.sf` extension) programmatically. These files can be executed by FAC command-line tools (`sfac`, `scrm`, `spol`).

## Key Features

- **No pfac dependency**: Generate FAC inputs without needing to compile pfac
- **Pythonic interface**: Clean, modern Python API
- **Atomkit integration**: Seamlessly use atomkit `Configuration` objects
- **Context manager**: Automatic file handling
- **Type hints**: Full IDE support and type checking
- **Comments and formatting**: Generate readable, well-documented input files

## Quick Start

### Basic Example

```python
from atomkit.fac import SFACWriter

with SFACWriter("calculation.sf") as fac:
    fac.SetAtom("Fe")
    fac.Closed("1s")
    fac.Config("2*8", group="n2")
    fac.Config("2*7 3*1", group="n3")
    
    fac.ConfigEnergy(0)
    fac.OptimizeRadial(["n2"])
    fac.ConfigEnergy(1)
    
    fac.Structure("ne.lev.b", ["n2", "n3"])
    fac.PrintTable("ne.lev.b", "ne.lev", 1)
```

Execute the generated file:
```bash
sfac calculation.sf
```

### Integration with atomkit

```python
from atomkit import Configuration
from atomkit.fac import SFACWriter

# Generate configurations using atomkit
ground = Configuration.from_element("Fe", 23)
excited = ground.generate_excitations(["2s", "2p", "3s"], 1)

# Export to FAC
with SFACWriter("fe_calc.sf") as fac:
    fac.SetAtom("Fe")
    
    # Add configurations from atomkit
    fac.config_from_atomkit(ground, "ground")
    for i, state in enumerate(excited):
        fac.config_from_atomkit(state, f"excited{i}")
    
    # Continue with FAC calculation
    fac.OptimizeRadial(["ground"])
    # ... rest of calculation
```

## Available Functions

The `SFACWriter` class implements all major FAC functions:

### Atomic Structure
- `SetAtom(symbol)` - Set atomic element
- `Closed(shells)` - Define closed shells
- `Config(config, group)` - Define configuration
- `ConfigEnergy(mode)` - Calculate configuration energies
- `OptimizeRadial(groups)` - Optimize radial potential
- `Structure(file, groups)` - Calculate energy levels
- `MemENTable(file)` - Load energy table to memory
- `PrintTable(input, output, verbose)` - Convert binary to ASCII

### Transitions and Rates
- `TransitionTable(file, lower, upper)` - Radiative transitions
- `CETable(file, lower, upper)` - Collisional excitation
- `CITable(file, bound, free)` - Collisional ionization
- `RRTable(file, bound, free)` - Radiative recombination
- `AITable(file, bound, free)` - Autoionization rates

### Settings
- `SetBreit(mode)` - Breit interaction
- `SetSE(mode)` - Self-energy corrections
- `SetVP(mode)` - Vacuum polarization
- `SetUTA(mode)` - Unresolved transition arrays
- `SetUsrCEGrid(grid)` - Custom energy grids
- `InitializeMPI(n)` - Parallel computation

### Utility
- `add_comment(text)` - Add comments to file
- `add_blank_line()` - Add blank line for readability
- `get_content()` - Preview content before writing

## Examples

See `examples/fac_wrapper_demo.py` for comprehensive examples including:
- Basic structure calculations
- Atomkit integration
- Parallel calculations with MPI
- Autoionization rates
- Photoionization cross sections

## Comparison with pfac

| Feature | pfac | atomkit.fac |
|---------|------|-------------|
| Installation | Requires compilation | Pure Python, no compilation |
| Dependencies | C/Fortran compilers, Python bindings | Standard library only |
| Platform | Linux/Unix primary | Cross-platform |
| Integration | Separate package | Built into atomkit |
| Output | Direct execution or .sf | .sf files for manual execution |
| Debugging | Binary execution | Readable text files |
| Version control | N/A | .sf files can be tracked |

## Workflow

1. **Generate** `.sf` file using `SFACWriter`
2. **Review** the generated file (plain text, human-readable)
3. **Execute** with FAC command-line tools
4. **Analyze** output files

```python
# 1. Generate
with SFACWriter("calc.sf") as fac:
    # ... define calculation

# 2. Review (optional)
# Open calc.sf in editor

# 3. Execute
# $ sfac calc.sf
# $ mpirun -n 24 sfac calc.sf  # parallel

# 4. Analyze output
# Binary: calc.lev.b, calc.tr.b, etc.
# ASCII: calc.lev, calc.tr, etc.
```

## Notes

- All generated `.sf` files use SFAC syntax, which is compatible with FAC 1.0.7+
- The wrapper follows FAC function naming conventions (CamelCase)
- Generated files can be edited manually if needed
- For pfac-specific features, you may still need the pfac package

## References

- [FAC Homepage](http://sprg.ssl.berkeley.edu/~mfgu/fac/)
- [FAC Manual](../manual.tex) - LaTeX documentation
- [atomkit Documentation](../../README.md)
