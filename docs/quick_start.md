# Quick Start Guide

Get up and running with AtomKit in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit

# Create conda environment
conda create -n atomkit python=3.13
conda activate atomkit

# Install AtomKit
pip install -e .

# Verify installation
python verify_install.py
```

## Your First AtomKit Script

Create a file called `my_first_script.py`:

```python
from atomkit import Configuration

# Create a configuration from a string
neon = Configuration.from_string("1s2.2s2.2p6")
print(f"Neon configuration: {neon}")
print(f"Total electrons: {neon.total_electrons()}")

# Create from element symbol
argon = Configuration.from_element("Ar")
print(f"\nArgon configuration: {argon}")

# Generate excited states
excited = neon.generate_excitations(
    source_shells=["2p"],
    target_shells=["3s", "3p"],
    num_electrons=1
)

print(f"\nGenerated {len(excited)} excited configurations:")
for i, config in enumerate(excited[:3], 1):  # Show first 3
    print(f"  {i}. {config}")
```

Run it:
```bash
conda activate atomkit
python my_first_script.py
```

Expected output:
```
Neon configuration: 1s2.2s2.2p6
Total electrons: 10

Argon configuration: 1s2.2s2.2p6.3s2.3p6

Generated 2 excited configurations:
  1. 1s2.2s2.2p5.3s1
  2. 1s2.2s2.2p5.3p1
```

## Common Tasks

### Read FAC Level Data

```python
from atomkit.readers import read_fac

# Read energy levels from FAC output
levels = read_fac("my_data.lev.asc", energy_unit="eV")

# Display first few levels
print(levels[["iLev", "Energy_eV", "config"]].head())
```

### Energy Conversion

```python
from atomkit.physics import energy_converter

# Convert between units
energy_ev = 13.6  # eV
energy_ry = energy_converter.ev_to_rydberg(energy_ev)
energy_cm = energy_converter.ev_to_wavenumber(energy_ev)

print(f"{energy_ev} eV = {energy_ry:.3f} Ry = {energy_cm:.0f} cm⁻¹")
```

### Plot Cross Sections

```python
import numpy as np
from atomkit.physics.plotting import quick_plot_cross_section

# Generate sample data
energies = np.linspace(10, 200, 100)
cross_sections = 1e-18 / energies  # Simple 1/E behavior

# Create plot
fig, ax = quick_plot_cross_section(
    energies,
    cross_sections,
    title="Electron Impact Cross Section"
)
```

## Next Steps

Now that you have AtomKit running, explore:

1. **[Examples Directory](../examples/)** - Working code examples for all features
2. **[Configuration Guide](configuration_guide.md)** - Deep dive into electron configurations
3. **[FAC Reader Guide](fac_reader_guide.md)** - Processing atomic structure data
4. **[API Reference](api/)** - Complete API documentation

## Getting Help

- Run examples: `python examples/basic_usage.py`
- Check API docs: See `docs/api/` directory
- Report issues: [GitHub Issues](https://github.com/rfsilva13/atomkit/issues)

## Common Issues

### "Module not found" error

Make sure you're in the atomkit environment:
```bash
conda activate atomkit
python -c "import atomkit; print('AtomKit installed!')"
```

### Import errors

Reinstall in development mode:
```bash
conda activate atomkit
cd /path/to/atomkit
pip install -e .
```

### Tests failing

Verify your installation:
```bash
conda activate atomkit
python verify_install.py
pytest tests/ -v
```

---

**Ready to dive deeper?** Check out the [full documentation](README.md) and [examples](../examples/)!
