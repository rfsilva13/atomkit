# AtomKit

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-226%20passed-brightgreen)

AtomKit is a comprehensive Python toolkit for atomic structure and spectral data analysis. It provides powerful tools for parsing, manipulating, and analyzing data from atomic physics codes, with a focus on electron configurations and FAC (Flexible Atomic Code) output.

## Features

### 🔬 Atomic Structure Management
- **Shell Representation:** Complete representation of electron shells with quantum numbers (n, l, j), occupation, and spectroscopic notation
- **Configuration Management:** Powerful `Configuration` class for handling electron configurations with support for:
  - Multiple notation formats (standard, compact, with j-quantum numbers)
  - Automatic validation and manipulation
  - Core/valence splitting
  - Hole and excited state generation
  - X-ray notation labels

### 📊 Data Parsing
- **FAC Reader:** Parse energy levels (`.lev.asc`) and transitions (`.tr.asc`) from FAC
  - Multi-block file support
  - Flexible unit conversion (eV, cm⁻¹, Ry, nm, Å)
  - Automatic calculation of derived quantities
  - Pandas DataFrame output for easy analysis

### 🧮 Physics Utilities
- **Cross Sections & Collision Strengths:** Conversion utilities and resonance calculations
- **Energy Conversion:** Convert between eV, Rydberg, cm⁻¹, Joules, Hz, Kelvin, and Hartree
- **Visualization:** Publication-quality plots with matplotlib integration (optional)

## Installation

### Using Conda (Recommended)

AtomKit requires Python 3.10 or higher. We recommend using conda to create a dedicated environment:

```bash
# Clone the repository
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit

# Create and activate the atomkit environment
conda create -n atomkit python=3.13
conda activate atomkit

# Install dependencies
pip install -e .

# Or install with optional features
pip install -e ".[all]"  # Includes plotting and enhanced logging
```

### Using Poetry

If you prefer using Poetry for dependency management:

```bash
# Clone the repository
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit

# Install with Poetry (automatically creates virtual environment)
poetry install

# Install with all optional features
poetry install --extras "all"

# Activate the Poetry shell
poetry shell
```

### From Conda Environment File

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate atomkit
```

## Quick Start

### Basic Configuration Usage

```python
from atomkit import Configuration, Shell

# Create configuration from string
config = Configuration.from_string("1s2.2s2.2p6")
print(config)  # Output: 1s2.2s2.2p6

# Create from element
ne_config = Configuration.from_element("Ne")
print(f"Neon has {ne_config.total_electrons()} electrons")

# Generate excited states
excited = config.generate_excitations(
    target_shells=["3s", "3p", "3d"],
    excitation_level=1,
    source_shells=["2p"]
)
print(f"Generated {len(excited)} excited configurations")

# Parse orbital shells
shell = Shell.from_string("3d10")
print(f"n={shell.n}, l={shell.l_quantum}, occupation={shell.occupation}")
```

### AUTOSTRUCTURE Integration (Code-Agnostic Workflow)

```python
from atomkit import Configuration
from atomkit.converters import configurations_to_autostructure

# Step 1: Generate configurations (PHYSICS - code-agnostic)
ground = Configuration.from_string('1s2 2s2 2p6 3s2 3p6 3d6 4s2')
excited = ground.generate_excitations(
    target_shells=['4p', '5s', '4d'],  # Where to excite TO
    excitation_level=1,                 # Single excitations
    source_shells=['3d', '4s']          # Excite FROM valence
)

# Step 1.5: Filter/modify (physics is done, now manipulate!)
filtered = [c for c in excited if '5s' in c.to_string()]

# Step 2: Format for AUTOSTRUCTURE (I/O - only now code-specific)
result = configurations_to_autostructure(
    [ground] + filtered,
    last_core_orbital='3p',
    output_file='fe_configs.txt'
)

print(f"Generated {result['mxconf']} configurations")
# Same configs could be formatted for FAC, GRASP, or any other code!
```

### Reading FAC Data

```python
from atomkit.readers import read_fac_levels, read_fac_transitions

# Read energy levels
levels_df = read_fac_levels("my_data.lev.asc", energy_unit="eV")
print(levels_df.head())

# Read transitions
transitions_df = read_fac_transitions("my_data.tr.asc", wavelength_unit="nm")
print(transitions_df[["wavelength_nm", "gf", "config_initial", "config_final"]])
```

### Utilities and Element Information

```python
from atomkit import get_element_info, parse_ion_notation

# Get element information
info = get_element_info('Fe')
print(f"Iron: Z={info['Z']}, name={info['name']}")

info = get_element_info(26)  # Also accepts atomic number
print(f"Element {info['symbol']}")

# Parse spectroscopic ion notation
element, charge, electrons = parse_ion_notation('Fe I')   # Neutral
element, charge, electrons = parse_ion_notation('Fe II')  # Singly ionized
print(f"Fe II: charge=+{charge}, {electrons} electrons")
```

### Energy Conversion

```python
from atomkit.physics import energy_converter

# Convert between units
energy_ry = energy_converter.ev_to_rydberg(13.6)
energy_cm = energy_converter.convert(100, from_unit="eV", to_unit="cm-1")
temp_k = energy_converter.ev_to_kelvin(1.0)

print(f"13.6 eV = {energy_ry:.2f} Ry")
print(f"100 eV = {energy_cm:.2e} cm⁻¹")
print(f"1 eV = {temp_k:.2e} K")
```

### Plotting Cross Sections

```python
from atomkit.physics.plotting import quick_plot_cross_section
import numpy as np

# Generate sample data
energies = np.linspace(10, 200, 100)
cross_sections = 1e-18 / energies  # Simple 1/E behavior

# Create publication-quality plot
fig, ax = quick_plot_cross_section(
    energies, 
    cross_sections,
    title="Electron Impact Cross Section",
    save_path="cross_section.png"
)
```

### FAC (Flexible Atomic Code) Integration

Atom kit provides a Python wrapper for generating FAC input files without requiring pfac compilation:

```python
from atomkit.fac import SFACWriter
from atomkit import Configuration

# Option 1: Generate FAC input from atomkit configurations
ground = Configuration.from_element("Fe", 23)  # Fe XXIV
excited = ground.generate_excitations(["2s", "2p", "3s"], 1)

with SFACWriter("fe_calculation.sf") as fac:
    fac.SetAtom("Fe")
    fac.SetBreit(-1)  # Enable Breit interaction
    fac.SetSE(-1)     # Enable QED corrections
    
    # Add configurations from atomkit
    fac.config_from_atomkit(ground, "ground")
    for i, state in enumerate(excited):
        fac.config_from_atomkit(state, f"excited{i}")
    
    # Perform calculation
    fac.OptimizeRadial(["ground"])
    fac.Structure("output.lev.b", ["ground"] + [f"excited{i}" for i in range(len(excited))])
    fac.PrintTable("output.lev.b", "output.lev", 1)

# Execute the generated file
# $ sfac fe_calculation.sf
# $ mpirun -n 24 sfac fe_calculation.sf  # parallel

# Option 2: Direct SFAC syntax (like FAC manual examples)
with SFACWriter("ne_like_fe.sf") as fac:
    fac.SetAtom("Fe")
    fac.Closed("1s")
    fac.Config("2*8", group="n2")       # n=2 complex
    fac.Config("2*7 3*1", group="n3")   # n=2 -> n=3 excitation
    
    fac.ConfigEnergy(0)
    fac.OptimizeRadial(["n2"])
    fac.ConfigEnergy(1)
    
    fac.Structure("ne.lev.b", ["n2", "n3"])
    fac.TransitionTable("ne.tr.b", ["n2"], ["n3"])
```

The FAC wrapper:
- ✅ No pfac dependency - pure Python
- ✅ Generates readable .sf files
- ✅ Full integration with atomkit configurations
- ✅ All major FAC functions supported
- ✅ MPI parallel calculation support

See `examples/fac_wrapper_demo.py` for comprehensive examples and `src/atomkit/fac/README.md` for complete documentation.

## Documentation

For detailed API documentation, see:
- [API Reference](docs/API_REFERENCE.md) - Complete function and class documentation
- [FAC Integration Guide](src/atomkit/fac/README.md) - FAC wrapper documentation
- [Configuration Guide](DOCUMENTATION.md) - Detailed guide for configuration manipulation
- [Examples Directory](examples/) - Working examples demonstrating key features

## Development

### Setting Up Development Environment

```bash
# Create conda environment
conda create -n atomkit python=3.13
conda activate atomkit

# Install in development mode with all dependencies
pip install -e ".[all]"

# Install development dependencies
pip install pytest pytest-cov black ruff mypy

# Or using Poetry
poetry install --extras "all" --with dev
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/atomkit --cov-report=html

# Run specific test file
pytest tests/test_configuration.py -v
```

### Code Quality

```bash
# Format code with black
black src/ tests/

# Lint with ruff
ruff check src/ tests/

# Type checking with mypy
mypy src/
```

## Requirements

### Core Dependencies
- Python ≥ 3.10
- NumPy ≥ 1.24
- Pandas ≥ 2.2
- SciPy ≥ 1.10
- Mendeleev ≥ 0.9

### Optional Dependencies
- Matplotlib ≥ 3.7 (for plotting)
- Seaborn ≥ 0.12 (enhanced plotting)
- Colorlog ≥ 6.7 (colored logging)

## Project Structure

```
atomkit/
├── src/atomkit/          # Main package
│   ├── configuration.py  # Configuration class
│   ├── shell.py          # Shell representation
│   ├── utils.py          # Element info & ion notation parsing
│   ├── definitions.py    # Constants and mappings
│   ├── converters/       # Format converters
│   │   ├── as_generator.py     # AUTOSTRUCTURE
│   │   ├── fac_to_as.py        # FAC to AS
│   │   └── ls_to_icr.py        # LS to ICR
│   ├── readers/          # Data file parsers
│   │   ├── levels.py           # FAC levels
│   │   ├── transitions.py      # FAC transitions
│   │   ├── autostructure.py    # AS parser
│   │   └── autoionization.py   # Autoionization
│   └── physics/          # Physics utilities
│       ├── units.py            # Energy conversion
│       ├── cross_sections.py   # Collision strengths
│       ├── potentials.py       # Atomic potentials
│       └── plotting.py         # Visualization
├── tests/                # Test suite (226 tests)
├── examples/             # Usage examples
└── docs/                 # Additional documentation
```

## Roadmap

- [ ] NIST database parsing and utilities
- [ ] AUTOSTRUCTURE parser
- [ ] ADAS file format support
- [ ] Mixing coefficients analysis
- [ ] Grotrian diagram plotting
- [ ] Boltzmann and Saha analysis tools
- [ ] ColRadPy integration for plasma modeling
- [ ] JJ2LSJ transformation utilities

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use AtomKit in your research, please cite:

```bibtex
@software{atomkit,
  author = {Silva, Ricardo Ferreira da},
  title = {AtomKit: A Python Toolkit for Atomic Physics},
  year = {2025},
  url = {https://github.com/rfsilva13/atomkit}
}
```

## Contact

Ricardo Ferreira da Silva - rfsilva@lip.pt

Project Link: [https://github.com/rfsilva13/atomkit](https://github.com/rfsilva13/atomkit)
