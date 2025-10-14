# AtomKit

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-162%20passed-brightgreen)

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
from atomkit import Configuration

# Create configuration from string
config = Configuration.from_string("1s2.2s2.2p6")
print(config)  # Output: 1s2 2s2 2p6

# Create from element
ne_config = Configuration.from_element("Ne")
print(f"Neon has {ne_config.total_electrons} electrons")

# Generate excited states
excited = config.generate_excitations(
    source_shells=["2p"],
    target_shells=["3s", "3p", "3d"],
    num_electrons=1
)
print(f"Generated {len(excited)} excited configurations")
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

## Documentation

For detailed API documentation, see:
- [API Reference](API_REFERENCE.md) - Complete function and class documentation
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
│   ├── configuration.py  # Configuration and Shell classes
│   ├── definitions.py    # Constants and mappings
│   ├── shell.py          # Shell representation
│   ├── readers/          # Data file parsers
│   │   ├── levels.py     # FAC level reader
│   │   ├── transitions.py # FAC transition reader
│   │   └── autoionization.py
│   └── physics/          # Physics utilities
│       ├── units.py      # Energy conversion
│       ├── cross_sections.py
│       └── plotting.py   # Visualization
├── tests/                # Test suite (162 tests)
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

Ricardo Ferreira da Silva - ricardo.apf.silva@gmail.com

Project Link: [https://github.com/rfsilva13/atomkit](https://github.com/rfsilva13/atomkit)
