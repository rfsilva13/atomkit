# AtomKit

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-37%20passed-brightgreen)

## AtomKit: Code-Agnostic Atomic Physics

AtomKit is a modern Python framework for atomic structure calculations and spectroscopic data analysis. It acts as a universal translator and a unified analysis platform, allowing you to define your physics calculations once and run them across multiple atomic physics codes.

### Why Use AtomKit?

AtomKit's design philosophy is **"Physics First, Code Second."** It abstracts the underlying code implementation, letting you focus on the physics.

---

### Core Features

#### 🔬 Physics-First Abstraction
* **Define Calculations Physically:** Structure your problem using high-level physical terms (e.g., coupling schemes, transition types, configurations) instead of code-specific syntax.
* **Agnostic Workflow:** Automatically translate your physics definitions into the specific input formats for various supported codes (e.g., AUTOSTRUCTURE, FAC, GRASP).
* **Interchangeable Codes:** Seamlessly switch between different atomic codes without needing to rewrite your entire workflow.

#### 📊 Unified Analysis & Visualization
* **Agnostic Parsing:** Ingest and parse the outputs from different codes into a single, consistent, and easy-to-use data structure.
* **Comparative Analysis:** Directly compare results (e.g., energy levels, transition probabilities) from multiple codes in one environment.
* **Data Visualization:** Generate a wide range of analysis plots and publication-ready tables directly from the unified data objects.

#### 🎯 Versatile Data Export
* **Standardized Output:** Export your processed and analyzed data into common formats.
* **Cross-Platform Integration:** Create outputs ready for use in other scientific platforms, including radiative transfer codes, plasma modeling suites, and spectroscopic databases.

## Core Capabilities

### 🎯 Code-Agnostic Interface

Define your atomic physics calculation once in physical terms:

```python
from atomkit.core import AtomicCalculation
from atomkit import Configuration

# Define calculation in physics terms - no code-specific syntax!
calc = AtomicCalculation(
    element="Fe",
    charge=16,                      # Fe XVII
    calculation_type="radiative",  # What physics to compute
    coupling="IC",                  # Intermediate coupling
    relativistic="Breit",           # Include Breit interaction
    radiation_types=["E1", "M1"],   # Electric & magnetic dipole
    configurations=[
        Configuration.from_string("1s2 2s2 2p6"),
        Configuration.from_string("1s2 2s2 2p5 3s1"),
    ],
    code="autostructure"  # ← Only code-specific choice!
)

# Generate input file
input_file = calc.write_input()

# Switch codes by changing ONE parameter:
calc.code = "fac"
fac_input = calc.write_input()  # Same physics, different code!
```

**Supported Codes:**
- ✅ AUTOSTRUCTURE (Badnell 2011) - all namelists and options
- ✅ FAC (Flexible Atomic Code, Gu 2008) - via SFAC wrapper
- 🚧 GRASP (coming soon)
- 🚧 MCDF (planned)

### 📊 Unified Data Analysis

Parse and analyze outputs from any code consistently:

```python
from atomkit.readers import read_autostructure_levels, read_fac_levels

# Read data from different codes
as_levels = read_autostructure_levels("output.j")
fac_levels = read_fac_levels("output.lev.asc")

# Both return pandas DataFrames with standardized columns:
# - energy, configuration, term, J, parity, g-factor, ...
# Compare results directly!
```

### � Publication-Ready Output

Generate tables and plots ready for papers, databases, and radiative transfer codes:

```python
from atomkit.physics.plotting import plot_grotrian_diagram, plot_cross_section
from atomkit.export import to_chianti_format, to_adas_format, to_latex_table

# Create publication-quality visualizations
plot_grotrian_diagram(levels_df, transitions_df, save="grotrian.pdf")
plot_cross_section(energy, cross_section, save="cross_section.png")

# Export for other platforms
to_chianti_format(levels, transitions, "fe_17.elvlc")  # For CHIANTI database
to_adas_format(transitions, "fe_17.dat")                # For ADAS
to_latex_table(levels, "levels.tex")                    # For publications
```

### 🔬 Atomic Structure Tools

Powerful configuration management that speaks physics:

- **Configuration Generation:** Ground states, excitations, ionization states
- **Electronic Structure:** Holes, core-valence splitting, configuration interaction
- **Quantum Numbers:** Full support for LS, IC, jj, LSJ coupling schemes
- **Validation:** Automatic checks for occupancy, quantum number rules, parity
- **Notation:** Parse and convert between different spectroscopic notations

### 🧮 Physics Utilities

Everything you need for atomic physics analysis:

- **Energy Conversion:** eV ↔ Rydberg ↔ cm⁻¹ ↔ wavelength ↔ temperature
- **Cross Sections:** Photoionization, electron impact, autoionization rates
- **Collision Strengths:** Resonance calculations, effective collision strengths
- **Atomic Potentials:** Thomas-Fermi, Hartree-Fock, screening potentials
- **Statistical Equilibrium:** Population analysis, ionization balance

## The AtomKit Advantage

### Traditional Workflow ❌
```
Define physics in AUTOSTRUCTURE syntax
    ↓
Run AUTOSTRUCTURE
    ↓
Parse AUTOSTRUCTURE output (custom script)
    ↓
Want to compare with FAC? → Start over with completely different syntax!
```

### AtomKit Workflow ✅
```
Define physics ONCE in AtomKit (code-agnostic)
    ↓
    ├→ Generate AUTOSTRUCTURE input → Run → Parse results ──┐
    ├→ Generate FAC input → Run → Parse results ────────────┤
    └→ Generate GRASP input → Run → Parse results ──────────┤
                                                             ↓
                                        Unified DataFrame for analysis
                                                             ↓
                            ├─ Compare codes ─ Publication plots ─┤
                            ├─ Export to CHIANTI/ADAS ────────────┤
                            └─ Generate LaTeX tables ─────────────┘
```

**Key Benefits:**
- 🎯 **Reproducibility:** Same physics definition across all codes
- ⚡ **Efficiency:** Switch codes by changing one parameter
- 📊 **Consistency:** All outputs in same format for direct comparison
- 🔬 **Validation:** Easily benchmark codes against each other
- 🚀 **Productivity:** Focus on physics, not code syntax

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

## Use Cases

### 🌟 Stellar Astrophysics
- Calculate atomic data for stellar atmosphere models (TLUSTY, PHOENIX, MARCS)
- Generate opacity tables for radiative transfer
- Compare codes to quantify systematic uncertainties
- Export line lists for spectral synthesis

### 🔥 Plasma Physics & Fusion
- Electron collision cross sections for plasma diagnostics
- Ionization balance and level populations
- Spectral line emission for tokamak modeling
- ADAS-format output for plasma codes

### 🗄️ Atomic Databases
- Generate data for CHIANTI, NIST, VAMDC
- Validate and compare database entries
- Create custom atomic datasets
- Quality control and consistency checks

### 📊 Laboratory Spectroscopy
- Identify spectral lines from experimental data
- Predict transition wavelengths and intensities
- Analyze fine structure and isotope shifts
- Generate synthetic spectra for comparison

### 🎓 Research & Education
- Explore atomic structure interactively
- Compare theoretical predictions with experiment
- Benchmark different computational methods
- Teach atomic physics concepts with real calculations

## Quick Start

### 1. Define Your Physics (Code-Agnostic)

```python
from atomkit.core import AtomicCalculation
from atomkit import Configuration

# Physics problem: Fe XVII radiative transitions
calc = AtomicCalculation(
    element="Fe",
    charge=16,
    calculation_type="radiative",
    coupling="IC",                  # Intermediate coupling
    relativistic="Breit",           # Breit-Pauli Hamiltonian
    radiation_types=["E1"],         # Electric dipole transitions
    configurations=[
        Configuration.from_string("1s2 2s2 2p6"),        # Ground: 2p⁶
        Configuration.from_string("1s2 2s2 2p5 3s1"),    # Excited: 2p⁵3s
    ],
    code="autostructure"
)

# Generate input - physics is translated automatically!
input_file = calc.write_input()
```

### 2. Analyze Results (Any Code)

```python
from atomkit.readers import read_levels, read_transitions
from atomkit.physics.plotting import plot_energy_levels

# Read output (works for AUTOSTRUCTURE, FAC, etc.)
levels = read_levels("output.lev")
transitions = read_transitions("output.tr")

# Analyze
print(f"Found {len(levels)} levels")
print(f"Found {len(transitions)} transitions")

# Find strongest lines
strongest = transitions.nlargest(10, 'gf')
print(strongest[['wavelength_nm', 'gf', 'transition']])

# Visualize
plot_energy_levels(levels, save="energy_diagram.pdf")
```

### 3. Export for Your Platform

```python
from atomkit.export import (
    to_chianti_format,    # For CHIANTI database
    to_adas_format,       # For plasma modeling
    to_latex_table,       # For publications
    to_radtrans_format    # For radiative transfer
)

# Export for CHIANTI atomic database
to_chianti_format(levels, transitions, "fe_17.elvlc", "fe_17.wgfa")

# Generate LaTeX table for paper
to_latex_table(
    strongest,
    "table1.tex",
    caption="Strongest transitions in Fe XVII",
    columns=['wavelength_nm', 'gf', 'A_value']
)
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

## Roadmap & Development

### Code Support
- [x] AUTOSTRUCTURE - Complete support (all namelists, validated with official test suite)
- [x] FAC - Full support via SFAC wrapper
- [ ] GRASP - Reader implemented, writer in progress
- [ ] MCDF - Planned
- [ ] R-Matrix codes - Under consideration

### Analysis & Visualization
- [x] Energy level diagrams
- [x] Cross section plotting
- [ ] Grotrian diagrams (in progress)
- [ ] Interactive spectral line identification
- [ ] Atomic structure visualization (orbitals, densities)
- [ ] Collisional-radiative modeling integration

### Data Export & Integration
- [x] AUTOSTRUCTURE format
- [x] FAC format
- [ ] CHIANTI database format (in progress)
- [ ] ADAS format (in progress)
- [ ] NIST ASD format
- [ ] VAMDC format
- [ ] Kurucz/VALD line lists
- [ ] Opacity Project format

### Physics Tools
- [x] Energy unit conversions
- [x] Configuration generation and manipulation
- [x] LS/IC/jj coupling schemes
- [ ] Mixing coefficient analysis
- [ ] JJ2LSJ transformation
- [ ] Boltzmann and Saha equation solvers
- [ ] Collisional-radiative equilibrium
- [ ] Autoionization rate calculations

### Databases & Validation
- [ ] NIST ASD integration for validation
- [ ] Comparison tools for benchmarking codes
- [ ] Experimental data integration
- [ ] Systematic uncertainty quantification

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
