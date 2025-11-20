# AtomKit Package Structure

## Overview
The AtomKit package is organized into logical subdirectories with NO standalone utility files at the root level (only `__init__.py`). This provides a clean, maintainable structure.

## Directory Structure

```
src/atomkit/
├── __init__.py              # Main package entry point (ONLY file at root)
│
├── constants/               # Physical constants and definitions
│   └── __init__.py         # EV_TO_CM1, HARTREE_EV, L_SYMBOLS, etc.
│
├── utils/                   # General utility functions
│   └── __init__.py         # get_element_info(), parse_ion_notation()
│
├── structure/               # Fundamental atomic structure classes
│   ├── __init__.py
│   ├── shell.py            # Shell class for electron shells
│   └── configuration.py    # Configuration class for electron configurations
│
├── analysis/                # Spectral analysis and data processing
│   └── __init__.py         # calculate_g(), process_*_intensities(), etc.
│
├── readers/                 # File format readers
│   ├── __init__.py
│   ├── base.py
│   ├── levels.py
│   ├── transitions.py
│   ├── autoionization.py
│   ├── autostructure.py
│   ├── siegbahn.py
│   └── labeling.py
│
├── converters/              # Format conversion utilities
│   ├── __init__.py
│   ├── as_generator.py
│   ├── fac_to_as.py
│   └── ls_to_icr.py
│
├── physics/                 # Physical calculations
│   ├── __init__.py
│   ├── cross_sections.py
│   ├── potentials.py
│   ├── units.py
│   └── plotting.py
│
├── fac/                     # FAC (Flexible Atomic Code) interface
│   ├── __init__.py
│   └── sfac_writer.py
│
├── autostructure/           # AUTOSTRUCTURE interface
│   ├── __init__.py
│   └── as_writer.py
│
└── core/                    # Unified calculation interface
    ├── __init__.py
    ├── calculation.py
    ├── backends.py
    └── specs.py
```

## Module Descriptions

### constants/
Physical constants, quantum number mappings, and validation functions:
- `EV_TO_CM1`, `HARTREE_EV`, `RYD_EV` - Unit conversion constants
- `L_SYMBOLS`, `ANGULAR_MOMENTUM_MAP` - Quantum number mappings
- `validate_j_quantum()`, `get_max_shell_occupation()` - Validation utilities

### utils/
General-purpose utility functions:
- `get_element_info()` - Get element data by symbol or Z
- `parse_ion_notation()` - Parse ion notation strings

### structure/
Fundamental classes for representing atomic structure:
- **Shell**: Represents an electron shell/subshell with quantum numbers (n, l, j) and occupation
- **Configuration**: Represents complete electron configurations as collections of shells

### analysis/
Spectral analysis and data processing functions:
- `calculate_g()` - Statistical weight calculations
- `get_spectator_hole()` - Identify spectator holes
- `process_diagram_intensities()`, `process_satellite_intensities()` - Intensity calculations
- `plot_k_alpha_spectrum()` - Visualization

### readers/
Parsers for various atomic structure file formats from different codes (FAC, AUTOSTRUCTURE, etc.)

### converters/
Tools for converting between different file formats and data representations

### physics/
Physical calculations including cross-sections, potentials, and unit conversions

### fac/
Interface to the Flexible Atomic Code (FAC) for generating input files

### autostructure/
Interface to AUTOSTRUCTURE for generating input files

### core/
Unified high-level interface (`AtomicCalculation` class) that abstracts over different backends

## Import Examples

### Basic imports (recommended)
```python
from atomkit import Shell, Configuration
from atomkit import get_element_info, parse_ion_notation

# Create a shell
s = Shell(2, 1, 2, 0.5)  # 2p- with 2 electrons

# Create a configuration
c = Configuration.from_string('1s2 2s2 2p6')
```

### Constants and utilities
```python
from atomkit.constants import EV_TO_CM1, HARTREE_EV, L_SYMBOLS
from atomkit.utils import get_element_info
from atomkit.analysis import calculate_g, plot_k_alpha_spectrum

# Use constants
print(f"Energy conversion: {EV_TO_CM1} cm^-1/eV")

# Get element info
info = get_element_info('Fe')

# Calculate statistical weight
g = calculate_g('1s_2p')
```

### Module imports
```python
import atomkit
from atomkit import structure, analysis, physics, core

# Access classes through module
shell = atomkit.structure.Shell(2, 0, 2)

# Use analysis functions
data = atomkit.analysis.load_spectral_data(file_paths)

# Use unified interface
calc = atomkit.AtomicCalculation(element='Fe', charge=15)
```

### Submodule imports
```python
from atomkit.structure import Shell, Configuration
from atomkit.analysis import plot_k_alpha_spectrum
from atomkit.physics import cross_sections
from atomkit.core import AtomicCalculation
from atomkit.constants import EV_TO_CM1
```

## Migration Guide

### From old structure
```python
# OLD (no longer works)
from atomkit.shell import Shell
from atomkit.configuration import Configuration
from atomkit.definitions import EV_TO_CM1
from atomkit.utils import get_element_info
from atomkit.analysis import calculate_g

# NEW (correct)
from atomkit import Shell, Configuration, get_element_info
from atomkit.constants import EV_TO_CM1
from atomkit.analysis import calculate_g
```

### Recommended patterns
```python
# Top-level imports for common classes
from atomkit import Shell, Configuration, get_element_info

# Submodule imports for specific functionality
from atomkit.constants import EV_TO_CM1, HARTREE_EV
from atomkit.analysis import calculate_g, process_diagram_intensities
from atomkit.physics.units import energy_converter
```

## Benefits
✓ **Clean root**: Only `__init__.py` at package root  
✓ **Logical organization**: Related modules grouped in subdirectories  
✓ **Scalability**: Each subdirectory can grow independently  
✓ **Clear imports**: Users know where to find functionality  
✓ **Maintainability**: Easy to locate and update code  
✓ **Backward compatible**: Top-level exports preserved
