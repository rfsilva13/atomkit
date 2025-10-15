# AtomKit API Reference (2025)

**Last Updated**: October 15, 2025  
**Version**: 1.0  
**Test Coverage**: 226 tests passing

This document provides a comprehensive reference to the AtomKit API after recent refactoring.

---

## Core Classes

### `Configuration`

Represents an atomic electron configuration as a collection of shells.

**Import:**
```python
from atomkit import Configuration
```

#### Class Methods

##### `from_string(config_str: str) -> Configuration`
Parse configuration from string notation.

**Supported formats:**
- Standard: `"1s2.2s2.2p6"` or `"1s2 2s2 2p6"`
- With j-quantum: `"1s2.2p-2.2p+4"`
- Compact: `"1s22s22p6"`

**Example:**
```python
config = Configuration.from_string("1s2.2s2.2p6")
config = Configuration.from_string("1s2 2s2 2p6")  # Space separator
config = Configuration.from_string("1s2.2p-2.2p+4")  # With j
```

##### `from_element(element_identifier: str | int, ion_charge: int = 0) -> Configuration`
Create ground state configuration from element.

**Parameters:**
- `element_identifier`: Element symbol ('Fe'), name ('Iron'), or atomic number (26)
- `ion_charge`: Ionization state (0=neutral, 1=+1, etc.)

**Example:**
```python
ne = Configuration.from_element("Ne")           # Neutral neon
ne_plus = Configuration.from_element("Ne", 1)   # Ne+
fe = Configuration.from_element(26)             # Iron from Z
```

##### `from_compact_string(compact_str: str, generate_permutations: bool = False) -> Configuration | List[Configuration]`
Parse compact notation like `"1*2.2*8.3*3"`.

**Example:**
```python
config = Configuration.from_compact_string("1*2.2*8")
# Returns: 1s2.2s2.2p6
```

#### Instance Methods

##### `total_electrons() -> int`
Count total electrons in configuration.

##### `to_string(separator: str = ".", include_j: bool = False) -> str`
Convert to string representation.

**Parameters:**
- `separator`: Separator between shells (`.`, ` `, or `""`)
- `include_j`: Include j-quantum numbers

**Example:**
```python
config.to_string()                    # "1s2.2s2.2p6"
config.to_string(separator=" ")       # "1s2 2s2 2p6"
config.to_string(separator="")        # "1s22s22p6"
config.to_string(include_j=True)      # "1s2.2p-2.2p+4"
```

##### `generate_excitations(source_shells, target_shells, num_electrons, num_holes=None) -> List[Configuration]`
Generate excited configurations.

**Parameters:**
- `source_shells`: List of shells to excite from (e.g., `["2p", "3d"]`)
- `target_shells`: List of shells to excite to (e.g., `["3s", "3p"]`)
- `num_electrons`: Number of electrons to excite
- `num_holes`: Number of holes to create (optional)

**Example:**
```python
ground = Configuration.from_string("1s2.2s2.2p6")
excited = ground.generate_excitations(
    source_shells=["2p"],
    target_shells=["3s", "3p", "3d"],
    num_electrons=1
)
# Returns list of configurations with 2p→3s, 2p→3p, 2p→3d excitations
```

##### `generate_autostructure_configurations(valence_shells, max_n, max_l) -> List[Configuration]`
Generate AUTOSTRUCTURE-style single excitations from valence shells.

**Parameters:**
- `valence_shells`: List of valence orbital prefixes (e.g., `["3d", "4s"]`)
- `max_n`: Maximum principal quantum number
- `max_l`: Maximum angular momentum (as integer: 0=s, 1=p, 2=d)

**Example:**
```python
config = Configuration.from_string("1s2.2s2.2p6.3s2.3p6.3d6.4s2")
all_configs = config.generate_autostructure_configurations(
    valence_shells=["3d", "4s"],
    max_n=5,
    max_l=2  # Up to d orbitals
)
```

##### `get_ionstage(element_identifier: str | int) -> int`
Calculate ion stage relative to neutral atom.

**Returns:** Ion stage (Z - number of electrons)

**Example:**
```python
config = Configuration.from_string("1s2.2s2.2p5")  # 9 electrons
stage = config.get_ionstage("Ne")  # Z=10, so stage=1 (Ne+)
```

##### `difference(other: Configuration) -> Dict[str, int]`
Find difference between configurations.

**Returns:** Dict mapping orbital notation to electron difference

**Example:**
```python
config1 = Configuration.from_string("1s2.2s2.2p6")
config2 = Configuration.from_string("1s2.2s2.2p5.3s1")
diff = config1.difference(config2)
# Returns: {"2p": 1, "3s": -1}  (config1 has 1 more in 2p, 1 fewer in 3s)
```

---

### `Shell`

Represents a single electron shell with quantum numbers.

**Import:**
```python
from atomkit import Shell
```

#### Constructor

```python
Shell(n: int, l_quantum: int, occupation: int, j_quantum: float = None)
```

**Parameters:**
- `n`: Principal quantum number (1, 2, 3, ...)
- `l_quantum`: Angular momentum (0=s, 1=p, 2=d, 3=f, ...)
- `occupation`: Number of electrons (0 to max_occupation)
- `j_quantum`: Total angular momentum (optional)

**Example:**
```python
shell = Shell(n=3, l_quantum=2, occupation=10)  # 3d10
shell = Shell(n=2, l_quantum=1, j_quantum=0.5, occupation=2)  # 2p-2
```

#### Class Methods

##### `from_string(shell_str: str) -> Shell`
Parse shell from string notation.

**Formats:**
- Standard: `"1s2"`, `"2p6"`, `"3d10"`
- With j: `"2p-2"` (j=l-1/2), `"2p+4"` (j=l+1/2)

**Example:**
```python
shell = Shell.from_string("3d10")
shell = Shell.from_string("2p-2")  # 2p with j=1/2
```

#### Instance Methods

##### `max_occupation() -> int`
Maximum electrons this shell can hold.

**Returns:** 2*(2*l+1) for no j, or 2*j+1 with j

##### `is_full() -> bool`
Check if shell is at maximum occupation.

##### `to_string(include_j: bool = False) -> str`
Convert to string representation.

---

## Utility Functions

### Element Information

#### `get_element_info(element_symbol_or_z: str | int) -> Dict`

Get element information from periodic table.

**Import:**
```python
from atomkit import get_element_info
```

**Parameters:**
- `element_symbol_or_z`: Element symbol ('Fe', 'Au') or atomic number (26, 79)

**Returns:**
Dict with keys: `'symbol'`, `'Z'`, `'name'`

**Example:**
```python
info = get_element_info('Fe')
# Returns: {'symbol': 'Fe', 'Z': 26, 'name': 'Iron'}

info = get_element_info(79)
# Returns: {'symbol': 'Au', 'Z': 79, 'name': 'Gold'}
```

---

### Ion Notation Parsing

#### `parse_ion_notation(ion_notation: str) -> Tuple[str, int, int]`

Parse spectroscopic ion notation with Roman numerals.

**Import:**
```python
from atomkit import parse_ion_notation
```

**Format:** `'Element RomanNumeral'` where:
- I = neutral (charge 0)
- II = singly ionized (+1)
- III = doubly ionized (+2)
- etc.

**Returns:** Tuple of (element_symbol, charge, num_electrons)

**Example:**
```python
element, charge, electrons = parse_ion_notation('Fe I')
# Returns: ('Fe', 0, 26)  # Neutral iron

element, charge, electrons = parse_ion_notation('Fe II')
# Returns: ('Fe', 1, 25)  # Fe+

element, charge, electrons = parse_ion_notation('Au III')
# Returns: ('Au', 2, 77)  # Au2+
```

---

## Converters

### AUTOSTRUCTURE Converters

Format atomic configurations for AUTOSTRUCTURE calculations.

**Import:**
```python
from atomkit import Configuration
from atomkit.converters import configurations_to_autostructure
```

#### ✅ `configurations_to_autostructure(configurations, core=None, last_core_orbital=None, output_file=None) -> Dict`

**RECOMMENDED** - Format configurations for AUTOSTRUCTURE with complete code-agnostic approach.

**Parameters:**
- `configurations` (List[Configuration] | List[str]): Full configurations OR valence-only if core provided
- `core` (Configuration | str, optional): Core configuration to prepend to all valence configs
- `last_core_orbital` (str, optional): Last core orbital notation (e.g., '3p'). Auto-detected if None.
- `output_file` (str | Path, optional): Write formatted output to file

**Returns:** Dict with keys:
- `'configurations'`: List of configuration strings
- `'mxconf'`: Number of configurations
- `'mxvorb'`: Number of valence orbitals
- `'kcor1'`, `'kcor2'`: Core orbital flags
- `'orbitals'`: List of (n, l) tuples
- `'occupation_matrix'`: 2D list of occupations

**Example 1 - Code-Agnostic Workflow (Full Configurations):**
```python
# Step 1: Generate configurations (PHYSICS - completely code-agnostic)
ground = Configuration.from_string('1s2 2s2 2p6 3s2 3p6 3d6 4s2')
excited = ground.generate_excitations(
    target_shells=['4p', '5s'],  # General excitation method
    excitation_level=1,
    source_shells=['3d', '4s']
)

# Step 1.5: Filter/modify (still code-agnostic!)
filtered = [c for c in excited if '4p' in c.to_string()]

# Step 2: Format for AUTOSTRUCTURE (I/O - only now code-specific)
result = configurations_to_autostructure([ground] + filtered, last_core_orbital='3p')
```

**Example 2 - Core + Valence Format (Maximum Flexibility):**
```python
# Define core once
core = Configuration.from_string('1s2 2s2 2p6 3s2 3p6')

# Work with valence only (simpler!)
valence_ground = Configuration.from_string('3d6 4s2')
valence_excited = valence_ground.generate_excitations(['4p', '5s'], excitation_level=1)

# Format with core prepended automatically
result = configurations_to_autostructure(
    [valence_ground] + valence_excited,
    core=core,
    last_core_orbital='3p'
)

# Step 1.5: Filter/modify configurations if needed
filtered = [c for c in all_configs if '5s' in c.to_string()]

# Step 2: Format for AUTOSTRUCTURE (I/O - converter)
result = configurations_to_autostructure(
    filtered, 
    last_core_orbital='3p',
    output_file='fe_configs.txt'  # Optional
)

print(f"Generated {result['mxconf']} configurations")
print(f"Orbitals: {result['orbitals']}")
```

**Why this approach?**
- Physics (configuration generation) separated from I/O (formatting)
- Full control to filter/modify/combine configurations
- Can use different generation strategies and combine results
- More flexible and testable

#### ⚠️ `generate_as_configurations(...) -> Dict` - DEPRECATED

Legacy all-in-one function. Use `Configuration.generate_autostructure_configurations()` 
followed by `configurations_to_autostructure()` instead.

**Will be removed in version 2.0.**

**Parameters:**
- `ion_notation` (str): Ion notation like 'Fe I', 'Nd II'
- `ground_config` (str): Ground state configuration
- `valence_orbitals` (str): Space-separated valence orbital prefixes
- `max_n` (int): Maximum n for excitations
- `max_l_symbol` (str): Maximum l symbol ('s', 'p', 'd', 'f', ...)
- `output_file` (str | Path, optional): Write output to file

**Old way:**
```python
result = generate_as_configurations(
    'Fe I', '1s2 2s2 2p6 3s2 3p6 3d6 4s2', '3d 4s', 5, 'd'
)
# Problem: Can't manipulate configurations before formatting
```

**New way:**
```python
config = Configuration.from_string('1s2 2s2 2p6 3s2 3p6 3d6 4s2')
all_configs = config.generate_autostructure_configurations(['3d', '4s'], 5, 2)
result = configurations_to_autostructure(all_configs, last_core_orbital='3p')
# Benefit: Full control over configurations!
```

#### `format_as_input(configurations: List[str], last_core_orbital: str) -> Dict`

Low-level formatter - usually you want `configurations_to_autostructure()` instead.

**Parameters:**
- `configurations`: List of configuration strings
- `last_core_orbital`: Last core orbital notation (e.g., '3p')

**Returns:** Same dict format as `generate_as_configurations`

**Example:**
```python
configs = [
    "1s2 2s2 2p6 3s2 3p6 3d6 4s2",
    "1s2 2s2 2p6 3s2 3p6 3d6 4s1 4p1",
    "1s2 2s2 2p6 3s2 3p6 3d5 4s2 4p1"
]

result = format_as_input(configs, last_core_orbital='3p')
```

---

## Readers

### FAC Level Reader

Read energy levels from FAC (Flexible Atomic Code) output.

**Import:**
```python
from atomkit.readers import read_fac_levels
```

#### `read_fac_levels(filename, energy_unit='eV', ...) -> DataFrame`

**Parameters:**
- `filename` (str | Path): FAC `.lev.asc` file
- `energy_unit` (str): 'eV', 'Ry', 'cm-1', 'Ha'
- `parse_config` (bool): Parse configuration strings (default: True)
- `parse_term` (bool): Parse term symbols (default: True)

**Returns:** Pandas DataFrame with columns:
- `iLev`: Level index
- `Energy_{unit}`: Energy in requested unit
- `P`: Parity
- `VNL`: Configuration
- `2J`: 2*J quantum number
- `config`: Parsed configuration (if parse_config=True)
- `term`: Term symbol (if parse_term=True)

**Example:**
```python
levels = read_fac_levels("fe_levels.lev.asc", energy_unit="eV")
print(levels[['iLev', 'Energy_eV', 'config', 'term']].head())
```

### FAC Transition Reader

Read radiative transitions from FAC output.

**Import:**
```python
from atomkit.readers import read_fac_transitions
```

#### `read_fac_transitions(filename, wavelength_unit='A', ...) -> DataFrame`

**Parameters:**
- `filename` (str | Path): FAC `.tr.asc` file
- `wavelength_unit` (str): 'A' (Angstrom), 'nm', 'eV'
- `energy_unit` (str): 'eV', 'Ry', 'cm-1'

**Returns:** Pandas DataFrame with columns:
- `upper_index`, `lower_index`: Level indices
- `wavelength_{unit}`: Transition wavelength
- `gf`: Oscillator strength
- `A`: Einstein A coefficient (s⁻¹)
- `config_initial`, `config_final`: Configurations
- And more...

**Example:**
```python
transitions = read_fac_transitions("fe_trans.tr.asc", wavelength_unit="nm")
strong = transitions[transitions['gf'] > 0.1]
print(strong[['wavelength_nm', 'gf', 'config_initial', 'config_final']])
```

---

## Physics

### Energy Conversion

Convert between different energy units.

**Import:**
```python
from atomkit.physics import energy_converter
```

#### Supported Units
- `'eV'`: Electron volts
- `'Ry'`: Rydberg
- `'cm-1'`: Wavenumbers (cm⁻¹)
- `'Ha'`: Hartree
- `'J'`: Joules
- `'Hz'`: Hertz (frequency)
- `'K'`: Kelvin (temperature)

#### Methods

##### `convert(value, from_unit, to_unit)`
Universal conversion between any supported units.

**Example:**
```python
energy_cm = energy_converter.convert(13.6, from_unit='eV', to_unit='cm-1')
temp_k = energy_converter.convert(1.0, from_unit='eV', to_unit='K')
```

##### Direct Conversion Methods
```python
# To eV
energy_converter.rydberg_to_ev(ry_value)
energy_converter.wavenumber_to_ev(cm_value)
energy_converter.hartree_to_ev(ha_value)

# From eV
energy_converter.ev_to_rydberg(ev_value)
energy_converter.ev_to_wavenumber(ev_value)
energy_converter.ev_to_hartree(ev_value)
energy_converter.ev_to_joules(ev_value)
energy_converter.ev_to_kelvin(ev_value)
```

---

## Constants and Definitions

### Angular Momentum

**Import:**
```python
from atomkit.definitions import ANGULAR_MOMENTUM_MAP, L_QUANTUM_MAP, L_SYMBOLS
```

#### `ANGULAR_MOMENTUM_MAP`
Dict mapping l symbols to quantum numbers.

```python
ANGULAR_MOMENTUM_MAP = {
    's': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, ...
}
```

#### `L_QUANTUM_MAP`
Dict mapping quantum numbers to l symbols.

```python
L_QUANTUM_MAP = {
    0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', ...
}
```

#### `L_SYMBOLS`
List of l symbols in order.

```python
L_SYMBOLS = ['s', 'p', 'd', 'f', 'g', 'h', 'i', ...]
```

---

## Best Practices

### 1. Always Use Configuration/Shell Classes

**❌ Don't do this:**
```python
# Manual parsing is error-prone
n, l, occ = int(s[0]), ANGULAR_MOMENTUM_MAP[s[1]], int(s[2:])
```

**✅ Do this:**
```python
from atomkit import Shell
shell = Shell.from_string("3d10")
n, l, occ = shell.n, shell.l_quantum, shell.occupation
```

### 2. Use Element Utilities

**❌ Don't do this:**
```python
# Manual atomic number lookup
ELEMENTS = {'H': 1, 'He': 2, 'Li': 3, ...}
Z = ELEMENTS['Fe']
```

**✅ Do this:**
```python
from atomkit import get_element_info
info = get_element_info('Fe')
Z = info['Z']
```

### 3. Use Ion Notation Parsing

**❌ Don't do this:**
```python
# Manual Roman numeral conversion
import roman
charge = roman.fromRoman('II') - 1
```

**✅ Do this:**
```python
from atomkit import parse_ion_notation
element, charge, electrons = parse_ion_notation('Fe II')
```

### 4. Leverage Configuration Methods

**❌ Don't do this:**
```python
# Manual excitation generation
for source in source_shells:
    for target in target_shells:
        # Complex logic to generate excitations...
```

**✅ Do this:**
```python
excited = config.generate_excitations(
    source_shells=["2p"],
    target_shells=["3s", "3p", "3d"],
    num_electrons=1
)
```

---

## Common Patterns

### Pattern: Working with Ions

```python
from atomkit import Configuration, parse_ion_notation

# Parse ion notation
element, charge, electrons = parse_ion_notation('Fe II')

# Create configuration
config = Configuration.from_element('Fe', ion_charge=charge)

# Verify electron count
assert config.total_electrons() == electrons
```

### Pattern: Generate and Format AS Configurations

```python
from atomkit.converters import generate_as_configurations

# One-liner to generate AUTOSTRUCTURE input
result = generate_as_configurations(
    'Fe I',
    '1s2 2s2 2p6 3s2 3p6 3d6 4s2',
    '3d 4s',
    max_n=5,
    max_l_symbol='d',
    output_file='fe_as.txt'
)
```

### Pattern: Read and Analyze FAC Data

```python
from atomkit.readers import read_fac_levels, read_fac_transitions
import pandas as pd

# Read levels and transitions
levels = read_fac_levels("data.lev.asc", energy_unit="eV")
transitions = read_fac_transitions("data.tr.asc", wavelength_unit="nm")

# Merge and analyze
merged = transitions.merge(
    levels[['iLev', 'config', 'term']],
    left_on='upper_index',
    right_on='iLev'
)

# Find strong transitions
strong = merged[merged['gf'] > 0.1]
print(strong[['wavelength_nm', 'gf', 'config', 'term']])
```

---

## Migration from Old API

If you have code using removed functions:

### ❌ Removed: `parse_orbital_string()`

**Old:**
```python
from atomkit.converters import parse_orbital_string
n, l_symbol, occ = parse_orbital_string('3d10')
```

**New:**
```python
from atomkit import Shell
shell = Shell.from_string('3d10')
n, l, occ = shell.n, shell.l_quantum, shell.occupation
```

### ❌ Removed: `validate_configuration()`

**Old:**
```python
from atomkit.converters import validate_configuration
is_valid = validate_configuration('1s2 2s2 2p6', expected_electrons=10)
```

**New:**
```python
from atomkit import Configuration
try:
    config = Configuration.from_string('1s2 2s2 2p6')
    is_valid = (config.total_electrons() == 10)
except ValueError:
    is_valid = False
```

### ❌ Removed: `generate_single_excitations()`

**Old:**
```python
from atomkit.converters import generate_single_excitations
excitations = generate_single_excitations(
    '1s2 2s2 2p6 3s2 3p6 3d6 4s2',
    '3d 4s',
    max_n=5,
    max_l_symbol='d',
    expected_electrons=26
)
```

**New:**
```python
from atomkit import Configuration
config = Configuration.from_string('1s2 2s2 2p6 3s2 3p6 3d6 4s2')
all_configs = config.generate_autostructure_configurations(
    valence_shells=['3d', '4s'],
    max_n=5,
    max_l=2  # 'd' = 2
)
excitations = [c.to_string(separator=' ') for c in all_configs[1:]]
```

---

## Further Reading

- [Quick Start Guide](quick_start.md) - Get started in 5 minutes
- [Examples Directory](../examples/) - Working code examples
- [Test Suite](../tests/) - Comprehensive test coverage showing usage patterns
- [GitHub Repository](https://github.com/rfsilva13/atomkit) - Source code and issues

---

**Questions or Issues?**  
Contact: rfsilva@lip.pt  
GitHub: https://github.com/rfsilva13/atomkit/issues
