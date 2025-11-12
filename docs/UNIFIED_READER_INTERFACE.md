# Unified Reader Interface for AtomKit

## Overview

The unified reader interface provides **format-agnostic** functions that automatically detect whether atomic structure data files are from **AUTOSTRUCTURE** or **FAC** (Flexible Atomic Code) and extract data using the appropriate reader.

## Key Features

✅ **Automatic Format Detection** - No need to know the file format  
✅ **Consistent API** - Same functions work with different atomic codes  
✅ **Comprehensive Data Extraction** - Levels, transitions, terms, autoionization  
✅ **Metadata Preservation** - Atomic system information included  
✅ **Future Extensibility** - Easy to add support for other atomic codes  

---

## Quick Start

```python
from atomkit.readers import get_levels, get_transitions, get_terms

# Works with both AUTOSTRUCTURE and FAC files
df_levels, metadata = get_levels('atomic_data_file')
df_transitions, metadata = get_transitions('atomic_data_file')
df_terms, metadata = get_terms('atomic_data_file')
```

---

## Available Functions

### 1. `detect_file_format(filename)`

Automatically detects the format of an atomic structure file.

**Parameters:**
- `filename`: Path to the atomic structure file

**Returns:**
- `str`: Format identifier ('autostructure', 'fac', or 'unknown')

**Detection Methods:**
1. **File extension**: `.olg`, `.ols` → AUTOSTRUCTURE; `.sf`, `.dat` → FAC
2. **File name**: `OLG`, `TERMS` → AUTOSTRUCTURE; `SF`, `DAT` → FAC
3. **File content**: Keywords like 'AUTOSTRUCTURE', 'LEVEL TABLE', 'FAC', etc.

**Example:**
```python
from atomkit.readers import detect_file_format

format_type = detect_file_format('output.olg')
print(f"Detected format: {format_type}")  # Output: autostructure
```

---

### 2. `get_levels(filename, output_file=None, coupling=None)`

Unified interface to extract level data from atomic structure files.

**Parameters:**
- `filename`: Path to the atomic structure file
- `output_file`: Optional path to save parsed data
- `coupling`: Optional coupling scheme preference ('jj', 'ls', or None for auto-detect)

**Returns:**
- `df_levels`: pandas DataFrame with level data
- `metadata`: Dictionary with atomic structure metadata

**DataFrame Columns (AUTOSTRUCTURE JJ coupling):**
- `K`: Level index
- `CF`: Configuration string
- `Level (Ry)`: Absolute energy in Rydbergs
- `2J`: Total angular momentum quantum number (2J)
- `2*S+1`: Spin multiplicity
- `L`: Orbital angular momentum quantum number
- `P`: Parity (0=even, 1=odd)

**DataFrame Columns (FAC):**
- `index`: Level index
- `configuration`: Electronic configuration
- `term`: Term symbol
- `J`: Total angular momentum
- `energy`: Energy level
- `g`: Statistical weight

**Example:**
```python
from atomkit.readers import get_levels

# AUTOSTRUCTURE file
df_levels, meta = get_levels('test_2/olg')
print(f"Found {len(df_levels)} levels")
print(f"Atomic system: Z={meta['Atomic number']}, N={meta['Number of electrons']}")

# Access level data
for idx, row in df_levels.head(3).iterrows():
    print(f"Level {row['K']}: 2J={row['2J']}, Energy={row['Level (Ry)']:.3f} Ry")
```

---

### 3. `get_transitions(filename, output_file=None)`

Unified interface to extract radiative transition data.

**Parameters:**
- `filename`: Path to the atomic structure file
- `output_file`: Optional path to save parsed data

**Returns:**
- `df_transitions`: pandas DataFrame with transition data
- `metadata`: Dictionary with atomic structure metadata

**DataFrame Columns (AUTOSTRUCTURE):**
- `index`: Transition index
- `K`: Upper level index
- `Klower`: Lower level index
- `WAVEL/AE`: Wavelength in Angstroms
- `A(K)*SEC`: Einstein A coefficient (s⁻¹)
- `F(ABS)`: Absorption oscillator strength
- `log(gf)`: Log of weighted oscillator strength

**DataFrame Columns (FAC):**
- `atomic_number`: Atomic number (Z)
- `level_index_lower`: Lower level index
- `level_index_upper`: Upper level index
- `2j_lower`: 2J value for lower level
- `2j_upper`: 2J value for upper level
- `energy`: Transition energy (eV)
- `lambda`: Wavelength (Å)
- `gf`: Weighted oscillator strength
- `A`: Einstein A coefficient (s⁻¹)
- `S`: Line strength (atomic units)
- `multipole`: Multipole mixing coefficient from file
- `type`: Transition type ('E1', 'M1', 'E2', etc., or 'ALL' for sum over all multipoles when MULTIP=0)

**Example:**
```python
from atomkit.readers import get_transitions

df_trans, meta = get_transitions('test_2/olg')
print(f"Found {len(df_trans)} transitions")

# Access transition data
for idx, row in df_trans.head(3).iterrows():
    print(f"{row['K']} ← {row['Klower']}: λ={row['WAVEL/AE']:.2f} Å, A={row['A(K)*SEC']:.2e} s⁻¹")
```

---

### 4. `get_terms(filename, output_file=None)`

Unified interface to extract term-averaged energies.

**Parameters:**
- `filename`: Path to the TERMS file or olg file
- `output_file`: Optional path to save parsed data

**Returns:**
- `df_terms`: pandas DataFrame with term data
- `metadata`: Dictionary with atomic structure metadata

**DataFrame Columns (AUTOSTRUCTURE):**
- `2*S+1`: Spin multiplicity
- `L`: Orbital angular momentum quantum number
- `P`: Parity (0=even, 1=odd)
- `CF`: Configuration index
- `NI`: Number of levels in term
- `Energy (Ry)`: Term energy in Rydbergs
- `Configuration`: Configuration string (if available)

**Example:**
```python
from atomkit.readers import get_terms

df_terms, meta = get_terms('test_3/TERMS')
print(f"Found {len(df_terms)} terms")
print(f"Energy range: {df_terms['Energy (Ry)'].min():.2f} to {df_terms['Energy (Ry)'].max():.2f} Ry")

# Access term data
for idx, row in df_terms.iterrows():
    multiplicity = row['2*S+1']
    L_symbol = ['S', 'P', 'D', 'F', 'G'][row['L']]
    parity = 'o' if row['P'] == 1 else 'e'
    print(f"{multiplicity}{L_symbol}{parity}: Energy={row['Energy (Ry)']:.3f} Ry")
```

---

### 5. `get_autoionization(filename, output_file=None)`

Unified interface to extract autoionization data (FAC only).

**Parameters:**
- `filename`: Path to the autoionization file
- `output_file`: Optional path to save parsed data

**Returns:**
- `df_auto`: pandas DataFrame with autoionization data
- `metadata`: Dictionary with atomic structure metadata

**Example:**
```python
from atomkit.readers import get_autoionization

df_auto, meta = get_autoionization('autoionization.sf')
print(f"Found {len(df_auto)} autoionization transitions")
```

---

## Format-Specific Readers

While the unified interface is recommended, you can still use format-specific readers for more control:

### AUTOSTRUCTURE Readers
```python
from atomkit.readers import (
    read_as_levels,      # JJ coupling levels
    read_as_transitions, # Radiative transitions
    read_as_terms,       # Term energies
    read_as_lambdas      # Lambda scaling parameters
)
```

### FAC Readers
```python
from atomkit.readers import (
    read_fac,                  # Levels
    read_fac_transitions,      # Transitions
    read_fac_autoionization    # Autoionization
)
```

---

## Data Hierarchy

AUTOSTRUCTURE provides multiple levels of detail:

| Data Type | Function | Description | Example Output |
|-----------|----------|-------------|----------------|
| **JJ Levels** | `get_levels()` | Individual levels with complete quantum numbers | 10 levels with K, 2J, 2*S+1, L, P |
| **Transitions** | `get_transitions()` | Radiative transitions between levels | 16 transitions with wavelengths, A-values |
| **LS Levels** | (skipped by default) | Term-averaged levels | Not extracted by current implementation |
| **Terms** | `get_terms()` | Term energies with quantum numbers | 7 terms with multiplicity, L, parity |

---

## Metadata Structure

All unified functions return consistent metadata:

```python
metadata = {
    "Atomic number": int,           # Atomic number Z
    "Number of electrons": int,     # Number of electrons
    "Closed": str,                  # Closed shell configuration
    "Ground state energy (Ry)": float,  # Ground state energy
    "CPU time": str,                # CPU time (AUTOSTRUCTURE only)
    "Total CPU time": str,          # Total CPU time (AUTOSTRUCTURE only)
    "Method": str                   # 'FAC' or not present for AUTOSTRUCTURE
}
```

---

## Best Practices

1. **Use unified interface for general tasks:**
   ```python
   df_levels, meta = get_levels('datafile')  # Auto-detects format
   ```

2. **Check format if needed:**
   ```python
   format_type = detect_file_format('datafile')
   if format_type == 'autostructure':
       # Handle AUTOSTRUCTURE-specific features
   ```

3. **Validate data extraction:**
   ```python
   df_levels, meta = get_levels('datafile')
   print(f"Extracted {len(df_levels)} levels for Z={meta['Atomic number']}")
   ```

4. **Handle different column names:**
   ```python
   # AUTOSTRUCTURE has 'K' for level index
   # FAC has 'index' for level index
   if 'K' in df_levels.columns:
       level_index = df_levels['K']
   elif 'index' in df_levels.columns:
       level_index = df_levels['index']
   ```

---

## Testing

Run the comprehensive test script:

```bash
python test_unified_interface.py
```

This tests:
- ✅ Format detection for AUTOSTRUCTURE and FAC files
- ✅ Level extraction from both formats
- ✅ Transition extraction
- ✅ Term extraction
- ✅ Data structure consistency

---

## Validation Results

### AUTOSTRUCTURE (test_2):
- **Format detection**: ✅ Correctly identified as 'autostructure'
- **Levels**: ✅ 10 JJ coupling levels extracted
- **Transitions**: ✅ 16 radiative transitions extracted
- **Metadata**: ✅ Z=6, N=4, Ground state = -72.877 Ry

### AUTOSTRUCTURE (test_3):
- **Format detection**: ✅ Correctly identified as 'autostructure'
- **Terms**: ✅ 7 terms extracted from TERMS file
- **Energy range**: ✅ -72.89 to 1.78 Ry

### FAC:
- **Format detection**: ✅ Correctly identified as 'fac'
- **Note**: Full testing requires properly named output files (.lev.asc, .tr.asc)

---

## Future Enhancements

Planned improvements:
- [ ] Add `coupling` parameter support for LS vs JJ preference
- [ ] Enhance error messages for format detection failures
- [ ] Add file validation before processing
- [ ] Support for additional atomic codes (e.g., GRASP, MCDHF)
- [ ] Unified autoionization interface for AUTOSTRUCTURE
- [ ] Configuration-based file format hints

---

## Migration Guide

If you're using format-specific readers, migration is simple:

**Before:**
```python
from atomkit.readers import read_as_levels
df_levels, meta = read_as_levels('output.olg')
```

**After:**
```python
from atomkit.readers import get_levels
df_levels, meta = get_levels('output.olg')  # Same result!
```

The unified interface is backward compatible and provides the same output structure.

---

## Contributing

To add support for a new atomic code:

1. Implement format-specific reader functions
2. Add detection logic to `detect_file_format()`
3. Add routing logic to unified functions (`get_levels()`, etc.)
4. Add tests to `test_unified_interface.py`
5. Update this documentation

---

## References

- AUTOSTRUCTURE: Badnell, N. R. (2011). Computer Physics Communications, 182(7), 1528-1535.
- FAC: Gu, M. F. (2008). Canadian Journal of Physics, 86(5), 675-689.

---

**Last Updated**: November 12, 2025  
**Version**: 0.1.0
