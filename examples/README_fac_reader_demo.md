# FAC Reader Demonstration

This example demonstrates how to use AtomKit's FAC readers to load and analyze output from FAC (Flexible Atomic Code) calculations.

## Overview

The `fac_reader_demo.py` script shows how to:
- Load energy levels from FAC `.lev.asc` files
- Load radiative transitions from FAC `.tr.asc` files
- Load autoionization rates from FAC `.ai.asc` files
- Perform basic analysis on the loaded data
- Extract insights like strongest transitions and fastest autoionization rates

## Running the Example

```bash
# Make sure you're in the atomkit environment
micromamba activate atomkit

# Run the demonstration
cd /home/rfsilva/Programs/atomkit
python examples/fac_reader_demo.py
```

## What It Demonstrates

### 1. Data Loading
```python
from atomkit.readers import read_fac, read_fac_transitions, read_fac_autoionization

# Load energy levels (note: no file extension needed)
levels = read_fac('examples/fac_outputs/fe24')

# Load radiative transitions
transitions = read_fac_transitions('examples/fac_outputs/fe24_ai')

# Load autoionization rates
ai_rates = read_fac_autoionization('examples/fac_outputs/fe24_ai')
```

### 2. Data Structure
All FAC readers return pandas DataFrames with standardized column names:

**Energy Levels:**
- `level_index`: FAC level index
- `energy`: Energy in eV
- `configuration`: Electron configuration string
- `term`: Term designation
- `conf_detail`, `rel_config`, `label`: Additional FAC metadata

**Radiative Transitions:**
- `level_index_lower/upper`: Lower/upper level indices
- `lambda`: Wavelength in Å
- `gf`: Oscillator strength × degeneracy
- `A`: Einstein A coefficient (s⁻¹)
- `S`: Line strength
- `type`: Multipole type (E1, E2, M1, etc.)

**Autoionization Rates:**
- `level_index_upper/lower`: Upper/lower level indices
- `ai_rate`: Autoionization rate (s⁻¹)
- `dc_rate`: Dielectronic capture rate (s⁻¹)

### 3. Analysis Examples

The script demonstrates:
- **Statistics**: Energy ranges, transition counts, rate distributions
- **Strongest transitions**: Top transitions by oscillator strength
- **Fastest autoionization**: Highest autoionization rates
- **Configuration analysis**: Most common electron configurations
- **Energy level distribution**: Levels grouped by energy ranges

## Sample Output

```
======================================================================
AtomKit FAC Reader Demonstration
======================================================================

1. Loading FAC Energy Levels
----------------------------------------
✓ Loaded 592 energy levels
✓ Columns: ['level_index', 'energy', 'configuration', 'term', ...]

Sample energy levels:
  Level 10: 6970.8 eV, 1s.2p.3p.3d, -
  Level 11: 6972.7 eV, 1s.2p.3p.3d, -
  ...

2. Loading FAC Radiative Transitions
----------------------------------------
✓ Loaded 38320 radiative transitions

Sample radiative transitions:
  10 → 11: λ=6636.8 Å, gf=6.94e-06, A=1.50e+02 s⁻¹, E0
  11 → 12: λ=2634.8 Å, gf=2.50e-05, A=2.67e+03 s⁻¹, E0
  ...

3. Loading FAC Autoionization Rates
----------------------------------------
✓ Loaded 5222 autoionization rates

Sample autoionization rates:
  Level 10.0 → 1.0: A_i = 1.48e+10 s⁻¹
  Level 10.0 → 2.0: A_i = 6.34e+09 s⁻¹
  ...
```

## Data Source

The example uses output from a simplified Fe XXIV autoionization calculation:
- **Element**: Fe XXIV (Lithium-like Iron)
- **Target configurations**: 3 (N-electron states)
- **Autoionizing configurations**: 4 (N+1-electron states)
- **Calculation**: Energy levels, radiative transitions, autoionization rates

## Files Used

- `examples/fac_outputs/fe24.lev.asc` - Energy levels
- `examples/fac_outputs/fe24_ai.tr.asc` - Radiative transitions
- `examples/fac_outputs/fe24_ai.ai.asc` - Autoionization rates

## Applications

The loaded data can be used for:
- **Spectral analysis**: Line identification and synthesis
- **Plasma physics**: Level populations and ionization balance
- **Atomic databases**: CHIANTI/ADAS format export
- **Visualization**: Grotrian diagrams and cross sections
- **Research**: Comparative studies between different codes

## Notes

- FAC readers automatically handle file splitting for large datasets
- All data is returned as pandas DataFrames for easy analysis
- Column names are standardized across all FAC output types
- Energy units are consistently in eV, wavelengths in Å
- Autoionization rates are in s⁻¹ (per second)
