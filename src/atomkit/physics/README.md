# AtomKit Physics Module

## Overview

The `atomkit.physics` module provides tools for calculating atomic physics quantities such as electron impact excitation cross sections. It currently includes:

- **Resonant electron impact excitation** via dielectronic capture and autoionization
- Lorentzian resonance profiles
- Branching ratio calculations
- Cross section plotting utilities

## Installation

The physics module is part of atomkit. Make sure you have atomkit installed with its dependencies:

```bash
pip install numpy pandas scipy matplotlib
```

## Quick Start

### Simple Usage

```python
import numpy as np
from atomkit import readers as rd
from atomkit.physics import calculate_resonant_excitation_cross_section

# Read FAC data
levels = rd.read_fac("save/FeXXIV_ai")
autoionization = rd.read_fac_autoionization("save/FeXXIV_ai")
transitions = rd.read_fac_transitions("save/FeXXIV_ai")

# Create energy grid
energy = np.linspace(100, 7000, 5000)

# Calculate cross section (one function call!)
cross_section, info = calculate_resonant_excitation_cross_section(
    initial_level=46,  # Level index
    final_level=47,    # Level index
    levels=levels,
    autoionization=autoionization,
    transitions=transitions,
    energy_grid=energy,
)

print(f"Max cross section: {cross_section.max():.3e} cm²")
```

### Using Configuration Strings

You can also specify levels by their electronic configuration:

```python
from atomkit.physics import ResonantExcitationCalculator

calculator = ResonantExcitationCalculator(
    levels=levels,
    autoionization=autoionization,
    transitions=transitions
)

cross_section, info = calculator.calculate_resonant_excitation(
    initial_level="1s2 2s1",  # Ground state of Li-like ion
    final_level="1s2 2p1",    # First excited state
    energy_grid=energy,
    ion_charge=23  # Required when using config strings
)
```

## Physics Background

### Resonant Excitation Process

The resonant electron impact excitation process proceeds in two steps:

1. **Dielectronic Capture**: A free electron is captured into a doubly-excited autoionizing state
   ```
   A^(n+) + e^- → A^((n-1)+)**  [resonance]
   ```

2. **Autoionization**: The resonance decays back to the target ion in an excited state
   ```
   A^((n-1)+)** → A^(n+)* + e^-
   ```

### Cross Section Formula

The resonant excitation cross section is given by:

$$\sigma_{if}(E) = \sum_r \sigma_{\\text{cap}}^{(r)}(E) \\times BR_{f}^{(r)}$$

where:
- $\sigma_{\\text{cap}}^{(r)}(E)$ is the capture cross section to resonance $r$
- $BR_f^{(r)}$ is the branching ratio for decay to the final state $f$

The capture cross section has a Lorentzian (Breit-Wigner) profile:

$$\sigma_{\\text{cap}}(E) = \\frac{\pi(\hbar c)^2}{2m_e c^2 E} \\frac{g_r}{g_i} A_a^{(i \\to r)} \\frac{\Gamma/2\pi}{(E - E_r)^2 + (\Gamma/2)^2}$$

where:
- $E_r$ is the resonance energy
- $\Gamma = (A_a^{\\text{tot}} + A_r^{\\text{tot}}) \hbar$ is the total width
- $A_a^{(i \\to r)}$ is the autoionization rate from initial state to resonance
- $g_r, g_i$ are statistical weights

The branching ratio is:

$$BR_f = \\frac{A_a^{(r \\to f)}}{A_a^{\\text{tot}} + A_r^{\\text{tot}}}$$

## API Reference

### `calculate_resonant_excitation_cross_section`

Convenience function for one-off calculations.

**Parameters:**
- `initial_level` (int or str): Initial level index or configuration string
- `final_level` (int or str): Final level index or configuration string
- `levels` (pd.DataFrame): Energy levels data
- `autoionization` (pd.DataFrame): Autoionization rates
- `transitions` (pd.DataFrame): Radiative transition rates
- `energy_grid` (np.ndarray): Energy grid in eV
- `ion_charge` (int, optional): Ion charge (required if using config strings)
- `resonant_levels` (list, optional): List of specific resonances to include

**Returns:**
- `cross_section` (np.ndarray): Cross section in cm²
- `resonance_info` (dict): Information about resonance contributions

### `ResonantExcitationCalculator`

Main class for resonant excitation calculations.

#### Methods

##### `__init__(levels, autoionization, transitions)`

Initialize calculator with atomic data.

##### `calculate_resonant_excitation(...)`

Calculate total resonant excitation cross section. Same parameters as the convenience function above.

##### `calculate_capture_cross_section(initial_level, resonant_level_index, energy_grid, ...)`

Calculate electron capture cross section to a specific resonant state.

**Returns:**
- `capture_cs` (np.ndarray): Capture cross section in cm²
- `resonance_energy` (float): Resonance energy in eV
- `gamma_total` (float): Total resonance width in eV

##### `calculate_branching_ratio(resonant_level_index, final_level, ...)`

Calculate branching ratio for decay to a specific final state.

**Returns:**
- `branching_ratio` (float): Probability of decay to final state

##### `get_level_by_index(level_index)`

Get level data by level index.

##### `get_level_by_config(configuration, ion_charge=None)`

Get level data by electronic configuration string.

##### `plot_cross_section(energy_grid, cross_section, ax=None, **kwargs)`

Plot cross section vs energy.

### `LorentzianProfile`

Class for Lorentzian (Breit-Wigner) resonance profiles.

```python
profile = LorentzianProfile(energy_center=6000, gamma=10)
values = profile(energy_array)
```

## Examples

### Example 1: Basic Calculation

```python
import numpy as np
from atomkit import readers as rd
from atomkit.physics import calculate_resonant_excitation_cross_section

# Load data
levels = rd.read_fac("save/FeXXIV_ai")
autoionization = rd.read_fac_autoionization("save/FeXXIV_ai")
transitions = rd.read_fac_transitions("save/FeXXIV_ai")

# Energy grid
energy = np.linspace(100, 7000, 5000)

# Calculate
cs, info = calculate_resonant_excitation_cross_section(
    initial_level=46,
    final_level=47,
    levels=levels,
    autoionization=autoionization,
    transitions=transitions,
    energy_grid=energy,
)
```

### Example 2: Analyzing Resonance Contributions

```python
from atomkit.physics import ResonantExcitationCalculator

calculator = ResonantExcitationCalculator(levels, autoionization, transitions)

cs, info = calculator.calculate_resonant_excitation(
    initial_level="2s",
    final_level="2p",
    energy_grid=energy,
    ion_charge=23
)

# Find strongest resonances
contributions = info['contributions']
peak_strengths = [c.max() for c in contributions]
top_5_indices = np.argsort(peak_strengths)[::-1][:5]

print("Top 5 resonances:")
for i, idx in enumerate(top_5_indices, 1):
    res_level = info['level_index'][idx]
    res_energy = info['energies'][idx]
    res_width = info['widths'][idx]
    strength = peak_strengths[idx]
    
    print(f"{i}. Level {res_level}: E={res_energy:.1f} eV, "
          f"Γ={res_width:.2e} eV, σ_max={strength:.2e} cm²")
```

### Example 3: Comparing Specific Resonances

```python
# Only include specific resonances
resonance_list = [130, 154, 234, 288, 307]

cs_subset, info_subset = calculator.calculate_resonant_excitation(
    initial_level=46,
    final_level=47,
    energy_grid=energy,
    resonant_levels=resonance_list
)

print(f"Included {len(resonance_list)} resonances")
print(f"Max cross section: {cs_subset.max():.3e} cm²")
```

### Example 4: Plotting

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# Plot total cross section
ax.plot(energy, cs, 'b-', linewidth=2, label='Total')

# Plot individual resonance contributions
for i, idx in enumerate(top_5_indices):
    contrib = info['contributions'][idx]
    ax.plot(energy, contrib, '--', alpha=0.6, 
            label=f"Resonance {info['level_index'][idx]}")

ax.set_xlabel('Electron Energy (eV)')
ax.set_ylabel('Cross Section (cm²)')
ax.set_title('Resonant Excitation Cross Section')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Data Requirements

The module expects pandas DataFrames with the following columns:

### `levels` DataFrame
- `level_index`: Unique level identifier
- `energy`: Level energy in eV
- `2j`: Twice the total angular momentum (for statistical weight)
- `configuration`: Electronic configuration string
- `ion_charge`: Ion charge state

### `autoionization` DataFrame  
- `level_index_upper`: Resonant (autoionizing) level index
- `level_index_lower`: Target ion level index
- `ai_rate`: Autoionization rate in s⁻¹
- `energy`: Transition energy in eV

### `transitions` DataFrame
- `level_index_upper`: Upper level index
- `level_index_lower`: Lower level index
- `A`: Einstein A coefficient (radiative decay rate) in s⁻¹

These can be read from FAC files using `atomkit.readers`:

```python
from atomkit import readers as rd

levels = rd.read_fac("filename")
autoionization = rd.read_fac_autoionization("filename")
transitions = rd.read_fac_transitions("filename")
```

## Physical Constants

The module uses the following physical constants from `scipy.constants`:

- $\hbar$ (reduced Planck constant): `scipy.constants.hbar`
- $m_e$ (electron mass): `scipy.constants.m_e`
- $e$ (elementary charge): `scipy.constants.e`
- $c$ (speed of light): `scipy.constants.c`

Derived constants:
- $m_e c^2 = 510998.9461$ eV (electron rest energy)
- Prefactor: $\pi(\hbar c)^2/(2m_e c^2) \\approx 6.65 \\times 10^{-14}$ eV²·cm²

## Notes

### Energy Grid Considerations

- Avoid starting the energy grid at exactly zero to prevent division by zero (the cross section has a 1/E dependence)
- Use `np.linspace(small_positive_value, max_energy, num_points)` instead of starting at 0
- For resonances, typical energy grids span from ~100 eV to several keV with 1000-10000 points

### Performance

- Calculation time scales with: (number of energy points) × (number of resonances)
- For ~10000 energy points and ~300 resonances: expect ~1-5 seconds
- Use `resonant_levels` parameter to limit calculation to specific resonances if needed

### Numerical Accuracy

- The Lorentzian profile is evaluated directly (no approximations)
- Very narrow resonances (Γ < 10⁻⁶ eV) may require finer energy grids
- Statistical weight factors use exact formula: $g = 2J + 1$

## Troubleshooting

### "KeyError: 'ion_charge'" or similar
Make sure your levels DataFrame has all required columns. Check with:
```python
print(levels.columns.tolist())
```

### "No direct capture channel from this initial state"
The autoionization data doesn't include a transition from the specified initial state to any resonance. Check your initial state selection.

### NaN or Inf values in cross section
- Check that energy grid doesn't include zero
- Verify that decay rates (Aa_total + Ar_total) are non-zero
- Ensure level indices match between DataFrames

### Very small or zero cross sections
- Verify that resonances exist in the energy range of interest
- Check that branching ratios are non-zero for your final state
- Confirm that statistical weights are calculated correctly

## References

1. Seaton, M. J. (1962). "Radiative Recombination of Hydrogenic Ions." Phys. Rev. 127, 1132.
2. Shore, B. W. (1969). "Phaseshift Breakdown in Electron-Atom Scattering." Rev. Mod. Phys. 41, 3.
3. Pindzola, M. S., et al. (1986). "Resonances in Electron-Atom and Electron-Ion Scattering." Phys. Rev. A 34, 4531.

## License

Part of the AtomKit package. See main package for license information.
