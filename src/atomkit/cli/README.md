# Interactive Energy Shifter CLI

A command-line tool for interactive energy shifting of atomic spectral lines using Plotly and ipywidgets.

## Installation

Make sure you have atomkit installed with the required dependencies:

```bash
pip install atomkit[all]  # Includes plotting dependencies
# or
pip install ipywidgets plotly pandas numpy
```

## Usage

```bash
# Basic usage - analyze 1s shell with all features enabled
interactive-plotter input_files/Pd

# Analyze 2s shell without satellite lines
interactive-plotter input_files/Pd --shell 2s --no-satellites

# Custom energy range and disable Lorentzian broadening
interactive-plotter input_files/Pd --energy-min 20000 --energy-max 22000 --no-lorentzian

# Disable shake-off corrections and save outputs
interactive-plotter input_files/Pd --no-shakeoff --output-dir results/

# Full options
interactive-plotter input_files/Pd \
    --shell 1s \
    --no-satellites \
    --no-shakeoff \
    --no-lorentzian \
    --energy-min 20500 \
    --energy-max 21800 \
    --energy-grid-points 5000 \
    --bin-width 0.5 \
    --output-dir ./output \
    --verbose
```

## Command Line Options

- `input_file`: Base filename for input data (required)
- `--shell`: Shell to analyze (default: 1s)
- `--no-satellites`: Exclude satellite lines from analysis
- `--no-shakeoff`: Disable shake-off corrections for diagram lines
- `--no-lorentzian`: Disable Lorentzian broadening (show delta functions only)
- `--energy-min`: Minimum energy for spectrum (default: 20500)
- `--energy-max`: Maximum energy for spectrum (default: 21800)
- `--energy-grid-points`: Number of points in energy grid (default: 5000)
- `--bin-width`: Bin width for binned spectrum in eV (default: 0.5)
- `--output-dir`: Directory to save output files
- `--verbose, -v`: Enable verbose output

## Output Files

When `--output-dir` is specified, the following files are saved:

- `diagram_lines.csv`: All diagram line data
- `satellite_lines.csv`: All satellite line data (if enabled)
- `spectrum_lorentzian.csv`: Lorentzian-broadened spectrum (if enabled)

## Interactive Features

The tool creates an interactive Plotly plot with the following controls:

- **Diagram Shift**: Adjusts all diagram line energies
- **Satellite Shift**: Adjusts all satellite line energies
- **X Min/Max**: Control the energy range displayed
- **Reset button**: Return all shifts to zero

The plot shows:
- Top panel: Individual spectral lines (delta functions)
- Bottom panel: Lorentzian-broadened spectrum (if enabled)

## Requirements

- Python 3.10+
- ipywidgets
- plotly
- pandas
- numpy
- atomkit

## Examples

### Compare Different Shells

```bash
# Compare K-shell (1s) and L-shell (2s) spectra
interactive-plotter input_files/Pd --shell 1s --output-dir k_shell/
interactive-plotter input_files/Pd --shell 2s --output-dir l_shell/
```

### Quick Analysis Without Broadening

```bash
# Fast analysis with delta functions only
interactive-plotter input_files/Pd --no-lorentzian --energy-grid-points 1000
```

### Full Analysis with All Corrections

```bash
# Complete analysis with all physics included
interactive-plotter input_files/Pd --verbose --output-dir full_analysis/
```