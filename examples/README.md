# AtomKit Examples

This directory contains working examples demonstrating the key features of AtomKit.

## Examples Overview

### Basic Usage
- `basic_usage.py` - Introduction to Configuration and Shell classes
- `configuration_manipulation.py` - Creating and manipulating electron configurations
- `element_configurations.py` - Working with element-based configurations

### FAC Data Reading
- `read_fac_levels.py` - Reading and analyzing FAC energy level files
- `read_fac_transitions.py` - Reading FAC transition data
- `fac_unit_conversion.py` - Unit conversion examples

### Physics Calculations
- `energy_conversion.py` - Converting between energy units
- `cross_sections.py` - Cross section and collision strength calculations
- `plotting_cross_sections.py` - Creating publication-quality plots

### Advanced Features
- `excited_states.py` - Generating excited configurations
- `hole_states.py` - Creating hole configurations
- `autoionizing_states.py` - Working with autoionizing configurations

## Running Examples

Make sure you have activated the atomkit environment first:

```bash
# Activate environment
conda activate atomkit

# Run an example
python examples/basic_usage.py
```

## Requirements

All examples require the core AtomKit installation. Some examples may need optional dependencies:

- Plotting examples require: `matplotlib`
- Some examples may need sample data files (`.lev.asc`, `.tr.asc`)

Install optional dependencies:
```bash
pip install matplotlib seaborn
```

## Sample Data

Some examples reference sample FAC output files. You can use your own FAC data files or create simple test files as shown in the examples.
