# AtomKit Examples

This directory contains working examples demonstrating the key features of AtomKit.

## Examples Overview

### Basic Usage
- `basic_usage.py` - Introduction to Configuration and Shell classes

### Configuration Generation
- `advanced_configuration_generation.py` - Comprehensive examples of advanced generation techniques
  - Single and double excitations
  - Autoionization states (doubly excited configurations)
  - Hole configurations (X-ray, Auger processes)
  - Recombined configurations (dielectronic recombination)
  - Complex filtering and combining strategies
  - Systematic generation for complete configuration spaces
  - Practical use case examples

### AUTOSTRUCTURE Integration
- `autostructure_workflow.py` - Code-agnostic workflow: generate (physics) → format (I/O)

### Format Converters
- `fac_to_as_converter.py` - Convert FAC data to AUTOSTRUCTURE format
- `ls_to_icr_converter.py` - Convert LS-coupling data to ICR format

## Note on Workflow Philosophy

AtomKit follows a clear separation of concerns:
- **Physics logic** (configuration generation, excitations, etc.) lives in `Configuration` class
- **I/O operations** (format conversion, file writing) lives in converters

This separation gives you maximum flexibility to manipulate configurations before formatting them for specific codes (AUTOSTRUCTURE, FAC, etc.).

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
