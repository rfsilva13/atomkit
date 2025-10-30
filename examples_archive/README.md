# Examples Archive

This directory contains additional example scripts that demonstrate specific features of AtomKit. These are archived to keep the main `examples/` directory minimal and focused.

## Archived Examples

### Configuration Generation
- **advanced_configuration_generation.py** - Advanced configuration generation techniques
- **explicit_parameters_demo.py** - Using explicit AUTOSTRUCTURE parameters

### AUTOSTRUCTURE Demos
- **autostructure_ux_examples.py** - User experience examples for AUTOSTRUCTURE
- **autostructure_wrapper_demo.py** - Low-level AUTOSTRUCTURE wrapper usage

### FAC Demos
- **fac_wrapper_demo.py** - FAC wrapper demonstrations

### Optimization
- **lambda_optimization_demo.py** - Lambda scaling optimization examples

### Converters
- **fac_to_as_converter.py** - Converting FAC inputs to AUTOSTRUCTURE
- **ls_to_icr_converter.py** - Converting LS coupling to ICR

## Main Examples (in examples/)

For everyday usage, see the main `examples/` directory which contains:

1. **basic_usage.py** - Start here for basic AtomKit usage
2. **autostructure_workflow.py** - Complete AUTOSTRUCTURE workflow
3. **unified_comparison.py** - Comparing different codes/approaches

## Using Archived Examples

These examples are still functional and can be run:

```bash
cd /home/rfsilva/Programs/atomkit/examples_archive
micromamba run -n atomkit python <example_name>.py
```

## Date Archived

October 21, 2025
