# Configuration Class Unit Tests

This directory contains comprehensive unit tests for the `Configuration` class in the atomkit library.

## Overview

The `Configuration` class represents the electron configuration of an atom or ion using a collection of Shell objects. The tests verify all public methods and edge cases.

## Test File

- `test_configuration.py` - Main test suite with 67 comprehensive tests

## Running Tests

### Using pytest (recommended)
```bash
cd /home/rfsilva/EIEres/atomkit
python -m pytest tests/test_configuration.py -v
```

### Using the custom test runner
```bash
cd /home/rfsilva/EIEres/atomkit
python run_tests.py
```

## Test Coverage

The test suite covers all major functionality of the Configuration class:

### Core Functionality
- ✅ Configuration initialization (empty, with shells, duplicate shells)
- ✅ Shell addition, removal, and manipulation 
- ✅ Shell retrieval and property access
- ✅ Total electron calculation
- ✅ Deep copying

### String Parsing and Creation
- ✅ Standard notation parsing (`1s2.2s2.2p6`)
- ✅ Compact notation parsing (`1*2.2*8`)
- ✅ Element-based configuration creation (using mendeleev)
- ✅ Ionization and charge calculations

### Advanced Operations
- ✅ Hole identification and calculation
- ✅ Excitation generation (single and multiple level)
- ✅ Core/valence splitting
- ✅ Configuration comparison
- ✅ X-ray notation labeling

### Magic Methods and Utilities
- ✅ String representation (`__str__`, `__repr__`)
- ✅ Equality and hashing (`__eq__`, `__hash__`)
- ✅ Container operations (`__len__`, `__iter__`, `__contains__`)
- ✅ Sorting and ordering

### Error Handling
- ✅ Invalid input validation
- ✅ Type checking
- ✅ Edge case handling
- ✅ Dependency management (graceful mendeleev handling)

## Dependencies

The tests require:
- `pytest` - Testing framework
- `mendeleev` - Element data (installed automatically)
- `scipy` - Scientific constants

## Test Results

All 67 tests pass successfully, demonstrating that the Configuration class is:
- ✅ Functionally correct
- ✅ Robust against edge cases
- ✅ Properly handling errors
- ✅ Following expected behavior patterns

## Environment

Tests are designed to run in the `kilopacities` conda environment and are confirmed working with:
- Python 3.11.5
- pytest 7.4.0
- mendeleev 1.1.0

## Notes

- Tests that depend on the `mendeleev` package are automatically skipped if the package is not available
- Zero occupation shells are handled correctly (filtered out during initialization)
- Special cases like fully ionized atoms (H+) are properly handled
- The test suite includes both unit tests and integration tests to ensure comprehensive coverage