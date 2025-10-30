# AtomKit Cleanup Plan

## Current Situation
- 529 tests archived to `tests_archive/`
- Multiple overlapping test files
- Confusing documentation
- Need to start fresh with minimal, sensible structure

## Essential Core Modules (Keep These)

### 1. Configuration & Shell (`configuration.py`, `shell.py`)
**What it does**: Represent electron configurations
**Why essential**: Core of everything - "1s2 2s2 2p6" parsing
**Status**: ✅ Working, well-tested

### 2. Utils (`utils.py`, `definitions.py`)
**What it does**: Element info, ion notation, constants
**Why essential**: Basic atomic data lookups
**Status**: ✅ Working

### 3. Core Unified Interface (`core/`)
**What it does**: Agnostic `AtomicCalculation` class
**Why essential**: Main user-facing API
**Status**: ✅ Working, just completed

### 4. Physics (`physics/`)
**What it does**: Unit conversion, cross sections, potentials
**Why essential**: Basic physics calculations
**Status**: ✅ Working

### 5. Readers (`readers/`)
**What it does**: Parse output files (FAC, AUTOSTRUCTURE)
**Why essential**: Read calculation results
**Status**: ✅ Working

### 6. Converters (`converters/`)
**What it does**: Convert between formats
**Why essential**: Bridge different codes
**Status**: ✅ Working

### 7. Backend Writers (`autostructure/`, `fac/`)
**What it does**: Generate input files
**Why essential**: Write calculations
**Status**: ✅ Working

## Minimal Test Suite (What We Actually Need)

### Essential Tests Only

1. **`test_configuration.py`** (Simple version)
   - Parse configurations from string
   - Generate excitations
   - Basic operations (add electrons, remove, etc.)
   - ~20-30 tests

2. **`test_shell.py`**
   - Parse shell notation
   - Basic shell properties
   - ~10-15 tests

3. **`test_unified_interface.py`** (Simplified)
   - Create calculations with AtomicCalculation
   - Write input files for AS and FAC
   - ~15-20 tests

4. **`test_physics.py`** (Core functionality)
   - Unit conversions
   - Cross section calculations
   - ~10-15 tests

5. **`test_readers.py`**
   - Read FAC levels
   - Read FAC transitions
   - ~10-15 tests

**Total: ~70-100 focused tests** instead of 529!

## Documentation Cleanup

### Keep
- `README.md` (main entry point)
- `docs/API_REFERENCE.md` (if up to date)
- `examples/` (working examples)

### Archive
- `WORK_SUMMARY.md` → Move to docs_archive/
- `QUICK_REFERENCE.md` → Move to docs_archive/
- `AUTOS_TESTS_README.md` → Move to docs_archive/
- Old phase documentation

## Action Plan

1. ✅ Archive all tests to `tests_archive/`
2. ⏳ Create minimal test suite (5 files, ~80 tests)
3. ⏳ Archive excessive documentation
4. ⏳ Update README to reflect clean structure
5. ⏳ Run tests to verify everything works

## Philosophy
- **Less is more**: Focus on what matters
- **Clear purpose**: Each test tests ONE thing
- **Simple syntax**: No over-engineering
- **Easy to understand**: New users should get it immediately
