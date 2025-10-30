# Clean Tests - Ready to Use

## Problem
The `tests/` directory has corrupted test files that got merged/concatenated incorrectly.

## Solution
Clean, minimal test files are in `tests_clean/`:

### Files Created
1. **test_configuration.py** - ~25 tests for Configuration class
   - Parsing, electron counting, validation, conversion, real-world cases
   
2. **test_shell.py** - ~15 tests for Shell class
   - Creation, validation, relativistic quantum numbers, string representation
   
3. **test_unified_interface.py** - ~12 tests for AtomicCalculation interface
   - Basic calculations, coupling schemes, input generation, code options, relativistic

Total: **~52 clean, simple tests** covering core functionality

## How to Use

### Option 1: Replace corrupted tests (recommended)
```bash
cd /home/rfsilva/Programs/atomkit
rm -rf tests/
mv tests_clean/ tests/
```

### Option 2: Test the clean version first
```bash
micromamba run -n atomkit pytest tests_clean/ -v
```

Then if it works, do Option 1.

## What's Different
- ✅ **Simple and readable** - no verbose assertions
- ✅ **Uses correct API** - `l_quantum`, `occupation`, `total_electrons()`
- ✅ **Minimal but complete** - covers essential functionality only
- ✅ **Clean code** - no corruption, no merge conflicts

## Test Coverage

### test_configuration.py
- Parsing: from_string(), multiple shells, partial occupancy
- Electron counting: total_electrons() for various atoms
- Generation: ground states, excited states
- Validation: max occupancies for s, p, d orbitals
- Conversion: to AUTOSTRUCTURE format
- Real-world: Fe, Fe XVII, C III

### test_shell.py
- Creation: 1s, 2p, 3d shells
- Validation: invalid n, l >= n, negative values
- Relativistic: j quantum numbers for p and s shells
- String representation: readable output

### test_unified_interface.py
- Calculations: structure, radiative
- Coupling: LS, IC, ICR
- Input generation: file creation, content validation
- Code options: backend-specific parameters
- Relativistic: Breit interaction, QED corrections

## Philosophy
These tests follow the KISS principle:
- Test one thing at a time
- Use clear, descriptive names
- Avoid over-assertion
- Focus on public API only
- Keep it maintainable
