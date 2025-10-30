# AUTOS Reference Tests

This directory contains the complete recreation of all 21 AUTOSTRUCTURE reference tests from the official AUTOS website: https://amdpp.phys.strath.ac.uk/autos/default/data/

## Test File

**`test_autos_reference.py`** - Complete test suite recreating all 21 AUTOS reference calculations using the agnostic AtomKit `AtomicCalculation` interface.

## Test Coverage

All 21 reference tests are fully implemented and passing:

### Basic Structure Tests
1. **das_1** - Be-like C structure (energies only)
2. **das_2** - Be-like C with radiative transitions (E1)
3. **das_3** - Lambda scaling optimization
4. **das_4** - KLL Auger process

### Continuum Process Tests
5. **das_5** - Photoionization (PI)
6. **das_6** - Dielectronic recombination (DR)
7. **das_7** - Radiative recombination (RR)

### Collision Tests
8. **das_8** - Collision with symmetry restrictions
9. **das_9** - Collision with relaxed basis (BASIS='RLX')
10. **das_10** - Distorted wave collision (DW)
11. **das_21** - Collision with Bethe-Peterkop approximation (BP)

### Configuration Generation Tests
12. **das_11** - ICFG automatic configuration generation (single excitations)
13. **das_12** - ICFG double excitations

### Advanced Options Tests
14. **das_13** - KCUT multipole restrictions
15. **das_14** - All multipoles (RAD='ALL')
16. **das_16** - ICR coupling with relativistic + QED corrections
17. **das_17** - Born approximation (BORN='INF')
18. **das_18** - Mg with core specification (KCOR)
19. **das_19** - LS coupling without fine structure
20. **das_20** - Inner shell DR calculation
21. **das_15** - R-matrix support (KCUTCC)

## Test Results

```
42 tests total (21 basic + 21 comparison)
42 passed ✓
0 failed
```

## Key Features Demonstrated

### Agnostic Interface Usage
Each test uses the `AtomicCalculation` class which provides a code-agnostic way to specify atomic calculations:

```python
calc = AtomicCalculation(
    element="C",
    charge=2,
    calculation_type="structure",  # or "radiative", "photoionization", "DR", "RR", "collision"
    coupling="IC",                 # LS, IC, ICR, etc.
    relativistic="Breit",          # none, Breit, QED, etc.
    qed_corrections=True,
    optimization="lambda",
    configurations=[...],
    code_options={...},            # Backend-specific overrides
    code="autostructure"
)
```

### Physical Concepts Mapped

- **Coupling schemes**: LS, IC, ICR, CA, LSM, MVD, CAR, LSR
- **Relativistic corrections**: Breit interaction, QED, retardation
- **Optimization**: Lambda scaling, energy minimization
- **Calculation types**: Structure, radiative, photoionization, DR, RR, collision
- **Advanced options**: ICFG, symmetry restrictions, basis control, multipole restrictions

### Code-Specific Options

For backend-specific parameters not covered by the agnostic interface, use `code_options`:

```python
code_options={
    "KCOR1": 1,    # One-body relativity
    "KCOR2": 1,    # Two-body relativity
    "KUTSO": 0,    # Spin-orbit
    "BORN": "INF", # Born approximation
    "BASIS": "RLX" # Relaxed basis
}
```

## Reference Data

Original AUTOS test files are stored in: `as_tests/amdpp.phys.strath.ac.uk/autos/default/data/test_X/`

Each test directory contains:
- `das_X` - Input file
- `dat_X` - Output data
- `lsg_X` - LS term output
- Additional output files depending on calculation type

## Running Tests

```bash
# Run all AUTOS reference tests
micromamba run -n atomkit pytest tests/test_autos_reference.py -v

# Run specific test
micromamba run -n atomkit pytest tests/test_autos_reference.py::TestAUTOSReference::test_das_1_be_like_c_structure -v

# Run with comparison to reference files
micromamba run -n atomkit pytest tests/test_autos_reference.py::TestAUTOSReferenceComparison -v
```

## Validation

Each test validates that:
1. The generated input file is created successfully
2. Key parameters appear in the output
3. The structure matches the reference (when applicable)

The `TestAUTOSReferenceComparison` class performs detailed comparison with the original reference files, checking:
- Correct SALGEB parameters
- Correct SMINIM parameters
- Configuration structure
- Special options and flags

## Notes

- Tests use temporary output directories to avoid file conflicts
- All tests are independent and can run in any order
- The agnostic interface automatically translates to backend-specific parameters
- Tests demonstrate the full range of AUTOSTRUCTURE capabilities through AtomKit

## Future Work

- Add validation of output files (not just input generation)
- Add numerical comparison of energy levels
- Add comparison with other codes (FAC) for same physical problem
- Extend to test data analysis and visualization features
