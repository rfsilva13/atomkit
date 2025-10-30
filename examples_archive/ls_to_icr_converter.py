"""
Example: Converting AUTOSTRUCTURE LS coupling to ICR coupling.

This example demonstrates how to use the LS to ICR converter to:
1. Extract optimized lambda parameters from LS calculations
2. Create ICR coupling input files
3. Optionally run AUTOSTRUCTURE calculations

Note: These examples assume you have AUTOSTRUCTURE executable available.
For demonstration purposes, we show how to use pre-computed lambdas.
"""

import numpy as np
import tempfile
from pathlib import Path
from atomkit.converters import convert_ls_to_icr, create_icr_input

print("=" * 70)
print("Example 1: Create ICR input with pre-computed lambdas")
print("=" * 70)

# Create a sample LS input file
ls_content = """&SYST
 NZ=26
 NE=24
 NELC=24
 CUP="LS"
 PKEY=2
&END

&SMINIM INCLUD=1 NVAR=5 &END
 1.0000  1.0000  1.0000  1.0000  1.0000

&CSFLS
 1 0 0 0 0
 0 1 0 0 0
 0 0 1 0 0
&END
"""

# Create temporary LS file
ls_file = Path(tempfile.mktemp(suffix="_LS.inp"))
ls_file.write_text(ls_content)

# Pre-computed lambda values (e.g., from a previous LS calculation)
lambdas = np.array(
    [0.9234567890, 0.9876543210, 1.0123456789, 0.9988776655, 1.0011223344]
)

try:
    # Convert LS to ICR using pre-computed lambdas
    icr_file, result_lambdas = convert_ls_to_icr(
        ls_file, run_ls_calculation=False, lambdas=lambdas  # Don't run AUTOSTRUCTURE
    )

    print(f"✓ Created ICR file: {icr_file}")
    print(f"✓ Used {len(result_lambdas)} lambda parameters")
    print(f"\nLambda values:")
    for i, lam in enumerate(result_lambdas, 1):
        print(f"  λ_{i} = {lam:.10f}")

    # Show ICR file content
    print(f"\nICR file content (first 20 lines):")
    print("-" * 70)
    with open(icr_file, "r") as f:
        lines = f.readlines()
        for line in lines[:20]:
            print(line.rstrip())
    print("-" * 70)

finally:
    # Cleanup
    if ls_file.exists():
        ls_file.unlink()
    if icr_file.exists():
        icr_file.unlink()


print("\n" + "=" * 70)
print("Example 2: Direct ICR file creation")
print("=" * 70)

# Create another LS file
ls_content2 = """&SYST
 NZ=79
 NE=78
 CUP="LS"
&END

&SMINIM INCLUD=1 NVAR=3 &END
 1.0  1.0  1.0
"""

ls_file2 = Path(tempfile.mktemp(suffix="_LS.inp"))
ls_file2.write_text(ls_content2)
icr_file2 = Path(tempfile.mktemp(suffix="_ICR.inp"))

# Lambdas from optimization
lambdas2 = np.array([0.95, 0.98, 1.02])

try:
    # Create ICR file directly
    create_icr_input(ls_file2, icr_file2, lambdas2)

    print(f"✓ Created: {icr_file2}")
    print(f"✓ Inserted lambda values: {lambdas2}")

    # Verify the changes
    with open(icr_file2, "r") as f:
        content = f.read()

    print("\nVerification:")
    print(f"  - CUP='ICR' present: {'✓' if 'CUP=\"ICR\"' in content else '✗'}")
    print(f"  - CUP='LS' removed: {'✓' if 'CUP=\"LS\"' not in content else '✗'}")
    print(f"  - INCLUD removed: {'✓' if 'INCLUD' not in content else '✗'}")
    print(f"  - NVAR removed: {'✓' if 'NVAR' not in content else '✗'}")
    print(f"  - Lambdas inserted: {'✓' if '0.9500000000' in content else '✗'}")

finally:
    if ls_file2.exists():
        ls_file2.unlink()
    if icr_file2.exists():
        icr_file2.unlink()


print("\n" + "=" * 70)
print("Example 3: Complete workflow (conceptual)")
print("=" * 70)

print(
    """
For a complete workflow WITH AUTOSTRUCTURE executable:

from atomkit.converters import ls_to_icr_full_workflow

# This will:
# 1. Run LS calculation
# 2. Extract lambdas from 'olg' file
# 3. Create ICR input file
# 4. Optionally run ICR calculation

result = ls_to_icr_full_workflow(
    'Fe_LS.inp',
    icr_output_file='Fe_ICR.inp',
    as_executable='./autostructure.x',
    run_icr_calculation=True
)

print(f"LS input: {result['ls_input']}")
print(f"ICR output: {result['icr_output']}")
print(f"Lambdas: {result['lambdas']}")
print(f"ICR calculation result: {result['icr_result']}")
"""
)


print("\n" + "=" * 70)
print("Example 4: Working with existing AUTOSTRUCTURE output")
print("=" * 70)

print(
    """
If you already have an 'olg' file from a previous LS calculation:

from atomkit.converters import convert_ls_to_icr
from atomkit.readers import read_as_lambdas

# Read lambdas from existing olg file
nl, lambdas = read_as_lambdas('previous_calculation.olg')

print(f"Found {len(lambdas)} lambda parameters")
for i, (n, l) in enumerate(nl):
    print(f"  n={n}, l={l}: λ={lambdas[i]:.6f}")

# Use these lambdas to create ICR input
icr_file, _ = convert_ls_to_icr(
    'new_LS_input.inp',
    'new_ICR_input.inp',
    run_ls_calculation=False,
    lambdas=lambdas
)

print(f"Created ICR file: {icr_file}")
"""
)


print("\n" + "=" * 70)
print("Example 5: Error handling")
print("=" * 70)

print(
    """
The converter includes comprehensive error handling:

try:
    # This will fail if AUTOSTRUCTURE executable is not found
    result = convert_ls_to_icr(
        'input.inp',
        as_executable='./nonexistent.x',
        run_ls_calculation=True
    )
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure AUTOSTRUCTURE executable is in the correct location")

try:
    # This will fail if no lambdas are available
    result = convert_ls_to_icr(
        'input.inp',
        run_ls_calculation=False,
        lambdas=None
    )
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Either run LS calculation or provide lambda parameters")
"""
)

print("\n" + "=" * 70)
print("Examples complete!")
print("=" * 70)
print("\nKey Functions:")
print("  - convert_ls_to_icr(): Main conversion function")
print("  - create_icr_input(): Create ICR file from LS file + lambdas")
print("  - run_autostructure_ls(): Run LS calculation")
print("  - run_autostructure_icr(): Run ICR calculation")
print("  - ls_to_icr_full_workflow(): Complete automation")
print("\nFor more information, see the module documentation:")
print("  >>> from atomkit.converters import ls_to_icr")
print("  >>> help(ls_to_icr)")
