#!/usr/bin/env python3
"""
Manual test to compare das_1 generation with reference.
This avoids numpy import issues.
"""

# Read the reference file
print("=" * 80)
print("REFERENCE das_1.dat")
print("=" * 80)
with open("reference_examples/autostructure/das_1.dat") as f:
    ref = f.read()
print(ref)

# Check if we have a generated file already
import os

gen_path = "outputs/as_reference_tests/das_1_recreate.dat"
if os.path.exists(gen_path):
    print("\n" + "=" * 80)
    print("GENERATED das_1_recreate.dat (from previous run)")
    print("=" * 80)
    with open(gen_path) as f:
        gen = f.read()
    print(gen)

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Reference length: {len(ref)} chars")
    print(f"Generated length: {len(gen)} chars")
    print(f"Match: {ref.strip() == gen.strip()}")
else:
    print(f"\n❌ Generated file not found: {gen_path}")
    print("Need to fix numpy environment to generate it.")
