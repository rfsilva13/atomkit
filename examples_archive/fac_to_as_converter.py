"""
Example: Converting FAC configurations to AUTOSTRUCTURE format.

This example demonstrates how to use the FAC to AUTOSTRUCTURE converter
to extract configuration information from FAC input files and format it
for use in AUTOSTRUCTURE calculations.
"""

from atomkit.converters import convert_fac_to_as, print_as_format

# Example 1: Simple conversion with console output
print("=" * 60)
print("Example 1: Convert FAC file and print to console")
print("=" * 60)

# Create a sample FAC file content (normally you'd read an actual file)
sample_fac_content = """
# Sample FAC input for Cu I (Z=29)
Closed('1s 2s 2p 3s 3p')

Config('MR', '3d10 4s1')
Config('MR', '3d9 4s2')
Config('MR', '3d10 4p1')
Config('MR', '3d9 4s1 4p1')
"""

# Save to a temporary file
import tempfile
from pathlib import Path

with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sf") as f:
    f.write(sample_fac_content)
    fac_file = Path(f.name)

try:
    # Convert the FAC file
    result = convert_fac_to_as(fac_file, config_label="MR")

    # Print the AUTOSTRUCTURE format
    print_as_format(result)

    print(f"\nSummary:")
    print(f"  - Configurations: {result['mxconf']}")
    print(f"  - Valence orbitals: {result['mxvorb']}")
    print(f"  - Closed shells: {result['kcor2']}")

finally:
    # Clean up temporary file
    fac_file.unlink()


# Example 2: Convert and save to file
print("\n" + "=" * 60)
print("Example 2: Convert FAC file and save to output file")
print("=" * 60)

# Create another sample FAC file for Fe I
iron_fac_content = """
# Sample FAC input for Fe I (Z=26)
Closed('1s 2s 2p 3s 3p')

Config('GS', '3d6 4s2')
Config('EX', '3d7 4s1')
Config('EX', '3d6 4s1 4p1')
Config('EX', '3d5 4s2 4p1')
"""

with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sf") as f:
    f.write(iron_fac_content)
    fac_file = Path(f.name)

output_file = Path("fe_autostructure_input.txt")

try:
    # Convert with output file
    result = convert_fac_to_as(fac_file, config_label="EX", output_file=output_file)

    print(f"Converted FAC configurations to AUTOSTRUCTURE format")
    print(f"Output saved to: {output_file}")
    print(f"\nConfiguration details:")
    print(f"  - Label: EX")
    print(f"  - Configurations: {result['mxconf']}")
    print(f"  - Orbitals: {result['orbitals']}")

    # Show file content
    print(f"\nOutput file content:")
    with open(output_file, "r") as f:
        print(f.read())

finally:
    # Clean up
    fac_file.unlink()
    if output_file.exists():
        output_file.unlink()


# Example 3: Working with the result data structure
print("\n" + "=" * 60)
print("Example 3: Accessing conversion result data")
print("=" * 60)

sample_fac = """
Closed('1s 2s 2p')
Config('MR', '3s2 3p5')
Config('MR', '3s1 3p6')
"""

with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sf") as f:
    f.write(sample_fac)
    fac_file = Path(f.name)

try:
    result = convert_fac_to_as(fac_file, "MR")

    print("Result data structure:")
    print(f"  ICFG (config flag): {result['icfg']}")
    print(f"  KCOR2 (closed shells): {result['kcor2']}")
    print(f"  MXCONF (num configs): {result['mxconf']}")
    print(f"  MXVORB (num orbitals): {result['mxvorb']}")

    print(f"\nOrbitals (n, l):")
    for i, (n, l) in enumerate(result["orbitals"]):
        l_symbols = ["s", "p", "d", "f", "g", "h"]
        l_str = l_symbols[l] if l < len(l_symbols) else str(l)
        print(f"    {i}: {n}{l_str} (n={n}, l={l})")

    print(f"\nOccupation matrix:")
    for i, row in enumerate(result["occupation_matrix"]):
        config_str = result["configurations"][i]
        print(f"    Config {i} ({config_str}): {row}")

finally:
    fac_file.unlink()


print("\n" + "=" * 60)
print("Examples complete!")
print("=" * 60)
