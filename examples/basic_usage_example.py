# atomkit/examples/basic_usage_example.py

"""
Example script demonstrating common usage patterns of the Shell and
Configuration classes from the atomkit library.

This script showcases how to create Shell objects, parse shell strings,
create Configuration objects from various inputs (strings, element identifiers,
compact notation), and utilize methods for analysis and manipulation like
splitting core/valence, finding holes, comparing configurations, calculating
X-ray labels, and generating excitations *from the valence part*.

Assumes atomkit and its dependencies (including mendeleev) are installed.
Run from the project root directory using:
poetry run python examples/basic_usage_example.py
"""

# mendeleev is needed for element-based operations
import mendeleev

from atomkit.configuration import Configuration

# Import necessary classes and functions
# Assuming atomkit is installed in the environment
from atomkit.shell import Shell

print("--- Shell Class Examples ---")
print("-" * 30)

# ========================================
# 1. Creating Shell objects directly
# ========================================
# Instantiate Shell objects by providing:
#   n: Principal quantum number (int >= 1)
#   l_quantum: Orbital angular momentum (int >= 0, must be < n)
#   occupation: Number of electrons (int >= 0, <= max_occupation)
#   j_quantum: Optional total angular momentum (float or int, half-integer >= 0.5)
print("1. Creating Shells directly:")
shell_1s = Shell(n=1, l_quantum=0, occupation=2)
shell_2p_minus = Shell(n=2, l_quantum=1, occupation=1, j_quantum=0.5)  # j=0.5 -> p-
shell_4f = Shell(n=4, l_quantum=3, occupation=5)
shell_valid_high_l = Shell(
    n=8, l_quantum=7, occupation=3
)  # Example of a valid high-l shell (l=7 is k)

# Access properties:
print(f"   - Directly created 1s: {shell_1s} (Max Occ: {shell_1s.max_occupation()})")
print(
    f"   - Directly created 2p-: {shell_2p_minus} (Max Occ: {shell_2p_minus.max_occupation()})"
)
print(f"   - Directly created 4f: {shell_4f} (Max Occ: {shell_4f.max_occupation()})")
print(
    f"   - Directly created valid high-l: {shell_valid_high_l} (Symbol: {shell_valid_high_l.l_symbol})"
)
print("-" * 30)

# ========================================
# 2. Creating Shell objects from strings
# ========================================
# Use the `Shell.from_string()` class method for convenience.
# It parses standard notation, relativistic (+/-), and high-l ([l=num]).
# It enforces l < n and validates occupation.
print("2. Creating Shells from strings:")
shell_str_3d = Shell.from_string("3d10")
shell_str_2p_plus = Shell.from_string("2p+3")  # j=1.5 implied
shell_str_5g = Shell.from_string("5g")  # Assumes occupation 1
shell_str_high_l = Shell.from_string("22[l=21]12")  # Valid high l (l<n)

# Check properties:
print(f"   - From string '3d10': {shell_str_3d} (Full? {shell_str_3d.is_full()})")
print(
    f"   - From string '2p+3': {shell_str_2p_plus} (Holes: {shell_str_2p_plus.holes()})"
)
print(f"   - From string '5g': {shell_str_5g}")
print(f"   - From string '22[l=21]12': {shell_str_high_l}")
print("-" * 30)

# ========================================
# 3. Shell manipulation
# ========================================
# `take_electron()` and `add_electron()` return *new* Shell objects.
# The original object remains unchanged (immutability).
print("3. Shell manipulation:")
shell_2p5 = Shell.from_string("2p5")
shell_2p4 = shell_2p5.take_electron()
shell_2p6 = shell_2p5.add_electron()
print(
    f"   - Original: {shell_2p5}, After take_electron(): {shell_2p4}, After add_electron(): {shell_2p6}"
)
print("-" * 30)


print("\n--- Configuration Class Examples ---")
print("-" * 30)

# ========================================
# 4. Creating Configurations from strings
# ========================================
# `Configuration.from_string()` parses strings with shells separated by '.' or space.
# It sorts shells and combines occupations for identical structures.
print("4. Creating Configurations from strings:")
config_neutral_fe_str = "1s2.2s2.2p6.3s2.3p6.3d6.4s2"
config_fe = Configuration.from_string(config_neutral_fe_str)
print(f"   - Parsed Fe I: {config_fe}")
print(f"     - Total electrons: {config_fe.total_electrons()}")
print(f"     - Number of distinct shells: {len(config_fe)}")

config_li = Configuration.from_string("1s2 2s1")  # Space separator
print(f"   - Parsed Li I: {config_li}")

config_rel = Configuration.from_string("2p-1.2p+3")  # Relativistic
print(f"   - Parsed Relativistic: {config_rel}")
print(f"     - Total electrons: {config_rel.total_electrons()}")  # 1+3=4

config_combine = Configuration.from_string("1s1.2s1.1s1")  # Combine 1s
print(f"   - Parsed Combined: {config_combine}")  # Expect 1s2.2s1

config_empty = Configuration.from_string("")
print(f"   - Parsed Empty: '{config_empty}', Length: {len(config_empty)}")
print("-" * 30)

# ==============================================================
# 5. Creating Configurations from element identifiers
# ==============================================================
# `Configuration.from_element()` uses `mendeleev` to get ground state configs.
# Provide Z (int), symbol (str), or name (str). `ion_charge` defaults to 0.
print("5. Creating Configurations from element identifiers:")
conf_ne = Configuration.from_element("Ne")  # Neutral Neon by symbol
conf_na_plus = Configuration.from_element(11, ion_charge=1)  # Sodium ion (Na+) by Z
conf_fe_3 = Configuration.from_element(
    "Iron", ion_charge=3
)  # Iron(IV) ion (Fe+3) by name
conf_ar = Configuration.from_element(18)  # Neutral Argon by Z

print(f"   - Ne I from element: {conf_ne} (e={conf_ne.total_electrons()})")
print(f"   - Na II from element: {conf_na_plus} (e={conf_na_plus.total_electrons()})")
print(f"   - Fe IV from element: {conf_fe_3} (e={conf_fe_3.total_electrons()})")
print(f"   - Ar I from element: {conf_ar} (e={conf_ar.total_electrons()})")

# --- Using the get_ionstage METHOD ---
# The `get_ionstage` method calculates the charge relative to a neutral element.
print("\n   --- Calculating Ion Stage ---")
print(
    f"   - Ion stage for '{conf_na_plus}' (relative to Na): {conf_na_plus.get_ionstage('Na')}"
)  # Expect 1
print(
    f"   - Ion stage for '{conf_fe_3}' (relative to Fe): {conf_fe_3.get_ionstage(26)}"
)  # Expect 3
print(
    f"   - Ion stage for '{conf_ar}' (relative to Argon): {conf_ar.get_ionstage('Argon')}"
)  # Expect 0
# --- End ---
print("-" * 30)

# ======================================================================
# 6. Creating Configurations from compact strings (N*E format)
# ======================================================================
# `Configuration.from_compact_string()` parses "N1*E1.N2*E2..."
print("6. Creating Configurations from compact strings (N*E format):")

# --- 6a. Default Behavior (Sequential Filling) ---
# `generate_permutations=False` (default) fills subshells sequentially (s, p, d...).
print("\n   --- 6a. Default (Sequential Fill) ---")
conf_compact_ne = Configuration.from_compact_string("1*2.2*8")  # Neon
print(f"   - From '1*2.2*8': {conf_compact_ne}")  # Expected: 1s2.2s2.2p6

conf_compact_p = Configuration.from_compact_string("1*2.2*8.3*5")  # Phosphorus
print(f"   - From '1*2.2*8.3*5': {conf_compact_p}")  # Expected: 1s2.2s2.2p6.3s2.3p3

# --- 6b. Generating Permutations ---
# `generate_permutations=True` generates all valid electron distributions
# for each N*E part and returns a list of unique combined Configurations.
print("\n   --- 6b. Generating Permutations ---")
compact_n2_e6 = "2*6"  # 6 electrons in n=2 shell
print(f"   - Permutations for '{compact_n2_e6}':")
perms_n2_e6 = Configuration.from_compact_string(
    compact_n2_e6, generate_permutations=True
)
for i, p_conf in enumerate(perms_n2_e6):
    print(f"     - Permutation {i+1}: {p_conf}")  # Expected: 2s2.2p4, 2s1.2p5, 2p6

compact_multi = "1*2.2*1"  # 2 electrons in n=1, 1 electron in n=2
print(f"\n   - Permutations for '{compact_multi}':")
perms_multi = Configuration.from_compact_string(
    compact_multi, generate_permutations=True
)
for i, p_conf in enumerate(perms_multi):
    print(f"     - Permutation {i+1}: {p_conf}")  # Expected: 1s2.2s1, 1s2.2p1
print("-" * 30)


# ======================================================================
# 7. Splitting Core and Valence & Operating on Valence
# ======================================================================
# Define a core configuration and separate it from the valence shells.
# This is useful for calculations focusing on outer electrons.
print("7. Splitting Core/Valence and Operating on Valence:")
config_al = Configuration.from_string("1s2.2s2.2p6.3s2.3p1")  # Al I ground state
print(f"   - Original Al I: {config_al}")

# Define the core shells (e.g., Neon core)
core_definition = ["1s", "2s", "2p"]
# Use the split_core_valence method. It returns two *new* Configuration objects.
core_config, valence_config = config_al.split_core_valence(core_definition)

print(f"   - Core defined by {core_definition}:")
print(f"     - Core Config:    {core_config} (e={core_config.total_electrons()})")
print(f"     - Valence Config: {valence_config} (e={valence_config.total_electrons()})")

# Now perform operations *only* on the valence configuration object
print(f"\n   - Operations on Valence Config ({valence_config}):")

# Get holes in valence
valence_holes = valence_config.get_holes()
print(f"     - Holes in valence: {valence_holes}")  # Expect {'3p': 5}

# Generate single holes in valence
valence_1_hole = valence_config.generate_hole_configurations(num_holes=1)
print("     - Valence single holes:")
for hc in valence_1_hole:
    print(f"       - {hc}")  # Expect 3s1.3p1, 3s2

# --- Generate Excitations from VALENCE ---
# Define target shells for excitation
valence_targets = ["3d", "4s", "4p"]
print(f"     - Single excitations from valence to {valence_targets}:")
# Generate excitations *from* the valence_config *to* the targets
valence_excitations_S = valence_config.generate_excitations(
    valence_targets, excitation_level=1
)
for ec in valence_excitations_S:
    # To see the full excited configuration, you could combine with the core:
    # full_excited_config = Configuration(core_config.shells + ec.shells)
    # print(f"       - Full Excited: {full_excited_config}")
    print(f"       - Valence Part: {ec}")  # e.g., 3s1.3p1.3d1, 3s2.3d1, etc.

print(f"\n     - Double excitations from valence to {valence_targets}:")
valence_excitations_D = valence_config.generate_excitations(
    valence_targets, excitation_level=2
)
print(f"       - Found {len(valence_excitations_D)} double excitations. First few:")
for i, ec in enumerate(valence_excitations_D):
    if i >= 5:
        break  # Print only the first few
    print(f"         - {ec}")

print(f"\n     - Triple excitations from valence to {valence_targets}:")
valence_excitations_T = valence_config.generate_excitations(
    valence_targets, excitation_level=3
)
print(f"       - Found {len(valence_excitations_T)} triple excitations. First few:")
for i, ec in enumerate(valence_excitations_T):
    if i >= 5:
        break  # Print only the first few
    print(f"         - {ec}")
# --- End Excitation Examples ---

# Example comparing the original valence to an excited valence state
excited_valence_example = Configuration.from_string("3s1.3p1.4s1")
valence_diff = valence_config.compare(excited_valence_example)
print(
    f"\n     - Difference between {valence_config} and {excited_valence_example}: {valence_diff}"
)

print("-" * 30)

# ========================================
# 8. Calculating X-ray Labels
# ========================================
# Use `calculate_xray_label` to find the label based on holes relative to a reference.
# It now returns a list of individual hole labels, sorted.
print("8. Calculating X-ray Labels:")
# Define a reference configuration (e.g., neutral Neon)
ref_ne = Configuration.from_string("1s2.2s2.2p6")
print(f"   - Reference Config: {ref_ne}")

# Create configurations with holes
conf_k_hole = Configuration.from_string("1s1.2s2.2p6")
conf_l1_hole = Configuration.from_string("1s2.2s1.2p6")
conf_l23_hole = Configuration.from_string("1s2.2s2.2p5")
conf_k_l1_hole = Configuration.from_string("1s1.2s1.2p6")
conf_double_l23_hole = Configuration.from_string("1s2.2s2.2p4")  # Two holes in 2p

# Calculate labels relative to the reference
print(
    f"   - Label for {conf_k_hole}: {conf_k_hole.calculate_xray_label(ref_ne)}"
)  # Expect ['K']
print(
    f"   - Label for {conf_l1_hole}: {conf_l1_hole.calculate_xray_label(ref_ne)}"
)  # Expect ['L1']
print(
    f"   - Label for {conf_l23_hole}: {conf_l23_hole.calculate_xray_label(ref_ne)}"
)  # Expect ['L23']
print(
    f"   - Label for {conf_k_l1_hole}: {conf_k_l1_hole.calculate_xray_label(ref_ne)}"
)  # Expect ['K', 'L1']
print(
    f"   - Label for {conf_double_l23_hole}: {conf_double_l23_hole.calculate_xray_label(ref_ne)}"
)  # Expect ['L23', 'L23']

# Example with relativistic shells
ref_rel = Configuration.from_string("1s2.2s2.2p-2.2p+4")  # Neutral Mg
conf_l2_hole = Configuration.from_string("1s2.2s2.2p-1.2p+4")
conf_l3_hole = Configuration.from_string("1s2.2s2.2p-2.2p+3")
conf_l2_l3_hole = Configuration.from_string("1s2.2s2.2p-1.2p+3")
print(f"\n   - Reference Relativistic: {ref_rel}")
print(
    f"   - Label for {conf_l2_hole}: {conf_l2_hole.calculate_xray_label(ref_rel)}"
)  # Expect ['L2']
print(
    f"   - Label for {conf_l3_hole}: {conf_l3_hole.calculate_xray_label(ref_rel)}"
)  # Expect ['L3']
print(
    f"   - Label for {conf_l2_l3_hole}: {conf_l2_l3_hole.calculate_xray_label(ref_rel)}"
)  # Expect ['L2', 'L3']

# Example where config matches reference
print(
    f"   - Label for {ref_ne}: {ref_ne.calculate_xray_label(ref_ne)}"
)  # Expect ['Ground']
print("-" * 30)


print("\nExample script finished.")
