# atomkit/examples/reading_transitions_example.py

import sys
from pathlib import Path

import pandas as pd

# --- Import the reader function ---
# Assuming atomkit is installed or src is in PYTHONPATH
try:
    # Import the new function for reading transitions
    from atomkit.readers import read_fac_transitions
except ImportError as e:
    print(f"\nError importing read_fac_transitions: {e}")
    print(
        "Please ensure atomkit is installed or the 'src' directory is in the Python path."
    )
    sys.exit(1)

# --- Define the input file ---
# Adjust the path as needed for your environment
# Tries to find the test_files directory relative to this script
try:
    project_root = Path(__file__).resolve().parent.parent
    # --- !! IMPORTANT !! ---
    # Change 'Pd' to the actual base name of your transition test file
    test_file_base = (
        project_root / "test_files" / "Pd"
    )  # <-- Using Pd example from logs
    # --- !! IMPORTANT !! ---
except NameError:
    # Fallback if __file__ is not defined
    project_root = Path(".").resolve()
    test_file_base = (
        project_root / "test_files" / "Pd"
    )  # <-- Using Pd example from logs

test_file_ext = ".tr.asc"  # Standard extension for FAC transitions
full_test_path = test_file_base.parent / f"{test_file_base.name}{test_file_ext}"

# --- Check if test file exists ---
if not full_test_path.exists():
    print(f"\nERROR: Transition test file not found at '{full_test_path}'")
    print(
        "Please ensure the test file exists and the 'test_file_base' variable is set correctly."
    )
    # sys.exit(1) # Exit if the file is crucial, commented out for demonstration flexibility

# --- Configure Pandas Display ---
pd.set_option("display.max_rows", 15)  # Show a few more rows
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)  # Adjust width as needed

# --- Example Use Cases for Transitions ---

print("\n--- Reading FAC Transitions (.tr.asc) ---")

print("\n--- 1. Basic Usage (Defaults) ---")
# Defaults: verbose=0 (Warnings/Errors only), energy_unit='ev', wavelength_unit='a', include_method=False
try:
    print("(Minimal logging expected)")
    df_default_tr = read_fac_transitions(
        str(test_file_base), file_extension=test_file_ext
    )
    if not df_default_tr.empty:
        print(f"DataFrame shape: {df_default_tr.shape}")
        print("Columns (Concise):", df_default_tr.columns.tolist())
        print("Index Names:", df_default_tr.index.names)
        print("Sample Data (Energy in eV, Wavelength in A):")
        print(df_default_tr.head(10))
    else:
        print(
            "Reading resulted in an empty DataFrame (check logs for warnings/errors)."
        )
except Exception as e:
    print(f"An error occurred during basic reading: {e}")

print("\n--- 2. Verbose Level 1 (Info) ---")
# Turn on informational logging messages
try:
    print("(Expecting INFO level logging output below...)")
    df_verbose_tr = read_fac_transitions(
        str(test_file_base),
        file_extension=test_file_ext,
        verbose=1,  # Set verbose to 1 for INFO
    )
    if not df_verbose_tr.empty:
        print(f"DataFrame shape: {df_verbose_tr.shape}")
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during verbose=1 reading: {e}")

print("\n--- 3. Verbose Level 2 (Debug) ---")
# Turn on debug logging messages
try:
    print("(Expecting DEBUG level logging output below...)")
    df_debug_tr = read_fac_transitions(
        str(test_file_base),
        file_extension=test_file_ext,
        verbose=2,  # Set verbose to 2 for DEBUG
    )
    if not df_debug_tr.empty:
        print(f"DataFrame shape: {df_debug_tr.shape}")
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during verbose=2 reading: {e}")


print("\n--- 4. Request Energy in cm-1 and Wavelength in nm ---")
# Use energy_unit='cm-1' and wavelength_unit='nm'
try:
    df_units_tr = read_fac_transitions(
        str(test_file_base),
        file_extension=test_file_ext,
        energy_unit="cm-1",
        wavelength_unit="nm",
    )
    if not df_units_tr.empty:
        print(f"DataFrame shape: {df_units_tr.shape}")
        print("Columns (Concise):", df_units_tr.columns.tolist())
        print("Sample Data (Energy in cm-1, Wavelength in nm):")
        print(df_units_tr[["delta_energy", "wavelength", "gf"]].head(5))
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during unit conversion reading: {e}")


print("\n--- 5. Include Method Column ---")
# Use include_method=True
try:
    df_method_tr = read_fac_transitions(
        str(test_file_base),
        file_extension=test_file_ext,
        include_method=True,  # Request method column
    )
    if not df_method_tr.empty:
        print(f"DataFrame shape: {df_method_tr.shape}")
        print("Columns (Includes Method):", df_method_tr.columns.tolist())
        print("Sample Data:")
        print(
            df_method_tr[["wavelength", "gf", "method"]].head(3)
        )  # Show relevant columns
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during include_method reading: {e}")


print("\n--- 6. Select Specific Columns (Using Concise Names) ---")
# Keep only a subset of columns, referring to them by their concise names
# Index columns are typically kept automatically if they exist before selection.
cols_tr = [
    "wavelength",
    "gf",
    "tr_rate",
    "multipole_type",
    "line_strength_S",
]  # Use concise names
try:
    df_selected_tr = read_fac_transitions(
        str(test_file_base), file_extension=test_file_ext, columns_to_keep=cols_tr
    )
    if not df_selected_tr.empty:
        print(f"DataFrame shape: {df_selected_tr.shape}")
        print("Columns (Selected):", df_selected_tr.columns.tolist())
        print("Index Names:", df_selected_tr.index.names)
        print("Sample Selected Data:")
        print(df_selected_tr.head(5))
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during column selection reading: {e}")


print("\n--- 7. Rename Columns (Using Concise Names) ---")
# Rename some columns using their concise names as keys
rename_map_tr = {
    "wavelength": "Wavelength",  # Rename concise name
    "tr_rate": "A_Value",
    "line_strength_S": "S_au",
}
try:
    df_renamed_tr = read_fac_transitions(
        str(test_file_base), file_extension=test_file_ext, rename_columns=rename_map_tr
    )
    if not df_renamed_tr.empty:
        print(f"DataFrame shape: {df_renamed_tr.shape}")
        print("Columns (Renamed):", df_renamed_tr.columns.tolist())
        print(
            "Index Names (note: index names also renamed if specified):",
            df_renamed_tr.index.names,
        )
        print("Sample Renamed Data:")
        print(df_renamed_tr[["Wavelength", "A_Value", "S_au"]].head(3))
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during column renaming reading: {e}")


print("\n--- 8. Combined Options (cm-1, nm, Rename, Include Method) ---")
# Combine unit conversion, renaming, and including method
cols_combo_tr = [
    "wavelength",
    "gf",
    "tr_rate",
    "line_strength_S",
    "method",
]  # Select concise names
rename_combo_tr = {
    "wavelength": "Wavelength_nm",
    "tr_rate": "A_Value",
}  # Rename concise names
try:
    df_combined_tr = read_fac_transitions(
        str(test_file_base),
        file_extension=test_file_ext,
        energy_unit="cm-1",  # Request energy in cm-1 (will affect delta_energy)
        wavelength_unit="nm",  # Request wavelength in nm
        columns_to_keep=cols_combo_tr,
        rename_columns=rename_combo_tr,
        include_method=True,  # Also include method
    )
    if not df_combined_tr.empty:
        print(f"DataFrame shape: {df_combined_tr.shape}")
        print("Columns (Combined):", df_combined_tr.columns.tolist())
        print("Index Names:", df_combined_tr.index.names)
        print("Sample Combined Data:")
        print(df_combined_tr.head(5))
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during combined options reading: {e}")


print("\n--- Example script for transitions finished. ---")
