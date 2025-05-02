# atomkit/examples/reading_example.py

import sys
from pathlib import Path

import pandas as pd

# --- Import the reader function ---
# Assuming atomkit is installed or src is in PYTHONPATH
try:
    from atomkit.readers import read_fac  # Using the level reader
except ImportError as e:
    print(f"\nError importing read_fac: {e}")
    print(
        "Please ensure atomkit is installed or the 'src' directory is in the Python path."
    )
    sys.exit(1)


# --- Define the input file ---
# Adjust the path as needed for your environment
# Tries to find the test_files directory relative to this script
try:
    project_root = Path(__file__).resolve().parent.parent
    # Use the Pr II level file example
    test_file_base = project_root / "test_files" / "59PrII"
except NameError:
    # Fallback if __file__ is not defined
    project_root = Path(".").resolve()
    test_file_base = project_root / "test_files" / "59PrII"

test_file_ext = ".lev.asc"  # Level file extension
full_test_path = test_file_base.parent / f"{test_file_base.name}{test_file_ext}"

# --- Check if test file exists ---
if not full_test_path.exists():
    print(f"\nERROR: Level test file not found at '{full_test_path}'")
    print(
        "Please ensure the test file exists and the 'test_file_base' variable is set correctly."
    )
    # sys.exit(1) # Exit if the file is crucial

# --- Configure Pandas Display ---
pd.set_option("display.max_rows", 15)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

# --- Example Use Cases for Levels ---

print("\n--- Reading FAC Levels (.lev.asc) ---")

print("\n--- 1. Basic Usage (Defaults) ---")
# Defaults: verbose=0 (Warnings/Errors only), energy_unit='ev', include_method=False
try:
    print("(Minimal logging expected)")
    df_default = read_fac(str(test_file_base), file_extension=test_file_ext)
    if not df_default.empty:
        print(f"DataFrame shape: {df_default.shape}")
        print("Columns (Concise):", df_default.columns.tolist())
        print("Index Names:", df_default.index.names)
        print("Sample Data (Energy in eV):")
        print(df_default.head(5))
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
    df_verbose = read_fac(
        str(test_file_base),
        file_extension=test_file_ext,
        verbose=1,  # Set verbose to 1 for INFO
    )
    if not df_verbose.empty:
        print(f"DataFrame shape: {df_verbose.shape}")
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during verbose=1 reading: {e}")

print("\n--- 3. Verbose Level 2 (Debug) ---")
# Turn on debug logging messages
try:
    print("(Expecting DEBUG level logging output below...)")
    df_debug = read_fac(
        str(test_file_base),
        file_extension=test_file_ext,
        verbose=2,  # Set verbose to 2 for DEBUG
    )
    if not df_debug.empty:
        print(f"DataFrame shape: {df_debug.shape}")
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during verbose=2 reading: {e}")


print("\n--- 4. Request Energy in cm-1 ---")
# Use energy_unit='cm-1'
try:
    df_cm1 = read_fac(
        str(test_file_base),
        file_extension=test_file_ext,
        energy_unit="cm-1",  # Request energy in cm-1
    )
    if not df_cm1.empty:
        print(f"DataFrame shape: {df_cm1.shape}")
        print(
            "Columns:", df_cm1.columns.tolist()
        )  # Note 'energy' column name is still concise
        print("Sample Data (Energy converted to cm-1):")
        if "energy" in df_cm1.columns:
            print(df_cm1["energy"].head(3))
        else:
            print(" -> 'energy' column not found.")
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during cm-1 reading: {e}")


print("\n--- 5. Include Method Column ---")
# Use include_method=True
try:
    df_method = read_fac(
        str(test_file_base),
        file_extension=test_file_ext,
        include_method=True,  # Request method column
    )
    if not df_method.empty:
        print(f"DataFrame shape: {df_method.shape}")
        print("Columns (Includes Method):", df_method.columns.tolist())
        print("Sample Data:")
        print(
            df_method[["energy", "configuration", "method"]].head(3)
        )  # Show relevant columns
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during include_method reading: {e}")


print("\n--- 6. Select Specific Columns (Using Concise Names) ---")
# Keep only a subset of columns, referring to them by their concise names
# Index columns are typically kept automatically if they exist before selection.
cols = ["energy", "configuration", "label", "rel_config"]  # Use concise names
try:
    df_selected = read_fac(
        str(test_file_base), file_extension=test_file_ext, columns_to_keep=cols
    )
    if not df_selected.empty:
        print(f"DataFrame shape: {df_selected.shape}")
        print("Columns (Selected):", df_selected.columns.tolist())
        print("Index Names:", df_selected.index.names)
        print("Sample Selected Data:")
        print(df_selected.head(3))
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during column selection reading: {e}")


print("\n--- 7. Rename Columns (Using Concise Names) ---")
# Rename some columns using their concise names as keys
rename_map = {"energy": "E_eV", "p": "Parity", "label": "Level_Label"}
try:
    df_renamed = read_fac(
        str(test_file_base), file_extension=test_file_ext, rename_columns=rename_map
    )
    if not df_renamed.empty:
        print(f"DataFrame shape: {df_renamed.shape}")
        print("Columns (Renamed):", df_renamed.columns.tolist())
        print(
            "Index Names (note: index names also renamed if specified):",
            df_renamed.index.names,
        )
        print("Sample Renamed Data:")
        print(df_renamed[["E_eV", "Parity", "Level_Label"]].head(3))
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during column renaming reading: {e}")


print("\n--- 8. Combined Options (cm-1, Rename, Include Method) ---")
# Combine unit conversion, renaming, and including method
cols_combo = ["energy", "2j", "configuration", "method"]  # Select concise names
rename_combo = {
    "energy": "Energy_cm1",
    "configuration": "Config",
}  # Rename concise names
try:
    df_combined = read_fac(
        str(test_file_base),
        file_extension=test_file_ext,
        energy_unit="cm-1",  # Request cm-1
        columns_to_keep=cols_combo,
        rename_columns=rename_combo,
        include_method=True,  # Also include method
    )
    if not df_combined.empty:
        print(f"DataFrame shape: {df_combined.shape}")
        print("Columns (Combined):", df_combined.columns.tolist())
        print("Index Names:", df_combined.index.names)
        print("Sample Combined Data:")
        print(df_combined.head(5))
    else:
        print("Reading resulted in an empty DataFrame.")
except Exception as e:
    print(f"An error occurred during combined options reading: {e}")


print("\n--- Example script finished. ---")
