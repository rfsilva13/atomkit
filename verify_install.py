"""
A simple script to verify that the AtomKit package is installed correctly.
"""

try:
    import atomkit
    print("✅ AtomKit was imported successfully!")
    print(f"   Version: {atomkit.__version__}")
    print(f"   Path: {atomkit.__file__}")
except ImportError as e:
    print("❌ Failed to import AtomKit.")
    print(f"   Error: {e}")
    print("   Please check your installation.")

