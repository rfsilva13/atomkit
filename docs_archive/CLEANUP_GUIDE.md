# Repository Cleanup Guide

## Overview
This guide shows what to move/delete to have a clean, minimal repository.

## Files to Archive or Remove

### 1. Move to docs_archive/ (for reference)
```bash
mv WORK_SUMMARY.md docs_archive/
mv QUICK_REFERENCE.md docs_archive/
mv CLEANUP_PLAN.md docs_archive/
mv CLEANUP_INSTRUCTIONS.txt docs_archive/
mv tests_clean_version.py docs_archive/
```

### 2. Replace Tests Directory
```bash
# Backup current corrupted tests (already in tests_archive)
rm -rf tests/

# Use clean tests
mv tests_clean/ tests/
```

### 3. Optional: Clean Up Examples (if too many)
The `examples/` directory might have too many files. Keep only:
- basic_usage.py
- autostructure_workflow.py
- unified_comparison.py

Move the rest to `examples_archive/` if needed.

### 4. Optional: Clean Up as_tests (AUTOS reference data)
If you don't need the AUTOS reference tests anymore:
```bash
# This is the downloaded reference data (large)
mv as_tests/ docs_archive/autos_reference_data/
```

## Final Clean Structure

After cleanup, your root should have:
```
atomkit/
├── src/atomkit/           # Core source code
├── tests/                 # Clean minimal tests (~52 tests)
├── docs/                  # Essential documentation
├── examples/              # Key examples only
├── docs_archive/          # Reference/historical docs
├── tests_archive/         # Old tests for reference
├── README.md              # Main readme
├── LICENSE                # License
├── pyproject.toml         # Project config
├── environment.yml        # Conda environment
└── CONTRIBUTING.md        # Contribution guide
```

## What Gets Removed

❌ **Delete permanently:**
- `.pytest_cache/` (temporary)
- `outputs/` (test outputs)
- `__pycache__/` directories (Python cache)

📦 **Archive (keep for reference):**
- WORK_SUMMARY.md
- QUICK_REFERENCE.md  
- CLEANUP_PLAN.md
- CLEANUP_INSTRUCTIONS.txt
- tests_clean_version.py
- AUTOS_TESTS_README.md (already in tests_archive)

## Quick Cleanup Commands

```bash
cd /home/rfsilva/Programs/atomkit

# Move docs to archive
mv WORK_SUMMARY.md QUICK_REFERENCE.md CLEANUP_PLAN.md CLEANUP_INSTRUCTIONS.txt tests_clean_version.py docs_archive/

# Replace tests
rm -rf tests/
mv tests_clean/ tests/

# Remove temporary files
rm -rf .pytest_cache/ outputs/
find . -type d -name "__pycache__" -exec rm -rf {} +

# Optional: Archive AUTOS reference data if not needed
# mv as_tests/ docs_archive/autos_reference_data/
```

## Test the Clean Setup

After cleanup:
```bash
micromamba run -n atomkit pytest tests/ -v
```

Should see ~52 tests pass.

## What to Keep

✅ **Essential files:**
- README.md - Main project documentation
- CONTRIBUTING.md - How to contribute
- LICENSE - Project license
- SETUP_GUIDE.md - Installation instructions
- pyproject.toml - Python project config
- environment.yml - Conda environment
- run_tests.py - Test runner
- verify_install.py - Installation checker

✅ **Essential directories:**
- src/atomkit/ - Core source code
- tests/ - Test suite (clean version)
- docs/ - Core documentation
- examples/ - Usage examples

✅ **Archive directories (keep for reference):**
- tests_archive/ - Old comprehensive tests
- docs_archive/ - Historical documentation
