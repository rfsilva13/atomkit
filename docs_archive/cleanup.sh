#!/bin/bash
# Quick cleanup script for atomkit repository

echo "🧹 Cleaning up atomkit repository..."

# Move documentation to archive
echo "📦 Archiving documentation..."
mv WORK_SUMMARY.md QUICK_REFERENCE.md CLEANUP_PLAN.md CLEANUP_INSTRUCTIONS.txt tests_clean_version.py docs_archive/ 2>/dev/null

# Replace tests with clean version
echo "✨ Replacing tests with clean version..."
rm -rf tests/
mv tests_clean/ tests/

# Remove temporary files
echo "🗑️  Removing temporary files..."
rm -rf .pytest_cache/ outputs/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "✅ Cleanup complete!"
echo ""
echo "📊 Test the clean setup:"
echo "   micromamba run -n atomkit pytest tests/ -v"
