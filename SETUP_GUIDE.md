# AtomKit Setup Guide

This guide ensures you're using AtomKit within the proper virtual environment for consistent behavior and dependency management.

## Quick Setup (Recommended)

### Using Conda

```bash
# 1. Clone the repository
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit

# 2. Create and activate the atomkit environment
conda create -n atomkit python=3.13
conda activate atomkit

# 3. Install AtomKit in development mode
pip install -e .

# 4. Verify installation
python verify_install.py

# 5. Run tests
pytest tests/ -v
```

### Using Conda with environment.yml

```bash
# 1. Clone the repository
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit

# 2. Create environment from file
conda env create -f environment.yml

# 3. Activate environment
conda activate atomkit

# 4. Install AtomKit
pip install -e .

# 5. Verify installation
python verify_install.py
```

### Using Poetry

```bash
# 1. Clone the repository
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit

# 2. Install with Poetry (creates virtual environment automatically)
poetry install

# 3. Activate Poetry shell
poetry shell

# 4. Verify installation
python verify_install.py
```

## Virtual Environment Best Practices

### Always Activate the Environment

Before using AtomKit, **always activate the virtual environment**:

```bash
# With Conda
conda activate atomkit

# With Poetry
poetry shell
```

### Check Current Environment

To verify you're in the correct environment:

```bash
# Check which Python is being used
which python

# Check if atomkit is installed
python -c "import atomkit; print(atomkit.__file__)"

# Run verification script
python verify_install.py
```

### Running Scripts

Always ensure the environment is active when running scripts:

```bash
# Activate first
conda activate atomkit

# Then run your script
python your_script.py

# Or use the FAC reader
python examples/fac_reader_example.py
```

### Running Tests

Tests should always be run within the atomkit environment:

```bash
# Activate environment
conda activate atomkit

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_configuration.py -v

# Run with coverage
pytest tests/ --cov=src/atomkit --cov-report=html
```

## Optional Dependencies

AtomKit has optional dependencies for enhanced features:

### Plotting Support

```bash
# Install plotting dependencies
pip install matplotlib seaborn

# Or with Poetry
poetry install --extras "plotting"
```

### Enhanced Logging

```bash
# Install colored logging
pip install colorlog

# Or with Poetry
poetry install --extras "logging"
```

### All Optional Features

```bash
# Install everything
pip install -e ".[all]"

# Or with Poetry
poetry install --extras "all"
```

## Troubleshooting

### "Module not found" errors

If you get `ModuleNotFoundError: No module named 'atomkit'`:

1. Check you're in the correct environment: `conda activate atomkit`
2. Reinstall in development mode: `pip install -e .`
3. Verify installation: `python verify_install.py`

### Wrong Python version

AtomKit requires Python 3.10 or higher. Check your version:

```bash
python --version
```

If needed, create a new environment with the correct version:

```bash
conda create -n atomkit python=3.13
```

### Missing dependencies

If dependencies are missing:

```bash
# Reinstall from pyproject.toml
pip install -e .

# Or recreate the conda environment
conda env remove -n atomkit
conda env create -f environment.yml
```

### Tests failing

If tests fail unexpectedly:

1. Ensure environment is activated
2. Reinstall dependencies: `pip install -e .`
3. Clear pytest cache: `pytest --cache-clear tests/`
4. Check for errors: Run verification script

## IDE Configuration

### VS Code

Add to `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "/path/to/anaconda/envs/atomkit/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Click gear icon → Add
3. Select "Conda Environment"
4. Choose existing environment: `atomkit`

### Jupyter Notebook

To use AtomKit in Jupyter:

```bash
# Activate environment
conda activate atomkit

# Install ipykernel
pip install ipykernel

# Add kernel to Jupyter
python -m ipykernel install --user --name=atomkit --display-name="Python (atomkit)"

# Launch Jupyter
jupyter notebook
```

Then select the "Python (atomkit)" kernel in your notebook.

## Development Workflow

### Making Changes

1. Activate environment: `conda activate atomkit`
2. Make your changes to the code
3. Run tests: `pytest tests/ -v`
4. Check types: `mypy src/` (optional)
5. Format code: `black src/ tests/`

### Adding Dependencies

When adding new dependencies:

1. Add to `pyproject.toml` under `[tool.poetry.dependencies]`
2. Reinstall: `pip install -e .`
3. Update `environment.yml` for conda users

### Creating a Release

```bash
# Activate environment
conda activate atomkit

# Run full test suite
pytest tests/ -v

# Check code quality
black --check src/ tests/
ruff check src/ tests/

# Update version in pyproject.toml

# Create git tag
git tag v0.1.0
git push origin v0.1.0
```

## Getting Help

- **Documentation**: See [README.md](README.md) and [API_REFERENCE.md](API_REFERENCE.md)
- **Examples**: Check the `examples/` directory
- **Issues**: Report bugs on [GitHub Issues](https://github.com/rfsilva13/atomkit/issues)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Summary Checklist

- [ ] Virtual environment created (`conda create -n atomkit python=3.13`)
- [ ] Environment activated (`conda activate atomkit`)
- [ ] AtomKit installed (`pip install -e .`)
- [ ] Installation verified (`python verify_install.py`)
- [ ] Tests passing (`pytest tests/ -v`)
- [ ] Optional dependencies installed (if needed)
- [ ] IDE configured to use atomkit environment

**Remember: Always activate the atomkit environment before working with the package!**
