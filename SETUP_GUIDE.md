# AtomKit — Modern Developer Setup Guide (Python 3.14+)

This guide explains a modern, reproducible development workflow for AtomKit targeting Python 3.14 (the latest stable version). We support Python 3.10+ for compatibility, but development and testing prioritize 3.14. 

---

## Prerequisites

- Git
- Python 3.14 (recommended) or Python 3.10+
- One of: micromamba, conda, venv, Poetry, pip, uv, or pip-tools

---

## Environment Setup Options

### Option 1: Micromamba (Recommended for Speed and Reproducibility)

Micromamba is a lightweight conda alternative that's fast and works across platforms.

1. Install micromamba: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

2. Clone and setup:

```bash
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit
micromamba create -n atomkit python=3.14 -y
micromamba activate atomkit
pip install -e .
python verify_install.py
pytest tests/ -v
```

### Option 2: Conda (Full Conda)

If you prefer the full conda experience:

```bash
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit
conda create -n atomkit python=3.14 -y
conda activate atomkit
pip install -e .
python verify_install.py
pytest tests/ -v
```

Or use the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate atomkit
pip install -e .
```

### Option 3: Python venv (Built-in, No Extra Tools)

For a pure Python setup without conda:

```bash
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit
python3.14 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
python verify_install.py
pytest tests/ -v
```

### Option 4: pyenv + venv (Version Management)

If you manage multiple Python versions:

```bash
# Install pyenv: https://github.com/pyenv/pyenv#installation
pyenv install 3.14.0
pyenv local 3.14.0
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python verify_install.py
pytest tests/ -v
```

### Option 5: Poetry (Dependency Management)

Poetry handles dependencies and creates environments automatically.

1. Install Poetry: https://python-poetry.org/docs/#installation

2. Setup:

```bash
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit
poetry install
poetry shell
python verify_install.py
pytest tests/ -v
```

### Option 6: uv (Fast Package Installer)

uv is a modern, fast alternative to pip.

1. Install uv: https://github.com/astral-sh/uv

2. Setup:

```bash
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit
uv venv .venv
source .venv/bin/activate
uv pip install -e .
python verify_install.py
pytest tests/ -v
```

### Option 7: pip-tools (Strict Dependency Pinning)

For advanced dependency management:

1. Install pip-tools: `pip install pip-tools`

2. Setup:

```bash
git clone https://github.com/rfsilva13/atomkit.git
cd atomkit
python3.14 -m venv .venv
source .venv/bin/activate
pip install pip-tools
pip-compile  # Generates requirements.txt from pyproject.toml
pip install -r requirements.txt
pip install -e .
python verify_install.py
pytest tests/ -v
```

---

## Developer Tooling (All Options)

Regardless of your environment setup, use these tools for code quality:

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Test
pytest tests/ -v

# Coverage
pytest --cov=src/atomkit --cov-report=html
```

---

## Installation with Extras

Add optional dependencies as needed:

```bash
# Plotting and logging
pip install -e "[plotting,logging]"

# All extras
pip install -e "[all]"
```

Poetry: `poetry install --extras "plotting logging"`

uv: `uv pip install -e "[all]"`

---

## Running Analyses and Examples

Analysis scripts are in `tools/` (ignored by git). Example data in `work/`.

```bash
# Activate your environment (choose your tool)
micromamba activate atomkit  # or conda, poetry shell, source .venv/bin/activate, etc.

# Run an analysis
python tools/fac_analyze.py --element Fe --shell K
```

---

## Testing and CI

Local tests: `pytest tests/ -q`

CI: Use GitHub Actions with matrix for Python 3.10, 3.11, 3.14.

---

## Contributing

- Use Python 3.14 for development.
- Update `environment.yml` for conda users.
- Follow black/ruff/mypy standards.

See `CONTRIBUTING.md` for details.

---

This guide ensures flexibility — pick the environment manager that fits your workflow!
