# Contributing to AtomKit

Thank you for your interest in contributing to AtomKit! This document provides guidelines and instructions for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Conda or Poetry (recommended for environment management)

### Development Setup

1. **Fork and Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/atomkit.git
cd atomkit
```

2. **Create a Development Environment**

Using Conda (recommended):
```bash
conda create -n atomkit-dev python=3.13
conda activate atomkit-dev
pip install -e ".[all]"
```

Or using the environment file:
```bash
conda env create -f environment.yml
conda activate atomkit
```

Using Poetry:
```bash
poetry install --extras "all" --with dev
poetry shell
```

3. **Verify Installation**

```bash
# Run the test suite
pytest tests/ -v

# Should see: 162 passed, 1 skipped
```

## Development Workflow

### Making Changes

1. **Create a Feature Branch**

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

2. **Make Your Changes**

- Write clear, documented code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

3. **Run Tests and Quality Checks**

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/atomkit --cov-report=html

# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Type checking (optional but encouraged)
mypy src/
```

4. **Commit Your Changes**

```bash
git add .
git commit -m "Brief description of your changes"
```

Use clear commit messages:
- `feat: Add support for AUTOSTRUCTURE parsing`
- `fix: Correct energy unit conversion in FAC reader`
- `docs: Update installation instructions`
- `test: Add tests for Configuration.from_element`

5. **Push and Create Pull Request**

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style Guidelines

### Python Style

- Follow PEP 8 guidelines
- Use Black for code formatting (line length: 88)
- Use type hints for function arguments and return values
- Use modern Python 3.10+ type hints (e.g., `list[str]` instead of `List[str]`)

### Example

```python
def calculate_energy(
    level: int,
    charge: int,
    unit: str = "eV"
) -> float:
    """
    Calculate energy for a given level.
    
    Parameters
    ----------
    level : int
        Quantum level number.
    charge : int
        Ion charge state.
    unit : str, optional
        Output energy unit, by default "eV".
    
    Returns
    -------
    float
        Energy in specified units.
    """
    energy = 13.6 * charge**2 / level**2
    return energy_converter.convert(energy, "eV", unit)
```

### Documentation

- Use NumPy-style docstrings
- Include Parameters, Returns, Examples sections
- Add type hints in code, not just in docstrings
- Update README.md and API_REFERENCE.md for new features

### Testing

- Write tests for all new functionality
- Aim for >90% code coverage
- Use descriptive test names: `test_from_string_with_j_quantum`
- Group related tests in classes

```python
class TestConfiguration:
    def test_from_string_basic(self):
        """Test basic configuration parsing from string."""
        config = Configuration.from_string("1s2.2s1")
        assert config.total_electrons == 3
        
    def test_from_string_invalid(self):
        """Test that invalid strings raise ValueError."""
        with pytest.raises(ValueError):
            Configuration.from_string("invalid")
```

## Type Checking

We use modern Python type hints throughout the codebase:

- **Use built-in types**: `list`, `dict`, `tuple`, `set` (not `List`, `Dict`, etc.)
- **Use `Union` or `|` for multiple types**: `Union[int, float]` or `int | float`
- **Use `Optional` for nullable**: `Optional[str]` or `str | None`
- **Import from `typing` only when needed**: `Any`, `Callable`, `Iterable`, etc.

### Type Checking Setup

```python
from typing import Any, Optional, Union
import numpy as np

def process_data(
    data: list[float],
    weights: Optional[np.ndarray] = None,
    method: str = "mean"
) -> Union[float, np.ndarray]:
    """Example function with proper type hints."""
    ...
```

Run type checking:
```bash
mypy src/atomkit
```

## Project Structure

```
atomkit/
├── src/atomkit/              # Main package source
│   ├── __init__.py          # Package initialization
│   ├── configuration.py     # Configuration and Shell classes
│   ├── shell.py             # Shell representation
│   ├── definitions.py       # Constants and mappings
│   ├── readers/             # File parsers
│   │   ├── __init__.py
│   │   ├── base.py          # Base reader class
│   │   ├── levels.py        # FAC level reader
│   │   ├── transitions.py   # FAC transition reader
│   │   └── autoionization.py
│   └── physics/             # Physics calculations
│       ├── __init__.py
│       ├── units.py         # Energy conversion
│       ├── cross_sections.py
│       └── plotting.py
├── tests/                   # Test suite
│   ├── test_configuration.py
│   ├── test_physics.py
│   └── test_plotting.py
├── examples/                # Usage examples
├── docs/                    # Additional documentation
├── pyproject.toml           # Poetry configuration
├── environment.yml          # Conda environment
└── README.md               # Main documentation
```

## Areas for Contribution

We welcome contributions in the following areas:

### High Priority
- [ ] NIST database parser and utilities
- [ ] AUTOSTRUCTURE output parser
- [ ] ADAS file format support
- [ ] Additional test coverage
- [ ] Documentation improvements

### Medium Priority
- [ ] Grotrian diagram plotting
- [ ] Mixing coefficients analysis
- [ ] Performance optimizations
- [ ] Extended examples and tutorials

### Low Priority
- [ ] Additional visualization options
- [ ] Export to different file formats
- [ ] Integration with other atomic codes

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Steps to Reproduce**: Minimal code example
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, package versions

Example:
```python
# Bug: Configuration.from_string fails with j-quantum numbers

from atomkit import Configuration

# This should work but raises ValueError
config = Configuration.from_string("2p-1.2p+3")

# Error message:
# ValueError: Invalid shell notation: 2p-1
```

## Questions and Support

- **Issues**: For bugs and feature requests, open an issue on GitHub
- **Discussions**: For questions and general discussion, use GitHub Discussions
- **Email**: For private inquiries, contact rfsilva@lip.pt

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them contribute
- Focus on what is best for the community
- Show empathy towards other community members

## License

By contributing to AtomKit, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in the README.md and release notes. Significant contributions may lead to co-authorship on related publications.

Thank you for contributing to AtomKit! 🚀
