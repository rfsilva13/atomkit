"""
Minimal essential tests for Shell module.
"""

import pytest
from atomkit import Shell


class TestShellCreation:
    """Test basic shell creation and validation."""

    def test_simple_shell(self):
        """Create a simple 1s shell."""
        shell = Shell(n=1, l_quantum=0, occupation=2)
        assert shell.n == 1
        assert shell.l_quantum == 0
        assert shell.occupation == 2

    def test_2p_shell(self):
        """Create a 2p shell."""
        shell = Shell(n=2, l_quantum=1, occupation=6)
        assert shell.n == 2
        assert shell.l_quantum == 1
        assert shell.occupation == 6

    def test_3d_shell(self):
        """Create a 3d shell."""
        shell = Shell(n=3, l_quantum=2, occupation=10)
        assert shell.n == 3
        assert shell.l_quantum == 2
        assert shell.occupation == 10


class TestShellValidation:
    """Test shell validation rules."""

    def test_invalid_n(self):
        """Test that n < 1 raises error."""
        with pytest.raises(ValueError):
            Shell(n=0, l_quantum=0, occupation=1)

    def test_invalid_l_greater_than_n(self):
        """Test that l >= n raises error."""
        with pytest.raises(ValueError):
            Shell(n=1, l_quantum=1, occupation=1)  # l must be < n

    def test_negative_l(self):
        """Test that negative l raises error."""
        with pytest.raises(ValueError):
            Shell(n=2, l_quantum=-1, occupation=1)

    def test_negative_occupation(self):
        """Test that negative occupation raises error."""
        with pytest.raises(ValueError):
            Shell(n=1, l_quantum=0, occupation=-1)


class TestShellRelativisticQuantumNumbers:
    """Test j quantum numbers for relativistic shells."""

    def test_j_quantum_for_p_shell(self):
        """Test p shell with j quantum number."""
        shell_p_minus = Shell(n=2, l_quantum=1, occupation=2, j_quantum=0.5)
        shell_p_plus = Shell(n=2, l_quantum=1, occupation=4, j_quantum=1.5)

        assert shell_p_minus.j_quantum == 0.5
        assert shell_p_plus.j_quantum == 1.5

    def test_j_quantum_for_s_shell(self):
        """Test s shell with j quantum number."""
        shell = Shell(n=1, l_quantum=0, occupation=2, j_quantum=0.5)
        assert shell.j_quantum == 0.5


class TestShellStringRepresentation:
    """Test shell string conversion."""

    def test_1s_repr(self):
        """Test 1s shell string representation."""
        shell = Shell(n=1, l_quantum=0, occupation=2)
        s = str(shell)
        assert "1s" in s
        assert "2" in s

    def test_2p_repr(self):
        """Test 2p shell string representation."""
        shell = Shell(n=2, l_quantum=1, occupation=6)
        s = str(shell)
        assert "2p" in s
        assert "6" in s
