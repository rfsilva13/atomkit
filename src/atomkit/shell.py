# atomkit/src/atomkit/shell.py

"""
Defines the Shell class representing an atomic electron shell, subshell,
or relativistic sub-subshell.
"""

import re
from typing import Optional, Tuple, Union

# Use absolute import based on package structure assuming definitions.py is in the same directory
# If definitions.py is in the parent directory or elsewhere, adjust the import path.
# Assuming definitions.py is in the same directory (src/atomkit/definitions.py)
from .definitions import validate_j_quantum  # Import the new validation function
from .definitions import ANGULAR_MOMENTUM_MAP, L_QUANTUM_MAP, get_max_shell_occupation


class Shell:
    """
    Represents an atomic electron shell, subshell, or relativistic sub-subshell.

    Encapsulates properties like n, l, j, occupation, and provides methods
    for parsing, manipulation, and representation. Shell structure is defined
    by (n, l, j), while occupation represents the state.

    Attributes:
        n (int): Principal quantum number (>= 1).
        l_quantum (int): Orbital angular momentum quantum number (0=s, 1=p, ...).
        occupation (int): Number of electrons in the shell (0 <= occupation <= max_occupation).
        j_quantum (Optional[float]): Total angular momentum (e.g., 1.5 for p+).
                                     None if not specified or non-relativistic.
                                     Must be half-integer (e.g., 0.5, 1.5).
    """

    # Using slots can slightly improve memory usage for many Shell objects
    __slots__ = ("n", "l_quantum", "_occupation", "_j_quantum")

    def __init__(
        self,
        n: int,
        l_quantum: int,
        occupation: int,
        j_quantum: Optional[Union[float, int]] = None,
    ):
        """
        Initializes a Shell object with validation.

        Args:
            n: Principal quantum number (>= 1).
            l_quantum: Orbital angular momentum quantum number (0=s, 1=p, ...). Must be < n.
            occupation: Number of electrons in the shell (>= 0).
            j_quantum: Total angular momentum (optional). Must be consistent with l
                       (j = l +/- 0.5, or j=0.5 for l=0). Accepts int or float.

        Raises:
            ValueError: If quantum numbers or occupation are invalid or inconsistent.
            TypeError: If input types are incorrect.
        """
        # Validate types FIRST
        if not isinstance(n, int):
            raise TypeError(
                f"Principal quantum number n must be an integer, got type {type(n)}."
            )
        if not isinstance(l_quantum, int):
            raise TypeError(
                f"l quantum number must be an integer, got type {type(l_quantum)}."
            )
        if not isinstance(occupation, int):
            raise TypeError(
                f"Occupation must be an integer, got type {type(occupation)}."
            )
        if j_quantum is not None and not isinstance(j_quantum, (float, int)):
            raise TypeError(
                f"j quantum number must be float or int, got type {type(j_quantum)}."
            )
        # End Type Validation

        # Validate n value
        if n < 1:
            raise ValueError(f"Principal quantum number n must be >= 1, got {n}.")

        # Validate l value
        if l_quantum < 0:
            raise ValueError(f"l quantum number must be non-negative, got {l_quantum}.")

        # --- FIXED: Enforce l < n ---
        if l_quantum >= n:
            raise ValueError(
                f"l quantum number ({l_quantum}) must be less than n ({n})."
            )
        # --- End Fix ---

        self.n: int = n
        self.l_quantum: int = l_quantum
        self._j_quantum: Optional[float] = None  # Internal storage for j

        # Validate and store j if provided, using the dedicated function
        if j_quantum is not None:
            # Delegate validation to the function from definitions.py
            self._j_quantum = validate_j_quantum(self.l_quantum, j_quantum)
        # else: j remains None

        # Validate and set occupation using the property setter *after* j is set
        self._occupation: int
        self.occupation = occupation  # Use property setter

    @property
    def occupation(self) -> int:
        """Gets the number of electrons in the shell."""
        if not hasattr(self, "_occupation"):
            return 0  # Should not happen in normal use after init
        return self._occupation

    @occupation.setter
    def occupation(self, value: int) -> None:
        """Sets the number of electrons, validating against max occupancy."""
        # Type check moved to __init__
        if value < 0:
            raise ValueError(f"Occupation must be a non-negative integer, got {value}.")

        max_occ = self.max_occupation()
        if value > max_occ:
            raise ValueError(
                f"Occupation ({value}) exceeds maximum ({max_occ}) for shell with n={self.n} l={self.l_quantum} and j={self.j_symbol}."
            )
        self._occupation = value

    # --- Rest of the Shell class methods remain the same ---
    # (Properties: j_quantum, l_symbol, j_symbol, max_occupation, is_full, holes, energy_key)
    # (__str__, __repr__, __eq__, has_same_structure, __lt__, __hash__)
    # (from_string, take_electron, add_electron)
    # ... (Keep the rest of the methods from the previous version) ...

    @property
    def j_quantum(self) -> Optional[float]:
        """Gets the total angular momentum quantum number j, or None."""
        return self._j_quantum

    @property
    def l_symbol(self) -> str:
        """Gets the spectroscopic symbol ('s', 'p', 'd', ...) for the shell's l."""
        # Handle high l values not in the map
        if self.l_quantum not in L_QUANTUM_MAP:
            # Use [l=num] notation as requested for high l
            return f"[l={self.l_quantum}]"
        return L_QUANTUM_MAP[self.l_quantum]

    @property
    def j_symbol(self) -> str:
        """
        Gets the relativistic subshell symbol ('-' for j=l-1/2, '+' for j=l+1/2, '' otherwise).
        Returns '-' or '+' only if j_quantum is set and l > 0.
        """
        # No j symbol if j is not defined or if it's an s-shell (l=0)
        if self.j_quantum is None or self.l_quantum == 0:
            return ""
        # Use proximity check due to potential float inaccuracies
        elif abs(self.j_quantum - (self.l_quantum - 0.5)) < 1e-6:
            return "-"
        elif abs(self.j_quantum - (self.l_quantum + 0.5)) < 1e-6:
            return "+"
        else:
            # This case should ideally not be reached if validation in __init__ is correct
            # Could indicate an issue if it appears.
            return "?"

    def max_occupation(self) -> int:
        """Calculates the maximum electron occupancy for this specific shell/subshell."""
        # Delegates to the function from definitions module
        return get_max_shell_occupation(self.l_quantum, self.j_quantum)

    def is_full(self) -> bool:
        """Checks if the shell is fully occupied."""
        return self.occupation == self.max_occupation()

    def holes(self) -> int:
        """Calculates the number of holes (vacancies) in the shell."""
        return self.max_occupation() - self.occupation

    def energy_key(self) -> Tuple[int, int]:
        """
        Returns a tuple (n+l, n) for sorting shells based on the Madelung (n+l) rule.
        Lower n is preferred for ties in n+l.
        """
        return (self.n + self.l_quantum, self.n)

    def __str__(self) -> str:
        """
        Formats the shell into standard notation (e.g., '3d10', '2p+3', '4[l=21]5').
        Returns empty string if occupation is 0.
        """
        # Access occupation via the property, which now handles potential early access
        current_occupation = self.occupation
        if current_occupation == 0:
            return ""

        # Use properties to get n, l_symbol, j_symbol
        # Always include occupation number for clarity, matching original code behavior
        occ_str = str(current_occupation)
        return f"{self.n}{self.l_symbol}{self.j_symbol}{occ_str}"

    def __repr__(self) -> str:
        """Provides an unambiguous string representation for debugging."""
        # Access occupation via property for safety during potential init issues
        current_occupation = (
            self.occupation if hasattr(self, "_occupation") else "Unset"
        )
        j_repr = f", j_quantum={self.j_quantum}" if self.j_quantum is not None else ""
        return f"Shell(n={self.n}, l_quantum={self.l_quantum}, occupation={current_occupation}{j_repr})"

    def __eq__(self, other: object) -> bool:
        """
        Checks for equality based on shell structure (n, l, j) AND occupation.
        Two Shell objects are equal if they represent the exact same state.
        """
        if not isinstance(other, Shell):
            return NotImplemented
        # Compare n, l, j, and occupation for full equality
        return (
            self.n == other.n
            and self.l_quantum == other.l_quantum
            and self.j_quantum == other.j_quantum  # Use property access
            and self.occupation == other.occupation
        )  # Use property access

    def has_same_structure(self, other: object) -> bool:
        """Checks if two shells have the same quantum numbers (n, l, j), ignoring occupation."""
        if not isinstance(other, Shell):
            return False
        return (
            self.n == other.n
            and self.l_quantum == other.l_quantum
            and self.j_quantum == other.j_quantum
        )  # Use property access

    def __lt__(self, other: "Shell") -> bool:
        """
        Compares shells for sorting. Primarily uses Madelung rule (n+l, n).
        For shells with the same (n+l, n), sorts by j (None < l-1/2 < l+1/2).
        If structure is identical, sorts by occupation (lower first).
        """
        if not isinstance(other, Shell):
            return NotImplemented

        # Primary sort: Madelung rule
        if self.energy_key() != other.energy_key():
            return self.energy_key() < other.energy_key()

        # Secondary sort: j value (if applicable)
        # Treat None as coming before any specific j value
        self_j = self.j_quantum
        other_j = other.j_quantum

        if self_j is None and other_j is not None:
            return True  # self (no j) comes before other (has j)
        if self_j is not None and other_j is None:
            return False  # self (has j) comes after other (no j)
        if self_j is not None and other_j is not None:
            # If j values are different (within tolerance)
            if abs(self_j - other_j) > 1e-6:
                return self_j < other_j
        # If n, l, and j status are the same, they are structurally equal for sorting.
        # Tertiary sort: Occupation (lower occupation comes first)
        if self.occupation != other.occupation:
            return self.occupation < other.occupation  # Use property access

        # If all else is equal, they are considered equal for sorting purposes
        return False

    def __hash__(self) -> int:
        """
        Allows Shell objects to be used as dictionary keys or in sets.
        Hashing is based on the shell structure (n, l, j) AND occupation
        to match the __eq__ definition.
        """
        # Use property access
        return hash((self.n, self.l_quantum, self.j_quantum, self.occupation))

    @classmethod
    def from_string(cls, shell_string: str) -> "Shell":
        """
        Parses a shell string into a Shell object.

        Handles standard notation (e.g., '3d10', '2s1', '4f') and
        relativistic notation (e.g., '2p+3', '2p-1').
        Also handles high-l notation like '4[l=21]5'.
        Assumes occupation is 1 if not specified at the end.

        Args:
            shell_string: The string representation of the shell.

        Returns:
            A Shell object instance.

        Raises:
            ValueError: If the string format is invalid.
        """
        shell_string = shell_string.strip()
        if not shell_string:
            raise ValueError("Input shell string cannot be empty.")

        # Regex to capture:
        # 1. n (digits)
        # 2. l_symbol (single letter OR [l=digits])
        # 3. optional j symbol (+ or -)
        # 4. optional occupation (digits)
        # Updated pattern to handle [l=num]
        pattern = r"^(\d+)(\[l=(\d+)\]|[a-zA-Z])([+\-])?(\d+)?$"
        match = re.fullmatch(pattern, shell_string)

        if not match:
            raise ValueError(f"Invalid shell string format: '{shell_string}'")

        n_str, l_part, l_num_in_bracket, j_sym, occ_str = match.groups()

        # --- Validation and Conversion ---
        try:
            n = int(n_str)
        except ValueError:
            raise ValueError(f"Invalid principal quantum number 'n': {n_str}")

        # Determine l_quantum
        if l_num_in_bracket:  # Case: [l=num]
            # --- FIXED: Disallow j_symbol with [l=num] ---
            if j_sym:
                raise ValueError(
                    f"Cannot use j symbol ('{j_sym}') with [l=num] notation in '{shell_string}'"
                )
            # --- End Fix ---
            try:
                l_quantum = int(l_num_in_bracket)
                if l_quantum < 0:
                    raise ValueError("l quantum number in bracket must be non-negative")
            except ValueError:
                raise ValueError(
                    f"Invalid l quantum number in bracket notation: '{l_num_in_bracket}'"
                )
        elif l_part:  # Case: standard letter symbol
            l_sym = l_part.lower()
            if l_sym not in ANGULAR_MOMENTUM_MAP:
                # --- FIXED: Raise error for invalid l symbol like 'x' ---
                raise ValueError(
                    f"Unknown or invalid l symbol: '{l_sym}' in '{shell_string}'"
                )
                # --- End Fix ---
            l_quantum = ANGULAR_MOMENTUM_MAP[l_sym]
        else:
            # Should not happen with the regex, but defensive check
            raise ValueError(
                f"Could not determine l quantum number from '{shell_string}'"
            )

        # Determine occupation (defaults to 1 if not present)
        try:
            occupation = int(occ_str) if occ_str else 1
        except ValueError:
            # --- FIXED: Catch non-digit occupation ---
            raise ValueError(
                f"Invalid occupation number: '{occ_str}' in '{shell_string}'"
            )
            # --- End Fix ---

        # Determine j_quantum based on j_symbol (+/-)
        j_quantum: Optional[float] = None
        if j_sym:
            # Check already done above to disallow j_sym with [l=num]
            if l_quantum == 0:  # s shell (l=0) only has j=1/2
                raise ValueError(
                    f"j symbol ('{j_sym}') not applicable for s shell (l=0) in '{shell_string}'"
                )
            elif j_sym == "+":
                j_quantum = l_quantum + 0.5
            elif j_sym == "-":
                j_quantum = l_quantum - 0.5
                # This check might be redundant due to validation in __init__, but safe to keep
                if j_quantum < 0.5:
                    raise ValueError(
                        f"Resulting j quantum number ({j_quantum}) is invalid for l={l_quantum}"
                    )
        # If no j_symbol is present, j_quantum remains None (non-relativistic subshell)

        # Create Shell object (further validation happens in __init__)
        # The __init__ will perform the final checks (e.g., occupation vs max_occ, l < n)
        return cls(n=n, l_quantum=l_quantum, occupation=occupation, j_quantum=j_quantum)

    # --- Instance Methods based on original functions ---
    def take_electron(self) -> "Shell":
        """
        Returns a *new* Shell instance with one fewer electron.
        Does not modify the original shell.

        Returns:
            A new Shell object with occupation decreased by 1.

        Raises:
            ValueError: If the shell occupation is already 0.
        """
        # Use property getter
        if self.occupation <= 0:
            raise ValueError(
                f"Cannot remove electron from shell with occupation 0: {self!s}"
            )
        # Create a new instance with decremented occupation
        return Shell(self.n, self.l_quantum, self.occupation - 1, self.j_quantum)

    def add_electron(self) -> "Shell":
        """
        Returns a *new* Shell instance with one more electron.
        Does not modify the original shell.

        Returns:
            A new Shell object with occupation increased by 1.

        Raises:
            ValueError: If adding an electron would exceed max occupancy.
        """
        # Use property getter
        if self.occupation >= self.max_occupation():
            raise ValueError(f"Cannot add electron to full shell: {self!s}")
        # Create a new instance with incremented occupation
        return Shell(self.n, self.l_quantum, self.occupation + 1, self.j_quantum)
