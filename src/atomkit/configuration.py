# atomkit/src/atomkit/configuration.py

"""
Defines the Configuration class representing the electron configuration
of an atom or ion using a collection of Shell objects.
"""

import copy  # For deep copying configurations during generation
import itertools  # For generating permutations/combinations
import re
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import mendeleev  # Import mendeleev package

# Import constants needed
from .definitions import SHELL_LABEL_MAP  # Import the X-ray label map
from .definitions import ANGULAR_MOMENTUM_MAP, L_SYMBOLS, get_max_shell_occupation

# Use absolute import based on package structure
from .shell import Shell


class Configuration:
    """
    Represents the electron configuration of an atom or ion.

    Stores a collection of Shell objects, keyed internally by their
    structure (n, l, j). Provides methods for parsing configuration strings,
    accessing shells, calculating total electrons, and generating related
    configurations (holes, excitations).

    Attributes:
        shells (List[Shell]): A sorted list of Shell objects in the configuration.
    """

    # No __slots__ here, as flexibility might be needed,
    # and the number of Configuration objects is usually much smaller than Shells.

    def __init__(self, shells: Optional[Iterable[Shell]] = None):
        """
        Initializes an electron configuration.

        Args:
            shells: An optional iterable of Shell objects to initialize with.
                    Shells with the same structure (n, l, j) will have their
                    occupations summed if provided multiple times. Shells with
                    zero occupation are ignored during initialization.
        """
        # Internal storage: Key is the hashable tuple (n, l_quantum, j_quantum)
        # Value is the Shell object itself.
        self._shells: Dict[Tuple[int, int, Optional[float]], Shell] = {}
        if shells:
            for shell in shells:
                # Combine occupations for shells with the same structure during init
                self.add_shell(shell, combine_occupation=True)

    def add_shell(self, shell: Shell, combine_occupation: bool = False) -> None:
        """
        Adds or updates a shell in the configuration.

        Args:
            shell: The Shell object to add.
            combine_occupation: If True and a shell with the same structure
                                (n, l, j) already exists, their occupations
                                are summed (raising ValueError if max occupancy
                                is exceeded). If False (default), the existing
                                shell is replaced by the new one. Shells with
                                zero occupation result in the removal of that
                                shell structure from the configuration.

        Raises:
            TypeError: If the input is not a Shell object.
            ValueError: If combining occupations exceeds the maximum allowed.
        """
        if not isinstance(shell, Shell):
            raise TypeError("Can only add Shell objects to a Configuration.")

        # Key based on shell structure (n, l, j)
        key = (shell.n, shell.l_quantum, shell.j_quantum)

        if shell.occupation <= 0:
            # Remove the shell if it exists and occupation is zero or less
            self._shells.pop(key, None)  # Safely remove if key exists
            return

        if combine_occupation and key in self._shells:
            existing_shell = self._shells[key]
            new_occupation = existing_shell.occupation + shell.occupation
            # Create a new shell with the combined occupation (validation in Shell init)
            try:
                # Use existing shell's n, l, j to ensure consistency
                combined_shell = Shell(
                    existing_shell.n,
                    existing_shell.l_quantum,
                    new_occupation,
                    existing_shell.j_quantum,
                )
                self._shells[key] = combined_shell
            except ValueError as e:
                # Re-raise with more context if combination exceeds max occupancy
                raise ValueError(
                    f"Cannot add shell {shell!s}: combining occupations "
                    f"({existing_shell.occupation} + {shell.occupation} = {new_occupation}) "
                    f"exceeds maximum for shell structure. Original error: {e}"
                )
        else:
            # Add new shell or replace existing one
            self._shells[key] = shell

    def get_shell(
        self, n: int, l_quantum: int, j_quantum: Optional[float] = None
    ) -> Optional[Shell]:
        """
        Retrieves a specific shell from the configuration based on its structure.

        Args:
            n: Principal quantum number.
            l_quantum: Orbital angular momentum quantum number.
            j_quantum: Total angular momentum (optional).

        Returns:
            The Shell object if found, otherwise None.
        """
        key = (n, l_quantum, j_quantum)
        return self._shells.get(key)

    @property
    def shells(self) -> List[Shell]:
        """
        Returns a list of all Shell objects currently in the configuration,
        sorted by energy key (Madelung rule), then j, then occupation.
        """
        # Sort the values (Shell objects) of the internal dictionary
        return sorted(self._shells.values())

    def total_electrons(self) -> int:
        """Calculates the total number of electrons in the configuration."""
        return sum(shell.occupation for shell in self._shells.values())

    def copy(self) -> "Configuration":
        """Creates a deep copy of the configuration."""
        # Create new Shell objects to ensure deep copy
        new_shells = [
            Shell(s.n, s.l_quantum, s.occupation, s.j_quantum)
            for s in self._shells.values()
        ]
        return Configuration(new_shells)

    def get_ionstage(self, element_identifier: Union[int, str]) -> int:
        """
        Calculates the ion stage (charge) of this configuration relative
        to the neutral atom of the specified element.

        Ion Stage = Z - Ne
        where Z is the atomic number and Ne is the number of electrons in this config.

        Args:
            element_identifier: The element's atomic number (int), symbol (str),
                                or name (str). Used to determine Z.

        Returns:
            The calculated ion stage (integer).

        Raises:
            ValueError: If the element identifier is invalid.
            ImportError: If the mendeleev package is not installed.
        """
        # Check if mendeleev is available (optional, could let it fail on import)
        try:
            import mendeleev
        except ImportError:
            raise ImportError(
                "The 'mendeleev' package is required for get_ionstage method."
            )

        try:
            element = mendeleev.element(element_identifier)
            atomic_number = element.atomic_number
        except Exception as e:
            raise ValueError(f"Invalid element identifier '{element_identifier}': {e}")

        num_electrons = self.total_electrons()
        ion_stage = atomic_number - num_electrons
        return ion_stage  # type: ignore

    def __str__(self) -> str:
        """
        Formats the configuration into standard notation (e.g., '1s2.2s2.2p6'),
        sorted by energy (Madelung rule). Shells are separated by dots.
        Returns an empty string for an empty configuration.
        """
        # Use to_string with default separator
        return self.to_string(separator=".")

    def to_string(self, separator: str = ".") -> str:
        """
        Formats the configuration into string notation with a custom separator.

        Shells are sorted by energy (Madelung rule) and joined with the specified
        separator. This is useful when different programs require different formats.

        Args:
            separator: The string to use between shells. Common values:
                      - "." (default) for dot notation: "1s2.2s2.2p6"
                      - " " for space notation: "1s2 2s2 2p6"
                      - "" for compact notation: "1s22s22p6"
                      - Any other string as needed

        Returns:
            A string representation of the configuration using the specified separator.
            Returns an empty string for an empty configuration.

        Examples:
            >>> config = Configuration.from_string("1s2.2s2.2p6")
            >>> config.to_string()  # Default: dots
            '1s2.2s2.2p6'
            >>> config.to_string(separator=" ")  # Spaces
            '1s2 2s2 2p6'
            >>> config.to_string(separator="")  # Compact
            '1s22s22p6'
        """
        # self.shells property already returns sorted shells
        # Filter out shells that might have ended up with 0 occupation internally
        shell_strings = [str(shell) for shell in self.shells if shell.occupation > 0]
        return separator.join(shell_strings)

    def __repr__(self) -> str:
        """Provides an unambiguous string representation for debugging."""
        # Get repr of sorted shells
        shells_repr = ", ".join(repr(s) for s in self.shells)
        return f"Configuration([{shells_repr}])"

    def __eq__(self, other: object) -> bool:
        """
        Checks if two configurations are identical (contain the exact same set
        of Shell objects, considering n, l, j, and occupation).
        Order does not matter due to internal dictionary storage.
        """
        if not isinstance(other, Configuration):
            return NotImplemented
        # Direct comparison of the internal dictionaries is sufficient
        # because Shell objects with same state hash and compare equally.
        return self._shells == other._shells

    def __len__(self) -> int:
        """Returns the number of distinct shell structures in the configuration."""
        return len(self._shells)

    def __iter__(self) -> Iterator[Shell]:
        """Allows iteration over the shells in the configuration (sorted)."""
        # Iterate over the sorted list provided by the property
        return iter(self.shells)

    def __hash__(self) -> int:
        """
        Makes Configuration objects hashable based on their shells.
        Allows them to be stored in sets.
        """
        # Hash based on a frozenset of the internal shell objects
        # Important: Relies on Shell.__hash__ being correctly implemented
        return hash(frozenset(self._shells.values()))

    def __contains__(
        self, item: Union[Shell, Tuple[int, int, Optional[float]]]
    ) -> bool:
        """
        Checks if a specific shell structure (n, l, j) exists in the configuration,
        regardless of occupation (unless a Shell object with specific occupation is passed).

        Args:
            item: Either a Shell object or a tuple (n, l_quantum, j_quantum).
                  If a Shell object is passed, checks for exact match including occupation.
                  If a tuple is passed, checks only for the shell structure's presence.

        Returns:
            True if the item is found, False otherwise.

        Raises:
            TypeError: If the item is not a Shell or a valid tuple.
        """
        if isinstance(item, Shell):
            # Check for exact shell match (including occupation)
            key = (item.n, item.l_quantum, item.j_quantum)
            # Ensure the shell exists *and* has the same occupation
            return (
                key in self._shells and self._shells[key].occupation == item.occupation
            )
        elif isinstance(item, tuple) and len(item) == 3:
            # Check only for the presence of the shell structure (n, l, j)
            key = item
            return key in self._shells
        else:
            raise TypeError("Can only check for Shell objects or (n, l, j) tuples.")

    @classmethod
    def from_string(cls, config_string: str) -> "Configuration":
        """
        Parses a configuration string into a Configuration object.

        Supports shells separated by dots ('.') or spaces (' ').
        Examples: '1s2.2s2.2p6', '5d10 6p1', '2p-1.2p+3'

        Args:
            config_string: The string representation of the configuration.
                           An empty string results in an empty Configuration.

        Returns:
            A Configuration object instance.

        Raises:
            ValueError: If the string format is invalid or contains invalid shells.
        """
        config_string = config_string.strip()
        if not config_string:
            return cls()  # Return empty configuration for empty string

        # Split by dot or space, filtering out empty parts resulting from multiple spaces
        parts = [part for part in re.split(r"[.\s]+", config_string) if part]

        shells = []
        for part in parts:
            try:
                # Use Shell's parser for each part
                shells.append(Shell.from_string(part))
            except ValueError as e:
                # Add context to the error message if parsing fails
                raise ValueError(
                    f"Error parsing part '{part}' in configuration string '{config_string}': {e}"
                )

        # Initialize the configuration (handles combining occupations if needed)
        return cls(shells)

    @classmethod
    def from_element(
        cls, element_identifier: Union[int, str], ion_charge: int = 0
    ) -> "Configuration":
        """
        Creates a ground state Configuration for an element or ion.

        Uses the mendeleev package to get the neutral ground state configuration
        string and then ionizes it using mendeleev's logic.

        Args:
            element_identifier: The element's atomic number (int), symbol (str),
                                or name (str).
            ion_charge: The charge of the ion (default is 0 for neutral).
                        Currently only supports non-negative charges.

        Returns:
            A Configuration object representing the ground state of the ion.

        Raises:
            ValueError: If the element identifier is invalid, ion_charge is negative,
                        or if ionization charge is invalid for the element.
            ImportError: If the mendeleev package is not installed.
            AttributeError: If mendeleev fails to provide the electronic configuration object.
            RuntimeError: For unexpected errors during processing.
        """
        if ion_charge < 0:
            # TODO: Implement adding electrons for anions if needed later.
            raise ValueError(
                "Negative ion charges (anions) are not currently supported by from_element."
            )

        try:
            element = mendeleev.element(element_identifier)
            element_symbol = element.symbol  # Store for error messages
        except Exception as e:  # Catch potential errors from mendeleev
            raise ValueError(f"Invalid element identifier '{element_identifier}': {e}")

        config_str = None
        try:
            # Get the neutral ElectronicConfiguration object
            neutral_ec = element.ec
            if neutral_ec is None:
                raise AttributeError(
                    f"Could not retrieve electronic configuration attribute for neutral {element_symbol}."
                )

            # Ionize the neutral configuration object if needed
            if ion_charge == 0:
                target_ec = neutral_ec
            else:
                # Let mendeleev handle ionization errors (e.g., charge too high)
                target_ec = neutral_ec.ionize(ion_charge)  # Returns a *new* object

            # Get the string representation from the target object
            config_str = target_ec.to_str()
            if (
                not config_str or config_str == "None"
            ):  # Check for empty or 'None' string
                # This is valid for fully ionized atoms (e.g., H+)
                # Return empty configuration
                return cls()  # Return empty configuration

            # TODO: Handle potential noble gas core notation [Core]... if needed
            # This might require calling get_valence() and combining with core config string
            # if config_str and '[' in config_str:
            #     config_str = cls._expand_noble_gas_core(config_str) # Hypothetical helper

        except AttributeError as e:
            # Catch AttributeErrors from accessing .ec or potentially .ionize/.to_str
            ion_str = (
                f"{element_symbol}{ion_charge:+}"
                if ion_charge != 0
                else f"neutral {element_symbol}"
            )
            raise AttributeError(
                f"Error accessing mendeleev configuration for {ion_str}: {e}. Data might be missing or incomplete."
            )
        except ValueError as e:
            # Catch ValueErrors from ionize() if charge is invalid
            raise ValueError(
                f"Error ionizing {element_symbol} to charge {ion_charge}: {e}"
            )
        except Exception as e:  # Catch other potential errors
            raise RuntimeError(
                f"An unexpected error occurred retrieving/processing configuration from mendeleev: {e}"
            )

        # Parse the final configuration string
        try:
            # Use the class's own string parser
            return cls.from_string(config_str)
        except ValueError as e:
            # Add context if parsing the string from mendeleev fails
            ion_str = (
                f"{element_symbol}{ion_charge:+}"
                if ion_charge != 0
                else f"neutral {element_symbol}"
            )
            raise ValueError(
                f"Error parsing configuration string '{config_str}' obtained from mendeleev for {ion_str}: {e}"
            )

    # --- Updated Method ---
    @classmethod
    def from_compact_string(
        cls, compact_string: str, generate_permutations: bool = False
    ) -> Union["Configuration", List["Configuration"]]:
        """
        Parses a compact configuration string (e.g., '1*2.2*8.3*5')
        into one or more Configuration objects.

        By default (`generate_permutations=False`), fills subshells sequentially
        (s, p, d, ...) for each principal shell N to create a single ground-state-like
        configuration.

        If `generate_permutations=True`, generates all possible valid distributions
        of electrons within each principal shell N and returns a list of all
        resulting unique Configuration objects.

        Args:
            compact_string: String in the format "N1*E1.N2*E2..." where N is the
                            principal quantum number and E is the total number
                            of electrons in that principal shell.
            generate_permutations: If True, generate all valid electron distributions
                                   for each N*E part. Defaults to False.

        Returns:
            If generate_permutations is False: A single Configuration object.
            If generate_permutations is True: A list of unique Configuration objects.

        Raises:
            ValueError: If the string format is invalid, N or E are not integers,
                        or if E exceeds the capacity of shell N.
        """
        if not compact_string.strip():
            # Return empty list if permutations requested, else empty config
            return [] if generate_permutations else cls()

        parts = compact_string.strip().split(".")
        parsed_parts: List[Tuple[int, int]] = []  # Store (n, electrons_in_n)

        for part in parts:
            if not part:
                continue  # Skip empty parts from double dots etc.

            match = re.fullmatch(
                r"(\d+)\*(\d+)", part.strip()
            )  # Use fullmatch and strip part
            if not match:
                raise ValueError(
                    f"Invalid format for compact part: '{part}'. Expected 'N*E'."
                )

            try:
                n = int(match.group(1))
                electrons_in_n = int(match.group(2))
                if n < 1 or electrons_in_n < 0:
                    raise ValueError("N must be >= 1 and E must be >= 0 in N*E.")
                # Check if E exceeds max capacity for shell N (2*N^2)
                max_capacity_n = 2 * (n**2)
                if electrons_in_n > max_capacity_n:
                    raise ValueError(
                        f"Electron count {electrons_in_n} for n={n} exceeds maximum capacity ({max_capacity_n})."
                    )
                # Store N*E part (even if E=0 for permutation logic)
                parsed_parts.append((n, electrons_in_n))
            except ValueError as e:
                if "N must be >= 1" in str(e) or "exceeds maximum capacity" in str(e):
                    raise  # Re-raise the specific value error
                else:  # Assume it was an int conversion error
                    raise ValueError(f"Invalid numbers in compact part '{part}': {e}")

        if not parsed_parts:  # Handle case where input was like "."
            return [] if generate_permutations else cls()

        # --- Default Logic (generate_permutations=False) ---
        if not generate_permutations:
            config = cls()
            for n, electrons_in_n in parsed_parts:
                if electrons_in_n == 0:
                    continue  # Skip N*0 parts
                electrons_remaining_in_n = electrons_in_n
                for l_quantum in range(n):  # l goes from 0 to n-1
                    if electrons_remaining_in_n <= 0:
                        break
                    max_occ_l = get_max_shell_occupation(l_quantum)
                    occupation = min(max_occ_l, electrons_remaining_in_n)
                    if occupation > 0:
                        config.add_shell(
                            Shell(n=n, l_quantum=l_quantum, occupation=occupation)
                        )
                        electrons_remaining_in_n -= occupation
                if electrons_remaining_in_n > 0:
                    raise RuntimeError(
                        f"Internal error: Electrons remaining ({electrons_remaining_in_n}) for n={n} after default filling."
                    )
            return config

        # --- Permutation Logic (generate_permutations=True) ---
        else:
            all_part_permutations: List[List[Configuration]] = []
            for n, electrons_in_n in parsed_parts:
                # Generate permutations for this single N*E part
                single_part_configs = cls._generate_subshell_permutations(
                    n, electrons_in_n
                )
                # _generate_subshell_permutations returns [Configuration()] for E=0
                # Check for errors if E>0 but no configs generated
                if not single_part_configs and electrons_in_n > 0:
                    raise RuntimeError(
                        f"Internal error: No permutations generated for n={n}, E={electrons_in_n}"
                    )
                all_part_permutations.append(single_part_configs)

            # Combine permutations from different N*E parts using Cartesian product
            final_configurations_set: Set[Configuration] = set()
            if not all_part_permutations:
                return []  # Should not happen if parsed_parts was not empty

            # Use itertools.product to get all combinations of configurations from each part
            # Example: [[1s2], [2s1, 2p1]] -> (1s2, 2s1), (1s2, 2p1)
            for config_combination in itertools.product(*all_part_permutations):
                # Combine the shells from each configuration in the combination
                combined_config = Configuration()
                for config_part in config_combination:
                    # Add shells from this part to the combined config
                    # Using add_shell with combine=True handles potential overlaps if N values were repeated (unlikely but safe)
                    for (
                        shell
                    ) in config_part:  # Iterate through shells of the partial config
                        combined_config.add_shell(shell, combine_occupation=True)
                final_configurations_set.add(combined_config)

            return sorted(
                list(final_configurations_set), key=lambda c: str(c)
            )  # Sort final list for consistency

    @classmethod
    def _generate_subshell_permutations(
        cls, n: int, total_electrons: int
    ) -> List["Configuration"]:
        """
        Internal helper to generate all valid distributions of electrons within
        subshells (l=0 to n-1) for a given principal shell n.

        Args:
            n: Principal quantum number.
            total_electrons: Total electrons to distribute in this shell.

        Returns:
            A list of Configuration objects, each representing one valid
            distribution for the given n. Returns a list containing one empty
            Configuration if total_electrons is 0.
        """
        # Base case: No electrons to distribute for this n
        if total_electrons == 0:
            return [cls()]  # Represent N*0 as a config with no shells for that N

        subshells_info = []  # Store tuples of (l_quantum, max_occupation)
        # Iterate through possible l values for the given n (l = 0 to n-1)
        for l_quantum in range(n):
            max_occ = get_max_shell_occupation(l_quantum)
            # Only consider subshells that can actually exist (l < n)
            if l_quantum < n:
                subshells_info.append((l_quantum, max_occ))

        if not subshells_info:  # Should only happen if n=0 (caught earlier) or n=1
            if n == 1 and total_electrons <= 2:  # Handle n=1 case explicitly
                return (
                    [Configuration([Shell(1, 0, total_electrons)])]
                    if total_electrons > 0
                    else [cls()]
                )
            else:  # Should not be reachable if input validation is correct
                return []

        valid_distributions = []  # Store lists of shells

        # Recursive helper function to find combinations
        def find_combinations(subshell_idx, electrons_left, current_shells_list):
            nonlocal valid_distributions
            # Base case: All subshells considered
            if subshell_idx == len(subshells_info):
                # If all electrons are distributed, this is a valid combination
                if electrons_left == 0:
                    valid_distributions.append(list(current_shells_list))  # Add copy
                return  # Stop recursion path

            # If somehow electrons_left < 0, prune this path
            if electrons_left < 0:
                return

            # Recursive step: Try all possible occupations for the current subshell
            l_s, max_occ_s = subshells_info[subshell_idx]

            # Iterate from max possible down to 0 for this subshell
            # Max possible is min(max_occ_s, electrons_left)
            for occ in range(min(max_occ_s, electrons_left), -1, -1):
                new_shells_list = list(current_shells_list)  # Create copy for this path
                if occ > 0:
                    # Add the shell for this occupation level
                    new_shells_list.append(Shell(n, l_s, occ))
                # Recurse for the next subshell
                find_combinations(
                    subshell_idx + 1, electrons_left - occ, new_shells_list
                )
                # No explicit backtrack needed as we pass copies

        find_combinations(0, total_electrons, [])
        # Convert lists of shells into Configuration objects
        return [Configuration(shells) for shells in valid_distributions]

    # --- Methods corresponding to original functions (Implementations required) ---

    def remove_filled_shells(self) -> "Configuration":
        """
        Creates a *new* Configuration containing only the partially filled shells
        from the original configuration.
        (Refactoring of `remove_max_occ`)

        Returns:
            A new Configuration object with only partially filled shells.
        """
        partial_shells = [
            shell for shell in self._shells.values() if not shell.is_full()
        ]
        return Configuration(partial_shells)

    def get_holes(self) -> Dict[str, int]:
        """
        Calculates the number of holes for each shell structure in the configuration.
        (Refactoring of `parse_holes`)

        Returns:
            A dictionary mapping the string representation of each shell structure
            (e.g., '3d', '2p-') to the number of holes in it. Only includes
            shell structures with holes > 0.
        """
        holes_dict = {}
        for shell in self._shells.values():
            num_holes = shell.holes()
            if num_holes > 0:
                # Create a key representing the shell structure (n, l, j) without occupation
                structure_shell = Shell(
                    shell.n, shell.l_quantum, 1, shell.j_quantum
                )  # Occ 1 for formatting
                structure_key_str = f"{structure_shell.n}{structure_shell.l_symbol}{structure_shell.j_symbol}"
                holes_dict[structure_key_str] = num_holes
        return holes_dict

    def compare(self, other: "Configuration") -> Dict[str, int]:
        """
        Compares the occupation numbers of structurally identical shells
        between this configuration and another.
        (Refactoring of `compare_shells_config`)

        Args:
            other: The other Configuration object to compare against.

        Returns:
            A dictionary mapping the string representation of shell structures
            (e.g., '3d', '2p+') to the absolute difference in their occupation
            numbers between the two configurations. Only includes shells where
            occupations differ or shells present in one config but not the other.

        Raises:
            TypeError: If 'other' is not a Configuration object.
        """
        if not isinstance(other, Configuration):
            raise TypeError(
                "Can only compare Configuration with another Configuration."
            )

        differences = {}
        # Get all unique shell structure keys from both configurations
        all_shell_keys: Set[Tuple[int, int, Optional[float]]] = set(
            self._shells.keys()
        ) | set(other._shells.keys())

        for key in all_shell_keys:
            # Get occupation, defaulting to 0 if shell structure doesn't exist
            # Create dummy shell only if needed, otherwise access directly
            occ_self = self._shells[key].occupation if key in self._shells else 0
            occ_other = other._shells[key].occupation if key in other._shells else 0

            if occ_self != occ_other:
                # Create a representative Shell object to get its string format without occupation
                # Use the structure key directly
                repr_shell = Shell(
                    key[0], key[1], 1, key[2]
                )  # Use occupation 1 for formatting
                shell_key_str = (
                    f"{repr_shell.n}{repr_shell.l_symbol}{repr_shell.j_symbol}"
                )

                differences[shell_key_str] = abs(occ_self - occ_other)

        return differences

    def split_core_valence(
        self, core_definition: Iterable[Union[str, Tuple[int, int, Optional[float]]]]
    ) -> Tuple["Configuration", "Configuration"]:
        """
        Splits the current configuration into core and valence configurations.

        The core contains shells whose structure matches the core_definition.
        The valence contains all other shells. Shells are copied, not modified.

        Args:
            core_definition: An iterable defining the core shell structures.
                Can contain shell strings (e.g., "1s", "2p", "3d-") or
                structure tuples (n, l_quantum, j_quantum). Occupation in
                any input Shell objects is ignored.

        Returns:
            A tuple containing two new Configuration objects: (core_config, valence_config).

        Raises:
            ValueError: If a string in core_definition is invalid.
            TypeError: If core_definition contains invalid types.
        """
        core_keys: Set[Tuple[int, int, Optional[float]]] = set()
        for item in core_definition:
            if isinstance(item, str):
                try:
                    # Add dummy occupation for parsing, handle potential l>=n error
                    shell = Shell.from_string(item + "1")
                    core_keys.add((shell.n, shell.l_quantum, shell.j_quantum))
                except ValueError as e:
                    # Re-raise with context if parsing fails
                    raise ValueError(
                        f"Invalid shell string '{item}' in core_definition: {e}"
                    )
            elif isinstance(item, tuple) and len(item) == 3:
                # Basic validation for tuple format
                n_core, l_core, j_core_in = item
                if (
                    not isinstance(n_core, int)
                    or not isinstance(l_core, int)
                    or (
                        j_core_in is not None
                        and not isinstance(j_core_in, (float, int))
                    )
                ):
                    raise TypeError(
                        f"Invalid structure tuple format in core_definition: {item}. Expected (int, int, float/int/None)."
                    )
                # Convert potential int j to float
                j_core = float(j_core_in) if isinstance(j_core_in, int) else j_core_in
                # Optional: Add validation for l_core < n_core here if desired for core defs
                # if l_core >= n_core: raise ValueError(...)
                # Optional: Add validation for j consistency here if desired for core defs
                # try: validate_j_quantum(l_core, j_core) except (ValueError, TypeError): raise TypeError(...)
                core_keys.add((n_core, l_core, j_core))
            else:
                raise TypeError(
                    f"core_definition must contain strings or (n, l, j) tuples, got {type(item)}"
                )

        core_shells = []
        valence_shells = []
        for key, shell in self._shells.items():
            # Create copies to ensure original config is unchanged
            shell_copy = Shell(
                shell.n, shell.l_quantum, shell.occupation, shell.j_quantum
            )
            if key in core_keys:
                core_shells.append(shell_copy)
            else:
                valence_shells.append(shell_copy)

        return Configuration(core_shells), Configuration(valence_shells)

    # --- Placeholder/TODO Methods for further refactoring ---

    # from_z is now implemented as from_element

    def _generate_holes_recursive(
        self,
        num_holes_left: int,
        start_shell_index: int,
        current_config: "Configuration",
    ) -> Set["Configuration"]:
        """Internal recursive helper for generating hole configurations."""
        # Base case: No more holes to create
        if num_holes_left == 0:
            # Use hash of string representation to ensure uniqueness based on content
            return {current_config.copy()}  # Return a set with a copy

        valid_next_configs = set()
        # Use a stable list of shells for consistent iteration order
        shells_list = sorted(
            list(current_config._shells.values())
        )  # Sort for determinism

        # Iterate through shells starting from start_shell_index to avoid reordering issues
        for i in range(start_shell_index, len(shells_list)):
            shell = shells_list[i]

            if shell.occupation > 0:
                # Create a new configuration with one electron removed from this shell
                next_config = current_config.copy()
                try:
                    removed_shell = shell.take_electron()
                    # Use add_shell which handles removal if occupation becomes 0
                    next_config.add_shell(
                        removed_shell, combine_occupation=False
                    )  # Replace
                except ValueError:
                    # Should not happen if occupation > 0, but safety check
                    continue  # Should log error?

                # Recursively find configurations with one less hole needed
                # Start recursion from the current shell index 'i' to allow multiple holes in the same shell structure
                results = self._generate_holes_recursive(
                    num_holes_left - 1, i, next_config
                )
                valid_next_configs.update(results)

        return valid_next_configs

    def generate_hole_configurations(self, num_holes: int = 1) -> List["Configuration"]:
        """
        Generates all possible unique configurations by creating the specified
        number of holes (removing electrons) in the current configuration.
        (Refactoring of `generate_hole_configurations` and `_generate_holes_recursive`)

        Args:
            num_holes: The number of electrons to remove (must be >= 1).

        Returns:
            A list of unique Configuration objects, sorted by string representation,
            each with `num_holes` fewer electrons distributed across the original shells.

        Raises:
            ValueError: If num_holes is less than 1 or greater than the total
                        number of electrons in the configuration.
        """
        if not isinstance(num_holes, int) or num_holes < 1:
            raise ValueError("Number of holes must be a positive integer.")

        total_electrons = self.total_electrons()
        if num_holes > total_electrons:
            raise ValueError(
                f"Cannot create {num_holes} holes in a configuration with only {total_electrons} electrons."
            )

        # Use the recursive helper
        config_set = self._generate_holes_recursive(num_holes, 0, self.copy())

        # Sort the results for consistent output
        return sorted(list(config_set), key=lambda c: str(c))

    def _generate_single_excitation_filtered(
        self,
        target_shell_structures: List[Tuple[int, int, Optional[float]]],
        allowed_source_keys: Optional[Set[Tuple[int, int, Optional[float]]]] = None,
    ) -> Set["Configuration"]:
        """
        Helper to generate single excitations, optionally filtering source shells.

        Args:
            target_shell_structures: List of target (n, l, j) tuples.
            allowed_source_keys: Set of allowed source (n, l, j) tuples, or None to allow all.

        Returns:
            A set of new Configuration objects representing single excitations.
        """
        excited_configs = set()
        occupied_source_shells = [s for s in self.shells if s.occupation > 0]

        for source_shell in occupied_source_shells:
            source_key = (
                source_shell.n,
                source_shell.l_quantum,
                source_shell.j_quantum,
            )
            # Skip if this source shell is not allowed
            if (
                allowed_source_keys is not None
                and source_key not in allowed_source_keys
            ):
                continue

            # Create config with one electron removed from source
            try:
                config_after_removal = self.copy()
                removed_shell = source_shell.take_electron()
                config_after_removal.add_shell(removed_shell)  # Removes if occ=0
            except ValueError:
                continue

            for target_key in target_shell_structures:
                # Skip excitation to the same shell structure
                if source_key == target_key:
                    continue

                n_t, l_t, j_t = target_key
                target_shell_in_config = config_after_removal.get_shell(n_t, l_t, j_t)
                current_occ_target = (
                    target_shell_in_config.occupation if target_shell_in_config else 0
                )

                # Check if target shell can accept an electron
                try:
                    max_occ_target = get_max_shell_occupation(l_t, j_t)
                    if current_occ_target < max_occ_target:
                        shell_to_add = Shell(n_t, l_t, 1, j_t)
                        final_config = config_after_removal.copy()
                        # Add electron to target shell (combine needed if target already exists)
                        final_config.add_shell(shell_to_add, combine_occupation=True)
                        # Check if it's different from the *original* config before adding
                        if final_config != self:
                            excited_configs.add(final_config)
                except ValueError:
                    continue  # Skip if target shell creation fails

        return excited_configs

    def generate_excitations(
        self,
        target_shells: List[Union[Shell, str]],
        excitation_level: int = 1,
        source_shells: Optional[
            Iterable[Union[str, Shell, Tuple[int, int, Optional[float]]]]
        ] = None,
    ) -> List["Configuration"]:
        """
        Generates excited configurations by moving electrons to target shells,
        optionally restricting which electrons can be moved.

        Args:
            target_shells: A list of possible target Shell structures to excite *to*.
                           Can be Shell objects or valid shell strings (e.g., "3d", "4p+").
                           Occupation of input shells is ignored, only structure matters.
            excitation_level: The number of electrons to excite (1 for single, 2 for double, etc.).
            source_shells: Optional. An iterable definin    g the shell structures from which
                           electrons are allowed to be excited. If None (default), any electron
                           can be excited. Accepts strings, Shells, or (n,l,j) tuples.

        Returns:
            A list of unique excited Configuration objects, sorted by string representation.
            These are the configurations first reached at the specified excitation_level,
            excluding configurations reached at lower levels and the original configuration.

        Raises:
            ValueError: If excitation_level is less than 1 or target/source shell string is invalid.
            TypeError: If target_shells/source_shells contains invalid types.
        """
        if not isinstance(excitation_level, int) or excitation_level < 1:
            raise ValueError("Excitation level must be a positive integer.")

        # --- Validate target shells upfront ---
        target_shell_structures_set: Set[Tuple[int, int, Optional[float]]] = set()
        for t_shell in target_shells:
            if isinstance(t_shell, Shell):
                target_shell_structures_set.add(
                    (t_shell.n, t_shell.l_quantum, t_shell.j_quantum)
                )
            elif isinstance(t_shell, str):
                try:
                    # Parse the string to validate it and get structure
                    # Add dummy occupation '1' for parsing structure strings
                    parsed_shell = Shell.from_string(t_shell + "1")
                    target_shell_structures_set.add(
                        (parsed_shell.n, parsed_shell.l_quantum, parsed_shell.j_quantum)
                    )
                except ValueError as e:
                    raise ValueError(f"Invalid target shell string '{t_shell}': {e}")
            else:
                raise TypeError(
                    f"target_shells must contain Shell objects or strings, got {type(t_shell)}"
                )
        # --- End Target Validation ---

        target_shell_structures = list(target_shell_structures_set)

        # --- Validate and process source_shells if provided ---
        allowed_source_keys: Optional[Set[Tuple[int, int, Optional[float]]]] = None
        if source_shells is not None:
            allowed_source_keys = set()
            for s_shell in source_shells:
                if isinstance(s_shell, Shell):
                    allowed_source_keys.add(
                        (s_shell.n, s_shell.l_quantum, s_shell.j_quantum)
                    )
                elif isinstance(s_shell, str):
                    try:
                        # Add dummy occupation '1' for parsing structure strings
                        parsed = Shell.from_string(s_shell + "1")
                        allowed_source_keys.add(
                            (parsed.n, parsed.l_quantum, parsed.j_quantum)
                        )
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid source shell string '{s_shell}': {e}"
                        )
                elif isinstance(s_shell, tuple) and len(s_shell) == 3:
                    # Basic validation for tuple format
                    n_core, l_core, j_core_in = s_shell
                    if (
                        not isinstance(n_core, int)
                        or not isinstance(l_core, int)
                        or (
                            j_core_in is not None
                            and not isinstance(j_core_in, (float, int))
                        )
                    ):
                        raise TypeError(
                            f"Invalid structure tuple format in source_shells: {s_shell}. Expected (int, int, float/int/None)."
                        )
                    j_core = (
                        float(j_core_in) if isinstance(j_core_in, int) else j_core_in
                    )
                    allowed_source_keys.add((n_core, l_core, j_core))
                else:
                    raise TypeError(
                        f"source_shells must contain Shells, strings, or (n,l,j) tuples, got {type(s_shell)}"
                    )
        # --- End Source Validation ---

        # --- Modified excitation loop using a helper ---
        # Start with a set containing only the initial configuration
        # Use the Configuration object itself as the key in the set, relies on __hash__ and __eq__
        # --- FIXED: Logic to return only states reached at exact level ---
        if excitation_level == 0:
            return []  # No excitations for level 0

        configs_at_prev_level = {
            self.copy()
        }  # Configs at the previous excitation level (starts with ground)
        all_configs_found = {
            self.copy()
        }  # Keep track of all configs found so far (including start)
        final_level_configs = set()  # Store configs first reached at the target level

        for current_level in range(1, excitation_level + 1):
            current_level_new_configs = set()
            if (
                not configs_at_prev_level
            ):  # Stop if previous level generated nothing new
                # This means no states exist at the target excitation level
                final_level_configs = set()  # Ensure empty set is returned
                break

            for config in configs_at_prev_level:
                # Generate single excitations from this config
                single_excitations = config._generate_single_excitation_filtered(
                    target_shell_structures, allowed_source_keys
                )
                # Add only those genuinely new (not found in *any* previous level)
                current_level_new_configs.update(single_excitations - all_configs_found)

            if (
                not current_level_new_configs
            ):  # Stop if no new unique configs generated at this level
                # Means no states exist *at* or *beyond* this level
                # If we are looking for exactly level N, ensure empty set is returned
                if current_level == excitation_level:
                    final_level_configs = set()
                break  # Stop the loop

            all_configs_found.update(current_level_new_configs)
            # The configs to excite from in the *next* iteration are the ones *just* found
            configs_at_prev_level = current_level_new_configs

            # If this is the target level, store the results
            if current_level == excitation_level:
                final_level_configs = current_level_new_configs

        # Return the configurations first reached at the target excitation level
        return sorted(list(final_level_configs), key=lambda c: str(c))
        # --- End Fix ---

    def generate_recombined_configurations(
        self,
        max_n: int,
        max_l: Optional[int] = None,
    ) -> List["Configuration"]:
        """
        Generates (N+1)-electron autoionizing/recombined configurations by adding
        one electron to this N-electron configuration.

        This method systematically adds an electron to all possible shells up to
        the specified quantum numbers, creating configurations useful for modeling
        dielectronic recombination and autoionization processes.

        Args:
            max_n: The maximum principal quantum number of the shell to add the
                   electron to.
            max_l: The maximum orbital angular momentum quantum number of the shell
                   to add the electron to. If None, defaults to max_n-1 (all allowed
                   l values for each n).

        Returns:
            A sorted list of unique Configuration objects, each with one additional
            electron compared to the original configuration.

        Raises:
            ValueError: If max_n is less than 1 or max_l is negative.

        Example:
            >>> config = Configuration.from_string("1s2.2s2")  # Be-like ion
            >>> recombined = config.generate_recombined_configurations(max_n=3, max_l=2)
            >>> # Returns configs like: 1s2.2s2.3s1, 1s2.2s2.3p1, 1s2.2s2.3d1
        """
        if not isinstance(max_n, int) or max_n < 1:
            raise ValueError(f"max_n must be an integer >= 1, got {max_n}")

        if max_l is None:
            max_l = max_n - 1  # Allow all l values up to n-1 for each n
        elif not isinstance(max_l, int) or max_l < 0:
            raise ValueError(
                f"max_l must be a non-negative integer or None, got {max_l}"
            )

        # Create a list of all possible shells to add an electron to
        shells_to_add = []
        for n in range(1, max_n + 1):
            # l must be less than n, so use min(n, max_l + 1)
            for l_quantum in range(min(n, max_l + 1)):
                shells_to_add.append(Shell(n, l_quantum, 1))

        # Use a set to store unique configurations
        recombined_configs: Set[Configuration] = set()

        # Try adding an electron to each possible shell
        for shell_to_add in shells_to_add:
            try:
                # Create a copy of the current configuration
                new_config = self.copy()
                # Add the shell (combining if it already exists)
                new_config.add_shell(shell_to_add, combine_occupation=True)
                # Add to the set (automatically handles uniqueness)
                recombined_configs.add(new_config)
            except ValueError:
                # This happens if the shell is already full - skip it
                pass

        # Convert to sorted list for consistent output
        return sorted(list(recombined_configs), key=lambda c: str(c))

    @classmethod
    def generate_recombined_configurations_batch(
        cls,
        configurations: List["Configuration"],
        max_n: int,
        max_l: Optional[int] = None,
    ) -> List["Configuration"]:
        """
        Generates (N+1)-electron recombined configurations for a list of input
        configurations and merges them into a single unique list.

        This is a convenience method for batch processing multiple configurations.
        It calls generate_recombined_configurations() for each input configuration
        and returns all unique results combined.

        Args:
            configurations: A list of Configuration objects to process.
            max_n: The maximum principal quantum number of the shell to add the
                   electron to.
            max_l: The maximum orbital angular momentum quantum number of the shell
                   to add the electron to. If None, defaults to max_n-1 (all allowed
                   l values for each n).

        Returns:
            A sorted list of unique Configuration objects, containing all recombined
            configurations generated from all input configurations (duplicates removed).

        Raises:
            ValueError: If max_n is less than 1, max_l is negative, or configurations
                        list is empty.
            TypeError: If any element in configurations is not a Configuration object.

        Example:
            >>> # Generate Li-like configurations
            >>> li_like = Configuration.from_element("Li", ion_charge=0)
            >>> li_like_excited = li_like.generate_excitations(["3s", "3p", "3d"], 1)
            >>>
            >>> # Generate recombined configs for all Li-like states at once
            >>> all_li = [li_like] + li_like_excited
            >>> recombined = Configuration.generate_recombined_configurations_batch(
            ...     all_li, max_n=4, max_l=2
            ... )
            >>> # Returns all unique (N+1)-electron configs from all Li-like states
        """
        if not configurations:
            raise ValueError("configurations list cannot be empty")

        # Validate that all elements are Configuration objects
        for i, config in enumerate(configurations):
            if not isinstance(config, cls):
                raise TypeError(
                    f"Element at index {i} is not a Configuration object: {type(config)}"
                )

        # Use a set to automatically handle uniqueness across all inputs
        all_recombined: Set[Configuration] = set()

        # Generate recombined configs for each input configuration
        for config in configurations:
            recombined = config.generate_recombined_configurations(max_n, max_l)
            all_recombined.update(recombined)

        # Convert to sorted list for consistent output
        return sorted(list(all_recombined), key=lambda c: str(c))

    @classmethod
    def generate_doubly_excited_autoionizing(
        cls,
        configurations: List["Configuration"],
        max_n: int,
        max_l: Optional[int] = None,
        num_holes: int = 1,
    ) -> List["Configuration"]:
        """
        Generates doubly-excited autoionizing configurations from a list of base
        configurations by creating holes and then adding electrons.

        This creates (N+1)-electron configurations with core holes, useful for
        modeling autoionization and dielectronic recombination processes where
        the system is doubly excited.

        The process:
        1. Create (N-num_holes)-electron configurations by removing electrons
        2. Add (num_holes+1) electrons to create (N+1)-electron autoionizing configs

        For example, from 1s2.2p1 (3e) with num_holes=1:
        - Create hole: 1s1.2p1 (2e) or 1s2 (2e)
        - Add 2 electrons: 1s1.2s2.2p1 (4e), 1s1.2s1.2p2 (4e), etc.

        Args:
            configurations: A list of base Configuration objects (typically N electrons).
            max_n: The maximum principal quantum number for added electrons.
            max_l: The maximum orbital angular momentum for added electrons.
                   If None, defaults to max_n-1.
            num_holes: Number of electrons to remove before recombining.
                      Default is 1 (single core hole).

        Returns:
            A sorted list of unique (N+1)-electron Configuration objects with
            core holes, representing doubly-excited autoionizing states.

        Raises:
            ValueError: If parameters are invalid or configurations list is empty.
            TypeError: If any element in configurations is not a Configuration object.

        Example:
            >>> # Li-like configurations (3 electrons)
            >>> li_configs = [
            ...     Configuration.from_string("1s2.2p1"),
            ...     Configuration.from_string("1s2.3s1"),
            ... ]
            >>>
            >>> # Generate doubly-excited autoionizing (4 electrons with hole)
            >>> autoionizing = Configuration.generate_doubly_excited_autoionizing(
            ...     li_configs, max_n=4, max_l=3
            ... )
            >>> # Returns configs like: 1s1.2s2.2p1, 1s1.2s1.2p2, 1s1.2s2.3s1, etc.
        """
        if not configurations:
            raise ValueError("configurations list cannot be empty")

        if not isinstance(num_holes, int) or num_holes < 1:
            raise ValueError(f"num_holes must be a positive integer, got {num_holes}")

        # Validate that all elements are Configuration objects
        for i, config in enumerate(configurations):
            if not isinstance(config, cls):
                raise TypeError(
                    f"Element at index {i} is not a Configuration object: {type(config)}"
                )

        # Step 1: Generate hole configurations from all base configs
        hole_configs_set: Set[Configuration] = set()
        for config in configurations:
            try:
                holes = config.generate_hole_configurations(num_holes=num_holes)
                hole_configs_set.update(holes)
            except ValueError:
                # Skip if we can't create enough holes (not enough electrons)
                continue

        if not hole_configs_set:
            # No valid hole configurations could be generated
            return []

        hole_configs = list(hole_configs_set)

        # Step 2: Add (num_holes + 1) electrons to create autoionizing configs
        # We need to call generate_recombined_configurations multiple times
        # to add multiple electrons

        current_configs = hole_configs
        for _ in range(num_holes + 1):
            current_configs = cls.generate_recombined_configurations_batch(
                current_configs, max_n=max_n, max_l=max_l
            )
            if not current_configs:
                return []

        return current_configs

    @classmethod
    def configurations_to_string(
        cls,
        configurations: List["Configuration"],
        separator: str = ".",
        line_separator: str = "\n",
        numbered: bool = False,
        start_index: int = 1,
        list: bool = True,
    ) -> str:
        """
        Converts a list of Configuration objects to a formatted string.

        This is a convenience method for printing, saving, or exporting lists
        of configurations. Each configuration is formatted on a separate line.

        Args:
            configurations: A list of Configuration objects to convert.
            separator: The separator to use between shells within each configuration.
                      Default is "." for dot notation. Use " " for space notation,
                      "" for compact notation, etc.
            line_separator: The separator between configurations. Default is "\n"
                           (newline). Can use other separators like ", " or "; ".
            numbered: If True, prefix each configuration with a number.
                     Default is False.
            start_index: If numbered is True, start numbering from this value.
                        Default is 1.

        Returns:
            A formatted string containing all configurations.

        Raises:
            ValueError: If configurations list is empty.
            TypeError: If any element in configurations is not a Configuration object.

        Examples:
            >>> configs = [
            ...     Configuration.from_string("1s2"),
            ...     Configuration.from_string("1s2.2s1"),
            ...     Configuration.from_string("1s2.2p1"),
            ... ]
            >>>
            >>> # Simple list (default)
            >>> print(Configuration.configurations_to_string(configs))
            1s2
            1s2.2s1
            1s2.2p1
            >>>
            >>> # Numbered list
            >>> print(Configuration.configurations_to_string(configs, numbered=True))
            1. 1s2
            2. 1s2.2s1
            3. 1s2.2p1
            >>>
            >>> # Space-separated (for FAC), numbered
            >>> print(Configuration.configurations_to_string(
            ...     configs, separator=' ', numbered=True
            ... ))
            1. 1s2
            2. 1s2 2s1
            3. 1s2 2p1
            >>>
            >>> # Comma-separated on one line
            >>> print(Configuration.configurations_to_string(
            ...     configs, line_separator=", "
            ... ))
            1s2, 1s2.2s1, 1s2.2p1
        """
        if not configurations:
            raise ValueError("configurations list cannot be empty")

        # Validate that all elements are Configuration objects
        for i, config in enumerate(configurations):
            if not isinstance(config, cls):
                raise TypeError(
                    f"Element at index {i} is not a Configuration object: {type(config)}"
                )

        # Format each configuration
        lines = []
        for i, config in enumerate(configurations):
            config_str = config.to_string(separator=separator)
            if numbered:
                line = f"{start_index + i}. {config_str}"
            else:
                line = config_str
            lines.append(line)
        if list:
            return lines  # type: ignore

        # Join with the specified line separator
        return line_separator.join(lines)

    def calculate_xray_label(self, reference_config: "Configuration") -> List[str]:
        """
        Calculates the X-ray notation label(s) for this configuration relative
        to a reference configuration (typically the neutral ground state).

        Identifies holes (missing electrons) in the current configuration
        compared to the reference and maps them to X-ray labels (K, L1, M5, etc.).

        Args:
            reference_config: The reference Configuration object (e.g., ground state).

        Returns:
            A sorted list of X-ray labels corresponding to the holes found.
            Returns ["Ground"] if the configuration matches the reference.
            Returns ["Unknown/Excited"] if differences are not simple holes.
            Returns ["UnknownLabel"] for holes in shells not in the standard map.

        Raises:
            TypeError: If reference_config is not a Configuration object.
        """
        if not isinstance(reference_config, Configuration):
            raise TypeError("reference_config must be a Configuration object.")

        hole_shells = []  # Store Shell objects representing the holes

        # Iterate through shells in the reference config
        for ref_shell in reference_config:
            current_shell = self.get_shell(
                ref_shell.n, ref_shell.l_quantum, ref_shell.j_quantum
            )
            current_occ = current_shell.occupation if current_shell else 0
            ref_occ = ref_shell.occupation

            if current_occ < ref_occ:
                # Calculate number of holes in this shell structure
                num_holes = ref_occ - current_occ
                # Create a temporary shell object representing the hole structure
                # We store the *structure* of the hole, number of holes handled by count
                hole_shell_structure = Shell(
                    ref_shell.n, ref_shell.l_quantum, 1, ref_shell.j_quantum
                )  # Occ=1 for sorting/mapping
                # Add this structure 'num_holes' times to the list
                hole_shells.extend([hole_shell_structure] * num_holes)

        # Check for electrons in shells not present in the reference (e.g., excited states)
        has_extra_electrons = False
        for current_shell in self:
            ref_shell_check = reference_config.get_shell(
                current_shell.n, current_shell.l_quantum, current_shell.j_quantum
            )
            if ref_shell_check is None and current_shell.occupation > 0:
                has_extra_electrons = True
                break
            # Also check if occupation increased relative to reference (less common for holes)
            elif (
                ref_shell_check
                and current_shell.occupation > ref_shell_check.occupation
            ):
                has_extra_electrons = True
                break

        if not hole_shells:
            # Check if the configurations are actually identical
            if self == reference_config:
                return ["Ground"]
            else:
                # Configurations differ, but not by simple holes in the reference shells
                # Could be an excited state relative to reference, or different ion stage.
                return ["Unknown/Excited"]  # Indicate it's not a simple hole state

        # If there are holes AND extra electrons, it's complex
        if has_extra_electrons:
            return ["Unknown/Excited"]

        # Sort the hole shells based on standard shell sorting (energy, j)
        sorted_hole_shells = sorted(hole_shells)

        # Map sorted hole shells to X-ray labels
        xray_labels = []
        for hole in sorted_hole_shells:
            # Create the key for the SHELL_LABEL_MAP (e.g., "1s", "2p-")
            map_key = f"{hole.n}{hole.l_symbol}{hole.j_symbol}"
            label = SHELL_LABEL_MAP.get(map_key)
            if label:
                xray_labels.append(label)
            else:
                # Fallback if label not found (e.g., for high n/l)
                xray_labels.append(f"Hole[{map_key}]")  # Indicate unknown hole
                print(
                    f"Warning: Could not map hole shell {map_key} to standard X-ray label."
                )

        # --- FIXED: Return the list directly ---
        return xray_labels
        # --- End Fix ---

    # --- Methods requiring more context or potentially belonging elsewhere ---

    # check_dict_patterns seems too specific for a general Configuration class method.
    # It compares hole dictionaries. This logic might belong in analysis or comparison functions.

    # sum_electrons is replaced by self.total_electrons()
    # OrderConfig is handled by the sorted self.shells property.
    # TakeElectron/AddElectron are handled by Shell methods, used within config generation.
    # JoinPairs is handled by Configuration.__init__ / add_shell(combine=True).
    # FilterEqual is a general utility, maybe atomkit.utils.filter_equal?


if __name__ == "__main__":
    # Unit tests have been moved to tests/test_configuration.py
    pass
