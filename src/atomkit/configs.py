import re
import numpy as np

l_numbers = list(range(21))
l_spec = [
    "s",
    "p",
    "d",
    "f",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "q",
    "r",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
ang_dict = dict(zip(l_spec, l_numbers))

def shelldict(shell):
    n, l, p, *_ = shell  # Using *_ catches all remaining unused variables
    label1 = chr(74 + int(n))
    return f'{label1}' if n == 1 else f'{label1}{2 *ang_dict[l] + (1 if p == "+" else 0)}'

def shells_config(atomic_number):
    electrons_left = int(atomic_number)
    shells = []
    
    for shell in range(1, int((atomic_number/2)**0.5) + 2):
        capacity = 2 * shell * shell
        electrons = min(electrons_left, capacity)
        if electrons <= 0:
            break
        shells.append(f"{shell}*{electrons}")
        electrons_left -= electrons
        
    return ".".join(shells)

def parse_holes(config: str) -> dict:
    """
    Parse electron holes from orbital configuration strings.
    Handles both regular subshell (e.g. '2p4', '4d10') and subsubshell (e.g. '2p+4(2)2') formats.
    
    Args:
        config: Orbital configuration string
    Returns:
        Dictionary mapping shells to number of holes
    """
    is_subsubshell = '(' in config
    holes = {}
    for subshell in config.split('.'):
        # Extract shell and electron count based on format
        if is_subsubshell:
            shell, count = re.match(r'(.*?)(\d+)(?:\(|$)', subshell).groups()
            count = int(count)
            l_char = next(c for c in reversed(shell[:-1]) if c.isalpha())
            l = ang_dict[l_char]
            max_e = 2*(l+1) if '+' in shell else 2*l
        else:
            # Find where the numbers end and letters begin
            i = 0
            while i < len(subshell) and subshell[i].isdigit():
                i += 1
            
            # Find where letters end and count begins
            j = i
            while j < len(subshell) and subshell[j].isalpha():
                j += 1
                
            # Now split into parts
            shell = subshell[:j]  # Everything up to where numbers start again
            count = int(subshell[j:]) if j < len(subshell) else 1  # All remaining digits
            
            l = ang_dict[shell[-1]]
            max_e = 2*(2*l + 1)
        
        holes[shell] = max_e - count
            
    return holes

def compare_shells_config(config1, config2):
    """
    Compare two shell configurations efficiently.
    Uses a single pass through each dictionary without creating intermediate sets.
    
    Args:
        config1, config2: Configuration dictionaries to compare
    Returns:
        Dictionary of differences where values differ
    """
    differences = {}
    
    # Process all keys in config1
    for shell, value1 in config1.items():        # O(len(config1))
        value2 = config2.get(shell, 0)           # O(1)
        if value1 != value2:
            differences[shell] = abs(value2 - value1)
    
    # Process remaining keys in config2 that weren't in config1
    for shell, value2 in config2.items():        # O(len(config2))
        if shell not in config1:                 # O(1) - dict lookup
            differences[shell] = abs(value2)      # These must be different since they're not in config1
    
    return differences

def parse_shells_config(config):
    electron_dict = {}
    for shell in config.split('.'):
        shell_num, electron_count = map(int, shell.split('*'))
        electron_dict[shell_num] = electron_count
    return electron_dict


def check_dict_patterns(dict1, dict2):
    """
    Check if two dictionaries match based on their key patterns.
    Simpler forms match with more complex forms:
    - Numbers match with any modifier: 3 matches with '3s' or '3d'
    - Base+modifier matches with same base+modifier+extras: '3d' matches with '3d+5'
    
    Example:
    {3: 1, 4: 1} matches with {'3s': 1, '4s': 1} or {'3d': 1, '4d': 1}
    {'3d': 1, '4d': 1} matches with {'3d+5': 1, '4d+5': 1}
    """
    
    def get_base_and_modifier(key):
        """
        Extract the numeric base and the modifier from a key.
        Returns (base_number, modifier)
        Example: 
        '3d+5' → ('3', 'd')
        '4s' → ('4', 's')
        3 → ('3', '')  # Note: now returns empty string instead of None
        """
        if isinstance(key, (int, float)):
            return str(key), ''
        
        key_str = str(key)
        base = ''
        modifier = ''
        
        # Get the numeric base
        i = 0
        while i < len(key_str) and key_str[i].isdigit():
            base += key_str[i]
            i += 1
            
        # Get the first non-numeric character (if any) as modifier
        if i < len(key_str):
            modifier = key_str[i]
            
        return base, modifier

    # if len(dict1) != len(dict2):
    #     return False

    # Convert all keys to their base patterns
    dict1_patterns = {k: get_base_and_modifier(k) for k in dict1.keys()}
    dict2_patterns = {k: get_base_and_modifier(k) for k in dict2.keys()}

    # Match each key from dict1 to dict2
    for key1, (base1, mod1) in dict1_patterns.items():
        # Look for a matching key in dict2
        found_match = False
        for key2, (base2, mod2) in dict2_patterns.items():
            # If bases match and either:
            # - dict1 key is numeric (mod1 is empty)
            # - both modifiers are the same
            if base1 == base2 and (mod1 == '' or mod1 == mod2):
                found_match = True
                break
        if not found_match:
            return False

    return True

def XRlabel(atomic_number, shell_conf, config, label): 
    ## Get shell hole
    conf1=parse_shells_config(shell_conf)
    ref_conf=parse_shells_config(shells_config(atomic_number))
    shell_holes=compare_shells_config(ref_conf, conf1)
    ## Get configuration holes
    config_holes=parse_holes(config)
    ## Get label holes
    label_holes=parse_holes(label)
    ## Check if they match
    t1=check_dict_patterns(shell_holes, config_holes)
    t2=check_dict_patterns(config_holes, label_holes)

    # if t1 is false --> it's double hole on an s subsshell in shell indicated in shell holes 
    # if t1 is true but all others are false --> it's a double hole in a - subshell in the subshell indicated in config_holes
    # if t1 and t2 are true --> follow label from label holes

    ##check if keys of label holes are 0 
    if all(value == 0 for value in shell_holes.values()):
        return 'No Holes'
    if not all(value == 0 for value in shell_holes.values()) and all(value == 0 for value in config_holes.values()):
        for key in shell_holes:
            hole_label={f'{key}s+0': shell_holes[key]}
        XRlabels=[' '.join([shelldict(key)]*hole_label[key]) for key in hole_label]
        return ' '.join(XRlabels)


    ## getting hole label
    if t1 and t2:
        hole_label=label_holes
    elif t1 and not t2:
        for key in config_holes:
            shell, ne = get_shell_parts(key)
            hole_label={f'{shell}-{ne}': config_holes[key]}
    elif not t1:
        for key in shell_holes:
            hole_label={f'{key}s+0': shell_holes[key]}
    else:
        return 'No Holes'

    XRlabels=[' '.join([shelldict(key)]*hole_label[key]) for key in hole_label]
    return ' '.join(XRlabels)
    
    


def sum_electrons(electron_config):
    """
    Calculates the total number of electrons in an atom based on its electron configuration.

    Args:
        electron_config (str): A string representation of the electron configuration of an atom.

    Returns:
        int: The total number of electrons in the atom.

    Example:
        >>> sum_electrons('1s2 2s2 2p6 3s2 3p6')
        36
    """

    # Split the electron configuration into subshells
    subshells = electron_config.split()

    # Extract the last number from each subshell
    numbers = [re.findall(r"\d+$", subshell)[0] for subshell in subshells]

    # Convert the numbers to integers and sum them
    total = sum(int(x) for x in numbers)

    return total

def get_shell_parts(config: str) -> tuple[str, int]:
    """
    Split a configuration into shell name and count.
    
    Args:
        config: Configuration string like '3d1' or '3d10'
    Returns:
        Tuple of (shell_name, count)
    
    Examples:
        >>> get_shell_parts('3d1')
        ('3d', 1)
        >>> get_shell_parts('3d10')
        ('3d', 10)
        >>> get_shell_parts('2p6')
        ('2p', 6)
    """
    # Find position where digits end
    i = len(config) - 1
    while i >= 0 and config[i].isdigit():
        i -= 1
    
    shell = config[:i + 1]
    count = int(config[i + 1:]) if i + 1 < len(config) else 1
    
    return shell, count


def get_ionstage(electron_config, electron_config_neutral):
    """
    Calculates the ion stage of an element based on its electron configuration.

    Args:
    - electron_config (str): The electron configuration of the ion.
    - electron_config_neutral (str): The electron configuration of the neutral atom.

    Returns:
    - ion_stage (int): The ion stage of the element.

    Example:
        >>> get_ionstage('1s2 2s2 2p6 3s2 3p6', '1s2 2s2 2p6 3s2 3p6')
        0
        >>> get_ionstage('1s2 2s2 2p4', '1s2 2s2 2p6')
        2
        >>> get_ionstage('1s2', '1s1')
        -1
    """
    # Get the number of electrons in the ion
    electrons_ion = sum_electrons(electron_config)

    # Get the number of electrons in the neutral atom
    electrons_neutral = sum_electrons(electron_config_neutral)

    # Calculate the ion stage
    ion_stage = electrons_neutral - electrons_ion

    return ion_stage


def CreateShells(nmin, nmax, l):
    """
    Creates an array with a list of shells with increasing n and fixed l.
    :param nmin: minimum value of n
    :param nmax: maximum value of n
    :param l: fixed value of l
    :return: list of shells in format "nl"
    """
    return ["{}{}".format(n, l) for n in range(nmin, nmax + 1)]


# def ReadShell(s):
#     """
#     Reads a shell and returns a tuple of (n, l, ne)
#     :param s: string representing a shell
#     :return: tuple of (n, l, ne)
#     """
#     if len(s) == 2:
#         s = s + "1"
#     res = re.split("(\d+)", s)
#     #print(s)
#     #print(res)
#     n = res[1]
#     l = res[2]
#     ne = int(res[3])

def get_xray_labels(conf1, label, configuration, l_spec=l_spec, ang_dict=ang_dict):
    xray_map = {
        ('1', 's'): 'K',
        ('2', 's'): 'L1',
        ('2', 'p', '-'): 'L2',
        ('2', 'p', '+'): 'L3',
        ('3', 's'): 'M1',
        ('3', 'p', '-'): 'M2',
        ('3', 'p', '+'): 'M3',
        ('3', 'd', '-'): 'M4',
        ('3', 'd', '+'): 'M5',
        ('4', 's'): 'N1',
        ('4', 'p', '-'): 'N2',
        ('4', 'p', '+'): 'N3',
        ('4', 'd', '-'): 'N4',
        ('4', 'd', '+'): 'N5'
    }
    
    holes = []
    
    # First check if configuration shows a full shell (like 4d10)
    if configuration.endswith('10') and '.' not in configuration:
        # Check conf1 for complete subshell holes
        shells = conf1.split('.')
        for shell in shells:
            n, count = shell.split('*')
            n = int(n)
            count = int(count)
            if n == 2 and count < 8:
                holes.extend(['L1'] * 2)
            elif n == 3 and count < 18:
                holes.extend(['M1'] * 2)
            elif n == 4 and count < 18:
                holes.extend(['N1'] * 2)
            elif n == 1 and count < 2:
                holes.extend(['K'] * (2 - count))
    
    else:
        # Process configurations with explicit holes
        configs = configuration.split('.')
        label_parts = label.split('.')

        for config in configs:
            if not config:
                continue
                
            n = config[0]
            l = config[1]
            
            # Get number of holes
            if len(config) > 2:
                count = int(config[2:])
                max_count = 2 * (2 * ang_dict[l] + 1) if l != 's' else 2
                num_holes = max_count - count
            else:
                num_holes = 1
                
            # For s shells
            if l == 's':
                key = (n, l)
                if key in xray_map:
                    holes.extend([xray_map[key]] * num_holes)
            
            # For p,d shells
            else:
                # Find corresponding label part
                matching_label = next((lab for lab in label_parts if lab.startswith(f"{n}{l}")), None)
                if matching_label:
                    parts = matching_label.split('.')
                    # If we have both + and - in label
                    if len(parts) > 1 and num_holes > 1:
                        if '-' in parts[0] and '+' in parts[1]:
                            holes.append(xray_map[(n, l, '-')])
                            holes.append(xray_map[(n, l, '+')])
                        elif '+' in parts[0] and '-' in parts[1]:
                            holes.append(xray_map[(n, l, '+')])
                            holes.append(xray_map[(n, l, '-')])
                    # Single type of hole
                    else:
                        if '-' in matching_label:
                            holes.extend([xray_map[(n, l, '-')]] * num_holes)
                        elif '+' in matching_label:
                            holes.extend([xray_map[(n, l, '+')]] * num_holes)
    
    return ' '.join(sorted(holes))


def ReadShell(s):
    """
    Reads a shell and returns a tuple of (n, l, ne)
    :param s: string representing a shell
    :return: tuple of (n, l, ne)
    """
    # Adjusted regex pattern to correctly match multiple digits for n and ne
    pattern = r"(\d+)([a-zA-Z]+)(\d*)"
    match = re.match(pattern, s)
    if not match:
        raise ValueError("Invalid shell format")

    n, l, ne = match.groups()
    # Convert n and ne to integers, handling cases where ne might be empty (defaults to 1)
    n = int(n)
    ne = int(ne) if ne else 1

    return (n, l, ne)


def OrderConfig(shells):
    """
    Orders a list of shells based on n and l values
    :param shells: list of shells
    :return: ordered list of shells
    """
    Shell_number = list(map(ReadShell, shells))
    # print(Shell_number)
    ordered = sorted(Shell_number, key=lambda tup: (tup[0], ang_dict[tup[1]]))
    shells = [f"{s[0]}{s[1]}{s[2]}" for s in ordered]
    return shells


def TakeElectron(s, clean=True):
    """
    Removes an electron from a shell and returns the new shell
    :param s: string representing a shell
    :return: string representing new shell
    """
    (n, l, ne) = ReadShell(s)
    if int(ne) < 0 and clean:
        s_new = ""
    else:
        s_new = f"{n}{l}{int(ne)-1}"
    return s_new


def AddShells(s1, s2):
    """
    Adds two shells together and returns the resulting shell
    :param s1: string representing first shell
    :param s2: string representing second shell
    :return: tuple of (True/False, resulting shell)
    """
    (n1, l1, ne1) = ReadShell(s1)
    (n2, l2, ne2) = ReadShell(s2)
    added = False
    shell = ""
    if n1 == n2 and l1 == l2:
        ne12 = ne1 + ne2
        added = True
        if ne12 <= 2 * (2 * ang_dict[l2] + 1):
            shell = f"{n1}{l1}{ne12}"
    return (added, shell)


def get_full_configuration(shell_numbers, holes_config):
    config = {}
    
    # Parse shell numbers
    shells = shell_numbers.split('.')
    for shell in shells:
        n, count = shell.split('*')
        count = int(count)
        remaining = count
        
        # Fill shells until remaining electrons are used up
        l = 0  # Start with first angular momentum
        while remaining > 0 and l < len(l_spec):
            l_sym = l_spec[l]
            shell_name = f"{n}{l_sym}"
            max_occ = 2 * (2 * l + 1)
            config[shell_name] = min(max_occ, remaining)
            remaining -= max_occ
            l += 1
    
    # Apply holes configuration
    if holes_config:
        holes = holes_config.split('.')
        for hole in holes:
            if not hole:
                continue
            shell = hole[:2]
            # Handle case where no number is given (assume 1)
            electrons = 1 if len(hole) == 2 else int(hole[2:])
            config[shell] = electrons
    
    # Energy ordering (n+l, n) Madelung rule
    def shell_key(shell):
        n = int(shell[0])
        l = ang_dict[shell[1]]
        return (n + l, n)
    
    # Create configuration string
    config_str = ' '.join(f"{shell}{config[shell]}" 
                         for shell in sorted(config.keys(), key=shell_key) 
                         if config[shell] > 0)
    
    return config_str


def JoinPairs(c):
    """
    Joins shells in a configuration together and returns the new configuration
    :param c: string representing configuration
    :return: string representing new configuration
    """
    Shells = c.split(" ")
    tShells = c.split(" ")
    to_remove = []
    to_add = []
    k = 0
    break_out_flag = False
    while len(tShells) > 1:
        s1 = tShells[0]
        for i, s2 in enumerate(tShells[1:]):
            (added, new_s) = AddShells(s1, s2)
            if added == True:
                Shells.pop(0 + i + 1)
                Shells.pop(0)
                Shells.insert(0, new_s)
                tShells.pop(i + 1)
                tShells.pop(0)
                tShells.insert(0, new_s)
                k = 0
                if new_s == "":
                    return ""
                break_out_flag = True
                break
        if not break_out_flag:
            tShells.pop(0)
            k += 1
            Shells.append(Shells.pop(0))
        break_out_flag = False
    return " ".join(OrderConfig(Shells))


def FilterEqual(a, b):
    """
    Finds the elements in list 'a' that are not in list 'b'
    :param a: list
    :param b: list
    :return: list of elements that are only in 'a'
    """
    return np.setdiff1d(a, np.intersect1d(a, b))


def remove_max_occ(configuration):
    shells = [ReadShell(x) for x in configuration.split(".")]
    max_occ = [max_occupancy(x[1]) for x in shells]
    shells_new = []
    for i in range(len(shells)):
        if shells[i][-1] != max_occ[i]:
            shells_new.append(shells[i])
    shells_new = [f"{s[0]}{s[1]}{s[2]}" for s in shells_new]
    shells_new = OrderConfig(shells_new)
    config_new = ".".join(shells_new)
    config_new = re.sub(r"(?<=[a-zA-Z])1(?![0-9])", "", config_new)
    return config_new


def ExcitationConfigs(configs, excitation):
    """
    Finds all possible configurations by adding excitation shells to existing configurations
    :param configs: list of configurations
    :param excitation: list of excitation shells
    :return: list of new configurations
    """
    new_configs = []
    for c in configs:
        shells = np.array(c.split(" "))
        for i, s in enumerate(shells):
            s_new = TakeElectron(s)
            for ls in excitation:
                rest = shells[np.arange(len(shells)) != i]
                rest = " ".join(rest).strip() if len(rest) > 0 else ""
                new_config = f"{s_new} {ls} {rest}".strip()
                if new_config != c:
                    new_configs.append(JoinPairs(new_config))
    new_configs = [x for x in new_configs if x]  # eliminate empty strings
    return FilterEqual(np.unique(new_configs), configs)


def Exciting(configs, excitation, type):
    """
    Finds all possible configurations with excitations.
    :param configs: list of current configurations
    :param excitation: list of possible excitations as strings "nl"
    :param type: type of excitations to perform (S, D or T)
    :return: a list with the new states as strings "nl nl nl ..."
    """
    match type:
        case "S":
            return ExcitationConfigs(configs, excitation)
        case "D":
            single = ExcitationConfigs(configs, excitation)
            double = FilterEqual(
                np.unique(ExcitationConfigs(single, excitation)), configs
            )
            return double
        case "T":
            single = ExcitationConfigs(configs, excitation)
            double = FilterEqual(
                np.unique(ExcitationConfigs(single, excitation)), configs
            )
            triple = FilterEqual(
                np.unique(ExcitationConfigs(double, excitation)), configs
            )
            return triple
        case "Q":
            single = ExcitationConfigs(configs, excitation)
            double = FilterEqual(
                np.unique(ExcitationConfigs(single, excitation)), configs
            )
            triple = FilterEqual(
                np.unique(ExcitationConfigs(double, excitation)), configs
            )
            quadruple = FilterEqual(
                np.unique(ExcitationConfigs(triple, excitation)), configs
            )
            return quadruple


def max_occupancy(orbital):
    if isinstance(orbital, str):
        orbitals = "spdfghiklmnoqrtuvwxyz"
        ## get position from orbital
        l = orbitals.index(orbital.lower())
    elif isinstance(orbital, int):
        l = orbital
    else:
        raise ValueError("Invalid input")
    return 2 * (2 * l + 1)


def parse_excitation_notation(notation):
    """Parse the SD{...} notation."""
    match = re.match(r"([SDT]+)\{(.*?)\}", notation)
    if not match:
        raise ValueError("Invalid excitation notation")
    exc_types, shells = match.groups()
    return exc_types, shells.split(",")


def get_excitation_shells(core, notation):
    """Get all shells for excitation based on core and notation."""
    _, max_shells = parse_excitation_notation(notation)
    all_shells = []
    core_shells = core.split()

    for shell in max_shells:
        n, l, ne = ReadShell(shell)

        # Get the minimum n for this l from the core configuration
        n_min_core = max(
            [ReadShell(s)[0] for s in core_shells if l == ReadShell(s)[1]], default=1
        )
        # Get the absolute minimum n allowed for this l
        n_min_allowed = get_min_n_for_l(l)

        # Use the larger of n_min_core and n_min_allowed
        n_min = max(n_min_core, n_min_allowed) + 1
        # print(n_min_core)

        # Add shells from n_min to n
        all_shells.extend([f"{i}{l}" for i in range(n_min, n + 1)])

    return sorted(list(set(all_shells)))  # Remove duplicates and sort


def get_min_n_for_l(l):
    """Get the minimum allowed principal quantum number for a given orbital angular momentum."""
    if isinstance(l, str):
        l = ang_dict.get(l.lower(), -1)
    else:
        raise ValueError("Invalid input for orbital angular momentum")

    return max(1, l + 1)





def _generate_holes_recursive(shells, remaining_holes, start_idx=0, memo=None):
    """
    Helper function to recursively generate hole configurations.
    Uses memoization to avoid recalculating same configurations.
    
    Args:
        shells (list): List of shell strings
        remaining_holes (int): Number of holes still to create
        start_idx (int): Index to start considering shells from
        memo (dict): Memoization dictionary
    
    Returns:
        set: Set of possible configurations
    """
    if memo is None:
        memo = {}
    
    # Create key for memoization
    key = (tuple(shells), remaining_holes, start_idx)
    if key in memo:
        return memo[key]
    
    # Base case: no more holes to create
    if remaining_holes == 0:
        return {".".join(shells)}
    
    configurations = set()
    
    # Try removing electrons from each valid shell
    for i in range(start_idx, len(shells)):
        if not shells[i].endswith('0') or len(shells[i]) > 3:
            new_shells = shells.copy()
            new_shells[i] = TakeElectron(shells[i], clean=False)
            # print(new_shells[i])
            # print(new_shells)

            # Recursively generate configurations:
            # 1. Can take another electron from same shell (i)
            # 2. Can take electron from next shells (i+1)
            same_shell_configs = _generate_holes_recursive(
                new_shells, remaining_holes - 1, i, memo
            )
            configurations.update(same_shell_configs)
    
    memo[key] = configurations
    return configurations

def generate_hole_configurations(config, num_holes=1):
    """
    Generate electron configurations with specified number of holes.
    Keeps explicit zero electron states for clarity.
    
    Args:
        config (str): Electron configuration (e.g., '1s2.2s2.2p6')
        num_holes (int): Number of holes to generate
    
    Returns:
        list: Sorted list of possible electron configurations with holes
    """
    # Input validation
    if num_holes <= 0:
        return [config]
        
    # Split into individual shells and use helper function
    shells = config.split(' ')
    configurations = _generate_holes_recursive(shells, num_holes)
    
    return sorted(list(configurations))
import unittest


class TestModule(unittest.TestCase):

    def test_CreateShells(self):
        self.assertEqual(CreateShells(1, 3, "s"), ["1s", "2s", "3s"])
        self.assertEqual(CreateShells(2, 4, "p"), ["2p", "3p", "4p"])

    def test_ReadShell(self):
        self.assertEqual(ReadShell("1s"), ("1", "s", 1))
        self.assertEqual(ReadShell("2p"), ("2", "p", 1))

    def test_OrderConfig(self):
        self.assertEqual(OrderConfig(["2s1", "3d1"]), ["2s1", "3d1"])
        self.assertEqual(OrderConfig(["3d1", "2s1"]), ["2s1", "3d1"])

    def test_TakeElectron(self):
        self.assertEqual(TakeElectron("1s2"), "1s1")
        self.assertEqual(TakeElectron("2p1"), "")

    def test_AddShells(self):
        self.assertEqual(AddShells("1s1", "1s1"), (True, "1s2"))
        self.assertEqual(AddShells("2p0", "3d5"), (False, ""))

    def test_JoinPairs(self):
        self.assertEqual(JoinPairs("1s2 2p5 3d1 3d3"), "1s2 2p5 3d4")

    def test_FilterEqual(self):
        a = [1, 2, 3]
        b = [2]
        c = [4]

        self.assertTrue((FilterEqual(a, b) == np.array([1, 3])).all())
        self.assertTrue((FilterEqual(a, c) == np.array([1, 2, 3])).all())

    def test_ExcitationConfigs(self):
        configs = ["1s2", "2s2"]
        excitation = ["3d", "4f"]

        self.assertEqual(
            set(ExcitationConfigs(configs, excitation)),
            set(["1s1 3d1", "1s1 4f1", "2s1 3d1", "2s1 4f1"]),
        )

    def test_Exciting(self):
        configs = ["1s2", "2s2"]
        excitation = ["3d", "4f"]

        self.assertCountEqual(
            set(Exciting(configs, excitation, "S")),
            set(["1s1 3d1", "1s1 4f1", "2s1 3d1", "2s1 4f1"]),
        )

        self.assertCountEqual(
            Exciting(configs, excitation, "D"), ["3d2", "4f2", "3d1 4f1"]
        )


if __name__ == "__main__":
    unittest.main()
