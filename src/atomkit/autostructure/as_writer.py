"""
AUTOSTRUCTURE input file writer.

This module provides the ASWriter class for generating AUTOSTRUCTURE .dat files
with a Pythonic interface, including helper classes for common configurations
and high-level presets for typical calculation types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

# Import atomkit utilities
if TYPE_CHECKING:
    from atomkit import Configuration
    from atomkit.structure import Shell

# Runtime imports
try:
    from atomkit import Configuration
    from atomkit.constants import L_QUANTUM_MAP
    from atomkit.structure import Shell
except ImportError:
    # Fallback for development/testing
    Configuration = None  # type: ignore
    Shell = None  # type: ignore
    L_QUANTUM_MAP = {}  # type: ignore


# ============================================================================
#                           HELPER CLASSES
# ============================================================================


@dataclass
class CoreSpecification:
    """
    Helper class for specifying closed core shells.

    Makes it easy to specify cores like He, Ne, Ar, etc. without
    remembering orbital indices.

    Examples
    --------
    >>> core = CoreSpecification.helium_like()  # 1s2
    >>> core = CoreSpecification.neon_like()    # 1s2 2s2 2p6
    >>> core = CoreSpecification.from_orbitals(1, 3)  # 1s through 2p
    """

    kcor1: int
    kcor2: int

    @classmethod
    def helium_like(cls) -> CoreSpecification:
        """1s² core (He-like)."""
        return cls(kcor1=1, kcor2=1)

    @classmethod
    def neon_like(cls) -> CoreSpecification:
        """1s² 2s² 2p⁶ core (Ne-like)."""
        return cls(kcor1=1, kcor2=3)

    @classmethod
    def argon_like(cls) -> CoreSpecification:
        """1s² 2s² 2p⁶ 3s² 3p⁶ core (Ar-like)."""
        return cls(kcor1=1, kcor2=6)

    @classmethod
    def from_orbitals(cls, first: int, last: int) -> CoreSpecification:
        """
        Define core from first and last orbital indices.

        Parameters
        ----------
        first : int
            First orbital index (usually 1 for 1s)
        last : int
            Last orbital index in core
        """
        return cls(kcor1=first, kcor2=last)


@dataclass
class SymmetryRestriction:
    """
    Helper class for symmetry restrictions in calculations.

    Simplifies specifying which terms/levels to include in calculations.

    Examples
    --------
    >>> # Restrict to ground state only
    >>> sym = SymmetryRestriction.single_term(S=0, L=0, parity=1)

    >>> # Restrict to specific levels in IC coupling
    >>> sym = SymmetryRestriction.levels([(1, 0, 0, 1), (1, 0, 2, 1)])
    """

    nast: int | None = None
    nastj: int | None = None
    term_list: list[tuple[int, int, int]] | None = field(
        default=None
    )  # [(2S+1, L, π), ...]
    level_list: list[tuple[int, int, int, int]] | None = field(
        default=None
    )  # [(2S+1, L, 2J, π), ...]

    @classmethod
    def single_term(cls, S: int, L: int, parity: int) -> SymmetryRestriction:
        """
        Restrict to a single LS term.

        Parameters
        ----------
        S : int
            Total spin
        L : int
            Total orbital angular momentum
        parity : int
            Parity (1 for even, -1 for odd)
        """
        return cls(nast=1, term_list=[(2 * S + 1, L, parity)])

    @classmethod
    def terms(cls, terms: list[tuple[int, int, int]]) -> SymmetryRestriction:
        """
        Restrict to specific LS terms.

        Parameters
        ----------
        terms : list of (2S+1, L, π)
            List of terms as (multiplicity, L, parity)
        """
        return cls(nast=len(terms), term_list=terms)

    @classmethod
    def levels(cls, levels: list[tuple[int, int, int, int]]) -> SymmetryRestriction:
        """
        Restrict to specific fine-structure levels.

        Parameters
        ----------
        levels : list of (2S+1, L, 2J, π)
            List of levels as (multiplicity, L, 2J, parity)
        """
        return cls(nastj=len(levels), level_list=levels)


@dataclass
class EnergyShifts:
    """
    Helper class for energy shift specifications.

    Useful for correcting known deficiencies in calculated energies.

    Examples
    --------
    >>> shifts = EnergyShifts(ls_shift=0.5, ic_shift=0.3)
    """

    ls_shift: float = 0.0  # ISHFTLS value
    ic_shift: float = 0.0  # ISHFTIC value
    continuum_ls: float = 0.0  # ECORLS
    continuum_ic: float = 0.0  # ECORIC


@dataclass
class CollisionParams:
    """
    Helper class for collision calculation parameters.

    Simplifies setting up electron impact excitation calculations.

    Examples
    --------
    >>> # Basic collision setup
    >>> coll = CollisionParams(min_L=0, max_L=10, min_J=0, max_J=20)

    >>> # With exchange control
    >>> coll = CollisionParams(min_L=0, max_L=10, max_exchange_L=8)
    """

    min_L: int | None = None
    max_L: int | None = None
    min_S: int | None = None
    max_S: int | None = None
    min_J: int | None = None
    max_J: int | None = None
    max_exchange_L: int | None = None
    max_exchange_multipole: int | None = None
    include_orbit_orbit: bool = False
    include_fine_structure: bool = False
    max_J_fine_structure: int | None = None


@dataclass
class OptimizationParams:
    """
    Helper class for orbital optimization parameters.

    Simplifies variational calculations and orbital optimization.

    Examples
    --------
    >>> # Basic optimization
    >>> opt = OptimizationParams(include_lowest=10, n_lambdas=5)

    >>> # With specific weighting
    >>> opt = OptimizationParams(include_lowest=20, weighting='equal')
    """

    include_lowest: int = 0  # INCLUD
    n_lambdas: int = 0  # NLAM
    n_variational: int = 0  # NVAR
    weighting: str = "statistical"  # 'statistical', 'equal', or 'custom'
    orthogonalization: str | None = None  # 'YES', 'NO', 'LPS'
    fix_orbitals: int | None = None  # IFIX

    def get_iwght(self) -> int:
        """Convert weighting string to IWGHT value."""
        mapping = {"statistical": 1, "equal": 0, "custom": -1}
        return mapping.get(self.weighting, 1)


@dataclass
class RydbergSeries:
    """
    Helper class for Rydberg series specification in DR/RR calculations.

    Examples
    --------
    >>> # Basic Rydberg series
    >>> ryd = RydbergSeries(n_min=3, n_max=15, l_max=7)
    """

    n_min: int
    n_max: int
    l_min: int = 0
    l_max: int = 7
    use_internal_mesh: bool = True  # NMESH=-1 for production
    limit_radiative: int | None = None  # NRAD


class ASWriter:
    """
    Generate AUTOSTRUCTURE input files (.dat format) with a Pythonic interface.

    AUTOSTRUCTURE uses NAMELIST-based input (Fortran style) with free-format
    configuration specifications. This class automates the generation of
    properly formatted input files.

    Key Differences from FAC
    ------------------------
    - Non-relativistic by default (though relativistic modes exist)
    - NAMELIST format: &NAME VAR=value &END
    - Configurations specified as orbital occupation numbers
    - More explicit coupling scheme specification (LS, IC, CA)

    Parameters
    ----------
    filename : str
        Output .dat filename

    Examples
    --------
    Basic structure calculation:

    >>> from atomkit.autostructure import ASWriter
    >>> from atomkit import Configuration
    >>>
    >>> ground = Configuration.from_string("1s2.2s2.2p6")
    >>> excited = ground.generate_excitations(["3s", "3p"], excitation_level=1)
    >>>
    >>> with ASWriter("ne_structure.dat") as asw:
    ...     asw.write_header("Ne-like structure")
    ...     asw.add_salgeb(CUP='LS', RAD='E1', MXCONF=1+len(excited))
    ...     asw.configs_from_atomkit([ground] + excited, last_core_orbital='1s')
    ...     asw.add_sminim(NZION=10)
    """

    def __init__(self, filename: str):
        """
        Initialize AUTOSTRUCTURE writer.

        Parameters
        ----------
        filename : str
            Output filename (typically .dat extension)
        """
        self.filename = Path(filename)
        self.lines: list[str] = []
        self.orbitals: list[tuple[int, int]] = []  # (n, l) pairs
        self.configurations: list[list[int]] = []  # occupation numbers
        self._file_handle = None

    def __enter__(self) -> ASWriter:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - writes file automatically."""
        if exc_type is None:
            self.close()
        return False

    def write_header(self, comment: str = "") -> None:
        """
        Write mandatory A.S. header line.

        The first 4 characters MUST be 'A.S.' for AUTOSTRUCTURE to recognize
        the file format. The rest of the line is for comments.

        Parameters
        ----------
        comment : str, optional
            Descriptive comment for this calculation

        Notes
        -----
        This must be called first before any other methods.
        """
        if self.lines:
            raise ValueError("Header must be written first!")
        self.lines.append(f"A.S. {comment}")
        self.add_comment(
            f"Generated by atomkit on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def add_comment(self, text: str) -> None:
        """
        Add a comment line.

        Parameters
        ----------
        text : str
            Comment text (# will be prepended automatically)
        """
        self.lines.append(f"# {text}")

    def add_blank_line(self) -> None:
        """Add a blank line for readability."""
        self.lines.append("")

    def add_salgeb(
        self,
        MXCONF: int | None = None,
        MXVORB: int | None = None,
        MXCCF: int = 0,
        CUP: str = "LS",
        RAD: str = "  ",
        RUN: str = "  ",
        # Core specification
        KCOR1: int | None = None,
        KCOR2: int | None = None,
        KORB1: int | None = None,
        KORB2: int | None = None,
        # Collision/autoionization control
        AUGER: str | None = None,
        BORN: str | None = None,
        # Fine-structure interaction control
        KUTSS: int | None = None,
        KUTSO: int | None = None,
        KUTOO: int | None = None,
        # Orbital basis control
        BASIS: str | None = None,
        # Configuration handling
        KCUT: int | None = None,
        KCUTCC: int | None = None,
        KCUTI: int | None = None,
        # Symmetry restrictions
        NAST: int | None = None,
        NASTJ: int | None = None,
        NASTS: int | None = None,
        NASTP: int | None = None,
        NASTPJ: int | None = None,
        # CI expansion control
        ICFG: int | None = None,
        NXTRA: int | None = None,
        LXTRA: int | None = None,
        IFILL: int | None = None,
        # Direct excitation range control
        MINLT: int | None = None,
        MAXLT: int | None = None,
        MINST: int | None = None,
        MAXST: int | None = None,
        MINJT: int | None = None,
        MAXJT: int | None = None,
        # Collision exchange control
        MAXLX: int | None = None,
        MXLAMX: int | None = None,
        LRGLAM: int | None = None,
        # Collision fine-structure control
        KUTOOX: int | None = None,
        KUTSSX: int | None = None,
        MAXJFS: int | None = None,
        # Multipole radiation control
        KPOLE: int | None = None,
        KPOLM: int | None = None,
        # Metastable and target state control
        NMETA: int | None = None,
        NMETAJ: int | None = None,
        INAST: int | None = None,
        INASTJ: int | None = None,
        TARGET: int | None = None,
        # Very large-scale calculation optimizations
        MSTART: int | None = None,
        KUTDSK: int | None = None,
        KUTLS: int | None = None,
        **kwargs,
    ) -> None:
        """
        Add SALGEB namelist (algebra and configuration specification).

        Parameters
        ----------
        MXCONF : int, optional
            Number of N-electron configurations (will be auto-set if using
            configs_from_atomkit)
        MXVORB : int, optional
            Number of distinct valence orbitals (will be auto-set if using
            configs_from_atomkit)
        MXCCF : int, optional
            Number of (N+1)-electron bound configurations for autoionization.
            Default is 0 (no autoionization)
        CUP : str, optional
            Coupling scheme:
            - 'LS': LS-coupling (default, non-relativistic)
            - 'IC': Intermediate coupling (non-relativistic with fine structure)
            - 'CA': Configuration average
            - 'LSM'/'MVD': LS with mass-velocity and Darwin
            - 'ICM': IC with mass-velocity and Darwin
            - 'ICR': Kappa-averaged relativistic (IC)
        RAD : str, optional
            Radiation type:
            - '  ' or 'NO': No radiation (default)
            - 'E1' or 'YES': Electric dipole
            - 'E2' or 'M1': Add quadrupole and magnetic dipole
            - 'E3' or 'M2': Add octupole and magnetic quadrupole
            - 'ALL': All available radiation
        RUN : str, optional
            Calculation type:
            - '  ': Structure only (default)
            - 'PI': Photoionization
            - 'DR': Dielectronic recombination
            - 'RR': Radiative recombination
            - 'PE': Photoexcitation-autoionization
            - 'RE': Resonant excitation
            - 'DE': Direct electron impact excitation
        KCOR1 : int, optional
            First orbital index defining closed core (use with KCOR2).
            Example: KCOR1=1, KCOR2=1 for 1s core; KCOR1=1, KCOR2=3 for Ne-like core.
            Alternative to KORB1/KORB2. Defines core model potential for R-matrix.
        KCOR2 : int, optional
            Last orbital index defining closed core (use with KCOR1).
        KORB1 : int, optional
            Alternative to KCOR1 for closed shell specification.
        KORB2 : int, optional
            Alternative to KCOR2 for closed shell specification.
        AUGER : str, optional
            Control autoionization rate calculation:
            - '  ' or 'YES': Calculate autoionization when continuum present (default)
            - 'NO': Do not calculate autoionization rates
            Note: Automatically 'NO' when RUN='PI' or 'RR'
        BORN : str, optional
            Control Born collision strength calculation:
            - 'INF': Infinite energy limit Born collision strengths
            - 'YES': Finite energy Born collision strengths (Type-1 adf04)
            - 'NO': Do not calculate (default, except if RAD='ALL' then 'INF')
        KUTSS : int, optional
            Fine-structure in 2-body spin-spin interactions:
            - -1: No fine-structure (LS coupling only)
            - 0: Perturbative fine-structure correction (default)
            - 1: Include fully in interaction matrix
            - 2: Include via direct integration
            Note: For LS coupling (CUP='LS'), KUTSS is forced to -1
        KUTSO : int, optional
            Fine-structure in 2-body spin-orbit interactions:
            - Same options as KUTSS (default: 0)
            - Controls relativistic spin-orbit contributions
        KUTOO : int, optional
            Fine-structure in 2-body orbit-orbit interactions:
            - Same options as KUTSS (default: 0)
            - Important for heavy elements and high precision
        BASIS : str, optional
            Orbital basis optimization mode (3 characters):
            - '   ' (3 spaces): Each configuration has unique orbitals (default)
            - 'RLX': Relaxed orbitals - optimize for each configuration separately
            - 'SRLX': Simplified relaxed - partially relaxed basis
            Note: Use with KCOR specification for core-valence separation.
            RLX/SRLX reduce size of STO parameter space in R-matrix calculations.
        KCUT : int, optional
            Configuration cutoff control:
            - Positive: Include only first KCUT configurations (spectroscopic)
            - Negative: Include first |KCUT| configs + all correlation configs
            - 0 or None: Include all configurations (default)
            Spectroscopic configs are those with holes in valence only.
            Correlation configs have holes in core (important for CI).
        KCUTCC : int, optional
            Cutoff for (N+1)-electron bound configurations in autoionization:
            - Similar to KCUT but applies to MXCCF configurations
            - Used with RUN='AI' or when MXCCF > 0
            - Separates spectroscopic from correlation (N+1) configs
        KCUTI : int, optional
            Cutoff for continuum configurations:
            - Controls which target states can photoionize/autoionize
            - Positive: First KCUTI target states only
            - Used with RUN='PI', 'AI', or 'RR' calculations
        NAST : int, optional
            Number of allowed spectroscopic terms (LS coupling):
            - Restricts calculation to specific SLπ terms
            - Followed by NAST values of (2S+1, L, π) in data section
            - Use with CUP='LS' for term-specific calculations
            - Example: NAST=2 to restrict to ground and first excited term
        NASTJ : int, optional
            Number of allowed levels (IC coupling):
            - Restricts calculation to specific LSJπ levels
            - Followed by NASTJ values of (2S+1, L, 2J, π) in data section
            - Use with CUP='IC' for level-specific calculations
            - More restrictive than NAST (specifies J quantum number)
        NASTS : int, optional
            Number of allowed LS terms for continuum states:
            - Restricts which LS terms can couple to continuum
            - Use with RUN='PI', 'RR' for term-selective photoionization
            - Followed by NASTS values of (2S+1, L, π)
        NASTP : int, optional
            Number of allowed parent terms for autoionization:
            - Restricts parent ion terms in autoionization
            - Use with RUN='AI' or when MXCCF > 0
            - Followed by NASTP values of (2S+1, L, π)
        NASTPJ : int, optional
            Number of allowed parent levels for autoionization:
            - Similar to NASTP but for IC coupling (includes J)
            - Followed by NASTPJ values of (2S+1, L, 2J, π)
            - Most restrictive parent specification
        ICFG : int, optional
            Configuration generation mode:
            - 0: No automatic generation (default, use explicit configs)
            - 1: Generate single excitations from reference configs
            - 2: Generate single + double excitations
            - 3: Generate triple excitations
            Use with NXTRA to extend orbital basis for CI expansion.
        NXTRA : int, optional
            Number of extra orbitals to add for CI expansion:
            - Adds NXTRA orbitals beyond those in reference configs
            - Use with LXTRA to specify maximum l of extra orbitals
            - Essential for systematic CI convergence studies
            - Example: NXTRA=5 adds 5 more n-values for each l
        LXTRA : int, optional
            Maximum l quantum number for extra orbitals:
            - Limits angular momentum of orbitals added by NXTRA
            - Example: LXTRA=2 means add only s, p, d orbitals
            - Default: Same as highest l in reference configuration
            - Use to control size of CI expansion
        IFILL : int, optional
            Configuration filling control:
            - 0: Standard filling based on energy order (default)
            - 1: Fill configurations by l quantum number priority
            - 2: Alternative filling scheme for specific systems
            - Affects which configurations ICFG generates
        MINLT : int, optional
            Minimum target level index for direct excitation:
            - Restricts which target states can be excited from
            - Use with MAXLT to define a range of initial states
            - Useful for calculating specific transition subsets
            - Example: MINLT=1, MAXLT=5 calculates from first 5 levels
        MAXLT : int, optional
            Maximum target level index for direct excitation:
            - Use with MINLT to restrict initial state range
            - Helps reduce calculation size for large level schemes
        MINST : int, optional
            Minimum total spin multiplicity (2S+1) for collision calculations:
            - Use with RUN='DE' for direct electron impact excitation
            - Restricts collision algebra to specific spin channels
            - Example: MINST=1, MAXST=3 for singlet and triplet only
            - Not necessary to set (default: all multiplicities)
        MAXST : int, optional
            Maximum total spin multiplicity (2S+1) for collision calculations:
            - Use with MINST to restrict spin channel range
            - Reduces computational cost by excluding high-spin channels
        MINJT : int, optional
            Minimum 2J value for initial states:
            - Restricts calculations to specific J quantum numbers
            - Use with CUP='IC' for fine-structure selection
            - Example: MINJT=0, MAXJT=4 for J=0, 1/2, 1, 3/2, 2
        MAXJT : int, optional
            Maximum 2J value for initial states:
            - Use with MINJT for J quantum number range restriction
            - Note: 2J used (not J) so integers work for half-integer J
        MAXLX : int, optional
            Maximum total L for exchange in collision calculations:
            - Use with RUN='DE' for electron impact excitation
            - Exchange neglected for L > MAXLX
            - Default: twice max exchange multipole (2*MXLAMX)
            - You should not normally need to change this
        MXLAMX : int, optional
            Maximum exchange multipole for collisions:
            - Controls exchange interaction range in scattering
            - Default: 2*(max target orbital l) + 3
            - Negative values: neglect exchange completely
            - Smaller values: restrict direct scattering as well
        LRGLAM : int, optional
            Top-up lambda for collision calculations:
            - Automatically set to MAXLT or MAXJT
            - Can be reduced/switched-off for testing
            - Controls convergence of partial wave expansion
        KUTOOX : int, optional
            Collisional two-body orbit-orbit interaction:
            - -1: Off (default)
            - 1: Include in same fashion as target KUTOO
            - Use with RUN='DE' for high-precision collisions
            - Independent of target KUTOO setting
        KUTSSX : int, optional
            Collisional two-body fine-structure:
            - -1: Off (default)
            - -999: All possible fine-structure terms
            - 1: First target configuration only
            - Similar interpretation as target KUTSS
            - Time-consuming; recommended only for weak transitions
        MAXJFS : int, optional
            Maximum 2J for collisional fine-structure:
            - Limits expensive fine-structure calculations
            - Use with KUTSSX to control computational cost
            - Only important for weak transitions
        KPOLE : int, optional
            Maximum multipole order for radiation:
            - 1: Electric dipole (E1) only (default for RAD='E1')
            - 2: E1 + E2 (electric quadrupole)
            - 3: E1 + E2 + E3, etc.
            - Automatically set based on RAD parameter if not specified
            - Use for high-precision forbidden transition rates
        KPOLM : int, optional
            Include magnetic multipoles:
            - 0 or None: Electric multipoles only (default)
            - 1: Include M1 (magnetic dipole)
            - 2: Include M1 + M2 (magnetic quadrupole)
            - Important for forbidden transitions and fine-structure mixing
        NMETA : int, optional
            Number of metastable terms (LS coupling):
            - Specifies which terms are considered metastable
            - Followed by NMETA values of (2S+1, L, π) in data section
            - Use with CUP='LS' for metastable-state-specific calculations
            - Important for excitation from metastable states
        NMETAJ : int, optional
            Number of metastable levels (IC coupling):
            - Similar to NMETA but for fine-structure levels
            - Followed by NMETAJ values of (2S+1, L, 2J, π)
            - Use with CUP='IC' for J-specific metastable states
            - More precise than NMETA for heavy elements
        INAST : int, optional
            Number of initial terms (LS coupling):
            - Specifies which terms are initial states for transitions
            - Similar to NAST but explicitly for initial states
            - Followed by INAST values of (2S+1, L, π)
            - Use when initial and final state restrictions differ
        INASTJ : int, optional
            Number of initial levels (IC coupling):
            - Similar to INAST but for fine-structure levels
            - Followed by INASTJ values of (2S+1, L, 2J, π)
            - Use with CUP='IC' for explicit initial level selection
            - Allows different initial/final level restrictions
        TARGET : int, optional
            Target state index for collision calculations:
            - Specifies single target state for scattering calculations
            - Used in electron impact excitation/ionization
            - Value is the index of the target state in level list
            - Useful for calculating cross sections to specific final states
        MSTART : int, optional
            Starting configuration number for isoelectronic sequence restarts:
            - > 0: Start from configuration MSTART (restart facility)
            - = 0: Start from beginning (default)
            - Use for restarting long isoelectronic sequence calculations
            - Requires previous calculation output for configurations < MSTART
        KUTDSK : int, optional
            Disk storage control for vector coupling coefficients:
            - > 0: Store on SCRATCH disk for configurations > KUTDSK
            - <= 0: All on disk with small I/O buffer (default)
            - Default: Store all in memory
            - Essential for terabyte-scale calculations (thousands of configs)
            - Use when memory limitations prevent in-core storage
        KUTLS : int, optional
            Configuration mixing control for large calculations:
            - < 0: Restrict mixing to *within* each (nl) configuration
            - > 0: Allow mixing between first KUTLS configs, restrict rest
            - Default: No restriction (full LS mixing)
            - Negative: Speeds up by repartitioning Hamiltonian
            - Positive: Eliminates unphysical n-mixing in high Rydberg states
        **kwargs : dict
            Additional SALGEB parameters for advanced usage.
            See AUTOSTRUCTURE manual for complete list.

        Notes
        -----
        If using configs_from_atomkit(), MXCONF and MXVORB will be set
        automatically and don't need to be specified here.

        Core specification (KCOR1/KCOR2 or KORB1/KORB2) is important for:
        - Defining correlation in structure calculations
        - Core model potential for R-matrix calculations
        - Separating core from valence in large calculations

        Examples
        --------
        Basic structure calculation:
        >>> asw.add_salgeb(CUP='IC', RAD='E1')

        With 1s closed core (He-like core):
        >>> asw.add_salgeb(CUP='LS', RAD='E1', KCOR1=1, KCOR2=1)

        With Ne-like core (1s2.2s2.2p6):
        >>> asw.add_salgeb(CUP='IC', RAD='E1', KCOR1=1, KCOR2=3)

        Photoionization without autoionization:
        >>> asw.add_salgeb(RUN='PI', CUP='LS', AUGER='NO')

        With Born collision strengths:
        >>> asw.add_salgeb(CUP='IC', RAD='ALL', BORN='INF')
        """
        params: dict[str, int | str | float] = {}

        # Only add CUP if specified (not None)
        if CUP is not None:
            params["CUP"] = self._quote_value(CUP)

        # Only add RAD if specified (not None)
        if RAD is not None:
            params["RAD"] = self._quote_value(RAD)

        if RUN and RUN.strip():
            params["RUN"] = self._quote_value(RUN)

        if MXCONF is not None:
            params["MXCONF"] = MXCONF
        if MXVORB is not None:
            params["MXVORB"] = MXVORB
        if MXCCF != 0:
            params["MXCCF"] = MXCCF

        # Core specification
        if KCOR1 is not None:
            params["KCOR1"] = KCOR1
        if KCOR2 is not None:
            params["KCOR2"] = KCOR2
        if KORB1 is not None:
            params["KORB1"] = KORB1
        if KORB2 is not None:
            params["KORB2"] = KORB2

        # Collision/autoionization control
        if AUGER is not None:
            params["AUGER"] = self._quote_value(AUGER)
        if BORN is not None:
            params["BORN"] = self._quote_value(BORN)

        # Fine-structure interaction control
        if KUTSS is not None:
            params["KUTSS"] = KUTSS
        if KUTSO is not None:
            params["KUTSO"] = KUTSO
        if KUTOO is not None:
            params["KUTOO"] = KUTOO

        # Orbital basis control
        if BASIS is not None:
            params["BASIS"] = self._quote_value(BASIS)

        # Configuration handling
        if KCUT is not None:
            params["KCUT"] = KCUT
        if KCUTCC is not None:
            params["KCUTCC"] = KCUTCC
        if KCUTI is not None:
            params["KCUTI"] = KCUTI

        # Symmetry restrictions
        if NAST is not None:
            params["NAST"] = NAST
        if NASTJ is not None:
            params["NASTJ"] = NASTJ
        if NASTS is not None:
            params["NASTS"] = NASTS
        if NASTP is not None:
            params["NASTP"] = NASTP
        if NASTPJ is not None:
            params["NASTPJ"] = NASTPJ

        # CI expansion control
        if ICFG is not None:
            params["ICFG"] = ICFG
        if NXTRA is not None:
            params["NXTRA"] = NXTRA
        if LXTRA is not None:
            params["LXTRA"] = LXTRA
        if IFILL is not None:
            params["IFILL"] = IFILL

        # Direct excitation range control
        if MINLT is not None:
            params["MINLT"] = MINLT
        if MAXLT is not None:
            params["MAXLT"] = MAXLT
        if MINST is not None:
            params["MINST"] = MINST
        if MAXST is not None:
            params["MAXST"] = MAXST
        if MINJT is not None:
            params["MINJT"] = MINJT
        if MAXJT is not None:
            params["MAXJT"] = MAXJT

        # Collision exchange control
        if MAXLX is not None:
            params["MAXLX"] = MAXLX
        if MXLAMX is not None:
            params["MXLAMX"] = MXLAMX
        if LRGLAM is not None:
            params["LRGLAM"] = LRGLAM

        # Collision fine-structure control
        if KUTOOX is not None:
            params["KUTOOX"] = KUTOOX
        if KUTSSX is not None:
            params["KUTSSX"] = KUTSSX
        if MAXJFS is not None:
            params["MAXJFS"] = MAXJFS

        # Multipole radiation control
        if KPOLE is not None:
            params["KPOLE"] = KPOLE
        if KPOLM is not None:
            params["KPOLM"] = KPOLM

        # Metastable and target state control
        if NMETA is not None:
            params["NMETA"] = NMETA
        if NMETAJ is not None:
            params["NMETAJ"] = NMETAJ
        if INAST is not None:
            params["INAST"] = INAST
        if INASTJ is not None:
            params["INASTJ"] = INASTJ
        if TARGET is not None:
            params["TARGET"] = TARGET

        # Very large-scale calculation optimizations
        if MSTART is not None:
            params["MSTART"] = MSTART
        if KUTDSK is not None:
            params["KUTDSK"] = KUTDSK
        if KUTLS is not None:
            params["KUTLS"] = KUTLS

        # Add any additional parameters
        for key, value in kwargs.items():
            if isinstance(value, str):
                params[key] = self._quote_value(value)
            else:
                params[key] = value

        self._write_namelist("SALGEB", params)

    def add_orbitals(self, orbitals: list[tuple[int, int]]) -> None:
        """
        Manually add orbital definitions (n, l pairs).

        Parameters
        ----------
        orbitals : list of (n, l) tuples
            Orbital quantum numbers
            Example: [(1, 0), (2, 0), (2, 1)] for 1s, 2s, 2p

        Notes
        -----
        Typically you should use configs_from_atomkit() which handles this
        automatically. This method is for advanced/manual usage.
        """
        self.orbitals = orbitals
        # Write orbital definitions: n1 l1  n2 l2  n3 l3 ...
        nl_line = "  ".join(f"{n} {l}" for n, l in orbitals)
        self.lines.append(nl_line)

    def add_configurations(self, configs: list[list[int]]) -> None:
        """
        Manually add configuration occupation numbers.

        Parameters
        ----------
        configs : list of lists
            Each inner list contains occupation numbers for the orbitals
            Example: [[2, 2, 0], [2, 1, 1], [2, 0, 2]] for 1s2.2s2, 1s2.2s1.2p1, 1s2.2p2

        Notes
        -----
        Typically you should use configs_from_atomkit() which handles this
        automatically. This method is for advanced/manual usage.
        """
        self.configurations = configs
        for occ_numbers in configs:
            occ_line = "  ".join(f"{occ:2d}" for occ in occ_numbers)
            self.lines.append(occ_line)

    def configs_from_atomkit(
        self,
        configurations: list[Any],  # list[Configuration]
        last_core_orbital: str | None = None,
        auto_detect_core: bool = True,
        optimize_from_orbital: str | None = None,
    ) -> dict:
        """
        Convert atomkit Configuration objects to AUTOSTRUCTURE format.

        This is the primary method for adding configurations - it automatically
        handles the conversion from Configuration objects to AS occupation numbers.

        Parameters
        ----------
        configurations : list of Configuration
            List of atomkit Configuration objects
        last_core_orbital : str, optional
            Last orbital in the closed core (e.g., '1s', '2p', '3p').
            If specified, core orbitals up to and including this one will be
            treated as closed shells via KCOR1/KCOR2.
            NOTE: If optimize_from_orbital is set, it overrides this for lambda optimization.
        auto_detect_core : bool, optional
            If True, automatically detect common core orbitals across all
            configurations. Default is True.
        optimize_from_orbital : str, optional
            If lambda optimization is planned, specify the first orbital to optimize from
            (e.g., '2s'). All orbitals from this one onwards will be written explicitly
            (not in core) so lambda parameters can act on them. Orbitals before this
            will be treated as core. This automatically handles the requirement that
            lambda parameters only act on explicitly listed orbitals.

        Returns
        -------
        dict
            Information about the conversion including:
            - 'n_orbitals': Number of distinct orbitals
            - 'n_configs': Number of configurations
            - 'core_orbitals': List of core orbital labels
            - 'valence_orbitals': List of valence orbital labels

        Examples
        --------
        >>> # Normal calculation with core
        >>> ground = Configuration.from_string("1s2.2s2.2p6")
        >>> excited = ground.generate_excitations(["3s", "3p"], 1)
        >>> asw.configs_from_atomkit([ground] + excited, last_core_orbital='1s')

        >>> # Lambda optimization starting from 2s
        >>> asw.configs_from_atomkit([ground] + excited, optimize_from_orbital='2s')
        >>> # This will write: 2s, 2p, 3s, 3p (all explicit) and put 1s in core

        Notes
        -----
        This method will update MXCONF and MXVORB automatically, so you don't
        need to specify them in add_salgeb() if you call this first.

        For lambda optimization, use optimize_from_orbital instead of last_core_orbital
        to ensure orbitals you want to optimize are explicitly written.
        """
        if not configurations:
            raise ValueError("Must provide at least one configuration")

        # Collect all unique orbitals across all configurations
        all_orbitals = set()
        for config in configurations:
            for shell in config.shells:
                all_orbitals.add((shell.n, shell.l_quantum))

        # Sort orbitals by (n, l)
        sorted_orbitals = sorted(all_orbitals)
        self.orbitals = sorted_orbitals

        # Detect core if requested
        core_orbitals = []

        # Priority: optimize_from_orbital > last_core_orbital > auto_detect
        if optimize_from_orbital:
            # Lambda optimization mode: orbitals BEFORE optimize_from_orbital go to core
            # Orbitals FROM optimize_from_orbital onwards are written explicitly
            if Shell is None:
                raise ImportError(
                    "Shell class not available - cannot parse optimize_from_orbital"
                )
            opt_shell = Shell.from_string(optimize_from_orbital + "1")
            opt_n, opt_l = opt_shell.n, opt_shell.l_quantum

            # Core = orbitals BEFORE the optimization starting point
            core_orbitals = [
                (n, l) for n, l in sorted_orbitals if (n, l) < (opt_n, opt_l)
            ]

        elif auto_detect_core or last_core_orbital:
            if last_core_orbital:
                # User specified core - use Shell class
                if Shell is None:
                    raise ImportError(
                        "Shell class not available - cannot parse core orbital"
                    )
                core_shell = Shell.from_string(
                    last_core_orbital + "1"
                )  # Add dummy occupation
                core_n, core_l = core_shell.n, core_shell.l_quantum
                core_orbitals = [
                    (n, l) for n, l in sorted_orbitals if (n, l) <= (core_n, core_l)
                ]
            else:
                # Auto-detect: find orbitals with same occupation in all configs
                orbital_occs = {orb: [] for orb in sorted_orbitals}
                for config in configurations:
                    for orb in sorted_orbitals:
                        n, l = orb
                        occ = sum(
                            shell.occupation
                            for shell in config.shells
                            if shell.n == n and shell.l_quantum == l
                        )
                        orbital_occs[orb].append(occ)

                # Core orbitals have constant, non-zero occupation
                for orb, occs in orbital_occs.items():
                    if len(set(occs)) == 1 and occs[0] > 0:
                        core_orbitals.append(orb)

        # Valence orbitals are non-core orbitals
        valence_orbitals = [orb for orb in sorted_orbitals if orb not in core_orbitals]

        # Write orbital definitions (valence only if core is separated)
        orbitals_to_write = valence_orbitals if core_orbitals else sorted_orbitals
        nl_line = "  ".join(f"{n} {l}" for n, l in orbitals_to_write)
        self.lines.append(nl_line)

        # Convert each configuration to occupation numbers
        config_occs = []
        for config in configurations:
            occ_numbers = []
            for n, l in orbitals_to_write:
                occ = sum(
                    shell.occupation
                    for shell in config.shells
                    if shell.n == n and shell.l_quantum == l
                )
                occ_numbers.append(occ)
            config_occs.append(occ_numbers)

        # Write configuration occupation numbers
        for occ_numbers in config_occs:
            occ_line = "  ".join(f"{occ:2d}" for occ in occ_numbers)
            self.lines.append(occ_line)

        self.configurations = config_occs

        # Update SALGEB parameters in the most recent namelist if it exists
        # Remove any existing MXCONF/MXVORB and add new values
        for i in range(len(self.lines) - 1, -1, -1):
            if "&SALGEB" in self.lines[i]:
                # Remove existing MX parameters if present
                import re

                line = self.lines[i]
                line = re.sub(r"\s*MXCONF=\d+", "", line)
                line = re.sub(r"\s*MXVORB=\d+", "", line)
                # Add new values before &END
                line = line.replace("&END", "")
                line += f" MXCONF={len(configurations)} MXVORB={len(orbitals_to_write)} &END"
                self.lines[i] = line
                break

        return {
            "n_orbitals": len(orbitals_to_write),
            "n_configs": len(configurations),
            "core_orbitals": [self._orbital_label(n, l) for n, l in core_orbitals],
            "valence_orbitals": [
                self._orbital_label(n, l) for n, l in valence_orbitals
            ],
        }

    def add_sminim(
        self,
        NZION: int,
        INCLUD: int = 0,
        NLAM: int = 0,
        NVAR: int = 0,
        # Optimization control
        IWGHT: int = 1,
        ORTHOG: str | None = None,
        MCFMX: int = 0,
        NFIX: int | None = None,
        MGRP: int | None = None,
        NOCC: int = 0,
        IFIX: int | None = None,
        # Potential specification
        MEXPOT: int = 0,
        PPOT: str | None = None,
        # Output control
        PRINT: str = "FORM",
        RADOUT: str = "NO",
        MAXE: float | None = None,
        # Energy shifts
        ISHFTLS: int = 0,
        ISHFTIC: int = 0,
        # Relativistic options (for CUP='ICR')
        IREL: int = 1,
        INUKE: int | None = None,
        IBREIT: int = 0,
        QED: int = 0,
        IRTARD: int = 0,
        # Advanced bundling
        NMETAR: int | None = None,
        NMETARJ: int | None = None,
        NRSLMX: int = 10000,
        NMETAP: int | None = None,
        NMETAPJ: int | None = None,
        NDEN: int | None = None,
        **kwargs,
    ) -> None:
        """
        Add SMINIM namelist (radial potential and minimization).

        Parameters
        ----------
        NZION : int
            Nuclear charge (atomic number)
            > 0: Thomas-Fermi-Dirac-Amaldi potential (non-relativistic)
            < 0: Hartree potential with Slater-Type-Orbitals
        INCLUD : int, optional
            Variational minimization:
            = 0: No minimization (default)
            > 0: Include lowest INCLUD terms in energy functional
            < 0: Read specific terms and weights
        NLAM : int, optional
            Number of scaling parameters (lambdas) for optimization.
            Default is 0 (all lambdas = 1.0)
        NVAR : int, optional
            Number of variational parameters to optimize.
            Default is 0 (no optimization)

        Optimization Control
        --------------------
        IWGHT : int, optional
            Weighting scheme for variational minimization:
            = 1: Weight by 2J+1 (default)
            = 0: Equal weights
            = -1: Use user-specified weights
        ORTHOG : str, optional
            Orthogonalization method:
            'YES': Schmidt orthogonalization
            'NO': No orthogonalization
            'LPS': Löwdin-Pauncz-Schwartz orthogonalization
            None: Auto-select (default)
        MCFMX : int, optional
            Configuration index for statistical TFD potential.
            = 0: Use average (default)
            > 0: Use specific configuration
        NFIX : int, optional
            Number of tied scaling parameters.
            Allows multiple orbitals to share the same lambda.
        MGRP : int, optional
            Number of orbital epsilon groups for grouping orbitals
            with similar energies in optimization.
        NOCC : int, optional
            Number of user-defined occupation numbers.
            = 0: Use default occupations (default)
            > 0: Read NOCC occupation specifications
        IFIX : int, optional
            Fix certain orbitals in self-consistent calculation:
            = 0: Optimize all orbitals
            > 0: Fix first IFIX orbitals from STO/previous calc

        Potential Specification
        -----------------------
        MEXPOT : int, optional
            Exchange potential:
            = 0: Hartree only (default)
            = 1: Hartree + local exchange approximation
        PPOT : str, optional
            Plasma potential specification:
            'SCCA': Self-consistent continuum approximation
            'FAC': Flexible atomic code potential
            None: No plasma effects (default)
            Or specify 'ION' for ion-sphere model

        Output Control
        --------------
        PRINT : str, optional
            Output format:
            'FORM': Formatted detailed output (default)
            'UNFORM': Unformatted compact output
        RADOUT : str, optional
            Radial function output for R-matrix:
            'YES': Write radial functions
            'NO': Don't write (default)
        MAXE : float, optional
            Maximum scattering energy in Rydbergs.
            Used for photoionization/collision calculations.

        Energy Shifts
        -------------
        ISHFTLS : int, optional
            Energy shifts in LS coupling:
            = 0: No shifts (default)
            > 0: Read ISHFTLS term energy shifts
        ISHFTIC : int, optional
            Energy shifts in IC coupling:
            = 0: No shifts (default)
            > 0: Read ISHFTIC level energy shifts

        Relativistic Options (for CUP='ICR')
        ------------------------------------
        IREL : int, optional
            Relativistic treatment:
            = 1: Large component only (default)
            = 2: Large + small components
        INUKE : int, optional
            Nuclear charge distribution:
            = -1: Point nucleus
            = 0: Uniform sphere
            = 1: Fermi distribution
            None: Auto-select based on Z (default)
        IBREIT : int, optional
            Breit interaction:
            = 0: Standard Breit (default)
            = 1: Generalized Breit
        QED : int, optional
            QED corrections:
            = 0: No QED (default)
            = 1: Vacuum polarization + self-energy
            = -1: Include additional QED terms
        IRTARD : int, optional
            Retardation effects:
            = 0: No retardation (default)
            = 1: Full retardation

        Advanced Data Bundling (for large calculations)
        -----------------------------------------------
        NMETAR : int, optional
            Electron-target bundling resolution.
            Groups (N+1)-electron terms for memory efficiency.
        NMETARJ : int, optional
            Electron-target level bundling resolution.
            Groups (N+1)-electron levels for memory efficiency.
        NRSLMX : int, optional
            Radiative data bundling limit.
            = 10000: Default maximum
        NMETAP : int, optional
            Photon-target bundling resolution.
            Groups target terms for photoionization.
        NMETAPJ : int, optional
            Photon-target level bundling resolution.
            Groups target levels for photoionization.
        NDEN : int, optional
            Number of plasma density/temperature pairs.
            Used with PPOT for plasma calculations.

        **kwargs : dict
            Additional SMINIM parameters not explicitly listed.

        Examples
        --------
        >>> # Basic structure calculation
        >>> asw.add_sminim(NZION=6)

        >>> # Optimized calculation with lambda parameters
        >>> asw.add_sminim(NZION=26, INCLUD=6, NLAM=3, NVAR=2, IWGHT=1)

        >>> # Relativistic calculation with QED
        >>> asw.add_sminim(NZION=92, IREL=2, QED=1, INUKE=1)

        >>> # Large DR calculation with bundling
        >>> asw.add_sminim(NZION=26, NMETAR=2, NRSLMX=50000)

        >>> # Plasma calculation
        >>> asw.add_sminim(NZION=26, PPOT='ION', NDEN=5)
        """
        params: dict[str, int | str | float] = {"NZION": NZION}

        if INCLUD != 0:
            params["INCLUD"] = INCLUD
        if NLAM != 0:
            params["NLAM"] = NLAM
        if NVAR != 0:
            params["NVAR"] = NVAR

        # Optimization control
        if IWGHT != 1:
            params["IWGHT"] = IWGHT
        if ORTHOG is not None:
            params["ORTHOG"] = self._quote_value(ORTHOG)
        if MCFMX != 0:
            params["MCFMX"] = MCFMX
        if NFIX is not None:
            params["NFIX"] = NFIX
        if MGRP is not None:
            params["MGRP"] = MGRP
        if NOCC != 0:
            params["NOCC"] = NOCC
        if IFIX is not None:
            params["IFIX"] = IFIX

        # Potential specification
        if MEXPOT != 0:
            params["MEXPOT"] = MEXPOT
        if PPOT is not None:
            params["PPOT"] = self._quote_value(PPOT)

        # Output control
        if PRINT != "FORM":
            params["PRINT"] = self._quote_value(PRINT)
        if RADOUT != "NO":
            params["RADOUT"] = self._quote_value(RADOUT)
        if MAXE is not None:
            params["MAXE"] = MAXE

        # Energy shifts
        if ISHFTLS != 0:
            params["ISHFTLS"] = ISHFTLS
        if ISHFTIC != 0:
            params["ISHFTIC"] = ISHFTIC

        # Relativistic options
        if IREL != 1:
            params["IREL"] = IREL
        if INUKE is not None:
            params["INUKE"] = INUKE
        if IBREIT != 0:
            params["IBREIT"] = IBREIT
        if QED != 0:
            params["QED"] = QED
        if IRTARD != 0:
            params["IRTARD"] = IRTARD

        # Advanced bundling
        if NMETAR is not None:
            params["NMETAR"] = NMETAR
        if NMETARJ is not None:
            params["NMETARJ"] = NMETARJ
        if NRSLMX != 10000:
            params["NRSLMX"] = NRSLMX
        if NMETAP is not None:
            params["NMETAP"] = NMETAP
        if NMETAPJ is not None:
            params["NMETAPJ"] = NMETAPJ
        if NDEN is not None:
            params["NDEN"] = NDEN

        # Add any additional parameters
        for key, value in kwargs.items():
            if isinstance(value, str):
                params[key] = self._quote_value(value)
            else:
                params[key] = value

        self._write_namelist("SMINIM", params)

    def add_sradcon(
        self,
        MENG: int = 0,
        EMIN: float | None = None,
        EMAX: float | None = None,
        # Additional energy grids
        MENGI: int | None = None,
        NDE: int = 0,
        DEMIN: float | None = None,
        DEMAX: float | None = None,
        NIDX: int | None = None,
        # Energy corrections
        ECORLS: float = 0.0,
        ECORIC: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Add SRADCON namelist (continuum energy grid for autoionization/photoionization).

        Only needed for calculations with continuum electrons (autoionization,
        photoionization, recombination, etc.)

        Parameters
        ----------
        MENG : int, optional
            Number of interpolation energies:
            = 0: Auto-select (default)
            > 0: Read MENG energies after namelist
            < 0: Generate -MENG energies between EMIN and EMAX
        EMIN : float, optional
            Minimum continuum energy (Rydbergs)
        EMAX : float, optional
            Maximum continuum energy (Rydbergs)

        Additional Energy Grids
        -----------------------
        MENGI : int, optional
            Number of interpolation energies for intermediate calculations.
            Similar to MENG but for internal interpolation.
        NDE : int, optional
            Number of excitation energies:
            = 0: No excitation energies (default)
            > 0: Read NDE excitation energies
            < 0: Generate -NDE energies between DEMIN and DEMAX
        DEMIN : float, optional
            Minimum excitation energy (Rydbergs).
            Used when NDE < 0.
        DEMAX : float, optional
            Maximum excitation energy (Rydbergs).
            Used when NDE < 0.
        NIDX : int, optional
            Number of extra energies beyond EMAX.
            Useful for extending the energy grid for specific transitions.

        Energy Corrections
        ------------------
        ECORLS : float, optional
            Energy correction for target continuum in LS coupling (Rydbergs).
            = 0.0: No correction (default)
            Adjusts the target continuum threshold.
        ECORIC : float, optional
            Energy correction for target continuum in IC coupling (Rydbergs).
            = 0.0: No correction (default)
            Adjusts the target continuum threshold.

        **kwargs : dict
            Additional SRADCON parameters not explicitly listed.

        Examples
        --------
        >>> # Standard photoionization grid
        >>> asw.add_sradcon(MENG=-15, EMIN=0.0, EMAX=25.0)

        >>> # With excitation energies
        >>> asw.add_sradcon(MENG=-15, EMIN=0.0, EMAX=100.0, NDE=-10, DEMIN=0.0, DEMAX=50.0)

        >>> # With energy correction
        >>> asw.add_sradcon(MENG=-20, EMIN=0.0, EMAX=150.0, ECORIC=0.5)

        >>> # Auto-select energies
        >>> asw.add_sradcon()
        """
        params = {}

        if MENG != 0:
            params["MENG"] = MENG
        if EMIN is not None:
            params["EMIN"] = EMIN
        if EMAX is not None:
            params["EMAX"] = EMAX

        # Additional energy grids
        if MENGI is not None:
            params["MENGI"] = MENGI
        if NDE != 0:
            params["NDE"] = NDE
        if DEMIN is not None:
            params["DEMIN"] = DEMIN
        if DEMAX is not None:
            params["DEMAX"] = DEMAX
        if NIDX is not None:
            params["NIDX"] = NIDX

        # Energy corrections
        if ECORLS != 0.0:
            params["ECORLS"] = ECORLS
        if ECORIC != 0.0:
            params["ECORIC"] = ECORIC

        # Add any additional parameters
        for key, value in kwargs.items():
            if isinstance(value, str):
                params[key] = self._quote_value(value)
            else:
                params[key] = value

        self._write_namelist("SRADCON", params)

    def add_drr(
        self,
        NMIN: int,
        NMAX: int,
        LMIN: int = 0,
        LMAX: int = 7,
        NMESH: int | None = None,
        # Radiation control
        NRAD: int | None = None,
        # Continuum specification
        LCON: int | None = None,
        **kwargs,
    ) -> None:
        """
        Add DRR namelist (Rydberg series for DR/RR calculations).

        Only needed when RUN='DR', 'RR', 'PE', or 'RE' in SALGEB.

        Parameters
        ----------
        NMIN : int
            Minimum principal quantum number for Rydberg series
        NMAX : int
            Maximum principal quantum number for Rydberg series
        LMIN : int, optional
            Minimum angular momentum. Default is 0.
        LMAX : int, optional
            Maximum angular momentum. Default is 7.
        NMESH : int, optional
            Number of additional n-values for interpolation:
            = 0: None
            > 0: Read NMESH values after namelist
            < 0: Use internal n-mesh (RECOMMENDED for production)
            = None: Don't specify (use AS default)

        Radiation Control
        -----------------
        NRAD : int, optional
            Principal quantum number above which no new radiative rates
            are calculated. Default is 1000 in AUTOSTRUCTURE.
            Useful for limiting computational cost in large calculations.
            Example: NRAD=100 means rates only computed up to n=100

        Continuum Specification
        -----------------------
        LCON : int, optional
            Number of continuum l-values to include in calculation.
            Allows control over angular momentum channels in continuum.

        **kwargs : dict
            Additional DRR parameters not explicitly listed.

        Examples
        --------
        >>> # Standard DR calculation
        >>> asw.add_drr(NMIN=3, NMAX=15, LMIN=0, LMAX=7)

        >>> # With internal n-mesh for production
        >>> asw.add_drr(NMIN=3, NMAX=10, LMIN=0, LMAX=5, NMESH=-1)

        >>> # Limit radiative rates for efficiency
        >>> asw.add_drr(NMIN=3, NMAX=20, LMAX=7, NRAD=100)

        >>> # Control continuum channels
        >>> asw.add_drr(NMIN=3, NMAX=15, LMAX=7, LCON=8)
        """
        params: dict[str, int | str | float] = {
            "NMIN": NMIN,
            "NMAX": NMAX,
            "LMIN": LMIN,
            "LMAX": LMAX,
        }

        if NMESH is not None:
            params["NMESH"] = NMESH

        # Radiation control
        if NRAD is not None:
            params["NRAD"] = NRAD

        # Continuum specification
        if LCON is not None:
            params["LCON"] = LCON

        # Add any additional parameters
        for key, value in kwargs.items():
            if isinstance(value, str):
                params[key] = self._quote_value(value)
            else:
                params[key] = value

        self._write_namelist("DRR", params)

    def add_sradwin(self, KEY: int = -9, **kwargs) -> None:
        """
        Add SRADWIN namelist (external orbital specification).

        This namelist is used to specify external orbitals read from files,
        typically for calculations using Opacity/Iron Project orbitals or
        Slater-Type-Orbitals.

        Parameters
        ----------
        KEY : int, optional
            Format specifier:
            = -9: Opacity/Iron/RmaX/APAP Project format (default)
            = -10: Free-formatted STO/Clementi orbitals from UNIT5
            Other values: See AUTOSTRUCTURE manual
        **kwargs : dict
            Additional SRADWIN parameters.
            May include specifications for which orbitals to read.

        Examples
        --------
        >>> # Read external orbitals in APAP format
        >>> asw.add_sradwin(KEY=-9)

        >>> # Read STO orbitals
        >>> asw.add_sradwin(KEY=-10)

        Notes
        -----
        External orbital files must be provided separately and specified
        via UNIT assignments in the AUTOSTRUCTURE execution environment.
        This is an advanced feature - most calculations use internally
        generated orbitals (no SRADWIN needed).
        """
        params: dict[str, int | str | float] = {"KEY": KEY}

        # Add any additional parameters
        for key, value in kwargs.items():
            if isinstance(value, str):
                params[key] = self._quote_value(value)
            else:
                params[key] = value

        self._write_namelist("SRADWIN", params)

    def add_icfg_occupation(self, min_occ: list[int], max_occ: list[int]) -> None:
        """
        Add minimum and maximum occupation arrays for ICFG automatic configuration generation.

        When ICFG is specified in SALGEB, the configurations are not listed explicitly.
        Instead, min/max occupation arrays define the allowed range for each orbital.

        Parameters
        ----------
        min_occ : list of int
            Minimum occupation numbers for each orbital
        max_occ : list of int
            Maximum occupation numbers for each orbital

        Examples
        --------
        >>> # Allow 0-2 electrons in first 14 orbitals
        >>> asw.add_icfg_occupation([0]*14, [2]*14)

        >>> # For Be-like: core 1s2, valence 0-2 in remaining orbitals
        >>> asw.add_icfg_occupation([2,0,0,0], [2,2,2,2])

        Notes
        -----
        This must be called after add_salgeb() if ICFG is specified.
        The arrays define the search space for automatic configuration generation.
        """
        # Format arrays compactly using * notation when possible
        min_str = self._format_array_compact(min_occ)
        max_str = self._format_array_compact(max_occ)
        self.lines.append(f" {min_str}")
        self.lines.append(f" {max_str}")

    def add_icfg_base_config(self, base_config: list[int], n_promotions: int) -> None:
        """
        Add base configuration and number of promotions for ICFG.

        Parameters
        ----------
        base_config : list of int
            Base configuration occupation numbers
        n_promotions : int
            Number of electron promotions from base

        Examples
        --------
        >>> # Single excitations from 1s2
        >>> asw.add_icfg_base_config([2,0,0,0], 1)

        >>> # Double excitations from 1s2 2s2
        >>> asw.add_icfg_base_config([2,2,0,0], 2)

        Notes
        -----
        Multiple base configurations can be specified by calling this method
        multiple times. Each defines a separate excitation series.
        """
        cfg_str = " ".join(str(x) for x in base_config)
        self.lines.append(f" {cfg_str}  {n_promotions}")

    def _format_array_compact(self, arr: list[int]) -> str:
        """
        Format an integer array using compact notation (e.g., 14*0 instead of 0 0 0...).

        Parameters
        ----------
        arr : list of int
            Array to format

        Returns
        -------
        str
            Compact string representation

        Examples
        --------
        >>> self._format_array_compact([0,0,0,0])
        '4*0'
        >>> self._format_array_compact([2,0,0,1,1])
        '2 3*0 2*1'
        """
        if not arr:
            return ""

        result = []
        i = 0
        while i < len(arr):
            val = arr[i]
            count = 1
            # Count consecutive identical values
            while i + count < len(arr) and arr[i + count] == val:
                count += 1

            # Use compact notation if beneficial (3+ repetitions)
            if count >= 3:
                result.append(f"{count}*{val}")
            elif count == 2:
                result.append(f"2*{val}")
            else:
                result.append(str(val))

            i += count

        return " ".join(result)

    def add_plasma_density_data(self, density: float, temperature: float) -> None:
        """
        Add electron density and temperature data for plasma screening calculations.

        Parameters
        ----------
        density : float
            Electron density in cm^-3
        temperature : float
            Temperature in Rydbergs (positive) or Kelvin (negative)

        Examples
        --------
        >>> # 1e24 cm^-3 at -1e6 K
        >>> asw.add_plasma_density_data(1e24, -1e6)

        Notes
        -----
        This should be called after add_sminim() when PPOT and NDEN are specified.
        Multiple density/temperature pairs can be added for calculations at different
        plasma conditions.
        """
        self.lines.append(
            f"{density:.6e} {temperature:.6e}       !NDEN electron density/temp pairs: N(cm/3) / T(>0 Ryd or <0 K)"
        )

    def get_content(self) -> str:
        """
        Get the generated AUTOSTRUCTURE input as a string.

        Returns
        -------
        str
            Complete file content

        Examples
        --------
        >>> content = asw.get_content()
        >>> print(content)
        """
        return "\n".join(self.lines)

    def close(self) -> None:
        """
        Write the file to disk.

        This is called automatically when using the context manager.
        """
        content = self.get_content()
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filename, "w") as f:
            f.write(content)
            f.write("\n")  # Ensure file ends with newline

    # Helper methods

    def _write_namelist(
        self, name: str, params: Mapping[str, int | str | float]
    ) -> None:
        """Write a NAMELIST block."""
        param_str = " ".join(f"{key}={value}" for key, value in params.items())
        self.lines.append(f" &{name} {param_str} &END")

    def _quote_value(self, value: str) -> str:
        """Add quotes to string values for NAMELIST."""
        if not value:
            return "'  '"
        if not value.startswith("'"):
            return f"'{value}'"
        return value

    def _orbital_label(self, n: int, l: int) -> str:
        """Convert (n, l) to spectroscopic notation using atomkit's definitions."""
        # Use atomkit's L_QUANTUM_MAP for consistency
        if l in L_QUANTUM_MAP:
            l_symbol = L_QUANTUM_MAP[l]
        else:
            # Handle high l values
            l_symbol = f"[l={l}]"
        return f"{n}{l_symbol}"

    # ========================================================================
    #                     HIGH-LEVEL PRESET METHODS
    # ========================================================================

    @classmethod
    def for_structure_calculation(
        cls,
        filename: str,
        nzion: int,
        coupling: str = "LS",
        radiation: str = "E1",
        core: CoreSpecification | None = None,
        optimization: OptimizationParams | None = None,
        comment: str = "",
    ) -> ASWriter:
        """
        Create an ASWriter configured for a basic structure calculation.

        This is the most common use case: calculate energy levels and
        radiative transition rates.

        Parameters
        ----------
        filename : str
            Output .dat filename
        nzion : int
            Nuclear charge (atomic number)
        coupling : str, optional
            Coupling scheme ('LS', 'IC', 'CA'). Default is 'LS'.
        radiation : str, optional
            Radiation type ('E1', 'E2', 'M1', 'ALL'). Default is 'E1'.
        core : CoreSpecification, optional
            Closed core specification
        optimization : OptimizationParams, optional
            Orbital optimization parameters
        comment : str, optional
            Description for header

        Returns
        -------
        ASWriter
            Pre-configured writer instance

        Examples
        --------
        >>> from atomkit.autostructure import ASWriter, CoreSpecification
        >>>
        >>> asw = ASWriter.for_structure_calculation(
        ...     "ne_structure.dat",
        ...     nzion=10,
        ...     coupling="IC",
        ...     core=CoreSpecification.helium_like()
        ... )
        >>> # Now add configurations and close
        """
        writer = cls(filename)
        writer.write_header(comment or f"Structure calculation for Z={nzion}")

        # Configure SALGEB
        salgeb_params: dict[str, Any] = {"CUP": coupling, "RAD": radiation}
        if core:
            salgeb_params["KCOR1"] = core.kcor1
            salgeb_params["KCOR2"] = core.kcor2
        writer.add_salgeb(**salgeb_params)  # type: ignore  # type: ignore

        # Will add configurations next (user's responsibility)
        # Then add SMINIM
        sminim_params: dict[str, Any] = {"NZION": nzion}
        if optimization:
            sminim_params["INCLUD"] = optimization.include_lowest
            sminim_params["NLAM"] = optimization.n_lambdas
            sminim_params["NVAR"] = optimization.n_variational
            sminim_params["IWGHT"] = optimization.get_iwght()
            if optimization.orthogonalization:
                sminim_params["ORTHOG"] = optimization.orthogonalization
            if optimization.fix_orbitals:
                sminim_params["IFIX"] = optimization.fix_orbitals

        writer._pending_sminim = sminim_params  # Store for later
        return writer

    @classmethod
    def for_photoionization(
        cls,
        filename: str,
        nzion: int,
        energy_min: float,
        energy_max: float,
        n_energies: int = 15,
        coupling: str = "IC",
        core: CoreSpecification | None = None,
        comment: str = "",
    ) -> ASWriter:
        """
        Create an ASWriter configured for photoionization calculations.

        Parameters
        ----------
        filename : str
            Output .dat filename
        nzion : int
            Nuclear charge
        energy_min : float
            Minimum photon energy (Rydbergs)
        energy_max : float
            Maximum photon energy (Rydbergs)
        n_energies : int, optional
            Number of energy points. Default is 15.
        coupling : str, optional
            Coupling scheme. Default is 'IC'.
        core : CoreSpecification, optional
            Closed core specification
        comment : str, optional
            Description for header

        Returns
        -------
        ASWriter
            Pre-configured writer instance

        Examples
        --------
        >>> asw = ASWriter.for_photoionization(
        ...     "ne_pi.dat",
        ...     nzion=10,
        ...     energy_min=0.0,
        ...     energy_max=50.0,
        ...     n_energies=20
        ... )
        """
        writer = cls(filename)
        writer.write_header(comment or f"Photoionization for Z={nzion}")

        # SALGEB with PI run type
        salgeb_params: dict[str, Any] = {"CUP": coupling, "RAD": "E1", "RUN": "PI"}
        if core:
            salgeb_params["KCOR1"] = core.kcor1
            salgeb_params["KCOR2"] = core.kcor2
        writer.add_salgeb(**salgeb_params)  # type: ignore

        # Store SMINIM and SRADCON parameters
        writer._pending_sminim = {"NZION": nzion}
        writer._pending_sradcon = {
            "MENG": -abs(n_energies),
            "EMIN": energy_min,
            "EMAX": energy_max,
        }
        return writer

    @classmethod
    def for_dielectronic_recombination(
        cls,
        filename: str,
        nzion: int,
        rydberg: RydbergSeries,
        energy_min: float = 0.0,
        energy_max: float = 100.0,
        coupling: str = "IC",
        core: CoreSpecification | None = None,
        comment: str = "",
    ) -> ASWriter:
        """
        Create an ASWriter configured for dielectronic recombination (DR).

        Parameters
        ----------
        filename : str
            Output .dat filename
        nzion : int
            Nuclear charge
        rydberg : RydbergSeries
            Rydberg series specification
        energy_min : float, optional
            Minimum energy (Rydbergs). Default is 0.
        energy_max : float, optional
            Maximum energy (Rydbergs). Default is 100.
        coupling : str, optional
            Coupling scheme. Default is 'IC'.
        core : CoreSpecification, optional
            Closed core specification
        comment : str, optional
            Description for header

        Returns
        -------
        ASWriter
            Pre-configured writer instance

        Examples
        --------
        >>> from atomkit.autostructure import RydbergSeries
        >>> ryd = RydbergSeries(n_min=3, n_max=15, l_max=7)
        >>> asw = ASWriter.for_dielectronic_recombination(
        ...     "ne_dr.dat",
        ...     nzion=10,
        ...     rydberg=ryd
        ... )
        """
        writer = cls(filename)
        writer.write_header(comment or f"Dielectronic recombination for Z={nzion}")

        # SALGEB with DR run type
        salgeb_params: dict[str, Any] = {"CUP": coupling, "RAD": "E1", "RUN": "DR"}
        if core:
            salgeb_params["KCOR1"] = core.kcor1
            salgeb_params["KCOR2"] = core.kcor2
        writer.add_salgeb(**salgeb_params)  # type: ignore

        # Store pending namelists
        writer._pending_sminim = {"NZION": nzion}
        writer._pending_sradcon = {"MENG": -15, "EMIN": energy_min, "EMAX": energy_max}
        writer._pending_drr = {
            "NMIN": rydberg.n_min,
            "NMAX": rydberg.n_max,
            "LMIN": rydberg.l_min,
            "LMAX": rydberg.l_max,
            "NMESH": -1 if rydberg.use_internal_mesh else 0,
        }
        if rydberg.limit_radiative:
            writer._pending_drr["NRAD"] = rydberg.limit_radiative

        return writer

    @classmethod
    def for_collision(
        cls,
        filename: str,
        nzion: int,
        collision: CollisionParams,
        coupling: str = "IC",
        core: CoreSpecification | None = None,
        comment: str = "",
    ) -> ASWriter:
        """
        Create an ASWriter configured for electron impact excitation.

        Parameters
        ----------
        filename : str
            Output .dat filename
        nzion : int
            Nuclear charge
        collision : CollisionParams
            Collision calculation parameters
        coupling : str, optional
            Coupling scheme. Default is 'IC'.
        core : CoreSpecification, optional
            Closed core specification
        comment : str, optional
            Description for header

        Returns
        -------
        ASWriter
            Pre-configured writer instance

        Examples
        --------
        >>> from atomkit.autostructure import CollisionParams
        >>> coll = CollisionParams(min_L=0, max_L=10, min_J=0, max_J=20)
        >>> asw = ASWriter.for_collision(
        ...     "ne_collision.dat",
        ...     nzion=10,
        ...     collision=coll
        ... )
        """
        writer = cls(filename)
        writer.write_header(comment or f"Electron impact excitation for Z={nzion}")

        # SALGEB with DE run type and collision parameters
        salgeb_params: dict[str, Any] = {"CUP": coupling, "RAD": "E1", "RUN": "DE"}
        if core:
            salgeb_params["KCOR1"] = core.kcor1
            salgeb_params["KCOR2"] = core.kcor2

        # Add collision-specific parameters
        if collision.min_L is not None:
            salgeb_params["MINLT"] = collision.min_L
        if collision.max_L is not None:
            salgeb_params["MAXLT"] = collision.max_L
        if collision.min_S is not None:
            salgeb_params["MINST"] = collision.min_S
        if collision.max_S is not None:
            salgeb_params["MAXST"] = collision.max_S
        if collision.min_J is not None:
            salgeb_params["MINJT"] = collision.min_J
        if collision.max_J is not None:
            salgeb_params["MAXJT"] = collision.max_J
        if collision.max_exchange_L is not None:
            salgeb_params["MAXLX"] = collision.max_exchange_L
        if collision.max_exchange_multipole is not None:
            salgeb_params["MXLAMX"] = collision.max_exchange_multipole
        if collision.include_orbit_orbit:
            salgeb_params["KUTOOX"] = 1
        if collision.include_fine_structure:
            salgeb_params["KUTSSX"] = 1
        if collision.max_J_fine_structure is not None:
            salgeb_params["MAXJFS"] = collision.max_J_fine_structure

        writer.add_salgeb(**salgeb_params)  # type: ignore
        writer._pending_sminim = {"NZION": nzion}
        return writer

    # ========================================================================
    #                     FLUENT INTERFACE (METHOD CHAINING)
    # ========================================================================

    def with_core(self, core: CoreSpecification) -> ASWriter:
        """
        Add core specification (fluent interface).

        This method allows method chaining for a more fluent API.

        Parameters
        ----------
        core : CoreSpecification
            Core specification

        Returns
        -------
        ASWriter
            self for method chaining

        Examples
        --------
        >>> asw = (ASWriter("file.dat")
        ...     .with_core(CoreSpecification.neon_like())
        ...     .with_optimization(OptimizationParams(include_lowest=10))
        ... )
        """
        # Update last SALGEB if it exists, or store for later
        if hasattr(self, "_pending_core"):
            self._pending_core = core
        else:
            self._pending_core = core
        return self

    def with_optimization(self, opt: OptimizationParams) -> ASWriter:
        """
        Add optimization parameters (fluent interface).

        Parameters
        ----------
        opt : OptimizationParams
            Optimization parameters

        Returns
        -------
        ASWriter
            self for method chaining
        """
        self._pending_optimization = opt
        return self

    def with_symmetry(self, sym: SymmetryRestriction) -> ASWriter:
        """
        Add symmetry restrictions (fluent interface).

        Parameters
        ----------
        sym : SymmetryRestriction
            Symmetry restriction

        Returns
        -------
        ASWriter
            self for method chaining
        """
        self._pending_symmetry = sym
        return self

    def with_energy_shifts(self, shifts: EnergyShifts) -> ASWriter:
        """
        Add energy shift corrections (fluent interface).

        Parameters
        ----------
        shifts : EnergyShifts
            Energy shift parameters

        Returns
        -------
        ASWriter
            self for method chaining
        """
        self._pending_shifts = shifts
        return self

    # ========================================================================
    #                     VALIDATION METHODS
    # ========================================================================

    def validate(self) -> list[str]:
        """
        Validate the current configuration and return warnings/errors.

        Checks for common mistakes and missing required parameters.

        Returns
        -------
        list of str
            List of warning/error messages. Empty if valid.

        Examples
        --------
        >>> asw = ASWriter("test.dat")
        >>> warnings = asw.validate()
        >>> if warnings:
        ...     for w in warnings:
        ...         print(f"Warning: {w}")
        """
        warnings = []

        # Check if header exists
        if not self.lines or not self.lines[0].startswith("A.S."):
            warnings.append("Missing header! Call write_header() first.")

        # Check if SALGEB exists
        if not any("&SALGEB" in line for line in self.lines):
            warnings.append("Missing SALGEB namelist! Call add_salgeb().")

        # Check if configurations exist
        if not self.configurations and not any(
            "&SALGEB" in line for line in self.lines
        ):
            warnings.append(
                "No configurations specified! Call add_configurations() or configs_from_atomkit()."
            )

        # Check if SMINIM exists
        if not any("&SMINIM" in line for line in self.lines):
            warnings.append("Missing SMINIM namelist! Call add_sminim().")

        # Check for RUN type specific requirements
        for line in self.lines:
            if "RUN='PI'" in line or 'RUN="PI"' in line:
                if not any("&SRADCON" in l for l in self.lines):
                    warnings.append(
                        "RUN='PI' requires SRADCON namelist for continuum energy grid."
                    )
            if "RUN='DR'" in line or 'RUN="DR"' in line:
                if not any("&DRR" in l for l in self.lines):
                    warnings.append(
                        "RUN='DR' requires DRR namelist for Rydberg series."
                    )
                if not any("&SRADCON" in l for l in self.lines):
                    warnings.append(
                        "RUN='DR' requires SRADCON namelist for continuum energy grid."
                    )
            if "RUN='RR'" in line or 'RUN="RR"' in line:
                if not any("&DRR" in l for l in self.lines):
                    warnings.append(
                        "RUN='RR' requires DRR namelist for Rydberg series."
                    )

        # Check for coupling scheme consistency
        for line in self.lines:
            if "CUP='LS'" in line or 'CUP="LS"' in line:
                if "NASTJ" in line or "NMETAJ" in line or "INASTJ" in line:
                    warnings.append(
                        "LS coupling doesn't use J quantum numbers. Use NAST/NMETA/INAST instead."
                    )

        return warnings

    def validate_and_raise(self) -> None:
        """
        Validate configuration and raise exception if errors found.

        Raises
        ------
        ValueError
            If validation errors are found

        Examples
        --------
        >>> asw = ASWriter("test.dat")
        >>> asw.write_header("Test")
        >>> asw.validate_and_raise()  # Raises ValueError with details
        """
        warnings = self.validate()
        if warnings:
            msg = "AUTOSTRUCTURE input validation failed:\n" + "\n".join(
                f"  - {w}" for w in warnings
            )
            raise ValueError(msg)

    def check_completeness(self) -> bool:
        """
        Check if the input file is complete and ready to write.

        Returns
        -------
        bool
            True if complete, False otherwise

        Examples
        --------
        >>> if asw.check_completeness():
        ...     asw.close()
        ... else:
        ...     print("Still missing required sections!")
        """
        required = ["A.S.", "&SALGEB", "&SMINIM"]
        return all(any(req in line for line in self.lines) for req in required)
