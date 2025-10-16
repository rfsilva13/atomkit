"""
Unified atomic calculation interface.

Express calculations in physical terms, automatically translate to backend-specific.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass, field

from .specs import (
    CouplingType,
    RelativisticType,
    OptimizationType,
    CalculationTypeStr,
    RadiationTypeStr,
)


@dataclass
class AtomicCalculation:
    """
    Unified interface for atomic structure calculations.

    **Philosophy**: Express WHAT you want (physical concepts), let the system
    figure out HOW (backend-specific implementation).

    This class provides a code-agnostic way to specify atomic calculations
    that works across multiple backends (AUTOSTRUCTURE, FAC, etc.).

    Parameters
    ----------
    element : str
        Element symbol (e.g., "Fe", "O", "Ne")
    charge : int
        Ionic charge (0 = neutral, 15 = +15 for Fe XVI)
    calculation_type : str
        Type of calculation: "structure", "radiative", "photoionization",
        "autoionization", "DR", "RR", "collision"

    coupling : str, optional
        Angular momentum coupling scheme (matches AUTOSTRUCTURE CUP):
        - "CA": Configuration average
        - "LS": Pure LS-coupling (default)
        - "LSM"/"MVD": LS with mass-velocity + Darwin
        - "IC": Intermediate coupling
        - "ICM": IC with mass-velocity + Darwin
        - "CAR"/"LSR"/"ICR": Kappa-averaged relativistic (rel in radial eqs)

        **Note**: FAC is always fully relativistic (jj-based) and issues warnings
        for LS/IC requests. AS supports all these natively.

    relativistic : str, optional
        Relativistic corrections (INDEPENDENT of coupling choice):
        - "none": No additional corrections (default)
        - "Breit": Breit interaction (IBREIT=1 in AS)
        - "Breit_full": Generalized Breit
        - "QED": QED corrections
        - "retardation": Full retardation

        **Key**: These are ADDITIONS to your coupling choice!
        E.g., coupling="IC" + relativistic="Breit" → CUP='IC' IBREIT=1
        (NOT CUP='ICR'! That's a different coupling where rel is in radial eqs)

        **Note**: FAC is always fully relativistic (Dirac). AS adds corrections.

    qed_corrections : bool, optional
        Include QED corrections (vacuum polarization, self-energy)
        Default: False

    optimization : str | None, optional
        Optimization target: "energy", "potential", "lambda", None
        Default: None (no optimization)

        **Translation**:
        - AS: "energy"/"lambda" → INCLUD=Y, NLAM=5
        - FAC: "energy"/"potential" → OptimizeRadial()

    core : str | None, optional
        Core configuration: "He-like", "Ne-like", "Ar-like", or explicit "1s2.2s2"
        Default: None (no core)

    configurations : list | None, optional
        List of Configuration objects or configuration strings
        Default: None (must be provided later)

    energy_range : tuple | None, optional
        Energy grid: (min_eV, max_eV, n_points)
        Default: None (required for PI/DR/RR calculations)

    radiation_types : list[str], optional
        Radiation types to calculate: ["E1", "M1", "E2", etc.]
        Default: ["E1"]

    code : str, optional
        Which code to use: "autostructure" or "fac"
        Default: "autostructure"

    code_options : dict | None, optional
        Code-specific parameters that don't map to physical concepts.
        These are passed directly to the backend writer.

        **Examples**:

        For AUTOSTRUCTURE:
            {"SCFRAC": 0.85, "NLAM": 5, "IBREIT": 1}

        For FAC:
            {"MaxLevels": 1000, "SetPEGrid": [0.1, 10.0, 100]}

    output_dir : str, optional
        Directory for output files
        Default: "."

    name : str | None, optional
        Calculation name (used for filenames)
        Default: "{element}_{charge}_{calculation_type}"

    Examples
    --------
    Simple structure calculation:

    >>> calc = AtomicCalculation(
    ...     element="Fe",
    ...     charge=15,
    ...     calculation_type="structure"
    ... )
    >>> calc.write_input()  # Generates fe_15_structure.dat

    Intermediate coupling with Breit interaction:

    >>> calc = AtomicCalculation(
    ...     element="Fe",
    ...     charge=15,
    ...     calculation_type="structure",
    ...     coupling="IC",
    ...     relativistic="Breit"
    ... )

    DR calculation with energy optimization:

    >>> calc = AtomicCalculation(
    ...     element="Fe",
    ...     charge=15,
    ...     calculation_type="DR",
    ...     coupling="IC",
    ...     optimization="energy",
    ...     energy_range=(0, 100, 1000)
    ... )

    Code-specific tuning (AUTOSTRUCTURE):

    >>> calc = AtomicCalculation(
    ...     element="Fe",
    ...     charge=15,
    ...     calculation_type="structure",
    ...     coupling="IC",
    ...     code="autostructure",
    ...     code_options={
    ...         "SCFRAC": 0.85,  # SCF convergence fraction
    ...         "NLAM": 5,       # Lambda scaling points
    ...         "IBREIT": 1,     # Breit interaction
    ...     }
    ... )

    Compare codes (same physical input):

    >>> for code in ["autostructure", "fac"]:
    ...     calc = AtomicCalculation(
    ...         element="Fe",
    ...         charge=15,
    ...         calculation_type="structure",
    ...         coupling="IC",
    ...         relativistic="Breit",
    ...         qed_corrections=True,
    ...         code=code
    ...     )
    ...     calc.write_input()
    ...     # Warnings shown for limitations (e.g., FAC can't change coupling)
    """

    # Required parameters
    element: str
    charge: int
    calculation_type: CalculationTypeStr

    # Physical specifications (code-agnostic)
    coupling: CouplingType | None = (
        None  # Optional: defaults per backend (ICR for AS, N/A for FAC)
    )
    relativistic: RelativisticType = "none"
    qed_corrections: bool = False
    optimization: OptimizationType = None

    # Calculation configuration
    core: str | None = None
    configurations: list[Any] | None = None
    energy_range: tuple[float, float, int] | None = None
    radiation_types: list[RadiationTypeStr] = field(default_factory=lambda: ["E1"])

    # Advanced configuration generation (AUTOSTRUCTURE ICFG)
    auto_generate_configs: int | None = None  # 1=single, 2=double, 3=triple excitations
    min_occupation: list[int] | None = None  # Minimum electrons per orbital
    max_occupation: list[int] | None = None  # Maximum electrons per orbital
    base_config_promotions: tuple[list[int], int] | None = (
        None  # (base_config, n_promotions)
    )

    # Pseudo-state expansion (AUTOSTRUCTURE NXTRA/LXTRA)
    n_extra_orbitals: int | None = None  # Number of extra n-shells for CI
    l_max_extra: int | None = None  # Maximum l for extra orbitals
    orthogonality: str | None = None  # Orthogonality enforcement

    # Orbital basis control (AUTOSTRUCTURE BASIS)
    orbital_basis: str | None = None  # 'RLX' (relaxed), 'SRLX' (simplified relaxed)

    # Plasma screening (AUTOSTRUCTURE PPOT/NDEN)
    plasma_potential: str | None = None  # 'MCKS', 'TFKS', etc.
    plasma_density: float | None = None  # Electron density in cm^-3

    # Custom orbital input (AUTOSTRUCTURE SRADWIN)
    custom_orbitals_file: str | None = None  # Path to external orbital file

    # Code selection and tuning
    code: Literal["autostructure", "fac"] = "autostructure"
    code_options: dict[str, Any] | None = None

    # Output control
    output_dir: str = "outputs"  # Changed default from "." to "outputs"
    name: str | None = None

    def __post_init__(self):
        """Initialize derived attributes and validate."""
        if self.name is None:
            self.name = f"{self.element.lower()}_{self.charge}_{self.calculation_type}"

        # Import backend here to avoid circular imports
        from .backends import get_backend

        self._backend = get_backend(self.code)
        self._warnings: list[str] = []

        # Validate and collect warnings
        self._validate()

    def _validate(self) -> None:
        """Validate configuration and collect warnings."""
        # Check backend capabilities
        caps = self._backend.capabilities()

        # Set default coupling per backend if not specified
        if self.coupling is None:
            if self.code == "autostructure":
                self.coupling = "ICR"  # Default for AUTOSTRUCTURE
            # FAC doesn't use coupling - leave as None

        # Coupling scheme warnings (only for AUTOSTRUCTURE)
        if self.coupling is not None and self.code == "autostructure":
            if self.coupling not in caps.get("coupling_schemes", []):
                self._warnings.append(
                    f"{self.code.upper()} does not support {self.coupling} coupling. "
                    f"Using default: {caps.get('default_coupling', 'ICR')}"
                )
                self.coupling = caps.get("default_coupling", "ICR")
        elif self.coupling is not None and self.code == "fac":
            self._warnings.append(
                f"FAC is always jj-coupling (Dirac-based). The 'coupling' parameter is ignored."
            )

        # Relativistic treatment warnings
        if self.relativistic not in caps.get("relativistic", []):
            self._warnings.append(
                f"{self.code.upper()} does not support '{self.relativistic}' treatment. "
                f"Using closest available: {caps.get('default_relativistic', 'Dirac')}"
            )

        # Optimization warnings
        if self.optimization and self.optimization not in caps.get("optimization", []):
            self._warnings.append(
                f"{self.code.upper()} does not support '{self.optimization}' optimization. "
                f"Using equivalent: {caps.get('optimization', [])[0] if caps.get('optimization') else 'none'}"
            )

        # Required parameters for calculation types
        if self.calculation_type in ["photoionization", "DR", "RR"]:
            if not self.energy_range:
                raise ValueError(
                    f"{self.calculation_type} requires energy_range parameter: "
                    "(min_eV, max_eV, n_points)"
                )

    def write_input(self, verbose: bool = True) -> Path:
        """
        Generate input file for the selected code.

        Parameters
        ----------
        verbose : bool, optional
            Print warnings about code limitations
            Default: True

        Returns
        -------
        Path
            Path to the generated input file

        Notes
        -----
        This method translates the physical specifications to code-specific
        parameters and generates the appropriate input file format.
        """
        if verbose and self._warnings:
            print("⚠️  Warnings:")
            for warning in self._warnings:
                print(f"   {warning}")

        return self._backend.write_input(self)

    @property
    def warnings(self) -> list[str]:
        """Get list of compatibility warnings."""
        return self._warnings.copy()

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        Useful for saving/loading calculations or generating reports.
        """
        return {
            "element": self.element,
            "charge": self.charge,
            "calculation_type": self.calculation_type,
            "coupling": self.coupling,
            "relativistic": self.relativistic,
            "qed_corrections": self.qed_corrections,
            "optimization": self.optimization,
            "core": self.core,
            "energy_range": self.energy_range,
            "radiation_types": self.radiation_types,
            "code": self.code,
            "code_options": self.code_options,
            "output_dir": self.output_dir,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtomicCalculation:
        """
        Create from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with calculation parameters

        Returns
        -------
        AtomicCalculation
            New calculation instance
        """
        return cls(**data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AtomicCalculation("
            f"element={self.element!r}, "
            f"charge={self.charge}, "
            f"type={self.calculation_type!r}, "
            f"code={self.code!r})"
        )
