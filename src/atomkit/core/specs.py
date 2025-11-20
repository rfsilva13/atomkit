"""
Physical specifications for atomic calculations.

This module defines code-agnostic physical concepts that translate
to backend-specific implementations.
"""

from typing import Literal


class CouplingScheme:
    """
    Angular momentum coupling schemes (matches AUTOSTRUCTURE CUP parameter).

    From AS manual:
    - CA: Configuration average (non-relativistic)
    - LS: Russell-Saunders LS-coupling (non-relativistic, default)
    - LSM/MVD: LS-coupling with mass-velocity and Darwin operators
    - IC: Intermediate coupling (gives non-relativistic LS results as well)
    - ICM: Intermediate coupling (with LS results as per LSM/MVD)
    - CAR/LSR/ICR: Kappa-averaged relativistic wavefunctions (XXR where XX=CA/LS/IC)
      (mass-velocity and Darwin operators included in radial equations)

    FAC note: FAC is always fully relativistic (jj-coupling based), cannot change
    """

    # Non-relativistic schemes
    CA: Literal["CA"] = "CA"  # Configuration average
    LS: Literal["LS"] = "LS"  # Pure LS-coupling (default)
    LSM: Literal["LSM"] = "LSM"  # LS with mass-velocity+Darwin (same as MVD)
    MVD: Literal["MVD"] = "MVD"  # Mass-Velocity+Darwin (same as LSM)
    IC: Literal["IC"] = "IC"  # Intermediate coupling
    ICM: Literal["ICM"] = "ICM"  # IC with mass-velocity+Darwin

    # Kappa-averaged relativistic (rel corrections in radial equations)
    CAR: Literal["CAR"] = "CAR"  # Configuration average, relativistic
    LSR: Literal["LSR"] = "LSR"  # LS-coupling, relativistic
    ICR: Literal["ICR"] = "ICR"  # Intermediate coupling, relativistic

    # Fully relativistic schemes (primarily for FAC)
    JJ: Literal["jj"] = "jj"  # Pure jj-coupling (FAC default)
    LSJ: Literal["LSJ"] = "LSJ"  # LS to jj transformation

    # Backend translations:
    # AUTOSTRUCTURE: Maps directly to CUP parameter (CA/LS/IC variants)
    # FAC: Always jj-based/fully-relativistic, issues warnings for LS requests


class RelativisticTreatment:
    """
    Relativistic corrections (separate from coupling choice in AS).

    In AUTOSTRUCTURE, these are ADDITIONS to the coupling scheme:
    - IBREIT=1: Breit interaction (default=0)
    - IRTARD=1: Full retardation in multipole radiation (requires IREL=2, default=0)
    - IREL=1/2: Neglects/includes small component for kappa-averaged (ICR, etc.)
    - QED=1: Vacuum polarization + self-energy (default=0)

    Note: For XXR couplings (CAR/LSR/ICR), mass-velocity+Darwin are already
    included in radial equations (kappa-averaged), so they're implicit.
    """

    NONE: Literal["none"] = "none"  # No additional relativistic corrections
    BREIT: Literal["Breit"] = "Breit"  # Breit interaction (IBREIT=1)
    BREIT_FULL: Literal["Breit_full"] = "Breit_full"  # Generalized Breit (IBREIT=1)
    QED: Literal["QED"] = "QED"  # QED corrections (QED=1)
    FULL_RETARDATION: Literal["retardation"] = (
        "retardation"  # Full retardation (IRTARD=1)
    )
    DIRAC: Literal["Dirac"] = "Dirac"  # Full Dirac equation (FAC default)

    # Backend translations:
    # AUTOSTRUCTURE: IBREIT, IRTARD, QED, IREL parameters
    # FAC: Always fully relativistic (Dirac), SetBreit() for Breit, SetVP/SetSE for QED


class OptimizationTarget:
    """
    What to optimize in variational calculations.
    """

    ENERGY: Literal["energy"] = "energy"  # Minimize total energy
    POTENTIAL: Literal["potential"] = "potential"  # Optimize radial potential
    LAMBDA: Literal["lambda"] = "lambda"  # Lambda scaling parameters
    NONE: None = None  # No optimization

    # Backend translations:
    # AUTOSTRUCTURE: INCLUD, NLAM parameters (energy → lambda scaling)
    # FAC: OptimizeRadial() (potential optimization)


class CalculationType:
    """
    Type of atomic process to calculate.
    """

    STRUCTURE: Literal["structure"] = "structure"  # Energy levels only
    RADIATIVE: Literal["radiative"] = "radiative"  # Radiative transitions
    PHOTOIONIZATION: Literal["photoionization"] = "photoionization"  # PI cross sections
    AUTOIONIZATION: Literal["autoionization"] = "autoionization"  # Autoionization rates
    DR: Literal["DR"] = "DR"  # Dielectronic recombination
    RR: Literal["RR"] = "RR"  # Radiative recombination
    COLLISION: Literal["collision"] = "collision"  # Electron impact excitation

    # Backend translations:
    # AUTOSTRUCTURE: RUN parameter ('  ', 'PI', 'DR', 'RR', 'DE'), various namelists
    # FAC: Different table functions (TRTable, AITable, RRTable, CETable, etc.)


class RadiationType:
    """
    Types of radiative transitions to calculate.
    """

    E1: Literal["E1"] = "E1"  # Electric dipole
    E2: Literal["E2"] = "E2"  # Electric quadrupole
    E3: Literal["E3"] = "E3"  # Electric octupole
    M1: Literal["M1"] = "M1"  # Magnetic dipole
    M2: Literal["M2"] = "M2"  # Magnetic quadrupole
    M3: Literal["M3"] = "M3"  # Magnetic octupole

    # Backend translations:
    # AUTOSTRUCTURE: RAD parameter (binary flags: 'E1', 'E2', 'M1', etc.)
    # FAC: TRTable() multipole argument


class SpecialModes:
    """
    Special calculation modes for specific codes.

    UTA: Unresolved Transition Arrays (FAC-specific)
    """

    UTA: Literal["UTA"] = "UTA"  # Unresolved transition arrays (FAC SetUTA)
    UTA_FULL: Literal["UTA_full"] = "UTA_full"  # Full UTA mode
    UTA_STATISTICAL: Literal["UTA_stat"] = "UTA_stat"  # Statistical UTA


# Type aliases for convenience
CouplingType = Literal[
    "CA", "LS", "LSM", "MVD", "IC", "ICM", "CAR", "LSR", "ICR", "jj", "LSJ"
]
RelativisticType = Literal["none", "Breit", "Breit_full", "QED", "retardation", "Dirac"]
OptimizationType = Literal["energy", "potential", "lambda"] | None
CalculationTypeStr = Literal[
    "structure",
    "radiative",
    "photoionization",
    "autoionization",
    "auger",
    "DR",
    "RR",
    "collision",
]
RadiationTypeStr = Literal["E1", "E2", "E3", "M1", "M2", "M3"]
SpecialModeStr = Literal["UTA", "UTA_full", "UTA_stat"]
