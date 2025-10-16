# AUTOSTRUCTURE Complete Implementation Plan

**Goal**: Implement ALL AUTOSTRUCTURE features explicitly with proper class methods, objects, and improved user experience.

**Status**: Currently 95% complete with kwargs. Target: 100% explicit implementation.

---

## Phase 1: SALGEB Advanced Parameters (High Priority)

### 1.1 Core Specification Parameters
```python
def add_salgeb(
    self,
    # Existing parameters...
    MXCONF: int | None = None,
    MXVORB: int | None = None,
    MXCCF: int = 0,
    CUP: str = "LS",
    RAD: str = "  ",
    RUN: str = "  ",
    
    # NEW: Core specification
    KCOR1: int | None = None,  # First core orbital index
    KCOR2: int | None = None,  # Last core orbital index
    KORB1: int | None = None,  # Alternative to KCOR1
    KORB2: int | None = None,  # Alternative to KCOR2
    
    # NEW: Collision/autoionization control
    AUGER: str | None = None,  # 'YES', 'NO', or None (default)
    BORN: str | None = None,   # 'INF', 'YES', 'NO'
    
    # NEW: Fine structure interactions
    KUTSS: int | None = None,  # Valence-valence 2-body fine-structure
    KUTSO: int | None = None,  # Generalized spin-orbit
    KUTOO: int | None = None,  # Two-body non-fine-structure
    
    # NEW: Orbital basis
    BASIS: str | None = None,  # '   ', 'RLX', 'SRLX'
    
    # NEW: Configuration handling
    KCUT: int = 0,            # Correlation configuration handling
    KCUTCC: int = 0,          # For MXCCF configs
    KCUTI: int | None = None, # Rydberg+continuum configs
    
    # NEW: Symmetry restrictions
    NAST: int = 0,            # Number of term symmetries
    NASTJ: int = 0,           # Number of level symmetries
    NASTS: int | None = None, # Subshell symmetries
    NASTP: int | None = None, # Parent term symmetries
    NASTPJ: int | None = None, # Parent level symmetries
    
    # NEW: Configuration generation (CI)
    ICFG: int = 0,            # CI expansion mode
    NXTRA: int | None = None, # Extend nl to n=NXTRA
    LXTRA: int | None = None, # Extend nl to l=LXTRA
    IFILL: int | None = None, # Fill for extended orbitals
    
    # NEW: Mixing restrictions
    KUTLS: int | None = None, # LS-mixing restrictions
    KUTDSK: int | None = None, # Disk storage control
    
    # NEW: Restart facility
    MSTART: int = 0,          # Angular algebra restart
    
    # NEW: Direct excitation parameters
    MINLT: int | None = None, # Min total L
    MAXLT: int | None = None, # Max total L
    MINST: int | None = None, # Min total 2S+1
    MAXST: int | None = None, # Max total 2S+1
    MINJT: int | None = None, # Min 2J
    MAXJT: int | None = None, # Max 2J
    MAXLX: int | None = None, # Max L for exchange
    MXLAMX: int | None = None, # Max exchange multipole
    LRGLAM: int | None = None, # Top-up L value
    KUTOOX: int = -1,         # Collisional 2-body non-fs
    KUTSSX: int = -1,         # Collisional 2-body fs
    MAXJFS: int | None = None, # Max 2J for 2-body fs
    
    # NEW: Metastable specification
    NMETA: int | None = None,  # Number of metastable terms
    NMETAJ: int | None = None, # Number of metastable levels
    INAST: int | None = None,  # (N+1)-electron symmetries
    INASTJ: int | None = None, # (N+1)-electron level symmetries
    TARGET: str | None = None, # 'OLD' or 'NEW'
    
    # NEW: Multipole radiation
    KPOLE: int | None = None,  # Max Ek pole
    KPOLM: int | None = None,  # Max Mk pole
    
    # NEW: R-matrix specific
    KUTSO: int | None = None,  # For R-matrix
    
    **kwargs,
) -> None:
```

### 1.2 Helper Classes for Complex Parameters

```python
class CoreSpecification:
    """Helper class for specifying closed shells."""
    def __init__(self, first_orbital: int, last_orbital: int):
        self.first = first_orbital
        self.last = last_orbital
    
    @classmethod
    def from_shell_name(cls, last_core_shell: str) -> "CoreSpecification":
        """Create from shell name like '2p' or '3d'."""
        # Auto-detect first/last indices
        pass

class SymmetryRestriction:
    """Helper for restricting term/level symmetries."""
    def __init__(self):
        self.terms: list[tuple[int, int, int]] = []  # [(2S+1, L, parity)]
        self.levels: list[tuple[int, int]] = []      # [(2J, parity)]
    
    def add_term(self, S2_plus_1: int, L: int, parity: int) -> None:
        """Add a term symmetry (2S+1, L, parity)."""
        self.terms.append((S2_plus_1, L, parity))
    
    def add_level(self, J2: int, parity: int) -> None:
        """Add a level symmetry (2J, parity)."""
        self.levels.append((J2, parity))

class CIExpansion:
    """Helper for configuration interaction expansion."""
    def __init__(self, mode: int = 0):
        self.mode = mode  # 0, 1, 2, -1
        self.min_occupations: list[int] = []
        self.max_occupations: list[int] = []
        self.promotions: list[int] = []
```

---

## Phase 2: SMINIM Advanced Parameters (High Priority)

```python
def add_sminim(
    self,
    # Existing parameters
    NZION: int,
    INCLUD: int = 0,
    NLAM: int = 0,
    NVAR: int = 0,
    
    # NEW: Optimization control
    IWGHT: int = 1,           # Weighting scheme
    ORTHOG: str | None = None, # 'YES', 'NO', 'LPS'
    MCFMX: int = 0,           # Config for STO potential
    NFIX: int | None = None,  # Tie scaling parameters
    MGRP: int | None = None,  # Orbital epsilon groups
    NOCC: int = 0,            # User-defined occupations
    IFIX: int | None = None,  # Fix orbitals in SCCA
    
    # NEW: Potential specification
    MEXPOT: int = 0,          # 0=Hartree, 1=Hartree+X
    PPOT: str | None = None,  # 'SCCA', 'FAC', or plasma
    
    # NEW: Output control
    PRINT: str = "FORM",      # 'FORM' or 'UNFORM'
    RADOUT: str = "NO",       # 'YES' or 'NO' for R-matrix
    
    # NEW: Energy range
    MAXE: float | None = None, # Max scattering energy (Ry)
    
    # NEW: Energy shifts
    ISHFTLS: int = 0,         # LS energy shifts
    ISHFTIC: int = 0,         # IC energy shifts
    
    # NEW: Relativistic options (for CUP='ICR')
    IREL: int = 1,            # 1=large only, 2=large+small
    INUKE: int | None = None, # -1=point, 0=uniform, 1=Fermi
    IBREIT: int = 0,          # 0=usual, 1=generalized
    QED: int = 0,             # 0=none, 1=VP+SE, -1=also terms
    IRTARD: int = 0,          # 0=no retardation, 1=full
    
    # NEW: Advanced data handling (bundling)
    NMETAR: int | None = None,  # Electron target bundling
    NMETARJ: int | None = None, # Electron target level bundling
    NRSLMX: int = 10000,        # Radiative data bundling
    NMETAP: int | None = None,  # Photon target bundling
    NMETAPJ: int | None = None, # Photon target level bundling
    NDEN: int | None = None,    # Plasma density/temp pairs
    
    **kwargs,
) -> None:
```

### 2.2 Helper Classes

```python
class EnergyShifts:
    """Helper for energy shift specification."""
    def __init__(self):
        self.term_shifts: dict[int, float] = {}
        self.level_shifts: dict[int, float] = {}
        self.units: float = 1.0  # In units of IP(H)
    
    @classmethod
    def from_file(cls, filename: str) -> "EnergyShifts":
        """Load from SHFTLS/SHFTIC file."""
        pass
    
    def add_term_shift(self, term_index: int, energy_shift: float) -> None:
        """Add energy shift for specific term."""
        self.term_shifts[term_index] = energy_shift

class PlasmaConditions:
    """Helper for plasma potential specification."""
    def __init__(self):
        self.densities: list[float] = []   # cm^-3
        self.temperatures: list[float] = []  # eV
    
    def add_condition(self, density: float, temperature: float) -> None:
        """Add density/temperature pair."""
        self.densities.append(density)
        self.temperatures.append(temperature)

class BundlingOptions:
    """Helper for controlling data bundling in large calculations."""
    def __init__(self):
        self.electron_target_resolution: int | None = None
        self.photon_target_resolution: int | None = None
        self.radiative_resolution: int = 10000
    
    @classmethod
    def for_dr(cls, n_metastables: int = 2) -> "BundlingOptions":
        """Preset for DR calculations."""
        obj = cls()
        obj.electron_target_resolution = n_metastables
        return obj
    
    @classmethod
    def for_photoionization(cls, n_targets: int = 999999) -> "BundlingOptions":
        """Preset for PI calculations."""
        obj = cls()
        obj.photon_target_resolution = n_targets
        return obj
```

---

## Phase 3: SRADCON Advanced Parameters

```python
def add_sradcon(
    self,
    # Existing
    MENG: int = 0,
    EMIN: float | None = None,
    EMAX: float | None = None,
    
    # NEW: Additional energies
    MENGI: int | None = None,  # Interpolation energies
    NDE: int = 0,              # Excitation energies
    DEMIN: float | None = None, # Min excitation energy
    DEMAX: float | None = None, # Max excitation energy
    NIDX: int | None = None,    # Extra energies beyond EMAX
    
    # NEW: Energy corrections
    ECORLS: float = 0.0,       # LS target continuum correction
    ECORIC: float = 0.0,       # IC target continuum correction
    
    **kwargs,
) -> None:
```

### 3.2 Helper Class

```python
class ContinuumEnergyGrid:
    """Helper for specifying continuum energy grids."""
    def __init__(self):
        self.final_energies: list[float] = []
        self.excitation_energies: list[float] = []
        self.interpolation_energies: list[float] = []
    
    @classmethod
    def auto_generate(cls, emax: float, n_points: int = 15) -> "ContinuumEnergyGrid":
        """Auto-generate logarithmic grid."""
        pass
    
    @classmethod
    def for_photoionization(cls) -> "ContinuumEnergyGrid":
        """Preset for PI cross sections."""
        pass
    
    @classmethod
    def for_recombination(cls) -> "ContinuumEnergyGrid":
        """Preset for DR/RR rate coefficients."""
        pass
```

---

## Phase 4: DRR Advanced Parameters

```python
def add_drr(
    self,
    # Existing
    NMIN: int,
    NMAX: int,
    LMIN: int = 0,
    LMAX: int = 7,
    NMESH: int | None = None,
    
    # NEW: Radiation control
    NRAD: int = 1000,         # n above which no new rad rates
    
    # NEW: Continuum specification
    LCON: int | None = None,  # Number of continuum l-values
    
    **kwargs,
) -> None:
```

---

## Phase 5: SRADWIN Implementation (External Orbitals)

```python
def add_sradwin(
    self,
    KEY: int = -9,            # -9=APAP format, -10=STO from UNIT5
    external_orbitals: list[ExternalOrbital] | None = None,
) -> None:
    """
    Add SRADWIN namelist for external orbital specification.
    
    Parameters
    ----------
    KEY : int
        -9: Opacity/Iron/RmaX/APAP Project format (default)
        -10: Free-formatted STO/Clementi orbitals from UNIT5
    external_orbitals : list of ExternalOrbital, optional
        List of external orbitals to read
    """
    params = {"KEY": KEY}
    self._write_namelist("SRADWIN", params)

class ExternalOrbital:
    """Representation of an external orbital."""
    def __init__(self, n: int, l: int, source: str):
        self.n = n
        self.l = l
        self.source = source  # Filename or data
```

---

## Phase 6: Enhanced User Experience Methods

### 6.1 High-Level Presets

```python
@classmethod
def for_structure_calculation(
    cls,
    filename: str,
    configurations: list[Configuration],
    nuclear_charge: int,
    coupling: str = "IC",
    radiation: str = "E1",
    core_orbital: str | None = None,
) -> "ASWriter":
    """Create ASWriter preset for structure calculations."""
    asw = cls(filename)
    asw.write_header(f"Structure calculation for Z={nuclear_charge}")
    asw.add_salgeb(CUP=coupling, RAD=radiation)
    info = asw.configs_from_atomkit(configurations, last_core_orbital=core_orbital)
    asw.add_sminim(NZION=nuclear_charge)
    return asw

@classmethod
def for_photoionization(
    cls,
    filename: str,
    target_configs: list[Configuration],
    initial_configs: list[Configuration],
    nuclear_charge: int,
    energy_range: tuple[float, float] = (0.0, 1500.0),
) -> "ASWriter":
    """Create ASWriter preset for photoionization."""
    asw = cls(filename)
    asw.write_header(f"Photoionization calculation for Z={nuclear_charge}")
    asw.add_salgeb(RUN="PI", CUP="LS", RAD="  ", MXCCF=len(initial_configs))
    asw.configs_from_atomkit(target_configs, last_core_orbital="1s")
    # Add initial configs
    asw.add_sminim(NZION=nuclear_charge)
    asw.add_sradcon(MENG=-15, EMIN=energy_range[0], EMAX=energy_range[1])
    return asw

@classmethod
def for_dielectronic_recombination(
    cls,
    filename: str,
    target_configs: list[Configuration],
    autoionizing_configs: list[Configuration],
    nuclear_charge: int,
    n_range: tuple[int, int] = (3, 15),
    l_max: int = 7,
) -> "ASWriter":
    """Create ASWriter preset for DR calculations."""
    asw = cls(filename)
    asw.write_header(f"DR calculation for Z={nuclear_charge}")
    asw.add_salgeb(RUN="DR", CUP="IC", RAD="E1", MXCCF=len(autoionizing_configs))
    asw.configs_from_atomkit(target_configs, last_core_orbital="1s")
    # Add autoionizing configs
    asw.add_sminim(NZION=nuclear_charge)
    asw.add_sradcon()
    asw.add_drr(NMIN=n_range[0], NMAX=n_range[1], LMIN=0, LMAX=l_max, NMESH=-1)
    return asw
```

### 6.2 Fluent Interface

```python
# Allow chaining
asw = (ASWriter("calculation.dat")
       .header("My calculation")
       .coupling("IC")
       .radiation("E1")
       .with_core("1s")
       .optimize_from("2s")
       .add_configs(configs)
       .nuclear_charge(26)
       .lambda_optimization(n_orbitals=5, n_vary=4)
       .write())
```

### 6.3 Validation Methods

```python
def validate(self) -> list[str]:
    """
    Validate the input for common errors.
    
    Returns
    -------
    list of str
        List of warnings/errors found
    """
    warnings = []
    
    # Check mandatory namelists
    if "SALGEB" not in self.lines:
        warnings.append("ERROR: SALGEB namelist required")
    if "SMINIM" not in self.lines:
        warnings.append("ERROR: SMINIM namelist required")
    
    # Check for common issues
    if self.has_continuum and "SRADCON" not in self.lines:
        warnings.append("WARNING: Continuum present but no SRADCON specified")
    
    # Check lambda optimization setup
    if self.nlam > 0 and self.nvar == 0:
        warnings.append("WARNING: NLAM>0 but NVAR=0, no optimization will occur")
    
    return warnings
```

---

## Phase 7: Implementation Timeline

### Week 1: SALGEB Extended Parameters
- Add all explicit SALGEB parameters
- Implement CoreSpecification helper class
- Add symmetry restriction helpers
- Update tests

### Week 2: SMINIM Extended Parameters  
- Add all explicit SMINIM parameters
- Implement EnergyShifts, PlasmaConditions, BundlingOptions classes
- Add relativistic and QED parameters
- Update tests

### Week 3: SRADCON/DRR/SRADWIN
- Complete SRADCON parameters
- Add ContinuumEnergyGrid helper
- Implement SRADWIN method
- Update tests

### Week 4: Enhanced User Experience
- Implement high-level presets
- Add fluent interface
- Add validation methods
- Create comprehensive examples

### Week 5: Documentation & Testing
- Update all docstrings
- Create usage guides for advanced features
- Expand test coverage to 100%
- Performance testing

---

## Success Criteria

✅ All AUTOSTRUCTURE parameters explicitly available
✅ Helper classes for complex parameter types
✅ High-level presets for common calculation types
✅ Validation to catch user errors
✅ Fluent interface for cleaner code
✅ 100% test coverage
✅ Comprehensive documentation
✅ Backward compatible with existing code

---

## Notes

- Keep **kwargs for absolute backward compatibility
- All new parameters should have sensible defaults
- Helper classes should be optional (can still use raw int/str values)
- Presets should cover 80% of use cases
- Validation should be helpful but not block advanced users
