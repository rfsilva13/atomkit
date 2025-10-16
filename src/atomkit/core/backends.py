"""
Backend adapters for translating physical specs to code-specific implementations.

Each backend knows how to:
1. Translate physical concepts to code parameters
2. Generate input files
3. Report its capabilities and limitations
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .calculation import AtomicCalculation


class Backend(ABC):
    """Base class for backend adapters."""

    @abstractmethod
    def translate(self, calc: AtomicCalculation) -> dict[str, Any]:
        """
        Translate physical specifications to backend-specific parameters.

        Parameters
        ----------
        calc : AtomicCalculation
            Calculation with physical specifications

        Returns
        -------
        dict
            Backend-specific parameters
        """
        pass

    @abstractmethod
    def write_input(self, calc: AtomicCalculation) -> Path:
        """
        Generate input file for this backend.

        Parameters
        ----------
        calc : AtomicCalculation
            Calculation to write

        Returns
        -------
        Path
            Path to generated input file
        """
        pass

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """
        Report backend capabilities and limitations.

        Returns
        -------
        dict
            Dictionary with:
            - coupling_schemes: list of supported schemes
            - relativistic: list of supported treatments
            - qed: bool
            - optimization: list of optimization types
            - calculation_types: list of supported calculation types
            - limitations: list of limitation strings
            - default_coupling: default coupling if requested not available
            - default_relativistic: default treatment if requested not available
        """
        pass


class AutostructureBackend(Backend):
    """
    AUTOSTRUCTURE backend adapter.

    Strengths
    ---------
    - Multiple coupling schemes (LS, IC, ICR, jj-approximated)
    - Fine control over relativistic corrections
    - Lambda scaling optimization
    - Extensive collision parameters

    Limitations
    -----------
    - Non-relativistic base (corrections added)
    - Slower for large calculations
    - jj-coupling approximated by IC
    """

    def capabilities(self) -> dict[str, Any]:
        """Report AUTOSTRUCTURE capabilities."""
        return {
            "coupling_schemes": [
                "CA",
                "LS",
                "LSM",
                "MVD",
                "IC",
                "ICM",
                "CAR",
                "LSR",
                "ICR",
            ],
            "relativistic": ["none", "Breit", "Breit_full", "QED", "retardation"],
            "qed": True,
            "optimization": ["energy", "lambda"],
            "calculation_types": [
                "structure",
                "radiative",
                "photoionization",
                "autoionization",
                "DR",
                "RR",
                "collision",
            ],
            "limitations": [
                "No potential optimization (use lambda scaling)",
                "XXR couplings include mass-velocity+Darwin in radial equations",
            ],
            "default_coupling": "LS",
            "default_relativistic": "none",
        }

    def translate(self, calc: AtomicCalculation) -> dict[str, Any]:
        """
        Translate physical specs to AUTOSTRUCTURE parameters.

        Translation Rules (CORRECTED to match AS manual)
        -------------------------------------------------
        coupling → CUP (maps directly):
            "CA"  → "CA"   (configuration average)
            "LS"  → "LS"   (pure LS-coupling, non-relativistic)
            "LSM"/"MVD" → "LSM" (LS + mass-velocity + Darwin)
            "IC"  → "IC"   (intermediate coupling)
            "ICM" → "ICM"  (IC + mass-velocity + Darwin)
            "CAR" → "CAR"  (CA, kappa-averaged relativistic)
            "LSR" → "LSR"  (LS, kappa-averaged relativistic)
            "ICR" → "ICR"  (IC, kappa-averaged relativistic)

        relativistic → IBREIT, IRTARD, QED (INDEPENDENT of coupling):
            "none" → No flags
            "Breit" → IBREIT=1
            "Breit_full" → IBREIT=1 (generalized Breit)
            "QED" → QED=1
            "retardation" → IRTARD=1

        KEY: Relativistic corrections are ADDITIONS to coupling choice!
             ICR is NOT "IC + relativistic" - it's a coupling choice where
             mass-velocity+Darwin are in radial equations!

        qed_corrections → QED:
            True → QED=1
            False → (not set, defaults to 0)

        optimization → INCLUD + NLAM:
            None → (not set)
            "energy"/"lambda" → INCLUD=10, NLAM=5
            "potential" → Warning (AS uses lambda), INCLUD=10, NLAM=5

        calculation_type → RUN:
            "structure" → RUN='  '
            "radiative" → RUN='  '
            "photoionization" → RUN='PI'
            "autoionization" → RUN='AI'
            "DR" → RUN='DR'
            "RR" → RUN='RR'
            "collision" → RUN='DE'
        """
        params: dict[str, Any] = {}

        # Coupling: Direct mapping to CUP (no modification needed!)
        # Only add if not default ICR to match reference files
        if calc.coupling is not None:
            coupling_map = {
                "CA": "CA",
                "LS": "LS",
                "LSM": "LSM",
                "MVD": "LSM",  # MVD is alias for LSM
                "IC": "IC",
                "ICM": "ICM",
                "CAR": "CAR",
                "LSR": "LSR",
                "ICR": "ICR",
            }
            cup_value = coupling_map.get(calc.coupling, "ICR")
            if cup_value != "ICR":  # Only add if not default
                params["CUP"] = cup_value

        # Relativistic: Independent additions (NOT modifying CUP!)
        if calc.relativistic == "Breit" or calc.relativistic == "Breit_full":
            params["IBREIT"] = 1
        if calc.relativistic == "retardation":
            params["IRTARD"] = 1
            params["IREL"] = 2  # Requires small component
        if calc.relativistic == "QED":
            params["QED"] = 1

        # QED corrections
        if calc.qed_corrections:
            params["QED"] = 1

        # Optimization
        if calc.optimization in ["energy", "lambda", "potential"]:
            params["INCLUD"] = 10  # Include 10 lowest states in optimization
            params["NLAM"] = (
                calc.code_options.get("NLAM", 5) if calc.code_options else 5
            )

        # Radiation types (only for radiative calculations)
        if calc.radiation_types and calc.calculation_type in [
            "radiative",
            "photoionization",
            "autoionization",
        ]:
            # Convert list to AS RAD string
            rad_str = "".join(calc.radiation_types)
            params["RAD"] = rad_str[:4].ljust(4)  # AS uses 4-char field

        # Calculation type → RUN parameter
        run_map = {
            "structure": "  ",
            "radiative": "  ",
            "photoionization": "PI",
            "autoionization": "AI",
            "DR": "DR",
            "RR": "RR",
            "collision": "DE",
        }
        params["RUN"] = run_map.get(calc.calculation_type, "  ")

        # Core specification
        if calc.core:
            # Translate He-like, Ne-like, etc. to KCOR1/KCOR2
            core_map = {
                "He-like": (1, 1),
                "Ne-like": (1, 3),
                "Ar-like": (1, 6),
            }
            if calc.core in core_map:
                params["KCOR1"], params["KCOR2"] = core_map[calc.core]

        # Energy range for continuum calculations
        if calc.energy_range:
            emin, emax, npoints = calc.energy_range
            params["EMIN"] = emin
            params["EMAX"] = emax
            params["MENG"] = -abs(npoints)  # Negative for log spacing

        # Advanced configuration generation (ICFG)
        if calc.auto_generate_configs:
            params["ICFG"] = calc.auto_generate_configs

        # Pseudo-state expansion (NXTRA/LXTRA)
        if calc.n_extra_orbitals:
            params["NXTRA"] = calc.n_extra_orbitals
        if calc.l_max_extra:
            params["LXTRA"] = calc.l_max_extra
        if calc.orthogonality:
            params["ORTHOG"] = calc.orthogonality

        # Orbital basis control (BASIS)
        if calc.orbital_basis:
            params["BASIS"] = calc.orbital_basis

        # Plasma screening (PPOT/NDEN)
        if calc.plasma_potential:
            params["PPOT"] = calc.plasma_potential
        if calc.plasma_density:
            params["NDEN"] = calc.plasma_density

        # Merge with user's code_options (user can override anything)
        if calc.code_options:
            params.update(calc.code_options)

        return params

    def write_input(self, calc: AtomicCalculation) -> Path:
        """
        Generate AUTOSTRUCTURE .dat input file.

        Uses ASWriter from atomkit.autostructure module.
        """
        from atomkit.autostructure import ASWriter
        from atomkit import Configuration

        # Get translated parameters
        params = self.translate(calc)

        # Create output directory if it doesn't exist
        output_path = Path(calc.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = output_path / f"{calc.name}.dat"

        # Create writer
        asw = ASWriter(str(filename))
        asw.write_header(f"{calc.element} {calc.charge}+ {calc.calculation_type}")

        # Add SALGEB with translated parameters
        # Only include non-default values to match reference files
        salgeb_params = {}

        # Only add CUP if not default LS
        if "CUP" in params and params["CUP"] != "LS":
            salgeb_params["CUP"] = params["CUP"]
        else:
            salgeb_params["CUP"] = None  # Explicitly None to skip it

        # Only add RAD if specified (not empty/default)
        if "RAD" in params and params["RAD"].strip():
            salgeb_params["RAD"] = params["RAD"]
        else:
            salgeb_params["RAD"] = None  # Explicitly None to skip it

        # Always add RUN, KCOR if present
        for key in ["RUN", "KCOR1", "KCOR2"]:
            if key in params:
                salgeb_params[key] = params[key]

        # Add advanced parameters to SALGEB
        for key in ["ICFG", "NXTRA", "LXTRA", "BASIS"]:
            if key in params:
                salgeb_params[key] = params[key]

        asw.add_salgeb(**salgeb_params)

        # Handle ICFG special format (min/max occupation)
        if calc.auto_generate_configs:
            # Write min/max occupation arrays
            if calc.min_occupation and calc.max_occupation:
                asw.add_icfg_occupation(calc.min_occupation, calc.max_occupation)
            # Write base config and promotions if specified
            if calc.base_config_promotions:
                base_cfg, n_promo = calc.base_config_promotions
                asw.add_icfg_base_config(base_cfg, n_promo)

        # Add configurations if provided (not used with ICFG auto-generation)
        if calc.configurations and not calc.auto_generate_configs:
            if isinstance(calc.configurations[0], str):
                # Convert strings to Configuration objects
                configs = [Configuration.from_string(c) for c in calc.configurations]
            else:
                configs = calc.configurations

            # Only auto-detect core if user specified one OR used KCOR
            # Otherwise write all orbitals explicitly (like das_1)
            has_core = calc.core is not None
            has_kcor = bool(calc.code_options and "KCOR1" in calc.code_options)
            use_auto_detect = has_core or has_kcor

            asw.configs_from_atomkit(configs, auto_detect_core=use_auto_detect)

        # Add SMINIM with optimization and QED params
        # NZION is nuclear charge (atomic number Z), not ionic charge!
        import mendeleev

        nuclear_charge = mendeleev.element(calc.element).atomic_number

        sminim_params: dict[str, Any] = {
            "NZION": nuclear_charge,
        }
        for k in ["INCLUD", "NLAM", "QED", "IBREIT", "IRTARD", "IREL"]:
            if k in params:
                sminim_params[k] = params[k]

        # Add advanced parameters (ORTHOG, PPOT, NDEN)
        for k in ["ORTHOG", "PPOT", "NDEN"]:
            if k in params:
                sminim_params[k] = params[k]

        # Add any user code_options that belong in SMINIM
        # Only pass params that exist and match expected types
        if calc.code_options:
            for k, v in calc.code_options.items():
                # Only allow known SMINIM parameters to avoid type mismatches
                if k in [
                    "SCFRAC",
                    "NVAR",
                    "IWGHT",
                    "MCFMX",
                    "NOCC",
                    "MEXPOT",
                    "MAXE",
                    "ISHFTLS",
                    "ISHFTIC",
                    "INUKE",
                    "NMETAR",
                    "NMETARJ",
                    "NRSLMX",
                    "NMETAP",
                    "NMETAPJ",
                ]:
                    sminim_params[k] = v
                # String parameters need explicit conversion if they come as ints
                elif k in ["PRINT", "RADOUT"] and isinstance(v, str):
                    sminim_params[k] = v

        asw.add_sminim(**sminim_params)

        # Add plasma density data if specified
        if calc.plasma_density and calc.plasma_potential:
            # Default temperature if not specified
            temperature = (
                calc.code_options.get("plasma_temperature", -1e6)
                if calc.code_options
                else -1e6
            )
            asw.add_plasma_density_data(calc.plasma_density, temperature)

        # Add SRADCON for continuum calculations
        if calc.energy_range:
            sradcon_params = {
                k: v for k, v in params.items() if k in ["MENG", "EMIN", "EMAX"]
            }
            if sradcon_params:
                asw.add_sradcon(**sradcon_params)

        # Add DRR for DR/RR calculations
        if calc.calculation_type in ["DR", "RR"]:
            # Use sensible defaults if not specified
            drr_params = calc.code_options.get("DRR", {}) if calc.code_options else {}
            drr_params.setdefault("NMIN", 3)
            drr_params.setdefault("NMAX", 15)
            drr_params.setdefault("LMAX", 7)
            drr_params.setdefault("NMESH", -1)  # Production mesh
            asw.add_drr(**drr_params)

        # Add SRADWIN for custom orbital input
        if calc.custom_orbitals_file:
            # KEY=-10 for free-format STO orbitals (most common)
            # KEY=-9 for Opacity/Iron Project format
            key = (
                calc.code_options.get("SRADWIN_KEY", -10) if calc.code_options else -10
            )
            asw.add_sradwin(KEY=key)

        # Write file
        asw.close()

        return filename


class FACBackend(Backend):
    """
    FAC (Flexible Atomic Code) backend adapter.

    Strengths
    ---------
    - Fully relativistic (Dirac equation)
    - Fast for large calculations
    - Excellent for highly-charged ions
    - Built-in parallelization

    Limitations
    -----------
    - Always jj-coupling (cannot change)
    - Always relativistic (Dirac-based)
    - LS coupling only approximate
    - No lambda scaling
    """

    def capabilities(self) -> dict[str, Any]:
        """Report FAC capabilities."""
        return {
            "coupling_schemes": [],  # FAC cannot change coupling (always jj-based)
            "relativistic": ["Breit", "QED"],  # Always Dirac, can add Breit/QED
            "qed": True,
            "optimization": ["energy", "potential"],
            "calculation_types": [
                "structure",
                "radiative",
                "photoionization",
                "autoionization",
                "DR",
                "RR",
                "collision",
            ],
            "limitations": [
                "Always fully relativistic (Dirac equation)",
                "Always jj-coupling based (cannot change to LS/IC)",
                "Can request LS term labels but physics is jj",
                "No lambda scaling (uses potential optimization)",
            ],
            "default_coupling": "ICR",  # Closest to FAC's jj+relativistic
            "default_relativistic": "none",  # Already Dirac-based
        }

    def translate(self, calc: AtomicCalculation) -> dict[str, Any]:
        """
        Translate physical specs to FAC function calls.

        Translation Rules
        -----------------
        coupling:
            N/A - FAC is always jj-based (Dirac equation)
            The coupling parameter is ignored for FAC

        relativistic:
            "Dirac" → Default (no SetBreit)
            "Breit" → SetBreit(1)
            Others → Warning (FAC always Dirac-based)

        qed_corrections:
            True → SetVP(1); SetSE(1)
            False → No calls

        optimization:
            "energy"/"potential" → OptimizeRadial()
            "lambda" → Warning, uses potential optimization
            None → No optimization

        calculation_type:
            "structure" → Structure(), MemENTable()
            "radiative" → TRTable()
            "photoionization" → SetUsrPEGrid(), RecStates(), RRTable()
            "DR" → SetUsrCEGrid(), RecStates(), AITable()
            "RR" → SetUsrPEGrid(), RecStates(), RRTable()
            "collision" → SetUsrCEGrid(), CETable()
        """
        fac_calls: list[str] = []

        # FAC script header
        fac_calls.append("from pfac import fac")
        fac_calls.append("")
        fac_calls.append(f"# {calc.element} {calc.charge}+ {calc.calculation_type}")
        fac_calls.append("")

        # Always set atom
        fac_calls.append(f"fac.SetAtom('{calc.element}')")

        # Breit interaction
        if calc.relativistic == "Breit":
            fac_calls.append("fac.SetBreit(1)")

        # QED corrections
        if calc.qed_corrections:
            fac_calls.append("fac.SetVP(1)  # Vacuum polarization")
            fac_calls.append("fac.SetSE(1)  # Self-energy")

        # Core (closed configurations)
        if calc.core:
            core_configs = {
                "He-like": "1s2",
                "Ne-like": "1s2 2s2 2p6",
                "Ar-like": "1s2 2s2 2p6 3s2 3p6",
            }
            core_str = core_configs.get(calc.core, calc.core)
            fac_calls.append(f"fac.Closed('{core_str}')")

        # Configurations
        if calc.configurations:
            fac_calls.append("")
            fac_calls.append("# Configurations")
            for i, config in enumerate(calc.configurations):
                if isinstance(config, str):
                    config_str = config
                else:
                    # Convert atomkit Configuration to FAC string
                    config_str = str(config).replace(".", " ")
                fac_calls.append(f"fac.Config('{config_str}', group='g{i}')")

        # Optimization
        if calc.optimization in ["energy", "potential"]:
            fac_calls.append("")
            fac_calls.append("# Optimization")
            if calc.configurations:
                groups = [f"'g{i}'" for i in range(len(calc.configurations))]
                fac_calls.append(f"fac.OptimizeRadial([{', '.join(groups)}])")
            else:
                fac_calls.append("# fac.OptimizeRadial([...])  # Add groups")

        # Structure calculation
        fac_calls.append("")
        fac_calls.append("# Structure")
        if calc.configurations:
            groups = [f"'g{i}'" for i in range(len(calc.configurations))]
            fac_calls.append(f"fac.Structure([{', '.join(groups)}])")
            fac_calls.append(f"fac.MemENTable('{calc.name}.en')")

        # Radiative transitions
        if calc.calculation_type in ["radiative", "structure"]:
            fac_calls.append("")
            fac_calls.append("# Radiative transitions")
            if calc.configurations:
                groups = [f"'g{i}'" for i in range(len(calc.configurations))]
                fac_calls.append(
                    f"fac.TRTable([{', '.join(groups)}], [{', '.join(groups)}])"
                )
                fac_calls.append(f"fac.PrintTable('{calc.name}.tr', 'TR')")

        # Photoionization
        if calc.calculation_type == "photoionization":
            fac_calls.append("")
            fac_calls.append("# Photoionization")
            if calc.energy_range:
                emin, emax, npoints = calc.energy_range
                fac_calls.append(f"fac.SetUsrPEGrid([{emin}, {emax}, {npoints}])")
            fac_calls.append("# Add continuum configurations")
            fac_calls.append("# fac.RecStates(...)")
            fac_calls.append(f"# fac.RRTable(...)")
            fac_calls.append(f"# fac.PrintTable('{calc.name}.rr', 'RR')")

        # DR/AI
        if calc.calculation_type in ["DR", "autoionization"]:
            fac_calls.append("")
            fac_calls.append("# Dielectronic recombination / Autoionization")
            if calc.energy_range:
                emin, emax, npoints = calc.energy_range
                fac_calls.append(f"fac.SetUsrCEGrid([{emin}, {emax}, {npoints}])")
            fac_calls.append("# Add autoionizing configurations")
            fac_calls.append("# fac.RecStates(...)")
            fac_calls.append(f"# fac.AITable(...)")
            fac_calls.append(f"# fac.PrintTable('{calc.name}.ai', 'AI')")

        # Collision
        if calc.calculation_type == "collision":
            fac_calls.append("")
            fac_calls.append("# Electron impact excitation")
            if calc.energy_range:
                emin, emax, npoints = calc.energy_range
                fac_calls.append(f"fac.SetUsrCEGrid([{emin}, {emax}, {npoints}])")
            if calc.configurations:
                groups = [f"'g{i}'" for i in range(len(calc.configurations))]
                fac_calls.append(
                    f"fac.CETable([{', '.join(groups)}], [{', '.join(groups)}])"
                )
                fac_calls.append(f"fac.PrintTable('{calc.name}.ce', 'CE')")

        return {"fac_calls": fac_calls}

    def write_input(self, calc: AtomicCalculation) -> Path:
        """
        Generate FAC .sf input file using SFACWriter.

        Now properly uses the atomkit.fac.SFACWriter module!
        """
        from atomkit.fac import SFACWriter
        from atomkit import Configuration

        # Create output directory if it doesn't exist
        output_path = Path(calc.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = output_path / f"{calc.name}.sf"

        # Use SFACWriter to generate the file
        with SFACWriter(filename) as fac:
            # Add header comment
            fac.add_comment(f"{calc.element} {calc.charge}+ {calc.calculation_type}")
            fac.add_blank_line()

            # Set atom
            fac.SetAtom(calc.element)

            # Breit interaction
            if calc.relativistic == "Breit" or calc.relativistic == "Breit_full":
                fac.SetBreit(1)

            # QED corrections
            if calc.qed_corrections or calc.relativistic == "QED":
                fac.SetVP(1)  # Vacuum polarization
                fac.SetSE(1)  # Self-energy

            # User-specified code options
            if calc.code_options:
                fac.add_blank_line()
                fac.add_comment("User-specified options")

                # Apply FAC-specific options that have direct SFACWriter methods
                if "SetUTA" in calc.code_options:
                    fac.SetUTA(calc.code_options["SetUTA"])

                if "SetMS" in calc.code_options:
                    nms, sms = calc.code_options["SetMS"]
                    fac.SetMS(nms, sms)

                # Note: Other options like MaxLevels, accuracy settings etc.
                # might not have direct SFACWriter methods yet
                # Add them as comments for manual editing
                unsupported = []
                for key, value in calc.code_options.items():
                    if key not in ["SetUTA", "SetMS", "optimize_groups"]:
                        unsupported.append(f"{key} = {value}")

                if unsupported:
                    fac.add_comment("Additional options (add manually if needed):")
                    for opt in unsupported:
                        fac.add_comment(f"  {opt}")

            # Core (closed shells)
            if calc.core:
                core_map = {
                    "He-like": "1s2",
                    "Ne-like": "1s2 2s2 2p6",
                    "Ar-like": "1s2 2s2 2p6 3s2 3p6",
                }
                core_str = core_map.get(calc.core, calc.core)
                fac.Closed(core_str)

            # Configurations
            if calc.configurations:
                fac.add_blank_line()
                fac.add_comment("Configurations")
                groups = []
                for i, config in enumerate(calc.configurations):
                    if isinstance(config, str):
                        # String configuration
                        config_str = config
                    else:
                        # atomkit Configuration object
                        # Convert to FAC format (e.g., "2s2 2p6" or "2*8")
                        config_str = self._config_to_fac(config)

                    fac.Config(config_str, group=f"g{i}")
                    groups.append(f"g{i}")

            # Structure calculation (includes optimization and energy references)
            if calc.configurations:
                groups = [f"g{i}" for i in range(len(calc.configurations))]

                fac.add_blank_line()
                fac.add_comment("Self-consistent field optimization")

                # Pattern from FAC manual/examples:
                # ConfigEnergy(0) -> OptimizeRadial() -> ConfigEnergy(1) -> Structure()

                # Initial reference energy
                fac.ConfigEnergy(0)

                # Radial optimization
                # Check if user wants to optimize specific groups via code_options
                if calc.code_options and "optimize_groups" in calc.code_options:
                    opt_groups = calc.code_options["optimize_groups"]
                    fac.OptimizeRadial(opt_groups)
                elif calc.optimization in ["energy", "potential"]:
                    # Optimize all configurations if explicitly requested
                    fac.OptimizeRadial(groups)
                else:
                    # By default, only optimize ground state (first configuration)
                    fac.OptimizeRadial([groups[0]])

                # Final reference energy
                fac.ConfigEnergy(1)

                # Calculate energy levels
                fac.add_blank_line()
                fac.add_comment("Calculate energy levels")
                fac.Structure(f"{calc.name}.lev.b", groups)
                fac.MemENTable(f"{calc.name}.lev.b")  # Store in memory - IMPORTANT!
                fac.PrintTable(f"{calc.name}.lev.b", f"{calc.name}.lev", 1)

            # Radiative transitions
            if (
                calc.calculation_type in ["radiative", "structure"]
                and calc.configurations
            ):
                fac.add_blank_line()
                fac.add_comment("Radiative transitions")
                groups = [f"g{i}" for i in range(len(calc.configurations))]

                # Map radiation types to FAC multipole codes
                # E1=1, E2=2, E3=3, E4=4, M1=-1, M2=-2, M3=-3, M4=-4
                # multipole=0 means all multipoles
                multipole_map = {
                    "E1": 1,
                    "E2": 2,
                    "E3": 3,
                    "E4": 4,
                    "M1": -1,
                    "M2": -2,
                    "M3": -3,
                    "M4": -4,
                }

                if calc.radiation_types and calc.radiation_types != ["E1"]:
                    # User specified specific radiation types
                    # Create separate TRTable for each type
                    for rad_type in calc.radiation_types:
                        if rad_type in multipole_map:
                            multipole = multipole_map[rad_type]
                            fac.TRTable(
                                f"{calc.name}.tr_{rad_type}.b",
                                groups,
                                groups,
                                multipole=multipole,
                            )
                            # Convert each binary table to ASCII
                            fac.PrintTable(
                                f"{calc.name}.tr_{rad_type}.b",
                                f"{calc.name}.tr_{rad_type}",
                                1,
                            )
                        else:
                            fac.add_comment(f"Unknown radiation type: {rad_type}")
                else:
                    # Default: E1 transitions only
                    fac.TRTable(f"{calc.name}.tr.b", groups, groups, multipole=1)
                    # Convert binary transition tables to ASCII
                    # PrintTable(input_binary, output_ascii, format)
                    fac.PrintTable(f"{calc.name}.tr.b", f"{calc.name}.tr", 1)

            # Energy range for continuum calculations
            if calc.energy_range:
                emin, emax, npoints = calc.energy_range

                # Photoionization
                if calc.calculation_type == "photoionization":
                    fac.add_blank_line()
                    fac.add_comment("Photoionization")
                    fac.SetUsrPEGrid([emin, emax, npoints])
                    fac.add_comment("Add continuum configurations with RecStates()")
                    fac.add_comment(
                        f"Then: RRTable(), PrintTable('{calc.name}.rr', 'RR')"
                    )

                # DR/Autoionization
                elif calc.calculation_type in ["DR", "autoionization"]:
                    fac.add_blank_line()
                    fac.add_comment("Dielectronic recombination / Autoionization")
                    fac.SetUsrCEGrid([emin, emax, npoints])
                    fac.add_comment("Add autoionizing configurations with RecStates()")
                    fac.add_comment(
                        f"Then: AITable(), PrintTable('{calc.name}.ai', 'AI')"
                    )

                # Collision
                elif calc.calculation_type == "collision":
                    fac.add_blank_line()
                    fac.add_comment("Electron impact excitation")
                    fac.SetUsrCEGrid([emin, emax, npoints])
                    if calc.configurations:
                        groups = [f"g{i}" for i in range(len(calc.configurations))]
                        fac.CETable(f"{calc.name}.ce.b", groups, groups)
                        fac.PrintTable(f"{calc.name}.ce", "CE", 1)

        return filename

    def _config_to_fac(self, config) -> str:
        """
        Convert atomkit Configuration object to FAC configuration string.

        Parameters
        ----------
        config : Configuration
            atomkit Configuration object

        Returns
        -------
        str
            FAC-format configuration string (e.g., "2s2 2p6" or "2*8")
        """
        # Simple conversion: just get string representation
        # FAC uses "2s2 2p6" format similar to atomkit
        config_str = str(config)
        # Convert atomkit's "2s2.2p6" to FAC's "2s2 2p6"
        config_str = config_str.replace(".", " ")
        return config_str


# Factory function
def get_backend(code: str) -> Backend:
    """
    Get backend adapter for specified code.

    Parameters
    ----------
    code : str
        Code name: "autostructure" or "fac"

    Returns
    -------
    Backend
        Backend adapter instance

    Raises
    ------
    ValueError
        If code is not recognized
    """
    backends = {
        "autostructure": AutostructureBackend(),
        "fac": FACBackend(),
    }

    if code.lower() not in backends:
        raise ValueError(
            f"Unknown code: {code}. " f"Supported: {', '.join(backends.keys())}"
        )

    return backends[code.lower()]
