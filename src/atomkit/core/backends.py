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
                "jj",  # Approximated by IC in AUTOSTRUCTURE
            ],
            "relativistic": ["none", "Breit", "Breit_full", "QED", "retardation"],
            "qed": True,
            "optimization": ["energy", "lambda"],
            "calculation_types": [
                "structure",
                "radiative",
                "photoionization",
                "autoionization",
                "auger",
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
            "mpi_support": False,  # AUTOSTRUCTURE doesn't support MPI
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
                "jj": "IC",  # AUTOSTRUCTURE approximates jj with IC
            }
            cup_value = coupling_map.get(calc.coupling, "LS")
            params["CUP"] = cup_value  # Always add CUP parameter

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
            "auger": "  ",  # Like radiative, but will add AI tables
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
        from atomkit import Configuration
        from atomkit.autostructure import ASWriter

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
        salgeb_params = {}

        # SALGEB-specific parameters (not SMINIM, SRADCON, DRR)
        # Parameters that belong in SALGEB namelist
        salgeb_keys = {
            "CUP",
            "RAD",
            "RUN",
            "MXCONF",
            "MXVORB",
            "MXCCF",
            "KCOR1",
            "KCOR2",
            "KORB1",
            "KORB2",
            "KORB",
            "AUGER",
            "BORN",
            "BP",
            "KUTSS",
            "KUTSO",
            "KUTOO",
            "BASIS",
            "KCUT",
            "KCUTCC",
            "KCUTI",
            "NAST",
            "NASTJ",
            "NASTS",
            "NASTP",
            "NASTPJ",
            "ICFG",
            "NXTRA",
            "LXTRA",
            "IFILL",
            "MINLT",
            "MAXLT",
            "MINST",
            "MAXST",
            "MINJT",
            "MAXJT",
            "MAXLX",
            "MXLAMX",
            "LRGLAM",
            "KUTOOX",
            "KUTSSX",
            "MAXJFS",
            "KPOLE",
            "KPOLM",
            "NMETA",
            "NMETAJ",
            "INAST",
            "INASTJ",
            "TARGET",
            "MSTART",
            "KUTDSK",
            "KUTLS",
        }

        # Add all SALGEB parameters from params
        for key in salgeb_keys:
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
            # Collect DRR parameters from code_options
            drr_params = {}
            drr_keys = {"NMIN", "NMAX", "JND", "LMIN", "LMAX", "NMESH", "RNUC"}
            if calc.code_options:
                for k in drr_keys:
                    if k in calc.code_options:
                        drr_params[k] = calc.code_options[k]

            # Use sensible defaults if not specified
            drr_params.setdefault("NMIN", 3)
            drr_params.setdefault("NMAX", 15)
            drr_params.setdefault("LMIN", 0)
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
            "coupling_schemes": ["jj"],  # FAC is always jj-based (Dirac)
            "relativistic": [
                "Dirac",
                "Breit",
                "QED",
            ],  # Always Dirac, can add Breit/QED
            "qed": True,
            "optimization": [],  # FAC always optimizes ground state, no user control
            "calculation_types": [
                "structure",
                "radiative",
                "photoionization",
                "autoionization",
                "auger",
                "DR",
                "RR",
                "collision",
            ],
            "limitations": [
                "Always jj-coupling based (cannot change to LS/IC)",
                "Always fully relativistic (Dirac equation)",
                "Can request LS term labels but physics is jj",
                "No lambda scaling (uses potential optimization)",
            ],
            "default_coupling": "ICR",  # Closest to FAC's jj+relativistic
            "default_relativistic": "none",  # Already Dirac-based
            "mpi_support": True,  # FAC supports MPI parallelization
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
        # FAC always optimizes ground state - no user control
        # if calc.optimization in ["energy", "potential"]:
        #     fac_calls.append("")
        #     fac_calls.append("# Optimization")
        #     if calc.configurations:
        #         groups = [f"'g{i}'" for i in range(len(calc.configurations))]
        #         fac_calls.append(f"fac.OptimizeRadial([{', '.join(groups)}])")
        #     else:
        #         fac_calls.append("# fac.OptimizeRadial([...])  # Add groups")

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
                fac_calls.append(f"fac.PrintTable('{calc.name}.tr', 'TR.asc')")

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
            fac_calls.append(f"# fac.PrintTable('{calc.name}.rr', 'RR.asc')")

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
            fac_calls.append(f"# fac.PrintTable('{calc.name}.ai', 'AI.asc')")

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
                fac_calls.append(
                    f"fac.PrintTable('{calc.name}.ce', '{calc.name}.ce.asc')"
                )

        return {"fac_calls": fac_calls}

    def write_input(self, calc: AtomicCalculation) -> Path:
        """
        Generate FAC .sf input file using SFACWriter.

        Now properly uses the atomkit.fac.SFACWriter module!
        """
        from atomkit import Configuration
        from atomkit.fac import SFACWriter

        # Create output directory if it doesn't exist
        output_path = Path(calc.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = output_path / f"{calc.name}.sf"

        # Use SFACWriter to generate the file
        with SFACWriter(filename) as fac:
            # MPI initialization (if requested)
            if calc.n_cores is not None and calc.n_cores > 1:
                fac.InitializeMPI(calc.n_cores)

            # Add header comment
            fac.add_comment(f"{calc.element} {calc.charge}+ {calc.calculation_type}")
            fac.add_blank_line()
            # Place relativistic/QED settings first (requested ordering)
            if calc.relativistic == "Breit" or calc.relativistic == "Breit_full":
                fac.SetBreit(-1)
            if calc.qed_corrections or calc.relativistic == "QED":
                fac.SetVP(-1)  # Vacuum polarization
                fac.SetSE(-1)  # Self-energy

            # Set atom (FAC typically expects this early; after settings is acceptable)
            fac.SetAtom(calc.element)

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
                electrons_per_group: dict[str, int] = {}
                for i, config in enumerate(calc.configurations):
                    if isinstance(config, str):
                        # String configuration
                        config_str = config
                        # Cannot derive electrons reliably from raw string; skip counting
                        electron_count = None  # type: ignore
                        group_name = f"g{i}"
                    else:
                        # atomkit Configuration object
                        # Convert to FAC format (e.g., "2s2 2p6" or "2*8")
                        config_str = self._config_to_fac(config)
                        try:
                            electron_count = config.total_electrons()
                        except Exception:
                            electron_count = None  # type: ignore

                        # Prefer a human-friendly group label if present on the
                        # Configuration (generated by generate_hole_configurations).
                        group_name = getattr(config, "group_label", f"g{i}")

                    fac.Config(config_str, group=group_name)
                    groups.append(group_name)
                    if electron_count is not None:
                        electrons_per_group[group_name] = electron_count

            # Structure calculation (includes optimization and energy references)
            if calc.configurations:
                # `groups` was populated during Config() creation above and may
                # already contain friendly labels. Reuse it rather than
                # regenerating generic `g{i}` names.

                fac.add_blank_line()
                fac.add_comment("Potential Optimization")

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
                fac.PrintTable(f"{calc.name}.lev.b", f"{calc.name}.lev.asc", 1)

            # Radiative transitions
            if (
                calc.calculation_type in ["radiative", "structure", "auger"]
                and calc.configurations
            ):
                fac.add_blank_line()
                fac.add_comment("Radiative transitions")
                # `groups` already contains the correct group names

                # Map radiation types to FAC multipole codes
                multipole_map = {
                    "E1": -1,
                    "E2": -2,
                    "E3": -3,
                    "E4": -4,
                    "M1": 1,
                    "M2": 2,
                    "M3": 3,
                    "M4": 4,
                }

                # Determine multipole list to emit TRTable calls for.
                multipoles_to_emit: list[int] = []

                # Priority: explicit `multipoles` field -> `max_multipole` shorthand ->
                # user-specified `radiation_types` -> defaults (auger-specific)
                if calc.multipoles:
                    multipoles_to_emit = list(calc.multipoles)
                elif calc.max_multipole and calc.max_multipole > 0:
                    # Generate electric (negative) multipoles up to max (E1..EN)
                    # and include M1 (magnetic dipole) by default. This keeps
                    # the common Auger default as E1, E2 and M1 for max_multipole=2.
                    negs = [-i for i in range(1, calc.max_multipole + 1)]
                    poss = [1]
                    multipoles_to_emit = negs + poss
                elif calc.radiation_types and calc.radiation_types != ["E1"]:
                    # Convert textual radiation_types into multipole integers
                    for rad_type in calc.radiation_types:
                        if rad_type in multipole_map:
                            multipoles_to_emit.append(multipole_map[rad_type])
                        else:
                            fac.add_comment(f"Unknown radiation type: {rad_type}")
                else:
                    # Default behavior
                    if calc.calculation_type == "auger":
                        # Auger-focused default: emit E1,E2,M1 as separate calls
                        # expressed as FAC multipole flags: E1=-1, E2=-2, M1=1
                        multipoles_to_emit = [-1, -2, 1]
                    else:
                        # Regular radiative/structure default: E1 only
                        multipoles_to_emit = [ -1 ]

                # Emit multiple TRTable calls into a single .tr.b file, then
                # write a single PrintTable at the end. This keeps all
                # multipole contributions in one transitions file.
                tr_base = f"{calc.name}.tr.b"
                for m in multipoles_to_emit:
                    fac.TRTable(tr_base, groups, groups, multipole=m)
                # Single print of the combined transitions table
                fac.PrintTable(tr_base, f"{calc.name}.tr.asc", 1)

            # Autoionization (Auger) transitions
            if (
                calc.calculation_type in ["autoionization", "auger"]
                and calc.configurations
            ):
                fac.add_blank_line()
                fac.add_comment("Autoionization (Auger) transitions")
                groups_all = groups

                # Infer electron counts to classify gc/one-hole/two-hole sets
                # Use maximum electron count as ground, then N-1, N-2 as hole sets
                classified = {"gc": [], "oneH": [], "twoH": []}
                if "electrons_per_group" in locals() and electrons_per_group:
                    try:
                        max_e = max(electrons_per_group.values())
                        for g in groups_all:
                            ne = electrons_per_group.get(g)
                            if ne is None:
                                continue
                            if ne == max_e:
                                classified["gc"].append(g)
                            elif ne == max_e - 1:
                                classified["oneH"].append(g)
                            elif ne == max_e - 2:
                                classified["twoH"].append(g)
                    except ValueError:
                        # Fallback: if no counts, do nothing
                        pass

                # Single AI table file for all autoionization transitions
                ai_base = f"{calc.name}.ai.b"
                
                # Ground → one-hole AI table (ionization widths)
                if classified["gc"] and classified["oneH"]:
                    fac.AITable(
                        ai_base, classified["gc"], classified["oneH"]
                    )

                # One-hole → two-hole AI table (Auger decay)
                if classified["oneH"] and classified["twoH"]:
                    fac.AITable(
                        ai_base, classified["oneH"], classified["twoH"]
                    )
                
                # Single print at the end for all AI transitions
                if classified["oneH"] and (classified["gc"] or classified["twoH"]):
                    fac.PrintTable(ai_base, f"{calc.name}.ai.asc", 1)

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
                        f"Then: RRTable(), PrintTable('{calc.name}.rr', 'RR.asc')"
                    )

                # DR/Autoionization
                elif calc.calculation_type in ["DR", "autoionization"]:
                    fac.add_blank_line()
                    fac.add_comment(
                        "Dielectronic recombination / Autoionization energy grid"
                    )
                    fac.SetUsrCEGrid([emin, emax, npoints])
                    # If autoionization, AITable calls were added above based on configurations

                # Collision
                elif calc.calculation_type == "collision":
                    fac.add_blank_line()
                    fac.add_comment("Electron impact excitation")
                    fac.SetUsrCEGrid([emin, emax, npoints])
                    if calc.configurations:
                        # reuse existing `groups` list
                        fac.CETable(f"{calc.name}.ce.b", groups, groups)
                        fac.PrintTable(f"{calc.name}.ce.b", f"{calc.name}.ce.asc", 1)

            # MPI finalization (if initialized)
            if calc.n_cores is not None and calc.n_cores > 1:
                fac.FinalizeMPI()

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
