"""
Effective potential calculations for atomic structure.

Provides models for calculating effective potentials used in atomic
structure calculations, including Thomas-Fermi-Dirac-Amaldi (TFDA)
and Slater-Type-Orbital (STO) potentials.

Compatible with NumPy 2.x and modern Python 3.13+.

Original Author: Tomás Campante (October 2025)
Adapted by: Ricardo Silva (rfsilva@lip.pt)
Date: October 2025
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging
import math

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class EffectivePotentialCalculator:
    """
    Calculator for effective potentials in atomic structure calculations.

    Implements Thomas-Fermi-Dirac-Amaldi (TFDA) and Slater-Type-Orbital (STO)
    model potentials for use in atomic structure codes like AUTOSTRUCTURE.

    Parameters
    ----------
    atom_data : dict
        Dictionary containing atomic information:
        - Z : int, atomic number
        - ion_charge : int, ionization charge (0 for neutral)
        - spectator_electron : dict with 'n' and 'l' quantum numbers
        - s : float, spin quantum number
        - shells : list of dict (optional), electronic configuration
    r_values : array-like
        Radial grid points in atomic units

    Examples
    --------
    >>> atom_data = {
    ...     'Z': 26,
    ...     'ion_charge': 1,
    ...     'spectator_electron': {'n': 3, 'l': 2},
    ...     's': 0.5
    ... }
    >>> r_grid = np.linspace(0.01, 10, 1000)
    >>> calc = EffectivePotentialCalculator(atom_data, r_grid)
    >>> lambdas = [1.0] * calc.num_shells
    >>> V = calc.calculate_potential_curve('tfda', lambdas)

    Notes
    -----
    Preserves all original calculation logic from Tomás Campante's
    potential_calculator.py, modernized for NumPy 2.x compatibility.
    """

    # Physical constants for TFDA potential
    DCON1: float = 140.0  # Exponential cutoff threshold
    DCON2: float = 0.2075  # TFDA screening parameter
    DCON3: float = 1.19  # TFDA potential parameter
    DCON4: float = (6.0 / (math.pi**2)) ** (1 / 3)

    # Numerical constants
    AJUSTX: float = 1.0
    D1THRD: float = 1 / 3
    D2THRD: float = 2 / 3
    D1M70: float = 1e-70

    # Aufbau principle orbital filling order
    _ORBITAL_FILL_ORDER: List[Tuple[int, int, int]] = [
        (1, 0, 2),  # 1s
        (2, 0, 2),  # 2s
        (2, 1, 6),  # 2p
        (3, 0, 2),  # 3s
        (3, 1, 6),  # 3p
        (4, 0, 2),  # 4s
        (3, 2, 10),  # 3d
        (4, 1, 6),  # 4p
        (5, 0, 2),  # 5s
        (4, 2, 10),  # 4d
        (5, 1, 6),  # 5p
        (6, 0, 2),  # 6s
        (4, 3, 14),  # 4f
        (5, 2, 10),  # 5d
        (6, 1, 6),  # 6p
        (7, 0, 2),  # 7s
        (5, 3, 14),  # 5f
        (6, 2, 10),  # 6d
        (7, 1, 6),  # 7p
        (8, 0, 2),  # 8s
    ]

    # Exceptions to Aufbau filling (e.g., Au: [Xe] 4f14 5d10 6s1)
    _AUFBAU_EXCEPTIONS: Dict[int, Dict[str, List[Tuple[int, int, int]]]] = {
        79: {  # Gold (Au)
            "shells": [
                (1, 0, 2),
                (2, 0, 2),
                (2, 1, 6),
                (3, 0, 2),
                (3, 1, 6),
                (3, 2, 10),
                (4, 0, 2),
                (4, 1, 6),
                (4, 2, 10),
                (4, 3, 14),
                (5, 0, 2),
                (5, 1, 6),
                (5, 2, 10),
                (6, 0, 1),  # 6s1 exception
            ]
        },
    }

    def __init__(self, atom_data: Dict[str, Any], r_values: List[float] | np.ndarray):
        """Initialize effective potential calculator."""
        self.atom_data = atom_data
        # Use np.asarray for NumPy 2.x compatibility
        self.r_values = np.asarray(r_values, dtype=np.float64)

        # Generate or use provided shell configuration
        if "shells" in self.atom_data and self.atom_data["shells"]:
            self.shells = self.atom_data["shells"]
        else:
            self.shells = self._generate_shells_from_z(
                self.atom_data["Z"], self.atom_data["ion_charge"]
            )

        # Extract key parameters
        self.nuclear_charge = -(self.atom_data["Z"] - self.atom_data["ion_charge"])
        self.spectator_n = self.atom_data["spectator_electron"]["n"]
        self.spectator_l = self.atom_data["spectator_electron"]["l"]
        self.spectator_spin = self.atom_data["s"]

        # Shell arrays for fast access
        self.num_shells = len(self.shells)
        self.principal_n = [shell["n"] for shell in self.shells]
        self.electron_occupations = [shell["occupation"] for shell in self.shells]

        logger.debug(
            f"Initialized potential calculator: Z={atom_data['Z']}, "
            f"ion_charge={atom_data['ion_charge']}, {self.num_shells} shells"
        )

    def _generate_shells_from_z(
        self, Z: float, ion_charge: int
    ) -> List[Dict[str, float]]:
        """
        Generate electronic configuration using Aufbau principle.

        Parameters
        ----------
        Z : float
            Atomic number
        ion_charge : int
            Ionization charge

        Returns
        -------
        shells : list of dict
            Electronic shell configuration
        """
        electron_count = int(Z - ion_charge)

        # Check for Aufbau exceptions (e.g., Au)
        if ion_charge == 0 and Z in self._AUFBAU_EXCEPTIONS:
            logger.debug(f"Using Aufbau exception for Z={Z}")
            return [
                {"n": n, "l": l, "occupation": float(occ)}
                for n, l, occ in self._AUFBAU_EXCEPTIONS[Z]["shells"]
            ]

        # Standard Aufbau filling
        shells = []
        current_electrons = 0

        for n, l, max_occ in self._ORBITAL_FILL_ORDER:
            if current_electrons >= electron_count:
                break

            fill_count = min(max_occ, electron_count - current_electrons)
            if fill_count > 0:
                shells.append({"n": n, "l": l, "occupation": float(fill_count)})
                current_electrons += fill_count

        return shells

    def _calculate_effective_potential(
        self,
        potential_type: str,
        r: float,
        lam_set: List[float],
        exchange_model: int,
        exchange_scaling_factor: float,
    ) -> Tuple[float, float]:
        """
        Calculate effective potential at a single radial point.

        Parameters
        ----------
        potential_type : str
            Either 'tfda' or 'sto'
        r : float
            Radial distance (atomic units)
        lam_set : list of float
            Lambda scaling parameters for each shell
        exchange_model : int
            Exchange model identifier (not currently used)
        exchange_scaling_factor : float
            Exchange scaling factor (not currently used)

        Returns
        -------
        potential : float
            Effective potential value
        weight : float
            Weight factor (used in STO model)
        """
        t, vx, wkt = 0.0, 0.0, 0.0

        if self.num_shells <= 0:
            return 0.0, 0.0

        if potential_type == "tfda":
            # Thomas-Fermi-Dirac-Amaldi potential
            t2 = (-self.nuclear_charge) ** self.D1THRD

            for j in range(self.num_shells):
                rho = lam_set[j] * r * t2
                if self.electron_occupations[j] > 0:
                    # Exponential with cutoff to avoid overflow
                    t4 = (
                        math.exp(-self.DCON2 * rho)
                        if abs(self.DCON2 * rho) < self.DCON1
                        else 0.0
                    )
                    t += self.electron_occupations[j] * t4 / (1 + self.DCON3 * rho)

            return -t / r, 0.0

        elif potential_type == "sto":
            # Slater-Type-Orbital potential
            sz = abs(self.spectator_spin)
            z = -self.nuclear_charge

            for j in range(self.num_shells):
                if self.electron_occupations[j] == 0.0:
                    continue

                x = lam_set[j] * r
                t1 = self.electron_occupations[j] - 1.0
                z -= t1 * sz / 2.0

                en = float(self.principal_n[j])
                rho = 2.0 * (z / en) ** 0.5 * x

                # Laguerre polynomial calculation
                t2, t3, t4 = en + en, 1.0, 1.0 / (en + en)
                for i1 in range(1, int(2 * en - 1) + 1):
                    t2 -= 1.0
                    t4 *= rho / float(i1)
                    t3 += t4 * t2

                # Exponential with cutoff
                t6 = math.exp(-rho) if abs(rho) < self.DCON1 else 0.0

                t += self.electron_occupations[j] * t3 * t6
                wkt += self.electron_occupations[j]

                t1 = self.electron_occupations[j] + 1.0
                z -= t1 * sz / 2.0

            return -t / r - vx, wkt

        else:
            raise ValueError(
                f"Potential type must be 'tfda' or 'sto', got '{potential_type}'"
            )

    def calculate_potential_curve(
        self,
        potential_type: str,
        lam_set: List[float],
        exchange_model: int = 0,
        exchange_scaling_factor: float = 0.0,
    ) -> np.ndarray:
        """
        Calculate effective potential over radial grid.

        Parameters
        ----------
        potential_type : str
            Either 'tfda' or 'sto'
        lam_set : list of float
            Lambda scaling parameters for each shell
        exchange_model : int, optional
            Exchange model identifier (default: 0)
        exchange_scaling_factor : float, optional
            Exchange scaling factor (default: 0.0)

        Returns
        -------
        potential : np.ndarray
            Effective potential values at each radial grid point

        Raises
        ------
        ValueError
            If lambda set length doesn't match number of shells

        Examples
        --------
        >>> calc = EffectivePotentialCalculator(atom_data, r_grid)
        >>> lambdas = [1.0] * calc.num_shells
        >>> V_tfda = calc.calculate_potential_curve('tfda', lambdas)
        >>> V_sto = calc.calculate_potential_curve('sto', lambdas)
        """
        if len(lam_set) != self.num_shells:
            raise ValueError(
                f"Lambda set must have {self.num_shells} values for this ion, "
                f"got {len(lam_set)}"
            )

        logger.debug(f"Calculating {potential_type.upper()} potential curve")

        potential_vals = [
            self._calculate_effective_potential(
                potential_type, r, lam_set, exchange_model, exchange_scaling_factor
            )[0]
            for r in self.r_values
        ]

        # Use NumPy 2.x compatible array creation with explicit dtype
        return np.array(potential_vals, dtype=np.float64)

    def get_orbital_peak_radius(
        self, n_orb: int, l_orb: int, lam_set: List[float]
    ) -> float:
        """
        Calculate radial peak position of orbital probability density.

        Computes the radius at which the radial probability density
        (r² |ψ(r)|²) is maximized for a given orbital.

        Parameters
        ----------
        n_orb : int
            Principal quantum number
        l_orb : int
            Orbital angular momentum quantum number
        lam_set : list of float
            Lambda scaling parameters for each shell

        Returns
        -------
        peak_radius : float
            Radius of maximum probability density (atomic units)

        Examples
        --------
        >>> calc = EffectivePotentialCalculator(atom_data, r_grid)
        >>> lambdas = [1.0] * calc.num_shells
        >>> r_peak = calc.get_orbital_peak_radius(3, 2, lambdas)  # 3d orbital

        Notes
        -----
        Uses Slater's rules to estimate effective nuclear charge and shielding.
        """
        # Calculate shielding using Slater's rules
        shielding = 0.0
        for shell in self.shells:
            if shell["n"] < n_orb:
                # Inner shells contribute full shielding
                shielding += shell["occupation"]
            elif shell["n"] == n_orb and shell["l"] < l_orb:
                # Same n, lower l
                shielding += shell["occupation"]
            elif shell["n"] == n_orb and shell["l"] == l_orb:
                # Same orbital: other electrons shield partially
                shielding += (shell["occupation"] - 1) * 0.35

        z_eff = self.atom_data["Z"] - shielding

        # Find lambda for target orbital
        target_lam = 1.0
        shell_exists_in_basis = False
        for i, shell in enumerate(self.shells):
            if shell["n"] == n_orb and shell["l"] == l_orb:
                target_lam = lam_set[i]
                shell_exists_in_basis = True
                break

        if not shell_exists_in_basis:
            logger.warning(f"Orbital ({n_orb},{l_orb}) not in basis set, using λ=1.0")

        # Calculate radial wavefunction
        radial_psi = []
        en = float(n_orb)

        for r in self.r_values:
            # Ensure z_eff is positive
            z_eff_calc = max(z_eff, 1.0)
            rho = 2.0 * (z_eff_calc / en) ** 0.5 * target_lam * r

            # Laguerre polynomial
            t2, t3, t4 = en + en, 1.0, 1.0 / (en + en)
            for i1 in range(1, int(2 * en - 1) + 1):
                t2 -= 1.0
                t4 *= rho / float(i1)
                t3 += t4 * t2

            # Exponential with cutoff
            t6 = math.exp(-rho) if abs(rho) < self.DCON1 else 0.0
            radial_psi.append(t3 * t6)

        # Use NumPy 2.x compatible array creation
        radial_psi_array = np.array(radial_psi, dtype=np.float64)

        # Probability density: r² |ψ(r)|²
        prob_density = self.r_values**2 * radial_psi_array**2
        peak_index = np.argmax(prob_density)

        logger.debug(
            f"Orbital ({n_orb},{l_orb}) peak at r = {self.r_values[peak_index]:.3f} a.u."
        )

        return float(self.r_values[peak_index])

    def plot_potential_comparison(
        self,
        lam_set: List[float],
        output_file: Optional[str | Path] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot TFDA and STO potentials for comparison.

        Parameters
        ----------
        lam_set : list of float
            Lambda scaling parameters
        output_file : str or Path, optional
            If provided, save plot to this file
        show_plot : bool, optional
            Whether to display the plot (default: True)

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        V_tfda = self.calculate_potential_curve("tfda", lam_set)
        V_sto = self.calculate_potential_curve("sto", lam_set)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.r_values, V_tfda, label="TFDA", linewidth=2)
        ax.plot(self.r_values, V_sto, label="STO", linewidth=2, linestyle="--")

        ax.set_xlabel("r (a.u.)", fontsize=12)
        ax.set_ylabel("Effective Potential (a.u.)", fontsize=12)
        ax.set_title(
            f'Effective Potentials for Z={self.atom_data["Z"]}, '
            f'q={self.atom_data["ion_charge"]}',
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        if output_file:
            output_path = Path(output_file)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved potential comparison plot to {output_path}")

        if show_plot:
            plt.show()

        return fig
