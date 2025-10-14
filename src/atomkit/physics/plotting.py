"""
Plotting utilities for resonant excitation cross sections.

This module provides convenient plotting functions for visualizing
resonant cross sections, individual resonance contributions, and
energy level diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, List, Tuple, Dict, Union
import pandas as pd


class ResonancePlotter:
    """
    Utility class for creating publication-quality plots of resonant cross sections.

    Examples
    --------
    >>> plotter = ResonancePlotter()
    >>> fig, ax = plotter.plot_cross_section(energy, cross_section)
    >>> plotter.add_resonance_markers(ax, resonance_energies, resonance_widths)
    >>> plt.show()
    """

    def __init__(self, style: str = "default"):
        """
        Initialize the plotter.

        Parameters
        ----------
        style : str
            Matplotlib style to use. Options: 'default', 'seaborn', 'ggplot', etc.
        """
        self.style = style
        if style != "default":
            plt.style.use(style)

    def plot_cross_section(
        self,
        energy: np.ndarray,
        cross_section: np.ndarray,
        ax: Optional[Axes] = None,
        figsize: Tuple[float, float] = (10, 6),
        title: Optional[str] = None,
        xlabel: str = "Electron Energy (eV)",
        ylabel: str = "Cross Section (cm²)",
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot cross section vs energy.

        Parameters
        ----------
        energy : np.ndarray
            Energy grid in eV.
        cross_section : np.ndarray
            Cross section values in cm².
        ax : Axes, optional
            Matplotlib axes to plot on. If None, creates new figure.
        figsize : tuple
            Figure size (width, height) in inches.
        title : str, optional
            Plot title.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        label : str, optional
            Label for the curve (for legend).
        **plot_kwargs
            Additional arguments passed to ax.plot().

        Returns
        -------
        fig, ax : Figure, Axes
            Matplotlib figure and axes objects.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Default plot styling
        default_kwargs = {"linewidth": 2, "color": "blue", "alpha": 0.8}
        default_kwargs.update(plot_kwargs)

        ax.plot(energy, cross_section, label=label, **default_kwargs)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if title:
            ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle="--")

        if label:
            ax.legend()

        return fig, ax

    def plot_resonance_contributions(
        self,
        energy: np.ndarray,
        total_cross_section: np.ndarray,
        resonance_info: Dict,
        num_resonances: Optional[int] = None,
        ax: Optional[Axes] = None,
        figsize: Tuple[float, float] = (12, 7),
        show_total: bool = True,
        title: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot individual resonance contributions.

        Parameters
        ----------
        energy : np.ndarray
            Energy grid in eV.
        total_cross_section : np.ndarray
            Total cross section.
        resonance_info : dict
            Dictionary with keys 'contributions', 'level_index', 'energies', 'widths'.
        num_resonances : int, optional
            Number of strongest resonances to show. If None, shows all.
        ax : Axes, optional
            Matplotlib axes to plot on.
        figsize : tuple
            Figure size.
        show_total : bool
            Whether to plot the total cross section.
        title : str, optional
            Plot title.

        Returns
        -------
        fig, ax : Figure, Axes
            Matplotlib figure and axes objects.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot total cross section
        if show_total:
            ax.plot(
                energy,
                total_cross_section,
                "k-",
                linewidth=2.5,
                label="Total",
                zorder=10,
                alpha=0.9,
            )

        # Find strongest resonances
        contributions = resonance_info["contributions"]
        peak_strengths = [c.max() for c in contributions]

        if num_resonances is None:
            num_resonances = len(contributions)

        # Get indices of strongest resonances
        top_indices = np.argsort(peak_strengths)[::-1][:num_resonances]

        # Color map for resonances
        colors = plt.cm.Set2(np.linspace(0, 1, min(num_resonances, 8)))

        # Plot individual contributions
        for i, idx in enumerate(top_indices):
            contribution = contributions[idx]
            level_idx = resonance_info["level_index"][idx]
            res_energy = resonance_info["energies"][idx]

            # Only plot where contribution is significant
            significant = contribution > 1e-30

            if np.any(significant):
                color = colors[i % len(colors)]
                ax.plot(
                    energy[significant],
                    contribution[significant],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    color=color,
                    label=f"Res {i+1} (Lv {level_idx}, {res_energy:.1f} eV)",
                )

        ax.set_xlabel("Electron Energy (eV)", fontsize=12)
        ax.set_ylabel("Cross Section (cm²)", fontsize=12)
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title("Resonance Contributions to Cross Section", fontsize=14)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="best", framealpha=0.9)

        return fig, ax

    def add_resonance_markers(
        self,
        ax: Axes,
        resonance_energies: Union[List[float], np.ndarray],
        resonance_widths: Optional[Union[List[float], np.ndarray]] = None,
        marker_style: str = "vlines",
        **kwargs,
    ) -> Axes:
        """
        Add vertical markers at resonance positions.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to add markers to.
        resonance_energies : array-like
            Resonance center energies in eV.
        resonance_widths : array-like, optional
            Resonance widths (FWHM) in eV. If provided, shows width as shaded region.
        marker_style : str
            Style of markers: 'vlines', 'arrows', or 'shaded'.
        **kwargs
            Additional styling arguments.

        Returns
        -------
        ax : Axes
            Modified axes object.
        """
        ymin, ymax = ax.get_ylim()

        if marker_style == "vlines":
            # Vertical dashed lines
            default_kwargs = {
                "color": "red",
                "linestyle": "--",
                "alpha": 0.5,
                "linewidth": 1,
            }
            default_kwargs.update(kwargs)

            for energy in resonance_energies:
                ax.axvline(energy, **default_kwargs)

        elif marker_style == "arrows":
            # Arrows pointing down
            default_kwargs = {"color": "red", "alpha": 0.6}
            default_kwargs.update(kwargs)

            arrow_y = ymax * 0.9
            for energy in resonance_energies:
                ax.annotate(
                    "",
                    xy=(energy, ymin),
                    xytext=(energy, arrow_y),
                    arrowprops=dict(arrowstyle="->", lw=1.5, **default_kwargs),
                )

        elif marker_style == "shaded" and resonance_widths is not None:
            # Shaded regions showing width
            default_kwargs = {"color": "red", "alpha": 0.2}
            default_kwargs.update(kwargs)

            for energy, width in zip(resonance_energies, resonance_widths):
                ax.axvspan(energy - width / 2, energy + width / 2, **default_kwargs)

        return ax

    def plot_collision_strength(
        self,
        energy: np.ndarray,
        collision_strength: np.ndarray,
        ax: Optional[Axes] = None,
        figsize: Tuple[float, float] = (10, 6),
        title: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot collision strength vs energy.

        Parameters
        ----------
        energy : np.ndarray
            Energy grid in eV.
        collision_strength : np.ndarray
            Collision strength (dimensionless).
        ax : Axes, optional
            Matplotlib axes to plot on.
        figsize : tuple
            Figure size.
        title : str, optional
            Plot title.
        **kwargs
            Additional plot styling arguments.

        Returns
        -------
        fig, ax : Figure, Axes
            Matplotlib figure and axes objects.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        default_kwargs = {"linewidth": 2, "color": "green", "alpha": 0.8}
        default_kwargs.update(kwargs)

        ax.plot(energy, collision_strength, **default_kwargs)

        ax.set_xlabel("Electron Energy (eV)", fontsize=12)
        ax.set_ylabel("Collision Strength Ω", fontsize=12)
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title("Collision Strength", fontsize=14)
        ax.grid(True, alpha=0.3, linestyle="--")

        return fig, ax

    def plot_comparison_subplots(
        self,
        energy: np.ndarray,
        cross_section: np.ndarray,
        collision_strength: np.ndarray,
        resonance_info: Optional[Dict] = None,
        figsize: Tuple[float, float] = (14, 10),
        suptitle: Optional[str] = None,
    ) -> Tuple[Figure, np.ndarray]:
        """
        Create a multi-panel comparison plot.

        Creates a figure with:
        - Top: Cross section
        - Middle: Collision strength
        - Bottom: Individual resonance contributions (if resonance_info provided)

        Parameters
        ----------
        energy : np.ndarray
            Energy grid in eV.
        cross_section : np.ndarray
            Cross section in cm².
        collision_strength : np.ndarray
            Collision strength (dimensionless).
        resonance_info : dict, optional
            Resonance contribution information.
        figsize : tuple
            Figure size.
        suptitle : str, optional
            Overall figure title.

        Returns
        -------
        fig : Figure
            Matplotlib figure object.
        axes : np.ndarray
            Array of axes objects.
        """
        n_panels = 3 if resonance_info else 2
        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)

        # Cross section panel
        self.plot_cross_section(
            energy, cross_section, ax=axes[0], title="Cross Section"
        )

        # Collision strength panel
        self.plot_collision_strength(
            energy, collision_strength, ax=axes[1], title="Collision Strength"
        )

        # Resonance contributions panel
        if resonance_info:
            self.plot_resonance_contributions(
                energy,
                cross_section,
                resonance_info,
                ax=axes[2],
                num_resonances=5,
                title="Top 5 Resonance Contributions",
            )

        if suptitle:
            fig.suptitle(suptitle, fontsize=16, y=0.995)

        plt.tight_layout()

        return fig, axes

    def plot_energy_level_diagram(
        self,
        levels: pd.DataFrame,
        initial_level_index: int,
        final_level_index: int,
        resonant_level_indices: Optional[List[int]] = None,
        ax: Optional[Axes] = None,
        figsize: Tuple[float, float] = (8, 10),
        max_resonances: int = 10,
    ) -> Tuple[Figure, Axes]:
        """
        Plot energy level diagram showing initial, final, and resonant states.

        Parameters
        ----------
        levels : pd.DataFrame
            Levels data with 'level_index', 'energy', 'configuration' columns.
        initial_level_index : int
            Initial state level index.
        final_level_index : int
            Final state level index.
        resonant_level_indices : list, optional
            List of resonant level indices to show.
        ax : Axes, optional
            Matplotlib axes to plot on.
        figsize : tuple
            Figure size.
        max_resonances : int
            Maximum number of resonances to show.

        Returns
        -------
        fig, ax : Figure, Axes
            Matplotlib figure and axes objects.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Get level data
        initial = levels[levels["level_index"] == initial_level_index].iloc[0]
        final = levels[levels["level_index"] == final_level_index].iloc[0]

        # Plot initial state
        ax.hlines(
            initial["energy"], 0, 0.3, colors="blue", linewidth=3, label="Initial"
        )
        ax.text(
            0.32,
            initial["energy"],
            f"{initial['configuration']}",
            va="center",
            fontsize=10,
            color="blue",
        )

        # Plot final state
        ax.hlines(final["energy"], 0.7, 1.0, colors="green", linewidth=3, label="Final")
        ax.text(
            0.68,
            final["energy"],
            f"{final['configuration']}",
            va="center",
            ha="right",
            fontsize=10,
            color="green",
        )

        # Plot resonances
        if resonant_level_indices:
            resonances = levels[
                levels["level_index"].isin(resonant_level_indices[:max_resonances])
            ]

            for i, (_, res) in enumerate(resonances.iterrows()):
                y = res["energy"]
                x_start = 0.4
                x_end = 0.6

                ax.hlines(y, x_start, x_end, colors="red", linewidth=1.5, alpha=0.6)
                if i < 5:  # Only label first 5
                    ax.text(
                        x_end + 0.02,
                        y,
                        f"Res {i+1}",
                        va="center",
                        fontsize=8,
                        color="red",
                        alpha=0.7,
                    )

        # Draw transition arrows
        # Initial -> resonance (capture)
        if resonant_level_indices:
            res_energy = levels[
                levels["level_index"] == resonant_level_indices[0]
            ].iloc[0]["energy"]
            ax.annotate(
                "",
                xy=(0.5, res_energy),
                xytext=(0.15, initial["energy"]),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="orange", alpha=0.5),
            )
            ax.text(
                0.25,
                (initial["energy"] + res_energy) / 2,
                "capture",
                fontsize=9,
                color="orange",
                rotation=60,
                alpha=0.7,
            )

            # Resonance -> final (autoionization)
            ax.annotate(
                "",
                xy=(0.85, final["energy"]),
                xytext=(0.5, res_energy),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="purple", alpha=0.5),
            )
            ax.text(
                0.65,
                (final["energy"] + res_energy) / 2,
                "autoionization",
                fontsize=9,
                color="purple",
                rotation=-60,
                alpha=0.7,
            )

        ax.set_ylabel("Energy (eV)", fontsize=12)
        ax.set_title("Energy Level Diagram", fontsize=14)
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks([])
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

        return fig, ax


def quick_plot_cross_section(
    energy: np.ndarray,
    cross_section: np.ndarray,
    resonance_info: Optional[Dict] = None,
    show_resonances: bool = True,
    num_resonances: int = 5,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """
    Quick plotting function for cross sections with resonance markers.

    Parameters
    ----------
    energy : np.ndarray
        Energy grid in eV.
    cross_section : np.ndarray
        Cross section in cm².
    resonance_info : dict, optional
        Resonance contribution information.
    show_resonances : bool
        Whether to show individual resonance contributions.
    num_resonances : int
        Number of resonances to show if show_resonances is True.
    figsize : tuple
        Figure size.
    save_path : str, optional
        If provided, saves figure to this path.

    Returns
    -------
    fig, ax : Figure, Axes
        Matplotlib figure and axes objects.
    """
    plotter = ResonancePlotter()

    if show_resonances and resonance_info:
        fig, ax = plotter.plot_resonance_contributions(
            energy,
            cross_section,
            resonance_info,
            num_resonances=num_resonances,
            figsize=figsize,
        )
    else:
        fig, ax = plotter.plot_cross_section(
            energy,
            cross_section,
            figsize=figsize,
            title="Resonant Excitation Cross Section",
        )

        # Add resonance markers if info provided
        if resonance_info:
            plotter.add_resonance_markers(
                ax,
                resonance_info["energies"][:num_resonances],
                (
                    resonance_info["widths"][:num_resonances]
                    if "widths" in resonance_info
                    else None
                ),
                marker_style="vlines",
            )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig, ax
