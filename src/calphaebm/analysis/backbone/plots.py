# src/calphaebm/analysis/backbone/plots.py

"""Plotting + data-product functions for backbone geometry analysis.

This module writes canonical, pipeline-centric filenames (NOT paper figure names).

Written artifacts (data_dir):
- theta_edges_deg.npy
- phi_edges_deg.npy
- theta_phi_energy.npy              # 2D energy surface E(θ,φ) = -log P(θ,φ)
- theta_phi_hist.npy                # 2D probability density P(θ,φ) (for validation Rama correlation)

- theta_i_edges_deg.npy
- theta_ip1_edges_deg.npy
- theta_theta_energy.npy            # 2D energy surface E(θ_i, θ_{i+1})

- phi_i_edges_deg.npy
- phi_ip1_edges_deg.npy
- phi_phi_energy.npy                # 2D energy surface E(φ_i, φ_{i+1})

Δφ persistence artifacts:
- delta_phi_energy.npy
- delta_phi_centers.npy

Notes
-----
- Energies are shifted so min(E)=0 for convenience.
- For φ dimensions we treat the axis as periodic when smoothing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from calphaebm.utils.logging import get_logger

from .config import (
    BOND_LENGTH_IDEAL,
    BOND_LENGTH_MAX,
    BOND_LENGTH_MIN,
    DELTA_PHI_BINS,
    DELTA_PHI_EPS,
    MAX_POINTS_FOR_KDE,
    PHI_BINS,
    PHI_MAX,
    PHI_MIN,
    PLOT_DPI,
    THETA_BINS,
    THETA_MAX,
    THETA_MIN,
)

logger = get_logger()

# Small pseudocount for 2D histograms to avoid log(0)
_EPS_2D = 1e-6

# Smoothing (bins). Keep modest to avoid washing out basins.
_SMOOTH_SIGMA_2D = 1.0


def _maybe_write_csv(df, path: Path) -> None:
    """Write CSV with warnings on real I/O errors."""
    try:
        df.to_csv(path, index=False)
    except OSError as e:
        logger.warning(f"Failed to write CSV to {path}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error writing CSV to {path}: {e}")


def _hist2d_to_energy(
    x: np.ndarray,
    y: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    smooth_sigma: float = _SMOOTH_SIGMA_2D,
    periodic_y: bool = False,
    eps: float = _EPS_2D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2D density histogram and convert to an energy surface E=-log P.

    Returns:
        energy: (H, W) float32
        hist:   (H, W) float32 — smoothed probability density
        x_edges: (H+1,) float32
        y_edges: (W+1,) float32
    """
    hist, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges], density=True)
    hist = hist.astype(np.float32)

    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            mode_y = "wrap" if periodic_y else "nearest"
            hist = gaussian_filter(hist, sigma=(smooth_sigma, smooth_sigma), mode=("nearest", mode_y)).astype(
                np.float32
            )
        except ImportError:
            logger.warning("scipy not available, skipping 2D smoothing")

    energy = -np.log(hist + float(eps)).astype(np.float32)
    energy -= float(np.min(energy))
    return energy, hist, xedges.astype(np.float32), yedges.astype(np.float32)


def plot_bond_length_distribution(
    bond_lengths_all: np.ndarray,
    output_dir: Path,
    data_dir: Path,
    save_figures: bool = True,
):
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd

        _maybe_write_csv(pd.DataFrame({"bond_length": bond_lengths_all}), data_dir / "bond_lengths_data.csv")
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(BOND_LENGTH_MIN, BOND_LENGTH_MAX + 1e-6, 0.05)

    ax.hist(bond_lengths_all, bins=bins, density=True, alpha=0.7, edgecolor="black", label="Histogram")

    kde = gaussian_kde(bond_lengths_all)
    x_kde = np.linspace(BOND_LENGTH_MIN, BOND_LENGTH_MAX, 1000)
    ax.plot(x_kde, kde(x_kde), linewidth=2, label="KDE")

    ax.set_xlim(BOND_LENGTH_MIN, BOND_LENGTH_MAX)
    ax.set_xlabel("Cα-Cα Bond Length (Å)", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title("Distribution of Cα-Cα Bond Lengths", fontsize=14)
    ax.axvline(BOND_LENGTH_IDEAL, linestyle="--", alpha=0.6, label=f"Ideal ({BOND_LENGTH_IDEAL} Å)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_figures:
        plt.savefig(output_dir / "bond_length_distribution.png", dpi=PLOT_DPI)
        plt.savefig(output_dir / "bond_length_distribution.pdf")
    plt.close(fig)


def plot_figure_2(
    theta_angles: np.ndarray,
    phi_angles: np.ndarray,
    output_dir: Path,
    data_dir: Path,
    save_figures: bool = True,
):
    """
    Simple θ and φ 1D distributions (plot-only). Data products are written elsewhere.
    """
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd

        _maybe_write_csv(pd.DataFrame({"theta_deg": theta_angles}), data_dir / "theta_angles_deg.csv")
        _maybe_write_csv(pd.DataFrame({"phi_deg": phi_angles}), data_dir / "phi_angles_deg.csv")
    except ImportError:
        pass

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    bins_theta = np.linspace(THETA_MIN, THETA_MAX, THETA_BINS + 1)
    ax1.hist(theta_angles, bins=bins_theta, density=True, alpha=0.7, edgecolor="black", label="Histogram")
    kde_theta = gaussian_kde(theta_angles)
    x_theta = np.linspace(THETA_MIN, THETA_MAX, 500)
    ax1.plot(x_theta, kde_theta(x_theta), linewidth=2, label="KDE")
    ax1.set_xlim(THETA_MIN, THETA_MAX)
    ax1.set_xlabel(r"$\theta$ (degrees)", fontsize=10)
    ax1.set_ylabel("Probability density", fontsize=10)
    ax1.set_title("Distribution of pseudo-bond angles θ", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    bins_phi = np.linspace(PHI_MIN, PHI_MAX, PHI_BINS + 1)
    ax2.hist(phi_angles, bins=bins_phi, density=True, alpha=0.7, edgecolor="black", label="Histogram")
    kde_phi = gaussian_kde(phi_angles)
    x_phi = np.linspace(PHI_MIN, PHI_MAX, 1000)
    ax2.plot(x_phi, kde_phi(x_phi), linewidth=2, label="KDE")
    ax2.set_xlim(PHI_MIN, PHI_MAX)
    ax2.set_xlabel(r"$\phi$ (degrees)", fontsize=10)
    ax2.set_ylabel("Probability density", fontsize=10)
    ax2.set_title("Distribution of pseudotorsion angles φ", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Distribution of Cα Backbone Angles", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_figures:
        plt.savefig(output_dir / "theta_phi_distributions.png", dpi=PLOT_DPI)
        plt.savefig(output_dir / "theta_phi_distributions.pdf")
    plt.close(fig)


def _kde_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    max_points: int = MAX_POINTS_FOR_KDE,
    seed: int = 42,
):
    """Density-colored scatter with reproducible subsampling."""
    if len(x) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), max_points, replace=False)
        x_plot = x[idx]
        y_plot = y[idx]
    else:
        x_plot = x
        y_plot = y

    xy = np.vstack([x_plot, y_plot])
    z = gaussian_kde(xy)(xy)
    order = z.argsort()
    sc = ax.scatter(x_plot[order], y_plot[order], c=z[order], s=1, cmap="viridis", alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    return sc


def plot_figure_3(
    theta_i: np.ndarray,
    theta_ip1: np.ndarray,
    phi: np.ndarray,
    output_dir: Path,
    data_dir: Path,
    save_figures: bool = True,
):
    """
    Write the 2D backbone energy surfaces used downstream.

    Always writes .npy artifacts, even if save_figures=False.
    """
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    theta_edges = np.linspace(THETA_MIN, THETA_MAX, THETA_BINS + 1, dtype=np.float32)
    phi_edges = np.linspace(PHI_MIN, PHI_MAX, PHI_BINS + 1, dtype=np.float32)

    # --- θ–φ energy (Ramachandran-like) ---
    theta_phi_energy, theta_phi_hist, xedges, yedges = _hist2d_to_energy(
        theta_i,
        phi,
        theta_edges,
        phi_edges,
        periodic_y=True,  # φ periodic
    )
    np.save(data_dir / "theta_edges_deg.npy", xedges)
    np.save(data_dir / "phi_edges_deg.npy", yedges)
    np.save(data_dir / "theta_phi_energy.npy", theta_phi_energy)
    np.save(data_dir / "theta_phi_hist.npy", theta_phi_hist)

    # --- θ_i – θ_{i+1} energy ---
    theta_theta_energy, _, x_tt, y_tt = _hist2d_to_energy(
        theta_i,
        theta_ip1,
        theta_edges,
        theta_edges,
        periodic_y=False,
    )
    np.save(data_dir / "theta_i_edges_deg.npy", x_tt)
    np.save(data_dir / "theta_ip1_edges_deg.npy", y_tt)
    np.save(data_dir / "theta_theta_energy.npy", theta_theta_energy)

    # --- φ_i – φ_{i+1} energy (requires paired torsions) ---
    if len(phi) >= 2:
        phi_i = phi[:-1]
        phi_ip1 = phi[1:]
        phi_phi_energy, _, x_pp, y_pp = _hist2d_to_energy(
            phi_i,
            phi_ip1,
            phi_edges,
            phi_edges,
            periodic_y=True,  # φ periodic
        )
        np.save(data_dir / "phi_i_edges_deg.npy", x_pp)
        np.save(data_dir / "phi_ip1_edges_deg.npy", y_pp)
        np.save(data_dir / "phi_phi_energy.npy", phi_phi_energy)

    if not save_figures:
        return

    # Plots (diagnostic only)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sc1 = _kde_scatter(
        axes[0],
        theta_i,
        phi,
        r"$\theta_i$ (deg)",
        r"$\phi$ (deg)",
        "(a) φ vs θ_i",
        (THETA_MIN, THETA_MAX),
        (PHI_MIN, PHI_MAX),
    )
    plt.colorbar(sc1, ax=axes[0], label="Density")

    sc2 = _kde_scatter(
        axes[1],
        theta_i,
        theta_ip1,
        r"$\theta_i$ (deg)",
        r"$\theta_{i+1}$ (deg)",
        "(b) θ_i vs θ_{i+1}",
        (THETA_MIN, THETA_MAX),
        (THETA_MIN, THETA_MAX),
    )
    axes[1].plot([THETA_MIN, THETA_MAX], [THETA_MIN, THETA_MAX], "k--", linewidth=1, alpha=0.5)
    plt.colorbar(sc2, ax=axes[1], label="Density")

    if len(phi) >= 2:
        sc3 = _kde_scatter(
            axes[2],
            phi[:-1],
            phi[1:],
            r"$\phi_i$ (deg)",
            r"$\phi_{i+1}$ (deg)",
            "(c) φ_i vs φ_{i+1}",
            (PHI_MIN, PHI_MAX),
            (PHI_MIN, PHI_MAX),
        )
        axes[2].plot([PHI_MIN, PHI_MAX], [PHI_MIN, PHI_MAX], "k--", linewidth=1, alpha=0.5)
        plt.colorbar(sc3, ax=axes[2], label="Density")
    else:
        axes[2].set_axis_off()

    plt.suptitle("Correlations in Cα Backbone Geometry", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "backbone_correlations.png", dpi=PLOT_DPI)
    plt.savefig(output_dir / "backbone_correlations.pdf")
    plt.close(fig)


def plot_phi_phi_correlation(phi, output_dir: Path, data_dir: Path, save_figures: bool = True):
    """(Optional) plot-only helper; no data products here."""
    if len(phi) < 2 or not save_figures:
        return

    phi_i = phi[:-1]
    phi_ip1 = phi[1:]

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = _kde_scatter(
        ax,
        phi_i,
        phi_ip1,
        r"$\phi_i$ (deg)",
        r"$\phi_{i+1}$ (deg)",
        "φ-φ Correlation",
        (PHI_MIN, PHI_MAX),
        (PHI_MIN, PHI_MAX),
    )
    ax.plot([PHI_MIN, PHI_MAX], [PHI_MIN, PHI_MAX], "r--", linewidth=2, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Density")
    plt.tight_layout()
    plt.savefig(output_dir / "phi_phi_correlation.png", dpi=PLOT_DPI)
    plt.savefig(output_dir / "phi_phi_correlation.pdf")
    plt.close(fig)


def plot_delta_phi_potential(phi, output_dir: Path, data_dir: Path, save_figures: bool = True):
    """Compute and optionally plot Δφ potential (always writes .npy files)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if len(phi) < 2:
        return None, None

    phi_i = phi[:-1]
    phi_ip1 = phi[1:]
    delta_phi = phi_ip1 - phi_i
    delta_phi = np.remainder(delta_phi + 180.0, 360.0) - 180.0

    edges = np.linspace(-180.0, 180.0, DELTA_PHI_BINS + 1)
    hist, edges = np.histogram(delta_phi, bins=edges, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    energy = -np.log(hist + DELTA_PHI_EPS)

    np.save(data_dir / "delta_phi_energy.npy", energy.astype(np.float32))
    np.save(data_dir / "delta_phi_centers.npy", centers.astype(np.float32))

    if not save_figures:
        return centers, energy

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(centers, hist, width=(edges[1] - edges[0]), alpha=0.7, edgecolor="black")
    ax1.set_xlabel(r"$\Delta\phi$ (deg)")
    ax1.set_ylabel("Density")
    ax1.set_title("Δφ Distribution")
    ax1.set_xlim(-180, 180)
    ax1.grid(True, alpha=0.3)

    ax2.plot(centers, energy, linewidth=2)
    ax2.set_xlabel(r"$\Delta\phi$ (deg)")
    ax2.set_ylabel("Energy")
    ax2.set_title(r"Δφ Potential ($E=-\log P$)")
    ax2.set_xlim(-180, 180)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "delta_phi_potential.png", dpi=PLOT_DPI)
    plt.savefig(output_dir / "delta_phi_potential.pdf")
    plt.close(fig)

    return centers, energy
