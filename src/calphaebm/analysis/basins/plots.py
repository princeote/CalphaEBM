# src/calphaebm/analysis/basins/plots.py

"""Plotting functions for basin analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import PLOT_DPI


def plot_cluster_scatter(
    theta: np.ndarray,
    phi: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    basin_names: List[str],
    output_dir: Path,
    max_points: int = 10_000,
    random_state: int = 42,
) -> plt.Figure:
    """
    Create a scatter plot of clustered (θ,φ) points.

    Notes:
      Uses a fixed RNG seed for reproducible subsampling.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # Reproducible subsample
    if len(theta) > max_points:
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(len(theta), int(max_points), replace=False)
        theta_plot = theta[idx]
        phi_plot = phi[idx]
        labels_plot = labels[idx]
    else:
        theta_plot = theta
        phi_plot = phi
        labels_plot = labels

    for k in range(n_clusters):
        mask = labels_plot == k
        ax.scatter(
            theta_plot[mask],
            phi_plot[mask],
            c=[colors[k]],
            alpha=0.25,
            s=2,
            label=f"{basin_names[k]} (k={k})",
        )
        ax.scatter(
            centers[k, 0],
            centers[k, 1],
            c=[colors[k]],
            marker="*",
            s=250,
            edgecolors="black",
            linewidth=1,
            zorder=10,
        )

    ax.set_xlabel("θ (degrees)", fontsize=12)
    ax.set_ylabel("φ (degrees)", fontsize=12)
    ax.set_title(f"Clustering of (θ,φ) Space into {n_clusters} Basins", fontsize=14)
    ax.set_xlim(50, 180)
    ax.set_ylim(-180, 180)
    ax.legend(markerscale=4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "basin_clusters.png", dpi=PLOT_DPI)
    plt.savefig(output_dir / "basin_clusters.pdf")
    plt.close(fig)

    return fig


def plot_basin_histogram(
    energy: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    basin_idx: int,
    basin_name: str,
    output_dir: Path,
) -> plt.Figure:
    """Plot a single basin energy surface."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        energy.T,
        origin="lower",
        extent=[theta_edges[0], theta_edges[-1], phi_edges[0], phi_edges[-1]],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="Energy (shifted)")

    ax.set_xlabel("θ (degrees)", fontsize=12)
    ax.set_ylabel("φ (degrees)", fontsize=12)
    ax.set_title(f"Basin {basin_idx}: {basin_name}", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / f"basin_{basin_idx}_histogram.png", dpi=PLOT_DPI)
    plt.savefig(output_dir / f"basin_{basin_idx}_histogram.pdf")
    plt.close(fig)

    return fig


def plot_basin_comparison(
    energies: List[np.ndarray],
    basin_names: List[str],
    output_dir: Path,
) -> plt.Figure:
    """Create a comparison plot of all basin energy surfaces."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_basins = len(energies)
    fig, axes = plt.subplots(1, n_basins, figsize=(5 * n_basins, 4))
    if n_basins == 1:
        axes = [axes]

    vmin = min(e.min() for e in energies)
    vmax = max(e.max() for e in energies)

    for k, (ax, energy, name) in enumerate(zip(axes, energies, basin_names)):
        im = ax.imshow(
            energy.T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Basin {k}: {name}", fontsize=12)
        ax.set_xlabel("θ bin")
        ax.set_ylabel("φ bin")
        plt.colorbar(im, ax=ax, label="Energy (shifted)")

    plt.tight_layout()
    plt.savefig(output_dir / "basin_comparison.png", dpi=PLOT_DPI)
    plt.savefig(output_dir / "basin_comparison.pdf")
    plt.close(fig)

    return fig
