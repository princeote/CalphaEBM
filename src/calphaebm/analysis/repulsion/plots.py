from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .config import AA_NAMES, PLOT_DPI


def plot_rdf_analysis(
    rdf_edges_A: np.ndarray,
    rdf_centers_A: np.ndarray,
    rdf_counts: np.ndarray,
    g_r: np.ndarray,
    pmf: np.ndarray,
    out_dir: Path,
) -> None:
    """Plot RDF and PMF analysis results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rdf_centers_A, g_r, linewidth=2)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title("RDF (tail-normalized)")
    fig.tight_layout()
    fig.savefig(out_dir / "rdf.png", dpi=PLOT_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rdf_centers_A, pmf, linewidth=2)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("W(r) = -log g(r)")
    ax.set_title("PMF from RDF (dimensionless)")
    fig.tight_layout()
    fig.savefig(out_dir / "pmf.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_repulsive_wall(
    r_dense_A: np.ndarray,
    W_dense: np.ndarray,
    r_star_A: float,
    out_dir: Path,
) -> None:
    """Plot repulsive wall."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r_dense_A, W_dense, linewidth=2)
    ax.axvline(float(r_star_A), linestyle=":", linewidth=1)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("E_rep(r)")
    ax.set_title("Repulsive wall")
    fig.tight_layout()
    fig.savefig(out_dir / "repulsive_wall.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_enrichment_matrices(
    oe: Dict[str, np.ndarray],
    log_oe: Dict[str, np.ndarray],
    z: Dict[str, np.ndarray],
    q: Dict[str, np.ndarray],
    contact_bins: Dict[str, Dict[str, float]],
    out_dir: Path,
) -> None:
    """
    Plot enrichment matrices for each contact bin.

    Args:
        oe: Dictionary of observed/expected matrices per bin
        log_oe: Dictionary of log(observed/expected) matrices per bin
        z: Dictionary of z-score matrices per bin
        q: Dictionary of q-value matrices per bin
        contact_bins: Dictionary of bin definitions
        out_dir: Output directory
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in contact_bins.keys():
        if name not in log_oe:
            continue

        # Plot log(OE)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(log_oe[name], origin="lower", aspect="equal", cmap="RdBu_r")
        ax.set_title(f"log(OE) enrichment: {name}")
        ax.set_xticks(range(20))
        ax.set_yticks(range(20))
        ax.set_xticklabels(AA_NAMES, fontsize=8)
        ax.set_yticklabels(AA_NAMES, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"enrichment_logoe_{name}.png", dpi=PLOT_DPI)
        plt.close(fig)

        # Plot significance (-log10 q)
        if name in q:
            fig, ax = plt.subplots(figsize=(7, 6))
            sig = -np.log10(np.maximum(q[name], 1e-12))
            im = ax.imshow(sig, origin="lower", aspect="equal", cmap="viridis")
            ax.set_title(f"-log10(q) significance: {name}")
            ax.set_xticks(range(20))
            ax.set_yticks(range(20))
            ax.set_xticklabels(AA_NAMES, fontsize=8)
            ax.set_yticklabels(AA_NAMES, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out_dir / f"enrichment_sig_{name}.png", dpi=PLOT_DPI)
            plt.close(fig)
