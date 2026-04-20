"""Plotting functions for evaluation results."""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_rg(
    rg_series: np.ndarray,
    rg_ref: float,
    out_path: Optional[Path] = None,
    dpi: int = 200,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot radius of gyration vs frame."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(rg_series))
    ax.plot(x, rg_series, "b-", linewidth=1.5, label="Rg(t)")
    ax.axhline(rg_ref, color="k", linestyle="--", linewidth=1.0, label="Reference")

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Rg (Å)", fontsize=12)
    ax.set_title(title or "Radius of Gyration", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_delta_rg(
    delta_rg_series: np.ndarray,
    out_path: Optional[Path] = None,
    dpi: int = 200,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot ΔRg vs frame."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(delta_rg_series))
    ax.plot(x, delta_rg_series, "b-", linewidth=1.5)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("ΔRg (Å)", fontsize=12)
    ax.set_title(title or "ΔRg vs Reference", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_rmsd(
    rmsd_series: np.ndarray,
    out_path: Optional[Path] = None,
    dpi: int = 200,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot RMSD vs frame."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(rmsd_series))
    ax.plot(x, rmsd_series, "r-", linewidth=1.5)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("RMSD (Å)", fontsize=12)
    ax.set_title(title or "Cα RMSD to Reference", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_drmsd(
    drmsd_series: np.ndarray,
    out_path: Optional[Path] = None,
    dpi: int = 200,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot dRMSD vs frame."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(drmsd_series))
    ax.plot(x, drmsd_series, "purple", linewidth=1.5)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("dRMSD (Å)", fontsize=12)
    ax.set_title(title or "Pairwise Distance RMSD", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_q(
    q_series: np.ndarray,
    title: str = "Q (Native Contacts)",
    color: str = "g",
    out_path: Optional[Path] = None,
    dpi: int = 200,
    subtitle: Optional[str] = None,
) -> plt.Figure:
    """Plot Q vs frame."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(q_series))
    ax.plot(x, q_series, f"{color}-", linewidth=1.5)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Q", fontsize=12)
    full_title = title if subtitle is None else f"{title} {subtitle}"
    ax.set_title(full_title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_q_comparison(
    q_hard_series: np.ndarray,
    q_smooth_series: np.ndarray,
    out_path: Optional[Path] = None,
    dpi: int = 200,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot Q_hard and Q_smooth together."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(q_hard_series))
    ax.plot(x, q_hard_series, "b-", linewidth=1.5, label="Q_hard")
    ax.plot(x, q_smooth_series, "r-", linewidth=1.5, label="Q_smooth")
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Q", fontsize=12)
    ax.set_title(title or "Native Contact Fraction", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_min_distance(
    min_series: np.ndarray,
    median_series: np.ndarray,
    clash_threshold: float = 3.8,
    out_path: Optional[Path] = None,
    dpi: int = 200,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot min and median nonbonded distances."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(min_series))
    ax.plot(x, min_series, "r-", linewidth=1.5, label="Min")
    ax.plot(x, median_series, "b-", linewidth=1.5, label="Median")
    ax.axhline(
        clash_threshold,
        color="k",
        linestyle="--",
        linewidth=1.0,
        label=f"Clash ({clash_threshold:.1f} Å)",
    )

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Distance (Å)", fontsize=12)
    ax.set_title(title or "Nonbonded Distances", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_rdf(
    centers: np.ndarray,
    counts: np.ndarray,
    norm: np.ndarray,
    out_dir: Optional[Path] = None,
    dpi: int = 200,
) -> Dict[str, plt.Figure]:
    """Plot RDF in various forms."""
    figs = {}

    # Raw counts
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(centers, counts, width=centers[1] - centers[0], alpha=0.7)
    ax1.set_xlabel("r (Å)", fontsize=12)
    ax1.set_ylabel("Counts", fontsize=12)
    ax1.set_title("RDF - Raw Counts", fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    figs["counts"] = fig1

    # Normalized
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(centers, norm, "b-", linewidth=2)
    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("r (Å)", fontsize=12)
    ax2.set_ylabel("g(r)", fontsize=12)
    ax2.set_title("RDF - Normalized", fontsize=14)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    figs["norm"] = fig2

    if out_dir:
        fig1.savefig(out_dir / "rdf_counts.png", dpi=dpi)
        fig2.savefig(out_dir / "rdf_norm.png", dpi=dpi)
        plt.close(fig1)
        plt.close(fig2)

    return figs


def plot_all(
    report: "EvaluationReport",
    out_dir: Path,
    dpi: int = 200,
    burnin_steps: int = 0,
) -> None:
    """Generate all plots for an evaluation report, optionally after burn-in steps."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get metadata
    n_frames = len(report.rmsd_series)
    save_every = report.metadata.get("save_every", 10000)

    # Convert steps to frames
    if burnin_steps > 0 and save_every > 0:
        burnin_frames = burnin_steps // save_every
    else:
        burnin_frames = 0

    # Apply burn-in if specified
    if burnin_frames > 0 and burnin_frames < n_frames:
        plot_slice = slice(burnin_frames, None)
        burnin_note = f"(after {burnin_steps} steps burn-in)"
    else:
        plot_slice = slice(None)
        burnin_note = ""

    # Rg plots with burn-in
    plot_rg(
        report.rg_series[plot_slice],
        report.rg_ref,
        out_dir / "rg.png",
        dpi,
        title=f"Radius of Gyration {burnin_note}".strip(),
    )

    plot_delta_rg(
        report.delta_rg_series[plot_slice],
        out_dir / "delta_rg.png",
        dpi,
        title=f"ΔRg vs Reference {burnin_note}".strip(),
    )

    # RMSD and dRMSD with burn-in
    plot_rmsd(
        report.rmsd_series[plot_slice],
        out_dir / "rmsd.png",
        dpi,
        title=f"Cα RMSD to Reference {burnin_note}".strip(),
    )

    if hasattr(report, "drmsd_series") and report.drmsd_series.size > 0:
        plot_drmsd(
            report.drmsd_series[plot_slice],
            out_dir / "drmsd.png",
            dpi,
            title=f"Pairwise Distance RMSD {burnin_note}".strip(),
        )

    # Q plots with burn-in
    if report.q_hard_series.size > 0:
        plot_q(
            report.q_hard_series[plot_slice],
            "Q_hard",
            "b",
            out_dir / "q_hard.png",
            dpi,
            subtitle=burnin_note,
        )

    if report.q_smooth_series.size > 0:
        plot_q(
            report.q_smooth_series[plot_slice],
            "Q_smooth",
            "r",
            out_dir / "q_smooth.png",
            dpi,
            subtitle=burnin_note,
        )

    if report.q_hard_series.size > 0 and report.q_smooth_series.size > 0:
        plot_q_comparison(
            report.q_hard_series[plot_slice],
            report.q_smooth_series[plot_slice],
            out_dir / "q_comparison.png",
            dpi,
            title=f"Native Contact Fraction {burnin_note}".strip(),
        )

    # Min distance plots with burn-in
    if hasattr(report, "median_distances_series") and report.median_distances_series.size > 0:
        plot_min_distance(
            report.min_distances_series[plot_slice],
            report.median_distances_series[plot_slice],
            clash_threshold=3.8,
            out_path=out_dir / "min_distance.png",
            dpi=dpi,
            title=f"Nonbonded Distances {burnin_note}".strip(),
        )
    else:
        # Fallback for backward compatibility
        plot_min_distance(
            report.min_distances_series[plot_slice],
            report.min_distances_series[plot_slice],
            clash_threshold=3.8,
            out_path=out_dir / "min_distance.png",
            dpi=dpi,
            title=f"Nonbonded Distances {burnin_note}".strip(),
        )

    # RDF (already averaged over all frames - doesn't need burn-in)
    plot_rdf(report.rdf_centers, report.rdf_counts, report.rdf_norm, out_dir, dpi)
