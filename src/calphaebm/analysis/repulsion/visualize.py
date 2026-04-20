"""Convenience visualization wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CONTACT_BINS
from .plots import plot_enrichment_matrices


def visualize_enrichment_bundle(data_dir: Path, bin_name: str, out_dir: Path) -> None:
    """
    Load enrichment data for a specific bin and generate plots.

    Args:
        data_dir: Directory containing the analysis output files
        bin_name: Name of the contact bin (tight, medium, loose)
        out_dir: Output directory for plots
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try new naming convention first, fall back to old
    try:
        oe = np.load(data_dir / f"oe_{bin_name}.npy")
        log_oe = np.load(data_dir / f"log_oe_{bin_name}.npy")
        z = np.load(data_dir / f"z_{bin_name}.npy")
        q = np.load(data_dir / f"q_{bin_name}.npy")
    except FileNotFoundError:
        # Fall back to old naming convention
        oe = np.load(data_dir / f"OE_{bin_name}.npy")
        log_oe = np.log(np.maximum(oe, 1e-12))
        z = np.load(data_dir / f"Z_{bin_name}.npy") if (data_dir / f"Z_{bin_name}.npy").exists() else None
        q = np.load(data_dir / f"Q_{bin_name}.npy") if (data_dir / f"Q_{bin_name}.npy").exists() else None

    # Create dictionaries for the plot function
    oe_dict = {bin_name: oe}
    log_oe_dict = {bin_name: log_oe}
    z_dict = {bin_name: z if z is not None else np.zeros_like(oe)}
    q_dict = {bin_name: q if q is not None else np.ones_like(oe)}

    # Generate plots
    plot_enrichment_matrices(
        oe=oe_dict,
        log_oe=log_oe_dict,
        z=z_dict,
        q=q_dict,
        contact_bins={bin_name: CONTACT_BINS[bin_name]},
        out_dir=out_dir,
    )
