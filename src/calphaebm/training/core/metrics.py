"""Validation metrics and correlation calculations."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from calphaebm.utils.logging import get_logger

logger = get_logger()


def compute_ramachandran_correlation(
    theta: torch.Tensor, phi: torch.Tensor, ref_path: str = "analysis/backbone_geometry/data/figure_3a_histogram.npy"
) -> float:
    """Compute correlation between current (θ,φ) distribution and PDB reference."""
    try:
        # Load reference Ramachandran distribution
        ref_path = Path(ref_path)
        if not ref_path.exists():
            logger.warning(f"Reference Ramachandran distribution not found at {ref_path}")
            return 0.0

        ref_hist = np.load(ref_path)  # Shape: [24, 36]

        # Convert angles to degrees
        theta_deg = theta.cpu().numpy().flatten() * 180 / np.pi
        phi_deg = phi.cpu().numpy().flatten() * 180 / np.pi

        # Create histogram with same bins as reference
        n_theta_bins, n_phi_bins = ref_hist.shape
        theta_bins = np.linspace(50, 180, n_theta_bins + 1)
        phi_bins = np.linspace(-180, 180, n_phi_bins + 1)

        current_hist, _, _ = np.histogram2d(theta_deg, phi_deg, bins=[theta_bins, phi_bins])

        # Normalize
        current_hist = current_hist + 1e-6
        current_hist = current_hist / current_hist.sum()
        ref_hist_norm = ref_hist / ref_hist.sum()

        # Compute correlation
        correlation = np.corrcoef(current_hist.flatten(), ref_hist_norm.flatten())[0, 1]

        return float(max(0, correlation))

    except Exception as e:
        logger.warning(f"Failed to compute Ramachandran correlation: {e}")
        return 0.0


def compute_delta_phi_correlation(phi: torch.Tensor, ref_dir: str = "analysis/backbone_geometry/data") -> float:
    """Compute correlation between current Δφ distribution and PDB reference."""
    try:
        # Load reference Δφ distribution
        ref_path = Path(ref_dir) / "delta_phi_energy.npy"
        centers_path = Path(ref_dir) / "delta_phi_centers.npy"

        if not ref_path.exists() or not centers_path.exists():
            logger.warning(f"Reference Δφ distribution not found in {ref_dir}")
            return 0.0

        ref_energy = np.load(ref_path)
        ref_centers = np.load(centers_path)

        # Convert energy to probability (Boltzmann inversion)
        ref_prob = np.exp(-ref_energy)
        ref_prob = ref_prob / ref_prob.sum()

        # Compute Δφ from torsion angles
        phi_deg = (phi.cpu().numpy() * 180 / np.pi).flatten()
        delta_phi = np.diff(phi_deg)

        # Wrap to [-180, 180]
        delta_phi = (delta_phi + 180) % 360 - 180

        # Create histogram of current Δφ
        current_hist, _ = np.histogram(delta_phi, bins=len(ref_centers), range=(-180, 180))
        current_hist = current_hist + 1e-6
        current_hist = current_hist / current_hist.sum()

        # Compute correlation
        correlation = np.corrcoef(current_hist, ref_prob)[0, 1]

        return float(max(0, correlation))

    except Exception as e:
        logger.warning(f"Failed to compute Δφ correlation: {e}")
        return 0.0


def compute_bond_metrics(bond_lengths: np.ndarray) -> tuple:
    """Compute bond length statistics."""
    bond_mean = float(np.mean(bond_lengths)) if len(bond_lengths) > 0 else 0.0
    bond_std = float(np.std(bond_lengths)) if len(bond_lengths) > 0 else 0.0
    bond_rmsd = float(np.sqrt(np.mean((bond_lengths - 3.8) ** 2))) if len(bond_lengths) > 0 else 0.0

    return bond_mean, bond_std, bond_rmsd
