"""Correlation functions for validation metrics.

Changes from previous version:
- FIX: wrong reference filename figure_3a_histogram.npy → theta_phi_hist.npy
- FIX: hardcoded bin ranges replaced with actual theta_edges_deg.npy / phi_edges_deg.npy
- FIX: removed max(0, correlation) clipping — negative correlations are diagnostic signal
- PERF: reference arrays cached on first load (previously re-read from disk every call)
- ROBUSTNESS: delta_phi reference also cached; both caches are module-level dicts
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from calphaebm.utils.logging import get_logger

logger = get_logger()

_DATA_DIR = Path("analysis/backbone_geometry/data")

# Module-level caches — populated on first call, never reloaded
_rama_cache: dict = {}
_dphi_cache: dict = {}


# ---------------------------------------------------------------------------
# Cache loaders
# ---------------------------------------------------------------------------


def _load_rama_ref() -> Optional[tuple]:
    """Load and cache Ramachandran reference data.

    Returns (ref_hist, theta_edges, phi_edges) or None on failure.
    """
    if "data" in _rama_cache:
        return _rama_cache["data"]

    hist_path = _DATA_DIR / "theta_phi_hist.npy"
    theta_path = _DATA_DIR / "theta_edges_deg.npy"
    phi_path = _DATA_DIR / "phi_edges_deg.npy"

    if not hist_path.exists():
        logger.warning(
            "Reference Ramachandran histogram not found at %s — "
            "Ramachandran correlation will be 0.0 until the file is present.",
            hist_path,
        )
        _rama_cache["data"] = None
        return None

    try:
        ref_hist = np.load(hist_path).astype(np.float64)
        theta_edges = np.load(theta_path) if theta_path.exists() else None
        phi_edges = np.load(phi_path) if phi_path.exists() else None
        _rama_cache["data"] = (ref_hist, theta_edges, phi_edges)
        logger.info(
            "Loaded Ramachandran reference: shape=%s  " "theta_edges=%s  phi_edges=%s",
            ref_hist.shape,
            theta_edges.shape if theta_edges is not None else "inferred",
            phi_edges.shape if phi_edges is not None else "inferred",
        )
        return _rama_cache["data"]
    except Exception as exc:
        logger.warning("Failed to load Ramachandran reference: %s", exc)
        _rama_cache["data"] = None
        return None


def _load_dphi_ref() -> Optional[tuple]:
    """Load and cache Δφ reference data.

    Returns (ref_prob, ref_centers) or None on failure.
    """
    if "data" in _dphi_cache:
        return _dphi_cache["data"]

    energy_path = _DATA_DIR / "delta_phi_energy.npy"
    centers_path = _DATA_DIR / "delta_phi_centers.npy"

    if not energy_path.exists() or not centers_path.exists():
        logger.warning(
            "Reference Δφ files not found in %s — Δφ correlation will be 0.0.",
            _DATA_DIR,
        )
        _dphi_cache["data"] = None
        return None

    try:
        ref_energy = np.load(energy_path).astype(np.float64)
        ref_centers = np.load(centers_path).astype(np.float64)
        ref_prob = np.exp(-ref_energy)
        ref_prob /= ref_prob.sum()
        _dphi_cache["data"] = (ref_prob, ref_centers)
        logger.info(
            "Loaded Δφ reference: %d bins, center range [%.1f, %.1f]°",
            len(ref_centers),
            ref_centers.min(),
            ref_centers.max(),
        )
        return _dphi_cache["data"]
    except Exception as exc:
        logger.warning("Failed to load Δφ reference: %s", exc)
        _dphi_cache["data"] = None
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ramachandran_correlation(
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> float:
    """Compute Pearson correlation between generated (θ, φ) distribution and PDB reference.

    Returns a value in [-1, 1].  Negative values indicate anti-correlation with
    the PDB reference — a meaningful diagnostic (do not clip to 0).

    Args:
        theta: Bond-angle tensor (any shape), values in radians.
        phi:   Torsion-angle tensor (any shape), values in radians.

    Returns:
        Pearson r, or 0.0 if reference data is unavailable or inputs are empty.
    """
    ref = _load_rama_ref()
    if ref is None:
        return 0.0

    ref_hist, theta_edges, phi_edges = ref

    try:
        theta_deg = theta.cpu().numpy().flatten() * (180.0 / np.pi)
        phi_deg = phi.cpu().numpy().flatten() * (
            -180.0 / np.pi
        )  # sign flip: matches secondary.py convention used when building reference

        valid = ~(np.isnan(theta_deg) | np.isnan(phi_deg))
        theta_deg = theta_deg[valid]
        phi_deg = phi_deg[valid]

        if theta_deg.size == 0:
            return 0.0

        # Build bin edges — use files when available, otherwise infer from hist shape
        if theta_edges is not None and phi_edges is not None:
            t_bins = theta_edges
            p_bins = phi_edges
        else:
            n_t, n_p = ref_hist.shape
            t_bins = np.linspace(float(theta_deg.min()), float(theta_deg.max()), n_t + 1)
            p_bins = np.linspace(-180.0, 180.0, n_p + 1)

        current_hist, _, _ = np.histogram2d(theta_deg, phi_deg, bins=[t_bins, p_bins])

        # Normalise with small epsilon to avoid zero-division
        cur_flat = current_hist.flatten() + 1e-8
        cur_flat /= cur_flat.sum()
        ref_flat = ref_hist.flatten() + 1e-8
        ref_flat /= ref_flat.sum()

        corr = float(np.corrcoef(cur_flat, ref_flat)[0, 1])
        # Return raw value — caller decides how to interpret sign
        return corr if np.isfinite(corr) else 0.0

    except Exception as exc:
        logger.warning("compute_ramachandran_correlation failed: %s", exc)
        return 0.0


def compute_delta_phi_correlation(phi: torch.Tensor) -> float:
    """Compute Pearson correlation between generated Δφ distribution and PDB reference.

    Returns a value in [-1, 1].  Negative values are not clipped.

    Args:
        phi: Torsion-angle tensor (any shape), values in radians.

    Returns:
        Pearson r, or 0.0 if reference data is unavailable or inputs are empty.
    """
    ref = _load_dphi_ref()
    if ref is None:
        return 0.0

    ref_prob, ref_centers = ref

    try:
        phi_deg = phi.cpu().numpy().flatten() * (180.0 / np.pi)
        delta_phi = np.diff(phi_deg)
        delta_phi = (delta_phi + 180.0) % 360.0 - 180.0  # wrap to (-180, 180]

        if delta_phi.size == 0:
            return 0.0

        n_bins = len(ref_centers)
        current_hist, _ = np.histogram(delta_phi, bins=n_bins, range=(-180.0, 180.0))
        cur_norm = current_hist.astype(np.float64) + 1e-8
        cur_norm /= cur_norm.sum()

        corr = float(np.corrcoef(cur_norm, ref_prob)[0, 1])
        return corr if np.isfinite(corr) else 0.0

    except Exception as exc:
        logger.warning("compute_delta_phi_correlation failed: %s", exc)
        return 0.0


def clear_reference_cache() -> None:
    """Force reference data to be reloaded on next call.  Useful in tests."""
    _rama_cache.clear()
    _dphi_cache.clear()
