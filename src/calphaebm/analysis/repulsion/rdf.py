from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from calphaebm.utils.logging import get_logger

from .config import (
    PMF_MIN_G,
    PMF_SMOOTH_SIGMA,
    RDF_TAIL_END_A,
    RDF_TAIL_START_A,
    WALL_DENSE_N,
    WALL_SMOOTH_SIGMA_BINS,
    WALL_SPARSE_N,
)

logger = get_logger()


@dataclass(frozen=True)
class RDFResult:
    """Container for RDF and derived quantities.

    All distances are in Å.
    pmf is dimensionless (kBT units) when computed as -log(g).
    """

    r_edges: np.ndarray  # (nbins+1,)
    r_centers: np.ndarray  # (nbins,)
    counts: np.ndarray  # (nbins,)
    g_r: np.ndarray  # (nbins,)
    pmf: np.ndarray  # (nbins,)
    tail_mean: float  # scalar used to normalize g(r) -> 1 in the tail


def compute_rdf_from_counts(
    counts: np.ndarray,
    r_edges: np.ndarray,
    tail_start: float = RDF_TAIL_START_A,
    tail_end: float = RDF_TAIL_END_A,
    pmf_min_g: float = PMF_MIN_G,
    pmf_smooth_sigma_bins: float = PMF_SMOOTH_SIGMA,
) -> RDFResult:
    """
    Compute a *normalized* RDF g(r) from histogram counts and bin edges.

    We normalize g(r) by forcing mean(g) in the tail window [tail_start, tail_end] to be ~1.
    This avoids having to estimate density explicitly; for our purposes (PMF shape / wall),
    tail normalization is sufficient and stable.

    Args:
        counts: (nbins,) raw pair counts per bin
        r_edges: (nbins+1,) bin edges in Å
        tail_start, tail_end: tail window in Å for normalization
        pmf_min_g: floor for g(r) before log
        pmf_smooth_sigma_bins: optional smoothing on PMF in *bins* (0 disables)

    Returns:
        RDFResult
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    r_edges = np.asarray(r_edges, dtype=np.float64).reshape(-1)
    if r_edges.size != counts.size + 1:
        raise ValueError(f"r_edges must have len(counts)+1. Got {r_edges.size} vs {counts.size}")

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = np.diff(r_edges)

    # "raw" g(r) up to a constant factor; divide by shell volume ~ 4πr^2 dr
    shell = 4.0 * np.pi * np.maximum(r_centers, 1e-8) ** 2 * np.maximum(dr, 1e-12)
    g_raw = counts / np.maximum(shell, 1e-24)

    tail_mask = (r_centers >= float(tail_start)) & (r_centers <= float(tail_end))
    if not np.any(tail_mask):
        tail_mean = 1.0
        logger.warning(
            f"RDF tail window [{tail_start},{tail_end}] Å contains no bins "
            f"(r range is [{r_centers.min():.3g},{r_centers.max():.3g}] Å). "
            "Using tail_mean=1.0 (no tail normalization)."
        )
    else:
        tail_mean = float(np.mean(g_raw[tail_mask]))
        if not np.isfinite(tail_mean) or tail_mean <= 0:
            logger.warning(f"Non-finite/invalid tail_mean={tail_mean}; using 1.0")
            tail_mean = 1.0

    g = g_raw / max(tail_mean, 1e-24)

    # PMF in reduced units
    pmf = -np.log(np.maximum(g, float(pmf_min_g)))

    # Optional PMF smoothing
    if float(pmf_smooth_sigma_bins) > 0:
        try:
            from scipy.ndimage import gaussian_filter1d

            pmf = gaussian_filter1d(pmf, sigma=float(pmf_smooth_sigma_bins), mode="nearest").astype(np.float64)
        except Exception as e:
            logger.warning(f"scipy smoothing not available for PMF; continuing unsmoothed. ({e})")

    return RDFResult(
        r_edges=r_edges,
        r_centers=r_centers,
        counts=counts,
        g_r=g,
        pmf=pmf,
        tail_mean=tail_mean,
    )


def extract_repulsive_wall(
    rdf: RDFResult,
    r_star: Optional[float] = None,
    enforce_monotone: bool = True,
    ensure_nonnegative: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Extract a repulsive wall from g(r)/PMF.

    We define:
        W_raw(r) = -log(g(r))  (dimensionless, kBT units)
    Then choose r_star as the first crossing where g(r) >= 1 (i.e., W_raw <= 0),
    and set the repulsive wall to:
        W_rep(r) = max(W_raw(r) - W_raw(r_star), 0) for r <= r_star
                 = 0                            for r > r_star

    Then optionally enforce monotonicity as r decreases.

    Returns:
        r_centers, W_raw, r_star, W_rep (all on the original center grid)
    """
    r = rdf.r_centers.astype(np.float64)
    g = rdf.g_r.astype(np.float64)

    W_raw = -np.log(np.maximum(g, PMF_MIN_G))
    W_raw = W_raw - W_raw[-1]  # shift so tail is ~0 at the largest r

    # Pick r_star if not provided: first bin where g>=1 (equivalently W_raw<=0)
    if r_star is None:
        idxs = np.where(g >= 1.0)[0]
        if idxs.size == 0:
            # fallback: choose the last bin (no crossing)
            r_star = float(r[-1])
            logger.warning("g(r) never crosses 1.0; using r_star=r_max.")
        else:
            r_star = float(r[idxs[0]])

    # Build repulsive wall
    W_rep = W_raw.copy()
    W_rep[r > r_star] = 0.0
    # shift so that W_rep(r_star)=0
    # locate closest index to r_star
    k_star = int(np.argmin(np.abs(r - r_star)))
    W_rep = W_rep - W_rep[k_star]
    W_rep[r > r_star] = 0.0

    if enforce_monotone:
        # enforce that W_rep increases as r decreases (monotone wall)
        W_rep = np.maximum.accumulate(W_rep[::-1])[::-1]

    if ensure_nonnegative:
        W_rep = np.maximum(W_rep, 0.0)

    return r, W_raw, float(r_star), W_rep


def densify_wall(
    r_centers: np.ndarray,
    W_rep: np.ndarray,
    sparse_n: int = WALL_SPARSE_N,
    dense_n: int = WALL_DENSE_N,
    smooth_sigma_bins: float = WALL_SMOOTH_SIGMA_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a dense, smooth wall by:
      1) sampling sparse_n points over the r_centers range
      2) linear interpolation to dense_n points
      3) optional smoothing in bins (nearest mode)

    Returns:
        r_dense, W_dense
    """
    r_centers = np.asarray(r_centers, dtype=np.float64).reshape(-1)
    W_rep = np.asarray(W_rep, dtype=np.float64).reshape(-1)
    if r_centers.size != W_rep.size:
        raise ValueError("r_centers and W_rep must have same length")

    # choose sparse indices uniformly in index-space
    if sparse_n <= 2:
        sparse_n = 2
    idx = np.linspace(0, r_centers.size - 1, int(sparse_n), dtype=np.int64)
    r_sparse = r_centers[idx]
    W_sparse = W_rep[idx]

    r_dense = np.linspace(r_centers.min(), r_centers.max(), int(dense_n), dtype=np.float64)
    W_dense = np.interp(r_dense, r_sparse, W_sparse).astype(np.float64)

    # smooth if requested
    if float(smooth_sigma_bins) > 0:
        try:
            from scipy.ndimage import gaussian_filter1d

            W_dense = gaussian_filter1d(W_dense, sigma=float(smooth_sigma_bins), mode="nearest").astype(np.float64)
        except Exception as e:
            logger.warning(f"scipy smoothing not available for wall; continuing unsmoothed. ({e})")

    # ensure nonnegative (repulsive)
    W_dense = np.maximum(W_dense, 0.0)
    return r_dense.astype(np.float64), W_dense.astype(np.float64)


def save_repulsive_wall(
    out_dir: Path,
    r_dense: np.ndarray,
    W_dense: np.ndarray,
    r_star: float,
    meta_extra: Optional[dict] = None,
) -> None:
    """
    Save repulsive wall arrays + metadata to disk.

    Files:
      - repulsive_wall_r_A.npy
      - repulsive_wall_energy.npy
      - repulsive_wall_metadata.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "repulsive_wall_r_A.npy", np.asarray(r_dense, dtype=np.float32))
    np.save(out_dir / "repulsive_wall_energy.npy", np.asarray(W_dense, dtype=np.float32))

    meta = {
        "format_version": "1.0",
        "quantity": "-log(g(r)) repulsive wall (monotone, nonnegative)",
        "units": "dimensionless; scaled by lambda_rep in the model",
        "r_star_A": float(r_star),
        "min_value": float(np.min(W_dense)),
        "max_value": float(np.max(W_dense)),
    }
    if meta_extra:
        meta.update(dict(meta_extra))

    with open(out_dir / "repulsive_wall_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved repulsive wall to {out_dir / 'repulsive_wall_energy.npy'}")
