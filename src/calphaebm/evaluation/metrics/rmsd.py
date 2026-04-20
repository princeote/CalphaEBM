# src/calphaebm/evaluation/metrics/rmsd.py
"""RMSD calculation with Kabsch alignment."""

from typing import Tuple

import numpy as np


def kabsch_rotate(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find optimal rotation aligning P onto Q after centering.

    Args:
        P: (N, 3) source points.
        Q: (N, 3) target points.

    Returns:
        (P_aligned, rotation_matrix) where P_aligned = (P - P_com) @ R
    """
    # Center
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)

    # Covariance matrix
    C = P_centered.T @ Q_centered

    # SVD
    V, _, Wt = np.linalg.svd(C)

    # Ensure right-handed coordinate system
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt

    P_aligned = P_centered @ R
    return P_aligned, R


def rmsd_kabsch(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute RMSD between two point sets after optimal alignment.

    Args:
        P: (N, 3) source points.
        Q: (N, 3) target points.

    Returns:
        RMSD value in same units as input.
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: {P.shape} vs {Q.shape}")

    if len(P) == 0:
        return 0.0

    P_aligned, _ = kabsch_rotate(P, Q)
    Q_centered = Q - Q.mean(axis=0)

    diff = P_aligned - Q_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def batch_rmsd(trajectory: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute RMSD for each frame in a trajectory.

    Args:
        trajectory: (n_frames, N, 3) coordinates.
        reference: (N, 3) reference structure.

    Returns:
        (n_frames,) RMSD values.
    """
    n_frames = trajectory.shape[0]
    rmsds = np.zeros(n_frames)

    for i in range(n_frames):
        rmsds[i] = rmsd_kabsch(trajectory[i], reference)

    return rmsds


def pairwise_distances(R: np.ndarray) -> np.ndarray:
    """Compute all pairwise distances."""
    diff = R[:, None, :] - R[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1) + 1e-12)


def drmsd(P: np.ndarray, Q: np.ndarray, mode: str = "all", exclude: int = 2) -> float:
    """Compute distance RMSD between two structures.

    Args:
        P: (N, 3) coordinates.
        Q: (N, 3) reference coordinates.
        mode: 'all' or 'nonlocal' - which pairs to include.
        exclude: Sequence separation for nonlocal mode.

    Returns:
        dRMSD value.
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: {P.shape} vs {Q.shape}")

    N = P.shape[0]
    D_P = pairwise_distances(P)
    D_Q = pairwise_distances(Q)

    # Create mask for pairs to include
    iu = np.triu_indices(N, k=1)
    if mode == "all":
        mask = np.ones(len(iu[0]), dtype=bool)
    elif mode == "nonlocal":
        sep = np.abs(iu[0] - iu[1])
        mask = sep > exclude
    else:
        raise ValueError(f"Unknown mode: {mode}")

    diff = (D_P[iu] - D_Q[iu])[mask]
    if len(diff) == 0:
        return 0.0

    return float(np.sqrt(np.mean(diff * diff)))


def batch_drmsd(trajectory: np.ndarray, reference: np.ndarray, mode: str = "nonlocal", exclude: int = 2) -> np.ndarray:
    """Compute dRMSD for each frame in a trajectory.

    Args:
        trajectory: (n_frames, N, 3) coordinates.
        reference: (N, 3) reference structure.
        mode: 'all' or 'nonlocal' - which pairs to include.
        exclude: Sequence separation for nonlocal mode.

    Returns:
        (n_frames,) dRMSD values.
    """
    n_frames = trajectory.shape[0]
    drmsds = np.zeros(n_frames)

    for i in range(n_frames):
        drmsds[i] = drmsd(trajectory[i], reference, mode=mode, exclude=exclude)

    return drmsds


# ── k-nearest dRMSD (k64dRMSD) ──────────────────────────────────────────


def _top_k_contacts(
    R_ref: np.ndarray,
    K: int = 64,
    exclude: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the K closest non-local Cα pairs in the native structure.

    Numpy equivalent of geometry/pairs.py:topk_nonbonded_pairs (which is
    PyTorch/GPU for differentiable training). Uses same exclude=3 default.

    Args:
        R_ref: (N, 3) reference coordinates.
        K: Number of closest contacts to keep (default 64, matching packing topk).
        exclude: Minimum sequence separation (|i-j| > exclude). Default 3,
                 consistent with geometry/pairs.py.

    Returns:
        (i_indices, j_indices, d0_distances) for the K closest pairs.
    """
    N = R_ref.shape[0]
    D = pairwise_distances(R_ref)

    # Upper triangle, non-local pairs only
    iu = np.triu_indices(N, k=1)
    sep = np.abs(iu[0] - iu[1])
    nonlocal_mask = sep > exclude

    i_all = iu[0][nonlocal_mask]
    j_all = iu[1][nonlocal_mask]
    d_all = D[iu][nonlocal_mask]

    # Take K closest
    n_available = len(d_all)
    k_actual = min(K, n_available)
    if k_actual == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    idx = np.argpartition(d_all, k_actual)[:k_actual]
    return i_all[idx], j_all[idx], d_all[idx]


def k_drmsd(
    P: np.ndarray,
    Q: np.ndarray,
    K: int = 64,
    exclude: int = 3,
    native_contacts: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> float:
    """Compute distance RMSD over the K closest native contacts.

    Focuses on the most structurally important contacts rather than all
    pairs. More discriminating than full dRMSD for fold quality assessment.

    k64dRMSD = sqrt(mean_k((d_ij^sample - d_ij^native)²))

    where the mean is over the K closest non-local Cα pairs in the native.

    Args:
        P: (N, 3) sample coordinates.
        Q: (N, 3) native/reference coordinates.
        K: Number of closest native contacts to use (default 64).
        exclude: Minimum sequence separation for contacts.
        native_contacts: Pre-computed (i, j, d0) from _top_k_contacts.
                         If None, computed from Q.

    Returns:
        k-dRMSD value in Å.
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: {P.shape} vs {Q.shape}")

    if native_contacts is None:
        ci, cj, d0 = _top_k_contacts(Q, K=K, exclude=exclude)
    else:
        ci, cj, d0 = native_contacts

    if len(ci) == 0:
        return 0.0

    # Compute distances in sample
    rij = P[ci] - P[cj]
    d_sample = np.sqrt(np.sum(rij * rij, axis=1) + 1e-12)

    diff = d_sample - d0
    return float(np.sqrt(np.mean(diff * diff)))


def batch_k_drmsd(
    trajectory: np.ndarray,
    reference: np.ndarray,
    K: int = 64,
    exclude: int = 3,
) -> np.ndarray:
    """Compute k-dRMSD for each frame in a trajectory.

    Pre-computes native contacts once, reuses for all frames.

    Args:
        trajectory: (n_frames, N, 3) coordinates.
        reference: (N, 3) reference structure.
        K: Number of closest native contacts.
        exclude: Minimum sequence separation.

    Returns:
        (n_frames,) k-dRMSD values.
    """
    native_contacts = _top_k_contacts(reference, K=K, exclude=exclude)

    n_frames = trajectory.shape[0]
    kdrmsds = np.zeros(n_frames)

    for i in range(n_frames):
        kdrmsds[i] = k_drmsd(trajectory[i], reference, K=K, exclude=exclude, native_contacts=native_contacts)

    return kdrmsds
