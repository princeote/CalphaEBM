"""Native contact analysis (Q) and contact counting."""

from typing import Tuple

import numpy as np


def pairwise_distances(R: np.ndarray) -> np.ndarray:
    """Compute all pairwise distances.

    Args:
        R: (N, 3) coordinates or (B, N, 3) batched coordinates.

    Returns:
        Distance matrix of shape (N, N) or (B, N, N).
    """
    if R.ndim == 2:
        # Single structure: (N, 3)
        diff = R[:, None, :] - R[None, :, :]
        return np.sqrt((diff * diff).sum(axis=-1) + 1e-12)
    elif R.ndim == 3:
        # Batched structures: (B, N, 3)
        diff = R[:, :, None, :] - R[:, None, :, :]
        return np.sqrt((diff * diff).sum(axis=-1) + 1e-12)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {R.shape}")


def native_contact_set(
    R_ref: np.ndarray,
    cutoff: float = 8.0,
    exclude: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build native contact set from reference structure.

    Args:
        R_ref: (N, 3) reference coordinates.
        cutoff: Distance cutoff for defining contacts (Å).
        exclude: Sequence separation cutoff (|i-j| <= exclude excluded).
                 Default 3, consistent with geometry/pairs.py.

    Returns:
        (i_indices, j_indices, d0_distances) for native contacts.
    """
    # Ensure R_ref is 2D
    if R_ref.ndim == 3:
        R_ref = R_ref[0]  # Take first batch if batched

    D = pairwise_distances(R_ref)  # (N, N)
    N = D.shape[0]

    # Sequence separation mask
    sep = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    mask = sep > exclude  # (N, N)

    # Upper triangle indices
    iu = np.triu_indices(N, k=1)  # (2, M) where M = N*(N-1)/2

    # Extract values at upper triangle indices
    # D[iu] returns a 1D array of shape (M,)
    d_vals = D[iu]
    mask_vals = mask[iu]

    # Find contacts
    good = mask_vals & (d_vals < cutoff)

    # Get indices and distances for good contacts
    i = iu[0][good].astype(np.int64)
    j = iu[1][good].astype(np.int64)
    d0 = d_vals[good].astype(np.float64)

    return i, j, d0


def q_hard(
    R: np.ndarray,
    native_i: np.ndarray,
    native_j: np.ndarray,
    cutoff: float = 8.0,
) -> float:
    """Compute Q_hard: fraction of native contacts present with hard cutoff.

    Args:
        R: (N, 3) coordinates.
        native_i, native_j: Native contact indices.
        cutoff: Distance cutoff for considering contact present.

    Returns:
        Q value in [0, 1].
    """
    if native_i.size == 0:
        return 0.0

    # Handle batched input
    if R.ndim == 3:
        R = R[0]  # Take first batch

    rij = R[native_i] - R[native_j]
    dij = np.sqrt(np.sum(rij * rij, axis=1) + 1e-12)

    return float(np.mean(dij < cutoff))


def q_smooth(
    R: np.ndarray,
    native_i: np.ndarray,
    native_j: np.ndarray,
    d0: np.ndarray,
    beta: float = 5.0,
    lam: float = 1.8,
) -> float:
    """Compute Q_smooth: Best-style smooth contact fraction.

    s_ij = 1 / (1 + exp(beta * (d_ij - lam * d0_ij)))
    Q = mean(s_ij)

    Args:
        R: (N, 3) coordinates.
        native_i, native_j: Native contact indices.
        d0: Native contact distances.
        beta: Sharpness parameter (1/Å).
        lam: Tolerance factor multiplying native distance.

    Returns:
        Q value in [0, 1].
    """
    if native_i.size == 0:
        return 0.0

    # Handle batched input
    if R.ndim == 3:
        R = R[0]  # Take first batch

    rij = R[native_i] - R[native_j]
    dij = np.sqrt(np.sum(rij * rij, axis=1) + 1e-12)

    x = beta * (dij - lam * d0)
    x = np.clip(x, -60.0, 60.0)  # Prevent overflow
    s = 1.0 / (1.0 + np.exp(x))

    return float(np.mean(s))


def contact_count(
    R: np.ndarray,
    cutoff: float = 8.0,
    exclude: int = 3,
) -> int:
    """Count number of contacts (pairs within cutoff, excluding local).

    Args:
        R: (N, 3) Cα coordinates.
        cutoff: Distance cutoff (Å).
        exclude: Minimum sequence separation (|i-j| > exclude). Default 3,
                 consistent with geometry/pairs.py.
    """
    # Handle batched input
    if R.ndim == 3:
        R = R[0]  # Take first batch

    D = pairwise_distances(R)
    N = D.shape[0]

    sep = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    mask = sep > exclude

    iu = np.triu_indices(N, k=1)
    d_vals = D[iu]
    mask_vals = mask[iu]
    good = mask_vals & (d_vals < cutoff)

    return int(good.sum())


# ── Contact Order ────────────────────────────────────────────────────────


def contact_order(
    R: np.ndarray,
    cutoff: float = 8.0,
    exclude: int = 3,
) -> Tuple[float, float, int]:
    """Compute relative and absolute contact order.

    Relative CO = (1 / (N_c * L)) * Σ |i - j|   for all contacts (i, j)
    Absolute CO = CO * L = (1 / N_c) * Σ |i - j|

    Contact order measures the average sequence separation of contacting
    residues, normalized by chain length. High CO → complex topology
    (β-sheets, long-range contacts), low CO → simple topology (helices).

    Strongly correlates with folding rate: ln(k_f) ∝ −CO for two-state
    folders (Plaxco et al., J Mol Biol, 1998).

    Args:
        R: (N, 3) Cα coordinates.
        cutoff: Distance cutoff for defining contacts (Å).
        exclude: Minimum sequence separation (|i-j| > exclude). Default 3,
                 consistent with topk_nonbonded_pairs in geometry/pairs.py.

    Returns:
        (relative_CO, absolute_CO, n_contacts) tuple.
        relative_CO: CO in [0, 1], normalized by L.
        absolute_CO: CO * L = mean sequence separation of contacts.
        n_contacts: Number of contacts found.
    """
    # Handle batched input
    if R.ndim == 3:
        R = R[0]

    N = R.shape[0]
    D = pairwise_distances(R)

    # Upper triangle, non-local pairs
    iu = np.triu_indices(N, k=1)
    sep = np.abs(iu[0] - iu[1])
    mask = sep > exclude

    d_vals = D[iu]
    contacts = mask & (d_vals < cutoff)

    n_contacts = int(contacts.sum())
    if n_contacts == 0:
        return 0.0, 0.0, 0

    # Sum of sequence separations for contacts
    total_sep = float(sep[contacts].sum())

    # Relative CO: normalized by both N_contacts and chain length
    relative_co = total_sep / (n_contacts * N)

    # Absolute CO: just mean sequence separation
    absolute_co = total_sep / n_contacts

    return relative_co, absolute_co, n_contacts
