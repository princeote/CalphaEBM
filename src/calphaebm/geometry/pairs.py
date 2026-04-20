# src/calphaebm/geometry/pairs.py

"""
Pair-selection utilities for coarse-grained Cα models.

This module provides `topk_nonbonded_pairs`, used by PackingEnergy/RepulsionEnergy
to select a manageable number of nonbonded interaction pairs per residue.

Design goals:
- Deterministic and fully differentiable w.r.t. coordinates (distances come from R).
- Enforces a sequence-separation exclusion: only |i-j| > exclude.
- Selects per-residue top-k nearest nonbonded partners (within optional cutoff).
- Works on batched coordinates (B, L, 3).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from calphaebm.utils.math import safe_norm


@torch.no_grad()
def _make_exclusion_mask(L: int, exclude: int, device: torch.device) -> torch.Tensor:
    """
    Returns a boolean mask (L, L) where True means the pair is allowed.
    Enforces:
      - i != j
      - |i-j| > exclude
    """
    idx = torch.arange(L, device=device)
    di = (idx[:, None] - idx[None, :]).abs()
    allowed = di > int(exclude)
    allowed.fill_diagonal_(False)
    return allowed


def topk_nonbonded_pairs(
    R: torch.Tensor,
    k: int = 64,
    exclude: int = 3,
    cutoff: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select per-residue top-k nearest nonbonded partners.

    Args:
        R: (B, L, 3) coordinates
        k: number of partners per residue i
        exclude: exclude local pairs with |i-j| <= exclude
        cutoff: optional distance cutoff (Å). Pairs beyond cutoff are treated as invalid.
                Returned distances for invalid slots will be +inf and indices may be arbitrary.

    Returns:
        d_topk: (B, L, k) distances (Å), sorted ascending along k
        j_topk: (B, L, k) partner indices j for each residue i

    Notes:
        - Gradients: distances are differentiable w.r.t. R. The *indices* are discrete.
        - If fewer than k valid pairs exist for a residue, the remaining slots will have
          distance = +inf and the corresponding j_topk values are arbitrary (not meaningful).

    FIX: The original implementation used torch.cdist(R, R) to compute pairwise distances.
    torch.cdist computes sqrt(sum of squared differences), whose second derivative is
    1/r³ — this diverges when two atoms coincide (r=0), producing NaN second-order
    gradients for downstream parameters (U_plus, U_minus, _lambda_pack_raw) under
    DSM's create_graph=True backward.

    Fix: Replace torch.cdist with an explicit safe_norm computation:
        diff = R[:, :, None, :] - R[:, None, :, :]   # (B, L, L, 3)
        D = safe_norm(diff, dim=-1)                    # sqrt(||diff||² + eps)
    safe_norm adds a small epsilon under the sqrt, keeping the second derivative
    finite everywhere including at r=0.
    """
    if R.ndim != 3 or R.shape[-1] != 3:
        raise ValueError(f"R must be (B, L, 3), got {tuple(R.shape)}")

    B, L, _ = R.shape
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")

    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, L - 1)

    device = R.device
    dtype = R.dtype

    # FIX: use safe_norm instead of torch.cdist to avoid NaN second-order gradients.
    # torch.cdist gradient is x/||x|| which is 0/0 at coincident atoms; the second
    # derivative 1/||x||³ blows up → NaN under create_graph=True.
    diff = R[:, :, None, :] - R[:, None, :, :]  # (B, L, L, 3)
    D = safe_norm(diff, dim=-1)  # (B, L, L)

    # Exclusion mask (L, L) - broadcastable to batch
    exclusion_mask = _make_exclusion_mask(L=L, exclude=exclude, device=device)  # (L, L)

    allowed = exclusion_mask.unsqueeze(0)  # (1, L, L) broadcasts to (B, L, L)

    if cutoff is not None:
        cutoff = float(cutoff)
        allowed = allowed & (D <= cutoff)

    inf = torch.tensor(float("inf"), device=device, dtype=dtype)
    D_masked = torch.where(allowed, D, inf)

    d_topk, j_topk = torch.topk(D_masked, k=k, dim=-1, largest=False, sorted=True)  # (B, L, k)

    return d_topk, j_topk
