"""Neighbor list utilities for efficient nonbonded calculations.

This module is kept for backward compatibility.

Preferred implementation lives in:
    calphaebm.geometry.pairs.topk_nonbonded_pairs

We keep:
- pairwise_distances (used broadly, simple and useful)
- topk_nonbonded_pairs wrapper (legacy signature: K/exclude/max_dist)
- NeighborList helper
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from calphaebm.geometry.pairs import topk_nonbonded_pairs as _topk_pairs


def pairwise_distances(R: torch.Tensor) -> torch.Tensor:
    """Compute all pairwise distances.

    Args:
        R: (B, L, 3) coordinates.

    Returns:
        (B, L, L) distance matrix.
    """
    return torch.cdist(R, R)


def topk_nonbonded_pairs(
    R: torch.Tensor,
    K: int = 64,
    exclude: int = 2,
    max_dist: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy wrapper for top-k nearest nonbonded neighbors for each residue.

    Legacy signature (utils.neighbors):
        topk_nonbonded_pairs(R, K=..., exclude=..., max_dist=...)

    Canonical signature (geometry.pairs):
        topk_nonbonded_pairs(R, k=..., exclude=..., cutoff=...)

    Returns:
        distances: (B, L, K)
        indices:   (B, L, K)
    """
    return _topk_pairs(R, k=int(K), exclude=int(exclude), cutoff=max_dist)


class NeighborList:
    """Verlet-style neighbor list with skin."""

    def __init__(
        self,
        cutoff: float = 12.0,
        skin: float = 2.0,
        max_neighbors: int = 64,
    ):
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.max_neighbors = int(max_neighbors)
        self.pairs = None
        self.last_positions = None

    def update(
        self,
        R: torch.Tensor,
        exclude: int = 2,
        force: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update neighbor list if needed.

        Args:
            R: (B, L, 3) current positions.
            exclude: Sequence separation cutoff.
            force: Force update even if skin not exceeded.

        Returns:
            (distances, indices) as in topk_nonbonded_pairs.
        """
        needs_update = force
        if not needs_update and self.last_positions is not None:
            drift = torch.norm(R - self.last_positions, dim=-1).max().item()
            if drift > self.skin:
                needs_update = True

        if needs_update or self.pairs is None:
            self.pairs = topk_nonbonded_pairs(R, K=self.max_neighbors, exclude=exclude, max_dist=self.cutoff)
            self.last_positions = R.clone()

        return self.pairs
