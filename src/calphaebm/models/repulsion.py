"""
Data-driven repulsion energy from PDB analysis with differentiable interpolation.

Nonbonded exclusion:
  We exclude pairs with sequence separation |i-j| <= exclude (default exclude=3),
  i.e. we include only |i-j| > 3. This matches the nonbonded definition used by
  the packing term and avoids double-counting near-local geometry already
  constrained by bond/angle/torsion terms.

lambda_rep:
  Trainable internal scale parameter (nn.Parameter, init ~0.172 from force calibration).
  This is the only trainable handle on repulsion magnitude because the energy table
  is a fixed PDB-derived buffer. The outer gate in TotalEnergy is frozen at 1.0.

Normalization (Option R1: per-residue):
  If normalize_by_length=True, the returned energy is normalized by L:
      E_rep = (1/L) * lambda_rep * sum_{i,k} E(r_{i,k}) * sw(r_{i,k})
  This makes the repulsion term consistent with other internally normalized terms.
  IMPORTANT: If you enable this, diagnostics should NOT divide repulsion by L again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calphaebm.geometry.pairs import topk_nonbonded_pairs
from calphaebm.utils.logging import get_logger
from calphaebm.utils.smooth import smooth_switch

logger = get_logger()


class RepulsionEnergy(nn.Module):
    """Data-driven repulsion energy from PDB analysis with differentiable interpolation.

    Notes:
      - lambda_rep is a trainable nn.Parameter (init from force calibration ~0.172).
        It is the sole internal trainable scale for repulsion because the energy table
        is fixed (non-trainable PDB-derived buffer). Outer gate is frozen at 1.0.
      - Optional internal per-residue normalization (Option R1) via normalize_by_length.
    """

    def __init__(
        self,
        data_dir: str = "analysis/repulsion_analysis/data",
        K: int = 64,
        exclude: int = 3,  # include only |i-j| > exclude
        r_on: float = 8.0,
        r_cut: float = 10.0,
        # Calibrated internal scale: init ~0.172 (= 1/5.8 from force balance measurement)
        init_lambda_rep: float = 0.172,
        # NEW: internal normalization (Option R1)
        normalize_by_length: bool = True,
        # Numerical stability
        r_min_safe: float = 3.8,
        softplus_beta: float = 20.0,
    ):
        super().__init__()

        # Trainable internal scale (the only trainable handle since the table is fixed)
        self._lambda_rep_raw = nn.Parameter(torch.tensor(float(init_lambda_rep), dtype=torch.float32))

        self.K = int(K)
        self.exclude = int(exclude)
        self.r_on = float(r_on)
        self.r_cut = float(r_cut)
        self.normalize_by_length = bool(normalize_by_length)

        self.r_min_safe = float(r_min_safe)
        self.softplus_beta = float(softplus_beta)

        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.exclude < 0:
            raise ValueError(f"exclude must be >= 0, got {self.exclude}")
        if not (self.r_on < self.r_cut):
            raise ValueError(f"Require r_on < r_cut, got r_on={self.r_on}, r_cut={self.r_cut}")
        if self.r_min_safe <= 0.0:
            raise ValueError(f"r_min_safe must be > 0, got {self.r_min_safe}")
        if self.softplus_beta <= 0.0:
            raise ValueError(f"softplus_beta must be > 0, got {self.softplus_beta}")

        data_path = Path(data_dir)

        # Try multiple possible filenames for centers
        center_candidates = [
            data_path / "repulsive_wall_centers.npy",  # preferred naming
            data_path / "repulsive_wall_r_A.npy",  # alternative naming (your logs show this)
        ]

        centers_path: Optional[Path] = None
        for candidate in center_candidates:
            if candidate.exists():
                centers_path = candidate
                logger.debug(f"Found repulsion centers: {candidate.name}")
                break

        if centers_path is None:
            raise FileNotFoundError(
                f"No repulsion centers file found in {data_dir}. " f"Tried: {[c.name for c in center_candidates]}"
            )

        energy_path = data_path / "repulsive_wall_energy.npy"
        if not energy_path.exists():
            raise FileNotFoundError(f"Repulsion energy file not found: {energy_path}")

        centers = np.load(centers_path).astype(np.float32).reshape(-1)  # grid points
        energy = np.load(energy_path).astype(np.float32).reshape(-1)  # energy values

        if centers.ndim != 1 or energy.ndim != 1:
            raise ValueError("repulsive wall centers/energy must be 1D arrays after reshape(-1)")
        if centers.shape[0] != energy.shape[0]:
            raise ValueError(f"centers and energy must have same length, got {centers.shape[0]} vs {energy.shape[0]}")
        if centers.shape[0] < 2:
            raise ValueError(f"centers must have >=2 points, got {centers.shape[0]}")
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(energy)):
            raise ValueError("repulsive wall arrays contain non-finite values")
        if not np.all(np.diff(centers) > 0):
            raise ValueError("repulsive wall centers must be strictly increasing")

        # Register as buffers (not parameters)
        self.register_buffer("r_centers", torch.tensor(centers, dtype=torch.float32))
        self.register_buffer("energy_table", torch.tensor(energy, dtype=torch.float32))

        # Grid metadata
        r_min = float(centers[0])
        r_max = float(centers[-1])
        n_points = int(centers.shape[0])
        dr = (r_max - r_min) / float(n_points - 1)

        self.register_buffer("r_min", torch.tensor(r_min, dtype=torch.float32))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float32))
        self.register_buffer("dr", torch.tensor(dr, dtype=torch.float32))
        self.register_buffer("n_points", torch.tensor(n_points, dtype=torch.long))
        self.register_buffer("r_star", torch.tensor(r_max, dtype=torch.float32))

        # Summary for consolidated init log
        self._init_summary = {
            "total_params": 1,  # just lambda_rep
            "n_grid": n_points,
            "r_range": (r_min, r_max),
            "init_lambda": init_lambda_rep,
        }
        logger.debug(
            "RepulsionEnergy: %d grid points [%.1f, %.1f]Å, λ_init=%.4f", n_points, r_min, r_max, init_lambda_rep
        )

    @property
    def lambda_rep(self) -> torch.Tensor:
        """Positive internal scale for repulsion energy (trainable)."""
        return torch.nn.functional.softplus(self._lambda_rep_raw) + 1e-6

    def _lookup_energy_differentiable(self, r: torch.Tensor) -> torch.Tensor:
        """Differentiable linear interpolation with robust boundary handling.

        Args:
            r: distances in Å, shape (...).

        Returns:
            energies with same shape as r.
        """
        orig_shape = r.shape
        r_flat = r.reshape(-1)

        # Smoothly push distances above r_min_safe to avoid extreme gradients / NaNs
        # r_eff = r_min_safe + softplus(r - r_min_safe)
        r_min_safe = self.r_min_safe
        beta = self.softplus_beta
        r_eff = r_min_safe + F.softplus(r_flat - r_min_safe, beta=beta)

        r_min = float(self.r_min.item())
        r_max = float(self.r_max.item())
        dr = float(self.dr.item())
        n = int(self.n_points.item())
        energy = self.energy_table  # (n,)

        energy_flat = torch.zeros_like(r_flat)

        in_range = (r_eff >= r_min) & (r_eff <= r_max)
        if in_range.any():
            r_in = r_eff[in_range]

            # u in [0, n-1]
            u = (r_in - r_min) / dr

            # i0 in [0, n-2]
            i0 = torch.floor(u).long()
            i0 = torch.clamp(i0, 0, n - 2)

            # t in [0,1]
            t = u - i0.to(dtype=u.dtype)
            t = torch.clamp(t, 0.0, 1.0)

            e0 = energy[i0]
            e1 = energy[i0 + 1]
            energy_flat[in_range] = (1.0 - t) * e0 + t * e1

        # Below grid minimum: clamp to first table value
        below_min = r_eff < r_min
        if below_min.any():
            energy_flat[below_min] = energy[0]

        # Above r_max: (shouldn't happen due to in_range mask, but safe) clamp to last value
        above_max = r_eff > r_max
        if above_max.any():
            energy_flat[above_max] = energy[-1]

        return energy_flat.reshape(orig_shape)

    def forward(self, R: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Compute repulsion energy (unscaled - outer gate will multiply).

        Args:
            R: (B, L, 3) coordinates
            seq: (B, L) amino acid indices (unused)
            lengths: (B,) actual chain lengths. If provided, padding atoms
                     are excluded from neighbour lists.

        Returns:
            (B,) raw energy per batch element (before outer gate multiplication).
            If normalize_by_length=True, energy is per-residue (divided by L).
        """
        _ = seq  # explicitly unused

        if R.dim() != 3 or R.size(-1) != 3:
            raise ValueError(f"R must have shape (B, L, 3), got {tuple(R.shape)}")

        B, L, _ = R.shape

        # Move padding atoms far away BEFORE computing neighbors.
        # Without this, real atoms near origin pick padding atoms at degenerate
        # positions (from NeRF on garbage ICs) as nearest neighbors → 0Å distances.
        if lengths is not None:
            valid_atom = torch.arange(L, device=R.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, L)
            R = R.masked_fill(~valid_atom.unsqueeze(2), 1e6)

        # Nonbonded pairs: include only |i-j| > exclude
        r, _ = topk_nonbonded_pairs(R, k=self.K, exclude=self.exclude)  # (B, L, K)

        # Belt-and-suspenders: also mask distances FROM padding atoms
        if lengths is not None:
            r = r.masked_fill(~valid_atom.unsqueeze(2), self.r_cut + 1.0)

        # Clamp distances to physical minimum (1.0Å for Cα-Cα).
        # Sub-1Å distances arise from extreme IC perturbations (σ=2.0 rad decoys)
        # and produce unreliable gradients. The repulsion wall is already maximal
        # at 1.0Å so clamping here doesn't change physics, just stabilises numerics.
        r = torch.clamp(r, min=1.0)

        # Look up energy for each distance (>= 0 typically)
        E_pair = self._lookup_energy_differentiable(r)  # (B, L, K)

        # Apply smooth switching function (typically ~1 at short range, to 0 near cutoff)
        sw = smooth_switch(r, self.r_on, self.r_cut)  # (B, L, K)

        # Sum over all pairs and scale by internal trainable lambda_rep
        E = self.lambda_rep * (E_pair * sw).sum(dim=(1, 2))  # (B,)

        # Option R1: per-residue normalization
        if self.normalize_by_length:
            if lengths is not None:
                denom = lengths.float().clamp(min=1.0)
            else:
                denom = float(max(L, 1))
            E = E / denom

        return E

    def energy_from_distances(self, r: torch.Tensor) -> torch.Tensor:
        """Compute repulsion energy from distances directly (testing/debug).

        If r is:
          - scalar: returns scalar
          - (...): returns same shape (pairwise energies after switching)
        """
        if r.dim() == 0:
            r = r.unsqueeze(0)

        E = self._lookup_energy_differentiable(r)
        sw = smooth_switch(r, self.r_on, self.r_cut)
        return E * sw


__all__ = ["RepulsionEnergy"]
