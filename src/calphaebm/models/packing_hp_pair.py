# src/calphaebm/models/packing_hp_pair.py
"""
ARCHIVED: Pairwise hydrophobic contact energy (_HydrophobicPairs).

Replaced by coordination-based _Hydrophobic in packing.py.

Reason for removal:
  The rank-1 pairwise potential E = -λ * Σ h(aa_i) * h(aa_j) * g(r_ij)
  produced systematically NEGATIVE discrimination: perturbed structures
  scored better than native because:
  1. top-K always finds 64 neighbors regardless of structure quality
  2. Gaussian kernel peaks at r=6.5Å, rewarding non-native contacts at that distance
  3. Rank-1 interaction matrix (h_i * h_j) covers only ~35% of the full 20×20
     contact frequency matrix — it rewards any same-type clustering, not
     specifically native packing

  The coordination-based approach measures whether each residue has the
  correct NUMBER of neighbors (from PDB statistics), which is:
  - Positive discrimination BY CONSTRUCTION (Gaussian peaks at native n*)
  - Physically grounded (per-AA coordination from PDB analysis)
  - More robust (scalar n_i vs pairwise h_i*h_j)

History:
  - Used in run1 full-stage (10 rounds) and early SC curriculum
  - Contact discrimination was negative at all noise scales
  - Other subterms (secondary, local, repulsion) compensated, but packing
    actively fought the funnel
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calphaebm.utils.logging import get_logger

logger = get_logger()


def _inv_softplus(y: float, eps: float = 1e-8) -> float:
    y = float(max(y, eps))
    return float(np.log(np.expm1(y)))


KYTE_DOOLITTLE_20 = [
    +1.8,
    -4.5,
    -3.5,
    -3.5,
    +2.5,
    -3.5,
    -3.5,
    -0.4,
    -3.2,
    +4.5,
    +3.8,
    -3.9,
    +1.9,
    +2.8,
    -1.6,
    -0.8,
    -0.7,
    -0.9,
    -1.3,
    +4.2,
]

H_SVD_MEDIUM_20 = [
    +0.3279,
    +1.0000,
    -0.9373,
    -0.8584,
    +0.6153,
    -0.5127,
    -0.3207,
    +0.8355,
    -0.7528,
    +0.6792,
    +0.5284,
    -0.8558,
    -0.4101,
    -0.7967,
    -0.6088,
    -0.5275,
    -0.2591,
    +0.8940,
    +0.0663,
    +0.1645,
]


class _HydrophobicPairs(nn.Module):
    """ARCHIVED: Pairwise hydrophobic interaction energy.

    E_hp = -lambda_hp * (1/L) * sum_i sum_j h(aa_i) * h(aa_j) * g(r_ij)

    h(aa): learned 20-dim pair interaction vector (init from SVD of log(OE)).
    g(r):  Gaussian bell: g(r) = exp(-(r - r_peak)^2 / (2*sigma^2))

    Parameters: 20 (h) + 2 (r_peak, sigma) + 1 (lambda_hp) = 23
    """

    def __init__(
        self,
        num_aa: int = 20,
        init_lambda: float = 1.0,
        init_r_half: float = 6.5,
        init_tau: float = 1.5,
        h_init: str = "svd",
    ):
        super().__init__()

        if h_init == "svd":
            h_vec = torch.tensor(H_SVD_MEDIUM_20[:num_aa], dtype=torch.float32)
            init_source = "SVD(log_OE_medium)"
        elif h_init == "kd":
            kd = torch.tensor(KYTE_DOOLITTLE_20[:num_aa], dtype=torch.float32)
            h_vec = kd / max(kd.abs().max().item(), 1e-6)
            init_source = "Kyte-Doolittle"
        elif h_init == "zeros":
            h_vec = torch.zeros(num_aa, dtype=torch.float32)
            init_source = "zeros"
        else:
            raise ValueError(f"h_init must be 'svd', 'kd', or 'zeros', got '{h_init}'")

        self.h = nn.Parameter(h_vec)
        self._r_half_raw = nn.Parameter(torch.tensor(_inv_softplus(init_r_half), dtype=torch.float32))
        self._tau_hp_raw = nn.Parameter(torch.tensor(_inv_softplus(init_tau), dtype=torch.float32))
        self._lambda_hp_raw = nn.Parameter(torch.tensor(_inv_softplus(init_lambda), dtype=torch.float32))

        logger.debug(
            "HydrophobicPairs: %d AA, h_init=%s, r_peak=%.1f, sigma=%.1f", num_aa, init_source, init_r_half, init_tau
        )

    @property
    def r_peak(self) -> torch.Tensor:
        return F.softplus(self._r_half_raw)

    @property
    def r_half(self) -> torch.Tensor:
        return self.r_peak

    @property
    def sigma_hp(self) -> torch.Tensor:
        return F.softplus(self._tau_hp_raw) + 0.1

    @property
    def tau_hp(self) -> torch.Tensor:
        return self.sigma_hp

    @property
    def lambda_hp(self) -> torch.Tensor:
        return F.softplus(self._lambda_hp_raw) + 1e-6

    def forward(self, seq, r, j_idx, max_dist=10.0):
        B, L, K = r.shape
        valid = (r < max_dist - 1e-4).float()
        h_i = self.h[seq]
        j_safe = j_idx.clamp(0, L - 1)
        seq_j = torch.gather(seq, 1, j_safe.view(B, -1)).view(B, L, K)
        h_j = self.h[seq_j]
        r_clamped = r.clamp(max=max_dist)
        g = torch.exp(-((r_clamped - self.r_peak) ** 2) / (2.0 * self.sigma_hp**2))
        g = g * valid
        pair_energy = h_i.unsqueeze(-1) * h_j * g
        E_hp_i = pair_energy.sum(dim=-1)
        return E_hp_i
