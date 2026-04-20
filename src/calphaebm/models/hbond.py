"""Hydrogen bond energy terms for secondary structure.

src/calphaebm/models/hbond.py

Two subterms that couple Ramachandran basin assignments with Cα-Cα distances:

  E_hb(α):  Helical H-bonds (local, i→i+4)
            E = -λ_α · (1/L) · Σ_i [p_helix(i) · p_helix(i+3)] · g_α(d(i,i+4))
            Product of 2 endpoint probs (not 4 consecutive) avoids p^4 sparsity.
            Rewards residue stretches with helical backbone AND correct i→i+4 distance.

  E_hb(β):  Sheet H-bonds (nonlocal)
            E = -λ_β · (1/L) · Σ_{i,j:|i-j|>4} p_ext(i) · p_ext(j) · g_β(d_ij)
            Rewards pairs of extended residues at correct Cα-Cα distance.

Basin probabilities are computed from learned E_ram surfaces via softmax:
  p_helix(i) = softmax(-E_basins(θ_i, φ_i))[HELIX_IDX]
  p_ext(i)   = softmax(-E_basins(θ_i, φ_i))[SHEET_IDX]

Distance functions are learned Gaussians initialized from HQ PDB statistics:
  g_α(d) = exp(-(d - μ_α)² / (2σ_α²))    μ_α=6.19Å, σ_α=0.19Å
  g_β(d) = G(d; μ1, σ1) + G(d; μ2, σ2)   μ1=5.79Å, σ1=0.87Å, μ2=10.68Å, σ2=1.78Å

Basin surface indices (from basins analysis on HQ data, our φ convention):
  HELIX_IDX = 1   (surface peak: θ≈92.5°, φ≈-55°)
  SHEET_IDX = 0   (surface peak: θ≈117.5°, φ≈+155°)

Parameters:
  E_hb(α): λ_α (learned) + μ_α, σ_α (fixed from PDB) = 1 learned
  E_hb(β): λ_β (learned) + μ_β1, σ_β1, μ_β2, σ_β2 (fixed from PDB) = 1 learned
  Total: 2 learned parameters + 6 fixed buffers
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calphaebm.models.learnable_buffers import reg
from calphaebm.utils.logging import get_logger

logger = get_logger()


def _inv_softplus(y: float, eps: float = 1e-6) -> float:
    y = float(max(y, eps))
    return float(np.log(np.expm1(y)))


# Basin surface indices (from basins analysis on HQ data)
#   Basin 0: Sheet  (θ≈117.5°, φ≈+155° in our convention)
#   Basin 1: Helix  (θ≈92.5°,  φ≈-55°  in our convention)
#   Basin 2: PPII   (θ≈92.5°,  φ≈-85°  in our convention)
#   Basin 3: Turn   (θ≈122.5°, φ≈+105° in our convention)
HELIX_BASIN_IDX = 1
SHEET_BASIN_IDX = 0


class HBondHelix(nn.Module):
    """Helical H-bond energy: E_hb(α).

    E = -λ_α · (1/L) · Σ_i [p_helix(i) · p_helix(i+3)] · g_α(d(Cα_i, Cα_{i+4}))

    Uses product of 2 endpoint probabilities (not 4 consecutive) to avoid
    p^4 sparsity that suppresses the signal. The distance constraint already
    ensures the backbone is helical between the endpoints.

    Parameters: 1 learned (λ_α) + 2 fixed (μ_α, σ_α from HQ PDB data)
    """

    def __init__(
        self,
        init_mu: float = 6.19,  # HQ data: mean d(i,i+4) for helices (3,252 structures)
        init_sigma: float = 0.19,  # HQ data: std of helical d(i,i+4)
        init_lambda: float = 1.0,
        learn_geometry: bool = False,
    ):
        super().__init__()
        # PDB statistics — optionally learnable
        reg(self, "mu", torch.tensor(init_mu, dtype=torch.float32), learnable=learn_geometry)
        reg(self, "sigma", torch.tensor(max(init_sigma, 0.05), dtype=torch.float32), learnable=learn_geometry)
        # Learned: overall H-bond strength
        self._lambda_raw = nn.Parameter(torch.tensor(_inv_softplus(init_lambda), dtype=torch.float32))
        logger.debug(
            "HBondHelix initialized: μ=%.2fÅ (fixed), σ=%.2fÅ (fixed), λ=%.1f (learned)",
            init_mu,
            init_sigma,
            init_lambda,
        )

    @property
    def lambda_hb(self) -> torch.Tensor:
        return F.softplus(self._lambda_raw) + 1e-6

    def forward(
        self,
        p_helix: torch.Tensor,  # (B, N) helix probability per position
        R: torch.Tensor,  # (B, L, 3) Cα coordinates
        normalize_by_length: bool = True,
        lengths: torch.Tensor | None = None,  # (B,) actual chain lengths
    ) -> torch.Tensor:
        """Compute helical H-bond energy.

        Args:
            p_helix: (B, N) probability each position is helical (from softmax of basins).
                     N = number of positions where basin assignment is defined.
            R: (B, L, 3) Cα coordinates.
            normalize_by_length: divide by chain length.
            lengths: (B,) actual chain lengths. If provided, normalizes by actual
                     lengths instead of padded L.

        Returns:
            (B,) energy values.
        """
        B, L, _ = R.shape
        N = p_helix.shape[1]

        if N < 4 or L < 5:
            return torch.zeros(B, device=R.device)

        p2 = p_helix[:, :-3] * p_helix[:, 3:]  # (B, N-3) endpoints only

        n_hb = min(N - 3, L - 5)
        if n_hb <= 0:
            return torch.zeros(B, device=R.device)

        p2 = p2[:, :n_hb]  # (B, n_hb)

        # Distances: d(residue k+1, residue k+5) for k = 0..n_hb-1
        diff = R[:, 1 : n_hb + 1] - R[:, 5 : n_hb + 5]  # (B, n_hb, 3)
        d_i4 = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)  # (B, n_hb)

        # Gaussian distance function
        g = torch.exp(-0.5 * ((d_i4 - self.mu) / self.sigma) ** 2)  # (B, n_hb)

        # H-bond signal: product of backbone conformation and distance
        signal = p2 * g  # (B, n_hb)

        # Smooth switch: x² / (x² + τ²)
        tau_sq = 0.01**2
        switch = signal.detach() ** 2 / (signal.detach() ** 2 + tau_sq)  # (B, n_hb)
        E_per_pos = signal * switch  # (B, n_hb)

        E = -self.lambda_hb * E_per_pos.sum(dim=-1)  # (B,)
        if normalize_by_length:
            if lengths is not None:
                E = E / lengths.float().clamp(min=1.0)
            else:
                E = E / float(max(L, 1))

        return E


class HBondSheet(nn.Module):
    """Sheet H-bond energy: E_hb(β) with two Gaussians.

    E = -λ_β · (1/L) · Σ_{i,j:|i-j|>4} p_ext(i) · p_ext(j) · g_β(d_ij)

    g_β(d) = G(d; μ1, σ1) + G(d; μ2, σ2)

    Two Gaussians capture the bimodal sheet Cα-Cα distance distribution:
      Peak 1 (μ1≈5.8Å, σ1≈0.88Å): anti-parallel β-sheet, closest Cα-Cα
      Peak 2 (μ2≈8.5Å, σ2≈1.2Å):  parallel β-sheet / wider strand spacing

    Parameters: 1 learned (λ_β) + 4 fixed (μ1, σ1, μ2, σ2 from HQ PDB data)
    """

    def __init__(
        self,
        init_mu1: float = 5.79,  # HQ data: anti-parallel peak
        init_sigma1: float = 0.87,
        init_mu2: float = 10.68,  # HQ data: parallel/wider peak
        init_sigma2: float = 1.78,
        init_lambda: float = 1.0,
        min_seq_sep: int = 5,
        max_dist: float = 12.0,
        learn_geometry: bool = False,
    ):
        super().__init__()
        # PDB statistics — optionally learnable
        reg(self, "mu1", torch.tensor(init_mu1, dtype=torch.float32), learnable=learn_geometry)
        reg(self, "sigma1", torch.tensor(max(init_sigma1, 0.05), dtype=torch.float32), learnable=learn_geometry)
        reg(self, "mu2", torch.tensor(init_mu2, dtype=torch.float32), learnable=learn_geometry)
        reg(self, "sigma2", torch.tensor(max(init_sigma2, 0.05), dtype=torch.float32), learnable=learn_geometry)
        # Learned: overall H-bond strength
        self._lambda_raw = nn.Parameter(torch.tensor(_inv_softplus(init_lambda), dtype=torch.float32))
        self.min_seq_sep = int(min_seq_sep)
        self.max_dist = float(max_dist)

        logger.debug(
            "HBondSheet initialized: μ1=%.2fÅ σ1=%.2fÅ, μ2=%.2fÅ σ2=%.2fÅ (all fixed), λ=%.1f (learned), min_sep=%d",
            init_mu1,
            init_sigma1,
            init_mu2,
            init_sigma2,
            init_lambda,
            min_seq_sep,
        )

    @property
    def lambda_hb(self) -> torch.Tensor:
        return F.softplus(self._lambda_raw) + 1e-6

    def forward(
        self,
        p_ext: torch.Tensor,  # (B, N) sheet probability per position
        R: torch.Tensor,  # (B, L, 3) Cα coordinates
        r: torch.Tensor,  # (B, L, K) neighbor distances from topk
        j_idx: torch.Tensor,  # (B, L, K) neighbor indices from topk
        normalize_by_length: bool = True,
        lengths: torch.Tensor | None = None,  # (B,) actual chain lengths
    ) -> torch.Tensor:
        """Compute sheet H-bond energy with two-Gaussian distance function."""
        B, L, K = r.shape
        N = p_ext.shape[1]

        if N < 1 or L < 6:
            return torch.zeros(B, device=R.device)

        # p_ext is defined for N positions starting at residue 1 (0-based).
        # Pad to full length L with zeros at boundaries.
        p_full = torch.zeros(B, L, device=R.device, dtype=p_ext.dtype)
        offset = 1  # basin assignment starts at residue 1
        end = min(offset + N, L)
        p_full[:, offset:end] = p_ext[:, : end - offset]

        p_i = p_full  # (B, L)

        j_safe = j_idx.clamp(0, L - 1)
        p_j = torch.gather(p_full, 1, j_safe.view(B, -1)).view(B, L, K)  # (B, L, K)

        valid = (r < self.max_dist - 1e-4).float()

        i_idx = torch.arange(L, device=R.device).view(1, L, 1).expand(B, L, K)
        seq_sep = (i_idx - j_safe).abs().float()
        sep_mask = (seq_sep > self.min_seq_sep).float()

        mask = valid * sep_mask  # (B, L, K)

        r_clamped = r.clamp(max=self.max_dist)
        g1 = torch.exp(-0.5 * ((r_clamped - self.mu1) / self.sigma1) ** 2)
        g2 = torch.exp(-0.5 * ((r_clamped - self.mu2) / self.sigma2) ** 2)
        g = (g1 + g2) * mask  # (B, L, K)

        signal = p_i.unsqueeze(-1) * p_j * g  # (B, L, K)

        tau_sq = 0.02**2
        switch = signal.detach() ** 2 / (signal.detach() ** 2 + tau_sq)
        E_per_pair = signal * switch
        E_per_residue = E_per_pair.sum(dim=-1)  # (B, L)

        E = -self.lambda_hb * E_per_residue.sum(dim=-1)  # (B,)
        if normalize_by_length:
            if lengths is not None:
                E = E / lengths.float().clamp(min=1.0)
            else:
                E = E / float(max(L, 1))

        return E


def compute_basin_probabilities(
    basin_potentials: nn.ModuleList,
    theta_deg: torch.Tensor,  # (B, N)
    phi_deg: torch.Tensor,  # (B, N)
    context: torch.Tensor,  # (B, N, ctx_dim) — AA context embeddings
    A: torch.Tensor,  # (K, ctx_dim) — mixture weight matrix
    a: torch.Tensor,  # (K,) — mixture bias
) -> torch.Tensor:
    """Compute per-position basin probabilities from E_ram surfaces.

    This reuses the same mixture-of-basins computation as E_ram, but returns
    the softmax probabilities instead of the energy.

    Returns:
        (B, N, K) probability of each basin at each position.
    """
    theta_deg = torch.nan_to_num(theta_deg, nan=0.0)
    phi_deg = torch.nan_to_num(phi_deg, nan=0.0)
    context = torch.nan_to_num(context, nan=0.0)

    # Logits: (B, N, K)
    logits = torch.einsum("kf,bnf->bnk", A, context) + a
    logits = torch.clamp(logits, -10.0, 10.0)

    # Basin energies: (B, N, K)
    with torch.no_grad():
        U = torch.stack([U_k(theta_deg, phi_deg) for U_k in basin_potentials], dim=-1)
    U = U.detach()
    U = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    U = torch.clamp(U, max=50.0)

    # s_k = logits_k - U_k → softmax gives basin probabilities
    s = logits - U  # (B, N, K)
    probs = torch.softmax(s, dim=-1)  # (B, N, K)

    return probs
