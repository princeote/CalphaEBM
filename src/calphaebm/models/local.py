"""
Local backbone energy — sliding window θφ MLP.
===============================================

A sliding window of W consecutive residues, each represented by
(cos θ, sin φ, cos φ) + AA embedding.

W=8 (default):  two helical turns, captures medium-range correlations.

Architecture: input_dim → hidden_dims → 1, bounded via scale*tanh(mlp/scale)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calphaebm.geometry.internal import bond_angles, torsions
from calphaebm.models.embeddings import AAEmbedding
from calphaebm.models.rama_gate import RamaValidityGate
from calphaebm.utils.logging import get_logger

logger = get_logger()

_MISSING_LENGTHS_WARNED = False


def _check_lengths(R: torch.Tensor, lengths: torch.Tensor | None, caller: str) -> None:
    global _MISSING_LENGTHS_WARNED
    if lengths is None and R.shape[0] > 1 and not _MISSING_LENGTHS_WARNED:
        logger.warning(
            "[%s] lengths=None with batch_size=%d — padding atoms will corrupt "
            "energy. Pass lengths to all model calls during training.",
            caller,
            R.shape[0],
        )
        _MISSING_LENGTHS_WARNED = True


def _inv_softplus(y: float, eps: float = 1e-8) -> float:
    y = float(max(y, eps))
    return float(np.log(np.expm1(y)))


def _cat(*xs: torch.Tensor) -> torch.Tensor:
    return torch.cat(xs, dim=-1)


class _StableMLP(nn.Module):
    """MLP with bounded output via scaled tanh."""

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int = 1, scale: float = 2.0):
        super().__init__()
        self.scale = scale
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.mlp(x) / self.scale)


class LocalEnergy(nn.Module):
    """Local backbone energy using a W-mer θφ sliding window MLP."""

    def __init__(
        self,
        window_size: int = 8,
        num_aa: int = 20,
        emb_dim: int = 16,
        hidden_dims: tuple[int, ...] = (64, 32),
        init_weight: float = 1.0,
        weight_eps: float = 1e-6,
        normalize_by_length: bool = True,
        secondary_data_dir: str = "analysis/secondary_analysis/data",
        **legacy_kwargs,
    ):
        super().__init__()

        for k in [
            "data_dir",
            "init_bond_spring",
            "bond_length_ideal",
            "init_theta_theta_weight",
            "init_delta_phi_weight",
            "init_phi_phi_weight",
            "smooth_delta_phi",
            "delta_phi_smooth_sigma_deg",
            "theta_theta_hidden",
            "phi_phi_hidden",
            "theta_theta_raw_scale",
            "delta_phi_raw_scale",
            "target_contribution",
            "bond_length_eps",
        ]:
            legacy_kwargs.pop(k, None)
        if legacy_kwargs:
            logger.warning("LocalEnergy: unexpected kwargs ignored: %s", list(legacy_kwargs.keys()))

        self.W = int(window_size)
        self.weight_eps = float(weight_eps)
        self.normalize_by_length = bool(normalize_by_length)
        self._emb_dim = emb_dim

        self.emb = AAEmbedding(num_aa=num_aa, dim=emb_dim)
        ctx_dim = self.W * emb_dim

        self._lambda_raw = nn.Parameter(
            torch.tensor(_inv_softplus(init_weight, eps=self.weight_eps), dtype=torch.float32)
        )

        angle_feat_dim = 3 * self.W
        self.f_theta_phi = _StableMLP(
            angle_feat_dim + ctx_dim,
            hidden_dims=hidden_dims,
            out_dim=1,
            scale=2.0,
        )

        n_mlp = sum(p.numel() for p in self.f_theta_phi.parameters())
        n_emb = sum(p.numel() for p in self.emb.parameters())
        self._init_summary = {
            "window_size": self.W,
            "total_params": n_mlp + n_emb + 1,
            "mlp_params": n_mlp,
            "emb_params": n_emb,
            "input_dim": angle_feat_dim + ctx_dim,
            "hidden_dims": hidden_dims,
        }
        logger.info(
            "LocalEnergy: %d-mer, input_dim=%d, hidden=%s, params=%d",
            self.W,
            angle_feat_dim + ctx_dim,
            hidden_dims,
            n_mlp + n_emb + 1,
        )

        # ── Rama validity gate — peaks from basin surface files ───────────
        self.rama_gate = RamaValidityGate.from_data_dir(secondary_data_dir)

    @property
    def weight(self) -> torch.Tensor:
        return F.softplus(self._lambda_raw) + self.weight_eps

    @property
    def theta_phi_weight(self) -> torch.Tensor:
        return self.weight

    def theta_phi_energy(self, R: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = R.shape
        W = self.W
        min_len = W + 3
        if L < min_len:
            return torch.zeros(B, device=R.device)

        theta = bond_angles(R)  # (B, L-2)
        phi = torsions(R)  # (B, L-3)

        N = phi.shape[1] - (W - 1)
        if N <= 0:
            return torch.zeros(B, device=R.device)

        # ── Angle features ───────────────────────────────────────────────
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        angle_parts = []
        for k in range(W):
            angle_parts.append(cos_theta[:, k : k + N])
            angle_parts.append(sin_phi[:, k : k + N])
            angle_parts.append(cos_phi[:, k : k + N])
        angle_feats = torch.stack(angle_parts, dim=-1)
        angle_feats = torch.nan_to_num(angle_feats, nan=0.0)

        # ── AA embedding context ─────────────────────────────────────────
        e = self.emb(seq)
        ctx_parts = [e[:, k : k + N] for k in range(W)]
        ctx = _cat(*ctx_parts)
        ctx = torch.nan_to_num(ctx, nan=0.0)

        # ── MLP forward ─────────────────────────────────────────────────
        x = _cat(angle_feats, ctx)
        mlp_out = self.f_theta_phi(x).squeeze(-1)
        mlp_out = torch.nan_to_num(mlp_out, nan=0.0)

        # Mask padding
        if lengths is not None:
            valid = torch.arange(N, device=R.device).unsqueeze(0) < (lengths.unsqueeze(1) - W - 2)
            mlp_out = mlp_out * valid.float()

        # ── Rama validity gate ───────────────────────────────────────────
        # Per-position validity, then mean over W consecutive → per-window
        with torch.no_grad():
            n_pos = min(theta.shape[1], phi.shape[1])
            v_pos = self.rama_gate(theta[:, :n_pos], phi[:, :n_pos])  # (B, n_pos)
            gate_parts = [v_pos[:, k : k + N] for k in range(W)]
            win_gate = torch.stack(gate_parts, dim=-1).mean(dim=-1)  # (B, N)

        mlp_out = mlp_out * win_gate

        # Sum and normalize
        E = self.weight * mlp_out.sum(dim=-1)
        if self.normalize_by_length:
            if lengths is not None:
                E = E / (lengths.float() - W - 2).clamp(min=1.0)
            else:
                E = E / float(max(1, N))

        return E

    def forward(self, R: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        _check_lengths(R, lengths, "LocalEnergy")
        return self.theta_phi_energy(R, seq, lengths=lengths)

    def forward_learned(self, R: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(R, seq, lengths=lengths)
