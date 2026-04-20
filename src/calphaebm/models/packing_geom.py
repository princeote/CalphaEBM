# src/calphaebm/models/packing_geom.py
"""
ARCHIVED — Packing geometry MLP subterm (E_geom).

Removed from PackingEnergy because the 1633-parameter MLP dominated the
energy landscape (~64% of total energy by round 4 of run1). The MLP's
flexible function approximation absorbed too much of the training signal,
preventing the physics-based E_hp (23 params) from learning meaningful
pair preferences.

The balance loss was supposed to prevent this, but a naming bug
(--disable-subterms geom matched "geom" in _skip, hiding packing_geom
from balance) allowed E_geom to grow unchecked.

Even without the bug, the MLP has a structural advantage: 1633 params vs
23 for E_hp means it can always fit faster and dominate. Removing it
forces the model to learn through physically interpretable pair interactions.

Kept here for reference in case a constrained version is needed later.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from calphaebm.utils.logging import get_logger

logger = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry feature extractor
# ─────────────────────────────────────────────────────────────────────────────


def _cosine_switch(r: torch.Tensor, r_on: float, r_cut: float) -> torch.Tensor:
    """Smooth switch: 1 for r<=r_on, cosine ramp to 0 over (r_on, r_cut), 0 for r>=r_cut."""
    x = (r - r_on) / (r_cut - r_on)
    x = torch.clamp(x, 0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(torch.pi * x))


def _cosine_gate(r: torch.Tensor, r_on: float, r_off: float) -> torch.Tensor:
    """Gate: 0 for r<=r_on, cosine ramp to 1 over (r_on, r_off), 1 for r>=r_off."""
    x = (r - r_on) / (r_off - r_on)
    x = torch.clamp(x, 0.0, 1.0)
    return 0.5 * (1.0 - torch.cos(torch.pi * x))


class GeometryFeatures(nn.Module):
    """
    Bidirectional per-residue geometry descriptor (v2).

    All outputs are differentiable w.r.t. R.

    Feature vector (7 scalars per residue, all ~O(1)):
        [n_dev_signed, n_dev_abs, shell_ratio, shell_asymmetry,
         mean_r_dev, std_r_norm, n_frac_band]
    """

    N_FEATURES: int = 7

    COORD_R_HALF = 7.0
    COORD_TAU = 1.0

    def __init__(
        self,
        tight_cut: float = 6.0,
        medium_cut: float = 8.0,
        loose_cut: float = 10.0,
        sig_tau: float = 0.5,
        short_gate_on: float = 4.5,
        short_gate_off: float = 5.0,
        r_on: float = 8.0,
        r_cut: float = 10.0,
        max_dist: float = 10.0,
        norm_n_tight: float = 5.0,
        norm_n_medium: float = 10.0,
        norm_n_loose: float = 15.0,
        norm_mean_r_centre: float = 7.0,
        norm_mean_r_scale: float = 2.0,
        norm_std_r: float = 1.5,
        norm_inv_sq: float = 5.0,
        n_mean_per_aa: "list | None" = None,
        n_lo_per_aa: "list | None" = None,
        n_hi_per_aa: "list | None" = None,
        num_aa: int = 20,
    ):
        super().__init__()
        self.tight_cut = tight_cut
        self.medium_cut = medium_cut
        self.loose_cut = loose_cut
        self.sig_tau = sig_tau
        self.short_gate_on = short_gate_on
        self.short_gate_off = short_gate_off
        self.r_on = r_on
        self.r_cut = r_cut
        self.max_dist = max_dist

        self.register_buffer("norm_n_tight", torch.tensor(norm_n_tight, dtype=torch.float32))
        self.register_buffer("norm_n_medium", torch.tensor(norm_n_medium, dtype=torch.float32))
        self.register_buffer("norm_n_loose", torch.tensor(norm_n_loose, dtype=torch.float32))
        self.register_buffer("norm_mean_r_centre", torch.tensor(norm_mean_r_centre, dtype=torch.float32))
        self.register_buffer("norm_mean_r_scale", torch.tensor(norm_mean_r_scale, dtype=torch.float32))
        self.register_buffer("norm_std_r", torch.tensor(norm_std_r, dtype=torch.float32))
        self.register_buffer("norm_inv_sq", torch.tensor(norm_inv_sq, dtype=torch.float32))

        if n_mean_per_aa is not None:
            _n_mean = torch.tensor(n_mean_per_aa[:num_aa], dtype=torch.float32)
        else:
            _n_mean = torch.full((num_aa,), 4.0, dtype=torch.float32)
        if n_lo_per_aa is not None:
            _n_lo = torch.tensor(n_lo_per_aa[:num_aa], dtype=torch.float32)
        else:
            _n_lo = torch.full((num_aa,), 0.5, dtype=torch.float32)
        if n_hi_per_aa is not None:
            _n_hi = torch.tensor(n_hi_per_aa[:num_aa], dtype=torch.float32)
        else:
            _n_hi = torch.full((num_aa,), 7.5, dtype=torch.float32)

        self.register_buffer("n_mean_aa", _n_mean)
        self.register_buffer("n_lo_aa", _n_lo)
        self.register_buffer("n_hi_aa", _n_hi)
        _n_scale = ((_n_hi - _n_lo) / 2.0).clamp(min=0.5)
        self.register_buffer("n_scale_aa", _n_scale)
        _n_band = (_n_hi - _n_lo).clamp(min=0.5)
        self.register_buffer("n_band_aa", _n_band)

    def forward(self, r: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        B, L, K = r.shape

        valid = r < self.max_dist - 1e-4
        valid_f = valid.float()
        n_valid = valid_f.sum(dim=-1).clamp(min=1)

        r_safe = r.clamp(max=self.max_dist)

        sw_short = _cosine_gate(r_safe, self.short_gate_on, self.short_gate_off)
        sw_short = sw_short * valid_f
        sw_long = _cosine_switch(r_safe, self.r_on, self.r_cut)

        tau = max(self.sig_tau, 1e-3)
        sig_tight = torch.sigmoid((self.tight_cut - r_safe) / tau) * sw_short
        sig_medium = torch.sigmoid((self.medium_cut - r_safe) / tau) * sw_short
        sig_loose = torch.sigmoid((self.loose_cut - r_safe) / tau) * sw_short * sw_long

        n_tight = sig_tight.sum(dim=-1)
        n_medium = sig_medium.sum(dim=-1)
        n_loose = sig_loose.sum(dim=-1)

        g_coord = torch.sigmoid((self.COORD_R_HALF - r_safe) / self.COORD_TAU)
        g_coord = g_coord * valid_f
        n_coord = g_coord.sum(dim=-1)

        mean_r = (r_safe * valid_f).sum(dim=-1) / n_valid
        diff_sq = (r_safe - mean_r.unsqueeze(-1)).pow(2) * valid_f
        std_r = (diff_sq.sum(dim=-1) / n_valid + 1e-4).sqrt()

        n_mean_i = self.n_mean_aa[seq]
        n_scale_i = self.n_scale_aa[seq]
        n_lo_i = self.n_lo_aa[seq]
        n_band_i = self.n_band_aa[seq]

        n_dev_signed = (n_coord - n_mean_i) / n_scale_i
        n_dev_abs = n_dev_signed.abs()
        shell_ratio = n_tight / n_medium.clamp(min=1.0)
        shell_asym = (n_tight - n_loose) / n_medium.clamp(min=1.0)
        mean_r_dev = (mean_r - self.norm_mean_r_centre) / self.norm_mean_r_scale.clamp(min=0.1)
        std_r_norm = std_r / self.norm_std_r.clamp(min=0.1)
        n_frac_band = (n_coord - n_lo_i) / n_band_i

        geom = torch.stack(
            [
                n_dev_signed,
                n_dev_abs,
                shell_ratio,
                shell_asym,
                mean_r_dev,
                std_r_norm,
                n_frac_band,
            ],
            dim=-1,
        )

        return geom


class PackingMLP(nn.Module):
    """
    Small MLP: [seq_embed ‖ geom_features] → scalar energy per residue.

    Default parameter count (seq_dim=16, n_geom=7, h1=32, h2=16):
        seq_embed: 20*16 = 320
        fc1: (23)*32 + 32 = 768
        fc2: 32*16  + 16 = 528
        fc3: 16*1   +  1 =  17
        Total: 1,633 params
    """

    def __init__(
        self,
        num_aa: int = 20,
        n_geom: int = GeometryFeatures.N_FEATURES,
        seq_dim: int = 16,
        hidden1: int = 32,
        hidden2: int = 16,
    ):
        super().__init__()
        self.seq_embed = nn.Linear(num_aa, seq_dim, bias=False)
        self.fc1 = nn.Linear(seq_dim + n_geom, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.weight.data.mul_(0.1)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, one_hot: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        h_seq = self.seq_embed(one_hot)
        x = torch.cat([h_seq, geom], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        E_i = self.fc3(x).squeeze(-1)
        return E_i


__all__ = ["GeometryFeatures", "PackingMLP"]
