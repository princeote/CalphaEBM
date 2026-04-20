"""Contrastive loss functions for energy-based training."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct

# PDB-derived ratio: std(θ) / std(φ) across training chains
THETA_PHI_RATIO = 0.161


def _sample_sigmas(B: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    """Per-sample log-uniform σ: returns (B, 1) for broadcasting."""
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    return torch.empty(B, 1, device=device).uniform_(log_min, log_max).exp()


def packing_contrastive_loss(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    margin: float = 0.5,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
    mode: str = "continuous",
    T_base: float = 2.0,
    return_diag: bool = False,
    lengths: torch.Tensor | None = None,
):
    """Packing contrastive loss for full phase — keeps MLP signal alive.

    Dynamics-focused: 100% IC-perturbed negatives (same sequence, perturbed
    backbone).  Each structure gets its own per-sample log-uniform σ so every
    batch covers the full perturbation range.

    Two modes:
      'continuous' (default): loss = exp(-gap / T_base).mean()
      'margin' (legacy):      loss = relu(margin - gap).mean()

    Args:
        model:     TotalEnergy model with .packing and .gate_packing.
        R:         (B, L, 3) native Cα coordinates.
        seq:       (B, L) amino acid indices.
        margin:    Required energy gap. Default 0.5.
        sigma_min: Min IC noise sigma (radians). Default 0.05.
        sigma_max: Max IC noise sigma (radians). Default 2.0.
        mode:      'continuous' or 'margin'.
        T_base:    Temperature for continuous mode.
        return_diag: If True, return (loss, diagnostics_dict).
        lengths:   (B,) real chain lengths for padding-aware computation.

    Returns:
        Scalar loss. Detached zero if unsafe or no packing term.
    """
    if not hasattr(model, "packing") or model.packing is None:
        z = torch.zeros((), device=R.device, dtype=R.dtype)
        return (z, {}) if return_diag else z

    B = R.shape[0]
    if B < 1:
        z = torch.zeros((), device=R.device, dtype=R.dtype)
        return (z, {}) if return_diag else z

    gate = float(getattr(model, "gate_packing", 1.0))
    diag: dict = {"margin": margin}

    try:
        with torch.no_grad():
            theta, phi = coords_to_internal(R)
            anchor = extract_anchor(R)

            sigmas = _sample_sigmas(B, sigma_min, sigma_max, R.device)  # (B, 1)
            noise_t = THETA_PHI_RATIO * sigmas * torch.randn_like(theta)
            noise_p = sigmas * torch.randn_like(phi)

            # Mask noise at padding positions
            if lengths is not None:
                idx_t = torch.arange(theta.shape[1], device=R.device)
                idx_p = torch.arange(phi.shape[1], device=R.device)
                vt = idx_t.unsqueeze(0) < (lengths.unsqueeze(1) - 2)
                vp = idx_p.unsqueeze(0) < (lengths.unsqueeze(1) - 3)
                noise_t = noise_t * vt.float()
                noise_p = noise_p * vp.float()

            theta_n = (theta + noise_t).clamp(0.01, math.pi - 0.01)
            phi_n = phi + noise_p
            phi_n = (phi_n + math.pi) % (2 * math.pi) - math.pi
            R_neg = nerf_reconstruct(theta_n, phi_n, anchor)

        E_pos = gate * model.packing(R, seq, lengths=lengths)
        E_neg = gate * model.packing(R_neg, seq, lengths=lengths)

        if not (torch.isfinite(E_pos).all() and torch.isfinite(E_neg).all()):
            z = torch.zeros((), device=R.device, dtype=R.dtype)
            return (z, diag) if return_diag else z

        if return_diag:
            L_mean = float(lengths.float().mean().item()) if lengths is not None else float(R.shape[1])
            diag["E_clean"] = float(E_pos.mean().item()) / L_mean
            diag["E_pert"] = float(E_neg.mean().item()) / L_mean
            diag["gap_pert"] = diag["E_pert"] - diag["E_clean"]

        gap = E_neg - E_pos  # positive = correct

        if mode == "continuous":
            L_mean = float(lengths.float().mean().item()) if lengths is not None else float(R.shape[1])
            gap_per_res = gap / L_mean
            loss = torch.exp(-gap_per_res / T_base).mean()
        else:
            loss = F.relu(margin - gap).mean()

        safe_loss = loss if torch.isfinite(loss) else torch.zeros((), device=R.device, dtype=R.dtype)
        return (safe_loss, diag) if return_diag else safe_loss

    except Exception:
        z = torch.zeros((), device=R.device, dtype=R.dtype)
        return (z, diag) if return_diag else z


def contrastive_logistic_loss(E_pos: torch.Tensor, E_neg: torch.Tensor) -> torch.Tensor:
    """Contrastive loss using logistic function (softplus).

    Args:
        E_pos: Energy of positive (native) samples, shape (B,)
        E_neg: Energy of negative (distorted) samples, shape (B,)

    Returns:
        Scalar loss
    """
    if E_pos.shape != E_neg.shape:
        min_len = min(E_pos.shape[0], E_neg.shape[0])
        E_pos = E_pos[:min_len]
        E_neg = E_neg[:min_len]

    delta = torch.clamp(E_pos - E_neg, min=-50, max=50)
    loss = F.softplus(delta).mean()

    return loss
