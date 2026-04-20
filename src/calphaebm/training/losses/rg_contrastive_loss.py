"""Rg contrastive loss — prevents compaction/swelling artifacts.

Scales coordinates by α ~ U(α_min, α_max) toward/away from COM,
then penalises E(native) > E(scaled) via softplus.

Unlike the old Rg gate (which modified the energy function during training
but not at inference), this is a pure training loss — E_θ is identical at
training and inference time.

Usage:
    from calphaebm.training.losses.rg_contrastive_loss import rg_contrastive_loss

    loss_rg = rg_contrastive_loss(model, R, seq, lengths=lengths)
"""

import torch
import torch.nn.functional as F

from calphaebm.utils.logging import get_logger

logger = get_logger()


def rg_contrastive_loss(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    lengths: torch.Tensor | None = None,
    alpha_min: float = 0.75,
    alpha_max: float = 1.25,
) -> torch.Tensor:
    """Compute Rg contrastive loss.

    Scales each structure's coordinates by α ~ U(α_min, α_max) relative to
    its center of mass, then returns softplus(E_native - E_scaled).

    When α < 1 the structure is compacted; when α > 1 it is swollen.
    The loss penalises whenever the scaled structure has lower energy
    than native — whether compacted or swollen.

    Args:
        model:     Energy model (returns per-structure E, already normalised).
        R:         (B, L, 3) native Cα coordinates.
        seq:       (B, L) integer residue types.
        lengths:   (B,) real chain lengths for padding-aware COM. None = no padding.
        alpha_min: Lower bound of scaling factor (< 1 = compaction).
        alpha_max: Upper bound of scaling factor (> 1 = swelling).

    Returns:
        Scalar loss (mean over batch).
    """
    B, L, _ = R.shape
    device = R.device

    # ── Padding-aware center of mass ──────────────────────────────────────
    with torch.no_grad():
        if lengths is not None:
            mask = (
                (torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(2)
            )  # (B, L, 1)
            n_atoms = lengths.float().clamp(min=1.0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            com = (R * mask).sum(dim=1, keepdim=True) / n_atoms  # (B, 1, 3)
        else:
            mask = None
            com = R.mean(dim=1, keepdim=True)  # (B, 1, 3)

        # ── Sample α ~ U(α_min, α_max) per structure ─────────────────────
        alpha = alpha_min + (alpha_max - alpha_min) * torch.rand(B, 1, 1, device=device)  # (B, 1, 1)

        # ── Scale real atoms; leave padding unchanged ─────────────────────
        R_scaled = com + alpha * (R - com)
        if mask is not None:
            R_scaled = R * (1.0 - mask) + R_scaled * mask

    # ── Energy on native and scaled (both need grad for backprop) ─────────
    E_native = model(R, seq, lengths=lengths).mean()
    E_scaled = model(R_scaled, seq, lengths=lengths).mean()

    # softplus(E_native - E_scaled) → 0 when native is lower
    loss = F.softplus(E_native - E_scaled)

    return loss
