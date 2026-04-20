# src/calphaebm/training/losses/native_gap_loss.py

"""Native gap loss — enforces a deep energy well around native structures.

Two modes:

mode='continuous' (default):
    L = exp(-gap / T(σ))      where T(σ) = T_base × σ
    Always provides gradient toward deeper wells.
    Diminishing returns at very deep wells (self-regulating).
    σ-dependent temperature: larger perturbations demand larger gaps.

mode='margin' (legacy):
    L = relu(margin - gap)    where gap = E(R_pert) - E(R_native)
    Binary: zero when gap >= margin, active when gap < margin.
    Problem: saturates once margin is met → no further well-deepening.

The continuous mode solves the fundamental DSM vs stability conflict:
DSM controls gradient direction, continuous native gap controls energy scale.
They're complementary — DSM shapes where the well is, native gap deepens it.

Negative data: IC perturbations with per-sample σ drawn from
LogUniform[sigma_min, sigma_max]. Per-sample (not per-batch) sigma gives
stable loss with low variance — every batch samples the full σ range.

  θ perturbed with 0.2×σ (bond angles are stiffer)
  φ perturbed with σ (dihedrals are the main DOF)
  Reconstructed via NeRF → bonds always 3.8Å

With sigma_max=2.0 rad (~115°), negatives include fully unfolded structures
with broken secondary structure and wrong contacts — covering all failure
modes that basin and pack-C losses were designed for.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.utils.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Helpers for steric-safe perturbation
# ---------------------------------------------------------------------------


def _perturb_and_reconstruct(
    theta: torch.Tensor,
    phi: torch.Tensor,
    anchor: torch.Tensor,
    sigmas: torch.Tensor,
    valid_t: torch.Tensor | None,
    valid_p: torch.Tensor | None,
) -> torch.Tensor:
    """Perturb ICs and reconstruct via NeRF.  Pure no-grad helper."""
    sig_theta = (0.2 * sigmas).unsqueeze(-1)  # (B, 1)
    sig_phi = sigmas.unsqueeze(-1)  # (B, 1)

    noise_theta = torch.randn_like(theta)
    noise_phi = torch.randn_like(phi)

    if valid_t is not None:
        noise_theta = noise_theta * valid_t.float()
    if valid_p is not None:
        noise_phi = noise_phi * valid_p.float()

    theta_pert = (theta + sig_theta * noise_theta).clamp(0.01, math.pi - 0.01)
    phi_pert = phi + sig_phi * noise_phi
    phi_pert = (phi_pert + math.pi) % (2 * math.pi) - math.pi

    return nerf_reconstruct(theta_pert, phi_pert, anchor)


def _batch_min_nonbonded_dist(
    R: torch.Tensor,
    lengths: torch.Tensor | None,
    seq_sep: int = 3,
) -> torch.Tensor:
    """Compute min nonbonded distance per sample.  Returns (B,) tensor."""
    B, L, _ = R.shape
    device = R.device

    idx = torch.arange(L, device=device)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    nonbond = sep > seq_sep  # (L, L)

    diff = R.unsqueeze(2) - R.unsqueeze(1)  # (B, L, L, 3)
    d = torch.sqrt((diff**2).sum(-1) + 1e-8)  # (B, L, L)

    # Mask bonded pairs and self
    d = d.masked_fill(~nonbond.unsqueeze(0), float("inf"))

    # Mask padding atoms
    if lengths is not None:
        valid = idx.unsqueeze(0) < lengths.unsqueeze(1)  # (B, L)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, L, L)
        d = d.masked_fill(~pair_valid, float("inf"))

    return d.reshape(B, -1).amin(dim=1)  # (B,)


def native_gap_loss(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    margin: float = 0.5,
    sigma_min: float = 0.05,
    sigma_max: float = 0.50,
    mode: str = "continuous",
    T_base: float = 5.0,
    return_diag: bool = False,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor | Tuple[torch.Tensor, dict]:
    """Full-model native gap loss with continuous or margin mode.

    Each sample in the batch gets its own σ drawn log-uniformly from
    [sigma_min, sigma_max]. This gives stable, low-variance loss signals
    compared to single-σ-per-batch.

    Args:
        model:       TotalEnergy model (all terms, all gates).
        R:           (B, L, 3) native Cα coordinates.
        seq:         (B, L) amino acid indices.
        margin:      Required energy gap (margin mode only). Default 0.5.
        sigma_min:   Minimum perturbation sigma (radians). Default 0.05.
        sigma_max:   Maximum perturbation sigma (radians). Default 0.50.
        mode:        'continuous' (default) or 'margin' (legacy).
        T_base:      Base temperature for continuous mode. Effective T = T_base × σ.
                     Larger T_base → gentler push. Default 5.0.
        return_diag: If True, return (loss, diag_dict).

    Returns:
        Scalar loss (or (loss, diag) if return_diag=True).

    Mode details:
        continuous: loss = exp(-gap_per_res / (T_base × σ)).mean()
                    Always active. Self-regulating via exponential decay.
        margin:     loss = relu(margin - gap).mean()
                    Zero when gap >= margin. Saturates.
    """
    z = torch.zeros((), device=R.device, dtype=R.dtype)
    B, L, _ = R.shape

    if B == 0:
        return (z, {}) if return_diag else z

    try:
        # ── Per-sample sigma from LogUniform ──────────────────────────
        log_sigma = torch.empty(B, device=R.device, dtype=R.dtype).uniform_(math.log(sigma_min), math.log(sigma_max))
        sigmas = log_sigma.exp()  # (B,)

        # ── Build perturbed structures in (theta, phi) space ──────────
        # After NeRF reconstruction, any sample with min nonbonded dist < 1Å
        # is resampled with halved sigma (up to 3 retries × 3 halvings).
        with torch.no_grad():
            theta, phi = coords_to_internal(R)  # (B, L-2), (B, L-3)
            anchor = extract_anchor(R)  # (B, 3, 3)

            # Validity masks for padding
            if lengths is not None:
                idx_t = torch.arange(theta.shape[1], device=R.device)
                idx_p = torch.arange(phi.shape[1], device=R.device)
                valid_t = idx_t.unsqueeze(0) < (lengths.unsqueeze(1) - 2)
                valid_p = idx_p.unsqueeze(0) < (lengths.unsqueeze(1) - 3)
            else:
                valid_t = valid_p = None

            # Per-sample effective sigma (may be halved for retries)
            eff_sigmas = sigmas.clone()  # (B,)

            R_pert = _perturb_and_reconstruct(
                theta,
                phi,
                anchor,
                eff_sigmas,
                valid_t,
                valid_p,
            )

            # Check min nonbonded dist per sample; retry failing ones
            MAX_RETRIES = 3
            MAX_HALVINGS = 3
            for _halving in range(MAX_HALVINGS):
                bad_mask = _batch_min_nonbonded_dist(R_pert, lengths) < 1.0  # (B,)
                if not bad_mask.any():
                    break
                for _retry in range(MAX_RETRIES):
                    if not bad_mask.any():
                        break
                    # Resample only failing samples with fresh noise at current sigma
                    R_retry = _perturb_and_reconstruct(
                        theta,
                        phi,
                        anchor,
                        eff_sigmas,
                        valid_t,
                        valid_p,
                    )
                    # Check which retries passed
                    retry_ok = _batch_min_nonbonded_dist(R_retry, lengths) >= 1.0
                    # Replace failing samples that now pass
                    replace = bad_mask & retry_ok
                    if replace.any():
                        R_pert[replace] = R_retry[replace]
                        bad_mask = bad_mask & ~replace
                if bad_mask.any():
                    # Halve sigma for remaining failures
                    eff_sigmas[bad_mask] *= 0.5
                    R_retry = _perturb_and_reconstruct(
                        theta,
                        phi,
                        anchor,
                        eff_sigmas,
                        valid_t,
                        valid_p,
                    )
                    R_pert[bad_mask] = R_retry[bad_mask]
                    bad_mask = _batch_min_nonbonded_dist(R_pert, lengths) < 1.0

            # Update sigmas to reflect any halvings (for T computation)
            sigmas = eff_sigmas

        # ── Evaluate full model ───────────────────────────────────────
        E_native = model(R.detach(), seq, lengths=lengths)  # (B,)
        E_pert = model(R_pert, seq, lengths=lengths)  # (B,)

        if not torch.isfinite(E_native).all() or not torch.isfinite(E_pert).all():
            return (z, {}) if return_diag else z

        gap = E_pert - E_native  # (B,) positive = correct

        # ── Compute loss ──────────────────────────────────────────────
        if mode == "continuous":
            # Use real lengths for per-residue normalization
            L_real = lengths.float() if lengths is not None else torch.full((B,), float(L), device=R.device)
            gap_per_res = gap / L_real  # (B,)
            T = T_base * sigmas  # (B,) per-sample temperature
            # Clamp to prevent explosion when gap is negative (perturbed < native)
            loss = torch.exp(-gap_per_res / T).clamp(max=10.0).mean()
        else:
            # Legacy margin mode
            loss = F.relu(margin - gap).mean()

        if not torch.isfinite(loss):
            return (z, {}) if return_diag else z

        # ── Diagnostics ───────────────────────────────────────────────
        if return_diag:
            with torch.no_grad():
                gap_det = gap.detach()
                sig_det = sigmas.detach()
                diag = {
                    "sigma": float(sig_det.mean().item()),
                    "E_native": float(E_native.mean().item()) / L,
                    "E_pert": float(E_pert.mean().item()) / L,
                    "gap_mean": float(gap_det.mean().item()),
                    "gap_min": float(gap_det.min().item()),
                    "ok": bool(gap_det.min().item() > 0),
                    "mode": mode,
                }
                if mode == "continuous":
                    T_det = T_base * sig_det
                    loss_per = torch.exp(-(gap_det / L) / T_det)
                    diag["T_mean"] = float(T_det.mean().item())
                    diag["loss_per_sample_mean"] = float(loss_per.mean().item())
                    diag["loss_per_sample_min"] = float(loss_per.min().item())
                    diag["loss_per_sample_max"] = float(loss_per.max().item())
                else:
                    diag["margin"] = margin
            return loss, diag

        return loss

    except Exception as exc:
        logger.warning("native_gap_loss failed: %s", exc)
        return (z, {}) if return_diag else z
