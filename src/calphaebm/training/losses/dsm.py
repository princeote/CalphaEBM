"""Denoising Score Matching (DSM) loss functions for energy-based models.

Two variants:

dsm_cartesian_loss  — legacy Cartesian DSM. Kept for ablations and DSM training
                      comparisons against IC. Noises R directly.

dsm_ic_loss         — IC DSM. The correct loss for run19+ training.
                      Noises (θ, φ) → reconstructs R via NeRF → bonds always 3.8Å.
                      Score target is in (θ, φ) space, not Cartesian.

Why IC DSM is correct for IC simulation
-----------------------------------------
If you simulate in (θ, φ) space but train with Cartesian DSM, there is a
fundamental train/inference mismatch:

  - Cartesian DSM trains the model to correct bond-length deviations (R noised
    along Cartesian axes can stretch bonds). The score dE/dR includes a large
    bond_spring component that dominates the residual.

  - IC simulation never has bond-length deviations. The model is used in a
    regime it was never trained on, and bond_spring provides no useful signal.

IC DSM fixes both problems:
  - Noise is in (θ, φ) → R always has bonds exactly 3.8Å, even during training.
  - Score target is in (θ, φ) → dE/dθ, dE/dφ via autograd through NeRF.
  - bond_spring is gone from the model entirely — not excluded, not fixed, gone.
  - train/inference are in the same geometric space.

Score matching objective (IC)
------------------------------
For a noisy sample (θ̃, φ̃) = (θ + σ_θ·ε_θ, φ + σ_φ·ε_φ):

  target_θ = (θ̃ - θ) / σ_θ²   ← score of q(θ̃|θ) w.r.t. θ̃
  target_φ = (φ̃ - φ) / σ_φ²

  loss = σ_θ² · ||dE/dθ̃ - target_θ||² + σ_φ² · ||dE/dφ̃ - target_φ||²

When σ_θ = σ_φ (shared sigma mode, the default), this reduces to the original
formulation. When differential sigma is enabled, each coordinate gets its own
noise range calibrated to its natural variance:

  θ (bond angle):  std ≈ 0.295 rad (16.9°)  → σ_max ≈ 0.5 rad
  φ (dihedral):    std ≈ 1.834 rad (105°)   → σ_max ≈ 1.5 rad

A shared noise level t ~ Uniform(0,1) maps to coordinate-specific σ values
via log-uniform interpolation, so both coordinates are always at the same
relative perturbation intensity.

The σ² weighting is the standard multi-scale DSM weighting that makes all
noise levels contribute equally (Vincent 2011, Song & Ermon 2019).

φ wrapping
----------
φ is periodic in [-π, π]. The noise target uses the shortest-arc difference:
  Δφ = wrap_to_pi(φ̃ - φ)
so σ-normalised target_φ = Δφ / σ_φ² (not (φ̃ - φ) / σ_φ²).

α-augmented bidirectional DSM (run53+)
--------------------------------------
Standard IC DSM only sees perturbed structures with Rg ≥ Rg_native (IC noise
almost always increases Rg — the coverage gap from Proposition 1 of the DSM
null space paper). The model learns that the gradient always points toward
smaller Rg, causing systematic compaction in long Langevin trajectories.

α-augmentation fixes this by also training on compacted and swollen structures:

  For each native R in the batch:
    1. Sample α ~ U(α_min, α_max), σ ~ LogUniform(σ_min, σ_max)
    2. Non-uniform scale: α_i = 1 + (α - 1) × (d_i / d_mean)
       where d_i = distance of residue i from center of mass.
       Core residues barely move, surface residues move most.
       This changes bond angles and dihedrals (unlike uniform scaling,
       which preserves all ICs exactly).
    3. Extract ICs from R_α → x_α, add IC noise → x̃_α
    4. Three DSM samples, ALL pointing to x_native:

    | Sample | Effective σ                 | Target                              |
    |--------|-----------------------------|-------------------------------------|
    | x̃     | σ                           | (x̃ - x_native) / σ²                |
    | x_α    | σ_α = ||x_α - x_native||/√d| (x_α - x_native) / σ_α²            |
    | x̃_α   | √(σ_α² + σ²)               | (x̃_α - x_native) / (σ_α² + σ²)    |

  where d = 2L-5 (L-2 bond angles + L-3 dihedrals).

  Native is ALWAYS the attractor. α-scaling is another perturbation channel,
  not a new equilibrium. 3× forward passes per batch element.
"""

from __future__ import annotations

import math

import torch

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.utils.math import wrap_to_pi

# ---------------------------------------------------------------------------
# Cartesian DSM (kept for ablations and legacy comparison)
# ---------------------------------------------------------------------------


def dsm_cartesian_loss(
    energy_model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    sigma: float = 0.25,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    min_dist_cutoff: float = 1.8,
    score_abs_max: float = 5e3,
    resid_abs_max: float = 5e3,
    eps: float = 1e-12,
) -> torch.Tensor:
    """DSM loss in Cartesian space. Kept for ablations — NOT used in run19+.

    When sigma_min and sigma_max are both provided, draws one sigma per sample
    log-uniformly from [sigma_min, sigma_max] and weights by sigma^2 per sample.
    Otherwise falls back to a single fixed sigma for the whole batch.

    Returns a scalar tensor. If unsafe, returns a detached zero (skip sentinel).
    """
    device = R.device
    dtype = R.dtype
    B = R.shape[0]

    multi_scale = (sigma_min is not None) and (sigma_max is not None) and (sigma_min < sigma_max)
    if multi_scale:
        log_sigmas = torch.empty(B, device=device, dtype=dtype).uniform_(
            math.log(float(sigma_min)), math.log(float(sigma_max))
        )
        sigmas = log_sigmas.exp()
        sigma_bcast = sigmas[:, None, None]
    else:
        sigmas = torch.full((B,), float(sigma), device=device, dtype=dtype)
        sigma_bcast = sigmas[:, None, None]

    with torch.no_grad():
        diff = R[:, :, None, :] - R[:, None, :, :]
        D = torch.sqrt((diff * diff).sum(dim=-1) + eps)
        L = D.shape[1]
        eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
        D_clean = D.masked_fill(eye, float("inf"))
        min_dist_clean = D_clean.amin(dim=(1, 2))
        if torch.any(min_dist_clean < float(min_dist_cutoff)):
            return torch.zeros((), device=device, dtype=dtype)

    noise = torch.randn_like(R)
    R_tilde = (R + sigma_bcast * noise).detach().requires_grad_(True)

    if hasattr(energy_model, "forward_dsm"):
        E = energy_model.forward_dsm(R_tilde, seq).sum()
    else:
        E = energy_model(R_tilde, seq).sum()

    grad = torch.autograd.grad(E, R_tilde, create_graph=True, retain_graph=True)[0]

    if (not torch.isfinite(grad).all()) or (grad.abs().max().detach() > float(score_abs_max)):
        return torch.zeros((), device=device, dtype=dtype)

    inv_sigma2 = 1.0 / (sigmas**2 + eps)
    target = (R_tilde - R) * inv_sigma2[:, None, None]

    if not torch.isfinite(target).all():
        return torch.zeros((), device=device, dtype=dtype)

    resid = torch.clamp(grad - target, min=-float(resid_abs_max), max=float(resid_abs_max))
    sigma2 = sigmas**2
    loss = (sigma2[:, None, None] * resid * resid).mean(dim=(1, 2)).mean()

    return loss if torch.isfinite(loss) else torch.zeros((), device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# IC DSM core — computes DSM loss for one set of perturbed ICs
# ---------------------------------------------------------------------------


def _dsm_ic_core(
    energy_model: torch.nn.Module,
    theta_tilde: torch.Tensor,  # (B, L-2) perturbed bond angles
    phi_tilde: torch.Tensor,  # (B, L-3) perturbed dihedrals
    theta_clean: torch.Tensor,  # (B, L-2) native bond angles
    phi_clean: torch.Tensor,  # (B, L-3) native dihedrals
    anchor: torch.Tensor,  # (B, 3, 3) first three atom positions
    seq: torch.Tensor,  # (B, L) amino acid indices
    sigmas_theta: torch.Tensor,  # (B,) effective sigma for θ
    sigmas_phi: torch.Tensor,  # (B,) effective sigma for φ
    valid_theta: torch.Tensor | None,  # (B, L-2) mask or None
    valid_phi: torch.Tensor | None,  # (B, L-3) mask or None
    lengths: torch.Tensor | None,
    bond: float = 3.8,
    eps: float = 1e-12,
    score_abs_max: float = 5e3,
    resid_abs_max: float = 5e3,
) -> torch.Tensor | None:
    """Compute DSM loss for a single set of perturbed ICs pointing to clean ICs.

    Returns scalar loss, or None if unsafe (caller should skip).

    The target is always (x̃ - x_native) / σ², meaning the score should point
    from the perturbed configuration back toward native.
    """
    device = theta_tilde.device
    dtype = theta_tilde.dtype

    # Clamp θ to (0, π), wrap φ to [-π, π]
    theta_tilde = theta_tilde.clamp(0.01, math.pi - 0.01)
    phi_tilde = wrap_to_pi(phi_tilde)

    # Detach and mark as leaf for autograd
    theta_tilde = theta_tilde.detach().requires_grad_(True)
    phi_tilde = phi_tilde.detach().requires_grad_(True)

    # Reconstruct R with exact bonds
    R_tilde = nerf_reconstruct(theta_tilde, phi_tilde, anchor, bond=bond)

    # Energy
    E = energy_model(R_tilde, seq, lengths=lengths).sum()

    # Score: dE/dθ̃ and dE/dφ̃
    grad_theta, grad_phi = torch.autograd.grad(
        E,
        [theta_tilde, phi_tilde],
        create_graph=True,
        retain_graph=True,
    )

    # Safety check
    if (
        not torch.isfinite(grad_theta).all()
        or not torch.isfinite(grad_phi).all()
        or grad_theta.abs().max().detach() > float(score_abs_max)
        or grad_phi.abs().max().detach() > float(score_abs_max)
    ):
        return None

    # Score targets
    inv_sigma2_theta = 1.0 / (sigmas_theta**2 + eps)  # (B,)
    inv_sigma2_phi = 1.0 / (sigmas_phi**2 + eps)  # (B,)

    # θ: no periodicity
    target_theta = (theta_tilde - theta_clean) * inv_sigma2_theta[:, None]
    # φ: periodic — shortest-arc difference
    delta_phi = wrap_to_pi(phi_tilde - phi_clean)
    target_phi = delta_phi * inv_sigma2_phi[:, None]

    if not torch.isfinite(target_theta).all() or not torch.isfinite(target_phi).all():
        return None

    # DSM residual
    resid_theta = torch.clamp(
        grad_theta - target_theta,
        min=-float(resid_abs_max),
        max=float(resid_abs_max),
    )
    resid_phi = torch.clamp(
        grad_phi - target_phi,
        min=-float(resid_abs_max),
        max=float(resid_abs_max),
    )

    # Zero residuals at padding positions
    if valid_theta is not None:
        resid_theta = resid_theta * valid_theta.float()
        resid_phi = resid_phi * valid_phi.float()

    # Loss: σ² weighted per coordinate
    sigma2_theta = sigmas_theta**2
    sigma2_phi = sigmas_phi**2

    if lengths is not None:
        n_theta = (lengths - 2).clamp(min=1).float()
        n_phi = (lengths - 3).clamp(min=1).float()
        loss_theta = (sigma2_theta[:, None] * resid_theta * resid_theta).sum(dim=1) / n_theta
        loss_phi = (sigma2_phi[:, None] * resid_phi * resid_phi).sum(dim=1) / n_phi
    else:
        loss_theta = (sigma2_theta[:, None] * resid_theta * resid_theta).mean(dim=1)
        loss_phi = (sigma2_phi[:, None] * resid_phi * resid_phi).mean(dim=1)

    loss = (loss_theta + loss_phi).mean()
    return loss if torch.isfinite(loss) else None


# ---------------------------------------------------------------------------
# IC DSM — the correct loss for run19+ training
# ---------------------------------------------------------------------------


def dsm_ic_loss(
    energy_model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    sigma: float = 0.05,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    # --- Differential sigma (optional, overrides sigma_min/max) ---
    sigma_min_theta: float | None = None,
    sigma_max_theta: float | None = None,
    sigma_min_phi: float | None = None,
    sigma_max_phi: float | None = None,
    score_abs_max: float = 5e3,
    resid_abs_max: float = 5e3,
    bond: float = 3.8,
    eps: float = 1e-12,
    lengths: torch.Tensor | None = None,
    # --- Alpha augmentation (run53+) ---
    alpha_min: float = 0.65,
    alpha_max: float = 1.25,
    # --- Diagnostics (optional, filled in-place) ---
    diag: dict | None = None,
) -> torch.Tensor:
    """IC DSM loss. The correct training objective for run19+ models.

    Noises (θ, φ) rather than Cartesian R. Reconstructed R always has bonds
    exactly `bond` Å — no bond_spring, no Cartesian noise artefacts.

    Score target is dE/dθ̃ and dE/dφ̃, not dE/dR.

    Sigma units are radians (not Å). Recommended range: [0.02, 0.3] rad.
    Comparable to Cartesian sigma_min/max=[0.05, 0.5] Å in physical effect.

    Alpha augmentation (run53+)
    ---------------------------
    When alpha_min and alpha_max are both provided, enables bidirectional
    Rg augmentation. For each structure in the batch:
      1. Sample α ~ U(alpha_min, alpha_max)
      2. Non-uniform scale: α_i = 1 + (α-1)×(d_i/d_mean) per residue
         (core barely moves, surface moves most — changes ICs)
      3. Extract ICs from R_α, add IC noise
      4. Compute three DSM losses, all pointing back to x_native:
         - Standard: x̃ → x_native (existing behavior)
         - Scaled:   x_α → x_native (non-uniform Rg perturbation)
         - Mixed:    x̃_α → x_native (scaling + IC noise)
      5. Return average of three losses

    This teaches the gradient field to point toward native from both
    compacted (α < 1) and swollen (α > 1) configurations, filling the
    coverage gap that causes systematic compaction in long dynamics.

    Cost: 3× forward passes when alpha augmentation is enabled.

    Args:
        energy_model:   TotalEnergy model. Takes R and seq as before.
        R:              (B, L, 3) clean PDB Cα coordinates.
        seq:            (B, L) amino acid indices.
        sigma:          Fixed noise std in radians.
        sigma_min:      Min sigma for log-uniform sampling (radians). Shared.
        sigma_max:      Max sigma for log-uniform sampling (radians). Shared.
        sigma_min_theta: Min sigma for θ (radians). Enables differential mode.
        sigma_max_theta: Max sigma for θ (radians).
        sigma_min_phi:   Min sigma for φ (radians). Enables differential mode.
        sigma_max_phi:   Max sigma for φ (radians).
        score_abs_max:  Max gradient magnitude before skipping step.
        resid_abs_max:  Clamp value for DSM residual.
        bond:           Fixed Cα-Cα bond length (Å).
        eps:            Numerical stability epsilon.
        lengths:        (B,) actual chain lengths for padding.
        alpha_min:      Min α for Rg scaling augmentation. Default 0.65.
                        Set both to 1.0 to disable.
        alpha_max:      Max α for Rg scaling augmentation. Default 1.25.
                        Set both to 1.0 to disable.

    Returns:
        Scalar loss. Returns detached zero if unsafe (skip-step sentinel).
    """
    device = R.device
    dtype = R.dtype
    B = R.shape[0]
    zero = torch.zeros((), device=device, dtype=dtype)

    # ---- Extract internal coordinates from clean structure ----
    with torch.no_grad():
        theta_clean, phi_clean = coords_to_internal(R)  # (B, L-2), (B, L-3)
        anchor = extract_anchor(R)  # (B, 3, 3)

    # ---- Padding masks for IC positions ----
    if lengths is not None:
        idx_t = torch.arange(theta_clean.shape[1], device=device)
        idx_p = torch.arange(phi_clean.shape[1], device=device)
        valid_theta = idx_t.unsqueeze(0) < (lengths.unsqueeze(1) - 2)  # (B, L-2)
        valid_phi = idx_p.unsqueeze(0) < (lengths.unsqueeze(1) - 3)  # (B, L-3)
    else:
        valid_theta = None
        valid_phi = None

    # ---- Sigma schedule ----
    diff_sigma = (
        sigma_min_theta is not None
        and sigma_max_theta is not None
        and sigma_min_phi is not None
        and sigma_max_phi is not None
    )

    if diff_sigma:
        t = torch.empty(B, device=device, dtype=dtype).uniform_(0.0, 1.0)
        log_smin_theta = math.log(float(sigma_min_theta))
        log_smax_theta = math.log(float(sigma_max_theta))
        log_smin_phi = math.log(float(sigma_min_phi))
        log_smax_phi = math.log(float(sigma_max_phi))
        sigmas_theta = (t * (log_smax_theta - log_smin_theta) + log_smin_theta).exp()
        sigmas_phi = (t * (log_smax_phi - log_smin_phi) + log_smin_phi).exp()
    else:
        multi_scale = (sigma_min is not None) and (sigma_max is not None) and (sigma_min < sigma_max)
        if multi_scale:
            log_sigmas = torch.empty(B, device=device, dtype=dtype).uniform_(
                math.log(float(sigma_min)), math.log(float(sigma_max))
            )
            sigmas = log_sigmas.exp()
        else:
            sigmas = torch.full((B,), float(sigma), device=device, dtype=dtype)
        sigmas_theta = sigmas
        sigmas_phi = sigmas

    # ---- Sample 1: Standard DSM — noise native ICs ----
    noise_theta = torch.randn_like(theta_clean)
    noise_phi = torch.randn_like(phi_clean)
    if valid_theta is not None:
        noise_theta = noise_theta * valid_theta.float()
        noise_phi = noise_phi * valid_phi.float()

    theta_tilde_std = theta_clean + sigmas_theta[:, None] * noise_theta
    phi_tilde_std = phi_clean + sigmas_phi[:, None] * noise_phi

    loss_std = _dsm_ic_core(
        energy_model,
        theta_tilde_std,
        phi_tilde_std,
        theta_clean,
        phi_clean,
        anchor,
        seq,
        sigmas_theta,
        sigmas_phi,
        valid_theta,
        valid_phi,
        lengths,
        bond=bond,
        eps=eps,
        score_abs_max=score_abs_max,
        resid_abs_max=resid_abs_max,
    )

    if loss_std is None:
        return zero

    # ---- Check for alpha augmentation ----
    # Disabled when α range is degenerate (min==max) or trivial ([1,1])
    use_alpha = (
        alpha_min is not None
        and alpha_max is not None
        and alpha_min < alpha_max
        and not (alpha_min == 1.0 and alpha_max == 1.0)
    )

    if not use_alpha:
        if diag is not None:
            diag["dsm_std"] = float(loss_std.detach().item())
            diag["dsm_alpha"] = None
            diag["dsm_mixed"] = None
            diag["dsm_total"] = float(loss_std.detach().item())
            diag["n_samples"] = 1
        return loss_std

    # ==================================================================
    # Alpha-augmented bidirectional DSM (run53+)
    # ==================================================================
    with torch.no_grad():
        # Sample α ~ U(alpha_min, alpha_max) per batch element
        alphas = torch.empty(B, device=device, dtype=dtype).uniform_(float(alpha_min), float(alpha_max))  # (B,)

        # Non-uniform scaling: α_i = 1 + (α_global - 1) × (d_i / d_mean)
        #
        # Uniform Cartesian scaling (com + α(R - com)) preserves ALL bond angles
        # and dihedrals — the IC displacement is exactly zero. Useless for DSM.
        #
        # Non-uniform scaling breaks angle preservation: adjacent residues at
        # different distances from COM get different scaling factors, changing
        # the vectors between them and thus changing θ and φ.
        #
        # Physical motivation: real compaction/swelling affects surface residues
        # more than core residues. This scaling mimics that:
        #   - Core (d_i << d_mean): α_i ≈ 1 (barely moves)
        #   - Surface (d_i >> d_mean): α_i ≈ α_global (full scaling)
        #
        # For α_global=0.8 and typical protein (d range 2-15 Å, d_mean ≈ 8 Å):
        #   - Core (d/d_mean=0.3): α_i = 0.94 (mild)
        #   - Surface (d/d_mean=1.8): α_i = 0.64 (strong)
        # Adjacent residues alternating core↔surface get Δα ≈ 0.3 → angles
        # change by several degrees → genuine IC displacement.

        # Compute per-structure COM respecting variable lengths
        if lengths is not None:
            mask = torch.arange(R.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, L)
            mask_f = mask.float()
            mask_3d = mask_f.unsqueeze(-1)  # (B, L, 1)
            com = (R * mask_3d).sum(dim=1, keepdim=True) / mask_3d.sum(dim=1, keepdim=True)  # (B, 1, 3)
        else:
            mask = None
            mask_f = None
            com = R.mean(dim=1, keepdim=True)  # (B, 1, 3)

        disp = R - com  # (B, L, 3) displacement from COM
        d_i = torch.sqrt((disp * disp).sum(dim=-1) + eps)  # (B, L) distance from COM

        # Mean distance (over valid residues only)
        if mask_f is not None:
            d_mean = (d_i * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True)  # (B, 1)
        else:
            d_mean = d_i.mean(dim=1, keepdim=True)  # (B, 1)

        # Per-residue weight: w_i = d_i / d_mean (mean ≈ 1)
        w_i = d_i / (d_mean + eps)  # (B, L)

        # Per-residue scaling factor
        alpha_i = 1.0 + (alphas[:, None] - 1.0) * w_i  # (B, L)

        # Non-uniform scaling
        R_alpha = com + alpha_i.unsqueeze(-1) * disp  # (B, L, 3)

        # Extract ICs from scaled structure
        theta_alpha, phi_alpha = coords_to_internal(R_alpha)  # (B, L-2), (B, L-3)

        # Compute per-element IC displacement for σ_α
        delta_theta_alpha = theta_alpha - theta_clean  # (B, L-2)
        delta_phi_alpha = wrap_to_pi(phi_alpha - phi_clean)  # (B, L-3)

        # Zero padding positions
        if valid_theta is not None:
            delta_theta_alpha = delta_theta_alpha * valid_theta.float()
            delta_phi_alpha = delta_phi_alpha * valid_phi.float()

        # d = 2L - 5 per element (L-2 bond angles + L-3 dihedrals)
        if lengths is not None:
            d = (2 * lengths - 5).float().clamp(min=1.0)  # (B,)
        else:
            L = R.shape[1]
            d = torch.full((B,), float(2 * L - 5), device=device, dtype=dtype)

        # σ_α = ||x_α - x_native|| / √d  per batch element
        disp_sq = (delta_theta_alpha**2).sum(dim=1) + (delta_phi_alpha**2).sum(dim=1)  # (B,)
        sigma_alpha = torch.sqrt(disp_sq / d + eps)  # (B,)

    # ---- Sample 2: x_α → x_native (scaled, no IC noise) ----
    loss_alpha = _dsm_ic_core(
        energy_model,
        theta_alpha,
        phi_alpha,
        theta_clean,
        phi_clean,
        anchor,
        seq,
        sigma_alpha,
        sigma_alpha,  # shared σ_α for both coordinates
        valid_theta,
        valid_phi,
        lengths,
        bond=bond,
        eps=eps,
        score_abs_max=score_abs_max,
        resid_abs_max=resid_abs_max,
    )

    # ---- Sample 3: x̃_α → x_native (scaled + IC noise) ----
    sigma_eff_theta = torch.sqrt(sigma_alpha**2 + sigmas_theta**2)  # (B,)
    sigma_eff_phi = torch.sqrt(sigma_alpha**2 + sigmas_phi**2)  # (B,)

    noise_theta2 = torch.randn_like(theta_clean)
    noise_phi2 = torch.randn_like(phi_clean)
    if valid_theta is not None:
        noise_theta2 = noise_theta2 * valid_theta.float()
        noise_phi2 = noise_phi2 * valid_phi.float()

    theta_tilde_mix = theta_alpha + sigmas_theta[:, None] * noise_theta2
    phi_tilde_mix = phi_alpha + sigmas_phi[:, None] * noise_phi2

    loss_mixed = _dsm_ic_core(
        energy_model,
        theta_tilde_mix,
        phi_tilde_mix,
        theta_clean,
        phi_clean,
        anchor,
        seq,
        sigma_eff_theta,
        sigma_eff_phi,
        valid_theta,
        valid_phi,
        lengths,
        bond=bond,
        eps=eps,
        score_abs_max=score_abs_max,
        resid_abs_max=resid_abs_max,
    )

    # ---- Combine: average of available losses ----
    losses = [loss_std]
    if loss_alpha is not None:
        losses.append(loss_alpha)
    if loss_mixed is not None:
        losses.append(loss_mixed)

    total = sum(losses) / len(losses)

    # Fill diagnostics dict if provided
    if diag is not None:
        diag["dsm_std"] = float(loss_std.detach().item())
        diag["dsm_alpha"] = float(loss_alpha.detach().item()) if loss_alpha is not None else None
        diag["dsm_mixed"] = float(loss_mixed.detach().item()) if loss_mixed is not None else None
        diag["dsm_total"] = float(total.detach().item()) if torch.isfinite(total) else None
        diag["n_samples"] = len(losses)

    return total if torch.isfinite(total) else zero
