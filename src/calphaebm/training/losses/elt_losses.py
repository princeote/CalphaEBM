"""Energy Landscape Theory (ELT) losses for CalphaEBM.

Implements complementary losses derived from protein folding
energy landscape theory (Bryngelson, Wolynes, Onuchic):

1. Q-funnel loss: enforces monotonic energy decrease with increasing
   native similarity Q. Engineers the folding funnel shape.
   Margin scales as m * (1 - exp(-α * ΔQ)) — saturating exponential.

2. Z-score loss: ensures the native state energy is statistically
   significantly lower than decoy energies -- the ELT foldability
   criterion dE/DE >> 1.  WARNING: this loss incentivises shrinking
   all energy scales to make decoy variance small, rather than
   making the absolute gap large.  Consider using gap loss instead.

3. Gap loss: pushes the absolute per-residue energy difference
   E_native < E_decoy by a Q-dependent margin.
   Margin scales as m * (1 - exp(-α * (1-Q_decoy))) — saturating.
   Near-native decoys (Q≈1) get zero margin; far decoys get full m.

4. Frustration loss: penalizes high frustration (native contacts that
   are energetically unfavorable compared to random contacts).
   Encourages a smooth, minimally frustrated landscape with high
   T_f/T_g ratio.

Q-funnel, Z-score, and gap share the same decoys (structural perturbations).
Frustration uses sequence permutations (independent).

Q definition: Best, Hummer & Eaton (2013) PNAS 110(44):17874-17879
  Q = (1/N) sum_{|i-j|>=4, r0_ij<9.5A}  1/(1 + exp(beta(r_ij - lambda*r0_ij)))
  beta = 5.0 A^-1 (switching steepness)
  lambda = 1.2    (20% expansion tolerance)

Margin form: m * (1 - exp(-α * Δ))   [saturating exponential]
  - Δ small: margin ≈ m*α*Δ (linear, gentle near native)
  - Δ large: margin → m (saturates, bounded)
  - 1/α = characteristic Δ for 63% of max margin
  - Encodes cooperative two-state folding via Boltzmann distribution

IMPORTANT: these losses require full protein chains, not segmented
fragments. The Q computation depends on having complete native contact
maps.

CRITICAL: All functions accept a `lengths` tensor (B,) giving the
actual chain length for each protein in the batch.  Padded positions
(indices >= lengths[b]) are masked out of ALL computations: contact
maps, Q values, decoy generation, and energy evaluation.  Without
this mask, padded (0,0,0) coordinates create spurious contacts at
0 A distance that corrupt Q_native to ~0.65 instead of ~1.0.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.utils.logging import get_logger

logger = get_logger()

THETA_PHI_RATIO = 0.161

# ---------------------------------------------------------------------------
# Default margin hyperparameters (Run5)
# ---------------------------------------------------------------------------
DEFAULT_MARGIN_M = 5.0  # maximum margin (saturates here)
DEFAULT_MARGIN_ALPHA = 5.0  # steepness (1/α = 0.2 ΔQ for 63% of max)


# ---------------------------------------------------------------------------
# Saturating exponential margin helper
# ---------------------------------------------------------------------------


def _saturating_margin(delta: torch.Tensor, m: float, alpha: float) -> torch.Tensor:
    """Compute saturating exponential margin: m * (1 - exp(-α * delta)).

    Args:
        delta: (any shape) structural difference (ΔQ, ΔRg, 1-Q, etc.)
               Should be non-negative.
        m:     Maximum margin (saturation level).
        alpha: Steepness. 1/α = characteristic delta for 63% of max.

    Returns:
        Margin tensor, same shape as delta. Range [0, m).
    """
    return m * (1.0 - torch.exp(-alpha * delta.clamp(min=0.0)))


# ---------------------------------------------------------------------------
# Padding mask helpers
# ---------------------------------------------------------------------------


def _make_valid_mask(L: int, lengths: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(L, device=lengths.device)
    return idx.unsqueeze(0) < lengths.unsqueeze(1)


def _make_pair_valid_mask(L: int, lengths: torch.Tensor) -> torch.Tensor:
    valid = _make_valid_mask(L, lengths)
    return valid.unsqueeze(2) & valid.unsqueeze(1)


# ---------------------------------------------------------------------------
# Native contact pre-computation
# ---------------------------------------------------------------------------


def _precompute_native_contacts(
    R_native: torch.Tensor,
    lengths: torch.Tensor,
    contact_cutoff: float = 9.5,
    seq_sep: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L, _ = R_native.shape

    diff = R_native.unsqueeze(2) - R_native.unsqueeze(1)
    d_native = torch.sqrt((diff**2).sum(-1) + 1e-8)

    idx = torch.arange(L, device=R_native.device)
    sep_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= seq_sep
    triu = torch.triu(torch.ones(L, L, device=R_native.device, dtype=torch.bool), diagonal=1)
    pair_mask = (sep_mask & triu).unsqueeze(0).expand(B, -1, -1)

    pair_valid = _make_pair_valid_mask(L, lengths)

    contact_mask = (d_native < contact_cutoff) & pair_mask & pair_valid
    N_contacts = contact_mask.float().sum(dim=(1, 2)).clamp(min=1.0)

    return d_native, contact_mask, N_contacts


# ---------------------------------------------------------------------------
# Q computation  (Best, Hummer & Eaton 2013)
# ---------------------------------------------------------------------------


def _compute_Q(
    R_current: torch.Tensor,
    d_native: torch.Tensor,
    contact_mask: torch.Tensor,
    N_contacts: torch.Tensor,
    beta: float = 5.0,
    lambda_factor: float = 1.2,
) -> torch.Tensor:
    diff = R_current.unsqueeze(2) - R_current.unsqueeze(1)
    d_current = torch.sqrt((diff**2).sum(-1) + 1e-8)

    exponent = beta * (d_current - lambda_factor * d_native)
    exponent = exponent.clamp(max=30.0)
    switching = 1.0 / (1.0 + torch.exp(exponent))

    Q = (switching * contact_mask.float()).sum(dim=(1, 2)) / N_contacts
    return Q


# ---------------------------------------------------------------------------
# Min-distance check for steric viability
# ---------------------------------------------------------------------------


def _min_nonbonded_dist(R_single: torch.Tensor, L_real: int, seq_sep: int = 3) -> float:
    coords = R_single[:L_real]
    if L_real < seq_sep + 1:
        return float("inf")
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    d = torch.sqrt((diff**2).sum(-1) + 1e-8)
    idx = torch.arange(L_real, device=R_single.device)
    nonbond = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > seq_sep
    if not nonbond.any():
        return float("inf")
    return float(d[nonbond].min().item())


# ---------------------------------------------------------------------------
# Decoy generation  (IC-space perturbation via NeRF + steric filtering)
# ---------------------------------------------------------------------------


def _generate_decoys(
    R: torch.Tensor,
    lengths: torch.Tensor,
    n_decoys: int = 10,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
    min_dist_threshold: float = 1.0,
    max_retries: int = 3,
    max_sigma_halvings: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate structural decoys at random perturbation levels.

    Perturbs (θ, φ) in IC space per-sample (avoiding padding contamination),
    reconstructs via NeRF, then checks for steric collapse.

    Each decoy gets an independent log-uniform random σ.
    """
    B, L, _ = R.shape

    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    sigmas = torch.empty(n_decoys, device=R.device).uniform_(log_min, log_max).exp()

    n_rejected = 0
    n_sigma_halved = 0
    n_total = 0

    R_decoys_list = []
    for sigma in sigmas.tolist():
        decoy_batch = torch.zeros_like(R)

        for b in range(B):
            Lb = int(lengths[b].item())
            if Lb < 4:
                decoy_batch[b, :Lb] = R[b, :Lb]
                continue

            R_b = R[b, :Lb].unsqueeze(0)
            theta_b, phi_b = coords_to_internal(R_b)
            anchor_b = extract_anchor(R_b)

            accepted = False
            current_sigma = sigma

            for halving in range(max_sigma_halvings + 1):
                for attempt in range(max_retries):
                    n_total += 1

                    theta_pert = (theta_b + THETA_PHI_RATIO * current_sigma * torch.randn_like(theta_b)).clamp(
                        0.3, math.pi - 0.3
                    )

                    phi_pert = phi_b + current_sigma * torch.randn_like(phi_b)
                    phi_pert = (phi_pert + math.pi) % (2 * math.pi) - math.pi

                    R_pert = nerf_reconstruct(theta_pert, phi_pert, anchor_b)

                    d_min = _min_nonbonded_dist(R_pert[0], Lb)
                    if d_min >= min_dist_threshold:
                        decoy_batch[b, :Lb] = R_pert[0]
                        accepted = True
                        break
                    else:
                        n_rejected += 1

                if accepted:
                    break

                current_sigma *= 0.5
                n_sigma_halved += 1

            if not accepted:
                decoy_batch[b, :Lb] = R_pert[0]  # type: ignore[possibly-undefined]

        R_decoys_list.append(decoy_batch)

    R_decoys = torch.cat(R_decoys_list, dim=0)

    if n_rejected > 0:
        logger.debug(
            "Decoy steric filter: %d/%d rejected (min_dist < %.1fÅ), " "%d sigma halvings, %.1f%% rejection rate",
            n_rejected,
            n_total,
            min_dist_threshold,
            n_sigma_halved,
            100.0 * n_rejected / max(n_total, 1),
        )

    return R_decoys, sigmas


# ---------------------------------------------------------------------------
# 1 + 2.  Q-funnel loss  &  Z-score loss  (shared decoys)
# ---------------------------------------------------------------------------


def elt_funnel_loss(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    lengths: torch.Tensor,
    n_decoys: int = 10,
    T_funnel: float = 2.0,
    target_zscore: float = 3.0,
    gap_margin: float = 0.5,
    slope_clamp: float = 10.0,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
    contact_cutoff: float = 9.5,
    min_dQ: float = 0.05,
    return_diag: bool = False,
    # --- Run5: saturating exponential margin ---
    funnel_m: float = DEFAULT_MARGIN_M,
    funnel_alpha: float = DEFAULT_MARGIN_ALPHA,
    gap_m: float = DEFAULT_MARGIN_M,
    gap_alpha: float = DEFAULT_MARGIN_ALPHA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Combined Q-funnel, Z-score, and gap ELT losses (shared decoys).

    Q-funnel: for pairs where Q_j > Q_i + min_dQ, want E_j < E_i.
    Margin = m * (1 - exp(-α * ΔQ)).  Saturating exponential.

    Gap: E(native) < E(decoy) by Q-dependent margin.
    Margin = m * (1 - exp(-α * (1 - Q_decoy))).  Saturating exponential.

    Z-score: Z = (<E_decoys> - E_native) / std(E_decoys).
    Loss = mean exp(clamp(target - Z, max=5)).
    WARNING: incentivises shrinking energy scales.

    Args:
        model:          TotalEnergy model.
        R:              (B, L, 3) native coordinates (padded).
        seq:            (B, L) amino acid indices (padded).
        lengths:        (B,) actual chain lengths per protein.
        n_decoys:       Decoys per structure. Default 10.
        T_funnel:       Funnel loss temperature. Default 2.0.
        target_zscore:  Target Z-score value. Default 3.0.
        gap_margin:     DEPRECATED constant gap margin (ignored when gap_m>0).
        slope_clamp:    DEPRECATED (unused).
        sigma_min:      Smallest decoy perturbation (rad). Default 0.05.
        sigma_max:      Largest decoy perturbation (rad). Default 2.0.
        contact_cutoff: Q contact cutoff (Å). Default 9.5.
        min_dQ:         Min Q difference to form a pair. Default 0.05.
        return_diag:    If True, fourth return is populated diagnostics.
        funnel_m:       Q-funnel margin maximum. Default 5.0.
        funnel_alpha:   Q-funnel margin steepness. Default 5.0.
        gap_m:          Gap margin maximum. Default 5.0.
        gap_alpha:      Gap margin steepness. Default 5.0.

    Returns:
        loss_funnel:  Scalar Q-funnel loss.
        loss_zscore:  Scalar Z-score loss.
        loss_gap:     Scalar gap loss.
        diag:         Dict of diagnostics (empty if return_diag=False).
    """
    z = torch.zeros((), device=R.device, dtype=R.dtype)
    B, L, _ = R.shape

    if B == 0 or L < 10:
        return z, z, z, {}

    try:
        # -- 1. Pre-compute native contacts (padding-safe) ------
        with torch.no_grad():
            d_native, contact_mask, N_contacts = _precompute_native_contacts(
                R,
                lengths,
                contact_cutoff=contact_cutoff,
            )

        # -- 2. Generate decoys (no grad for coordinates) --------
        with torch.no_grad():
            R_decoys, sigmas = _generate_decoys(
                R,
                lengths,
                n_decoys,
                sigma_min,
                sigma_max,
            )

        # -- 3. Compute Q for decoys and native (no grad) -------
        K = n_decoys
        with torch.no_grad():
            d_native_exp = d_native.repeat(K, 1, 1)
            contact_mask_exp = contact_mask.repeat(K, 1, 1)
            N_contacts_exp = N_contacts.repeat(K)

            Q_decoys = _compute_Q(
                R_decoys,
                d_native_exp,
                contact_mask_exp,
                N_contacts_exp,
            )

            Q_native = _compute_Q(
                R,
                d_native,
                contact_mask,
                N_contacts,
            )

        # -- 4. Evaluate energies (WITH grad through model) ------
        seq_exp = seq.repeat(K, 1)
        lengths_exp = lengths.repeat(K)
        E_native = model(R.detach(), seq, lengths=lengths)
        E_decoys = model(R_decoys, seq_exp, lengths=lengths_exp)

        if not torch.isfinite(E_native).all() or not torch.isfinite(E_decoys).all():
            return z, z, z, {}

        E_native_pr = E_native
        E_decoys_pr = E_decoys

        E_dk = E_decoys_pr.view(K, B).T  # (B, K)
        Q_dk = Q_decoys.view(K, B).T  # (B, K)

        Q_all = torch.cat([Q_dk, Q_native.unsqueeze(1)], dim=1)  # (B, K+1)
        E_all = torch.cat([E_dk, E_native_pr.unsqueeze(1)], dim=1)  # (B, K+1)

        K1 = K + 1

        # === Q-FUNNEL LOSS =====================================
        # dQ[i,j] = Q_j - Q_i;  dE[i,j] = E_j - E_i
        dQ = Q_all.unsqueeze(2) - Q_all.unsqueeze(1)  # (B, K1, K1)
        dE = E_all.unsqueeze(2) - E_all.unsqueeze(1)  # (B, K1, K1)

        valid = dQ > min_dQ  # (B, K1, K1)

        # Saturating exponential margin: m * (1 - exp(-α * ΔQ))
        required_gap_funnel = _saturating_margin(dQ, funnel_m, funnel_alpha)
        exponent = (dE + required_gap_funnel).clamp(max=5.0)
        pair_loss = torch.exp(exponent) * valid.float()

        n_valid = valid.float().sum().clamp(min=1.0)
        loss_funnel = pair_loss.sum() / n_valid

        # === Z-SCORE LOSS ======================================
        decoy_mean = E_dk.mean(dim=1)
        decoy_std = E_dk.std(dim=1).clamp(min=1e-4)
        Z = (decoy_mean - E_native_pr) / decoy_std

        loss_zscore = torch.exp((target_zscore - Z).clamp(max=5.0)).mean()

        # === GAP LOSS ==========================================
        # Saturating margin scaled by (1 - Q_decoy):
        # Near-native decoys (Q≈1) → margin≈0
        # Far decoys (Q≈0) → margin≈m
        delta_Q_gap = 1.0 - Q_dk  # (B, K)
        required_gap_gap = _saturating_margin(delta_Q_gap, gap_m, gap_alpha)
        gap_exponent = (E_native_pr.unsqueeze(1) - E_dk + required_gap_gap).clamp(max=5.0)
        loss_gap = torch.exp(gap_exponent).mean()

        gap = E_dk.mean(dim=1) - E_native_pr

        if not torch.isfinite(loss_funnel):
            loss_funnel = z
        if not torch.isfinite(loss_zscore):
            loss_zscore = z
        if not torch.isfinite(loss_gap):
            loss_gap = z

        # -- Diagnostics ----------------------------------------
        diag: dict = {}
        if return_diag:
            with torch.no_grad():
                dE_valid = dE[valid]
                n_anti = int((dE_valid >= 0).sum().item()) if dE_valid.numel() > 0 else 0
                n_corr = int((dE_valid < 0).sum().item()) if dE_valid.numel() > 0 else 0
                n_tot = n_anti + n_corr

                diag = {
                    "n_pairs": n_tot,
                    "n_anti_funnel": n_anti,
                    "frac_anti_funnel": n_anti / max(n_tot, 1),
                    "mean_dE": float(dE_valid.mean().item()) if n_tot > 0 else 0.0,
                    "Q_decoy_mean": float(Q_dk.mean().item()),
                    "Q_decoy_min": float(Q_dk.min().item()),
                    "Q_decoy_max": float(Q_dk.max().item()),
                    "Q_native_mean": float(Q_native.mean().item()),
                    "Z_mean": float(Z.mean().item()),
                    "Z_min": float(Z.min().item()),
                    "Z_max": float(Z.max().item()),
                    "E_native_pr": float(E_native_pr.mean().item()),
                    "E_decoy_pr": float(E_dk.mean().item()),
                    "gap_mean": float(gap.mean().item()),
                    "gap_min": float(gap.min().item()),
                    "gap_max": float(gap.max().item()),
                    "sigmas": [round(s, 3) for s in sigmas.tolist()],
                    "funnel_m": funnel_m,
                    "funnel_alpha": funnel_alpha,
                    "gap_m": gap_m,
                    "gap_alpha": gap_alpha,
                }

        return loss_funnel, loss_zscore, loss_gap, diag

    except Exception as exc:
        logger.warning("elt_funnel_loss failed: %s", exc)
        return z, z, z, {}


# ---------------------------------------------------------------------------
# 3.  Frustration loss  (sequence permutations)
# ---------------------------------------------------------------------------


def elt_frustration_loss(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    lengths: torch.Tensor,
    n_perms: int = 4,
    T_frust: float = 2.0,
    f_clamp: float = 10.0,
    return_diag: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """Frustration regulariser -- penalises high landscape frustration.

    f = (<E(R, seq_perm)> - E(R, seq_native)) / std(E(R, seq_perm))
    Positive f = good (native preferred). Matches Z-score convention.
    Loss = mean exp( -clamp(f, min=-f_clamp) / T_frust )

    Sequence permutations only shuffle valid (non-padded) positions.
    """
    z = torch.zeros((), device=R.device, dtype=R.dtype)
    B, L = seq.shape

    if B == 0 or L < 10:
        return z, {}

    try:
        E_native = model(R.detach(), seq, lengths=lengths)

        if not torch.isfinite(E_native).all():
            return z, {}

        E_perms = []
        for _ in range(n_perms):
            seq_perm = seq.clone()
            for b in range(B):
                n = int(lengths[b].item())
                perm = torch.randperm(n, device=seq.device)
                seq_perm[b, :n] = seq[b, perm]
            E_perm = model(R.detach(), seq_perm, lengths=lengths)
            if torch.isfinite(E_perm).all():
                E_perms.append(E_perm)

        if len(E_perms) < 2:
            return z, {}

        E_perms_t = torch.stack(E_perms, dim=1)

        perm_mean = E_perms_t.mean(dim=1)
        perm_std = E_perms_t.std(dim=1).clamp(min=1e-4)

        f = (perm_mean - E_native) / perm_std

        f_c = f.clamp(min=-min(f_clamp, T_frust * 5.0))
        loss = torch.exp(-f_c / T_frust).mean()

        if not torch.isfinite(loss):
            loss = z

        diag: dict = {}
        if return_diag:
            with torch.no_grad():
                diag = {
                    "f_mean": float(f.mean().item()),
                    "f_min": float(f.min().item()),
                    "f_max": float(f.max().item()),
                    "n_perms_ok": len(E_perms),
                    "E_native_pr": float(E_native.mean().item()),
                    "E_perm_pr": float(perm_mean.mean().item()),
                    "perm_std_pr": float(perm_std.mean().item()),
                    "frac_frustrated": float((f < 0).float().mean().item()),
                }

        return loss, diag

    except Exception as exc:
        logger.warning("elt_frustration_loss failed: %s", exc)
        return z, {}


# ---------------------------------------------------------------------------
# 4.  Pairwise energy ordering funnel losses (Q and Rg)
# ---------------------------------------------------------------------------
# Standalone versions called by self_consistent.py with pre-collected
# negatives that already have Q and Rg computed.


def q_funnel_loss(
    E_all: torch.Tensor,
    Q_all: torch.Tensor,
    m: float = DEFAULT_MARGIN_M,
    alpha: float = DEFAULT_MARGIN_ALPHA,
    threshold: float = 0.05,
    clamp_max: float = 5.0,
    # Backward compat
    margin: Optional[float] = None,
) -> Tuple[torch.Tensor, int, int]:
    """Pairwise Q-funnel: higher Q should have lower energy.

    For pairs (i,j) where Q_j > Q_i + threshold, we want E_j < E_i.
    Required gap = m * (1 - exp(-α * ΔQ)).  Saturating exponential.

    Args:
        E_all:     (N,) per-residue energies for native + negatives.
        Q_all:     (N,) fraction of native contacts for each structure.
        m:         Maximum margin (saturation level). Default 5.0.
        alpha:     Margin steepness. Default 5.0.
        threshold: Minimum delta-Q to form a valid pair.
        clamp_max: Clamp exponent to prevent overflow.
        margin:    DEPRECATED constant margin. Ignored when m > 0.

    Returns:
        loss:    scalar (mean over valid pairs), zero if no valid pairs.
        n_pairs: number of valid pairs.
        n_anti:  number of anti-funnel violations (E_j >= E_i).
    """
    dQ = Q_all.unsqueeze(0) - Q_all.unsqueeze(1)
    dE = E_all.unsqueeze(0) - E_all.unsqueeze(1)
    valid = dQ > threshold

    n_pairs = int(valid.sum().item())
    if n_pairs == 0:
        return torch.tensor(0.0, device=E_all.device, requires_grad=True), 0, 0

    # Saturating exponential margin
    required_gap = _saturating_margin(dQ, m, alpha)
    exponent = (dE + required_gap).clamp(max=clamp_max)
    pair_loss = torch.exp(exponent) * valid.float()
    loss = pair_loss.sum() / n_pairs

    with torch.no_grad():
        n_anti = int(((dE >= 0) & valid).sum().item())

    return loss, n_pairs, n_anti


def rg_funnel_loss(
    E_all: torch.Tensor,
    delta_all: torch.Tensor,
    m: float = DEFAULT_MARGIN_M,
    alpha: float = DEFAULT_MARGIN_ALPHA,
    threshold: float = 0.05,
    clamp_max: float = 5.0,
    # Backward compat
    margin: Optional[float] = None,
) -> Tuple[torch.Tensor, int, int]:
    """DEPRECATED — replaced by drmsd_funnel_loss.

    Pairwise Rg-funnel: higher |Rg/Rg_native - 1| should have higher energy.
    Kept for backward compatibility only. New code should use drmsd_funnel_loss.

    Problem: Rg is topology-blind — a compact misfold satisfies the Rg
    constraint while having completely wrong internal distances.
    dRMSD catches this; Rg does not.
    """
    dd = delta_all.unsqueeze(0) - delta_all.unsqueeze(1)
    dE = E_all.unsqueeze(0) - E_all.unsqueeze(1)
    valid = dd > threshold

    n_pairs = int(valid.sum().item())
    if n_pairs == 0:
        return torch.tensor(0.0, device=E_all.device, requires_grad=True), 0, 0

    required_gap = _saturating_margin(dd, m, alpha)
    exponent = (-dE + required_gap).clamp(max=clamp_max)
    pair_loss = torch.exp(exponent) * valid.float()
    loss = pair_loss.sum() / n_pairs

    with torch.no_grad():
        n_anti = int(((dE <= 0) & valid).sum().item())

    return loss, n_pairs, n_anti


def drmsd_funnel_loss(
    E_all: torch.Tensor,
    drmsd_all: torch.Tensor,
    m: float = DEFAULT_MARGIN_M,
    alpha: float = DEFAULT_MARGIN_ALPHA,
    threshold: float = 0.5,
    clamp_max: float = 5.0,
) -> Tuple[torch.Tensor, int, int]:
    """Pairwise full-dRMSD funnel: lower dRMSD → lower energy.

    For pairs (i,j) where drmsd_i > drmsd_j + threshold (i is less native),
    we want E_i > E_j.  Required gap = m * (1 - exp(-α * Δ_dRMSD)).

    Uses ALL pairwise Cα distances (|i-j| ≥ 4) — topology-sensitive.
    Unlike Rg, dRMSD cannot be satisfied by a compact misfold with wrong
    internal distances: strand-swaps, helix-register shifts, domain
    rotations all produce large dRMSD even at native Rg.

    The full distance matrix is FREE at training time: the Q computation
    in _precompute_native_contacts already materialises the (B,L,L) D
    tensor.  dRMSD adds only a masked mean over that pre-existing tensor.

    Default threshold: 0.5 Å — forms a valid pair when one structure has
    dRMSD at least 0.5 Å worse than the other.  Typical decoy range is
    2–15 Å so this threshold is generous.

    Functional form identical to rg_funnel_loss; the coordinate is
    absolute dRMSD (Å) instead of |Rg/Rg* − 1| (dimensionless).

    Args:
        E_all:     (N,) per-residue energies.  N = 1 native + n_decoys.
        drmsd_all: (N,) full dRMSD to native for each structure (Å).
                   Native has drmsd=0 by definition.
        m:         Maximum margin (E/res).  Default 5.0.
        alpha:     Margin steepness.  Default 5.0.
        threshold: Minimum Δ_dRMSD (Å) to form a valid pair.  Default 0.5.
        clamp_max: Clamp exponent to prevent overflow.  Default 5.0.

    Returns:
        loss:    Scalar mean loss over valid pairs.
        n_pairs: Number of valid pairs.
        n_anti:  Anti-funnel violations (i less native but E_i <= E_j).
    """
    dd = drmsd_all.unsqueeze(0) - drmsd_all.unsqueeze(1)  # dd[i,j] = dRMSD_i - dRMSD_j
    dE = E_all.unsqueeze(0) - E_all.unsqueeze(1)  # dE[i,j] = E_i - E_j
    valid = dd > threshold  # i has higher dRMSD (less native)

    n_pairs = int(valid.sum().item())
    if n_pairs == 0:
        return torch.tensor(0.0, device=E_all.device, requires_grad=True), 0, 0

    # Saturating margin: larger Δ_dRMSD → stronger push
    required_gap = _saturating_margin(dd, m, alpha)
    # Penalise when E_i ≤ E_j but i is less native (i should have higher energy)
    exponent = (-dE + required_gap).clamp(max=clamp_max)
    pair_loss = torch.exp(exponent) * valid.float()
    loss = pair_loss.sum() / n_pairs

    with torch.no_grad():
        n_anti = int(((dE <= 0) & valid).sum().item())

    return loss, n_pairs, n_anti
