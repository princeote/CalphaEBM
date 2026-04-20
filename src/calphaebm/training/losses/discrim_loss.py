"""Per-subterm discrimination maintenance loss for Phase 5.

Prevents discrimination collapse by requiring each energy subterm to
independently rank native below IC-perturbed structures:

    L_discrim = (1/K) * Σ_k softplus(E_k(R_native) - E_k(R_perturbed))

This is the same contrastive objective used in Phases 3-4 for individual
terms, applied to ALL terms simultaneously during joint training.

The loss is self-annealing: softplus(x) → 0 for x << 0 (terms already
discriminating), so only collapsing terms receive corrective gradients.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.utils.logging import get_logger

logger = get_logger()

# θ is perturbed at a fraction of φ's noise (bond angles are stiffer)
THETA_PHI_RATIO = 1.0 / (2.0 * math.pi)


def _ic_perturb(
    R: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate IC-perturbed structures. Returns R_pert (B, L, 3).

    Each structure gets its own log-uniform σ. Both θ and φ are perturbed.
    Bonds stay 3.8 Å (NeRF reconstruction).
    """
    with torch.no_grad():
        theta, phi = coords_to_internal(R)
        anchor = extract_anchor(R)
        B = R.shape[0]

        log_min = math.log(sigma_min)
        log_max = math.log(sigma_max)
        sigmas = torch.empty(B, 1, device=R.device).uniform_(log_min, log_max).exp()

        noise_t = THETA_PHI_RATIO * sigmas * torch.randn_like(theta)
        noise_p = sigmas * torch.randn_like(phi)

        # Mask padding
        if lengths is not None:
            Lt = theta.shape[1]
            Lp = phi.shape[1]
            idx_t = torch.arange(Lt, device=R.device).unsqueeze(0)
            idx_p = torch.arange(Lp, device=R.device).unsqueeze(0)
            valid_t = idx_t < (lengths.unsqueeze(1) - 2)
            valid_p = idx_p < (lengths.unsqueeze(1) - 3)
            noise_t = noise_t * valid_t.float()
            noise_p = noise_p * valid_p.float()

        theta_neg = (theta + noise_t).clamp(0.01, math.pi - 0.01)
        phi_neg = phi + noise_p
        phi_neg = (phi_neg + math.pi) % (2 * math.pi) - math.pi

        R_pert = nerf_reconstruct(theta_neg, phi_neg, anchor)

    return R_pert


def _gate(model: torch.nn.Module, term: str) -> float:
    """Read gate value for a term."""
    gate_name = f"gate_{term}"
    if hasattr(model, gate_name):
        return float(getattr(model, gate_name).detach().item())
    return 1.0


def _evaluate_subterms(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    exclude: Optional[Set[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Evaluate all active subterms, returning dict of (B,) tensors with gates applied.

    Mirrors the logic in balance_loss._evaluate_subterm_energies but as a
    standalone function for the discrimination loss.
    """
    out: Dict[str, torch.Tensor] = {}
    _skip = exclude or set()

    # ── local ──────────────────────────────────────────────────────
    local = getattr(model, "local", None)
    if local is not None:
        g = _gate(model, "local")
        # 4-mer architecture
        if hasattr(local, "theta_phi_energy") and "theta_phi" not in _skip:
            out["local_thetaphi"] = g * local.theta_phi_energy(R, seq, lengths=lengths)
        # Old 3-subterm architecture
        if hasattr(local, "theta_theta_energy") and "theta_theta" not in _skip:
            try:
                out["local_thetatheta"] = g * local.theta_theta_energy(R, seq, lengths=lengths)
            except TypeError:
                out["local_thetatheta"] = g * local.theta_theta_energy(R)
        if hasattr(local, "delta_phi_energy") and "delta_phi" not in _skip:
            out["local_deltaphi"] = g * local.delta_phi_energy(R, lengths=lengths)
        if hasattr(local, "phi_phi_energy") and "phi_phi" not in _skip:
            try:
                out["local_phiphi"] = g * local.phi_phi_energy(R, seq, lengths=lengths)
            except TypeError:
                out["local_phiphi"] = g * local.phi_phi_energy(R)

    # ── secondary ──────────────────────────────────────────────────
    sec = getattr(model, "secondary", None)
    if sec is not None:
        g = _gate(model, "secondary")
        try:
            E_ram, E_hb_a, E_hb_b = sec.subterm_energies(R, seq, lengths=lengths)
        except TypeError:
            E_ram, E_hb_a, E_hb_b = sec.subterm_energies(R, seq)
        if "ram" not in _skip:
            out["secondary_ram"] = g * E_ram
        if "hb_alpha" not in _skip:
            out["secondary_hb_alpha"] = g * E_hb_a
        if "hb_beta" not in _skip:
            out["secondary_hb_beta"] = g * E_hb_b

    # ── repulsion ──────────────────────────────────────────────────
    rep = getattr(model, "repulsion", None)
    if rep is not None and "repulsion" not in _skip:
        g = _gate(model, "repulsion")
        out["repulsion"] = g * rep(R, seq, lengths=lengths)

    # ── packing ────────────────────────────────────────────────────
    pack = getattr(model, "packing", None)
    if pack is not None:
        g = _gate(model, "packing")
        try:
            E_geom, E_hp, E_coord, E_rg = pack.subterm_energies(R, seq, lengths=lengths)
        except TypeError:
            E_geom, E_hp, E_coord, E_rg = pack.subterm_energies(R, seq)
        if "geom" not in _skip:
            out["packing_geom"] = g * E_geom
        if "contact" not in _skip:
            out["packing_contact"] = g * E_hp
        if "coord" not in _skip:
            out["packing_coord"] = g * E_coord
        if "rg" not in _skip:
            out["packing_rg"] = g * E_rg

    return out


def subterm_discrimination_loss(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
    exclude_subterms: Optional[Set[str]] = None,
    mode: str = "max",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute per-subterm discrimination maintenance loss.

    L = agg_k softplus(E_k(R_native) - E_k(R_perturbed))

    Args:
        model:     Full energy model
        R:         (B, L, 3) native Cα coordinates
        seq:       (B, L) amino acid indices
        lengths:   (B,) actual chain lengths
        sigma_min: Min IC perturbation noise (rad)
        sigma_max: Max IC perturbation noise (rad)
        exclude_subterms: Set of subterm names to skip
        mode:      Aggregation over subterms: 'max' (default, targets worst)
                   or 'mean' (legacy, dilutes collapsing subterms)

    Returns:
        loss:  Scalar loss (aggregated over subterms, mean over batch)
        diag:  Dict of per-subterm gaps (E_pert - E_native, positive = good)
    """
    # Generate IC-perturbed negatives (no grad through perturbation)
    R_pert = _ic_perturb(R, sigma_min, sigma_max, lengths)

    # Per-subterm energies on native and perturbed
    E_nat = _evaluate_subterms(model, R, seq, lengths, exclude_subterms)
    E_pert = _evaluate_subterms(model, R_pert, seq, lengths, exclude_subterms)

    losses = []
    diag: Dict[str, float] = {}

    for key in E_nat:
        if key not in E_pert:
            continue

        e_nat_k = E_nat[key]  # (B,)
        e_pert_k = E_pert[key]  # (B,)

        # Per-structure gap: positive = correct (native < perturbed)
        gap = (e_pert_k - e_nat_k).mean()

        # Loss: penalise native > perturbed (discrimination failure)
        term_loss = F.softplus(e_nat_k - e_pert_k).mean()
        losses.append(term_loss)

        diag[key] = float(gap.detach().item())

    if not losses:
        return torch.tensor(0.0, device=R.device, requires_grad=True), diag

    # Max over subterms: gradient targets the worst-discriminating subterm.
    # When all subterms are healthy (gap >> 0), softplus ≈ 0.5 for all,
    # and max ≈ mean.  When one collapses (gap < 0), max catches it
    # immediately rather than diluting 7× through the mean.
    stacked = torch.stack(losses)
    if mode == "max":
        loss = stacked.max()
    else:
        loss = stacked.mean()

    return loss, diag
