"""Force balance regularization loss for energy-based model training.

Two variants:

force_balance_loss       — legacy Cartesian perturbation. Kept for ablations.
force_balance_ic_loss    — IC perturbation. Correct for run19+ training.
                           Noises (θ, φ), reconstructs via NeRF → bonds always 3.8Å.

Why the IC variant is needed
-----------------------------
The Cartesian _perturb() adds Gaussian noise to R directly. This can stretch
bonds, producing a perturbed geometry that the IC model will never see during
simulation (IC simulation always has bonds exactly 3.8Å). Measuring force
balance at geometries with stretched bonds gives misleading force ratios for
terms like repulsion, which is sensitive to inter-atomic distances.

The IC variant noises (θ, φ) → reconstructs R via NeRF. The perturbed
geometry always has bonds exactly 3.8Å, matching the simulation distribution.
Clash perturbations are replaced by φ dihedral perturbations, which produce
realistic steric violations without breaking bond geometry.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.utils.math import safe_norm, wrap_to_pi

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _rms_force(F_tensor: torch.Tensor) -> torch.Tensor:
    """RMS force magnitude across all atoms."""
    return safe_norm(F_tensor, dim=-1).mean()


def _term_force_rms(
    gate: torch.Tensor,
    term_fn,
    R: torch.Tensor,
    seq: torch.Tensor,
    create_graph: bool = True,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMS gated force for one energy term, keeping graph for backward."""
    Rg = R.detach().requires_grad_(True)
    E = (gate * term_fn(Rg, seq, lengths=lengths)).sum()
    Fv = -torch.autograd.grad(E, Rg, create_graph=create_graph)[0]
    return _rms_force(Fv)


def _symmetric_hinge(F_a: torch.Tensor, F_b: torch.Tensor, target: float) -> torch.Tensor:
    """Penalize when either term dominates: ratio > target in either direction."""
    return F.relu(F_a / (F_b + 1e-8) - target) + F.relu(F_b / (F_a + 1e-8) - target)


# ---------------------------------------------------------------------------
# Cartesian perturbation (legacy — ablations only)
# ---------------------------------------------------------------------------


def _perturb_cartesian(
    R: torch.Tensor,
    sigma_thermal: float = 0.3,
    clash_frac: float = 0.05,
    clash_sigma: float = 0.5,
    clash_min_seq_sep: int = 4,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
) -> torch.Tensor:
    """Thermal + clash perturbation in Cartesian space. Legacy — ablations only.

    NOTE: Can stretch bonds. Use _perturb_ic() for run19+ training.
    """
    B, L, _ = R.shape
    device = R.device
    dtype = R.dtype

    multi_scale = sigma_min is not None and sigma_max is not None and sigma_min < sigma_max

    if multi_scale:
        log_sigmas = torch.empty(B, device=device, dtype=dtype).uniform_(
            math.log(float(sigma_min)), math.log(float(sigma_max))
        )
        sigma_bcast = log_sigmas.exp()[:, None, None]
    else:
        sigma_bcast = float(sigma_thermal)

    R_out = R.detach() + sigma_bcast * torch.randn_like(R)

    if clash_frac > 0 and L > clash_min_seq_sep + 1:
        R_out = R_out.clone()
        n_pairs = max(1, int(clash_frac * L))
        for b in range(B):
            for _ in range(n_pairs):
                i = int(torch.randint(0, L - clash_min_seq_sep, (1,)).item())
                j = int(torch.randint(i + clash_min_seq_sep, L, (1,)).item())
                direction = R_out[b, i] - R_out[b, j]
                dist = direction.norm().clamp(min=0.1)
                R_out[b, j] = R_out[b, j] + clash_sigma * (direction / dist)

    return R_out


# ---------------------------------------------------------------------------
# IC perturbation — correct for run19+ training
# ---------------------------------------------------------------------------


def _perturb_ic(
    R: torch.Tensor,
    sigma_theta: float = 0.15,
    sigma_phi: float = 0.3,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    clash_phi_frac: float = 0.1,
    clash_phi_sigma: float = 1.0,
    bond: float = 3.8,
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """IC perturbation: noise (θ, φ) → reconstruct via NeRF → bonds always 3.8Å.

    Replaces Cartesian thermal+clash perturbation for run19+ training.

    Two perturbation types, both in angle space:
      1. Gaussian noise on all (θ, φ) — simulates thermal fluctuations in
         angle space, matching the DSM training distribution exactly.
      2. Large φ perturbations on a random subset of residues — simulates
         dihedral violations / steric clashes without breaking bond geometry.

    When sigma_min and sigma_max are provided, draws one σ per sample
    log-uniformly from [sigma_min, sigma_max], matching the DSM schedule.
    sigma_min/max are applied to φ noise; θ noise is always sigma_theta.

    Args:
        R:               (B, L, 3) clean Cα coordinates.
        sigma_theta:     Std for θ Gaussian noise (radians). Default 0.15.
        sigma_phi:       Std for φ Gaussian noise (radians). Default 0.3.
        sigma_min:       Min σ for log-uniform schedule (radians, φ only).
        sigma_max:       Max σ for log-uniform schedule (radians, φ only).
        clash_phi_frac:  Fraction of residues receiving large φ perturbations.
        clash_phi_sigma: Std of large φ perturbations (radians). Default 1.0.
        bond:            Fixed Cα-Cα bond length (Å). Default 3.8.
        lengths:         (B,) actual chain lengths. If provided, noise is masked
                         at padding IC positions.

    Returns:
        R_perturbed: (B, L, 3) with bonds exactly 3.8Å.
    """
    B, L, _ = R.shape
    device = R.device
    dtype = R.dtype

    with torch.no_grad():
        theta, phi = coords_to_internal(R)  # (B, L-2), (B, L-3)
        anchor = extract_anchor(R)  # (B, 3, 3)

    # Valid IC masks for padding
    if lengths is not None:
        idx_t = torch.arange(theta.shape[1], device=device)
        idx_p = torch.arange(phi.shape[1], device=device)
        valid_theta = (idx_t.unsqueeze(0) < (lengths.unsqueeze(1) - 2)).float()
        valid_phi = (idx_p.unsqueeze(0) < (lengths.unsqueeze(1) - 3)).float()
    else:
        valid_theta = None
        valid_phi = None

    multi_scale = sigma_min is not None and sigma_max is not None and sigma_min < sigma_max

    # θ noise — fixed sigma_theta
    noise_t = torch.randn_like(theta)
    if valid_theta is not None:
        noise_t = noise_t * valid_theta
    theta_noisy = theta + sigma_theta * noise_t
    theta_noisy = theta_noisy.clamp(0.01, math.pi - 0.01)

    # φ noise — either fixed sigma_phi or log-uniform schedule
    if multi_scale:
        log_sigmas = torch.empty(B, device=device, dtype=dtype).uniform_(
            math.log(float(sigma_min)), math.log(float(sigma_max))
        )
        phi_sigmas = log_sigmas.exp()[:, None]  # (B, 1) → (B, L-3)
    else:
        phi_sigmas = float(sigma_phi)

    noise_p = torch.randn_like(phi)
    if valid_phi is not None:
        noise_p = noise_p * valid_phi
    phi_noisy = phi + phi_sigmas * noise_p

    # Large φ perturbations on random residues — dihedral clashes
    if clash_phi_frac > 0 and phi.shape[1] > 1:
        n_clash = max(1, int(clash_phi_frac * phi.shape[1]))
        clash_mask = torch.zeros_like(phi)
        for b in range(B):
            n_valid_p = int(lengths[b].item()) - 3 if lengths is not None else phi.shape[1]
            n_valid_p = max(1, n_valid_p)
            nc = min(n_clash, n_valid_p)
            idxs = torch.randperm(n_valid_p, device=device)[:nc]
            clash_mask[b, idxs] = 1.0
        clash_noise = torch.randn_like(phi)
        if valid_phi is not None:
            clash_noise = clash_noise * valid_phi
        phi_noisy = phi_noisy + clash_mask * clash_phi_sigma * clash_noise

    # φ ∈ [-π, π]: wrap periodically
    phi_noisy = wrap_to_pi(phi_noisy)

    # Reconstruct — bonds always exactly 3.8Å
    with torch.no_grad():
        R_perturbed = nerf_reconstruct(theta_noisy, phi_noisy, anchor, bond=bond)

    return R_perturbed


# ---------------------------------------------------------------------------
# Cartesian force_balance_loss (kept for ablations)
# ---------------------------------------------------------------------------


def force_balance_loss(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    sigma_thermal: float = 0.3,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    clash_frac: float = 0.05,
    clash_sigma: float = 0.5,
    target_ss_ratio: float = 2.0,
    target_pack_ratio: float = 2.0,
    target_rep_ratio: float = 2.0,
) -> torch.Tensor:
    """Cartesian force balance loss. Kept for ablations — use force_balance_ic_loss for run19+."""
    if not hasattr(model, "local"):
        return torch.zeros((), device=R.device, dtype=R.dtype)

    R_p = _perturb_cartesian(
        R,
        sigma_thermal=sigma_thermal,
        clash_frac=clash_frac,
        clash_sigma=clash_sigma,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    F_local = _term_force_rms(model.gate_local, model.local, R_p, seq)
    loss = torch.zeros((), device=R.device, dtype=R.dtype)

    if model.secondary is not None:
        F_ss = _term_force_rms(model.gate_secondary, model.secondary, R_p, seq)
        loss = loss + _symmetric_hinge(F_local, F_ss, target_ss_ratio)
    if model.packing is not None:
        F_pk = _term_force_rms(model.gate_packing, model.packing, R_p, seq)
        loss = loss + _symmetric_hinge(F_local, F_pk, target_pack_ratio)
    if model.repulsion is not None:
        F_rep = _term_force_rms(model.gate_repulsion, model.repulsion, R_p, seq)
        loss = loss + _symmetric_hinge(F_local, F_rep, target_rep_ratio)

    return loss


def force_balance_diagnostics(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    sigma_thermal: float = 0.3,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    clash_frac: float = 0.05,
    clash_sigma: float = 0.5,
    target_ss_ratio: float = 2.0,
    target_pack_ratio: float = 2.0,
    target_rep_ratio: float = 2.0,
) -> Dict[str, object]:
    """Cartesian force balance diagnostics. Kept for ablations."""
    with torch.no_grad():
        R_p = _perturb_cartesian(
            R,
            sigma_thermal=sigma_thermal,
            clash_frac=clash_frac,
            clash_sigma=clash_sigma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    scales: Dict[str, float] = {}

    def _compute(gate, fn):
        with torch.enable_grad():
            Rg = R_p.detach().requires_grad_(True)
            E = (gate * fn(Rg, seq)).sum()
            Fv = -torch.autograd.grad(E, Rg, create_graph=False)[0]
        return float(_rms_force(Fv).detach().item())

    scales["local"] = _compute(model.gate_local, model.local)
    if model.repulsion is not None:
        scales["repulsion"] = _compute(model.gate_repulsion, model.repulsion)
    if model.secondary is not None:
        scales["secondary"] = _compute(model.gate_secondary, model.secondary)
    if model.packing is not None:
        scales["packing"] = _compute(model.gate_packing, model.packing)

    ref = scales.get("local", 1.0)
    ratios = {k: ref / (v + 1e-8) for k, v in scales.items() if k != "local"}
    targets = {"repulsion": target_rep_ratio, "secondary": target_ss_ratio, "packing": target_pack_ratio}
    met = {
        k: (ratios.get(k, 0.0) <= targets.get(k, 2.0) and ratios.get(k, 0.0) >= 1.0 / targets.get(k, 2.0))
        for k in targets
        if k in ratios
    }
    return {"scales": scales, "ratios": ratios, "targets": targets, "met": met}


# ---------------------------------------------------------------------------
# IC force_balance_loss — correct for run19+ training
# ---------------------------------------------------------------------------


def force_balance_ic_loss(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    sigma_theta: float = 0.15,
    sigma_phi: float = 0.3,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    clash_phi_frac: float = 0.1,
    clash_phi_sigma: float = 1.0,
    bond: float = 3.8,
    target_ss_ratio: float = 2.0,
    target_pack_ratio: float = 2.0,
    target_rep_ratio: float = 2.0,
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """IC force balance loss. The correct variant for run19+ training.

    Perturbs (θ, φ) and reconstructs via NeRF — perturbed geometry always
    has bonds exactly 3.8Å, matching the IC simulation distribution.

    Force ratios are measured in Cartesian space (dE/dR evaluated at R_perturbed)
    — the energy terms still take R as input, so Cartesian forces are the
    natural quantity. The key difference from the Cartesian variant is that
    R_perturbed has no bond stretching artefacts.

    Args: see _perturb_ic() and force_balance_loss() for parameter details.
    """
    if not hasattr(model, "local"):
        return torch.zeros((), device=R.device, dtype=R.dtype)

    R_p = _perturb_ic(
        R,
        sigma_theta=sigma_theta,
        sigma_phi=sigma_phi,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        clash_phi_frac=clash_phi_frac,
        clash_phi_sigma=clash_phi_sigma,
        bond=bond,
        lengths=lengths,
    )
    F_local = _term_force_rms(model.gate_local, model.local, R_p, seq, lengths=lengths)
    loss = torch.zeros((), device=R.device, dtype=R.dtype)

    if model.secondary is not None:
        F_ss = _term_force_rms(model.gate_secondary, model.secondary, R_p, seq, lengths=lengths)
        loss = loss + _symmetric_hinge(F_local, F_ss, target_ss_ratio)
    if model.packing is not None:
        F_pk = _term_force_rms(model.gate_packing, model.packing, R_p, seq, lengths=lengths)
        loss = loss + _symmetric_hinge(F_local, F_pk, target_pack_ratio)
    if model.repulsion is not None:
        F_rep = _term_force_rms(model.gate_repulsion, model.repulsion, R_p, seq, lengths=lengths)
        loss = loss + _symmetric_hinge(F_local, F_rep, target_rep_ratio)

    return loss


def force_balance_ic_diagnostics(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    sigma_theta: float = 0.15,
    sigma_phi: float = 0.3,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    clash_phi_frac: float = 0.1,
    clash_phi_sigma: float = 1.0,
    bond: float = 3.8,
    target_ss_ratio: float = 2.0,
    target_pack_ratio: float = 2.0,
    target_rep_ratio: float = 2.0,
    lengths: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """IC force balance diagnostics for run19+ training."""
    with torch.no_grad():
        R_p = _perturb_ic(
            R,
            sigma_theta=sigma_theta,
            sigma_phi=sigma_phi,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            clash_phi_frac=clash_phi_frac,
            clash_phi_sigma=clash_phi_sigma,
            bond=bond,
            lengths=lengths,
        )

    scales: Dict[str, float] = {}

    def _compute(gate, fn):
        with torch.enable_grad():
            Rg = R_p.detach().requires_grad_(True)
            E = (gate * fn(Rg, seq, lengths=lengths)).sum()
            Fv = -torch.autograd.grad(E, Rg, create_graph=False)[0]
        return float(_rms_force(Fv).detach().item())

    scales["local"] = _compute(model.gate_local, model.local)
    if model.repulsion is not None:
        scales["repulsion"] = _compute(model.gate_repulsion, model.repulsion)
    if model.secondary is not None:
        scales["secondary"] = _compute(model.gate_secondary, model.secondary)
    if model.packing is not None:
        scales["packing"] = _compute(model.gate_packing, model.packing)

    ref = scales.get("local", 1.0)
    ratios = {k: ref / (v + 1e-8) for k, v in scales.items() if k != "local"}
    targets = {"repulsion": target_rep_ratio, "secondary": target_ss_ratio, "packing": target_pack_ratio}
    met = {
        k: (ratios.get(k, 0.0) <= targets.get(k, 2.0) and ratios.get(k, 0.0) >= 1.0 / targets.get(k, 2.0))
        for k in targets
        if k in ratios
    }
    return {"scales": scales, "ratios": ratios, "targets": targets, "met": met}
