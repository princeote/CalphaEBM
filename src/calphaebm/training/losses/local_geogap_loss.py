"""Local geometry gap loss for energy-based model training.

Two variants:

local_geogap_loss       — legacy Cartesian perturbation. Kept for ablations.
local_geogap_ic_loss    — IC perturbation. Correct for run19+ training.

Why the IC variant is needed
-----------------------------
The Cartesian _local_perturb() displaces individual Cα atoms in 3D space to
create bond/angle/dihedral violations. In IC training this is wrong for two
reasons:

  1. Displacing atoms can stretch bonds, producing geometries the IC model
     never sees (IC simulation always has bonds exactly 3.8Å).

  2. The perturbations are in Cartesian space but the model's local term scores
     (θ, φ) geometry. A Cartesian displacement that violates a bond angle also
     perturbs bond length, mixing the signal.

The IC variant perturbs (θ, φ) directly:
  - θ perturbations → bad bond angles (exactly what theta_theta_energy scores)
  - φ perturbations → bad dihedral transitions (exactly what delta_phi_energy scores)
  - Reconstruction via NeRF → bonds always exactly 3.8Å

The local term is thus tested on the exact geometry violations it was designed
to detect, without bond-length cross-contamination.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.utils.math import wrap_to_pi

# ---------------------------------------------------------------------------
# Cartesian local perturbation (kept for ablations)
# ---------------------------------------------------------------------------


def _local_perturb_cartesian(
    R: torch.Tensor,
    bond_stretch_sigma: float = 0.3,
    angle_perturb_sigma: float = 0.2,
    dihedral_perturb_sigma: float = 0.4,
    frac_perturbed: float = 0.3,
) -> torch.Tensor:
    """Cartesian local geometry perturbation. Legacy — ablations only.

    NOTE: Can stretch bonds. Use _local_perturb_ic() for run19+ training.
    """
    B, L, _ = R.shape
    R_out = R.detach().clone()

    if L < 3:
        return R_out

    n_perturb = max(1, int(frac_perturbed * L))

    for b in range(B):
        # Bond stretch
        idxs = torch.randperm(L - 1)[:n_perturb]
        for i in idxs.tolist():
            bond_vec = R_out[b, i] - R_out[b, i - 1] if i > 0 else R_out[b, i + 1] - R_out[b, i]
            unit = bond_vec / bond_vec.norm().clamp(min=0.1)
            R_out[b, i] = R_out[b, i] + bond_stretch_sigma * torch.randn(1).item() * unit

        # Angle perturbation (perpendicular displacement)
        idxs = torch.randperm(L - 2)[:n_perturb]
        for i in (idxs + 1).tolist():
            if i <= 0 or i >= L - 1:
                continue
            axis = R_out[b, i + 1] - R_out[b, i - 1]
            axis_u = axis / axis.norm().clamp(min=0.1)
            rand_vec = torch.randn(3)
            rand_vec = rand_vec - (rand_vec @ axis_u) * axis_u
            perp = rand_vec / rand_vec.norm().clamp(min=1e-6)
            R_out[b, i] = R_out[b, i] + abs(torch.randn(1).item()) * angle_perturb_sigma * perp

        # Dihedral perturbation (along dihedral normal)
        idxs = torch.randperm(L - 3)[:n_perturb]
        for i in (idxs + 1).tolist():
            if i <= 0 or i >= L - 2:
                continue
            b1 = R_out[b, i] - R_out[b, i - 1]
            b2 = R_out[b, i + 1] - R_out[b, i]
            normal = torch.linalg.cross(b1, b2)
            normal = normal / normal.norm().clamp(min=1e-6)
            R_out[b, i] = R_out[b, i] + torch.randn(1).item() * dihedral_perturb_sigma * normal

    return R_out


# ---------------------------------------------------------------------------
# IC local perturbation — correct for run19+ training
# ---------------------------------------------------------------------------


def _local_perturb_ic(
    R: torch.Tensor,
    theta_perturb_sigma: float = 0.25,
    phi_perturb_sigma: float = 0.5,
    frac_perturbed: float = 0.3,
    bond: float = 3.8,
) -> torch.Tensor:
    """IC local geometry perturbation. Correct for run19+ training.

    Perturbs (θ, φ) directly → reconstructs via NeRF → bonds always 3.8Å.

    Two targeted perturbation types:
      1. θ violations: perturb bond angles at a random subset of residues.
         This directly tests theta_theta_energy discrimination.
      2. φ violations: perturb dihedral angles at a random (different) subset.
         This directly tests delta_phi_energy discrimination.

    Unlike Cartesian perturbation, there is zero bond-length cross-contamination:
    bond lengths are exactly 3.8Å by NeRF reconstruction at every perturbed geometry.

    Args:
        R:                   (B, L, 3) clean Cα coordinates.
        theta_perturb_sigma: Std for θ perturbations (radians). Default 0.25.
        phi_perturb_sigma:   Std for φ perturbations (radians). Default 0.5.
        frac_perturbed:      Fraction of residues receiving each perturbation type.
        bond:                Fixed bond length (Å). Default 3.8.

    Returns:
        R_perturbed: (B, L, 3) with bonds exactly 3.8Å and local geometry violations.
    """
    B, L, _ = R.shape
    device = R.device

    with torch.no_grad():
        theta, phi = coords_to_internal(R)  # (B, L-2), (B, L-3)
        anchor = extract_anchor(R)  # (B, 3, 3)

    theta_p = theta.clone()
    phi_p = phi.clone()

    n_theta = max(1, int(frac_perturbed * theta.shape[1]))
    n_phi = max(1, int(frac_perturbed * phi.shape[1]))

    for b in range(B):
        # θ violations — bad bond angles
        t_idxs = torch.randperm(theta.shape[1], device=device)[:n_theta]
        theta_p[b, t_idxs] = theta_p[b, t_idxs] + theta_perturb_sigma * torch.randn(n_theta, device=device)

        # φ violations — bad dihedral transitions
        p_idxs = torch.randperm(phi.shape[1], device=device)[:n_phi]
        phi_p[b, p_idxs] = phi_p[b, p_idxs] + phi_perturb_sigma * torch.randn(n_phi, device=device)

    # θ ∈ (0, π): clamp to physical range
    theta_p = theta_p.clamp(0.01, math.pi - 0.01)
    # φ ∈ [-π, π]: wrap periodically
    phi_p = wrap_to_pi(phi_p)

    with torch.no_grad():
        R_perturbed = nerf_reconstruct(theta_p, phi_p, anchor, bond=bond)

    return R_perturbed


# ---------------------------------------------------------------------------
# Cartesian geogap loss (kept for ablations)
# ---------------------------------------------------------------------------


def local_geogap_loss(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    margin: float = 2.0,
    bond_stretch_sigma: float = 0.3,
    angle_perturb_sigma: float = 0.2,
    dihedral_perturb_sigma: float = 0.4,
    frac_perturbed: float = 0.3,
) -> torch.Tensor:
    """Cartesian local geogap loss. Kept for ablations — use local_geogap_ic_loss for run19+."""
    if not hasattr(model, "local") or model.local is None:
        return torch.zeros((), device=R.device, dtype=R.dtype)

    E_clean = (model.gate_local * model.local.forward_learned(R, seq)).mean()
    R_perturbed = _local_perturb_cartesian(
        R,
        bond_stretch_sigma=bond_stretch_sigma,
        angle_perturb_sigma=angle_perturb_sigma,
        dihedral_perturb_sigma=dihedral_perturb_sigma,
        frac_perturbed=frac_perturbed,
    ).to(R.device)
    E_perturbed = (model.gate_local * model.local.forward_learned(R_perturbed, seq)).mean()

    return F.relu(margin - (E_perturbed - E_clean))


def local_geogap_diagnostics(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    margin: float = 2.0,
    bond_stretch_sigma: float = 0.3,
    angle_perturb_sigma: float = 0.2,
    dihedral_perturb_sigma: float = 0.4,
    frac_perturbed: float = 0.3,
) -> dict:
    """Cartesian geogap diagnostics. Kept for ablations."""
    with torch.enable_grad():
        E_clean = float(
            (model.gate_local * model.local.forward_learned(R.detach().requires_grad_(False), seq))
            .mean()
            .detach()
            .item()
        )
        R_p = _local_perturb_cartesian(
            R,
            bond_stretch_sigma=bond_stretch_sigma,
            angle_perturb_sigma=angle_perturb_sigma,
            dihedral_perturb_sigma=dihedral_perturb_sigma,
            frac_perturbed=frac_perturbed,
        ).to(R.device)
        E_perturbed = float((model.gate_local * model.local.forward_learned(R_p, seq)).mean().detach().item())

    gap = E_perturbed - E_clean
    loss_val = max(0.0, margin - gap)
    return {
        "E_clean": E_clean,
        "E_perturbed": E_perturbed,
        "gap": gap,
        "margin": margin,
        "gap_active": gap < margin,
        "loss_value": loss_val,
    }


# ---------------------------------------------------------------------------
# IC geogap loss — correct for run19+ training
# ---------------------------------------------------------------------------


def local_geogap_ic_loss(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    margin: float = 2.0,
    theta_perturb_sigma: float = 0.25,
    phi_perturb_sigma: float = 0.5,
    frac_perturbed: float = 0.3,
    bond: float = 3.8,
) -> torch.Tensor:
    """IC local geogap loss. The correct variant for run19+ training.

    Ensures the local term (theta_theta_energy + delta_phi_energy) retains
    a positive energy gap between native and locally-perturbed geometry.

    Perturbation is in (θ, φ) space — no bond-length cross-contamination.
    Reconstructed geometry always has bonds exactly 3.8Å.

    The θ-perturbation tests theta_theta_energy directly.
    The φ-perturbation tests delta_phi_energy directly.

    Loss = relu(margin - (E_local(perturbed) - E_local(clean)))
    Zero when gap >= margin. Active when local term loses discrimination.

    Args:
        model:               TotalEnergy model with .local and .gate_local.
        R:                   (B, L, 3) clean Cα coordinates.
        seq:                 (B, L) amino acid indices.
        margin:              Required energy gap. Default 2.0.
        theta_perturb_sigma: Std for θ perturbations (radians). Default 0.25.
        phi_perturb_sigma:   Std for φ perturbations (radians). Default 0.5.
        frac_perturbed:      Fraction of residues perturbed. Default 0.3.
        bond:                Fixed bond length (Å). Default 3.8.

    Returns:
        Scalar loss. Zero when E_local gap >= margin.
    """
    if not hasattr(model, "local") or model.local is None:
        return torch.zeros((), device=R.device, dtype=R.dtype)

    E_clean = (model.gate_local * model.local(R, seq)).mean()

    R_perturbed = _local_perturb_ic(
        R,
        theta_perturb_sigma=theta_perturb_sigma,
        phi_perturb_sigma=phi_perturb_sigma,
        frac_perturbed=frac_perturbed,
        bond=bond,
    ).to(R.device)

    E_perturbed = (model.gate_local * model.local(R_perturbed, seq)).mean()

    return F.relu(margin - (E_perturbed - E_clean))


def local_geogap_ic_diagnostics(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    margin: float = 2.0,
    theta_perturb_sigma: float = 0.25,
    phi_perturb_sigma: float = 0.5,
    frac_perturbed: float = 0.3,
    bond: float = 3.8,
) -> dict:
    """IC geogap diagnostics for run19+ training."""
    with torch.no_grad():
        E_clean = float((model.gate_local * model.local(R, seq)).mean().detach().item())
        R_p = _local_perturb_ic(
            R,
            theta_perturb_sigma=theta_perturb_sigma,
            phi_perturb_sigma=phi_perturb_sigma,
            frac_perturbed=frac_perturbed,
            bond=bond,
        ).to(R.device)
        E_perturbed = float((model.gate_local * model.local(R_p, seq)).mean().detach().item())

    gap = E_perturbed - E_clean
    loss_val = max(0.0, margin - gap)
    return {
        "E_clean": E_clean,
        "E_perturbed": E_perturbed,
        "gap": gap,
        "margin": margin,
        "gap_active": gap < margin,
        "loss_value": loss_val,
    }
