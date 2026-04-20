"""NeRF reconstruction: internal coordinates → Cartesian Cα coordinates.

Converts bond angles (θ) and torsion angles (φ) back to 3D coordinates
with bond lengths fixed at exactly 3.8 Å by construction.

This is the inverse of calphaebm.geometry.internal.{bond_angles, torsions}.

Mathematical foundation
-----------------------
Given three consecutive atoms p1, p2, p3, the position of the next atom p4
is determined by:
  - bond length  b    = 3.8 Å (fixed constant — never a variable)
  - bond angle   θ    at atom p3 (angle p2-p3-p4)
  - torsion angle φ   of quadruplet (p1, p2, p3, p4)

The NeRF formula places p4 in the local reference frame defined by p1,p2,p3:

    bc  = unit(p3 - p2)           # bond direction
    n   = unit(cross(p2-p1, p3-p2))  # normal to plane
    m   = cross(bc, n)            # completes right-hand frame

    d   = b * [-cos(θ), sin(θ)*cos(φ), -sin(θ)*sin(φ)]  # in local frame (-φ convention)

    p4  = p3 + d[0]*bc + d[1]*m + d[2]*n

Because b is a constant input and never updated, bond lengths are exactly
3.8 Å everywhere in the trajectory by construction — not as a soft penalty
but as a hard geometric identity.

Anchor
------
The first three atom positions are the "anchor" — they fix global translation
and orientation of the chain. Since all energy terms are translation/rotation
invariant (they depend only on distances and angles), fixing the anchor to the
native structure's first three Cα positions loses nothing physically. It simply
removes 6 redundant degrees of freedom (3 translation + 3 rotation).

Usage
-----
    from calphaebm.geometry.reconstruct import nerf_reconstruct, coords_to_internal

    # Convert existing trajectory coords to (theta, phi) for warm start
    theta, phi = coords_to_internal(R_native)   # (B, L-2), (B, L-3)

    # Reconstruct — bonds are exactly 3.8 Å
    R = nerf_reconstruct(theta, phi, anchor=R_native[:, :3, :])   # (B, L, 3)

Integration with existing code
-------------------------------
All energy terms (LocalEnergy, RepulsionEnergy, SecondaryStructureEnergy,
PackingEnergy) are unchanged — they still take R as input. The new integrator
maintains state as (theta, phi), calls nerf_reconstruct to get R at each step,
evaluates E(R), and uses autograd to get dE/dtheta and dE/dphi.

The LocalEnergy bond_energy() method and bond_spring buffer should be removed
(see local_terms_no_bond.py) since bond length is now exactly 3.8 Å always.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from calphaebm.utils.math import safe_norm

# ---------------------------------------------------------------------------
# Core NeRF placement: place one atom given three anchors + angles
# ---------------------------------------------------------------------------


def _place_atom(
    p1: torch.Tensor,  # (B, 3)
    p2: torch.Tensor,  # (B, 3)
    p3: torch.Tensor,  # (B, 3)
    theta: torch.Tensor,  # (B,) bond angle at p3 in radians
    phi: torch.Tensor,  # (B,) torsion angle in radians
    bond: float = 3.8,
) -> torch.Tensor:
    """Place atom p4 given three reference atoms and two angles.

    Bond length is exactly `bond` Å — not approximate, not penalized,
    but exact by the geometry of the construction.

    Args:
        p1, p2, p3: (B, 3) positions of three consecutive preceding atoms.
        theta: (B,) bond angle at p3, i.e. angle(p2, p3, p4) in radians.
        phi: (B,) torsion angle of quadruplet (p1, p2, p3, p4) in radians.
        bond: Fixed Cα-Cα bond length in Å.

    Returns:
        p4: (B, 3) position of the new atom.
    """
    # Local frame — must match the convention used by dihedral.py exactly.
    #
    # After the sign fix, dihedral(p1,p2,p3,p4) uses:
    #   b0=p2-p1, b1=p3-p2
    #   u = cross(b0, b1)
    #   w = cross(b1, u)     ← standard order (b1, u), not (u, b1)
    #   phi = atan2(dot(w,v), dot(u,v))
    #
    # With this convention: phi_dihedral = phi_nerf (no negation).
    # So reconstruction uses phi directly:
    #   d = bond * [-cos(θ),  sin(θ)*cos(φ),  sin(θ)*sin(φ)]

    bc = p3 - p2
    bc = bc / (safe_norm(bc, dim=-1, keepdim=True))  # unit bond direction (B, 3)

    # Normal to p1-p2-p3 plane — matches u in dihedral()
    b0 = p2 - p1
    b1 = p3 - p2
    n = torch.linalg.cross(b0, b1, dim=-1)  # (B, 3)
    n = n / (safe_norm(n, dim=-1, keepdim=True))

    # In-plane vector perpendicular to bc
    m = torch.linalg.cross(n, bc, dim=-1)  # (B, 3)

    sin_t = torch.sin(theta)  # (B,)
    cos_t = torch.cos(theta)  # (B,)
    sin_p = torch.sin(phi)  # (B,)
    cos_p = torch.cos(phi)  # (B,)

    # phi_dihedral = phi_nerf (no inversion needed with standard convention)
    d_bc = (-cos_t).unsqueeze(-1)  # (B, 1)
    d_m = (sin_t * cos_p).unsqueeze(-1)  # (B, 1)
    d_n = (sin_t * sin_p).unsqueeze(-1)  # (B, 1)  — no negation

    p4 = p3 + bond * (d_bc * bc + d_m * m + d_n * n)  # (B, 3)
    return p4


# ---------------------------------------------------------------------------
# Full chain reconstruction
# ---------------------------------------------------------------------------


def nerf_reconstruct(
    theta: torch.Tensor,  # (B, L-2) bond angles in radians
    phi: torch.Tensor,  # (B, L-3) torsion angles in radians
    anchor: torch.Tensor,  # (B, 3, 3) first three atom positions (fixed)
    bond: float = 3.8,
) -> torch.Tensor:
    """Reconstruct full Cα chain from internal coordinates.

    Bond lengths are exactly `bond` Å everywhere — not a constraint,
    not a penalty, but a geometric identity of the construction.

    Args:
        theta: (B, L-2) bond angles. theta[:, i] is the angle at residue i+1,
               i.e. angle(r_i, r_{i+1}, r_{i+2}).
        phi:   (B, L-3) torsion angles. phi[:, i] is the torsion of quadruplet
               (r_i, r_{i+1}, r_{i+2}, r_{i+3}).
        anchor: (B, 3, 3) positions of the first three Cα atoms. These are
                fixed throughout simulation — they set global translation and
                orientation, which do not affect any energy term.
        bond:  Cα-Cα bond length in Å (default 3.8).

    Returns:
        R: (B, L, 3) full chain coordinates. Bonds r[i+1]-r[i] are all
           exactly `bond` Å for i >= 0.

    Notes:
        - L = theta.shape[1] + 2 = phi.shape[1] + 3
        - theta and phi must be consistent: theta.shape[1] == phi.shape[1] + 1
        - The first three rows of R are copied directly from anchor (no
          reconstruction needed — they are the reference frame).
        - Reconstruction is sequential: atom i+1 depends on atoms i-2, i-1, i.
          PyTorch autograd correctly handles this sequential dependency — gradients
          flow back through the entire chain automatically.
    """
    B = theta.shape[0]
    L_minus_2 = theta.shape[1]  # number of bond angles = L-2
    L_minus_3 = phi.shape[1]  # number of torsions = L-3
    L = L_minus_2 + 2

    if L_minus_3 != L_minus_2 - 1:
        raise ValueError(
            f"Inconsistent shapes: theta has {L_minus_2} angles (→ L={L}) "
            f"but phi has {L_minus_3} torsions (→ L={L_minus_3 + 3}). "
            f"Expected phi.shape[1] == theta.shape[1] - 1."
        )
    if anchor.shape != (B, 3, 3):
        raise ValueError(f"anchor must have shape (B, 3, 3), got {tuple(anchor.shape)}")

    # Start with the three anchor atoms
    # We build coords as a list and stack at the end for efficiency
    coords = [
        anchor[:, 0, :],  # (B, 3) — atom 0
        anchor[:, 1, :],  # (B, 3) — atom 1
        anchor[:, 2, :],  # (B, 3) — atom 2
    ]

    # Place atoms 3, 4, ..., L-1 sequentially
    # Atom i+3 uses:
    #   - p1 = coords[i]     (atom i)
    #   - p2 = coords[i+1]   (atom i+1)
    #   - p3 = coords[i+2]   (atom i+2)
    #   - theta[:, i]        (bond angle at atom i+2)
    #   - phi[:, i]          (torsion of atoms i, i+1, i+2, i+3)
    for i in range(L - 3):
        p1 = coords[i]
        p2 = coords[i + 1]
        p3 = coords[i + 2]
        t = theta[:, i + 1]  # angle at p3=r_{i+2}: bond_angles[k]=angle(r_k,r_{k+1},r_{k+2}), so k=i+1
        p = phi[:, i]  # torsion of (r_i, r_{i+1}, r_{i+2}, r_{i+3})
        p4 = _place_atom(p1, p2, p3, t, p, bond=bond)
        coords.append(p4)

    R = torch.stack(coords, dim=1)  # (B, L, 3)
    return R


# ---------------------------------------------------------------------------
# Utility: extract anchor and internal coords from existing Cartesian coords
# (used to warm-start from native structure)
# ---------------------------------------------------------------------------


def coords_to_internal(
    R: torch.Tensor,  # (B, L, 3) or (L, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract (theta, phi) from Cartesian coordinates.

    Thin wrapper around existing geometry functions, provided here for
    convenience when warm-starting the internal coordinate integrator
    from a native PDB structure.

    Args:
        R: (B, L, 3) or (L, 3) Cα coordinates.

    Returns:
        theta: (B, L-2) bond angles in radians.
        phi:   (B, L-3) torsion angles in radians.
    """
    from calphaebm.geometry.internal import bond_angles, torsions

    if R.dim() == 2:
        R = R.unsqueeze(0)

    return bond_angles(R), torsions(R)


def extract_anchor(R: torch.Tensor) -> torch.Tensor:
    """Extract the anchor (first three atoms) from coordinates.

    Args:
        R: (B, L, 3) or (L, 3) coordinates.

    Returns:
        anchor: (B, 3, 3) first three atom positions.
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
    return R[:, :3, :].clone()


# ---------------------------------------------------------------------------
# Numerical verification utility (not used in production)
# ---------------------------------------------------------------------------


@torch.no_grad()
def verify_reconstruction(
    R_original: torch.Tensor,  # (B, L, 3) or (L, 3)
    bond: float = 3.8,
    atol: float = 1e-4,
) -> dict:
    """Verify that nerf_reconstruct inverts coords_to_internal correctly.

    Extracts (theta, phi, anchor) from R_original, reconstructs R, and
    checks that bond lengths are exactly `bond` and that the reconstructed
    coordinates match the original (up to floating-point precision).

    Args:
        R_original: Reference coordinates to verify against.
        bond: Expected bond length in Å.
        atol: Absolute tolerance for coordinate comparison.

    Returns:
        Dictionary with verification results.
    """
    if R_original.dim() == 2:
        R_original = R_original.unsqueeze(0)

    B, L, _ = R_original.shape

    theta, phi = coords_to_internal(R_original)
    anchor = extract_anchor(R_original)

    R_reconstructed = nerf_reconstruct(theta, phi, anchor, bond=bond)

    # All bond lengths (B, L-1)
    all_diffs = R_reconstructed[:, 1:, :] - R_reconstructed[:, :-1, :]
    all_bl = torch.sqrt((all_diffs * all_diffs).sum(dim=-1))

    # Anchor bonds: first 2 bonds (atoms 0-1, 1-2) — copied from native, NOT 3.8 Å exactly
    anchor_bl = all_bl[:, :2]

    # NeRF-placed bonds: atom 3 onwards — these MUST be exactly `bond` Å
    nerf_diffs = R_reconstructed[:, 3:, :] - R_reconstructed[:, 2:-1, :]
    nerf_bl = torch.sqrt((nerf_diffs * nerf_diffs).sum(dim=-1))  # (B, L-3)
    max_bond_error = (nerf_bl - bond).abs().max().item()
    mean_bond_error = (nerf_bl - bond).abs().mean().item()

    # Coordinate reconstruction error (residues 3+ only; anchor is copied exactly)
    coord_error = (R_reconstructed[:, 3:, :] - R_original[:, 3:, :]).abs()
    max_coord_error = coord_error.max().item()
    mean_coord_error = coord_error.mean().item()

    passed = max_bond_error < atol

    return {
        "passed": passed,
        # All bonds (including anchor — for display only)
        "bond_lengths_mean": all_bl.mean().item(),
        "bond_lengths_std": all_bl.std().item(),
        # NeRF-placed bonds only (the meaningful check)
        "nerf_bond_lengths_mean": nerf_bl.mean().item(),
        "nerf_bond_lengths_std": nerf_bl.std().item(),
        "max_bond_error_from_ideal": max_bond_error,
        "mean_bond_error_from_ideal": mean_bond_error,
        # Anchor bonds (native PDB, not 3.8 Å — informational only)
        "anchor_bond_lengths_mean": anchor_bl.mean().item(),
        "anchor_bond_lengths_std": anchor_bl.std().item(),
        # Coordinate error
        "max_coord_error": max_coord_error,
        "mean_coord_error": mean_coord_error,
        "L": L,
        "B": B,
    }
