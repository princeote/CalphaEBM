"""Differentiable internal coordinates from Cartesian Cα coordinates.

Cα pseudo-angles follow Oldfield & Hubbard (Proteins 1994; PMID 8208725):
  θ (bond angle):    angle at Cα_i formed by Cα_{i-1}—Cα_i—Cα_{i+1}
  φ (pseudo-torsion): dihedral of Cα_{i-1}—Cα_i—Cα_{i+1}—Cα_{i+2}

Standard convention: α-helix θ≈91° φ≈+50°, β-sheet θ≈124° φ≈-170°.

R is expected to be [B, L, 3] or [L, 3].
Outputs are batched if input is batched.
"""
import numpy as np
import torch

from calphaebm.geometry.dihedral import dihedral
from calphaebm.utils.math import safe_norm


def _ensure_batch(R):
    """Convert 2D input to batched format."""
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float()
    if R.dim() == 2:
        return R.unsqueeze(0)
    return R


def bond_lengths(R: torch.Tensor) -> torch.Tensor:
    """Compute bond lengths ℓ_i = ||r_{i+1} - r_i||.
    Args:
        R: (B, L, 3) or (L, 3) coordinates.
    Returns:
        (B, L-1) bond lengths in Å.
    """
    Rb = _ensure_batch(R)
    diffs = Rb[:, 1:, :] - Rb[:, :-1, :]
    return safe_norm(diffs, dim=-1)


def bond_angles(R: torch.Tensor) -> torch.Tensor:
    """Compute bond angles θ_i at i=1..L-2 using (i-1,i,i+1).

    Args:
        R: (B, L, 3) or (L, 3) coordinates.
    Returns:
        (B, L-2) bond angles in radians.

    FIX: The original implementation used:
        cos_theta = clamp(dot / (|u||v|), -1, 1)
        theta = acos(cos_theta)

    This is numerically unstable for second-order gradients (DSM uses
    create_graph=True). Two problems:
      1. torch.clamp is non-differentiable at the boundary values ±1.
         The Hessian is undefined there, producing NaN in second-order backward.
      2. d/dx acos(x) = -1/sqrt(1-x²), which diverges as x → ±1.
         Even values slightly inside the boundary produce huge second derivatives.

    Fix: Use atan2(|u×v|, u·v) instead of acos(clamp(cos)).
    atan2 has well-behaved first AND second derivatives everywhere, including
    near θ=0 and θ=π. The cross-product magnitude gives sin(θ) and the dot
    product gives cos(θ), so atan2 recovers θ without ever touching acos.

    SECOND FIX: Use safe_norm (sqrt(||x||² + eps)) instead of torch.norm
    for the cross product magnitude. torch.norm has undefined gradient at
    exactly zero (d/dx ||x|| = x/||x|| → 0/0 when ||x||=0), which occurs
    whenever three consecutive Cα atoms are collinear. DSM's Gaussian noise
    makes this happen regularly, producing NaN gradients that poison the
    entire batch. safe_norm adds a small epsilon under the sqrt to keep the
    gradient finite everywhere.
    """
    Rb = _ensure_batch(R)

    # Vectors from central atom to neighbors
    u = Rb[:, :-2, :] - Rb[:, 1:-1, :]  # r_{i-1} - r_i
    v = Rb[:, 2:, :] - Rb[:, 1:-1, :]  # r_{i+1} - r_i

    # dot product: cos(θ) * |u| * |v|
    dot = torch.sum(u * v, dim=-1)

    # cross product magnitude: sin(θ) * |u| * |v|  (always >= 0)
    cross = torch.linalg.cross(u, v, dim=-1)  # (B, L-2, 3)
    cross_norm = safe_norm(cross, dim=-1)  # FIX: was torch.norm

    # atan2(sin, cos) — well-behaved second derivatives everywhere
    theta = torch.atan2(cross_norm, dot)

    return theta


def torsions(R: torch.Tensor) -> torch.Tensor:
    """Compute torsion angles φ_i for quadruplets (i-1,i,i+1,i+2).
    Args:
        R: (B, L, 3) or (L, 3) coordinates.
    Returns:
        (B, L-3) torsion angles in radians.
    """
    Rb = _ensure_batch(R)
    p0 = Rb[:, :-3, :]
    p1 = Rb[:, 1:-2, :]
    p2 = Rb[:, 2:-1, :]
    p3 = Rb[:, 3:, :]
    return dihedral(p0, p1, p2, p3)


def all_internal(R: torch.Tensor) -> dict:
    """Compute all internal coordinates at once.
    Args:
        R: (B, L, 3) or (L, 3) coordinates.
    Returns:
        Dictionary with keys:
            - 'l': bond lengths
            - 'theta': bond angles
            - 'phi': torsion angles
    """
    Rb = _ensure_batch(R)
    return {
        "l": bond_lengths(Rb),
        "theta": bond_angles(Rb),
        "phi": torsions(Rb),
    }


def check_geometry(R, max_jump: float = 4.5) -> dict:
    """Check geometric sanity of a structure.
    Args:
        R: (L, 3) coordinates (numpy or torch tensor).
        max_jump: Maximum allowed Cα-Cα distance.
    Returns:
        Dictionary with validation results.
    """
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float()
    R = _ensure_batch(R)[0]
    L = R.shape[0]

    # Check bond lengths
    l = bond_lengths(R)
    l_min = l.min().item()
    l_max = l.max().item()
    l_mean = l.mean().item()

    # Check for chain breaks
    broken = (l > max_jump).sum().item()

    # Check bond angles (should be ~90-140° for proteins)
    theta = bond_angles(R) * 180 / torch.pi
    theta_min = theta.min().item()
    theta_max = theta.max().item()

    # Check for steric clashes (very rough)
    from calphaebm.utils.neighbors import pairwise_distances

    D = pairwise_distances(R.unsqueeze(0))[0]
    # Exclude bonded pairs
    for i in range(L):
        D[i, max(0, i - 2) : min(L, i + 3)] = float("inf")
    min_nonbonded = D.min().item()

    return {
        "length": L,
        "bond_lengths": {"min": l_min, "max": l_max, "mean": l_mean},
        "broken_chain": broken > 0,
        "bond_angles_deg": {"min": theta_min, "max": theta_max},
        "min_nonbonded": min_nonbonded,
        "valid": min_nonbonded > 2.5 and l_max < max_jump,
    }
