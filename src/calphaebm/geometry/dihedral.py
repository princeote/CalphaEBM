"""Stable dihedral (torsion) angle implementation.

Convention: Standard dihedral (Oldfield & Hubbard, Proteins 1994;
PMID 8208725, DOI 10.1002/prot.340180404).

For Cα pseudo-torsions of four consecutive Cα atoms:
  α-helix: φ ≈ +50°
  β-sheet: φ ≈ -170°

This matches IUPAC, CHARMM, GROMACS, MDAnalysis conventions.
"""
import torch

from calphaebm.utils.math import safe_norm, wrap_to_pi


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors with a safe epsilon in the denominator.

    FIX: The original code used torch.where(norm > eps, v / norm, zeros).
    torch.where evaluates AND differentiates BOTH branches regardless of the
    condition. The v / norm branch produces NaN gradients when norm ≈ 0 (0/0
    in the Hessian), which then propagates even through the 'False' branch
    under DSM's create_graph=True second-order backward.

    The correct fix is to always divide by (norm + eps), which is smooth
    everywhere and avoids the 0/0 singularity entirely. The epsilon is small
    enough that it doesn't affect normal (non-degenerate) vectors.
    """
    norm = safe_norm(v, dim=-1, keepdim=True)
    return v / (norm + eps)


def dihedral(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
) -> torch.Tensor:
    """Compute dihedral angle for points p0-p1-p2-p3.

    Standard convention (Oldfield & Hubbard 1994):
      α-helix ≈ +50°, β-sheet ≈ -170° for Cα pseudo-torsions.
    """
    # Ensure all inputs are float32
    p0 = p0.float()
    p1 = p1.float()
    p2 = p2.float()
    p3 = p3.float()

    # Vectors along bonds
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    # Standard MD formula
    u = torch.cross(b0, b1, dim=-1)
    v = torch.cross(b1, b2, dim=-1)
    w = torch.cross(b1, u, dim=-1)  # Standard: cross(b1, u), not cross(u, b1)

    # FIX: Use epsilon-safe normalization instead of torch.where.
    # torch.where differentiates both branches, causing NaN Hessians when
    # norm ≈ 0 under DSM's create_graph=True second-order backward.
    u = _safe_normalize(u)
    v = _safe_normalize(v)
    w = _safe_normalize(w)

    x = torch.sum(u * v, dim=-1)
    y = torch.sum(w * v, dim=-1)

    # FIX: Remove clamp on x and y before atan2. atan2 handles all finite
    # real inputs natively, and clamping to [-1, 1] introduces non-differentiable
    # kinks that produce NaN/Inf second-order gradients at the boundaries.
    phi = torch.atan2(y, x)
    return wrap_to_pi(phi)


def dihedral_from_points(points: torch.Tensor) -> torch.Tensor:
    """Compute dihedral angles for all quadruplets in a chain."""
    points = points.float()
    p0 = points[..., :-3, :]
    p1 = points[..., 1:-2, :]
    p2 = points[..., 2:-1, :]
    p3 = points[..., 3:, :]
    return dihedral(p0, p1, p2, p3)
