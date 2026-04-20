# src/calphaebm/geometry/__init__.py

"""Geometry utilities for internal coordinate calculations.

Provides differentiable functions to compute:
- Bond lengths (ℓ)
- Bond angles (θ)
- Torsion angles (φ)
- Sin/cos features for periodic variables
- Nonbonded pair selection utilities
"""

from calphaebm.geometry.dihedral import dihedral
from calphaebm.geometry.features import phi_sincos
from calphaebm.geometry.internal import bond_angles, bond_lengths, torsions

# Pair selection for nonbonded interactions (packing/repulsion)
from calphaebm.geometry.pairs import topk_nonbonded_pairs

__all__ = [
    "bond_lengths",
    "bond_angles",
    "torsions",
    "dihedral",
    "phi_sincos",
    "topk_nonbonded_pairs",
]
