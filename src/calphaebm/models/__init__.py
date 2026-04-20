# src/calphaebm/models/__init__.py

from __future__ import annotations

from calphaebm.models.embeddings import AAEmbedding
from calphaebm.models.energy import TotalEnergy, create_total_energy
from calphaebm.models.local import LocalEnergy
from calphaebm.models.mlp import MLP
from calphaebm.models.packing import PackingEnergy
from calphaebm.models.repulsion import RepulsionEnergy
from calphaebm.models.secondary import SecondaryStructureEnergy

# Backward-compatibility: older code that imports cross_terms / local_terms
try:
    from calphaebm.models.packing import SimplePackingEnergy
except ImportError:
    SimplePackingEnergy = PackingEnergy  # type: ignore


__all__ = [
    "AAEmbedding",
    "LocalEnergy",
    "RepulsionEnergy",
    "SecondaryStructureEnergy",
    "PackingEnergy",
    "SimplePackingEnergy",
    "TotalEnergy",
    "create_total_energy",
    "MLP",
]
