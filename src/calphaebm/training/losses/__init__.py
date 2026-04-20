"""Loss functions for energy-based training.

Cartesian variants (dsm_cartesian_loss, force_balance_loss, local_geogap_loss)
are kept for ablations and legacy comparison.

IC variants (*_ic_loss) are the correct losses for run19+ training where
simulation runs in internal coordinate (θ, φ) space with bonds fixed at 3.8Å.
"""

# Contrastive
from calphaebm.training.losses.contrastive_losses import contrastive_logistic_loss, packing_contrastive_loss

# DSM
from calphaebm.training.losses.dsm import dsm_cartesian_loss  # ablations only
from calphaebm.training.losses.dsm import dsm_ic_loss  # run19+ default

# Force balance
from calphaebm.training.losses.force_balance import force_balance_diagnostics  # ablations only
from calphaebm.training.losses.force_balance import force_balance_ic_diagnostics  # run19+ default
from calphaebm.training.losses.force_balance import force_balance_ic_loss  # run19+ default (IC perturbation)
from calphaebm.training.losses.force_balance import force_balance_loss  # ablations only (Cartesian perturbation)

# Local geometry gap
from calphaebm.training.losses.local_geogap_loss import local_geogap_diagnostics  # ablations only
from calphaebm.training.losses.local_geogap_loss import local_geogap_ic_diagnostics  # run19+ default
from calphaebm.training.losses.local_geogap_loss import local_geogap_ic_loss  # run19+ default (IC perturbation)
from calphaebm.training.losses.local_geogap_loss import local_geogap_loss  # ablations only (Cartesian perturbation)

__all__ = [
    # DSM
    "dsm_cartesian_loss",
    "dsm_ic_loss",
    # Contrastive
    "contrastive_logistic_loss",
    "packing_contrastive_loss",
    # Force balance — Cartesian
    "force_balance_loss",
    "force_balance_diagnostics",
    # Force balance — IC
    "force_balance_ic_loss",
    "force_balance_ic_diagnostics",
    # Geogap — Cartesian
    "local_geogap_loss",
    "local_geogap_diagnostics",
    # Geogap — IC
    "local_geogap_ic_loss",
    "local_geogap_ic_diagnostics",
]
