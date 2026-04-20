"""Contrastive loss computation utilities for calibration."""

import torch
import torch.nn.functional as F

from calphaebm.geometry.internal import bond_angles, torsions
from calphaebm.training.losses.contrastive_losses import contrastive_logistic_loss  # Clear import
from calphaebm.utils.langevin_utils import check_langevin_available, get_langevin_sample
from calphaebm.utils.logging import get_logger

logger = get_logger()


class ContrastiveLossComputer:
    """Helper class for contrastive loss computation."""

    def __init__(self, reg_weight: float = 1e-4):
        self.reg_weight = reg_weight

    def secondary_loss(self, model, R: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for secondary phase."""
        theta = bond_angles(R)
        phi = torsions(R)
        phi_neg = torch.rand_like(phi) * 2 * torch.pi - torch.pi

        E_pos = model.secondary.energy_from_thetaphi(theta, phi, seq)
        E_neg = model.secondary.energy_from_thetaphi(theta, phi_neg, seq)

        reg = self.reg_weight * (E_pos**2 + E_neg**2).mean()
        loss = F.softplus(E_pos - E_neg).mean() + reg

        return loss

    def packing_loss(
        self,
        model,
        R: torch.Tensor,
        seq: torch.Tensor,
        langevin_steps: int = 20,
        step_size: float = 2e-4,
        noise_level: float = 0.5,
    ) -> torch.Tensor:
        """Compute contrastive loss for packing phase."""
        if not check_langevin_available():
            # Fallback: simple Gaussian noise
            logger.debug("Using Gaussian noise fallback for packing negative generation")
            R_neg = R + noise_level * torch.randn_like(R)
        else:
            langevin_sample = get_langevin_sample()
            R_noisy = R + noise_level * torch.randn_like(R)
            try:
                snaps = langevin_sample(
                    model,
                    R0=R_noisy,
                    seq=seq,
                    n_steps=langevin_steps,
                    step_size=step_size,
                    force_cap=50.0,
                    log_every=1000,
                )
                R_neg = snaps[-1].to(R.device)
            except Exception as e:
                logger.warning(f"Langevin sampling failed: {e}. Using fallback.")
                R_neg = R + noise_level * torch.randn_like(R)

        E_pos = model.packing(R, seq)
        E_neg = model.packing(R_neg, seq)
        loss = contrastive_logistic_loss(E_pos, E_neg)

        return loss
