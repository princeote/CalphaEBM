"""Contrastive loss for self-consistent training.

src/calphaebm/training/losses/self_consistent_loss.py

Exponential margin loss on (native, negative) pairs collected by the
NegativeCollector.  Ensures E(native) < E(negative) - margin(Q) for all
failure-mode configurations discovered during CG Langevin dynamics.

Run5: Q-scaled saturating exponential margin.
  margin(Q) = m * (1 - exp(-α * (1 - Q_neg)))
  - Near-native negatives (Q≈1): margin≈0 (allow thermal fluctuation)
  - Far negatives (Q≈0): margin→m (strong penalty)
  - Encodes funnel shape directly into the gap loss

Design choices:
  - Exponential (not ReLU): always provides gradient, even when the
    margin is already satisfied.  Stronger push when violated.
  - Per-residue energies: E/res normalization so loss is comparable
    across different chain lengths.
  - Category weighting: different failure types can receive different
    loss weights (e.g., false_basin > compacted > frozen).
  - Clamp at max=5.0 to prevent overflow (exp(5) ≈ 148).

Usage:
    from calphaebm.training.losses.self_consistent_loss import SelfConsistentLoss
    from calphaebm.training.negative_collector import NegativeExample

    sc_loss = SelfConsistentLoss(model, device, margin_m=5.0, margin_alpha=5.0)

    # During training:
    loss = sc_loss(negatives_batch)  # list of NegativeExample
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from calphaebm.utils.logging import get_logger

logger = get_logger()


DEFAULT_CATEGORY_WEIGHTS = {
    "false_basin": 2.0,
    "compacted": 1.0,
    "swollen": 1.0,
    "ss_loss": 1.0,
    "drift_preserved": 0.5,
    "frozen": 0.5,
}


def _saturating_margin(delta_q: torch.Tensor, m: float, alpha: float) -> torch.Tensor:
    """Compute saturating exponential margin: m * (1 - exp(-α * ΔQ)).

    Args:
        delta_q: (N,) structural difference from native (1 - Q_neg).
        m:       Maximum margin (saturation level).
        alpha:   Steepness. 1/α = characteristic ΔQ for 63% of max.

    Returns:
        (N,) margin tensor in [0, m).
    """
    return m * (1.0 - torch.exp(-alpha * delta_q.clamp(min=0.0)))


class SelfConsistentLoss(nn.Module):
    """Q-scaled exponential margin loss on native vs failure-mode configurations.

    For each (R_native, R_negative) pair with Q_neg:

        required_margin = m * (1 - exp(-α * (1 - Q_neg)))
        L = w_cat · exp(clamp(E_nat - E_neg + required_margin, max=clamp_max))

    where:
        E_nat     = E(R_native, seq)    (model returns per-residue energy)
        E_neg     = E(R_negative, seq)  (model returns per-residue energy)
        Q_neg     = fraction of native contacts in negative structure
        m         = maximum margin (saturation level)
        α         = margin steepness
        w_cat     = category-specific weight

    Args:
        model:            TotalEnergy model
        device:           torch device
        margin_m:         maximum margin (E/res). Default 5.0.
        margin_alpha:     margin steepness. Default 5.0.
        margin:           DEPRECATED constant margin (used as fallback if Q unavailable)
        clamp_max:        clamp exponent to prevent overflow (default 5.0)
        category_weights: dict mapping category name to loss weight
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        margin_m: float = 5.0,
        margin_alpha: float = 5.0,
        margin: float = 0.5,
        clamp_max: float = 5.0,
        category_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.margin_m = margin_m
        self.margin_alpha = margin_alpha
        self.margin = margin  # fallback if Q not available
        self.clamp_max = clamp_max
        self.category_weights = category_weights or DEFAULT_CATEGORY_WEIGHTS

    def forward(
        self,
        negatives: List,
        return_diagnostics: bool = False,
    ) -> "torch.Tensor | tuple[torch.Tensor, dict]":
        """Compute contrastive loss over a batch of negative examples.

        Each negative must have .q attribute (fraction of native contacts).
        If .q is missing, falls back to constant self.margin.
        """
        if not negatives:
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            if return_diagnostics:
                return zero, {"n_pairs": 0}
            return zero

        by_protein: dict = {}
        for neg in negatives:
            key = f"{neg.pdb_id}_{neg.chain_id}"
            if key not in by_protein:
                by_protein[key] = []
            by_protein[key].append(neg)

        total_loss = torch.tensor(0.0, device=self.device)
        n_pairs = 0
        category_losses = {}
        category_counts = {}
        margin_violations = 0

        for key, negs in by_protein.items():
            L = negs[0].R_native.shape[0]
            N = len(negs)

            R_all = torch.zeros(2 * N, L, 3, device=self.device)
            seq_t = negs[0].seq.unsqueeze(0).expand(2 * N, -1).to(self.device)
            lengths_t = torch.tensor([L] * (2 * N), device=self.device)

            for i, neg in enumerate(negs):
                R_all[2 * i] = neg.R_native.to(self.device)
                R_all[2 * i + 1] = neg.R_negative.to(self.device)

            E_all = self.model(R_all, seq_t, lengths=lengths_t)
            if isinstance(E_all, tuple):
                E_all = E_all[0]

            E_nat = E_all[0::2]  # (N,)
            E_neg = E_all[1::2]  # (N,)

            # Q-scaled margin per negative
            q_available = hasattr(negs[0], "q") and negs[0].q is not None
            if q_available:
                q_vals = torch.tensor([neg.q for neg in negs], device=self.device, dtype=torch.float32)
                delta_q = (1.0 - q_vals).clamp(min=0.0)
                per_pair_margin = _saturating_margin(delta_q, self.margin_m, self.margin_alpha)
            else:
                per_pair_margin = torch.full((N,), self.margin, device=self.device)

            # Per-pair loss: exp(clamp(E_nat - E_neg + margin, max))
            exponent = (E_nat - E_neg + per_pair_margin).clamp(max=self.clamp_max)
            pair_losses = torch.exp(exponent)

            for i, neg in enumerate(negs):
                cat = neg.category.value if neg.category else "unknown"
                w = self.category_weights.get(cat, 1.0)
                total_loss = total_loss + w * pair_losses[i]
                n_pairs += 1

                if exponent[i].item() > 0:
                    margin_violations += 1

                if cat not in category_losses:
                    category_losses[cat] = 0.0
                    category_counts[cat] = 0
                category_losses[cat] += pair_losses[i].item()
                category_counts[cat] += 1

        loss = total_loss / max(n_pairs, 1)

        if return_diagnostics:
            diag = {
                "n_pairs": n_pairs,
                "margin_violations": margin_violations,
                "violation_frac": margin_violations / max(n_pairs, 1),
                "mean_loss": loss.item(),
                "margin_m": self.margin_m,
                "margin_alpha": self.margin_alpha,
            }
            for cat in category_losses:
                diag[f"loss_{cat}"] = category_losses[cat] / category_counts[cat]
                diag[f"n_{cat}"] = category_counts[cat]
            return loss, diag

        return loss


def compute_self_consistent_loss(
    model: nn.Module,
    negatives: List,
    device: torch.device,
    margin_m: float = 5.0,
    margin_alpha: float = 5.0,
    margin: float = 0.5,
    clamp_max: float = 5.0,
    category_weights: Optional[Dict[str, float]] = None,
) -> "tuple[torch.Tensor, dict]":
    """Functional interface for self-consistent contrastive loss."""
    sc = SelfConsistentLoss(
        model=model,
        device=device,
        margin_m=margin_m,
        margin_alpha=margin_alpha,
        margin=margin,
        clamp_max=clamp_max,
        category_weights=category_weights,
    )
    return sc(negatives, return_diagnostics=True)
