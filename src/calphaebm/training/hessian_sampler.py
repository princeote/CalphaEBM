# src/calphaebm/training/hessian_sampler.py
"""Hessian-based harmonic negative generation.

Generates negatives by sampling from the diagonal harmonic approximation
of the energy surface at native structures.  Unlike MALA, this produces
negatives that probe the model's soft modes — directions where the energy
funnel is flat and discrimination is weakest.

Theory
======
Near native x₀, the energy surface is approximately quadratic:

    E(x₀ + δ) ≈ E(x₀) + ½ Σᵢ hᵢ δᵢ²

where hᵢ = ∂²E/∂φᵢ² is the diagonal Hessian (curvature) along torsion i.
Sampling δᵢ ~ N(0, T/hᵢ) gives:

    ΔE ~ (T/2) × χ²(n_dof)

This is harmonic BY CONSTRUCTION — the scHSM assumption is exact.

Soft modes (small hᵢ) get large perturbations → training signal where
the model needs it most.  Stiff modes (large hᵢ) get small perturbations
→ no wasted compute on what the model already discriminates.

Cost
====
Hessian computation: 2 batched forward passes per protein (not 4L individual).
Negative generation: 1 NeRF rebuild per negative (no model evaluation needed).
Anharmonicity check: 1 forward pass per negative (optional, recommended).

Usage
=====
    sampler = HessianNegativeSampler(model, device)

    # Compute curvature at native (once per protein per round)
    hessian = sampler.compute_hessian(R_native, seq, lengths)

    # Generate negatives (unlimited, essentially free)
    negatives = sampler.sample(R_native, seq, lengths, hessian,
                               n_samples=32, temps=[0.5, 1.0, 2.0, 5.0])

    # Each negative has: R_perturbed, d2_harm (analytic), E_actual (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.utils.logging import get_logger

logger = get_logger()

CA_BOND = 3.8  # Å, Cα-Cα virtual bond length


@dataclass
class HessianProfile:
    """Diagonal Hessian at a native structure."""

    h_theta: torch.Tensor  # (L-2,) curvatures for bond angles
    h_phi: torch.Tensor  # (L-3,) curvatures for torsions
    theta_native: torch.Tensor  # (L-2,) native bond angles
    phi_native: torch.Tensor  # (L-3,) native torsions
    anchor: torch.Tensor  # (3, 3) first three atoms
    E_native: float  # native energy
    L: int  # chain length


@dataclass
class HarmonicNegative:
    """A single negative generated from the harmonic approximation."""

    R: torch.Tensor  # (L, 3) perturbed Cartesian coords
    d2_harm: float  # analytic harmonic displacement: Σᵢ hᵢ δᵢ²
    dE_predicted: float  # predicted ΔE = ½ d2_harm
    T: float  # temperature used for sampling
    delta_theta: torch.Tensor  # (L-2,) perturbations applied to bond angles
    delta_phi: torch.Tensor  # (L-3,) perturbations applied to torsions
    # Filled in by anharmonicity check (optional)
    E_actual: Optional[float] = None
    dE_actual: Optional[float] = None
    anharmonicity: Optional[float] = None  # E_actual / E_predicted


class HessianNegativeSampler:
    """Generate negatives by sampling from the diagonal Hessian at native.

    Args:
        model: energy model (must be in eval mode for Hessian computation)
        device: torch device
        eps: finite difference step size (radians)
        h_min: minimum curvature clamp (prevents division by zero for flat modes)
        h_max: maximum curvature clamp (prevents vanishing perturbations for very stiff modes)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        eps: float = 0.01,  # ~0.6° in radians
        h_min: float = 1e-3,  # clamp for very flat modes
        h_max: float = 1e4,  # clamp for very stiff modes
    ):
        self.model = model
        self.device = device
        self.eps = eps
        self.h_min = h_min
        self.h_max = h_max

    @torch.no_grad()
    def compute_hessian(
        self,
        R: torch.Tensor,  # (1, L, 3) native Cα coords
        seq: torch.Tensor,  # (1, L) AA indices
        lengths: torch.Tensor,  # (1,) chain length
    ) -> HessianProfile:
        """Compute diagonal Hessian at native structure.

        Uses batched finite differences: 2 forward passes per DOF type
        (θ and φ), not 4L individual passes.

        Cost: ~3 forward passes total (1 native + 2 batched).
        """
        self.model.eval()
        L = int(R.shape[1])

        # Extract internal coordinates at native
        theta, phi = coords_to_internal(R)  # (1, L-2), (1, L-3)
        anchor = extract_anchor(R)  # (1, 3, 3)

        theta_0 = theta[0]  # (L-2,)
        phi_0 = phi[0]  # (L-3,)
        anchor_0 = anchor[0]  # (3, 3)

        # Native energy
        E0 = self.model(R, seq, lengths=lengths).item()

        n_theta = theta_0.shape[0]
        n_phi = phi_0.shape[0]
        n_dof = n_theta + n_phi

        # ── Build all perturbed IC vectors ────────────────────────────
        # Create +ε and -ε perturbations for all DOFs at once
        # theta perturbations: n_theta copies, each with one θ shifted
        # phi perturbations:   n_phi copies, each with one φ shifted

        theta_plus = theta_0.unsqueeze(0).expand(n_dof, -1).clone()  # (n_dof, L-2)
        theta_minus = theta_0.unsqueeze(0).expand(n_dof, -1).clone()
        phi_plus = phi_0.unsqueeze(0).expand(n_dof, -1).clone()  # (n_dof, L-3)
        phi_minus = phi_0.unsqueeze(0).expand(n_dof, -1).clone()

        # Perturb theta DOFs (indices 0..n_theta-1)
        for i in range(n_theta):
            theta_plus[i, i] += self.eps
            theta_minus[i, i] -= self.eps

        # Perturb phi DOFs (indices n_theta..n_dof-1)
        for i in range(n_phi):
            phi_plus[n_theta + i, i] += self.eps
            phi_minus[n_theta + i, i] -= self.eps

        # ── Rebuild all perturbed structures via NeRF ─────────────────
        anchor_batch = anchor_0.unsqueeze(0).expand(n_dof, -1, -1)  # (n_dof, 3, 3)

        R_plus = nerf_reconstruct(theta_plus, phi_plus, anchor_batch, bond=CA_BOND)
        R_minus = nerf_reconstruct(theta_minus, phi_minus, anchor_batch, bond=CA_BOND)

        # ── Batched energy evaluation ─────────────────────────────────
        seq_batch = seq.expand(n_dof, -1)  # (n_dof, L)
        lengths_batch = lengths.expand(n_dof)  # (n_dof,)

        E_plus = self.model(R_plus, seq_batch, lengths=lengths_batch)  # (n_dof,)
        E_minus = self.model(R_minus, seq_batch, lengths=lengths_batch)  # (n_dof,)

        # ── Central difference second derivative ──────────────────────
        h_all = (E_plus - 2.0 * E0 + E_minus) / (self.eps**2)  # (n_dof,)
        h_all = h_all.clamp(min=self.h_min, max=self.h_max)

        h_theta = h_all[:n_theta].cpu()  # (L-2,)
        h_phi = h_all[n_theta:].cpu()  # (L-3,)

        logger.info(
            "  Hessian [L=%d]: h_θ range [%.2f, %.2f] (mean %.2f)  "
            "h_φ range [%.2f, %.2f] (mean %.2f)  "
            "n_soft(h<1)=%d  n_stiff(h>100)=%d",
            L,
            h_theta.min().item(),
            h_theta.max().item(),
            h_theta.mean().item(),
            h_phi.min().item(),
            h_phi.max().item(),
            h_phi.mean().item(),
            int((h_all < 1.0).sum().item()),
            int((h_all > 100.0).sum().item()),
        )

        return HessianProfile(
            h_theta=h_theta,
            h_phi=h_phi,
            theta_native=theta_0.cpu(),
            phi_native=phi_0.cpu(),
            anchor=anchor_0.cpu(),
            E_native=E0,
            L=L,
        )

    def sample(
        self,
        hessian: HessianProfile,
        n_samples: int = 32,
        temps: Optional[List[float]] = None,
    ) -> List[HarmonicNegative]:
        """Generate negatives from the harmonic approximation.

        Essentially free — no model evaluation, just random draws + NeRF.

        Args:
            hessian: precomputed HessianProfile from compute_hessian()
            n_samples: total number of negatives to generate
            temps: list of temperatures. Samples are distributed evenly
                   across temperatures. If None, uses [0.5, 1.0, 2.0, 5.0].

        Returns:
            list of HarmonicNegative objects
        """
        if temps is None:
            temps = [0.5, 1.0, 2.0, 5.0]

        per_temp = max(1, n_samples // len(temps))
        negatives = []

        for T in temps:
            sigma_theta = torch.sqrt(T / hessian.h_theta)  # (L-2,)
            sigma_phi = torch.sqrt(T / hessian.h_phi)  # (L-3,)

            for _ in range(per_temp):
                # Draw perturbations
                delta_theta = torch.randn_like(hessian.theta_native) * sigma_theta
                delta_phi = torch.randn_like(hessian.phi_native) * sigma_phi

                # Harmonic displacement (analytic)
                d2_theta = (hessian.h_theta * delta_theta**2).sum().item()
                d2_phi = (hessian.h_phi * delta_phi**2).sum().item()
                d2_harm = d2_theta + d2_phi
                dE_pred = 0.5 * d2_harm

                # Rebuild perturbed structure via NeRF
                theta_pert = (hessian.theta_native + delta_theta).unsqueeze(0)
                phi_pert = (hessian.phi_native + delta_phi).unsqueeze(0)
                anchor = hessian.anchor.unsqueeze(0)

                R_pert = nerf_reconstruct(theta_pert, phi_pert, anchor, bond=CA_BOND)[0]  # (L, 3)

                negatives.append(
                    HarmonicNegative(
                        R=R_pert,
                        d2_harm=d2_harm,
                        dE_predicted=dE_pred,
                        T=T,
                        delta_theta=delta_theta,
                        delta_phi=delta_phi,
                    )
                )

        return negatives

    @torch.no_grad()
    def check_anharmonicity(
        self,
        negatives: List[HarmonicNegative],
        seq: torch.Tensor,  # (1, L)
        lengths: torch.Tensor,  # (1,)
        E_native: float,
    ) -> Dict[str, float]:
        """Compute actual energies and measure anharmonicity.

        Adds E_actual, dE_actual, and anharmonicity fields to each negative.
        Anharmonicity γ = dE_actual / dE_predicted:
            γ ≈ 1.0 → harmonic assumption is good
            γ < 1.0 → true funnel is flatter than predicted (needs more training)
            γ > 1.0 → true funnel is steeper than predicted

        Returns summary statistics.
        """
        self.model.eval()

        if not negatives:
            return {}

        # Batch all negatives for one forward pass
        R_batch = torch.stack([n.R for n in negatives]).to(self.device)
        n_neg = R_batch.shape[0]
        seq_batch = seq.expand(n_neg, -1)
        lengths_batch = lengths.expand(n_neg)

        E_batch = self.model(R_batch, seq_batch, lengths=lengths_batch)  # (n_neg,)

        gammas = []
        for i, neg in enumerate(negatives):
            E_act = E_batch[i].item()
            dE_act = E_act - E_native
            neg.E_actual = E_act
            neg.dE_actual = dE_act
            neg.anharmonicity = dE_act / max(neg.dE_predicted, 1e-8)
            gammas.append(neg.anharmonicity)

        gammas = np.array(gammas)

        summary = {
            "gamma_mean": float(gammas.mean()),
            "gamma_std": float(gammas.std()),
            "gamma_median": float(np.median(gammas)),
            "frac_subharmonic": float((gammas < 0.5).mean()),  # truly flat
            "frac_superharmonic": float((gammas > 2.0).mean()),  # steeper than expected
        }

        # Per-temperature breakdown
        for T in sorted(set(n.T for n in negatives)):
            t_gammas = [n.anharmonicity for n in negatives if n.T == T]
            if t_gammas:
                summary[f"gamma_T{T:.1f}_mean"] = float(np.mean(t_gammas))
                summary[f"gamma_T{T:.1f}_std"] = float(np.std(t_gammas))

        logger.info(
            "  Anharmonicity: γ=%.3f ± %.3f  (median %.3f)  " "sub-harmonic(<0.5)=%.0f%%  super(>2.0)=%.0f%%",
            summary["gamma_mean"],
            summary["gamma_std"],
            summary["gamma_median"],
            summary["frac_subharmonic"] * 100,
            summary["frac_superharmonic"] * 100,
        )

        return summary

    def generate_full(
        self,
        R: torch.Tensor,  # (1, L, 3)
        seq: torch.Tensor,  # (1, L)
        lengths: torch.Tensor,  # (1,)
        n_samples: int = 32,
        temps: Optional[List[float]] = None,
        check_anharmonicity: bool = True,
    ) -> tuple[List[HarmonicNegative], HessianProfile, Dict[str, float]]:
        """Full pipeline: compute Hessian → sample → anharmonicity check.

        This is the main entry point for training integration.

        Args:
            R: native Cα coordinates (batch_size=1)
            seq: amino acid indices
            lengths: chain length
            n_samples: negatives per protein
            temps: temperature schedule
            check_anharmonicity: if True, compute actual energies

        Returns:
            (negatives, hessian_profile, anharmonicity_summary)
        """
        hessian = self.compute_hessian(R, seq, lengths)
        negatives = self.sample(hessian, n_samples=n_samples, temps=temps)

        anharm_summary = {}
        if check_anharmonicity:
            anharm_summary = self.check_anharmonicity(negatives, seq, lengths, hessian.E_native)

        return negatives, hessian, anharm_summary


def hessian_curvature_report(
    hessian: HessianProfile,
    seq: Optional[torch.Tensor] = None,
) -> List[str]:
    """Generate diagnostic report of per-residue curvature.

    Shows which torsions are soft (model doesn't discriminate)
    vs stiff (model discriminates well).
    """
    lines = []
    h_phi = hessian.h_phi
    h_theta = hessian.h_theta
    L = hessian.L

    lines.append(f"  Curvature profile [L={L}]:")
    lines.append(
        f"    θ: mean={h_theta.mean():.2f}  min={h_theta.min():.2f}  "
        f"max={h_theta.max():.2f}  n_soft={int((h_theta < 1.0).sum())}"
    )
    lines.append(
        f"    φ: mean={h_phi.mean():.2f}  min={h_phi.min():.2f}  "
        f"max={h_phi.max():.2f}  n_soft={int((h_phi < 1.0).sum())}"
    )

    # Top 10 softest torsions (most informative for training)
    h_all = torch.cat([h_theta, h_phi])
    labels = [f"θ_{i+1}" for i in range(len(h_theta))] + [f"φ_{i+1}" for i in range(len(h_phi))]
    sorted_idx = h_all.argsort()

    lines.append("    Softest modes (flat funnel — training signal needed):")
    for rank, idx in enumerate(sorted_idx[:10]):
        lines.append(f"      {rank+1}. {labels[idx]:>6s}  h={h_all[idx]:.4f}")

    lines.append("    Stiffest modes (good discrimination):")
    for rank, idx in enumerate(sorted_idx[-5:].flip(0)):
        lines.append(f"      {rank+1}. {labels[idx]:>6s}  h={h_all[idx]:.2f}")

    return lines
