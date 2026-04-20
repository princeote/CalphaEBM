"""Validator that tests model behavior on specific test cases.

Padding-aware: all model calls pass lengths, all IC perturbation masks padding.
Tuned for full-phase validation with dynamics-relevant σ range.

Key metrics for dynamics:
  - native_vs_distorted_gap: E(perturbed) − E(native), positive = correct funnel
  - gap_profile: gap at multiple σ values — reveals basin shape
  - secondary_helix_gap: secondary term only, positive = helix preferred
  - per_subterm_gaps: gap decomposed by subterm — shows which terms contribute
"""

from __future__ import annotations

import math

import numpy as np
import torch

from calphaebm.data.synthetic import make_extended_chain, make_helix
from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.utils.logging import get_logger

from .base import BaseValidator

logger = get_logger()

ALA_IDX = 1
THETA_PHI_RATIO = 0.161


class BehaviorValidator(BaseValidator):
    """Validates model behaviour on synthetic and real test cases.

    All methods return a scalar "gap" defined as:
        gap = E(bad/distorted) − E(good/native)
    so positive gap → model correctly assigns lower energy to the preferred state.
    """

    # ------------------------------------------------------------------
    # IC perturbation helper — shared across all tests
    # ------------------------------------------------------------------

    @staticmethod
    def _ic_perturb(R, sigma, lengths=None):
        """IC-perturb a batch with scalar σ.

        Returns R_neg with bonds at 3.8Å. Padding masked.
        """
        with torch.no_grad():
            theta, phi = coords_to_internal(R)
            anchor = extract_anchor(R)

            noise_t = THETA_PHI_RATIO * sigma * torch.randn_like(theta)
            noise_p = sigma * torch.randn_like(phi)

            if lengths is not None:
                idx_t = torch.arange(theta.shape[1], device=R.device).unsqueeze(0)
                idx_p = torch.arange(phi.shape[1], device=R.device).unsqueeze(0)
                vt = idx_t < (lengths.unsqueeze(1) - 2)
                vp = idx_p < (lengths.unsqueeze(1) - 3)
                noise_t = noise_t * vt.float()
                noise_p = noise_p * vp.float()

            theta_p = (theta + noise_t).clamp(0.01, math.pi - 0.01)
            phi_p = phi + noise_p
            phi_p = (phi_p + math.pi) % (2 * math.pi) - math.pi

            return nerf_reconstruct(theta_p, phi_p, anchor)

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def validate_native_vs_distorted(
        self,
        R: torch.Tensor,
        seq: torch.Tensor,
        noise_scale: float = 0.3,
        lengths: torch.Tensor | None = None,
    ) -> float:
        """Return E(distorted) − E(native) averaged over the batch.

        Positive → model prefers native.
        """
        with torch.no_grad():
            E_native = self.model(R, seq, lengths=lengths).mean()
            R_dist = self._ic_perturb(R, noise_scale, lengths)
            E_distorted = self.model(R_dist, seq, lengths=lengths).mean()

        return float((E_distorted - E_native).item())

    def validate_helix_vs_random(
        self,
        length: int = 20,
        helix_noise: float = 0.02,
        random_noise: float = 0.5,
    ) -> float:
        """Return E(extended) − E(helix) using poly-alanine sequence.

        Positive → model assigns lower energy to helix than extended coil.
        """
        R_helix = make_helix(batch=1, length=length, noise=helix_noise).to(self.device)
        R_extended = make_extended_chain(batch=1, length=length, noise=random_noise).to(self.device)
        seq_ala = torch.full((1, length), fill_value=ALA_IDX, dtype=torch.long, device=self.device)
        lens = torch.tensor([length], dtype=torch.long, device=self.device)

        with torch.no_grad():
            E_helix = self.model(R_helix, seq_ala, lengths=lens).mean()
            E_extended = self.model(R_extended, seq_ala, lengths=lens).mean()

        return float((E_extended - E_helix).item())

    def validate_secondary_term(
        self,
        length: int = 20,
        helix_noise: float = 0.02,
        random_noise: float = 0.5,
    ) -> float:
        """Return E_secondary(extended) − E_secondary(helix) using poly-alanine.

        Isolates the secondary structure term. Positive → secondary prefers helix.
        """
        if not hasattr(self.model, "secondary") or self.model.secondary is None:
            return 0.0

        R_helix = make_helix(batch=1, length=length, noise=helix_noise).to(self.device)
        R_extended = make_extended_chain(batch=1, length=length, noise=random_noise).to(self.device)
        seq_ala = torch.full((1, length), fill_value=ALA_IDX, dtype=torch.long, device=self.device)
        lens = torch.tensor([length], dtype=torch.long, device=self.device)

        with torch.no_grad():
            E_helix_ss = self.model.secondary(R_helix, seq_ala, lengths=lens).mean()
            E_extended_ss = self.model.secondary(R_extended, seq_ala, lengths=lens).mean()

        gap = float((E_extended_ss - E_helix_ss).item())
        sign = "OK ✓" if gap > 0 else "INVERTED ✗"
        logger.info(
            "[secondary_term] E_extended=%.3f  E_helix=%.3f  gap=%.3f  [%s]",
            E_extended_ss.item(),
            E_helix_ss.item(),
            gap,
            sign,
        )
        return gap

    def _per_subterm_gap(
        self,
        R: torch.Tensor,
        seq: torch.Tensor,
        R_dist: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict:
        """Compute gap per subterm: E_sub(distorted) − E_sub(native)."""
        gaps = {}
        for name in ("local", "repulsion", "secondary", "packing"):
            term = getattr(self.model, name, None)
            if term is None:
                continue
            try:
                with torch.no_grad():
                    e_nat = term(R, seq, lengths=lengths).mean()
                    e_dist = term(R_dist, seq, lengths=lengths).mean()
                gaps[name] = float((e_dist - e_nat).item())
            except Exception:
                gaps[name] = float("nan")
        return gaps

    # ------------------------------------------------------------------
    # Main validate
    # ------------------------------------------------------------------

    def validate(self, val_loader) -> dict:
        """Run all behaviour tests and return a metrics dict.

        Gap profiling uses IC-space perturbations with padding masks.
        σ range includes dynamics-relevant small values (0.05, 0.1)
        in addition to larger discriminative values.

        Keys returned
        -------------
        native_vs_distorted_gap : float  — gap at σ=0.3, positive is good
        helix_vs_random_gap     : float  — full model helix preference
        secondary_helix_gap     : float  — secondary term only
        energy_mean / energy_std: float
        gap_profile             : dict[float, float] — gap at each sigma
        mean_gap                : float
        per_subterm_gaps        : dict — per-term gap at σ=0.3
        small_sigma_gap         : float — gap at σ=0.05 (dynamics-critical)
        """
        all_energies = []

        # Dynamics-relevant σ range: small values matter most for Langevin
        gap_sigmas = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
        native_gaps_by_sigma: dict[float, list] = {s: [] for s in gap_sigmas}

        # Per-subterm gaps at σ=0.3 (accumulated across batches)
        subterm_gap_accum: dict[str, list] = {}

        with torch.no_grad():
            for batch in val_loader:
                R, seq = batch[0].to(self.device), batch[1].to(self.device)
                lengths = batch[4].to(self.device) if len(batch) > 4 else None

                E = self.model(R, seq, lengths=lengths)
                all_energies.extend(E.cpu().numpy().flatten().tolist())

                E_native = self.model(R, seq, lengths=lengths).mean()

                try:
                    for sig in gap_sigmas:
                        R_dist = self._ic_perturb(R, sig, lengths)
                        E_dist = self.model(R_dist, seq, lengths=lengths).mean()
                        native_gaps_by_sigma[sig].append(float((E_dist - E_native).item()))

                        # Per-subterm at σ=0.3
                        if sig == 0.3 and not subterm_gap_accum:
                            sg = self._per_subterm_gap(R, seq, R_dist, lengths)
                            for k, v in sg.items():
                                subterm_gap_accum.setdefault(k, []).append(v)

                except Exception:
                    pass

        # Per-sigma mean gaps
        gap_profile = {sig: float(np.mean(gaps)) if gaps else 0.0 for sig, gaps in native_gaps_by_sigma.items()}
        mean_gap = float(np.mean(list(gap_profile.values()))) if gap_profile else 0.0

        native_gap_mean = gap_profile.get(0.3, 0.0)
        small_sigma_gap = gap_profile.get(0.05, 0.0)

        secondary_gap = self.validate_secondary_term()

        per_subterm_gaps = {k: float(np.mean(v)) for k, v in subterm_gap_accum.items()}

        metrics = {
            "native_vs_distorted_gap": native_gap_mean,
            "helix_vs_random_gap": secondary_gap,
            "secondary_helix_gap": secondary_gap,
            "energy_mean": float(np.mean(all_energies)) if all_energies else 0.0,
            "energy_std": float(np.std(all_energies)) if all_energies else 0.0,
            "energy_consistency": float(np.std(native_gaps_by_sigma.get(0.3, [])))
            if native_gaps_by_sigma.get(0.3)
            else 0.0,
            "gap_profile": gap_profile,
            "mean_gap": mean_gap,
            "small_sigma_gap": small_sigma_gap,
            "per_subterm_gaps": per_subterm_gaps,
        }

        gap_str = "  ".join(f"@{s:.2f}r={gap_profile[s]:+.3f}" for s in gap_sigmas)
        logger.info(
            "[behavior] native_gap=%.3f  small_σ_gap=%.3f  mean_gap=%.3f  [%s]  "
            "secondary_gap=%.3f  E_mean=%.2f ± %.2f",
            metrics["native_vs_distorted_gap"],
            small_sigma_gap,
            mean_gap,
            gap_str,
            metrics["secondary_helix_gap"],
            metrics["energy_mean"],
            metrics["energy_std"],
        )
        if per_subterm_gaps:
            st_str = "  ".join(f"{k}={v:+.4f}" for k, v in per_subterm_gaps.items())
            logger.info("[behavior] per-subterm gaps @σ=0.3: %s", st_str)

        return metrics
