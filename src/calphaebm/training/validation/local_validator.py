"""Lightweight validator for local phase training.

Padding-aware: all model calls pass lengths, IC negatives mask padding.
Uses THETA_PHI_RATIO for θ noise to match training convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from calphaebm.geometry.internal import bond_angles, bond_lengths, torsions
from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.utils.logging import get_logger
from calphaebm.utils.math import wrap_to_pi

logger = get_logger()

IDEAL_BOND = 3.8
THETA_PHI_RATIO = 0.161


@dataclass
class LocalValidationMetrics:
    # Clean (baseline / sanity)
    clean_bond_mean: float
    clean_bond_std: float
    clean_bond_rmsd: float
    clean_theta_roughness: float
    clean_dphi_roughness: float

    # Distorted (fair negative check)
    dist_bond_mean: float
    dist_bond_std: float
    dist_bond_rmsd: float
    dist_theta_roughness: float
    dist_dphi_roughness: float

    # Gap (model test)
    gap_mean: float
    gap_success_rate: float
    gap_p10: float
    gap_p90: float

    # Meta
    n_batches: int
    step: int


class LocalValidator:
    """Fast geometry-only validation for local phase."""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    # ----------------------------------------------------------------
    # Model API robustness
    # ----------------------------------------------------------------

    def _local_energy(self, R: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Access the local energy term regardless of model wrapper depth."""
        if hasattr(self.model, "local"):
            return self.model.local(R, seq, lengths=lengths)
        if hasattr(self.model, "energy") and hasattr(self.model.energy, "local"):
            return self.model.energy.local(R, seq, lengths=lengths)
        raise AttributeError("Model does not expose a local energy term (.local or .energy.local).")

    # ----------------------------------------------------------------
    # Geometry helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_roughness(R: torch.Tensor, lengths: torch.Tensor | None = None) -> Tuple[float, float]:
        """Mean-squared successive-difference roughness for angles and torsions.

        Padding-aware: only computes over valid residues.
        """
        theta = bond_angles(R)  # (B, L-2)
        phi = torsions(R)  # (B, L-3)

        if lengths is not None:
            B = R.shape[0]
            # Mask for valid θ differences: need two consecutive valid θ values
            idx_t = torch.arange(theta.shape[1] - 1, device=R.device).unsqueeze(0)
            valid_t = idx_t < (lengths.unsqueeze(1) - 3)  # θ has L-2 values, diffs have L-3
            idx_p = torch.arange(phi.shape[1] - 1, device=R.device).unsqueeze(0)
            valid_p = idx_p < (lengths.unsqueeze(1) - 4)  # φ has L-3 values, diffs have L-4

            theta_diffs = (theta[:, 1:] - theta[:, :-1]) ** 2
            dphi_diffs = wrap_to_pi(phi[:, 1:] - phi[:, :-1]) ** 2

            theta_rough = (
                float((theta_diffs * valid_t.float()).sum() / valid_t.float().sum().clamp(min=1))
                if valid_t.any()
                else float("nan")
            )
            dphi_rough = (
                float((dphi_diffs * valid_p.float()).sum() / valid_p.float().sum().clamp(min=1))
                if valid_p.any()
                else float("nan")
            )
        else:
            if theta.shape[-1] >= 2:
                theta_rough = float(((theta[:, 1:] - theta[:, :-1]) ** 2).mean().item())
            else:
                theta_rough = float("nan")
            if phi.shape[-1] >= 2:
                dphi = wrap_to_pi(phi[:, 1:] - phi[:, :-1])
                dphi_rough = float((dphi**2).mean().item())
            else:
                dphi_rough = float("nan")

        return theta_rough, dphi_rough

    @staticmethod
    def _compute_bond_stats(
        R: torch.Tensor,
        lengths: torch.Tensor | None = None,
        ideal: float = IDEAL_BOND,
    ) -> Tuple[float, float, float]:
        """Bond length mean / std / RMSD-to-ideal. Padding-aware."""
        bl = bond_lengths(R)  # (B, L-1)

        if lengths is not None:
            idx = torch.arange(bl.shape[1], device=R.device).unsqueeze(0)
            valid = idx < (lengths.unsqueeze(1) - 1)
            bl_valid = bl[valid]
        else:
            bl_valid = bl.flatten()

        if bl_valid.numel() == 0:
            return float("nan"), float("nan"), float("nan")

        mean = float(bl_valid.mean().item())
        std = float(bl_valid.std().item())
        rmsd = float(torch.sqrt(((bl_valid - ideal) ** 2).mean()).item())
        return mean, std, rmsd

    # ----------------------------------------------------------------
    # IC negative construction
    # ----------------------------------------------------------------

    @staticmethod
    def _create_ic_negative(
        R: torch.Tensor,
        noise_scale: float = 0.15,
        lengths: torch.Tensor | None = None,
        bond: float = IDEAL_BOND,
    ) -> torch.Tensor:
        """Add Gaussian noise in (θ, φ) space and reconstruct via NeRF.

        Uses THETA_PHI_RATIO for θ noise. Masks padding.
        Bonds guaranteed exactly 3.8Å by NeRF reconstruction.
        """
        with torch.no_grad():
            theta, phi = coords_to_internal(R)
            anchor = extract_anchor(R)

            noise_t = THETA_PHI_RATIO * noise_scale * torch.randn_like(theta)
            noise_p = noise_scale * torch.randn_like(phi)

            if lengths is not None:
                idx_t = torch.arange(theta.shape[1], device=R.device).unsqueeze(0)
                idx_p = torch.arange(phi.shape[1], device=R.device).unsqueeze(0)
                vt = idx_t < (lengths.unsqueeze(1) - 2)
                vp = idx_p < (lengths.unsqueeze(1) - 3)
                noise_t = noise_t * vt.float()
                noise_p = noise_p * vp.float()

            theta_noisy = (theta + noise_t).clamp(0.01, math.pi - 0.01)
            phi_noisy = wrap_to_pi(phi + noise_p)

            R_neg = nerf_reconstruct(theta_noisy, phi_noisy, anchor, bond=bond)
        return R_neg.detach()

    # ----------------------------------------------------------------
    # Main validate
    # ----------------------------------------------------------------

    def validate(
        self,
        val_loader,
        n_batches: int = 5,
        step: Optional[int] = None,
        noise_scale: float = 0.15,
        n_corruptions_per_batch: int = 5,
        warn_bond_rmsd_diff: float = 0.005,
    ) -> LocalValidationMetrics:
        """Run lightweight validation on a few batches using IC negatives.

        Padding-aware: lengths extracted from batch and passed to all calls.
        """
        self.model.eval()

        clean_bond_means, clean_bond_stds, clean_bond_rmsds = [], [], []
        clean_theta_roughs, clean_dphi_roughs = [], []

        dist_bond_means, dist_bond_stds, dist_bond_rmsds = [], [], []
        dist_theta_roughs, dist_dphi_roughs = [], []

        gaps = []
        batches_used = 0

        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break

            R, seq = batch[0].to(self.device), batch[1].to(self.device)
            lengths = batch[4].to(self.device) if len(batch) > 4 else None

            # ---- clean stats (once per batch) ----
            with torch.no_grad():
                bmean, bstd, brmsd = self._compute_bond_stats(R, lengths)
                tr, dr = self._compute_roughness(R, lengths)

            clean_bond_means.append(bmean)
            clean_bond_stds.append(bstd)
            clean_bond_rmsds.append(brmsd)
            clean_theta_roughs.append(tr)
            clean_dphi_roughs.append(dr)

            # ---- generate multiple IC-noise corrupted versions ----
            for _ in range(n_corruptions_per_batch):
                R_dist = self._create_ic_negative(R, noise_scale=noise_scale, lengths=lengths)

                with torch.no_grad():
                    bmean_d, bstd_d, brmsd_d = self._compute_bond_stats(R_dist, lengths)
                    tr_d, dr_d = self._compute_roughness(R_dist, lengths)

                    E_native = self._local_energy(R, seq, lengths=lengths).mean()
                    E_dist = self._local_energy(R_dist, seq, lengths=lengths).mean()

                dist_bond_means.append(bmean_d)
                dist_bond_stds.append(bstd_d)
                dist_bond_rmsds.append(brmsd_d)
                dist_theta_roughs.append(tr_d)
                dist_dphi_roughs.append(dr_d)
                gaps.append(float((E_dist - E_native).item()))

            batches_used += 1

        # ---- handle empty case ----
        if batches_used == 0 or not gaps:
            return LocalValidationMetrics(
                clean_bond_mean=float("inf"),
                clean_bond_std=float("inf"),
                clean_bond_rmsd=float("inf"),
                clean_theta_roughness=float("inf"),
                clean_dphi_roughness=float("inf"),
                dist_bond_mean=float("inf"),
                dist_bond_std=float("inf"),
                dist_bond_rmsd=float("inf"),
                dist_theta_roughness=float("inf"),
                dist_dphi_roughness=float("inf"),
                gap_mean=float("-inf"),
                gap_success_rate=0.0,
                gap_p10=float("-inf"),
                gap_p90=float("-inf"),
                n_batches=0,
                step=int(step) if step is not None else -1,
            )

        gaps_np = np.asarray(gaps, dtype=np.float64)
        success_rate = float(np.mean(gaps_np > 0.0))

        m = LocalValidationMetrics(
            clean_bond_mean=float(np.mean(clean_bond_means)),
            clean_bond_std=float(np.mean(clean_bond_stds)),
            clean_bond_rmsd=float(np.mean(clean_bond_rmsds)),
            clean_theta_roughness=float(np.nanmean(clean_theta_roughs)),
            clean_dphi_roughness=float(np.nanmean(clean_dphi_roughs)),
            dist_bond_mean=float(np.mean(dist_bond_means)),
            dist_bond_std=float(np.mean(dist_bond_stds)),
            dist_bond_rmsd=float(np.mean(dist_bond_rmsds)),
            dist_theta_roughness=float(np.nanmean(dist_theta_roughs)),
            dist_dphi_roughness=float(np.nanmean(dist_dphi_roughs)),
            gap_mean=float(np.mean(gaps_np)),
            gap_success_rate=success_rate,
            gap_p10=float(np.percentile(gaps_np, 10)),
            gap_p90=float(np.percentile(gaps_np, 90)),
            n_batches=batches_used,
            step=int(step) if step is not None else -1,
        )

        bond_rmsd_diff = abs(m.dist_bond_rmsd - m.clean_bond_rmsd)
        if warn_bond_rmsd_diff is not None and m.dist_bond_rmsd > warn_bond_rmsd_diff:
            logger.warning(
                "IC bond check: distorted bond RMSD=%.4f Å exceeds %.4f Å threshold "
                "— NeRF should guarantee exact 3.8Å bonds; check reconstruct pipeline.",
                m.dist_bond_rmsd,
                warn_bond_rmsd_diff,
            )

        return m

    def log_validation(self, m: LocalValidationMetrics) -> None:
        """Log with clean vs distorted separation and fairness check."""
        logger.info("\n%s", "=" * 60)
        logger.info("LOCAL VALIDATION at step %d  (batches=%d)", m.step, m.n_batches)
        logger.info("%s", "=" * 60)

        logger.info("CLEAN (baseline / sanity):")
        logger.info(
            "  Bonds: mean=%.3f Å  std=%.3f Å  rmsd=%.4f Å",
            m.clean_bond_mean,
            m.clean_bond_std,
            m.clean_bond_rmsd,
        )
        logger.info("  θ roughness:  %.6f", m.clean_theta_roughness)
        logger.info("  Δφ roughness: %.6f", m.clean_dphi_roughness)

        logger.info("DISTORTED (IC-noise negative — bonds exact by NeRF):")
        logger.info(
            "  Bonds: mean=%.3f Å  std=%.3f Å  rmsd=%.4f Å",
            m.dist_bond_mean,
            m.dist_bond_std,
            m.dist_bond_rmsd,
        )
        logger.info("  θ roughness:  %.6f", m.dist_theta_roughness)
        logger.info("  Δφ roughness: %.6f", m.dist_dphi_roughness)
        logger.info(
            "  Bond RMSD to ideal: dist=%.4f Å  clean=%.4f Å  (IC negatives should be ~0)",
            m.dist_bond_rmsd,
            m.clean_bond_rmsd,
        )

        logger.info("GAP (model test):")
        logger.info("  Gap mean:    %.4f  (target > 0)", m.gap_mean)
        logger.info("  Gap success: %.1f%%", m.gap_success_rate * 100)
        logger.info("  Gap p10/p90: %.4f / %.4f", m.gap_p10, m.gap_p90)
        logger.info("%s\n", "=" * 60)

    def sweep(
        self,
        val_loader,
        noise_levels: tuple = (0.10, 0.15, 0.20, 0.35),
        n_batches: int = 10,
        n_corruptions_per_batch: int = 5,
        step: Optional[int] = None,
    ) -> None:
        """Run validate() at multiple noise levels and print a compact summary table."""
        logger.info("\n%s", "=" * 60)
        logger.info("LOCAL VALIDATION SWEEP  (step=%s)", step if step is not None else "?")
        logger.info("%s", "=" * 60)
        logger.info("  %-10s  %-8s  %-8s  %-8s  %-8s", "σ (rad)", "σ (°)", "gap", "p10", "p90")
        logger.info("  %s", "-" * 50)

        results = {}
        for sigma in noise_levels:
            m = self.validate(
                val_loader=val_loader,
                n_batches=n_batches,
                step=step,
                noise_scale=sigma,
                n_corruptions_per_batch=n_corruptions_per_batch,
            )
            deg = sigma * (180.0 / 3.14159265)
            verdict = "✓" if m.gap_mean > 0 else "✗"
            logger.info(
                "  %-10.2f  %-8.1f  %-8.4f  %-8.4f  %-8.4f  %s",
                sigma,
                deg,
                m.gap_mean,
                m.gap_p10,
                m.gap_p90,
                verdict,
            )
            results[sigma] = m

        logger.info("  %s", "-" * 50)

        gaps = [results[s].gap_mean for s in noise_levels]
        monotone = all(gaps[i] <= gaps[i + 1] for i in range(len(gaps) - 1))
        if not monotone:
            logger.warning(
                "Gap is NOT monotonically increasing with σ — " "local term may have pathological sensitivity profile."
            )
        else:
            ratio_gap = gaps[-1] / max(gaps[0], 1e-9)
            ratio_sq = (noise_levels[-1] / noise_levels[0]) ** 2
            logger.info(
                "  Sensitivity ratio: gap(%.2f)/gap(%.2f) = %.1fx  " "(quadratic expectation: %.1fx)",
                noise_levels[-1],
                noise_levels[0],
                ratio_gap,
                ratio_sq,
            )
        logger.info("%s\n", "=" * 60)
        return results
