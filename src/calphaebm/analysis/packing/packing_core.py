"""Packing geometry feature calibration core.

Computes:
  sig_tau   — analytically: exp(mean(log σ_min, log σ_max)) × √2
              This maximises the expected sigmoid gradient averaged over the
              log-uniform DSM sigma distribution.
  norm_*    — per-feature normalisation denominators from training structures:
                count features (n_tight/medium/loose) → divide by mean
                mean_r                                 → subtract mean, divide by std
                std_r, inv_sq                          → divide by mean

Outputs:
  {output_dir}/data/geometry_feature_calibration.json  ← consumed by --packing-geom-calibration
  {output_dir}/data/feature_stats.npz                  ← raw arrays for inspection
  {output_dir}/calibration_summary.json                ← human-readable report
  {output_dir}/feature_distributions.png               ← diagnostic plots
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from calphaebm.utils.logging import get_logger

from .packing_config import (
    DEFAULT_N_STRUCTURES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEGMENTS_PT,
    DEFAULT_SIGMA_MAX,
    DEFAULT_SIGMA_MIN,
    EXCLUDE,
    MAX_DIST,
    R_CUT,
    R_ON,
    SHELL_CUTOFFS,
    SHELL_HALF_WIDTH,
    SHORT_GATE_OFF,
    SHORT_GATE_ON,
    TOPK,
)
from .parking_data_loader import load_segments, segments_to_coord_batch

logger = get_logger()

FEATURE_NAMES = ["n_tight", "n_medium", "n_loose", "mean_r", "std_r", "inv_sq"]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (self-contained — no model import required)
# ─────────────────────────────────────────────────────────────────────────────


def _cosine_gate(r, r_on: float, r_off: float):
    x = ((r - r_on) / (r_off - r_on)).clamp(0.0, 1.0)
    return 0.5 * (1.0 - (math.pi * x).cos())


def _cosine_switch(r, r_on: float, r_cut: float):
    x = ((r - r_on) / (r_cut - r_on)).clamp(0.0, 1.0)
    return 0.5 * (1.0 + (math.pi * x).cos())


def _topk_distances(R, k: int = TOPK, exclude: int = EXCLUDE, max_dist: float = MAX_DIST):
    """Return (B, L, k) Cα–Cα distances; excluded/OOB pairs → +inf."""
    import torch

    B, L, _ = R.shape
    diff = R.unsqueeze(2) - R.unsqueeze(1)  # (B,L,L,3)
    dist = diff.norm(dim=-1)  # (B,L,L)

    mask = torch.ones(L, L, dtype=torch.bool, device=R.device)
    for d in range(-exclude, exclude + 1):
        i = torch.arange(L, device=R.device)
        j = (i + d).clamp(0, L - 1)
        ok = (i + d >= 0) & (i + d < L)
        mask[i[ok], j[ok]] = False

    dist = dist.masked_fill(~mask.unsqueeze(0), float("inf"))
    dist = dist.masked_fill(dist > max_dist, float("inf"))

    k_use = min(k, L - 2 * exclude - 1)
    if k_use <= 0:
        import torch as t

        return t.full((B, L, k), float("inf"), device=R.device)

    topk, _ = dist.topk(k_use, dim=-1, largest=False)
    if k_use < k:
        import torch as t

        pad = t.full((B, L, k - k_use), float("inf"), device=R.device)
        topk = t.cat([topk, pad], dim=-1)

    return topk


def _raw_features(r, tau: float):
    """Raw (unnormalised) geometry features.  Returns (B, L, 6)."""
    import torch

    tight_cut, medium_cut, loose_cut = SHELL_CUTOFFS
    valid_f = (r < MAX_DIST - 1e-4).float()
    n_valid = valid_f.sum(-1).clamp(min=1)
    r_safe = r.clamp(max=MAX_DIST)

    sw_short = _cosine_gate(r_safe, SHORT_GATE_ON, SHORT_GATE_OFF) * valid_f
    sw_long = _cosine_switch(r_safe, R_ON, R_CUT)

    tau_c = max(tau, 1e-3)
    n_tight = (torch.sigmoid((tight_cut - r_safe) / tau_c) * sw_short).sum(-1)
    n_medium = (torch.sigmoid((medium_cut - r_safe) / tau_c) * sw_short).sum(-1)
    n_loose = (torch.sigmoid((loose_cut - r_safe) / tau_c) * sw_short * sw_long).sum(-1)

    mean_r = (r_safe * valid_f).sum(-1) / n_valid
    diff_sq = (r_safe - mean_r.unsqueeze(-1)).pow(2) * valid_f
    std_r = (diff_sq.sum(-1) / n_valid + 1e-4).sqrt()
    inv_sq = (sw_short / (r_safe.pow(2) + 1.0)).sum(-1)

    return torch.stack([n_tight, n_medium, n_loose, mean_r, std_r, inv_sq], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FeatureStats:
    mean: float
    std: float
    p5: float
    p95: float


@dataclass
class GeometryCalibration:
    """All parameters consumed by _GeometryFeatures via --packing-geom-calibration."""

    sig_tau: float
    norm_n_tight: float
    norm_n_medium: float
    norm_n_loose: float
    norm_mean_r_centre: float
    norm_mean_r_scale: float
    norm_std_r: float
    norm_inv_sq: float
    # Provenance
    sigma_min: float
    sigma_max: float
    n_structures: int
    sigma_geom: float

    def to_dict(self) -> dict:
        return asdict(self)

    def log(self) -> None:
        logger.info("=" * 60)
        logger.info("Geometry feature calibration results")
        logger.info("=" * 60)
        logger.info("  sigma range : [%.3f, %.3f] Å (log-uniform)", self.sigma_min, self.sigma_max)
        logger.info("  sigma_geom  : %.4f Å", self.sigma_geom)
        logger.info("  sig_tau     : %.4f Å  (= sigma_geom × √2)", self.sig_tau)
        logger.info("  norm_n_tight        : %.4f", self.norm_n_tight)
        logger.info("  norm_n_medium       : %.4f", self.norm_n_medium)
        logger.info("  norm_n_loose        : %.4f", self.norm_n_loose)
        logger.info("  norm_mean_r_centre  : %.4f", self.norm_mean_r_centre)
        logger.info("  norm_mean_r_scale   : %.4f", self.norm_mean_r_scale)
        logger.info("  norm_std_r          : %.4f", self.norm_std_r)
        logger.info("  norm_inv_sq         : %.6f", self.norm_inv_sq)
        logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main analyser
# ─────────────────────────────────────────────────────────────────────────────


class PackingGeometryAnalyzer:
    """Calibrates _GeometryFeatures parameters from a processed segments file."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ):
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        segments_pt: Path = DEFAULT_SEGMENTS_PT,
        n_structures: int = DEFAULT_N_STRUCTURES,
        sigma_min: float = DEFAULT_SIGMA_MIN,
        sigma_max: float = DEFAULT_SIGMA_MAX,
        quiet: bool = False,
    ) -> GeometryCalibration:
        import torch

        # ── Load data ─────────────────────────────────────────────────────────
        data, load_stats = load_segments(
            segments_pt=segments_pt,
            n_structures=n_structures,
            verbose=not quiet,
        )
        R_batch = segments_to_coord_batch(data, verbose=not quiet)

        # ── Step 1: sig_tau (analytical) ──────────────────────────────────────
        # tau_opt = sigma_geom × √2  where  sigma_geom = exp(mean(log σ_min, log σ_max))
        #
        # Derivation: the expected gradient of sigmoid((c - r) / τ) under DSM noise
        # N(0, σ²) is maximised when τ ≈ σ√2.  For log-uniform sigma in [σ_min, σ_max]
        # the score weighting is flat in log-space (σ² × 1/σ² = const), so the
        # representative sigma is the geometric mean.
        sigma_geom = math.exp(0.5 * (math.log(sigma_min) + math.log(sigma_max)))
        sig_tau = sigma_geom * math.sqrt(2)
        logger.info("sig_tau = exp(mean(log %.3f, log %.3f)) × √2 = %.4f Å", sigma_min, sigma_max, sig_tau)

        # ── Step 2: per-cutoff distance spread (diagnostic) ───────────────────
        with torch.no_grad():
            diff = R_batch.unsqueeze(2) - R_batch.unsqueeze(1)
            dist_all = diff.norm(dim=-1)
            eye = torch.eye(R_batch.shape[1], dtype=torch.bool).unsqueeze(0)
            dist_all = dist_all.masked_fill(eye, float("inf"))
            dist_flat = dist_all[dist_all < float("inf")]

        logger.info("Per-cutoff distance spread (diagnostic):")
        for cut in SHELL_CUTOFFS:
            lo, hi = cut - SHELL_HALF_WIDTH, cut + SHELL_HALF_WIDTH
            near = dist_flat[(dist_flat >= lo) & (dist_flat <= hi)]
            if len(near) >= 10:
                logger.info("  cutoff %.1fÅ: std=%.3fÅ  n=%d", cut, float(near.std()), len(near))
            else:
                logger.warning("  cutoff %.1fÅ: too few pairs for std estimate", cut)

        # ── Step 3: raw feature statistics at sig_tau ─────────────────────────
        logger.info("Computing feature statistics at tau=%.4fÅ ...", sig_tau)
        with torch.no_grad():
            r_batch = _topk_distances(R_batch)
            feats = _raw_features(r_batch, sig_tau)  # (B, max_L, 6)
            has_nbr = (r_batch < MAX_DIST - 1e-4).any(-1)  # (B, max_L) — valid residues

        feature_stats: Dict[str, FeatureStats] = {}
        raw_arrays: Dict[str, np.ndarray] = {}

        logger.info("%-12s  %8s  %8s  %8s  %8s", "feature", "mean", "std", "p5", "p95")
        for i, name in enumerate(FEATURE_NAMES):
            f = feats[:, :, i][has_nbr].numpy()
            raw_arrays[name] = f
            fs = FeatureStats(
                mean=float(f.mean()),
                std=float(f.std()),
                p5=float(np.percentile(f, 5)),
                p95=float(np.percentile(f, 95)),
            )
            feature_stats[name] = fs
            logger.info("  %-12s  %8.4f  %8.4f  %8.4f  %8.4f", name, fs.mean, fs.std, fs.p5, fs.p95)

        # ── Step 4: normalisation parameters ──────────────────────────────────
        cal = GeometryCalibration(
            sig_tau=sig_tau,
            norm_n_tight=max(feature_stats["n_tight"].mean, 1e-6),
            norm_n_medium=max(feature_stats["n_medium"].mean, 1e-6),
            norm_n_loose=max(feature_stats["n_loose"].mean, 1e-6),
            norm_mean_r_centre=feature_stats["mean_r"].mean,
            norm_mean_r_scale=max(feature_stats["mean_r"].std, 0.1),
            norm_std_r=max(feature_stats["std_r"].mean, 1e-6),
            norm_inv_sq=max(feature_stats["inv_sq"].mean, 1e-8),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            n_structures=len(data),
            sigma_geom=sigma_geom,
        )
        cal.log()

        # ── Step 5: verify normalisation ──────────────────────────────────────
        normed_stats: dict = {}
        logger.info("Verification (normalised features should be mean≈1 or mean≈0):")
        for i, name in enumerate(FEATURE_NAMES):
            f = feats[:, :, i][has_nbr].numpy()
            if name == "mean_r":
                fn = (f - cal.norm_mean_r_centre) / cal.norm_mean_r_scale
            elif name == "n_tight":
                fn = f / cal.norm_n_tight
            elif name == "n_medium":
                fn = f / cal.norm_n_medium
            elif name == "n_loose":
                fn = f / cal.norm_n_loose
            elif name == "std_r":
                fn = f / cal.norm_std_r
            else:  # inv_sq
                fn = f / cal.norm_inv_sq
            normed_stats[name] = {"mean": float(fn.mean()), "std": float(fn.std())}
            logger.info("  %-12s_n: mean=%.3f  std=%.3f", name, fn.mean(), fn.std())

        # ── Save outputs ──────────────────────────────────────────────────────
        self._save(cal, feature_stats, normed_stats, raw_arrays)

        return cal

    def _save(
        self,
        cal: GeometryCalibration,
        feature_stats: Dict[str, FeatureStats],
        normed_stats: dict,
        raw_arrays: Dict[str, np.ndarray],
    ) -> None:
        from dataclasses import asdict

        # 1. geometry_feature_calibration.json — consumed by --packing-geom-calibration
        cal_json_path = self.data_dir / "geometry_feature_calibration.json"
        payload = {
            "norm_params": {
                "sig_tau": cal.sig_tau,
                "norm_n_tight": cal.norm_n_tight,
                "norm_n_medium": cal.norm_n_medium,
                "norm_n_loose": cal.norm_n_loose,
                "norm_mean_r_centre": cal.norm_mean_r_centre,
                "norm_mean_r_scale": cal.norm_mean_r_scale,
                "norm_std_r": cal.norm_std_r,
                "norm_inv_sq": cal.norm_inv_sq,
            },
            "provenance": {
                "sigma_min": cal.sigma_min,
                "sigma_max": cal.sigma_max,
                "sigma_geom": cal.sigma_geom,
                "n_structures": cal.n_structures,
                "tau_derivation": "sig_tau = exp(mean(log σ_min, log σ_max)) × √2",
            },
        }
        cal_json_path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved calibration JSON to %s", cal_json_path)

        # 2. feature_stats.npz — raw arrays for downstream inspection
        np.savez(
            self.data_dir / "feature_stats.npz",
            **raw_arrays,
        )

        # 3. calibration_summary.json — human-readable
        summary = {
            "calibration": cal.to_dict(),
            "feature_stats": {name: asdict(fs) for name, fs in feature_stats.items()},
            "normalised_stats": normed_stats,
        }
        (self.output_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2))

        # 4. Diagnostic plot
        try:
            from .packing_plots import plot_feature_distributions

            plot_feature_distributions(
                raw_arrays=raw_arrays,
                feature_stats=feature_stats,
                sig_tau=cal.sig_tau,
                out_dir=self.output_dir,
            )
        except Exception as e:
            logger.warning("Could not generate plots: %s", e)

        logger.info("✅ Parking analysis complete. Outputs in %s", self.output_dir)
        logger.info("   Calibration JSON: %s", cal_json_path)
        logger.info("   Pass to training: --packing-geom-calibration %s", cal_json_path)


def run_packing_analysis(args) -> int:
    """Entry point called from CLI."""
    analyzer = PackingGeometryAnalyzer(output_dir=Path(args.output_dir))
    analyzer.run(
        segments_pt=Path(args.segments),
        n_structures=args.n_structures,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        quiet=getattr(args, "quiet", False),
    )
    return 0
