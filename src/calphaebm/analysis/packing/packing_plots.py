"""Diagnostic plots for packing geometry feature calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .packing_config import PLOT_DPI
from .packing_core import FEATURE_NAMES, FeatureStats


def plot_feature_distributions(
    raw_arrays: Dict[str, np.ndarray],
    feature_stats: Dict[str, FeatureStats],
    sig_tau: float,
    out_dir: Path,
) -> None:
    """Plot raw and normalised distributions for all 6 geometry features.

    Layout: 2 rows × 6 columns
      Top row:    raw feature histograms with mean/std annotations
      Bottom row: normalised feature histograms (count features → mean≈1; mean_r → mean≈0)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(FEATURE_NAMES)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle(f"Geometry Feature Distributions  (sig_tau = {sig_tau:.4f} Å)", fontsize=11)

    for col, name in enumerate(FEATURE_NAMES):
        f = raw_arrays[name]
        fs = feature_stats[name]

        # ── raw ───────────────────────────────────────────────────────────────
        ax = axes[0, col]
        ax.hist(f, bins=60, color="steelblue", alpha=0.7, density=True)
        ax.axvline(fs.mean, color="red", linewidth=1.5, label=f"μ={fs.mean:.3f}")
        ax.axvline(fs.mean - fs.std, color="red", linewidth=0.8, linestyle="--")
        ax.axvline(fs.mean + fs.std, color="red", linewidth=0.8, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("raw value", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        if col == 0:
            ax.set_ylabel("density", fontsize=8)

        # ── normalised ────────────────────────────────────────────────────────
        ax2 = axes[1, col]
        if name == "mean_r":
            fn = (f - fs.mean) / max(fs.std, 0.1)
            denom_label = f"(x − {fs.mean:.2f}) / {fs.std:.3f}"
        else:
            fn = f / max(fs.mean, 1e-9)
            denom_label = f"x / {fs.mean:.3f}"
        ax2.hist(fn, bins=60, color="darkorange", alpha=0.7, density=True)
        ax2.axvline(float(fn.mean()), color="navy", linewidth=1.5, label=f"μ={fn.mean():.2f}\nσ={fn.std():.2f}")
        ax2.set_xlabel(denom_label, fontsize=7)
        ax2.tick_params(labelsize=7)
        ax2.legend(fontsize=7)
        if col == 0:
            ax2.set_ylabel("normalised density", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "feature_distributions.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    from calphaebm.utils.logging import get_logger

    get_logger().info("Saved feature distribution plot to %s", out_path)
