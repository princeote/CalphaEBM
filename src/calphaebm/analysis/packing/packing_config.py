"""Configuration for packing geometry feature calibration.

Computes _GeometryFeatures parameters used by the packing MLP:
  - sig_tau:  sigmoid transition width (Å), set analytically from DSM sigma range
  - norm_*:   per-feature normalisation denominators, computed from data

Outputs:
  analysis/packing_analysis/data/geometry_feature_calibration.json  ← consumed by training
  analysis/packing_analysis/data/feature_stats.npz
  analysis/packing_analysis/calibration_summary.json
  analysis/packing_analysis/feature_distributions.png
"""

from __future__ import annotations

from pathlib import Path

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_SEGMENTS_PT = Path("processed_cache/segments_f1c082560ae42021.pt")
DEFAULT_OUTPUT_DIR = Path("analysis/packing_analysis")

# ── Calibration settings ──────────────────────────────────────────────────────

# Number of structures to sample from the segments file
DEFAULT_N_STRUCTURES: int = 500

# DSM sigma range — MUST match --sigma-min / --sigma-max used in training.
# sig_tau = exp(mean(log σ_min, log σ_max)) × √2
DEFAULT_SIGMA_MIN: float = 0.05  # Å
DEFAULT_SIGMA_MAX: float = 3.0  # Å

# ── Geometry feature settings (must match _GeometryFeatures defaults) ─────────
SHELL_CUTOFFS = (6.0, 8.0, 10.0)  # Å — tight / medium / loose
SHORT_GATE_ON = 4.5  # Å
SHORT_GATE_OFF = 5.0  # Å
R_ON = 8.0  # Å — long-range switch start
R_CUT = 10.0  # Å — long-range switch end
MAX_DIST = 10.0  # Å — distance clamp
TOPK = 64  # k-nearest neighbours
EXCLUDE = 3  # exclude |i-j| <= EXCLUDE

# Shell half-width used for per-cutoff distance-spread diagnostics
SHELL_HALF_WIDTH: float = 1.5  # Å

# ── Plotting ──────────────────────────────────────────────────────────────────
PLOT_DPI: int = 150
