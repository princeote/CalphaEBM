"""CLI interface for packing geometry feature calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from .packing_config import (
    DEFAULT_N_STRUCTURES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEGMENTS_PT,
    DEFAULT_SIGMA_MAX,
    DEFAULT_SIGMA_MIN,
)
from .packing_core import run_packing_analysis


def add_subparser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "packing",
        description=(
            "Calibrate packing geometry feature parameters.\n\n"
            "Computes sig_tau and normalisation denominators for the\n"
            "_GeometryFeatures module used by the packing MLP, from a\n"
            "processed segments file.\n\n"
            "  sig_tau = exp(mean(log σ_min, log σ_max)) × √2\n"
            "            maximises expected sigmoid gradient under log-uniform DSM\n\n"
            "  norm_*  = per-feature mean / std over training structures\n\n"
            "Outputs:\n"
            "  analysis/packing_analysis/data/geometry_feature_calibration.json\n"
            "      → pass to training: --packing-geom-calibration <path>\n"
            "  analysis/packing_analysis/feature_distributions.png\n"
            "  analysis/packing_analysis/calibration_summary.json\n"
            "  analysis/packing_analysis/data/feature_stats.npz"
        ),
        help="Calibrate packing geometry features (sig_tau, normalisation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--segments",
        type=str,
        default=str(DEFAULT_SEGMENTS_PT),
        help=(
            "Path to processed segments .pt file "
            f"(default: {DEFAULT_SEGMENTS_PT}). "
            "Generated automatically on first 'calphaebm train' run."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for plots and JSON (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--n-structures",
        type=int,
        default=DEFAULT_N_STRUCTURES,
        help=(
            f"Structures to sample from the segments file "
            f"(default: {DEFAULT_N_STRUCTURES}). "
            "200–500 gives stable normalisation statistics."
        ),
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=DEFAULT_SIGMA_MIN,
        help=(
            f"Lower DSM sigma bound in Å (default: {DEFAULT_SIGMA_MIN}). " "Must match --sigma-min used in training."
        ),
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=DEFAULT_SIGMA_MAX,
        help=(
            f"Upper DSM sigma bound in Å (default: {DEFAULT_SIGMA_MAX}). " "Must match --sigma-max used in training."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    parser.set_defaults(func=run_packing_analysis)
    return parser
