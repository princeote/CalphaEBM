# src/calphaebm/analysis/basins/cli.py

"""CLI interface for basin analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CIRCULAR_PHI,
    DEFAULT_CLUSTER_METHOD,
    DEFAULT_MAX_CHAINS,
    DEFAULT_MAX_PDBS,
    DEFAULT_N_BASINS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PDB_LIST,
    DEFAULT_PLOT_MAX_POINTS,
    DEFAULT_PSEUDOCOUNT,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_SMOOTH_SIGMA,
    DEFAULT_STANDARDIZE,
)
from .core import run_basin_analysis


def add_subparser(subparsers):
    """Add basin subcommand parser."""
    parser = subparsers.add_parser(
        "basins",
        description="Generate basin energy surfaces for mixture-of-basins Ramachandran",
        help="Basin analysis for Ramachandran mixture model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pdb-list",
        type=Path,
        default=DEFAULT_PDB_LIST,
        help=f"File containing PDB IDs (default: {DEFAULT_PDB_LIST})",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory for PDB cache (default: {DEFAULT_CACHE_DIR})",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--max-pdbs",
        type=int,
        default=DEFAULT_MAX_PDBS,
        help=f"Maximum number of PDB IDs to process (default: {DEFAULT_MAX_PDBS})",
    )

    parser.add_argument(
        "--max-chains",
        type=int,
        default=DEFAULT_MAX_CHAINS,
        help="Maximum number of chains to process across all PDBs (default: no limit)",
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=DEFAULT_SAMPLE_EVERY,
        help=f"Sample every N residues to reduce correlation (default: {DEFAULT_SAMPLE_EVERY})",
    )

    parser.add_argument(
        "--n-basins",
        type=int,
        default=DEFAULT_N_BASINS,
        help=f"Number of basins (K) (default: {DEFAULT_N_BASINS})",
    )

    parser.add_argument(
        "--cluster-method",
        type=str,
        default=DEFAULT_CLUSTER_METHOD,
        choices=["kmeans", "gmm"],
        help=f"Clustering method (default: {DEFAULT_CLUSTER_METHOD})",
    )

    parser.add_argument(
        "--circular-phi",
        action="store_true",
        default=DEFAULT_CIRCULAR_PHI,
        help="Cluster using circular phi features [sin(phi), cos(phi)] (recommended)",
    )

    parser.add_argument(
        "--no-circular-phi",
        action="store_false",
        dest="circular_phi",
        help="Cluster using raw phi values (not recommended; phi is circular)",
    )

    parser.add_argument(
        "--standardize",
        action="store_true",
        default=DEFAULT_STANDARDIZE,
        help="Standardize features before clustering (default: on)",
    )

    parser.add_argument(
        "--no-standardize",
        action="store_false",
        dest="standardize",
        help="Disable feature standardization",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE})",
    )

    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=DEFAULT_SMOOTH_SIGMA,
        help=f"Smoothing sigma for histograms in bins (default: {DEFAULT_SMOOTH_SIGMA})",
    )

    parser.add_argument(
        "--pseudocount",
        type=float,
        default=DEFAULT_PSEUDOCOUNT,
        help=f"Pseudocount for histogram stability (default: {DEFAULT_PSEUDOCOUNT})",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=DEFAULT_PLOT_MAX_POINTS,
        help=f"Max points to plot in scatter (default: {DEFAULT_PLOT_MAX_POINTS})",
    )

    parser.add_argument(
        "--force-reextract",
        action="store_true",
        default=False,
        help="Ignore cached (θ,φ) data and re-extract from PDB structures",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars / reduce verbosity",
    )

    parser.set_defaults(func=run_basin_analysis)
    return parser
