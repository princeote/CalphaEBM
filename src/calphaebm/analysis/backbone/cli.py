# src/calphaebm/analysis/backbone/cli.py

"""CLI interface for backbone analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from calphaebm.utils.logging import get_logger

from .config import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PDB_LIST
from .core import run_backbone_analysis

logger = get_logger()


def add_subparser(subparsers):
    """Add backbone subcommand parser."""
    parser = subparsers.add_parser(
        "backbone",
        description="Analyze backbone geometry (bond lengths, θ/φ distributions, correlations, Δφ potential)",
        help="Backbone geometry analysis",
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
        "--max-chains",
        type=int,
        default=None,
        help="Maximum number of chains to process across all PDBs (default: all)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plot images (still writes .npy data products)",
    )

    parser.set_defaults(func=run_backbone_analysis)
    return parser
