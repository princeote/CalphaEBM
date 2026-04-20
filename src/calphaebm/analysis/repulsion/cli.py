"""CLI interface for repulsion + packing enrichment analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PDB_LIST, PLOT_MAX_POINTS
from .core import run_repulsion_analysis


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "repulsion",
        description="Compute RDF/PMF repulsive wall + packing enrichment matrices",
        help="Repulsion + packing enrichment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--pdb-list", type=Path, default=DEFAULT_PDB_LIST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--max-pdbs", type=int, default=None, help="Maximum number of PDBs to process")
    parser.add_argument("--max-chains", type=int, default=None, help="Maximum chains per PDB")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bars")
    parser.add_argument("--plot-max-points", type=int, default=PLOT_MAX_POINTS, help="Maximum points for plotting")

    parser.add_argument("--no-enrichment", action="store_true", help="Skip OE/PMI enrichment computation")

    parser.set_defaults(func=run_repulsion_analysis)
    return parser
