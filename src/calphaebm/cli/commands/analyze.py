# src/calphaebm/cli/commands/analyze.py

"""Analysis commands for CalphaEBM."""

from __future__ import annotations

import argparse

from calphaebm.utils.logging import get_logger

logger = get_logger()


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add analyze command parser."""
    parser = subparsers.add_parser(
        "analyze",
        description="Run analysis on PDB data",
        help="Analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Create subparsers for different analysis types
    analyze_subparsers = parser.add_subparsers(
        dest="analysis_type",
        title="Analysis type",
        description="Valid analysis types",
        required=True,
    )

    # Lazy imports - only load when actually needed
    try:
        from calphaebm.analysis.repulsion import cli as repulsion_cli

        repulsion_cli.add_subparser(analyze_subparsers)
    except ImportError as e:
        logger.warning(f"Repulsion analysis module not available: {e}")

    try:
        from calphaebm.analysis.backbone import cli as backbone_cli

        backbone_cli.add_subparser(analyze_subparsers)
    except ImportError as e:
        logger.warning(f"Backbone analysis module not available: {e}")

    # Basins analysis (mixture-of-basins Ramachandran)
    try:
        from calphaebm.analysis.basins import cli as basins_cli

        basins_cli.add_subparser(analyze_subparsers)
    except ImportError as e:
        logger.warning(f"Basins analysis module not available: {e}")

    # Packing analysis (packing geometry feature calibration)
    try:
        from calphaebm.analysis.packing import packing_cli

        packing_cli.add_subparser(analyze_subparsers)
    except ImportError as e:
        logger.warning(f"Packing analysis module not available: {e}")

    # B-factor calibration (Langevin RMSF vs experimental B-factors)
    try:
        from calphaebm.analysis.bfactor import cli as bfactor_cli

        bfactor_cli.add_subparser(analyze_subparsers)
    except ImportError as e:
        logger.warning(f"B-factor analysis module not available: {e}")

    # Coordination analysis (per-AA coordination statistics for packing energy)
    try:
        from calphaebm.analysis.coordination import cli as coordination_cli

        coordination_cli.add_subparser(analyze_subparsers)
    except ImportError as e:
        logger.warning(f"Coordination analysis module not available: {e}")

    # H-bond distance analysis (helix/sheet distance distributions)
    try:
        from calphaebm.analysis.hbonds import cli as hbonds_cli

        hbonds_cli.add_subparser(analyze_subparsers)
    except ImportError as e:
        logger.warning(f"H-bond analysis module not available: {e}")

    return parser
