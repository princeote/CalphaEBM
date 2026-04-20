"""Main CLI entry point."""

import argparse
import sys

from calphaebm import __version__
from calphaebm.utils.logging import setup_logger

logger = setup_logger("calphaebm.cli")


def main(args=None):
    """Main CLI entry point."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="CalphaEBM: Cα Energy-Based Model for Protein Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"CalphaEBM {__version__}",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Valid commands",
        help="Additional help",
    )

    # Import command modules
    from calphaebm.cli.commands.analyze import add_parser as add_analyze_parser  # NEW
    from calphaebm.cli.commands.balance import add_parser as add_balance_parser
    from calphaebm.cli.commands.build_dataset import add_parser as add_build_dataset_parser
    from calphaebm.cli.commands.calibrate import add_geometry_parser as add_calibrate_geometry_parser
    from calphaebm.cli.commands.calibrate import add_parser as add_calibrate_parser
    from calphaebm.cli.commands.curate_dataset import add_parser as add_curate_dataset_parser
    from calphaebm.cli.commands.evaluate import add_parser as add_evaluate_parser
    from calphaebm.cli.commands.simulate import add_parser as add_simulate_parser
    from calphaebm.cli.commands.train import add_parser as add_train_parser
    from calphaebm.cli.commands.validate import add_parser as add_validate_parser

    add_train_parser(subparsers)
    add_simulate_parser(subparsers)
    add_balance_parser(subparsers)
    add_evaluate_parser(subparsers)
    add_build_dataset_parser(subparsers)
    add_curate_dataset_parser(subparsers)
    add_analyze_parser(subparsers)  # NEW
    add_calibrate_parser(subparsers)
    add_calibrate_geometry_parser(subparsers)
    add_validate_parser(subparsers)

    # Parse arguments
    parsed = parser.parse_args(args)

    # Set logging level
    if parsed.verbose:
        setup_logger(level="DEBUG")

    # Run command
    if parsed.command == "train":
        from calphaebm.cli.commands.train import run

        return run(parsed)
    elif parsed.command == "simulate":
        from calphaebm.cli.commands.simulate import run

        return run(parsed)
    elif parsed.command == "balance":
        from calphaebm.cli.commands.balance import run

        return run(parsed)
    elif parsed.command == "evaluate":
        from calphaebm.cli.commands.evaluate import run

        return run(parsed)
    elif parsed.command == "build-dataset":
        from calphaebm.cli.commands.build_dataset import run

        return run(parsed)
    elif parsed.command == "curate-dataset":
        return parsed.func(parsed)
    elif parsed.command == "analyze":  # NEW
        return parsed.func(parsed)
    elif parsed.command == "calibrate":
        return parsed.func(parsed)
    elif parsed.command == "validate":
        from calphaebm.cli.commands.validate import run

        return run(parsed)
    elif parsed.command == "calibrate-geometry":
        return parsed.func(parsed)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
