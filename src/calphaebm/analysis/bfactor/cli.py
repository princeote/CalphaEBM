"""CLI for B-factor calibration analysis.

Registers as: calphaebm analyze bfactor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from calphaebm.utils.logging import get_logger

logger = get_logger()


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the bfactor subcommand under 'analyze'."""
    parser = subparsers.add_parser(
        "bfactor",
        help="B-factor calibration: compare Langevin RMSF to experimental B-factors",
        description=(
            "Run Langevin dynamics at multiple inverse temperatures (β), "
            "compute per-residue RMSF, and correlate with experimental "
            "B-factors from X-ray crystallography. The β with highest "
            "Pearson correlation defines the physical temperature scale."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--pdb",
        nargs="+",
        required=True,
        help=(
            "PDB IDs to analyze. Use high-resolution X-ray structures "
            "(resolution < 1.5Å) for reliable B-factors. "
            "Recommended: 1crn 1ubq 1pga 2ci2"
        ),
    )

    # β sweep
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[10, 20, 50, 100, 200],
        help="Inverse temperature values to test (default: 10 20 50 100 200)",
    )

    # Langevin params
    parser.add_argument(
        "--langevin-steps",
        type=int,
        default=2000,
        help="Langevin steps per structure per β (default: 2000)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1e-4,
        help="Langevin step size η (default: 1e-4)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save trajectory snapshot every N steps (default: 10)",
    )

    # I/O
    parser.add_argument(
        "--cache-dir",
        default="./pdb_cache",
        help="PDB file cache directory",
    )
    parser.add_argument(
        "--output",
        default="analysis/bfactor/bfactor_analysis.json",
        help="Output JSON path (default: analysis/bfactor/bfactor_analysis.json)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu or cuda (default: cpu)",
    )

    # Model data dirs (must match training config)
    parser.add_argument("--backbone-data-dir", default="analysis/backbone_geometry/data")
    parser.add_argument("--secondary-data-dir", default="analysis/secondary_analysis/data")
    parser.add_argument("--repulsion-data-dir", default="analysis/repulsion_analysis/data")
    parser.add_argument("--packing-data-dir", default="analysis/repulsion_analysis/data")
    parser.add_argument(
        "--packing-geom-calibration",
        default="analysis/packing_analysis/data/geometry_feature_calibration.json",
    )

    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    """Entry point for: calphaebm analyze bfactor"""

    from calphaebm.analysis.bfactor.core import run_bfactor_analysis
    from calphaebm.models.total_energy import TotalEnergy

    # Load model
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_state = ckpt.get("model_state_dict", ckpt)

    model = TotalEnergy(
        backbone_data_dir=args.backbone_data_dir,
        secondary_data_dir=args.secondary_data_dir,
        repulsion_data_dir=args.repulsion_data_dir,
        packing_data_dir=args.packing_data_dir,
        packing_geom_calibration=args.packing_geom_calibration,
    )

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    logger.info("Model: %d missing, %d unexpected keys", len(missing), len(unexpected))
    model.to(args.device)
    model.eval()

    # Run analysis
    run_bfactor_analysis(
        model=model,
        pdb_ids=args.pdb,
        betas=args.betas,
        n_steps=args.langevin_steps,
        step_size=args.step_size,
        save_every=args.save_every,
        cache_dir=args.cache_dir,
        output_path=args.output,
        device=args.device,
    )
