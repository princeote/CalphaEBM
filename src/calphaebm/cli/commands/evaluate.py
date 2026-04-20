# src/calphaebm/cli/commands/evaluate.py
"""Evaluation CLI — thin dispatcher to the evaluation module.

Three modes:
  calphaebm evaluate --mode traj   --traj trajectory_dir/
  calphaebm evaluate --mode basin  --checkpoint path/to/ckpt.pt --pdb val_hq.txt
  calphaebm evaluate --mode watch  --ckpt-dir checkpoints/run6  --pdb val_hq.txt

Shared eval flags (basin + watch):
  --beta 100.0       inverse temperature (basin: list for sweep, watch: single value)
  --n-steps 10000    MALA steps per structure
  --n-samples 64     structures to evaluate
  --sampler mala     sampling algorithm

Implementation lives in:
  calphaebm.evaluation.basin_evaluation      — basin mode + shared model/data utils
  calphaebm.evaluation.trajectory_evaluation — traj mode
  calphaebm.evaluation.training_evaluation   — watch mode + FES
"""

import argparse

from calphaebm.utils.logging import get_logger

logger = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "evaluate",
        description="Evaluate a trajectory, run basin stability, or watch training",
        help="Evaluate trajectory / basin stability / training watcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["traj", "basin", "watch"],
        default="traj",
        help=(
            "'traj'  — analyse a pre-generated trajectory  "
            "'basin' — single-checkpoint basin sweep  "
            "'watch' — detached training watcher (polls for round checkpoints)"
        ),
    )

    # ── Trajectory mode (--mode traj) ────────────────────────────────
    parser.add_argument("--traj", help="Trajectory directory (required for --mode traj)")
    parser.add_argument("--ref-pt", help="Reference .pt file (default: snapshot_0000.pt)")
    parser.add_argument("--ref-xyz", help="Reference .xyz file (alternative to --ref-pt)")
    parser.add_argument("--burnin", type=int, default=0, help="Frames to discard as burn-in (default: 0)")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    # ── Basin mode (--mode basin) ─────────────────────────────────────
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint (required for --mode basin)")
    parser.add_argument(
        "--pdb",
        nargs="+",
        default=None,
        help=("PDB IDs, file of IDs, or val-list file. " "Basin: ID list or file. Watch: file path (e.g. val_hq.txt)."),
    )
    parser.add_argument(
        "--minimize-steps", type=int, default=0, help="L-BFGS minimization steps before Langevin (default: 0)"
    )
    parser.add_argument(
        "--perturb-sigma",
        type=float,
        default=0.0,
        help="IC perturbation σ before Langevin (default: 0). "
        "Non-zero: recovery test from perturbed start (try 0.3).",
    )
    parser.add_argument(
        "--save-every", type=int, default=50, help="Save trajectory snapshot every N steps (default: 50)"
    )
    parser.add_argument("--log-every", type=int, default=1000, help="Log progress every N steps (default: 1000)")
    parser.add_argument("--step-size", type=float, default=1e-4, help="Langevin step size (default: 1e-4)")
    parser.add_argument("--force-cap", type=float, default=100.0, help="Force clipping threshold (default: 100.0)")
    parser.add_argument("--out-dir", default=None, help="Output directory for basin results (default: basin_results/)")
    parser.add_argument("--min-len", type=int, default=40, help="Minimum chain length (default: 40)")
    parser.add_argument("--max-len", type=int, default=512, help="Maximum chain length (default: 512)")
    parser.add_argument(
        "--no-early-stop", action="store_true", help="Always run all Langevin steps (no early stopping)"
    )
    parser.add_argument("--cache-dir", default="./pdb_cache")
    parser.add_argument("--processed-cache-dir", default="./processed_cache")

    # ── Watch mode (--mode watch) ─────────────────────────────────────
    parser.add_argument(
        "--ckpt-dir", default=None, help="Checkpoint root (required for --mode watch, " "e.g. checkpoints/run6)"
    )
    parser.add_argument("--ckpt-prefix", default=None, help="Checkpoint prefix (required for --mode watch, e.g. run6)")
    parser.add_argument("--max-rounds", type=int, default=10, help="Stop after this many rounds (default: 10)")
    parser.add_argument("--start-round", type=int, default=1, help="First round to watch (default: 1)")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between checkpoint polls (default: 60)")

    # ── Shared eval flags (basin + watch) ────────────────────────────
    parser.add_argument(
        "--beta",
        type=float,
        nargs="+",
        default=[100.0],
        help="Inverse temperature(s). " "Basin: sweep over list. Watch: uses first value. " "(default: 100.0)",
    )
    parser.add_argument("--n-steps", type=int, default=10000, help="MALA/Langevin steps per structure (default: 10000)")
    parser.add_argument("--n-samples", type=int, default=64, help="Structures to evaluate per round (default: 64)")
    parser.add_argument(
        "--sampler", default="mala", choices=["mala", "langevin"], help="Sampling algorithm (default: mala)"
    )

    # ── Shared metric flags (traj + basin) ───────────────────────────
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--exclude", type=int, default=2)
    parser.add_argument("--rdf-rmax", type=float, default=20.0)
    parser.add_argument("--rdf-dr", type=float, default=0.25)
    parser.add_argument("--q-smooth-beta", type=float, default=5.0)
    parser.add_argument("--q-smooth-lambda", type=float, default=1.8)
    parser.add_argument("--clash-threshold", type=float, default=3.8)

    parser.set_defaults(func=run)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher — all logic lives in the evaluation module
# ─────────────────────────────────────────────────────────────────────────────


def run(args):
    if args.mode == "watch":
        from calphaebm.evaluation.training_evaluation import run_watch

        return run_watch(args)
    elif args.mode == "basin":
        from calphaebm.evaluation.basin_evaluation import run_basin

        return run_basin(args)
    else:
        from calphaebm.evaluation.trajectory_evaluation import run_traj

        return run_traj(args)
