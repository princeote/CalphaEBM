# src/calphaebm/evaluation/trajectory_evaluation.py
"""Trajectory evaluation — analyse a pre-generated Langevin trajectory.

Entry point:
    run_traj(args)   called by evaluate.py --mode traj

Thin wrapper around TrajectoryEvaluator (reporting.py) and plot_all (plotting.py).
No model or checkpoint loading — operates entirely on saved coordinate snapshots.
"""

from __future__ import annotations

from pathlib import Path

from calphaebm.utils.logging import get_logger

logger = get_logger()


def run_traj(args) -> int:
    """Run trajectory evaluation (--mode traj).

    Reads a directory of trajectory snapshots, computes RMSD, dRMSD, Q,
    Rg, RDF, clash diagnostics, and optionally generates plots.
    """
    from calphaebm.evaluation.reporting import TrajectoryEvaluator

    if not args.traj:
        logger.error("--traj is required for --mode traj")
        return 1

    # Reference structure: prefer explicit args, fall back to snapshot_0000.pt
    if getattr(args, "ref_xyz", None):
        ref_path = args.ref_xyz
    elif getattr(args, "ref_pt", None):
        ref_path = args.ref_pt
    else:
        ref_path = str(Path(args.traj) / "snapshot_0000.pt")

    evaluator = TrajectoryEvaluator(
        contact_cutoff=args.contact_cutoff,
        exclude=args.exclude,
        rdf_rmax=args.rdf_rmax,
        rdf_dr=args.rdf_dr,
        q_smooth_beta=args.q_smooth_beta,
        q_smooth_lambda=args.q_smooth_lambda,
        clash_threshold=args.clash_threshold,
    )

    logger.info("Evaluating trajectory: %s", args.traj)
    logger.info("Reference:             %s", ref_path)

    report = evaluator.evaluate_from_dir(
        args.traj,
        ref_path=ref_path,
        burnin_steps=args.burnin,
    )

    print("\n" + report.summary())

    out_dir = Path(args.traj) / "eval"
    report.save(out_dir, burnin_steps=args.burnin)
    logger.info("Saved results to %s", out_dir)

    if not getattr(args, "no_plots", False):
        from calphaebm.evaluation.plotting import plot_all

        logger.info("Generating plots...")
        plot_all(report, out_dir, burnin_steps=args.burnin)
        logger.info("Saved plots to %s", out_dir)

    return 0
