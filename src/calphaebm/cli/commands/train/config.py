"""Argument parsing for the train command.

This module does not import run() to avoid circular imports.
"""

import argparse

from calphaebm.defaults import MODEL as _M
from calphaebm.defaults import TRAIN as _T
from calphaebm.utils.constants import LEARNING_RATE

# Default IC DSM sigma range (radians).
# These are the noise levels for (θ, φ) angle space — not Å.
# Physically comparable to Cartesian sigma 0.05–0.5 Å.
IC_SIGMA_DEFAULT = 0.08  # fixed-sigma fallback (radians) — geometric midpoint of schedule range
IC_SIGMA_MIN_DEFAULT = 0.02  # log-uniform lower bound (radians)
IC_SIGMA_MAX_DEFAULT = 0.30  # log-uniform upper bound (radians)


def build_parser(subparsers):
    """Build train command parser."""
    parser = subparsers.add_parser(
        "train",
        description="Train energy model with phased IC training (run19+)",
        help="Train energy model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Stage-based training (new, preferred)
    parser.add_argument(
        "--stage",
        choices=["full", "sc"],
        default=None,
        help="Training stage: 'full' (PDB-only with rounds) or 'sc' (self-consistent). "
        "Replaces --phase. If set, --phase is ignored.",
    )
    # Legacy phase-based training
    parser.add_argument(
        "--phase",
        choices=["local", "secondary", "repulsion", "packing", "full", "self-consistent"],
        default=None,
        help="(Legacy) Training phase. Use --stage instead.",
    )
    # Stage-specific arguments (--stage full)
    parser.add_argument(
        "--max-rounds", type=int, default=_T["max_rounds"], help="Max training rounds (stage full, default: 8)"
    )
    parser.add_argument(
        "--steps-per-round", type=int, default=3000, help="Training steps per round (stage full, default: 3000)"
    )
    parser.add_argument(
        "--n-decoys",
        type=int,
        default=_T["n_decoys"],
        help="IC-noised decoys per protein per step (stage full, default: 8)",
    )
    parser.add_argument(
        "--disc-T", type=float, default=_T["disc_T"], help="Temperature for discrimination loss (default: 2.0)"
    )
    parser.add_argument(
        "--lambda-qf", type=float, default=_T["lambda_qf"], help="Weight for Q-funnel loss (stage full, default: 1.0)"
    )
    parser.add_argument(
        "--converge-q", type=float, default=_T["converge_q"], help="Convergence Q threshold (default: 0.95)"
    )
    parser.add_argument(
        "--converge-rmsd",
        type=float,
        default=_T["converge_rmsd"],
        help="Convergence RMSD threshold in A (default: 5.0)",
    )
    parser.add_argument(
        "--converge-rg-lo", type=float, default=_T["converge_rg_lo"], help="Convergence Rg%% lower bound (default: 95)"
    )
    parser.add_argument(
        "--converge-rg-hi", type=float, default=_T["converge_rg_hi"], help="Convergence Rg%% upper bound (default: 105)"
    )
    parser.add_argument(
        "--sigma-min-ic", type=float, default=0.0524, help="Min IC noise sigma in radians (stage full, default: 0.05)"
    )
    parser.add_argument(
        "--sigma-max-ic", type=float, default=1.0472, help="Max IC noise sigma in radians (stage full, default: 1.5)"
    )
    parser.add_argument("--decoy-every", type=int, default=1, help="Compute decoy losses every N steps (default: 1)")
    parser.add_argument(
        "--val-proteins", type=int, default=_T["eval_proteins"], help="Number of proteins for basin eval (default: 64)"
    )
    parser.add_argument("--val-steps", type=int, default=5000, help="Langevin steps for basin eval (default: 5000)")
    parser.add_argument(
        "--val-beta", type=float, default=_T["eval_beta"], help="Inverse temperature for basin eval (default: 100)"
    )
    parser.add_argument(
        "--T-funnel", type=float, default=_T["T_funnel"], help="Temperature for funnel losses (default: 2.0)"
    )

    parser.add_argument(
        "--train-pdb",
        type=str,
        default=None,
        help="File with training PDB IDs (one per line). If set with --val-pdb, "
        "overrides --pdb and uses explicit train/val split.",
    )
    parser.add_argument(
        "--val-pdb",
        type=str,
        default=None,
        help="File with validation PDB IDs (one per line). Must be used with --train-pdb.",
    )
    parser.add_argument(
        "--pdb",
        nargs="+",
        default=None,
        help="PDB IDs or file containing IDs (one per line)",
    )

    # Data options
    parser.add_argument("--cache-dir", default="./pdb_cache", help="Directory for PDB cache")
    parser.add_argument(
        "--processed-cache-dir",
        type=str,
        default="./processed_cache",
        help="Directory to cache processed chains/segments. "
        "Avoids re-parsing CIF files on every startup. "
        "Used by both PDBChainDataset (--stage full/sc) and "
        "PDBSegmentDataset (legacy phases). Default: ./processed_cache",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of processed chains/segments")
    parser.add_argument("--force-reprocess", action="store_true", help="Force re-parsing of PDBs even if cache exists")
    parser.add_argument("--seg-len", type=int, default=128, help="Segment length")
    parser.add_argument("--stride", type=int, default=64, help="Segment stride")
    parser.add_argument("--limit", type=int, default=100000, help="Maximum number of segments to load")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Training options
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["linear", "cosine", "exponential", "none"],
        default="none",
        help="Learning rate schedule",
    )
    parser.add_argument("--lr-final", type=float, default=None, help="Final learning rate for scheduling")
    parser.add_argument(
        "--scalar-lr-mult",
        type=float,
        default=20.0,
        help="LR multiplier for scalar lambda params relative to MLP (default: 20x)",
    )
    parser.add_argument(
        "--freeze-packing-scalar",
        action="store_true",
        default=False,
        help="Hold λ_pack fixed at lr=0; only MLP weights train",
    )

    # ============================================================
    # IC DSM SIGMA OPTIONS  (units: radians)
    # ============================================================
    # NOTE: sigma values are in radians (angle space), NOT Angstroms.
    # Comparable physical effect: 0.05 rad ≈ 0.05 Å at Cα–Cα bond length.
    # Recommended multi-scale range: --sigma-min-rad 0.02 --sigma-max-rad 0.30
    parser.add_argument(
        "--sigma-rad",
        type=float,
        default=IC_SIGMA_DEFAULT,
        help=(
            f"IC DSM noise sigma in RADIANS (default: {IC_SIGMA_DEFAULT}). "
            "Ignored when --sigma-min-rad and --sigma-max-rad are both set."
        ),
    )
    parser.add_argument(
        "--sigma-min-rad",
        type=float,
        default=IC_SIGMA_MIN_DEFAULT,
        help=(
            "Minimum sigma for multi-scale IC DSM (radians). "
            "When set with --sigma-max-rad, each sample draws sigma log-uniformly "
            "from [sigma-min-rad, sigma-max-rad], overriding --sigma-rad. "
            "Recommended: 0.02."
        ),
    )
    parser.add_argument(
        "--sigma-max-rad",
        type=float,
        default=IC_SIGMA_MAX_DEFAULT,
        help="Maximum sigma for multi-scale IC DSM (radians). See --sigma-min-rad. Recommended: 0.30.",
    )

    # --- Differential sigma: per-coordinate noise ranges ---
    # When all four are set, θ and φ get different noise scales calibrated to
    # their natural variance (θ std ≈ 0.295 rad, φ std ≈ 1.834 rad).
    # A shared noise level t ~ U(0,1) maps to coordinate-specific σ via
    # log-uniform interpolation, keeping both at the same relative intensity.
    parser.add_argument(
        "--sigma-min-theta",
        type=float,
        default=0.02,
        help=("Min sigma for θ (bond angle) noise (radians). " "θ std ≈ 0.295 rad in PDB data. Default: 0.02."),
    )
    parser.add_argument(
        "--sigma-max-theta",
        type=float,
        default=0.5,
        help=("Max sigma for θ (bond angle) noise (radians). " "0.5 rad ≈ 1.7× natural variation. Default: 0.5."),
    )
    parser.add_argument(
        "--sigma-min-phi",
        type=float,
        default=0.05,
        help=("Min sigma for φ (dihedral) noise (radians). " "φ std ≈ 1.83 rad in PDB data. Default: 0.05."),
    )
    parser.add_argument(
        "--sigma-max-phi",
        type=float,
        default=1.5,
        help=(
            "Max sigma for φ (dihedral) noise (radians). " "1.5 rad covers most of the torsion landscape. Default: 1.5."
        ),
    )

    # --- Alpha augmentation: bidirectional Rg perturbation for DSM (run53+) ---
    # Fixes the coverage gap where IC noise is asymmetric in Rg space.
    # When enabled, each DSM step computes 3 samples:
    #   1. Standard: x̃ → x_native (existing behavior)
    #   2. Scaled:   x_α → x_native (pure Rg scaling)
    #   3. Mixed:    x̃_α → x_native (scaling + IC noise)
    # Native is ALWAYS the attractor. 3× compute cost.
    parser.add_argument(
        "--dsm-alpha-min",
        type=float,
        default=0.65,
        help=(
            "Min α for DSM Rg scaling augmentation. "
            "α < 1 = compacted, α > 1 = swollen. "
            "Set both to 1.0 to disable augmentation. Default: 0.65."
        ),
    )
    parser.add_argument(
        "--dsm-alpha-max",
        type=float,
        default=1.25,
        help=("Max α for DSM Rg scaling augmentation. " "Set both to 1.0 to disable augmentation. Default: 1.25."),
    )

    # Validation options
    parser.add_argument("--validate-every", type=int, default=500, help="Run validation every N steps")
    parser.add_argument(
        "--val-max-samples", type=int, default=256, help="Max structures during training validation (default: 256)"
    )
    parser.add_argument(
        "--val-langevin-steps",
        type=int,
        default=500,
        help="IC Langevin steps per structure during validation (default: 500)",
    )
    parser.add_argument(
        "--val-step-size", type=float, default=None, help="IC Langevin step size for validation (default: 1e-4)"
    )
    parser.add_argument(
        "--langevin-beta",
        type=float,
        default=1.0,
        help="Inverse temperature β for validation Langevin dynamics. "
        "Higher β = lower temperature = less noise. "
        "β=1 is standard; β=5 reduces noise by √5≈2.24×.",
    )
    parser.add_argument("--early-stopping", type=int, default=None, help="Early stopping patience")

    # Checkpoint options
    parser.add_argument("--ckpt-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--ckpt-prefix", default="run1", help="Experiment prefix")
    parser.add_argument("--ckpt-every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from, or 'auto' for latest")
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help="Load only model weights, not optimizer/scheduler state. Gates are reset to 1.0.",
    )

    # Gate freezing for full phase
    parser.add_argument(
        "--freeze-gates-steps",
        type=int,
        default=0,
        help="Steps to freeze gates at start of full phase (default: 0 = no freezing)",
    )

    # ============================================================
    # GATE RAMPING OPTIONS (full phase)
    # ============================================================
    parser.add_argument("--ramp-gates", action="store_true", help="Ramp gates linearly during full phase")
    parser.add_argument("--ramp-steps", type=int, default=5000, help="Steps to ramp gates over")
    parser.add_argument("--ramp-local-start", type=float, default=1.0)
    parser.add_argument("--ramp-local-end", type=float, default=1.0)
    parser.add_argument("--ramp-rep-start", type=float, default=1.0)
    parser.add_argument("--ramp-rep-end", type=float, default=20.0)
    parser.add_argument("--ramp-ss-start", type=float, default=1.0)
    parser.add_argument("--ramp-ss-end", type=float, default=8.0)
    parser.add_argument("--ramp-pack-start", type=float, default=0.5)
    parser.add_argument("--ramp-pack-end", type=float, default=4.0)

    # Energy terms
    parser.add_argument(
        "--energy-terms",
        nargs="*",
        default=["local"],
        choices=["local", "repulsion", "secondary", "packing", "all"],
        help="Energy terms to include",
    )

    # Freeze
    parser.add_argument(
        "--freeze",
        nargs="*",
        default=[],
        choices=["local", "repulsion", "secondary", "packing"],
        help="Terms to freeze",
    )

    # ============================================================
    # LOCAL TERM OPTIONS
    # bond_spring is REMOVED — bonds are fixed at 3.8Å by NeRF in IC training.
    # Only the learned angle sub-terms remain.
    # ============================================================
    parser.add_argument(
        "--init-theta-theta-weight",
        type=float,
        default=1.0,
        help="Initial weight for theta-theta coupling (default: 1.0)",
    )
    parser.add_argument(
        "--init-delta-phi-weight", type=float, default=1.0, help="Initial weight for delta-phi potential (default: 1.0)"
    )

    # ============================================================
    # GATE OVERRIDE OPTIONS
    # Applied AFTER checkpoint load.
    # ============================================================
    parser.add_argument(
        "--set-gate-local", type=float, default=None, help="Override g_local gate after checkpoint load"
    )
    parser.add_argument(
        "--set-gate-secondary", type=float, default=None, help="Override g_secondary gate after checkpoint load"
    )
    parser.add_argument(
        "--set-gate-repulsion", type=float, default=None, help="Override g_repulsion gate after checkpoint load"
    )
    parser.add_argument(
        "--set-gate-packing", type=float, default=None, help="Override g_packing gate after checkpoint load"
    )

    # ============================================================
    # REPULSION OPTIONS
    # ============================================================
    parser.add_argument("--repulsion-data-dir", type=str, default="analysis/repulsion_analysis/data")
    parser.add_argument("--repulsion-K", type=int, default=64)
    parser.add_argument("--repulsion-exclude", type=int, default=3)
    parser.add_argument("--repulsion-r-on", type=float, default=8.0)
    parser.add_argument("--repulsion-r-cut", type=float, default=10.0)

    # ============================================================
    # PACKING OPTIONS
    # ============================================================
    parser.add_argument("--packing-data-dir", type=str, default="analysis/repulsion_analysis/data")
    parser.add_argument("--packing-r-on", type=float, default=8.0)
    parser.add_argument("--packing-r-cut", type=float, default=10.0)
    parser.add_argument(
        "--packing-init-from", type=str, default="log_oe", choices=["log_oe", "pmi", "pmi_signed", "zeros"]
    )
    parser.set_defaults(packing_normalize_by_length=True)
    parser.add_argument("--no-packing-normalize", action="store_false", dest="packing_normalize_by_length")
    parser.add_argument("--packing-short-gate-on", type=float, default=4.5)
    parser.add_argument("--packing-short-gate-off", type=float, default=5.0)
    parser.add_argument("--packing-rbf-centers", type=float, nargs=3, default=[5.5, 7.0, 9.0])
    parser.add_argument("--packing-rbf-width", type=float, default=1.0)
    parser.add_argument("--packing-max-dist", type=float, default=_M["max_dist"])
    parser.add_argument("--packing-debug-scale", action="store_true")
    parser.add_argument("--packing-debug-every", type=int, default=200)
    parser.add_argument(
        "--packing-geom-calibration",
        type=str,
        default="analysis/packing_analysis/data/geometry_feature_calibration.json",
        help="Path to geometry_feature_calibration.json",
    )
    parser.add_argument("--packing-pretrain", action="store_true", default=False)
    parser.add_argument("--packing-logoe-data-dir", type=str, default=None)
    parser.add_argument("--packing-logoe-scale", type=float, default=5.0)
    parser.add_argument(
        "--packing-rg-lambda",
        type=float,
        default=_M["rg_lambda"],
        help="Rg Flory size restraint strength inside packing energy. "
        "E_rg = lambda * (Rg / Rg_expected - 1)^2. "
        "Part of E_theta (active at training AND inference). Default 5.0.",
    )
    parser.add_argument(
        "--packing-rg-r0",
        type=float,
        default=_M["rg_r0"],
        help="Flory prefactor: Rg_expected = r0 * L^nu. Default 2.0 Å.",
    )
    parser.add_argument(
        "--packing-rg-nu", type=float, default=_M["rg_nu"], help="Flory exponent for globular proteins. Default 0.38."
    )
    parser.add_argument(
        "--hp-penalty-lambda",
        type=float,
        default=_M["coord_lambda"],
        help="Coordination penalty coefficient (0=disabled). "
        "E_hp_pen = λ · (1/L) · Σ_i penalty(n_i). "
        "Default 0.1.",
    )
    parser.add_argument(
        "--coord-n-star-file",
        type=str,
        default="coord_n_star.json",
        help="JSON file with per-AA optimal coordination counts. "
        "Generated by 'calphaebm analyze coordination --best-style'. "
        "Default: coord_n_star.json",
    )

    # Run5: exponential constraint params
    parser.add_argument(
        "--packing-rg-dead-zone",
        type=float,
        default=_M["rg_dead_zone"],
        help="Rg penalty dead zone: no penalty within ±δ of Rg*. Default 0.30 (±30%%).",
    )
    parser.add_argument(
        "--packing-rg-m", type=float, default=_M["rg_m"], help="Rg penalty saturation level. Default 1.0."
    )
    parser.add_argument(
        "--packing-rg-alpha", type=float, default=_M["rg_alpha"], help="Rg penalty steepness. Default 3.0."
    )
    parser.add_argument(
        "--packing-coord-m",
        type=float,
        default=_M["coord_m"],
        help="Coordination penalty saturation level. Default 1.0.",
    )
    parser.add_argument(
        "--packing-coord-alpha",
        type=float,
        default=_M["coord_alpha"],
        help="Coordination penalty steepness. Default 2.0.",
    )
    # Run5: contact density (ρ) energy
    parser.add_argument(
        "--packing-rho-lambda",
        type=float,
        default=_M["rho_lambda"],
        help="Contact density ρ reward scale. Default 0.1.",
    )
    parser.add_argument(
        "--packing-rho-sigma",
        type=float,
        default=_M["rho_sigma"],
        help="Contact density ρ Gaussian width. Default 0.7.",
    )
    parser.add_argument(
        "--packing-rho-m", type=float, default=_M["rho_m"], help="Contact density ρ penalty saturation. Default 1.0."
    )
    parser.add_argument(
        "--packing-rho-alpha",
        type=float,
        default=_M["rho_alpha"],
        help="Contact density ρ penalty steepness. Default 2.0.",
    )
    parser.add_argument(
        "--rho-penalty-lambda",
        type=float,
        default=_M["rho_penalty_lambda"],
        help="Contact density ρ penalty scale. Default 0.1.",
    )
    # Run5: saturating exponential margins
    parser.add_argument(
        "--funnel-m", type=float, default=_T["funnel_m"], help="Q/Rg funnel margin saturation level. Default 5.0."
    )
    parser.add_argument(
        "--funnel-alpha", type=float, default=_T["funnel_alpha"], help="Q/Rg funnel margin steepness. Default 5.0."
    )
    parser.add_argument(
        "--gap-m", type=float, default=_T["gap_m"], help="Gap loss margin saturation level. Default 5.0."
    )
    parser.add_argument(
        "--gap-alpha", type=float, default=_T["gap_alpha"], help="Gap loss margin steepness. Default 5.0."
    )
    parser.add_argument(
        "--lambda-drmsd",
        type=float,
        default=_T["lambda_drmsd"],
        help="dRMSD-funnel loss coefficient. Pairwise ordering: "
        "lower full-pairwise dRMSD → lower energy. "
        "Topology-sensitive — catches strand-swaps and register "
        "shifts that satisfy the Rg constraint. Default 2.0.",
    )
    parser.add_argument(
        "--lambda-rg",
        type=float,
        default=None,
        help="DEPRECATED — alias for --lambda-drmsd for backward "
        "compat with existing shell scripts. Overrides "
        "--lambda-drmsd when set.",
    )
    parser.add_argument(
        "--rg-alpha-min",
        type=float,
        default=0.75,
        help="Lower bound of Rg scaling factor (< 1 = compaction). Default 0.75.",
    )
    parser.add_argument(
        "--rg-alpha-max",
        type=float,
        default=1.25,
        help="Upper bound of Rg scaling factor (> 1 = swelling). Default 1.25.",
    )

    # ============================================================
    # IC FORCE BALANCE LOSS OPTIONS (full phase only)
    # Perturbation is in (θ, φ) space — bonds always 3.8Å.
    # ============================================================
    parser.add_argument(
        "--lambda-fb",
        type=float,
        default=0.0,
        help="Weight for IC force balance loss (full phase only). 0=disabled. Recommended: 0.1.",
    )
    parser.add_argument(
        "--fb-sigma-theta",
        type=float,
        default=0.15,
        help="Std for θ Gaussian noise in IC force balance perturbation (radians). Default: 0.15.",
    )
    parser.add_argument(
        "--fb-sigma-phi",
        type=float,
        default=0.3,
        help="Std for φ Gaussian noise in IC force balance perturbation (radians). Default: 0.3.",
    )
    parser.add_argument(
        "--fb-clash-phi-frac",
        type=float,
        default=0.1,
        help="Fraction of residues receiving large φ perturbations (dihedral clash). Default: 0.1.",
    )
    parser.add_argument(
        "--fb-clash-phi-sigma",
        type=float,
        default=1.0,
        help="Std of large φ clash perturbations (radians). Default: 1.0.",
    )
    parser.add_argument("--fb-target-ss-ratio", type=float, default=2.0)
    parser.add_argument("--fb-target-pack-ratio", type=float, default=2.0)
    parser.add_argument("--fb-target-rep-ratio", type=float, default=2.0)
    parser.add_argument("--fb-diag-every", type=int, default=200)

    # ============================================================
    # IC LOCAL GEOMETRY GAP LOSS OPTIONS (full phase only)
    # Perturbation is in (θ, φ) space — no bond-length cross-contamination.
    # ============================================================
    parser.add_argument(
        "--lambda-geogap",
        type=float,
        default=0.0,
        help=(
            "Weight for IC local geometry gap loss (full phase only). 0=disabled. "
            "Penalizes when E_local(θ/φ-perturbed) - E_local(clean) < --geogap-margin."
        ),
    )
    parser.add_argument(
        "--geogap-margin",
        type=float,
        default=2.0,
        help="Required energy gap: E_local(perturbed) - E_local(clean) must exceed this. Default: 2.0.",
    )
    parser.add_argument(
        "--geogap-theta-sigma",
        type=float,
        default=0.25,
        help=(
            "θ perturbation sigma for IC geogap loss (radians). Default: 0.25. "
            "Tests theta_theta_energy discrimination."
        ),
    )
    parser.add_argument(
        "--geogap-phi-sigma",
        type=float,
        default=0.5,
        help=(
            "φ perturbation sigma for IC geogap loss (radians). Default: 0.5. " "Tests delta_phi_energy discrimination."
        ),
    )
    parser.add_argument(
        "--geogap-frac-perturbed",
        type=float,
        default=0.3,
        help="Fraction of residues perturbed per batch. Default: 0.3.",
    )
    parser.add_argument(
        "--geogap-diag-every", type=int, default=200, help="Log geogap diagnostics every N steps. Default: 200."
    )

    # ============================================================
    # PACKING CONTRASTIVE LOSS OPTIONS (full phase only)
    # Keeps the packing MLP signal alive during full phase DSM training.
    # ============================================================
    parser.add_argument(
        "--lambda-pack-contrastive",
        type=float,
        default=0.0,
        help=(
            "Weight for packing contrastive loss during full phase. 0=disabled. "
            "Recommended: 1.0. Prevents packing MLP from collapsing to zero output."
        ),
    )
    parser.add_argument(
        "--pack-contrastive-margin",
        type=float,
        default=0.5,
        help="Required energy gap for packing contrastive loss (margin mode). Default: 0.5.",
    )
    parser.add_argument(
        "--pack-contrastive-mode",
        type=str,
        choices=["margin", "continuous"],
        default="continuous",
        help=(
            "Packing contrastive loss mode. "
            "'continuous' (default): exp(-gap/T), always rewards deeper wells. "
            "'margin' (legacy): relu(margin - gap), zero when satisfied."
        ),
    )
    parser.add_argument(
        "--pack-contrastive-T-base",
        type=float,
        default=2.0,
        help="Temperature for continuous packing contrastive loss. Default: 2.0.",
    )
    parser.add_argument(
        "--lambda-balance",
        type=float,
        default=0.001,
        help=(
            "Weight for energy balance loss. "
            "Penalises pairwise term ratios outside [1/r, r] where r=--balance-r. "
            "Default: 0.001."
        ),
    )
    parser.add_argument(
        "--balance-r",
        type=float,
        default=7.0,
        help="Allowed pairwise ratio bound for subterm-level energy balance (7 subterms). Default: 7.0.",
    )
    parser.add_argument(
        "--balance-r-term",
        type=float,
        default=4.0,
        help="Allowed pairwise ratio bound for term-level energy balance (4 terms). Default: 4.0.",
    )

    # ============================================================
    # PER-SUBTERM DISCRIMINATION MAINTENANCE (full phase only)
    # L = (1/K) Σ_k softplus(E_k(native) - E_k(perturbed))
    # Prevents individual subterms from collapsing under Z-score pressure.
    # ============================================================
    parser.add_argument(
        "--lambda-discrim",
        type=float,
        default=2.0,
        help=(
            "Weight for per-subterm discrimination maintenance loss. 0=disabled. "
            "Each subterm is independently required to rank native below IC-perturbed. "
            "Prevents collapse under total-energy-only objectives. Recommended: 0.1-0.3."
        ),
    )
    parser.add_argument(
        "--discrim-every",
        type=int,
        default=4,
        help="Run discrimination loss every N gradient steps. Default: 4.",
    )
    parser.add_argument(
        "--discrim-sigma-min",
        type=float,
        default=0.05,
        help="Min IC perturbation noise for discrimination loss (rad). Default: 0.05.",
    )
    parser.add_argument(
        "--discrim-sigma-max",
        type=float,
        default=2.0,
        help="Max IC perturbation noise for discrimination loss (rad). Default: 2.0.",
    )
    parser.add_argument(
        "--discrim-mode",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help=(
            "Aggregation mode for per-subterm discrimination loss. "
            "'mean' (default): (1/K) * sum_k softplus(-gap_k). "
            "'max': max_k softplus(-gap_k). Focuses gradient on the "
            "worst-performing subterm, preventing collapse under strong "
            "funnel pressure. Recommended: 'max'."
        ),
    )

    # ============================================================
    # SECONDARY STRUCTURE BASIN LOSS OPTIONS (full phase only)
    # Enforces E_ss(helix) < E_ss(extended) on real batch geometry.
    # Replaces --lambda-geogap for run20+: geogap caused θθ dominance;
    # this loss directly targets secondary term inversion without crowding local.
    # ============================================================
    parser.add_argument(
        "--lambda-basin",
        type=float,
        default=0.0,
        help=(
            "Weight for secondary structure basin loss (full phase only). 0=disabled. "
            "Enforces E_ss(helix) < E_ss(extended) on real batch geometry. "
            "Replaces --lambda-geogap for run20+. Recommended: 1.0."
        ),
    )
    parser.add_argument(
        "--basin-margin",
        type=float,
        default=0.5,
        help=(
            "Required energy gap E_extended − E_helix per residue for basin loss. "
            "Default: 0.5. Only used when --basin-mode=margin."
        ),
    )
    parser.add_argument(
        "--basin-mode",
        type=str,
        choices=["margin", "continuous"],
        default="continuous",
        help=(
            "Basin loss mode. "
            "'continuous' (default): exp(-gap/T), always rewards helix < extended. "
            "'margin' (legacy): relu(margin - gap)², zero when satisfied."
        ),
    )
    parser.add_argument(
        "--basin-T-base",
        type=float,
        default=2.0,
        help="Temperature for continuous basin loss. Default: 2.0.",
    )

    # ============================================================
    # NATIVE GAP LOSS OPTIONS (full phase only)
    # Enforces a deep energy well around native structures so that
    # Langevin dynamics at beta=1 cannot escape the native basin.
    # Operates on the FULL model energy (all terms), complementing DSM
    # which shapes gradients but does not constrain well depth.
    # ============================================================
    parser.add_argument(
        "--lambda-native",
        type=float,
        default=0.0,
        help=(
            "Weight for native gap loss (full phase only). 0=disabled. "
            "Penalises when E(R_perturbed) - E(R_native) < --native-margin. "
            "Deepens the full-model energy well around native structures. "
            "Recommended: 1.0."
        ),
    )
    parser.add_argument(
        "--native-margin",
        type=float,
        default=0.5,
        help=(
            "Required energy gap for native gap loss: "
            "E(R_perturbed) - E(R_native) must exceed this. "
            "Default: 0.5."
        ),
    )
    parser.add_argument(
        "--native-sigma-min",
        type=float,
        default=0.05,
        help="Min perturbation sigma for native gap loss (radians). Default: 0.05.",
    )
    parser.add_argument(
        "--native-sigma-max",
        type=float,
        default=0.50,
        help=(
            "Max perturbation sigma for native gap loss (radians). Default: 0.50. "
            "Covers the ~0.32 rad RMS exploration of 500 Langevin steps at beta=1, dt=1e-4. "
            "Use 2.0 for continuous mode to test global stability."
        ),
    )
    parser.add_argument(
        "--native-mode",
        type=str,
        choices=["margin", "continuous"],
        default="continuous",
        help=(
            "Native gap loss mode. "
            "'continuous' (default): exp(-gap/T), always rewards deeper wells. "
            "'margin' (legacy): relu(margin - gap), zero when satisfied. "
        ),
    )
    parser.add_argument(
        "--native-T-base",
        type=float,
        default=5.0,
        help=(
            "Base temperature for continuous native gap loss. "
            "Effective T = T_base × sigma. Larger T = gentler push. "
            "Default: 5.0. Only used when --native-mode=continuous."
        ),
    )

    # ============================================================
    # ENERGY LANDSCAPE THEORY (ELT) LOSSES  (full phase only)
    #
    # Three complementary losses from protein folding theory:
    #   Q-funnel:    monotonic energy decrease with increasing Q
    #   Z-score:     native energy significantly below decoy distribution
    #   Frustration: native sequence favoured over random permutations
    #
    # Q-funnel and Z-score share decoys (one set of structural
    # perturbations). Frustration uses sequence permutations.
    #
    # IMPORTANT: ELT losses require full protein chains, not segments.
    # Use --elt-full-chains to enable a separate full-chain data loader.
    #
    # Reference for Q: Best, Hummer & Eaton (2013) PNAS 110:17874
    # ============================================================

    # --- Q-funnel loss ---
    parser.add_argument(
        "--lambda-funnel",
        type=float,
        default=0.5,
        help=(
            "Weight for Q-funnel loss. 0=disabled. "
            "Enforces monotonic energy decrease toward native (Q->1). "
            "Default: 0.5."
        ),
    )
    parser.add_argument(
        "--funnel-T",
        type=float,
        default=2.0,
        help=(
            "Temperature for Q-funnel exp-decay loss. " "Larger = gentler penalty for anti-funnel slopes. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--funnel-n-decoys",
        type=int,
        default=10,
        help="Number of decoys per structure for Q-funnel/gap. Default: 10.",
    )
    parser.add_argument(
        "--funnel-slope-clamp",
        type=float,
        default=10.0,
        help="Max positive slope before exp (prevents overflow). Default: 10.0.",
    )
    parser.add_argument(
        "--funnel-sigma-min",
        type=float,
        default=0.05,
        help="Smallest decoy perturbation sigma (rad) for Q-funnel. Default: 0.05.",
    )
    parser.add_argument(
        "--funnel-sigma-max",
        type=float,
        default=2.0,
        help="Largest decoy perturbation sigma (rad) for Q-funnel. Default: 2.0.",
    )
    parser.add_argument(
        "--funnel-contact-cutoff",
        type=float,
        default=9.5,
        help="Ca contact cutoff (A) for Q computation (Best 2013). Default: 9.5.",
    )

    # --- Z-score loss ---
    parser.add_argument(
        "--lambda-zscore",
        type=float,
        default=0.0,
        help=(
            "Weight for Z-score loss. 0=disabled. "
            "Penalises when native energy is not significantly below decoys. "
            "Uses same decoys as Q-funnel. Recommended: 0.5-1.0."
        ),
    )
    parser.add_argument(
        "--target-zscore",
        type=float,
        default=3.0,
        help=(
            "Target Z-score for Z-score loss. Loss = exp(clamp(target - Z, max=5)). "
            "Self-annealing: strong push when Z < target, negligible above. Default: 3.0."
        ),
    )

    # --- Gap loss (replaces Z-score) ---
    parser.add_argument(
        "--lambda-gap",
        type=float,
        default=1.0,
        help=(
            "Weight for gap loss. 0=disabled. "
            "Per-decoy margin: E(native) must be below each decoy by at least margin. "
            "Uses same decoys as Q-funnel. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--gap-margin",
        type=float,
        default=0.5,
        help=(
            "Margin for gap loss (per-residue). "
            "Loss = exp(clamp(E_native - E_decoy + margin, max=5)). "
            "Each decoy must lie above native by at least this margin. Default: 0.5."
        ),
    )

    # --- Frustration loss ---
    parser.add_argument(
        "--lambda-frustration",
        type=float,
        default=0.0,
        help=(
            "Weight for frustration loss. 0=disabled. "
            "Penalises when native sequence energy is not favoured "
            "over random sequence permutations. Recommended: 0.3-0.5."
        ),
    )
    parser.add_argument(
        "--frustration-T",
        type=float,
        default=2.0,
        help="Temperature for frustration exp-decay loss. Default: 2.0.",
    )
    parser.add_argument(
        "--frustration-n-perms",
        type=int,
        default=4,
        help="Number of random sequence permutations per structure. Default: 4.",
    )

    # --- ELT data options ---
    parser.add_argument(
        "--elt-every",
        type=int,
        default=5,
        help=(
            "Compute ELT losses every N steps (amortize cost). Default: 5. "
            "DSM runs every step; ELT losses are global and need fewer updates."
        ),
    )
    parser.add_argument(
        "--elt-max-len",
        type=int,
        default=512,
        help=("Maximum chain length for ELT full-chain batches. " "Chains longer than this are skipped. Default: 512."),
    )
    parser.add_argument(
        "--elt-batch-size",
        type=int,
        default=16,
        help=(
            "Batch size for ELT/SC full-chain batches. Also controls number "
            "of proteins sampled from negatives per SC training step. Default: 16."
        ),
    )
    parser.add_argument(
        "--local-window-size",
        type=int,
        default=8,
        help=(
            "Window size for local backbone MLP. W=4 is the original 4-mer "
            "(one helical turn). W=8 captures two helical turns and medium-range "
            "backbone correlations. Default: 8."
        ),
    )
    parser.add_argument(
        "--max-rg-ratio",
        type=float,
        default=1.3,
        help=(
            "Maximum Rg/Rg_Flory ratio for training chains. Chains with "
            "Rg exceeding this multiple of the Flory prediction (2.0·L^0.38) "
            "are rejected. Filters elongated structures (coiled-coils, fibrils) "
            "that corrupt E_Rg and E_coord gradients. Default: 1.3 (~18%% filtered). "
            "Set to 0 to disable."
        ),
    )

    # ============================================================
    # NATIVE DEPTH LOSS (Phase 5) — deepen basin
    # ============================================================
    parser.add_argument(
        "--lambda-native-depth",
        type=float,
        default=2.0,
        help=(
            "Weight for native depth loss. "
            "Pushes E(native) below target via "
            "exp(clamp(E_native - target, max=5)). Self-annealing: "
            "strong push above target, negligible below."
        ),
    )
    parser.add_argument(
        "--target-native-depth",
        type=float,
        default=-3.0,
        help=(
            "Target native energy per residue for depth loss. "
            "At E_native=+0.2: loss=3.3. At target: loss=1.0. Below: fades."
        ),
    )

    # ============================================================
    # LAMBDA FLOOR (Phase 5) — prevent subterm collapse
    # ============================================================
    parser.add_argument(
        "--lambda-hb-beta-floor",
        type=float,
        default=0.0,
        help=(
            "Minimum lambda for hb_beta subterm. Default: 0 (disabled). "
            "Set to 0.1 to prevent hb_beta collapse (dies by step ~1200 "
            "when calibrated lambda=0.056 is too small). Applied as "
            "post-step clamp on the raw softplus parameter."
        ),
    )

    # ============================================================
    # SUBTERM CONTROL
    # ============================================================
    parser.add_argument(
        "--disable-subterms",
        nargs="+",
        type=str,
        default=[],
        help=(
            "List of subterms to disable (lambda clamped to zero after each step). "
            "Names: theta_theta, delta_phi, phi_phi, ram, hb_alpha, hb_beta, "
            "repulsion, geom, contact. "
            "Example: --disable-subterms theta_theta delta_phi"
        ),
    )

    # ============================================================
    # DEBUG OPTIONS
    # ============================================================
    parser.add_argument("--debug-mode", action="store_true", help="Enable debug prints from energy terms")

    # ── Learnable buffer flags ──────────────────────────────────────────
    parser.add_argument(
        "--learn-packing-coords", action="store_true", help="Make n_star, sigma in packing learnable (200 params)"
    )
    parser.add_argument(
        "--learn-packing-density", action="store_true", help="Make rho_fit_a/b/c, rho_sigma learnable (20 params)"
    )
    parser.add_argument(
        "--learn-penalty-shapes", action="store_true", help="Make dead_zone, m, alpha learnable (9 params)"
    )
    parser.add_argument(
        "--learn-packing-bounds", action="store_true", help="Make coord/rho lo/hi bounds learnable (210 params)"
    )
    parser.add_argument(
        "--learn-penalty-strengths", action="store_true", help="Make rg/coord/rho lambda strengths learnable (3 params)"
    )
    parser.add_argument(
        "--learn-gate-geometry", action="store_true", help="Make rama gate basin centers/widths learnable (10 params)"
    )
    parser.add_argument(
        "--learn-hbond-geometry", action="store_true", help="Make hbond mu/sigma distance params learnable (6 params)"
    )
    parser.add_argument(
        "--learn-all-buffers", action="store_true", help="Enable ALL learnable buffer flags (~462 params)"
    )

    # ============================================================
    # DATA OPTIONS
    # ============================================================
    parser.add_argument("--backbone-data-dir", type=str, default="analysis/backbone_geometry/data")
    parser.add_argument("--secondary-data-dir", type=str, default="analysis/secondary_analysis/data")

    # ============================================================
    # SELF-CONSISTENT PHASE OPTIONS (--phase self-consistent)
    # Defaults imported from sc_defaults.py (single source of truth)
    # ============================================================
    from calphaebm.training.sc_defaults import SC_DEFAULTS as _D

    sc = parser.add_argument_group("Self-consistent training (--phase self-consistent)")
    sc.add_argument(
        "--n-rounds",
        type=int,
        default=_D["n_rounds"],
        help=f"Number of collect→retrain rounds (default: {_D['n_rounds']})",
    )
    sc.add_argument(
        "--collect-proteins",
        type=int,
        default=_D["collect_proteins"],
        help=f"Proteins to randomly sample per round (default: {_D['collect_proteins']})",
    )
    sc.add_argument(
        "--collect-steps",
        type=int,
        default=_D["collect_steps"],
        help=f"Langevin steps per protein during collection (default: {_D['collect_steps']})",
    )
    sc.add_argument(
        "--collect-beta",
        type=float,
        default=_D["collect_beta"],
        help=f"Inverse temperature for collection dynamics (default: {_D['collect_beta']})",
    )
    sc.add_argument(
        "--collect-step-size",
        type=float,
        default=_D["collect_step_size"],
        help=f"Langevin step size for collection (default: {_D['collect_step_size']})",
    )
    sc.add_argument(
        "--collect-save-every",
        type=int,
        default=_D["collect_save_every"],
        help=f"Check for failures every N Langevin steps (default: {_D['collect_save_every']})",
    )
    sc.add_argument(
        "--n-workers",
        type=int,
        default=_D["collect_n_workers"],
        help=f"Parallel workers for Langevin collection (default: {_D['collect_n_workers']}). "
        "Fork-based, CPU only. Use 1 if fork issues on your platform.",
    )
    sc.add_argument(
        "--collect-max-len",
        type=int,
        default=_D["collect_max_len"],
        help=f"Maximum chain length for collection proteins (default: {_D['collect_max_len']})",
    )
    sc.add_argument(
        "--retrain-steps",
        type=int,
        default=_D["retrain_steps"],
        help=f"Training steps per SC round (default: {_D['retrain_steps']})",
    )
    sc.add_argument(
        "--retrain-lr",
        type=float,
        default=_D["retrain_lr"],
        help=f"Learning rate for SC retraining (default: {_D['retrain_lr']})",
    )
    # Model-sampled negative losses (3× Synthetic counterparts)
    sc.add_argument(
        "--lambda-sampled-hsm",
        type=float,
        default=_D["lambda_sampled_hsm"],
        help=f"Sampled HSM loss weight: HSM on Langevin-sampled negatives (default: {_D['lambda_sampled_hsm']})",
    )
    sc.add_argument(
        "--lambda-sampled-qf",
        type=float,
        default=_D["lambda_sampled_qf"],
        help=f"Sampled Q-funnel loss weight: 3× Synthetic QF (default: {_D['lambda_sampled_qf']})",
    )
    sc.add_argument(
        "--lambda-sampled-drmsd-funnel",
        type=float,
        default=_D["lambda_sampled_drmsd_funnel"],
        help=f"Sampled dRMSD-funnel loss weight: topology-sensitive ordering, "
        f"lower full-dRMSD → lower energy (default: {_D['lambda_sampled_drmsd_funnel']})",
    )
    sc.add_argument(
        "--lambda-sampled-gap",
        type=float,
        default=_D["lambda_sampled_gap"],
        help=f"Sampled gap loss weight: 3× Synthetic gap (default: {_D['lambda_sampled_gap']})",
    )
    sc.add_argument(
        "--sc-margin",
        type=float,
        default=_D["sc_margin"],
        help=f"Sampled gap margin in E/res units (default: {_D['sc_margin']})",
    )
    sc.add_argument(
        "--sc-eval-steps",
        type=int,
        default=_D["eval_steps"],
        help=f"Langevin steps for SC basin evaluation (default: {_D['eval_steps']})",
    )
    sc.add_argument(
        "--sc-eval-beta",
        type=float,
        default=_D["eval_beta"],
        help=f"Inverse temperature for SC basin evaluation (default: {_D['eval_beta']})",
    )
    sc.add_argument(
        "--sc-eval-proteins",
        type=int,
        default=_D["eval_proteins"],
        help=f"Proteins for SC basin evaluation (default: {_D['eval_proteins']})",
    )
    sc.add_argument(
        "--rg-compact",
        type=float,
        default=_D["rg_compact"],
        help=f"Rg ratio below this → compacted failure (default: {_D['rg_compact']})",
    )
    sc.add_argument(
        "--rg-swollen",
        type=float,
        default=_D["rg_swollen"],
        help=f"Rg ratio above this → swollen failure (default: {_D['rg_swollen']})",
    )
    sc.add_argument(
        "--q-false-basin",
        type=float,
        default=_D["q_false_basin"],
        help=f"Q below this + E < E_native → false basin (default: {_D['q_false_basin']})",
    )
    sc.add_argument(
        "--rmsd-drift",
        type=float,
        default=_D["rmsd_drift"],
        help=f"RMSD above this + Q > 0.9 → drift failure (default: {_D['rmsd_drift']})",
    )
    sc.add_argument(
        "--rmsf-frozen",
        type=float,
        default=_D["rmsf_frozen"],
        help=f"RMSF below this → frozen/over-stabilized (default: {_D['rmsf_frozen']})",
    )
    sc.add_argument(
        "--ss-change-thr",
        type=float,
        default=_D["ss_change_thr"],
        help=f"SS change above this → SS loss failure (default: {_D['ss_change_thr']})",
    )
    sc.add_argument(
        "--max-negatives-per-protein",
        type=int,
        default=_D["max_negatives_per_protein"],
        help=f"Cap negatives per protein (default: {_D['max_negatives_per_protein']})",
    )
    sc.add_argument(
        "--convergence-threshold",
        type=float,
        default=_D["convergence_threshold"],
        help=f"Stop if E_delta improvement < this between rounds (default: {_D['convergence_threshold']})",
    )
    sc.add_argument(
        "--min-negatives",
        type=int,
        default=_D["min_negatives"],
        help=f"Skip retraining if fewer negatives collected (default: {_D['min_negatives']})",
    )
    sc.add_argument(
        "--sc-resume-round", type=int, default=0, help="Resume from this round number (default: 0 = start fresh)"
    )

    return parser
