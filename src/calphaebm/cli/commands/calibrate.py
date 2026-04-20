# src/calphaebm/cli/commands/calibrate.py

"""Calibrate sub-term energy scales and geometry feature parameters.

Three calibration modes are available via flags:

  calphaebm calibrate
      Measures the mean raw output of each energy sub-term over PDB structures
      perturbed with sigma ~ LogUniform(sigma_min, sigma_max) in IC space
      (radians) — the same distribution used in DSM training — and computes init weights such that
      every sub-term contributes `target` energy per residue at weight=1.0.

      Run once before training. Results are printed and optionally applied
      directly to a checkpoint via --apply-to-ckpt.

  calphaebm calibrate --coord-n-star
      Calibrates per-AA optimal soft neighbor counts (n*) for the coordination
      penalty E_coord. Uses a fixed sigmoid g(r) = σ((7.0 - r) / 1.0) to count
      neighbors — independent of any checkpoint. Run once on training data;
      output is valid forever. Saves coord_n_star.json.

  calphaebm calibrate --rg-gate
      Calibrates Rg Gaussian gate σ for packing terms. Measures Rg variation
      at each noise level.

  calphaebm calibrate-geometry
      Calibrates the _GeometryFeatures parameters used by the packing MLP:
        - sig_tau: sigmoid transition width for soft neighbour counts.
          Chosen to maximise DSM-weighted gradient sensitivity — i.e. the
          value of tau that causes soft counts to change most when structures
          are perturbed at DSM-scale sigmas.
        - normalisation denominators (norm_n_tight, norm_n_medium, etc.):
          computed as mean/std of each raw feature over the training set so
          that after normalisation all features are ~N(0,1) / mean≈1.

      Run once before training the packing MLP (i.e. before run14+).
      Saves a JSON that is passed via --packing-geom-calibration at train time.

Usage
-----
    calphaebm calibrate \\
        --pdb train_entities.no_test_entries.txt \\
        --backbone-data-dir analysis/backbone_geometry/data \\
        --repulsion-data-dir analysis/repulsion_analysis/data \\
        --sigma-min 0.05 --sigma-max 2.0 \\
        --n-samples 512 \\
        --out calibration.json

    calphaebm calibrate --coord-n-star \\
        --pdb train_entities.no_test_entries.txt \\
        --n-samples 500 \\
        --out coord_n_star.json

    calphaebm calibrate-geometry \\
        --segments processed_cache/segments_f1c082560ae42021.pt \\
        --limit 500 \\
        --sigma-min 0.05 --sigma-max 3.0 \\
        --out geometry_feature_calibration.json

    # tau candidates are derived automatically from pairwise distance
    # spread near each shell cutoff (tau_min=std/5, tau_max=std*2,
    # 8 log-uniform steps). Override with explicit values if needed:
    calphaebm calibrate-geometry ... \\
        --tau-candidates 0.5 1.0 1.5 2.0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import types
from pathlib import Path

from calphaebm.utils.logging import get_logger

logger = get_logger()


def add_parser(subparsers) -> argparse.ArgumentParser:
    """Add calibrate command parser."""
    parser = subparsers.add_parser(
        "calibrate",
        description="Calibrate energy sub-term init weights from perturbed PDB structures",
        help="Calibrate sub-term energy scales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--pdb", required=True, nargs="+", help="PDB ID list file (or list of IDs)")
    parser.add_argument(
        "--backbone-data-dir",
        default="analysis/backbone_geometry/data",
        help="Path to backbone geometry data (default: analysis/backbone_geometry/data)",
    )
    parser.add_argument(
        "--secondary-data-dir",
        default="analysis/secondary_analysis/data",
        help="Path to secondary structure basin surfaces (default: analysis/secondary_analysis/data)",
    )
    parser.add_argument(
        "--repulsion-data-dir",
        default="analysis/repulsion_analysis/data",
        help="Path to repulsion analysis data (default: analysis/repulsion_analysis/data)",
    )
    parser.add_argument("--cache-dir", default="./pdb_cache", help="PDB cache directory (default: ./pdb_cache)")
    parser.add_argument("--limit", type=int, default=1000, help="Max PDB IDs to load (default: 1000)")

    # Calibration
    parser.add_argument("--n-samples", type=int, default=512, help="Number of segments to use (default: 512)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument(
        "--sigma-min", type=float, default=0.05, help="Lower sigma bound in radians (default: 0.05 rad ≈ 3°)"
    )
    parser.add_argument(
        "--sigma-max", type=float, default=2.0, help="Upper sigma bound in radians (default: 2.0 rad ≈ 115°)"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=1.0 / 7.0,
        help="Target mean energy per residue per sub-term (default: 1/7 ≈ 0.143 "
        "for 7 subterms with 4-mer architecture; use 1/9 for old 9-subterm)",
    )
    parser.add_argument(
        "--calibrate-mlp-terms",
        action="store_true",
        default=False,
        help="Also calibrate theta_theta and phi_phi MLP sub-terms in local. "
        "Only meaningful after training — at init these terms produce "
        "near-zero output and cannot be meaningfully calibrated.",
    )

    # Output
    parser.add_argument("--out", default="calibration.json", help="Output JSON path (default: calibration.json)")

    # Optional checkpoint patching
    parser.add_argument("--apply-to-ckpt", default=None, help="Apply calibrated weights to this checkpoint")
    parser.add_argument("--out-ckpt", default=None, help="Output checkpoint path (required with --apply-to-ckpt)")
    parser.add_argument(
        "--packing-geom-calibration",
        default="analysis/packing_analysis/data/geometry_feature_calibration.json",
        help="Path to geometry_feature_calibration.json for packing MLP "
        "(default: analysis/packing_analysis/data/geometry_feature_calibration.json)",
    )

    # Rg gate calibration mode
    parser.add_argument(
        "--rg-gate",
        action="store_true",
        default=False,
        help="Calibrate Rg Gaussian gate σ for packing terms. "
        "Measures Rg variation at each noise level. "
        "Uses --pdb, --n-samples, --sigma-min, --sigma-max.",
    )
    parser.add_argument(
        "--rg-gate-n-repeats",
        type=int,
        default=10,
        help="Perturbation repeats per structure per σ for Rg gate calibration (default: 10)",
    )

    # Coordination n* calibration mode
    parser.add_argument(
        "--coord-n-star",
        action="store_true",
        default=False,
        help="Calibrate per-AA optimal coordination counts (n*) for E_coord. "
        "Uses fixed sigmoid g(r)=σ((7.0-r)/1.0) — no checkpoint needed. "
        "Uses --pdb, --n-samples, --min-len, --max-len, --seed, --out.",
    )
    parser.add_argument(
        "--min-len", type=int, default=40, help="Min chain length for Rg gate calibration (default: 40)"
    )
    parser.add_argument(
        "--max-len", type=int, default=512, help="Max chain length for Rg gate calibration (default: 512)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    parser.set_defaults(func=run)
    return parser


def run(args) -> int:
    """Run calibrate command."""
    # ── Rg gate calibration mode ──────────────────────────────────────────────
    if getattr(args, "rg_gate", False):
        return _run_rg_gate_calibration(args)

    # ── Coordination n* calibration mode ──────────────────────────────────────
    if getattr(args, "coord_n_star", False):
        return _run_coord_n_star_calibration(args)

    import torch

    from calphaebm.cli.commands.train.data_utils import parse_pdb_arg
    from calphaebm.data.pdb_dataset import PDBSegmentDataset
    from calphaebm.training.calibrators.subterm_calibrators import SubtermScaleCalibrator

    # ── validate ──────────────────────────────────────────────────────────────
    if args.apply_to_ckpt and not args.out_ckpt:
        raise ValueError("--out-ckpt is required when --apply-to-ckpt is set")

    # ── load PDB IDs ──────────────────────────────────────────────────────────
    pdb_ids = parse_pdb_arg(args.pdb)
    if args.limit:
        pdb_ids = pdb_ids[: args.limit]
    logger.info("Using %d PDB IDs", len(pdb_ids))

    # ── load segments ─────────────────────────────────────────────────────────
    logger.info("Loading segments...")
    dataset = PDBSegmentDataset(
        pdb_ids=pdb_ids,
        cache_dir=args.cache_dir,
        return_dict=True,
    )
    segments = list(dataset)
    logger.info("Loaded %d segments", len(segments))

    # ── build a fresh model for calibration ───────────────────────────────────
    logger.info("Building model...")
    from calphaebm.cli.commands.train.model_builder import build_model

    mock_args = types.SimpleNamespace(
        # Paths
        backbone_data_dir=args.backbone_data_dir,
        secondary_data_dir=args.secondary_data_dir,
        repulsion_data_dir=args.repulsion_data_dir,
        packing_data_dir=args.repulsion_data_dir,  # same dir, matches train default
        # Nonbonded
        repulsion_K=64,
        repulsion_exclude=3,
        repulsion_r_on=8.0,
        repulsion_r_cut=10.0,
        packing_r_on=8.0,
        packing_r_cut=10.0,
        # Packing
        packing_short_gate_on=4.5,
        packing_short_gate_off=5.0,
        packing_rbf_centers=[5.5, 7.0, 9.0],
        packing_rbf_width=1.0,
        packing_max_dist=10.0,
        packing_init_from="log_oe",
        packing_normalize_by_length=True,
        packing_debug_scale=False,
        packing_debug_every=200,
        packing_geom_calibration=args.packing_geom_calibration,
        # Local term
        # init_bond_spring removed — bond_spring is gone in IC version
        init_theta_theta_weight=1.0,
        init_delta_phi_weight=1.0,
        # Gates (unused — calibration doesn't train, but build_model needs them)
        lambda_local=1.0,
        lambda_rep=1.0,
        lambda_ss=1.0,
        lambda_pack=1.0,
        # Misc
        debug_mode=False,
    )
    model = build_model(
        terms_set={"local", "secondary", "repulsion", "packing"},
        device=torch.device("cpu"),
        args=mock_args,
    )

    # ── run calibration ───────────────────────────────────────────────────────
    # ALWAYS load checkpoint weights before measuring. Without this, terms like
    # packing geom MLP (trained in run28, outputs ~1.0) get measured at random
    # init (output ~0.02) → λ=60 → explosion at training time.
    # The --calibrate-mlp-terms flag only controls whether NEW local MLPs
    # (θθ, φφ) are calibrated — those aren't in the checkpoint and output ~0.
    if args.apply_to_ckpt:
        logger.info("Loading checkpoint weights for calibration measurement: %s", args.apply_to_ckpt)
        ckpt_for_measure = torch.load(args.apply_to_ckpt, map_location="cpu", weights_only=False)
        state_for_measure = ckpt_for_measure.get("model_state", ckpt_for_measure)
        missing, unexpected = model.load_state_dict(state_for_measure, strict=False)
        logger.info(
            "  Loaded: %d keys, missing: %d (new params), unexpected: %d",
            len(state_for_measure) - len(missing),
            len(missing),
            len(unexpected),
        )

    calibrator = SubtermScaleCalibrator(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        target=args.target,
        calibrate_mlp=args.calibrate_mlp_terms,
    )
    results = calibrator.run(
        segments=segments,
        model=model,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )

    results.log_summary()
    results.save(args.out)

    # ── optionally patch a checkpoint ─────────────────────────────────────────
    if args.apply_to_ckpt:
        logger.info("Applying calibrated weights to checkpoint: %s", args.apply_to_ckpt)

        ckpt = torch.load(args.apply_to_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)

        # apply_to_model sets the parameter values on the freshly-built model;
        # we then copy those tensors into the checkpoint state dict
        results.apply_to_model(model)
        new_state = model.state_dict()

        patched_keys = []

        # Deterministic sub-terms (always calibrated)
        deterministic_keys = [
            # 4-mer architecture
            "local._lambda_raw",
            # Local: v2 uses _theta_theta_mlp_w, v1 uses _theta_theta_weight_raw
            "local._theta_theta_mlp_w",
            "local._theta_theta_weight_raw",
            "local._delta_phi_weight_raw",
        ]

        # MLP sub-terms (only when --calibrate-mlp-terms was used)
        mlp_keys = (
            [
                "local._phi_phi_mlp_w",
                "local._phi_phi_weight_raw",
            ]
            if args.calibrate_mlp_terms
            else []
        )

        # Secondary: ram weight + H-bond lambdas
        secondary_keys = [
            "secondary._lambda_ram_raw",
            "secondary._ram_weight_raw",
            "secondary.lambda_ram",
            "secondary.hb_helix._lambda_raw",  # H-bond alpha lambda
            "secondary.hb_sheet._lambda_raw",  # H-bond beta lambda
        ]

        # Repulsion and packing — always calibrated
        # Packing lambda — repulsion is NOT patched (safety constraint, preserved)
        rep_pack_keys = [
            "packing._lambda_pack_raw",
            "packing.burial._lambda_hp_raw",  # contact HP lambda
        ]

        for key in deterministic_keys + mlp_keys + secondary_keys + rep_pack_keys:
            if key in state and key in new_state:
                state[key] = new_state[key].clone()
                patched_keys.append(key)
                logger.info("  Patched %s", key)
            elif key in new_state and key not in state:
                # New key not in old checkpoint — add it
                state[key] = new_state[key].clone()
                patched_keys.append(key)
                logger.info("  Added %s (new param)", key)
            # else: key not in model, skip silently

        if "model_state" in ckpt:
            ckpt["model_state"] = state
        else:
            ckpt = state

        out_path = Path(args.out_ckpt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out_path)
        logger.info("Saved calibrated checkpoint to %s", out_path)
        logger.info("Patched %d keys: %s", len(patched_keys), patched_keys)

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# calphaebm calibrate-geometry  (delegates to analysis/packing module)
# ─────────────────────────────────────────────────────────────────────────────


def add_geometry_parser(subparsers) -> argparse.ArgumentParser:
    """Add calibrate-geometry subcommand.

    This is a thin wrapper that delegates to calphaebm.analysis.packing.
    The canonical command is:

        calphaebm analyze packing ...

    This alias is kept for backward compatibility with existing scripts.
    """
    from calphaebm.analysis.packing.packing_cli import add_subparser as _add

    # Register under 'calibrate-geometry' as an alias
    parser = subparsers.add_parser(
        "calibrate-geometry",
        description=(
            "Alias for 'calphaebm analyze packing'.\n"
            "Calibrate packing geometry feature parameters (sig_tau, normalisation).\n\n"
            "Prefer: calphaebm analyze packing --help"
        ),
        help="[alias] Calibrate packing geometry features — prefer 'analyze packing'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Re-use the same args as the packing CLI
    from calphaebm.analysis.packing.packing_config import (
        DEFAULT_N_STRUCTURES,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SEGMENTS_PT,
        DEFAULT_SIGMA_MAX,
        DEFAULT_SIGMA_MIN,
    )

    parser.add_argument("--segments", type=str, default=str(DEFAULT_SEGMENTS_PT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-structures", type=int, default=DEFAULT_N_STRUCTURES)
    parser.add_argument("--sigma-min", type=float, default=DEFAULT_SIGMA_MIN)
    parser.add_argument("--sigma-max", type=float, default=DEFAULT_SIGMA_MAX)
    parser.add_argument("--quiet", action="store_true")
    # Legacy --limit alias
    parser.add_argument(
        "--limit", type=int, default=None, dest="n_structures", help="Alias for --n-structures (deprecated)"
    )
    # Legacy --out alias
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="[ignored] Output JSON path now fixed to "
        "analysis/packing_analysis/data/geometry_feature_calibration.json",
    )

    from calphaebm.analysis.packing.packing_core import run_packing_analysis

    parser.set_defaults(func=run_packing_analysis)
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# calphaebm calibrate --rg-gate
# ─────────────────────────────────────────────────────────────────────────────


def _run_coord_n_star_calibration(args) -> int:
    """Calibrate per-AA optimal soft neighbor counts (n*) for E_coord.

    Dispatched from run() when --coord-n-star is passed.
    Uses shared args: --pdb, --n-samples, --out, --min-len, --max-len, --seed

    Uses FIXED sigmoid g(r) = σ((7.0 - r) / 1.0) — independent of any
    checkpoint. Run once on training data; output is valid forever.
    """
    from collections import defaultdict

    import numpy as np
    import torch

    from calphaebm.cli.commands.train.data_utils import parse_pdb_arg
    from calphaebm.data.pdb_chain_dataset import PDBChainDataset
    from calphaebm.geometry.pairs import topk_nonbonded_pairs

    # Fixed sigmoid constants — must match PackingEnergy.COORD_R_HALF / COORD_TAU
    COORD_R_HALF = 7.0
    COORD_TAU = 1.0

    # 1-letter alphabetical — matches the seq encoding in PDBChainDataset
    AA_NAMES = [
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "ILE",
        "LYS",
        "LEU",
        "MET",
        "ASN",
        "PRO",
        "GLN",
        "ARG",
        "SER",
        "THR",
        "VAL",
        "TRP",
        "TYR",
    ]

    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    min_len = getattr(args, "min_len", 40)
    max_len = getattr(args, "max_len", 512)
    n_samples = getattr(args, "n_samples", 500)
    out_path = getattr(args, "out", "coord_n_star.json")
    # Override default --out if user didn't specify (default is calibration.json)
    if out_path == "calibration.json":
        out_path = "coord_n_star.json"
    max_dist = 10.0

    logger.info("Calibrating coordination n* values")
    logger.info("  Fixed sigmoid: g(r) = σ((%.1f - r) / %.1f)", COORD_R_HALF, COORD_TAU)

    # Load PDB IDs
    pdb_ids = parse_pdb_arg(args.pdb)
    limit = getattr(args, "limit", None)
    if limit:
        pdb_ids = pdb_ids[:limit]
    logger.info("  PDB IDs: %d", len(pdb_ids))

    # Load chains
    dataset = PDBChainDataset(pdb_ids=pdb_ids, min_len=min_len, max_len=max_len)
    n_use = min(len(dataset), n_samples)
    logger.info("  Chains: %d available, using %d (L=%d-%d)", len(dataset), n_use, min_len, max_len)

    # Accumulate per-AA soft neighbor counts
    aa_counts: dict[int, list[float]] = defaultdict(list)

    with torch.no_grad():
        for idx in range(n_use):
            R_chain, seq_chain, pdb_id, chain_id = dataset[idx]
            L = R_chain.shape[0]

            # Add batch dim
            R = R_chain.unsqueeze(0)  # (1, L, 3)
            seq = seq_chain.unsqueeze(0)  # (1, L)

            # Compute neighbor distances
            r, j_idx = topk_nonbonded_pairs(R, k=64, exclude=3, cutoff=max_dist)

            # Fixed sigmoid — same as PackingEnergy._compute_coord_energy
            valid = (r < max_dist - 1e-4).float()
            g = torch.sigmoid((COORD_R_HALF - r.clamp(max=max_dist)) / COORD_TAU)
            g = g * valid
            n_soft = g.sum(dim=-1)  # (1, L)

            # Accumulate per AA type
            for i in range(L):
                aa = int(seq[0, i].item())
                if 0 <= aa < 20:
                    aa_counts[aa].append(float(n_soft[0, i].item()))

            if (idx + 1) % 100 == 0:
                logger.info("  Processed %d/%d", idx + 1, n_use)

    # Compute p5/p95 band per AA type
    n_lo_dict = {}
    n_hi_dict = {}
    n_mean_dict = {}
    logger.info("")
    logger.info("Calibrated coordination band (fixed sigmoid r_half=%.1f, tau=%.1f):", COORD_R_HALF, COORD_TAU)
    logger.info("%-5s  %6s  %6s  %6s  %6s  %6s", "AA", "n_lo", "mean", "n_hi", "std", "count")
    logger.info("-" * 55)
    for aa in range(20):
        vals = aa_counts.get(aa, [])
        if not vals:
            logger.warning("  %s: no samples — using defaults (0.0, 7.0, 99.0)", AA_NAMES[aa])
            n_lo_dict[AA_NAMES[aa]] = 0.0
            n_hi_dict[AA_NAMES[aa]] = 99.0
            n_mean_dict[AA_NAMES[aa]] = 7.0
            continue
        arr = np.array(vals)
        p5 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))
        mean_n = float(np.mean(arr))
        std_n = float(np.std(arr))
        count = len(vals)
        n_lo_dict[AA_NAMES[aa]] = round(p5, 3)
        n_hi_dict[AA_NAMES[aa]] = round(p95, 3)
        n_mean_dict[AA_NAMES[aa]] = round(mean_n, 3)
        logger.info("  %s  %6.2f  %6.2f  %6.2f  %6.2f  %5d", AA_NAMES[aa], p5, mean_n, p95, std_n, count)

    # Save as ordered lists for direct use
    n_lo_list = [n_lo_dict[AA_NAMES[aa]] for aa in range(20)]
    n_hi_list = [n_hi_dict[AA_NAMES[aa]] for aa in range(20)]
    n_mean_list = [n_mean_dict[AA_NAMES[aa]] for aa in range(20)]

    output = {
        "n_lo_per_aa": n_lo_dict,
        "n_hi_per_aa": n_hi_dict,
        "n_mean_per_aa": n_mean_dict,
        "n_lo_list": n_lo_list,
        "n_hi_list": n_hi_list,
        "n_mean_list": n_mean_list,
        "percentiles": [5, 95],
        "coord_r_half": COORD_R_HALF,
        "coord_tau": COORD_TAU,
        "n_structures": n_use,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("")
    logger.info("Saved to %s", out_path)
    logger.info("Use with: --coord-lambda 0.5 --coord-n-star-file %s", out_path)
    return 0


def _run_rg_gate_calibration(args) -> int:
    """Measure Rg ratio std at each DSM noise level.

    Dispatched from run() when --rg-gate is passed.
    Uses shared args: --pdb, --n-samples, --sigma-min, --sigma-max, --out, --limit
    Plus Rg-gate specific: --rg-gate-n-repeats, --min-len, --max-len, --seed
    """
    import numpy as np
    import torch

    from calphaebm.cli.commands.train.data_utils import parse_pdb_arg
    from calphaebm.data.pdb_chain_dataset import PDBChainDataset
    from calphaebm.evaluation.metrics.rg import radius_of_gyration
    from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct

    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load PDB IDs
    pdb_ids = parse_pdb_arg(args.pdb)
    if args.limit:
        pdb_ids = pdb_ids[: args.limit]
    logger.info("Parsed %d PDB IDs", len(pdb_ids))

    min_len = getattr(args, "min_len", 40)
    max_len = getattr(args, "max_len", 512)
    n_repeats = getattr(args, "rg_gate_n_repeats", 10)

    # Load full-chain dataset
    dataset = PDBChainDataset(
        pdb_ids=pdb_ids,
        cache_dir=getattr(args, "cache_dir", "./pdb_cache"),
        min_len=min_len,
        max_len=max_len,
    )
    logger.info("Dataset: %d chains (L=%d-%d)", len(dataset), min_len, max_len)

    n = min(args.n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)

    # Noise levels: log-uniform grid spanning DSM training range
    sigma_min = getattr(args, "sigma_min", 0.05)
    sigma_max = getattr(args, "sigma_max", 2.0)
    sigma_levels = sorted(
        set([round(x, 3) for x in np.exp(np.linspace(math.log(sigma_min), math.log(sigma_max), 10)).tolist()])
    )

    results = {}

    for sigma in sigma_levels:
        ratios = []

        for idx_i, ds_idx in enumerate(indices):
            sample = dataset[int(ds_idx)]
            R = sample[0]  # (L, 3) tensor
            L = R.shape[0]

            rg_native = radius_of_gyration(R.numpy())
            if rg_native < 1.0:
                continue

            R_b = R.unsqueeze(0)
            with torch.no_grad():
                theta, phi = coords_to_internal(R_b)
                anchor = extract_anchor(R_b)

            for _ in range(n_repeats):
                with torch.no_grad():
                    theta_noisy = theta + sigma * torch.randn_like(theta)
                    phi_noisy = phi + sigma * torch.randn_like(phi)
                    theta_noisy = theta_noisy.clamp(0.01, math.pi - 0.01)
                    R_pert = nerf_reconstruct(theta_noisy, phi_noisy, anchor, bond=3.8)

                rg_pert = radius_of_gyration(R_pert[0, :L].numpy())
                ratios.append(rg_pert / rg_native)

            if (idx_i + 1) % 100 == 0:
                logger.info("  σ=%.2f: %d/%d structures", sigma, idx_i + 1, n)

        ratios = np.array(ratios)
        r = {
            "sigma_noise": sigma,
            "n_samples": len(ratios),
            "mean_ratio": float(ratios.mean()),
            "std_ratio": float(ratios.std()),
            "median_ratio": float(np.median(ratios)),
            "p05": float(np.percentile(ratios, 5)),
            "p95": float(np.percentile(ratios, 95)),
        }
        results[str(sigma)] = r

        logger.info(
            "σ_noise=%.3f  Rg_ratio: mean=%.4f ± %.4f  " "median=%.4f  [p05=%.3f, p95=%.3f]  (n=%d)",
            sigma,
            r["mean_ratio"],
            r["std_ratio"],
            r["median_ratio"],
            r["p05"],
            r["p95"],
            r["n_samples"],
        )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("  Rg GATE CALIBRATION RESULTS")
    logger.info("=" * 70)
    logger.info("  %-10s  %-12s  %-12s  %-10s  %-14s", "σ_noise", "mean_ratio", "σ_Rg_ratio", "median", "p05-p95")
    logger.info("  " + "-" * 62)

    sigma_rg_table = {}
    for sigma in sigma_levels:
        r = results[str(sigma)]
        sigma_rg_table[sigma] = r["std_ratio"]
        logger.info(
            "  %-10.3f  %-12.4f  %-12.4f  %-10.4f  [%.3f, %.3f]",
            sigma,
            r["mean_ratio"],
            r["std_ratio"],
            r["median_ratio"],
            r["p05"],
            r["p95"],
        )

    logger.info("=" * 70)

    # Recommendation: use std at lowest noise level
    low_noise_std = sigma_rg_table.get(sigma_levels[0], 0.10)
    recommended = round(max(low_noise_std, 0.03), 3)

    logger.info("\n  Recommended --rg-gate-sigma: %.3f", recommended)
    logger.info("  (Based on σ_Rg at σ_noise=%.3f rad = near-native fluctuations)", sigma_levels[0])
    logger.info("  Gate effect at this σ:")
    for dev in [0.03, 0.05, 0.10, 0.20, 0.30]:
        gate_val = math.exp(-(dev**2) / (2 * recommended**2))
        logger.info("    Rg deviation ±%.0f%%: gate = %.3f", 100 * dev, gate_val)

    # Save
    output = {
        "sigma_levels": sigma_levels,
        "sigma_rg_lookup": {str(s): sigma_rg_table[s] for s in sigma_levels},
        "recommended_rg_gate_sigma": recommended,
        "full_results": results,
        "config": {
            "n_samples": n,
            "n_repeats": n_repeats,
            "min_len": min_len,
            "max_len": max_len,
            "seed": seed,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved to %s", out_path)

    return 0
