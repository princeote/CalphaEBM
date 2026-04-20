# src/calphaebm/cli/commands/validate.py

"""Validation command.

Runs one or more validators against a checkpoint without launching a training run.

Subcommands
-----------
  calphaebm validate local       -- LocalValidator: geometry gap on clean vs distorted structures
  calphaebm validate behavior    -- BehaviorValidator: helix/extended synthetic gap tests
  calphaebm validate generation  -- GenerationValidator: IC Langevin structure generation quality

Examples
--------
  calphaebm validate local \\
      --ckpt checkpoints/run19/secondary/step003000.pt \\
      --pdb  train_entities.no_test_entries.txt \\
      --n-batches 20

  calphaebm validate behavior \\
      --ckpt checkpoints/run19/secondary/step003000.pt

  calphaebm validate generation \\
      --ckpt checkpoints/run19/secondary/step003000.pt \\
      --pdb  train_entities.no_test_entries.txt \\
      --n-batches 10 --langevin-steps 200
"""

from __future__ import annotations

import argparse

from calphaebm.utils.logging import get_logger

logger = get_logger()


def _ckpt_step(args) -> "int | None":
    """Extract global_step from a checkpoint for display labels."""
    import torch as _torch

    try:
        ckpt = _torch.load(args.ckpt, map_location="cpu", weights_only=False)
        return ckpt.get("global_step") if isinstance(ckpt, dict) else None
    except Exception:
        return None


# Shared model-construction defaults (mirrors simulate.py)
_BACKBONE_DATA_DIR = "analysis/backbone_geometry/data"
_SECONDARY_DATA_DIR = "analysis/secondary_analysis/data"
_REPULSION_DATA_DIR = "analysis/repulsion_analysis/data"
_PACKING_DATA_DIR = "analysis/repulsion_analysis/data"
_PACKING_GEOM_CAL = "analysis/packing_analysis/data/geometry_feature_calibration.json"


# ─────────────────────────────────────────────────────────────────────────────
# Parser registration
# ─────────────────────────────────────────────────────────────────────────────


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "validate",
        description="Run validation against a checkpoint",
        help="Validate a checkpoint (local / behavior / generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(
        dest="validate_mode",
        title="Validation mode",
        required=True,
    )

    _add_local_parser(sub)
    _add_secondary_parser(sub)
    _add_repulsion_parser(sub)
    _add_packing_parser(sub)
    _add_model_parser(sub)
    _add_behavior_parser(sub)
    _add_generation_parser(sub)

    parser.set_defaults(func=run)
    return parser


def _common_args(p: argparse.ArgumentParser) -> None:
    """Args shared by all validate subcommands."""
    p.add_argument("--ckpt", required=True, help="Checkpoint .pt file")
    p.add_argument("--device", default="cpu", help="Device (default: cpu)")

    # Model construction (same defaults as simulate)
    p.add_argument("--backbone-data-dir", default=_BACKBONE_DATA_DIR)
    p.add_argument("--secondary-data-dir", default=_SECONDARY_DATA_DIR)
    p.add_argument("--repulsion-data-dir", default=_REPULSION_DATA_DIR)
    p.add_argument("--packing-data-dir", default=_PACKING_DATA_DIR)
    p.add_argument("--packing-geom-calibration", default=_PACKING_GEOM_CAL)

    p.add_argument(
        "--energy-terms",
        nargs="*",
        default=["local", "repulsion", "secondary"],
        choices=["local", "repulsion", "secondary", "packing", "all"],
        help="Energy terms to include in model (default: local repulsion secondary)",
    )


def _data_args(p: argparse.ArgumentParser) -> None:
    """Args for commands that need a data loader."""
    p.add_argument(
        "--pdb",
        default="train_entities.no_test_entries.txt",
        help="PDB ID list file (default: train_entities.no_test_entries.txt)",
    )
    p.add_argument("--cache-dir", default="./pdb_cache")
    p.add_argument("--seg-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-batches", type=int, default=20, help="Number of batches to validate (default: 20)")
    p.add_argument(
        "--n-ids", type=int, default=300, help="PDB IDs to load segments from (default: 300; more = slower startup)"
    )


def _add_model_args(p) -> None:
    """Add shared model-loading args to any validate subparser."""
    p.add_argument("--secondary-data-dir", default="analysis/secondary_analysis/data")
    p.add_argument("--repulsion-data-dir", default="analysis/repulsion_analysis/data")
    p.add_argument("--packing-data-dir", default="analysis/repulsion_analysis/data")
    p.add_argument("--backbone-data-dir", default="analysis/backbone_geometry/data")
    p.add_argument("--packing-geom-calibration", default="analysis/repulsion_analysis/packing_geom_calibration.pt")
    p.add_argument(
        "--energy-terms",
        nargs="+",
        default=["local", "repulsion", "secondary"],
        choices=["local", "repulsion", "secondary", "packing", "all"],
    )


def _add_local_parser(sub) -> None:
    p = sub.add_parser(
        "local",
        help="LocalValidator: geometry gap (native vs Cartesian-noise distorted)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _common_args(p)
    _data_args(p)
    p.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=None,
        metavar="σ",
        help="IC noise level(s) in radians. Pass one value for single run, "
        "multiple for sweep. Default: sweep [0.05, 0.10, 0.20, 0.30].",
    )
    p.add_argument("--n-corruptions", type=int, default=5, help="IC-noise samples per batch (default: 5)")


def _add_model_parser(sub) -> None:
    p = sub.add_parser("model", help="Validate combined model energy (all terms, full DSM gap).")
    _add_model_args(p)
    p.set_defaults(energy_terms=["local", "repulsion", "secondary", "packing"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pdb", required=True)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--n-ids", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seg-len", type=int, default=64)
    p.add_argument("--cache-dir", default="pdb_cache")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=None,
        metavar="σ",
        help="IC noise levels in radians (default: 0.05 0.10 0.20 0.30)",
    )


def _add_behavior_parser(sub) -> None:
    p = sub.add_parser(
        "behavior",
        help="BehaviorValidator: helix vs extended synthetic gaps (no data needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _common_args(p)
    p.add_argument("--length", type=int, default=20, help="Synthetic chain length (default: 20)")
    p.add_argument("--helix-noise", type=float, default=0.02)
    p.add_argument("--random-noise", type=float, default=0.5)


def _add_secondary_parser(sub) -> None:
    p = sub.add_parser("secondary", help="Validate secondary structure energy (IC-noise sweep).")
    _add_model_args(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pdb", required=True)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--n-ids", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seg-len", type=int, default=64)
    p.add_argument("--cache-dir", default="pdb_cache")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=None,
        metavar="σ",
        help="IC noise levels in radians (default: 0.10 0.15 0.20 0.35)",
    )


def _add_repulsion_parser(sub) -> None:
    p = sub.add_parser("repulsion", help="Validate repulsion energy (compression + IC-noise).")
    _add_model_args(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pdb", required=True)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--n-ids", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seg-len", type=int, default=64)
    p.add_argument("--cache-dir", default="pdb_cache")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--compress",
        type=float,
        nargs="+",
        default=[0.90, 0.80, 0.70],
        metavar="s",
        help="Compression scale factors (default: 0.90 0.80 0.70)",
    )


def _add_packing_parser(sub) -> None:
    p = sub.add_parser("packing", help="Validate packing energy (IC-noise + sequence shuffle).")
    _add_model_args(p)
    # Override energy-terms default to include packing
    p.set_defaults(energy_terms=["local", "repulsion", "secondary", "packing"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pdb", required=True)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--n-ids", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seg-len", type=int, default=64)
    p.add_argument("--cache-dir", default="pdb_cache")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=None,
        metavar="σ",
        help="IC noise levels in radians (default: 0.05 0.10 0.20 0.30)",
    )


def _add_generation_parser(sub) -> None:
    p = sub.add_parser(
        "generation",
        help="GenerationValidator: IC Langevin generation quality (RMSD, energy delta)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _common_args(p)
    _data_args(p)
    p.add_argument("--langevin-steps", type=int, default=200, help="IC Langevin steps per structure (default: 200)")
    p.add_argument("--step-size", type=float, default=1e-4, help="IC Langevin step size (default: 1e-4)")
    p.add_argument("--temperature", type=float, default=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_model(args, device):
    """Build and load model from checkpoint, following simulate.py pattern."""
    import torch

    from calphaebm.models.energy import create_total_energy

    terms_set = {t.lower() for t in args.energy_terms}
    if "all" in terms_set:
        terms_set = {"local", "repulsion", "secondary", "packing"}

    import os

    packing_geom = args.packing_geom_calibration if os.path.exists(args.packing_geom_calibration) else None

    model = create_total_energy(
        backbone_data_dir=args.backbone_data_dir,
        secondary_data_dir=args.secondary_data_dir,
        repulsion_data_dir=args.repulsion_data_dir,
        packing_data_dir=args.packing_data_dir,
        packing_geom_calibration=packing_geom,
        include_repulsion="repulsion" in terms_set,
        include_secondary="secondary" in terms_set,
        include_packing="packing" in terms_set,
        device=device,
    )

    logger.info("Loading checkpoint: %s", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    step = ckpt.get("global_step", "?") if isinstance(ckpt, dict) else "?"
    logger.info("Checkpoint loaded (global_step=%s) | missing=%d  unexpected=%d", step, len(missing), len(unexpected))
    if missing:
        logger.warning("Missing keys (first 8): %s", missing[:8])

    model.to(device).eval()
    return model


def _make_loader(args):
    """Build a simple DataLoader from PDB IDs, using processed_cache when available."""
    import glob

    import torch

    from calphaebm.data.pdb_dataset import PDBSegmentDataset
    from calphaebm.data.pdb_parse import load_pdb_segments

    # Strip entity suffix (e.g. "3NIR_1" → "3NIR") — pdb_parse needs entry IDs
    raw_ids = [l.strip().split()[0] for l in open(args.pdb) if l.strip() and not l.startswith("#")]
    entry_ids = list(dict.fromkeys(x.split("_")[0].upper() for x in raw_ids))  # deduplicate, preserve order

    # ── Try loading from processed_cache first (instant, no network) ──────
    cache_files = sorted(glob.glob("processed_cache/segments_*.pt"))
    if cache_files:
        # Use the largest cache file (most segments = training set cache)
        cache_path = max(cache_files, key=lambda p: __import__("os").path.getsize(p))
        logger.info("Loading segments from cache: %s", cache_path)
        try:
            dataset = PDBSegmentDataset.from_cache(cache_path)
            import random

            segs = list(dataset.segments)
            random.shuffle(segs)
            segments = segs[: args.n_batches * args.batch_size + 64]
            logger.info("Using %d segments from cache (total available: %d)", len(segments), len(segs))
        except Exception as e:
            logger.warning("Cache load failed (%s), falling back to network", e)
            segments = None
    else:
        segments = None

    if segments is None:
        ids = entry_ids[: args.n_ids]
        logger.info("Loading segments from %d entry IDs (network)...", len(ids))
        segments = load_pdb_segments(
            pdb_ids=ids,
            cache_dir=args.cache_dir,
            seg_len=args.seg_len,
            stride=64,
            limit_segments=args.n_batches * args.batch_size + 64,
        )
        logger.info("Loaded %d segments", len(segments))

    class _Loader:
        def __init__(self, segs, bsz):
            self._segs = segs
            self._bsz = bsz

        def __len__(self):
            return len(self._segs) // self._bsz

        def __iter__(self):
            for i in range(0, len(self._segs) - self._bsz + 1, self._bsz):
                batch = self._segs[i : i + self._bsz]
                R = torch.stack([torch.tensor(s["coords"], dtype=torch.float32) for s in batch])
                seq = torch.stack([torch.tensor(s["seq"], dtype=torch.long) for s in batch])
                R = R - R.mean(dim=1, keepdim=True)
                yield R, seq, "", ""  # pdb_id / chain_id stubs

    return _Loader(segments, args.batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# Mode runners
# ─────────────────────────────────────────────────────────────────────────────


def _run_local(args, device) -> int:
    from calphaebm.training.validation.local_validator import LocalValidator

    model = _load_model(args, device)
    loader = _make_loader(args)

    # Log local subterm weights
    if hasattr(model, "local") and model.local is not None:
        loc = model.local
        ttw = loc.theta_theta_weight.item() if hasattr(loc, "theta_theta_weight") else float("nan")
        dpw = loc.delta_phi_weight.item() if hasattr(loc, "delta_phi_weight") else float("nan")
        logger.info("Local weights  θθ=%.4f  Δφ=%.4f", ttw, dpw)

    ckpt_step = _ckpt_step(args)

    validator = LocalValidator(model=model, device=device)

    noise_levels = args.noise  # list[float] or None

    if noise_levels is None or len(noise_levels) > 1:
        # ── Sweep mode (default: 4 levels) ───────────────────────────────
        levels = tuple(noise_levels) if noise_levels else (0.05, 0.10, 0.20, 0.30)  # DSM range
        validator.sweep(
            val_loader=loader,
            noise_levels=levels,
            n_batches=max(5, args.n_batches // 2),
            n_corruptions_per_batch=args.n_corruptions,
            step=ckpt_step,
        )
    else:
        # ── Single-level mode ─────────────────────────────────────────────
        sigma = noise_levels[0]
        metrics = validator.validate(
            val_loader=loader,
            n_batches=args.n_batches,
            noise_scale=sigma,
            n_corruptions_per_batch=args.n_corruptions,
            step=ckpt_step,
        )
        validator.log_validation(metrics)
        print()
        if metrics.gap_mean > 0 and metrics.gap_success_rate >= 0.70:
            print(f"✓  Local term healthy   gap={metrics.gap_mean:+.3f}  success={metrics.gap_success_rate*100:.0f}%")
        elif metrics.gap_mean > 0:
            print(f"⚠  Local term marginal  gap={metrics.gap_mean:+.3f}  success={metrics.gap_success_rate*100:.0f}%")
        else:
            print(f"✗  Local term INVERTED  gap={metrics.gap_mean:+.3f}  success={metrics.gap_success_rate*100:.0f}%")
        print()
    return 0


def _run_behavior(args, device) -> int:
    from calphaebm.training.validation.behavior import BehaviorValidator

    model = _load_model(args, device)
    validator = BehaviorValidator(model=model, device=device)

    # helix vs extended (full model)
    helix_gap = validator.validate_helix_vs_random(
        length=args.length,
        helix_noise=args.helix_noise,
        random_noise=args.random_noise,
    )
    # secondary term in isolation
    ss_gap = validator.validate_secondary_term(
        length=args.length,
        helix_noise=args.helix_noise,
        random_noise=args.random_noise,
    )

    print()
    print(f"  Full-model helix gap   (E_extended − E_helix): {helix_gap:+.3f}  {'✓' if helix_gap > 0 else '✗'}")
    print(f"  Secondary-only ss gap  (E_extended − E_helix): {ss_gap:+.3f}   {'✓' if ss_gap > 0 else '✗'}")
    print()
    return 0


def _run_generation(args, device) -> int:
    from calphaebm.training.validation.generation import GenerationValidator

    model = _load_model(args, device)
    loader = _make_loader(args)

    validator = GenerationValidator(
        model=model,
        device=device,
        n_steps=args.langevin_steps,
        step_size=args.step_size,
    )
    results = validator.validate(
        val_loader=loader,
        max_samples=args.n_batches * args.batch_size,
        temperature=args.temperature,
    )

    print()
    if results.get("valid"):
        rmsd = results.get("rmsd_mean", float("nan"))
        dE = results.get("energy_delta_mean", float("nan"))
        rama = results.get("ramachandran_correlation", float("nan"))
        print(f"  RMSD mean:              {rmsd:.3f} Å")
        print(f"  Energy delta (final−init): {dE:+.3f}")
        print(f"  Ramachandran correlation:  {rama:.3f}")
    else:
        reason = results.get("failure_reason", "unknown")
        print(f"✗  Generation validation failed: {reason}")
    print()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────


def _run_secondary(args, device) -> int:
    import torch

    from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
    from calphaebm.utils.math import wrap_to_pi

    model = _load_model(args, device)
    loader = _make_loader(args)
    if model.secondary is None:
        logger.error("No secondary term in checkpoint.")
        return 1

    term = model.secondary
    gate = model.gate_secondary.item() if hasattr(model, "gate_secondary") else 1.0
    # Internal learned weights
    lam_ram = term.lambda_ram.item() if hasattr(term, "lambda_ram") else float("nan")
    lam_tp = term.lambda_theta_phi.item() if hasattr(term, "lambda_theta_phi") else float("nan")
    lam_pp = term.lambda_phi_phi.item() if hasattr(term, "lambda_phi_phi") else float("nan")
    step = _ckpt_step(args)
    noise_levels = tuple(args.noise) if args.noise else (0.05, 0.10, 0.20, 0.30)  # DSM range
    n_b = max(5, args.n_batches // 2)

    logger.info("\n%s", "=" * 60)
    logger.info("SECONDARY VALIDATION SWEEP  (step=%s  outer_gate=%.3f)", step or "?", gate)
    logger.info("  λ_ram=%.4f  λ_θφ=%.4f  λ_φφ=%.4f", lam_ram, lam_tp, lam_pp)
    logger.info("%s", "=" * 60)
    logger.info("  %-10s  %-8s  %-8s  %-8s  %-8s", "σ (rad)", "σ (°)", "gap", "p10", "p90")
    logger.info("  %s", "-" * 50)

    gaps = []
    for sigma in noise_levels:
        vals = []
        for i, batch in enumerate(loader):
            if i >= n_b:
                break
            R, seq = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                E0 = term(R, seq)
                th, ph = coords_to_internal(R)
                R_neg = nerf_reconstruct(
                    th + torch.randn_like(th) * sigma,
                    wrap_to_pi(ph + torch.randn_like(ph) * sigma),
                    extract_anchor(R),
                    bond=3.8,
                )
                vals.append(term(R_neg, seq) - E0)
        g = torch.cat(vals)
        gaps.append(g.mean().item())
        logger.info(
            "  %-10.2f  %-8.1f  %-8.4f  %-8.4f  %-8.4f  %s",
            sigma,
            sigma * 57.296,
            g.mean().item(),
            g.quantile(0.10).item(),
            g.quantile(0.90).item(),
            "✓" if g.mean().item() > 0 else "✗",
        )

    logger.info("  %s", "-" * 50)
    if gaps[0] > 1e-9:
        logger.info(
            "  Sensitivity ratio: gap(%.2f)/gap(%.2f) = %.1fx  (quadratic: %.1fx)",
            noise_levels[-1],
            noise_levels[0],
            gaps[-1] / gaps[0],
            (noise_levels[-1] / noise_levels[0]) ** 2,
        )
    logger.info("%s\n", "=" * 60)
    return 0


def _run_repulsion(args, device) -> int:
    import torch

    from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
    from calphaebm.utils.math import wrap_to_pi

    model = _load_model(args, device)
    loader = _make_loader(args)
    if model.repulsion is None:
        logger.error("No repulsion term in checkpoint.")
        return 1

    term = model.repulsion
    gate = model.gate_repulsion.item() if hasattr(model, "gate_repulsion") else 1.0
    lam = term.lambda_rep.item() if hasattr(term, "lambda_rep") else float("nan")
    step = _ckpt_step(args)

    logger.info("\n%s", "=" * 60)
    logger.info("REPULSION VALIDATION  (step=%s  gate=%.3f  λ_rep=%.4f)", step or "?", gate, lam)
    logger.info("%s", "=" * 60)

    logger.info("  Compression test (scale toward centroid → tighter packing → higher E):")
    logger.info("  %-8s  %-8s  %-8s  %-8s", "scale", "gap", "p10", "p90")
    logger.info("  %s", "-" * 36)
    for scale in args.compress:
        vals = []
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            R, seq = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                c = R.mean(dim=1, keepdim=True)
                vals.append(term(c + scale * (R - c), seq) - term(R, seq))
        g = torch.cat(vals)
        logger.info(
            "  %-8.2f  %-8.4f  %-8.4f  %-8.4f  %s",
            scale,
            g.mean().item(),
            g.quantile(0.10).item(),
            g.quantile(0.90).item(),
            "✓" if g.mean().item() > 0 else "✗",
        )

    logger.info("  IC-noise test (angle perturbation changes global contacts — expect moderate gap):")
    logger.info("  %-10s  %-8s", "σ (rad)", "gap")
    logger.info("  %s", "-" * 22)
    for sigma in (0.05, 0.10, 0.20, 0.30):  # DSM sigma range
        vals = []
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            R, seq = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                th, ph = coords_to_internal(R)
                R_neg = nerf_reconstruct(
                    th + torch.randn_like(th) * sigma,
                    wrap_to_pi(ph + torch.randn_like(ph) * sigma),
                    extract_anchor(R),
                    bond=3.8,
                )
                vals.append(term(R_neg, seq) - term(R, seq))
        g = torch.cat(vals)
        logger.info("  %-10.2f  %-8.4f", sigma, g.mean().item())

    logger.info("%s\n", "=" * 60)
    return 0


def _run_packing(args, device) -> int:
    import torch

    from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
    from calphaebm.utils.math import wrap_to_pi

    model = _load_model(args, device)
    loader = _make_loader(args)
    if model.packing is None:
        logger.error("No packing term in checkpoint.")
        return 1

    term = model.packing
    gate = model.gate_packing.item() if hasattr(model, "gate_packing") else 1.0
    lam = term.lambda_pack.item() if hasattr(term, "lambda_pack") else float("nan")
    step = _ckpt_step(args)

    logger.info("\n%s", "=" * 60)
    logger.info("PACKING VALIDATION  (step=%s  gate=%.3f  λ_pack=%.4f)", step or "?", gate, lam)
    logger.info("%s", "=" * 60)

    noise_levels = tuple(args.noise) if args.noise else (0.05, 0.10, 0.20, 0.30)  # DSM range

    logger.info("  IC-noise geometry test:")
    logger.info("  %-10s  %-8s  %-8s  %-8s", "σ (rad)", "gap", "p10", "p90")
    logger.info("  %s", "-" * 38)
    for sigma in noise_levels:
        vals = []
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            R, seq = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                th, ph = coords_to_internal(R)
                R_neg = nerf_reconstruct(
                    th + torch.randn_like(th) * sigma,
                    wrap_to_pi(ph + torch.randn_like(ph) * sigma),
                    extract_anchor(R),
                    bond=3.8,
                )
                vals.append(term(R_neg, seq) - term(R, seq))
        g = torch.cat(vals)
        logger.info(
            "  %-10.2f  %-8.4f  %-8.4f  %-8.4f  %s",
            sigma,
            g.mean().item(),
            g.quantile(0.10).item(),
            g.quantile(0.90).item(),
            "✓" if g.mean().item() > 0 else "✗",
        )

    logger.info("  Sequence shuffle test (native seq should fit geometry better than random):")
    vals = []
    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        R, seq = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            B, L = seq.shape
            idx = torch.stack([torch.randperm(L, device=device) for _ in range(B)])
            vals.append(term(R, seq.gather(1, idx)) - term(R, seq))
    g = torch.cat(vals)
    logger.info(
        "  seq-shuffle gap=%.4f  p10=%.4f  p90=%.4f  %s",
        g.mean().item(),
        g.quantile(0.10).item(),
        g.quantile(0.90).item(),
        "✓" if g.mean().item() > 0 else "✗",
    )
    logger.info("  (positive = native sequence fits geometry better than shuffled)")

    logger.info("%s\n", "=" * 60)
    return 0


def _run_model(args, device) -> int:
    """Combined model validation — all terms, full DSM gap sweep."""
    import torch

    from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
    from calphaebm.utils.math import wrap_to_pi

    model = _load_model(args, device)
    loader = _make_loader(args)
    step = _ckpt_step(args)

    # Summarise what's loaded
    has = lambda t: getattr(model, t, None) is not None
    terms_active = [t for t in ("local", "repulsion", "secondary", "packing") if has(t)]

    # Collect internal lambda values for display
    def _lam(term, attr):
        t = getattr(model, term, None)
        if t is None:
            return float("nan")
        p = getattr(t, attr, None)
        return p.item() if p is not None else float("nan")

    logger.info("\n%s", "=" * 60)
    logger.info("MODEL VALIDATION SWEEP  (step=%s)", step or "?")
    logger.info("  Terms: %s", "  ".join(terms_active))
    logger.info(
        "  λ_local(θθ)=%.4f  λ_local(Δφ)=%.4f", _lam("local", "theta_theta_weight"), _lam("local", "delta_phi_weight")
    )
    logger.info(
        "  λ_rep=%.4f  λ_ram=%.4f  λ_θφ=%.4f  λ_φφ=%.4f  λ_pack=%.4f",
        _lam("repulsion", "lambda_rep"),
        _lam("secondary", "lambda_ram"),
        _lam("secondary", "lambda_theta_phi"),
        _lam("secondary", "lambda_phi_phi"),
        _lam("packing", "lambda_pack"),
    )
    logger.info(
        "  outer gates: local=%.3f  rep=%.3f  ss=%.3f  pack=%.3f",
        model.gate_local.item(),
        model.gate_repulsion.item(),
        model.gate_secondary.item(),
        model.gate_packing.item(),
    )
    logger.info("%s", "=" * 60)
    logger.info("  %-10s  %-8s  %-8s  %-8s  %-8s  %-8s", "σ (rad)", "σ (°)", "gap", "p10", "p90", "success%")
    logger.info("  %s", "-" * 56)

    noise_levels = tuple(args.noise) if args.noise else (0.05, 0.10, 0.20, 0.30)
    gaps = []

    for sigma in noise_levels:
        vals = []
        n_total = 0
        n_pos = 0
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            R, seq = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                E_clean = model(R, seq)
                th, ph = coords_to_internal(R)
                anchor = extract_anchor(R)
                R_neg = nerf_reconstruct(
                    th + torch.randn_like(th) * sigma, wrap_to_pi(ph + torch.randn_like(ph) * sigma), anchor, bond=3.8
                )
                E_neg = model(R_neg, seq)
                gap = E_neg - E_clean
                vals.append(gap)
                n_pos += (gap > 0).sum().item()
                n_total += gap.numel()

        g = torch.cat(vals)
        gaps.append(g.mean().item())
        success = 100.0 * n_pos / max(n_total, 1)
        verdict = "✓" if g.mean().item() > 0 else "✗"
        logger.info(
            "  %-10.2f  %-8.1f  %-8.4f  %-8.4f  %-8.4f  %-8.1f  %s",
            sigma,
            sigma * 57.296,
            g.mean().item(),
            g.quantile(0.10).item(),
            g.quantile(0.90).item(),
            success,
            verdict,
        )

    logger.info("  %s", "-" * 56)
    if gaps[0] > 1e-9:
        ratio_gap = gaps[-1] / gaps[0]
        ratio_sq = (noise_levels[-1] / noise_levels[0]) ** 2
        logger.info(
            "  Sensitivity ratio: gap(%.2f)/gap(%.2f) = %.1fx  (quadratic: %.1fx)",
            noise_levels[-1],
            noise_levels[0],
            ratio_gap,
            ratio_sq,
        )
    logger.info("%s\n", "=" * 60)
    return 0


def run(args) -> int:
    import torch

    device = torch.device(args.device)
    logger.info("validate %s | ckpt=%s | device=%s", args.validate_mode, args.ckpt, device)

    if args.validate_mode == "local":
        return _run_local(args, device)
    elif args.validate_mode == "secondary":
        return _run_secondary(args, device)
    elif args.validate_mode == "repulsion":
        return _run_repulsion(args, device)
    elif args.validate_mode == "packing":
        return _run_packing(args, device)
    elif args.validate_mode == "model":
        return _run_model(args, device)
    elif args.validate_mode == "behavior":
        return _run_behavior(args, device)
    elif args.validate_mode == "generation":
        return _run_generation(args, device)
    else:
        logger.error("Unknown validate mode: %s", args.validate_mode)
        return 1
