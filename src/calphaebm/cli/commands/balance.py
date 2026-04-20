# src/calphaebm/cli/commands/balance.py
"""Lambda balancing command.

Measures force RMS for each energy term at a range of noise scales and
computes correction factors needed to hit the target force ratio:

    local : repulsion : secondary : packing  =  2 : 1 : 1 : 1

Unlike the old balance command (which recommended outer gate values), this
command targets the *internal* lambda parameters of each term, since outer
gates are now frozen at 1.0.

For each term T the correction factor is:

    correction(T)  =  current_rms(T) / target_rms(T)
    new_lambda(T)  =  current_lambda(T) / correction(T)

The command prints:
  - Per-term RMS force at each sigma
  - Summary table: mean RMS, target ratio, correction factor, new lambda
  - Ready-to-paste --init-* CLI flags for the next training run

Optionally patches the checkpoint in-place (--apply-to-ckpt).

Usage:
    calphaebm balance \\
        --pdb 1L2Y \\
        --ckpt checkpoints/run/full/step001500.pt \\
        --sigma 0.1 0.25 0.5 1.0 2.0 \\
        --n-samples 64 \\
        --out balance.json \\
        --apply-to-ckpt checkpoints/run/full/step001500_balanced.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from calphaebm.data.pdb_dataset import PDBSegmentDataset
from calphaebm.models.energy import TotalEnergy, create_total_energy
from calphaebm.utils.logging import get_logger

logger = get_logger()

# Default target force ratios
DEFAULT_TARGET_RATIOS: Dict[str, float] = {
    "local": 2.0,
    "repulsion": 1.0,
    "secondary": 1.0,
    "packing": 1.0,
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "balance",
        description=__doc__,
        help="Measure force scales and compute internal lambda correction factors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # data
    parser.add_argument("--pdb", nargs="+", required=True, help="PDB IDs or file of IDs (one per line)")
    parser.add_argument("--cache-dir", default="./pdb_cache")
    parser.add_argument("--seg-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--limit", type=int, default=1024, help="Max segments to load (default: 1024)")
    parser.add_argument("--n-samples", type=int, default=64, help="Structures used per sigma measurement (default: 64)")

    # measurement — mirrors the DSM training flags
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.05,
        help="Lower bound of log-uniform sigma distribution in Å (default: 0.05)",
    )
    parser.add_argument(
        "--sigma-max", type=float, default=8.0, help="Upper bound of log-uniform sigma distribution in Å (default: 8.0)"
    )
    parser.add_argument(
        "--n-sigma-levels",
        type=int,
        default=20,
        help="Number of sigma levels drawn log-uniformly for measurement (default: 20)",
    )
    parser.add_argument(
        "--n-passes", type=int, default=3, help="Number of independent measurement passes to average (default: 3)"
    )
    parser.add_argument(
        "--target-ratios",
        nargs=4,
        type=float,
        default=None,
        metavar=("LOCAL", "REP", "SS", "PACK"),
        help="Override target force ratios (default: 2 1 1 1)",
    )

    # model / checkpoint
    parser.add_argument("--ckpt", default=None, help="Checkpoint path (uses fresh model if omitted)")
    parser.add_argument("--backbone-data-dir", default="analysis/backbone_geometry/data")
    parser.add_argument("--repulsion-data-dir", default="analysis/repulsion_analysis/data")
    parser.add_argument("--packing-data-dir", default=None)

    # output
    parser.add_argument("--out", default=None, help="Write JSON report to this path")
    parser.add_argument(
        "--apply-to-ckpt",
        default=None,
        help="Write a corrected checkpoint (patches lambda raw parameters) " "to this path.  Requires --ckpt.",
    )

    parser.set_defaults(func=run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_id_lines(path: str) -> List[str]:
    ids: List[str] = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def _parse_pdb_arg(values: List[str]) -> List[str]:
    if len(values) == 1 and os.path.isfile(values[0]):
        raw = _read_id_lines(values[0])
    else:
        raw = values
    seen: set = set()
    out: List[str] = []
    for x in raw:
        key = x.split("_")[0].upper()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _inv_softplus(y: float, eps: float = 1e-6) -> float:
    """softplus^{-1}(y): the raw value r such that softplus(r) ≈ y."""
    y = max(float(y), eps)
    return math.log(math.expm1(y))


# ---------------------------------------------------------------------------
# Force measurement
# ---------------------------------------------------------------------------


def _rms_force(g: torch.Tensor) -> float:
    """RMS gradient magnitude: sqrt(mean_atoms ||g_i||^2).

    g: (B, L, 3)
    """
    return float(g.pow(2).sum(dim=-1).mean().sqrt().item())


def _term_rms_at_sigma(
    model: TotalEnergy,
    R_native: torch.Tensor,  # (B, L, 3)
    seq: torch.Tensor,  # (B, L)
    sigma: float,
) -> Dict[str, float]:
    """RMS of -dE/dR for each term, evaluated on R_native + Gaussian noise."""
    R_noisy = (R_native + sigma * torch.randn_like(R_native)).detach()

    results: Dict[str, float] = {}

    def _compute(name: str, E_fn) -> None:
        Rg = R_noisy.detach().requires_grad_(True)
        try:
            E = E_fn(Rg).sum()
            (g,) = torch.autograd.grad(E, Rg, create_graph=False)
            if torch.isfinite(g).all():
                results[name] = _rms_force(g)
            else:
                logger.warning("Non-finite gradients for %s at sigma=%.2f", name, sigma)
                results[name] = float("nan")
        except Exception as exc:
            logger.warning("Error computing %s forces at sigma=%.2f: %s", name, sigma, exc)
            results[name] = float("nan")

    # Local: measure only learned sub-terms (theta_theta + delta_phi).
    # bond_spring is a fixed buffer excluded from DSM — including it would
    # inflate the local RMS by ~3-20× and cause balance to over-correct all
    # other terms. Zero it temporarily, measure, then restore.
    def _compute_local_learned(r: torch.Tensor) -> torch.Tensor:
        orig = model.local._bond_spring_val.data.clone()
        try:
            model.local._bond_spring_val.data.zero_()
            return model.local(r, seq)
        finally:
            model.local._bond_spring_val.data.copy_(orig)

    _compute("local", _compute_local_learned)
    if model.repulsion is not None:
        _compute("repulsion", lambda r: model.repulsion(r, seq))
    if model.secondary is not None:
        _compute("secondary", lambda r: model.secondary(r, seq))
    if model.packing is not None:
        _compute("packing", lambda r: model.packing(r, seq))

    return results


def measure_force_scales(
    model: TotalEnergy,
    R: torch.Tensor,
    seq: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    n_sigma_levels: int,
    n_samples: int,
) -> Dict[str, Dict[float, float]]:
    """Measure per-term RMS force at sigma levels drawn log-uniformly from [sigma_min, sigma_max].

    Mirrors the DSM training distribution exactly — each level is an independent
    draw from log-Uniform[sigma_min, sigma_max], so the measurement covers the
    same noise range that the model will be trained on.

    Returns rms[term][sigma] = float.
    """
    model.eval()
    B_avail = R.shape[0]
    B = min(n_samples, B_avail)

    # Draw sigma levels log-uniformly, same distribution as DSM
    log_sigmas = torch.empty(n_sigma_levels).uniform_(math.log(sigma_min), math.log(sigma_max))
    sigmas = torch.exp(log_sigmas).tolist()
    sigmas_sorted = sorted(sigmas)  # sorted only for display; measurement uses drawn order

    result: Dict[str, Dict[float, float]] = {}

    for sigma in sigmas:
        logger.info("  sigma=%.3f Å ...", sigma)
        idx = torch.randperm(B_avail, device=R.device)[:B]
        R_sub = R[idx]
        seq_sub = seq[idx]

        rms = _term_rms_at_sigma(model, R_sub, seq_sub, sigma)
        for term, val in rms.items():
            result.setdefault(term, {})[sigma] = val

    return result, sigmas_sorted


# ---------------------------------------------------------------------------
# Correction computation
# ---------------------------------------------------------------------------


def compute_corrections(
    rms_by_term_sigma: Dict[str, Dict[float, float]],
    target_ratios: Dict[str, float],
) -> Dict[str, float]:
    """Compute correction = mean_rms / target_rms for each term.

    A correction > 1 means the term is currently too strong.
    A correction < 1 means it is too weak.
    To rebalance: new_lambda = current_lambda / correction.

    The target scale S is chosen so that:
        target_rms(t) = S * target_ratio(t)
    where S = mean_t( mean_rms(t) / target_ratio(t) ).
    """
    # Mean RMS across sigma levels, ignoring NaN
    mean_rms: Dict[str, float] = {}
    for term, rms_by_sigma in rms_by_term_sigma.items():
        vals = [v for v in rms_by_sigma.values() if math.isfinite(v)]
        mean_rms[term] = sum(vals) / len(vals) if vals else 0.0

    # Unit scale S
    units = [
        mean_rms[t] / target_ratios[t]
        for t in mean_rms
        if t in target_ratios and target_ratios[t] > 0 and mean_rms[t] > 0
    ]
    if not units:
        raise ValueError("No terms with positive mean RMS and target ratio found")
    S = sum(units) / len(units)

    corrections: Dict[str, float] = {}
    for term, rms in mean_rms.items():
        ratio = target_ratios.get(term, 1.0)
        target = S * ratio
        corrections[term] = (rms / target) if target > 0 else 1.0

    return corrections, mean_rms, S


# ---------------------------------------------------------------------------
# Lambda reading and recommendation
# ---------------------------------------------------------------------------


def get_current_lambdas(model: TotalEnergy) -> Dict[str, float]:
    """Read current positive lambda values from each term module."""
    lams: Dict[str, float] = {}

    loc = model.local
    lams["local.bond_spring"] = float(loc.bond_spring.item())  # FIXED buffer — shown for info only, not rebalanced
    lams["local.theta_theta"] = float(loc.theta_theta_weight.item())
    lams["local.delta_phi"] = float(loc.delta_phi_weight.item())

    if model.repulsion is not None and hasattr(model.repulsion, "lambda_rep"):
        lams["repulsion.lambda_rep"] = float(model.repulsion.lambda_rep.item())

    if model.secondary is not None:
        ss = model.secondary
        for attr, key in [
            ("lambda_ram", "secondary.lambda_ram"),
            ("lambda_theta_phi", "secondary.lambda_theta_phi"),
            ("lambda_phi_phi", "secondary.lambda_phi_phi"),
        ]:
            if hasattr(ss, attr):
                lams[key] = float(getattr(ss, attr).item())

    if model.packing is not None and hasattr(model.packing, "lambda_pack"):
        lams["packing.lambda_pack"] = float(model.packing.lambda_pack.item())

    return lams


def recommend_new_lambdas(
    corrections: Dict[str, float],
    current_lambdas: Dict[str, float],
) -> Dict[str, float]:
    """new_lambda(k) = current_lambda(k) / correction(term(k))."""
    new: Dict[str, float] = {}

    local_c = corrections.get("local", 1.0)
    # bond_spring is a fixed physical buffer — skip rebalancing, only adjust learned weights
    for key in ("local.theta_theta", "local.delta_phi"):
        if key in current_lambdas:
            new[key] = current_lambdas[key] / local_c

    rep_c = corrections.get("repulsion", 1.0)
    if "repulsion.lambda_rep" in current_lambdas:
        new["repulsion.lambda_rep"] = current_lambdas["repulsion.lambda_rep"] / rep_c

    ss_c = corrections.get("secondary", 1.0)
    for key in ("secondary.lambda_ram", "secondary.lambda_theta_phi", "secondary.lambda_phi_phi"):
        if key in current_lambdas:
            new[key] = current_lambdas[key] / ss_c

    pack_c = corrections.get("packing", 1.0)
    if "packing.lambda_pack" in current_lambdas:
        new["packing.lambda_pack"] = current_lambdas["packing.lambda_pack"] / pack_c

    return new


def format_cli_flags(new_lambdas: Dict[str, float]) -> str:
    lines: List[str] = []
    mapping = [
        # bond_spring omitted — fixed physical buffer, set via --init-bond-spring but not rebalanced
        ("local.theta_theta", "--init-theta-theta-weight"),
        ("local.delta_phi", "--init-delta-phi-weight"),
        ("repulsion.lambda_rep", "--init-lambda-rep"),
        ("secondary.lambda_ram", "--init-lambda-ram"),
        ("secondary.lambda_theta_phi", "--init-lambda-theta-phi"),
        ("secondary.lambda_phi_phi", "--init-lambda-phi-phi"),
        ("packing.lambda_pack", "--init-lambda-pack"),
    ]
    for key, flag in mapping:
        if key in new_lambdas:
            lines.append(f"  {flag} {new_lambdas[key]:.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoint patching
# ---------------------------------------------------------------------------


def patch_checkpoint(
    ckpt_path: str,
    new_lambdas: Dict[str, float],
    out_path: str,
    device: torch.device,
) -> None:
    """Rewrite raw (pre-softplus) parameters in checkpoint to match new_lambdas.

    Handles three cases:
      1. New key exists in checkpoint → overwrite directly.
      2. Old key exists (pre-rename) → overwrite under the old name so the
         checkpoint remains loadable by the old architecture, AND insert under
         the new name for the new architecture.
      3. Key is absent entirely → insert it fresh (e.g. lambda_rep, lambda_ram
         which didn't exist in old checkpoints at all).
    """
    # Maps our "lambda key" -> new state_dict key
    # NOTE: "local.bond_spring" is intentionally absent — it is now a FIXED
    # buffer (_bond_spring_val), not a trainable parameter. Balance must not
    # patch it. Use --init-bond-spring at training time to set it.
    new_key_map = {
        "local.theta_theta": "local._theta_theta_weight_raw",
        "local.delta_phi": "local._delta_phi_weight_raw",
        "repulsion.lambda_rep": "repulsion._lambda_rep_raw",
        "secondary.lambda_ram": "secondary._lambda_ram_raw",
        "secondary.lambda_theta_phi": "secondary._lambda_theta_phi_raw",
        "secondary.lambda_phi_phi": "secondary._lambda_phi_phi_raw",
        "packing.lambda_pack": "packing._lambda_pack_raw",
    }
    # Old key names that may exist in pre-rename checkpoints
    old_key_map = {
        "secondary.lambda_ram": "secondary._ram_weight_raw",
        "secondary.lambda_theta_phi": "secondary._theta_phi_weight_raw",
        "secondary.lambda_phi_phi": "secondary._phi_phi_weight_raw",
    }

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)

    patched = 0
    inserted = 0
    for src_key, new_val in new_lambdas.items():
        if src_key.endswith("_info"):
            continue
        new_state_key = new_key_map.get(src_key)
        if new_state_key is None:
            continue

        raw = _inv_softplus(new_val)
        tensor = torch.tensor(raw, dtype=torch.float32)

        if new_state_key in state:
            # Case 1: new key already present
            state[new_state_key] = tensor
            logger.info("  Updated  %-50s new_lambda=%.6f  raw=%.6f", new_state_key, new_val, raw)
            patched += 1
        else:
            # Case 3: key absent — insert it
            state[new_state_key] = tensor
            logger.info("  Inserted %-50s new_lambda=%.6f  raw=%.6f", new_state_key, new_val, raw)
            inserted += 1

        # Case 2: also patch old key if it exists (keeps checkpoint usable by old code)
        old_state_key = old_key_map.get(src_key)
        if old_state_key and old_state_key in state:
            state[old_state_key] = tensor
            logger.info("  Patched old key %-44s raw=%.6f", old_state_key, raw)

    if "model_state" in ckpt:
        ckpt["model_state"] = state
    else:
        ckpt = state

    torch.save(ckpt, out_path)
    logger.info("Saved patched checkpoint (%d updated, %d inserted) -> %s", patched, inserted, out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── validate ──────────────────────────────────────────────────────────────
    if args.apply_to_ckpt:
        if not args.ckpt:
            logger.error("--apply-to-ckpt requires --ckpt")
            return 1
        # Resolve both paths so symlinks / relative paths don't fool us
        src = os.path.realpath(args.ckpt)
        dst = os.path.realpath(args.apply_to_ckpt)
        if src == dst:
            logger.error(
                "--apply-to-ckpt must be a different path from --ckpt.\n"
                "  source : %s\n"
                "  output : %s\n"
                "Refusing to overwrite the source checkpoint in-place — "
                "pass a distinct output path, e.g. step007000_balanced.pt",
                args.ckpt,
                args.apply_to_ckpt,
            )
            return 1

    # target ratios
    if args.target_ratios is not None:
        lr, rr, sr, pr = args.target_ratios
        target_ratios = {"local": lr, "repulsion": rr, "secondary": sr, "packing": pr}
    else:
        target_ratios = dict(DEFAULT_TARGET_RATIOS)

    logger.info(
        "Target force ratios — local:%.1f  rep:%.1f  ss:%.1f  pack:%.1f",
        target_ratios["local"],
        target_ratios["repulsion"],
        target_ratios["secondary"],
        target_ratios["packing"],
    )

    # data
    pdb_ids = _parse_pdb_arg(args.pdb)
    if not pdb_ids:
        logger.error("No valid PDB IDs found")
        return 1

    logger.info("Loading dataset...")
    ds = PDBSegmentDataset(
        pdb_ids=pdb_ids,
        cache_dir=args.cache_dir,
        seg_len=args.seg_len,
        stride=args.stride,
        limit_segments=args.limit,
    )
    if len(ds) == 0:
        logger.error("No segments found")
        return 1

    dl = DataLoader(ds, batch_size=args.n_samples, shuffle=True, num_workers=0, drop_last=False)
    batch = next(iter(dl))
    R_native = batch[0].to(device)
    seq = batch[1].to(device)
    logger.info("Loaded %d structures × L=%d", R_native.shape[0], R_native.shape[1])

    # model
    logger.info("Building model...")
    model = create_total_energy(
        backbone_data_dir=args.backbone_data_dir,
        repulsion_data_dir=args.repulsion_data_dir,
        packing_data_dir=args.packing_data_dir,
        device=device,
        include_repulsion=True,
        include_secondary=True,
        include_packing=True,
    )

    if args.ckpt:
        logger.info("Loading checkpoint: %s", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected[:5])
    else:
        logger.info("No checkpoint — using freshly initialised model")

    model.eval()

    # current lambdas
    current_lambdas = get_current_lambdas(model)
    print("\nCurrent internal lambda values:")
    for k, v in sorted(current_lambdas.items()):
        print(f"  {k:<45} {v:.6f}")

    # measure — average across n_passes independent sigma draws to reduce noise
    sigma_min = float(args.sigma_min)
    sigma_max = float(args.sigma_max)
    n_sigma_levels = int(args.n_sigma_levels)
    n_passes = int(args.n_passes)
    if sigma_min >= sigma_max:
        logger.error("--sigma-min (%.3f) must be < --sigma-max (%.3f)", sigma_min, sigma_max)
        return 1

    print(
        f"\nMeasuring RMS force: {n_passes} passes × {n_sigma_levels} sigma levels "
        f"log-uniformly from [{sigma_min}, {sigma_max}] Å ..."
    )

    # Accumulate mean_rms across passes, then average
    all_pass_mean_rms: Dict[str, List[float]] = {}
    last_rms_by_term_sigma = None
    last_sigmas_sorted = None

    for pass_idx in range(n_passes):
        logger.info("Pass %d/%d ...", pass_idx + 1, n_passes)
        rms_by_term_sigma, sigmas_sorted = measure_force_scales(
            model,
            R_native,
            seq,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            n_sigma_levels=n_sigma_levels,
            n_samples=args.n_samples,
        )
        last_rms_by_term_sigma = rms_by_term_sigma
        last_sigmas_sorted = sigmas_sorted

        for term, rms_by_sigma in rms_by_term_sigma.items():
            vals = [v for v in rms_by_sigma.values() if math.isfinite(v)]
            mean_val = sum(vals) / len(vals) if vals else 0.0
            all_pass_mean_rms.setdefault(term, []).append(mean_val)

    # Average mean_rms across passes
    avg_mean_rms: Dict[str, float] = {term: sum(vals) / len(vals) for term, vals in all_pass_mean_rms.items()}
    std_mean_rms: Dict[str, float] = {
        term: (sum((v - avg_mean_rms[term]) ** 2 for v in vals) / max(len(vals), 1)) ** 0.5
        for term, vals in all_pass_mean_rms.items()
    }

    # Print per-pass summary
    if n_passes > 1:
        print(f"\nPer-pass mean RMS (averaged over sigma levels):")
        print(
            f"  {'term':>12}"
            + "".join(f"  {'pass'+str(i+1):>10}" for i in range(n_passes))
            + f"  {'avg':>10}  {'std':>8}"
        )
        for term in avg_mean_rms:
            row = f"  {term:>12}"
            for v in all_pass_mean_rms[term]:
                row += f"  {v:>10.6f}"
            row += f"  {avg_mean_rms[term]:>10.6f}  {std_mean_rms[term]:>8.6f}"
            print(row)

    # Use last pass for sigma table display
    rms_by_term_sigma = last_rms_by_term_sigma
    sigmas_sorted = last_sigmas_sorted

    # per-sigma table from last pass (sorted for readability)
    terms = list(rms_by_term_sigma.keys())
    print("\nPer-term RMS |∇E| at each sigma (last pass, sorted):")
    header = f"  {'sigma':>8}" + "".join(f"  {t:>12}" for t in terms)
    print(header)
    for s in sigmas_sorted:
        row = f"  {s:>8.3f}"
        for t in terms:
            val = rms_by_term_sigma[t].get(s, float("nan"))
            row += f"  {val:>12.6f}"
        print(row)

    # corrections — use averaged mean_rms instead of single-pass
    # Temporarily override compute_corrections to use avg_mean_rms directly
    units = [
        avg_mean_rms[t] / target_ratios[t]
        for t in avg_mean_rms
        if t in target_ratios and target_ratios[t] > 0 and avg_mean_rms[t] > 0
    ]
    S = sum(units) / len(units)
    corrections: Dict[str, float] = {}
    for term, rms in avg_mean_rms.items():
        ratio = target_ratios.get(term, 1.0)
        target = S * ratio
        corrections[term] = (rms / target) if target > 0 else 1.0
    mean_rms = avg_mean_rms

    print(f"\nUnit scale S = {S:.6f}")
    print(f"\n  {'term':>12}  {'mean_rms':>10}  {'target_rms':>10}  {'correction':>11}  interpretation")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*30}")
    for t in terms:
        corr = corrections.get(t, 1.0)
        tgt = S * target_ratios.get(t, 1.0)
        mrms = mean_rms.get(t, 0.0)
        if corr > 1.05:
            interp = f"too strong by {corr:.2f}×"
        elif corr < 0.95:
            interp = f"too weak by {1/corr:.2f}×"
        else:
            interp = "balanced"
        print(f"  {t:>12}  {mrms:>10.6f}  {tgt:>10.6f}  {corr:>11.4f}  {interp}")

    # new lambdas
    new_lambdas = recommend_new_lambdas(corrections, current_lambdas)

    print("\nRecommended lambda updates:")
    print(f"  {'parameter':<45}  {'current':>10}  {'new':>10}  {'×':>6}")
    print(f"  {'-'*45}  {'-'*10}  {'-'*10}  {'-'*6}")
    for k in sorted(new_lambdas.keys()):
        if k.endswith("_info"):
            print(f"  {k:<45}  {'—':>10}  {'—':>10}  {new_lambdas[k]:>5.3f}×")
        else:
            cur = current_lambdas.get(k, float("nan"))
            nw = new_lambdas[k]
            chg = nw / cur if cur > 0 else float("nan")
            print(f"  {k:<45}  {cur:>10.6f}  {nw:>10.6f}  {chg:>5.3f}×")

    cli = format_cli_flags(new_lambdas)
    if cli:
        print("\nCLI flags (paste into next training run):")
        print(cli)

    # optional checkpoint patch
    if args.apply_to_ckpt:
        print(f"\nPatching checkpoint -> {args.apply_to_ckpt}")
        patch_checkpoint(args.ckpt, new_lambdas, args.apply_to_ckpt, device)

    # JSON
    if args.out:
        report = {
            "target_ratios": target_ratios,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "n_sigma_levels": n_sigma_levels,
            "n_passes": n_passes,
            "unit_scale_S": S,
            "per_pass_mean_rms": all_pass_mean_rms,
            "avg_mean_rms": avg_mean_rms,
            "std_mean_rms": std_mean_rms,
            "rms_by_term_sigma": {t: {str(s): v for s, v in rms.items()} for t, rms in rms_by_term_sigma.items()},
            "mean_rms": mean_rms,
            "corrections": corrections,
            "current_lambdas": current_lambdas,
            "new_lambdas": {k: v for k, v in new_lambdas.items() if not k.endswith("_info")},
        }
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote report to %s", args.out)

    return 0
