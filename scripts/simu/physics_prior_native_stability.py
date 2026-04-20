#!/usr/bin/env python
"""Physics prior native basin stability — ablation sweep.

Tests whether the untrained physics prior can hold proteins in their native
basins across a range of betas (inverse temperatures) and across ablations
of which energy terms are active. Uses the SAME canonical components as
training and model_test.py:

  - build_model         (calphaebm.cli.commands.train.model_builder)
  - load_structures     (calphaebm.evaluation.core_evaluation)
  - lbfgs_minimize      (calphaebm.simulation.minimize)
  - get_simulator(mala) (calphaebm.simulation.backends)
  - native_contact_set, q_smooth, rmsd_kabsch (calphaebm.evaluation.metrics)

For each protein × each β:
  - Build a fresh TotalEnergy model (no checkpoint) with physics_prior=True
  - Optionally L-BFGS minimize starting from native (IC space)
  - Run N_STEPS MALA production via canonical simulator
  - Record Q, RMSD, dRMSD, Rg/Rg*, accept rate, E trajectory

Usage
-----
    python scripts/simu/physics_prior_native_stability.py \\
        --pdb 1FME \\
        --minimize \\
        --betas 5,50,100 \\
        --out runs/physics_prior/full

Term toggles (default: ram,hb,rep,rg,pack):
    --terms ram,hb,rep,rg,pack    # full prior (default)
    --terms ram,hb,rep,rg         # no packing
    --terms ram,rep,rg            # no H-bonds (ablation)
    --terms ram,rep               # ram + steric only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from calphaebm.models.energy import create_total_energy
from calphaebm.utils.constants import EMB_DIM
from calphaebm.utils.logging import get_logger

logger = get_logger()


ALL_TERMS = {"ram", "hb", "rep", "rg", "pack", "local"}


def parse_terms(terms_str: str) -> set[str]:
    terms = {t.strip().lower() for t in terms_str.split(",") if t.strip()}
    unknown = terms - ALL_TERMS
    if unknown:
        raise ValueError(f"Unknown terms: {unknown}. Valid: {ALL_TERMS}")
    return terms


def build_physics_prior_model(
    terms: set[str],
    device: torch.device,
    secondary_data_dir: str,
    repulsion_data_dir: str,
    packing_data_dir: str,
    backbone_data_dir: str,
    coord_n_star_file: str | None = None,
) -> torch.nn.Module:
    """Build a fresh model for physics prior evaluation.

    physics_prior=True zeros: (a) secondary logits, (b) local gate.
    --terms selects which module groups to include.  Within secondary,
    individual sub-terms (ram vs hb) are enabled by zeroing unused lambdas.

    REQUIRED paths — script hard-fails if any are missing. No defaults,
    no fallbacks. The caller must pass correct paths for every term they
    request.
    """
    include_secondary = ("ram" in terms) or ("hb" in terms)
    include_repulsion = "rep" in terms
    include_packing = ("pack" in terms) or ("rg" in terms)
    rg_only = ("rg" in terms) and ("pack" not in terms)

    # ── Strict path validation — fail loud on anything missing ─────────
    required_dirs: list[tuple[str, str, bool]] = [
        ("backbone_data_dir", backbone_data_dir, True),  # always needed
        ("secondary_data_dir", secondary_data_dir, include_secondary),
        ("repulsion_data_dir", repulsion_data_dir, include_repulsion),
        ("packing_data_dir", packing_data_dir, include_packing),
    ]
    for name, path, required in required_dirs:
        if required and not Path(path).is_dir():
            raise FileNotFoundError(
                f"Required directory {name}={path!r} does not exist. "
                f"Pass the correct --{name.replace('_', '-')} explicitly."
            )

    if include_packing:
        if coord_n_star_file is None:
            raise ValueError(
                "--terms includes 'pack' or 'rg' but --coord-n-star-file is None. " "Pass a concrete file path."
            )
        if not Path(coord_n_star_file).is_file():
            raise FileNotFoundError(
                f"coord_n_star_file={coord_n_star_file!r} does not exist. "
                f"Pass the correct --coord-n-star-file explicitly."
            )

    # ── Load n_star data for packing ───────────────────────────────────
    packing_extra = {}
    if include_packing:
        with open(coord_n_star_file) as f:
            nstar = json.load(f)
        required_keys = [
            ("group_assignment", "group_assignment"),
            ("n_group_mean_list", "n_group_mean"),
            ("n_group_std_list", "n_group_std"),
            ("n_group_lo_list", "n_group_lo"),
            ("n_group_hi_list", "n_group_hi"),
            ("rho_group_fits", "rho_group_fits"),
            ("rho_group_sigma", "rho_group_sigma"),
            ("rho_group_lo", "rho_group_lo"),
            ("rho_group_hi", "rho_group_hi"),
        ]
        missing = []
        for k_json, k_extra in required_keys:
            v = nstar.get(k_json)
            if v is None:
                missing.append(k_json)
            else:
                packing_extra[k_extra] = v
        if missing:
            raise KeyError(
                f"coord_n_star_file {coord_n_star_file!r} is missing required keys: "
                f"{missing}. This file is out of date or corrupt."
            )
        logger.info("Loaded coord_n_star from %s", coord_n_star_file)

    kwargs = dict(
        backbone_data_dir=backbone_data_dir,
        secondary_data_dir=secondary_data_dir,
        repulsion_data_dir=repulsion_data_dir,
        packing_data_dir=packing_data_dir,
        device=device,
        emb_dim=EMB_DIM,
        hidden_dims=(128, 128),
        include_repulsion=include_repulsion,
        include_secondary=include_secondary,
        include_packing=include_packing,
        physics_prior=True,
        packing_extra=packing_extra if packing_extra else None,
    )

    if rg_only:
        kwargs["coord_lambda"] = 0.0

    model = create_total_energy(**kwargs)

    # Post-construction: zero sub-lambdas based on which sub-terms are active
    with torch.no_grad():
        if rg_only and model.packing is not None:
            if hasattr(model.packing, "burial") and hasattr(model.packing.burial, "_lambda_hp_raw"):
                model.packing.burial._lambda_hp_raw.fill_(-20.0)
            if hasattr(model.packing, "_lambda_rho_raw"):
                model.packing._lambda_rho_raw.fill_(-20.0)
            logger.info("rg_only: zeroed λ_hp and λ_ρ (keeping only E_rg_penalty)")

        if include_secondary and ("hb" not in terms) and model.secondary is not None:
            model.secondary.hb_helix._lambda_raw.fill_(-20.0)
            model.secondary.hb_sheet._lambda_raw.fill_(-20.0)
            logger.info("ram-only: zeroed λ_α and λ_β")

        if include_secondary and ("hb" in terms) and ("ram" not in terms) and model.secondary is not None:
            model.secondary.lambda_ram.fill_(-20.0)
            logger.info("hb-only: zeroed λ_ram")

    return model


# ──────────────────────────────────────────────────────────────────────────
# Simulation & metrics — canonical imports from calphaebm
# ──────────────────────────────────────────────────────────────────────────
# Reuses the EXACT same simulator, minimizer, and metrics functions that
# train, evaluate, and model_test use. No reimplementation, no drift.


def minimize_structure(model, R_init, seq_tensor, lengths):
    """Energy minimize via L-BFGS in IC space (quadratic convergence).

    Returns (R_min, E_min, n_steps, drmsd, delta_E). Consistent with
    model_test.py, eval_subprocess.py, negative_collector.py.
    """
    from calphaebm.simulation.minimize import lbfgs_minimize

    L = int(lengths[0].item())
    result = lbfgs_minimize(model, R_init, seq_tensor, lengths=lengths)
    R_min = result["R_min"]
    E_min = result["E_minimized"]
    n_steps = result["min_steps"]
    delta_E = result["E_relax"]  # E_min - E_pdb

    coords_init = R_init[0, :L].detach().numpy()
    coords_min = R_min[0, :L].detach().numpy()
    d_init = np.sqrt(((coords_init[:, None] - coords_init[None, :]) ** 2).sum(-1))
    d_min = np.sqrt(((coords_min[:, None] - coords_min[None, :]) ** 2).sum(-1))
    triu = np.triu_indices(L, k=1)
    drmsd = float(np.sqrt(np.mean((d_init[triu] - d_min[triu]) ** 2)))

    return R_min, E_min, n_steps, drmsd, delta_E


def compute_metrics(coords, native_coords, ni, nj, d0, rg_native):
    """Compute Q, RMSD, Rg/Rg*, dRMSD via canonical metric functions."""
    from calphaebm.evaluation.metrics import q_smooth, rmsd_kabsch

    q = q_smooth(coords, ni, nj, d0)
    rmsd = rmsd_kabsch(coords, native_coords)
    rg = float(np.sqrt(((coords - coords.mean(0)) ** 2).sum(1).mean()))
    d_nat = np.sqrt(((native_coords[:, None] - native_coords[None, :]) ** 2).sum(-1))
    d_cur = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
    triu = np.triu_indices(len(coords), k=1)
    drmsd = float(np.sqrt(np.mean((d_nat[triu] - d_cur[triu]) ** 2)))
    return {
        "q": q,
        "rmsd": rmsd,
        "rg": rg,
        "rg_ratio": rg / max(rg_native, 1e-6),
        "drmsd": drmsd,
    }


# ──────────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────────


def run_sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    terms = parse_terms(args.terms)
    logger.info("Active terms: %s", sorted(terms))

    betas = [float(b) for b in args.betas.split(",")]
    logger.info("Betas: %s", betas)

    model = build_physics_prior_model(
        terms=terms,
        device=device,
        secondary_data_dir=args.secondary_data_dir,
        repulsion_data_dir=args.repulsion_data_dir,
        packing_data_dir=args.packing_data_dir,
        backbone_data_dir=args.backbone_data_dir,
        coord_n_star_file=args.coord_n_star_file,
    )
    model.eval()
    model.to(device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    terms_tag = "_".join(sorted(terms))
    results = []

    # Load structures via canonical loader — accepts file path OR list of PDB IDs
    from calphaebm.evaluation.core_evaluation import load_structures

    # args.pdb is always a list from nargs="+"
    # If single entry and it's a file that exists, treat as file path; else as PDB ID list
    if len(args.pdb) == 1 and Path(args.pdb[0]).is_file():
        pdb_source = args.pdb[0]
        logger.info("Loading structures from file: %s", pdb_source)
    else:
        pdb_source = list(args.pdb)
        logger.info(
            "Loading structures from %d PDB IDs: %s",
            len(pdb_source),
            pdb_source[:5] + (["..."] if len(pdb_source) > 5 else []),
        )
    structures = load_structures(
        pdb_source=pdb_source,
        cache_dir=args.cache_dir,
        n_samples=args.n_samples,
        max_len=args.max_len,
        min_len=args.min_len,
    )
    logger.info("Loaded %d structures", len(structures))
    if not structures:
        raise RuntimeError("No structures loaded — check --pdb and --cache-dir")

    # Import canonical sim/metrics once per run
    from calphaebm.evaluation.metrics import native_contact_set
    from calphaebm.simulation.backends import get_simulator

    for item in structures:
        # Handle both 5-tuple (R, seq, pdb_id, chain_id, L) and 3-tuple (R, seq, L)
        if len(item) == 5:
            R_native_raw, seq_raw, pdb_id, chain_id, L = item
        else:
            R_native_raw, seq_raw, L = item
            pdb_id = f"protein_{len(results)}"

        R_native = R_native_raw.unsqueeze(0).to(device)
        seq_tensor = seq_raw.unsqueeze(0).to(device)
        lengths = torch.tensor([L], device=device)

        # Precompute native reference data (native_contact_set + rg_native)
        native_np = R_native_raw.detach().cpu().numpy()
        ni, nj, d0 = native_contact_set(native_np)
        rg_native = float(np.sqrt(((native_np - native_np.mean(0)) ** 2).sum(1).mean()))

        with torch.no_grad():
            E_native = float(model(R_native, seq_tensor, lengths=lengths).item())
        logger.info("[%s] L=%d  E_native=%.4f  Rg=%.2fÅ  contacts=%d", pdb_id, L, E_native, rg_native, len(ni))

        for beta in betas:
            torch.manual_seed(args.seed)
            R_init = R_native.clone()  # start from native (basin stability)

            # Energy minimization via canonical L-BFGS in IC space
            min_info = None
            if args.minimize:
                R_init, E_min, n_min_steps, min_drmsd, min_dE = minimize_structure(model, R_init, seq_tensor, lengths)
                logger.info(
                    "    [%s] β=%.1f minimized in %d steps: " "ΔE=%+.3f  dRMSD=%.2f  E_min=%.3f",
                    pdb_id,
                    beta,
                    n_min_steps,
                    min_dE,
                    min_drmsd,
                    E_min,
                )
                min_info = {
                    "n_steps": int(n_min_steps),
                    "delta_E": float(min_dE),
                    "drmsd": float(min_drmsd),
                    "E_min": float(E_min),
                }

            # Initial metrics at production start
            init_coords = R_init[0, :L].detach().cpu().numpy()
            init_metrics = compute_metrics(init_coords, native_np, ni, nj, d0, rg_native)
            logger.info(
                "    [%s] β=%.1f start: Q=%.3f  RMSD=%.2f  dRMSD=%.2f  Rg%%=%.0f%%",
                pdb_id,
                beta,
                init_metrics["q"],
                init_metrics["rmsd"],
                init_metrics["drmsd"],
                init_metrics["rg_ratio"] * 100,
            )

            # Production MALA via canonical simulator
            sim = get_simulator(
                name="mala",
                model=model,
                seq=seq_tensor,
                R_init=R_init.detach(),
                step_size=args.dt,
                beta=beta,
                force_cap=args.force_cap,
                lengths=lengths,
            )

            trajectory = {
                "steps": [],
                "q": [],
                "rmsd": [],
                "drmsd": [],
                "rg_ratio": [],
                "energy": [],
                "accept_rate": [],
            }
            R_current = R_init.clone()
            for step in range(1, args.n_steps + 1):
                R_current, _, info = sim.step()
                if step % args.log_every == 0 or step == args.n_steps:
                    coords = R_current[0, :L].detach().cpu().numpy()
                    with torch.no_grad():
                        energy = float(model(R_current, seq_tensor, lengths=lengths).item())
                    m = compute_metrics(coords, native_np, ni, nj, d0, rg_native)
                    accept_rate = getattr(sim, "acceptance_rate", 0.0)
                    trajectory["steps"].append(step)
                    trajectory["q"].append(m["q"])
                    trajectory["rmsd"].append(m["rmsd"])
                    trajectory["drmsd"].append(m["drmsd"])
                    trajectory["rg_ratio"].append(m["rg_ratio"])
                    trajectory["energy"].append(energy)
                    trajectory["accept_rate"].append(accept_rate)
                    logger.info(
                        "    [%s] β=%.0f step %dK  Q=%.3f  RMSD=%.1f  "
                        "dRMSD=%.1f  Rg%%=%.0f%%  E=%.3f  accept=%.1f%%",
                        pdb_id,
                        beta,
                        step // 1000,
                        m["q"],
                        m["rmsd"],
                        m["drmsd"],
                        m["rg_ratio"] * 100,
                        energy,
                        accept_rate * 100,
                    )

            # Summary over final 20% of trajectory (pooled tail)
            n_tail = max(1, len(trajectory["q"]) // 5)
            final_q = float(np.mean(trajectory["q"][-n_tail:]))
            final_rmsd = float(np.mean(trajectory["rmsd"][-n_tail:]))
            final_drmsd = float(np.mean(trajectory["drmsd"][-n_tail:]))
            final_rg_r = float(np.mean(trajectory["rg_ratio"][-n_tail:]))
            final_accept = float(np.mean(trajectory["accept_rate"][-n_tail:]))
            logger.info(
                "  [%s] β=%.0f L=%d: final_Q=%.3f  RMSD=%.2f  " "dRMSD=%.2f  Rg/Rg*=%.2f  accept=%.2f",
                pdb_id,
                beta,
                L,
                final_q,
                final_rmsd,
                final_drmsd,
                final_rg_r,
                final_accept,
            )

            results.append(
                {
                    "pdb": pdb_id,
                    "L": L,
                    "beta": beta,
                    "E_native": E_native,
                    "rg_native": rg_native,
                    "init_metrics": init_metrics,
                    "minimization": min_info,
                    "final_q": final_q,
                    "final_rmsd": final_rmsd,
                    "final_drmsd": final_drmsd,
                    "final_rg_ratio": final_rg_r,
                    "final_accept_rate": final_accept,
                    "terms": terms_tag,
                    "trajectory": trajectory,
                }
            )

    out_file = out_dir / f"sweep_{terms_tag}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %d records to %s", len(results), out_file)


def main():
    ap = argparse.ArgumentParser(
        description="Physics prior native basin stability — ablation sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--terms", type=str, default="ram,hb,rep,rg,pack", help="Comma-separated subset of: ram,hb,rep,rg,pack,local"
    )
    ap.add_argument(
        "--pdb",
        nargs="+",
        required=True,
        help="Either a file path listing PDB IDs (one per line), or "
        "a whitespace-separated list of PDB IDs directly.",
    )
    ap.add_argument("--out", type=str, default="runs/physics_prior_sweep", help="Output directory for sweep results")
    ap.add_argument(
        "--cache-dir", type=str, default="data/pdb_cache", help="Directory to cache downloaded/parsed PDB structures"
    )
    ap.add_argument("--n-samples", type=int, default=10_000, help="Max structures to sample from the list")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--betas", type=str, default="5,50,100", help="Comma-separated list of β values to sweep")
    ap.add_argument("--n-steps", type=int, default=100_000)
    ap.add_argument("--step-size", type=float, default=1e-4, dest="dt", help="MALA integrator step size (η)")
    ap.add_argument("--force-cap", type=float, default=100.0, help="Force clipping magnitude for MALA simulator")
    ap.add_argument(
        "--minimize",
        action="store_true",
        help="L-BFGS energy-minimize start structure in IC space "
        "before MALA production (consistent with model_test.py).",
    )
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    # ── Data paths — defaults match actual repo layout. Script still
    #    hard-fails if any path is missing (see build_physics_prior_model).
    ap.add_argument(
        "--secondary-data-dir",
        type=str,
        default="analysis/secondary_analysis/data",
        help="Path to secondary_analysis/data directory",
    )
    ap.add_argument(
        "--repulsion-data-dir",
        type=str,
        default="analysis/repulsion_analysis/data",
        help="Path to repulsion_analysis/data directory",
    )
    ap.add_argument(
        "--packing-data-dir",
        type=str,
        default="analysis/repulsion_analysis/data",
        help="Path to packing data directory " "(canonical training uses repulsion's data)",
    )
    ap.add_argument(
        "--backbone-data-dir",
        type=str,
        default="analysis/backbone_geometry/data",
        help="Path to backbone_geometry/data directory",
    )
    ap.add_argument(
        "--coord-n-star-file",
        type=str,
        default="analysis/coordination_analysis/coord_n_star.json",
        help="Path to coord_n_star.json",
    )
    args = ap.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
