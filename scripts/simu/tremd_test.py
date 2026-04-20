"""
tremd_test.py — Basin stability and folding via Temperature REMD.

Unlike HREMD (which modifies gate scaling on individual energy terms),
TREMD preserves the exact energy landscape across all replicas and only
varies β (inverse temperature). This is physically correct: the native
basin remains the global minimum at every temperature; hot replicas
simply cross barriers more easily.

Each replica starts from an independent random configuration (different
seed), optionally energy-minimized via L-BFGS before dynamics begin.

Usage — folding from random with minimization:
    PYTHONPATH=src python scripts/simu/tremd_test.py \\
        --model checkpoints/run9/run9/full-stage/full_round009/step005000.pt \\
        --pdb-id 1YRF \\
        --start-mode random --minimize \\
        --n-replicas 4 --beta 100 --beta-min 5.0 \\
        --step-size 1e-4 \\
        --n-swaps 5000 --steps-per-swap 200 \\
        --log-every 100

Usage — basin stability (native start):
    PYTHONPATH=src python scripts/simu/tremd_test.py \\
        --model checkpoints/run9/run9/full-stage/full_round009/step005000.pt \\
        --pdb-id 1YRF \\
        --start-mode native --minimize \\
        --n-replicas 4 --beta 100 --beta-min 10.0 \\
        --step-size 3e-5 \\
        --n-swaps 500 --steps-per-swap 200

Usage — custom β ladder:
    PYTHONPATH=src python scripts/simu/tremd_test.py \\
        --model checkpoints/run9/run9/full-stage/full_round009/step005000.pt \\
        --pdb-id 1PGB \\
        --start-mode random --minimize \\
        --n-replicas 6 --beta 100 --beta-min 2.0 \\
        --step-size 1e-4 \\
        --n-swaps 10000 --steps-per-swap 200
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("calphaebm").propagate = False
logger = logging.getLogger(__name__)


# =====================================================================
#  Random chain generation + minimization (from model_test.py)
# =====================================================================


def generate_random_chain(L, seq, seed=None):
    """Generate a random chain with random torsion angles via NeRF."""
    from calphaebm.geometry.reconstruct import nerf_reconstruct

    if seed is not None:
        torch.manual_seed(seed)

    anchor = torch.zeros(1, 3, 3)
    anchor[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    anchor[0, 1] = torch.tensor([3.8, 0.0, 0.0])
    anchor[0, 2] = torch.tensor([3.8 + 3.8 * math.cos(2.09), 3.8 * math.sin(2.09), 0.0])

    theta = torch.empty(1, L - 2).uniform_(1.4, 2.4)
    phi = torch.empty(1, L - 3).uniform_(-math.pi, math.pi)

    R = nerf_reconstruct(theta, phi, anchor, bond=3.8)
    R = R - R.mean(dim=1, keepdim=True)
    return R


def minimize_structure(model, R_init, seq_tensor, lengths):
    """Energy minimize via L-BFGS in IC space.
    Returns: (R_min, E_min, n_steps, drmsd, delta_E)
    """
    from calphaebm.simulation.minimize import lbfgs_minimize

    L = int(lengths[0].item())
    result = lbfgs_minimize(model, R_init, seq_tensor, lengths=lengths)

    R_min = result["R_min"]
    E_min = result["E_minimized"]
    n_steps = result["min_steps"]
    delta_E = result["E_relax"]

    coords_init = R_init[0, :L].detach().numpy()
    coords_min = R_min[0, :L].detach().numpy()
    d_init = np.sqrt(((coords_init[:, None] - coords_init[None, :]) ** 2).sum(-1))
    d_min = np.sqrt(((coords_min[:, None] - coords_min[None, :]) ** 2).sum(-1))
    triu = np.triu_indices(L, k=1)
    drmsd = float(np.sqrt(np.mean((d_init[triu] - d_min[triu]) ** 2)))

    return R_min, E_min, n_steps, drmsd, delta_E


def drmsd_to_native(R, R_native, L):
    """Compute dRMSD between structure R and native R_native."""
    c1 = R[0, :L].detach().numpy()
    c2 = R_native[:L].detach().numpy() if R_native.dim() == 2 else R_native[0, :L].detach().numpy()
    d1 = np.sqrt(((c1[:, None] - c1[None, :]) ** 2).sum(-1))
    d2 = np.sqrt(((c2[:, None] - c2[None, :]) ** 2).sum(-1))
    triu = np.triu_indices(L, k=1)
    return float(np.sqrt(np.mean((d1[triu] - d2[triu]) ** 2)))


def compute_q(R, ni, nj, d0, L):
    """Compute fraction of native contacts Q for structure R."""
    from calphaebm.evaluation.metrics import q_smooth

    coords = R[0, :L].detach().numpy()
    return q_smooth(coords, ni, nj, d0)


def parse_args():
    p = argparse.ArgumentParser(description="TREMD basin stability / folding test for CalphaEBM")

    # Model + structure
    p.add_argument("--model", required=True, help="Checkpoint path (.pt)")
    p.add_argument("--pdb-id", default=None, help="PDB ID to download")
    p.add_argument("--pdb-file", default=None, help="Local PDB/mmCIF file")
    p.add_argument("--cache-dir", default="pdb_cache")

    # Sampling
    p.add_argument("--start-mode", choices=["native", "extended", "random"], default="random")
    p.add_argument("--step-size", type=float, default=3e-5)
    p.add_argument("--force-cap", type=float, default=100.0)
    p.add_argument(
        "--minimize",
        action="store_true",
        default=False,
        help="Energy-minimize starting structures via L-BFGS "
        "before dynamics. Each replica is minimized independently.",
    )

    # Temperature ladder
    p.add_argument("--n-replicas", type=int, default=4, help="Number of replicas (geometric β ladder)")
    p.add_argument("--beta", type=float, default=None, help="β for target (coldest) replica (default: β=L)")
    p.add_argument("--beta-min", type=float, default=5.0, help="β for hottest replica (default: 5.0)")
    p.add_argument(
        "--no-scale-step", action="store_true", default=False, help="Disable step-size scaling by sqrt(β_target/β_k)"
    )

    # TREMD settings
    p.add_argument("--n-swaps", type=int, default=500, help="Number of swap rounds")
    p.add_argument("--steps-per-swap", type=int, default=200, help="MALA steps between swap attempts")
    p.add_argument("--swap-scheme", choices=["adjacent", "all_pairs"], default="adjacent")

    # Output
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--output-dir", default="results/tremd")

    return p.parse_args()


def assess_target(traj: list) -> dict:
    """Summarise the target replica trajectory."""
    if not traj:
        return {}
    Es = [t["E"] for t in traj]
    acc = [t["accept"] for t in traj]
    Qs = [t["Q"] for t in traj if t["Q"] == t["Q"]]  # filter NaN
    dRs = [t["dRMSD"] for t in traj if t["dRMSD"] == t["dRMSD"]]
    return {
        "n_points": len(traj),
        "E_mean": sum(Es) / len(Es),
        "E_min": min(Es),
        "accept_mean": sum(acc) / len(acc),
        "Q_max": max(Qs) if Qs else 0.0,
        "Q_final": Qs[-1] if Qs else 0.0,
        "dRMSD_min": min(dRs) if dRs else 999.0,
        "dRMSD_final": dRs[-1] if dRs else 999.0,
    }


def main():
    args = parse_args()

    # ── Load model ─────────────────────────────────────────────────────────
    from calphaebm.evaluation.core_evaluation import load_model

    logger.info("Loading model from %s", args.model)
    model = load_model(Path(args.model))
    model.eval()

    # ── Load structure ──────────────────────────────────────────────────────
    from calphaebm.data.pdb_chain_dataset import PDBChainDataset

    pdb_ids = [args.pdb_id] if args.pdb_id else None
    dataset = PDBChainDataset(
        pdb_ids=pdb_ids,
        cache_dir=args.cache_dir,
        min_len=1,
        max_len=10000,
        require_monomeric=False,
        require_complete=False,
    )
    assert len(dataset) == 1, f"Expected 1 structure, got {len(dataset)}"
    coords, seq, pdb_id, chain_id = dataset[0]
    R_native = coords
    L = coords.shape[0]
    beta_target = args.beta if args.beta is not None else float(L)
    logger.info("Target: %s/%s  L=%d  β_target=%.1f", pdb_id, chain_id, L, beta_target)

    # ── Native contacts for Q computation ──────────────────────────────────
    from calphaebm.evaluation.metrics import native_contact_set

    native_coords = R_native[:L].detach().numpy() if R_native.dim() == 2 else R_native[0, :L].detach().numpy()
    ni, nj, d0 = native_contact_set(native_coords)

    # ── β ladder ───────────────────────────────────────────────────────────
    from calphaebm.simulation.tremd import TREMDSimulator, geometric_beta_ladder

    beta_ladder = geometric_beta_ladder(beta_target, args.beta_min, args.n_replicas)
    n_reps = len(beta_ladder)
    lengths = torch.tensor([L])

    # ── Generate per-replica starting configs ───────────────────────────────
    R_replicas = []
    if args.start_mode == "native":
        seq_batch = seq.unsqueeze(0)
        for i in range(n_reps):
            R_init = R_native.unsqueeze(0).clone()
            if args.minimize:
                R_init, E_min, min_steps, min_drmsd, min_dE = minimize_structure(model, R_init, seq_batch, lengths)
                dr_nat = drmsd_to_native(R_init, R_native, L)
                q = compute_q(R_init, ni, nj, d0, L)
                if i == 0:
                    print(f"\n  Minimizing native start for {n_reps} replicas...")
                print(
                    f"    Replica {i} (β={beta_ladder[i]:.1f}): minimized in {min_steps} steps  "
                    f"E={E_min:.3f}  Q={q:.3f}  dRMSD_nat={dr_nat:.2f}  ΔE={min_dE:.3f}"
                )
            R_replicas.append(R_init)
        if args.minimize:
            print()
        else:
            logger.info("All %d replicas start from native (no minimize)", n_reps)

    elif args.start_mode == "random":
        print(f"\n  Generating {n_reps} independent random starts...")
        seq_batch = seq.unsqueeze(0)
        for i in range(n_reps):
            seed = 42 + i * 1000
            R_rand = generate_random_chain(L, seq, seed=seed)
            if args.minimize:
                R_rand, E_min, min_steps, min_drmsd, min_dE = minimize_structure(model, R_rand, seq_batch, lengths)
                dr_nat = drmsd_to_native(R_rand, R_native, L)
                q = compute_q(R_rand, ni, nj, d0, L)
                print(
                    f"    Replica {i} (β={beta_ladder[i]:.1f}): minimized in {min_steps} steps  "
                    f"E={E_min:.3f}  Q={q:.3f}  dRMSD_nat={dr_nat:.2f}  ΔE={min_dE:.3f}"
                )
            else:
                dr_nat = drmsd_to_native(R_rand, R_native, L)
                q = compute_q(R_rand, ni, nj, d0, L)
                print(
                    f"    Replica {i} (β={beta_ladder[i]:.1f}): random (seed={seed})  "
                    f"Q={q:.3f}  dRMSD_nat={dr_nat:.2f}"
                )
            R_replicas.append(R_rand)
        print()

    elif args.start_mode == "extended":
        from calphaebm.geometry.reconstruct import nerf_reconstruct

        for i in range(n_reps):
            torch.manual_seed(42 + i * 1000)
            anchor = torch.zeros(1, 3, 3)
            anchor[0, 0] = torch.tensor([0.0, 0.0, 0.0])
            anchor[0, 1] = torch.tensor([3.8, 0.0, 0.0])
            anchor[0, 2] = torch.tensor([3.8 + 3.8 * math.cos(2.09), 3.8 * math.sin(2.09), 0.0])
            theta = torch.empty(1, L - 2).normal_(2.0, 0.05)
            phi = torch.empty(1, L - 3).normal_(math.pi, 0.1)
            R_ext = nerf_reconstruct(theta, phi, anchor, bond=3.8)
            R_ext = R_ext - R_ext.mean(dim=1, keepdim=True)
            if args.minimize:
                seq_batch = seq.unsqueeze(0)
                R_ext, E_min, min_steps, min_drmsd, min_dE = minimize_structure(model, R_ext, seq_batch, lengths)
                dr_nat = drmsd_to_native(R_ext, R_native, L)
                q = compute_q(R_ext, ni, nj, d0, L)
                print(
                    f"    Replica {i} (β={beta_ladder[i]:.1f}): extended+minimized  "
                    f"E={E_min:.3f}  Q={q:.3f}  dRMSD_nat={dr_nat:.2f}"
                )
            R_replicas.append(R_ext)

    # ── Build TREMD ─────────────────────────────────────────────────────────
    tremd = TREMDSimulator(
        model=model,
        seq=seq.unsqueeze(0),
        lengths=lengths,
        beta_ladder=beta_ladder,
        step_size=args.step_size,
        force_cap=args.force_cap,
        n_steps_per_swap=args.steps_per_swap,
        swap_scheme=args.swap_scheme,
        scale_step_size=not args.no_scale_step,
    )

    tremd.initialize(
        start_mode=args.start_mode,
        R_native=R_native.unsqueeze(0) if R_native is not None else None,
        R_replicas=R_replicas,
    )

    # ── Header ──────────────────────────────────────────────────────────────
    total_mala = args.n_swaps * args.steps_per_swap * n_reps
    min_tag = " + minimize" if args.minimize else ""
    print()
    print("=" * 70)
    print(f"  TREMD TEST: {pdb_id} (L={L})")
    print(f"  Start: {args.start_mode}{min_tag}  |  {n_reps} replicas")
    print(
        f"  {args.n_swaps} swap rounds × {args.steps_per_swap} steps × {n_reps} replicas  "
        f"=  {total_mala:,} total MALA steps"
    )
    print(f"  Step-size scaling: {'sqrt(β_target/β)' if not args.no_scale_step else 'OFF'}")
    print(f"  β ladder:")
    for i, b in enumerate(beta_ladder):
        tag = "  ← TARGET" if i == 0 else ""
        print(f"    [{i}] β={b:7.2f}  (T={1/b:.4f}){tag}")
    print("=" * 70)
    print()

    # ── Run ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    tremd.run(n_swaps=args.n_swaps, log_every=args.log_every)
    elapsed = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────────────
    swap_rates = tremd.swap_acceptance_rates()
    traj = tremd.target_trajectory()
    stats = assess_target(traj)

    print()
    print("=" * 70)
    print(f"  TREMD RESULTS: {pdb_id}  ({elapsed/60:.1f} min)")
    print("-" * 70)
    print(f"  Target replica (idx=0, β={beta_ladder[0]:.1f}):")
    print(
        f"    E_mean={stats.get('E_mean', 0):.3f}  "
        f"E_min={stats.get('E_min', 0):.3f}  "
        f"MALA_accept={stats.get('accept_mean', 0):.1%}"
    )
    print(
        f"    Q_max={stats.get('Q_max', 0):.3f}  "
        f"Q_final={stats.get('Q_final', 0):.3f}  "
        f"dRMSD_min={stats.get('dRMSD_min', 999):.2f}  "
        f"dRMSD_final={stats.get('dRMSD_final', 999):.2f}"
    )
    print("-" * 70)
    print("  Swap acceptance (target 20-40% per pair):")
    for k, rate in enumerate(swap_rates):
        bi, bj = beta_ladder[k], beta_ladder[k + 1]
        flag = ""
        if rate < 0.10:
            flag = "  ← TOO LOW (need more replicas or smaller β gap)"
        if rate > 0.50:
            flag = "  ← TOO HIGH (can remove intermediate replica)"
        print(f"    [{k}]↔[{k+1}]  β: {bi:.1f}→{bj:.1f}  rate={rate:.1%}{flag}")
    print("=" * 70)

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / pdb_id / args.start_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "L": L,
        "beta_target": beta_target,
        "beta_min": args.beta_min,
        "start_mode": args.start_mode,
        "minimize": args.minimize,
        "n_replicas": n_reps,
        "n_swaps": args.n_swaps,
        "steps_per_swap": args.steps_per_swap,
        "beta_ladder": beta_ladder,
        "swap_acceptance_rates": swap_rates,
        "target_stats": stats,
        "elapsed_min": elapsed / 60,
    }
    out_path = out_dir / "tremd_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved to %s", out_path)

    fes_dir = str(out_dir / "fes")
    tremd.save_fes(fes_dir, pdb_id=pdb_id)


if __name__ == "__main__":
    main()
