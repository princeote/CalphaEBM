"""
hremd_test.py — Basin stability and folding via Hamiltonian REMD.

The key difference from model_test.py: instead of a single trajectory
at β=L, HREMD runs N replicas in parallel with progressively flattened
packing and secondary landscapes. Compact misfolded states that trap single
trajectories can be escaped via the hot replicas, and folded configurations
propagate down the ladder to the target (β=L, all gates=1.0) replica.

Each replica starts from an INDEPENDENT random configuration (different
seed per replica), optionally energy-minimized via L-BFGS before dynamics.

Usage — basin stability (native start):
    PYTHONPATH=src python scripts/hremd_test.py \\
        --model checkpoints/run9/run9/full-stage/full_round009/step005000.pt \\
        --pdb-id 1YRF \\
        --start-mode native \\
        --preset small \\
        --n-swaps 500 --steps-per-swap 200 --step-size 3e-5

Usage — folding from random with minimization:
    PYTHONPATH=src python scripts/hremd_test.py \\
        --model checkpoints/run9/run9/full-stage/full_round009/step005000.pt \\
        --pdb-id 1YRF \\
        --start-mode random \\
        --minimize \\
        --n-replicas 4 --ss-min 0.08 --pack-min 0.05 \\
        --n-swaps 5000 --steps-per-swap 200 --step-size 1e-4 \\
        --beta 100 --log-every 100

Usage — custom ladder:
    PYTHONPATH=src python scripts/hremd_test.py \\
        --model checkpoints/run9/run9/full-stage/full_round009/step005000.pt \\
        --pdb-id 1ENH \\
        --start-mode random \\
        --minimize \\
        --n-replicas 5 --ss-min 0.05 --pack-min 0.03 \\
        --n-swaps 1000 --steps-per-swap 300
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
# Prevent duplicate output: calphaebm's internal logger has its own
# StreamHandler; disabling propagation stops it from also firing root's handler.
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
    """Energy minimize via L-BFGS in IC space (quadratic convergence).
    Consistent with eval_subprocess.py and negative_collector.py.
    Returns: (R_min, E_min, n_steps, drmsd, delta_E)
    """
    from calphaebm.simulation.minimize import lbfgs_minimize

    L = int(lengths[0].item())
    result = lbfgs_minimize(model, R_init, seq_tensor, lengths=lengths)

    R_min = result["R_min"]
    E_min = result["E_minimized"]
    n_steps = result["min_steps"]
    delta_E = result["E_relax"]  # E_min - E_pdb

    # dRMSD between input and minimized
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
    p = argparse.ArgumentParser(description="HREMD basin stability / folding test for CalphaEBM")

    # Model + structure
    p.add_argument("--model", required=True, help="Checkpoint path (.pt)")
    p.add_argument("--pdb-id", default=None, help="PDB ID to download")
    p.add_argument("--pdb-file", default=None, help="Local PDB/mmCIF file")
    p.add_argument("--cache-dir", default="pdb_cache")

    # Sampling
    p.add_argument("--start-mode", choices=["native", "extended", "random"], default="random")
    p.add_argument("--step-size", type=float, default=3e-5)
    p.add_argument("--force-cap", type=float, default=100.0)
    p.add_argument("--beta", type=float, default=None, help="Override β (default β=L)")
    p.add_argument(
        "--minimize",
        action="store_true",
        default=False,
        help="Energy-minimize starting structures via L-BFGS "
        "before dynamics. Each replica is minimized independently.",
    )

    # Gate ladder — preset OR manual
    p.add_argument(
        "--preset",
        choices=["small", "medium", "large", "ood_small", "ss_only", "pack_only"],
        default=None,
        help="Use a built-in gate ladder preset",
    )
    p.add_argument(
        "--n-replicas", type=int, default=4, help="Total replicas (1 target + n-1 hot). Ignored if --preset used."
    )
    p.add_argument("--ss-min", type=float, default=0.08, help="Secondary gate at hottest replica (geometric ladder)")
    p.add_argument("--pack-min", type=float, default=0.05, help="Packing gate at hottest replica (geometric ladder)")

    # HREMD settings
    p.add_argument("--n-swaps", type=int, default=500, help="Number of swap rounds")
    p.add_argument("--steps-per-swap", type=int, default=200, help="MALA steps between swap attempts")
    p.add_argument("--swap-scheme", choices=["adjacent", "all_pairs"], default="adjacent")

    # Output
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--output-dir", default="results/hremd")

    return p.parse_args()


def assess_target(traj: list, Q_threshold: float = 0.9) -> dict:
    """Summarise the target replica trajectory."""
    if not traj:
        return {}
    Es = [t["E"] for t in traj]
    acc = [t["accept"] for t in traj]
    return {
        "n_points": len(traj),
        "E_mean": sum(Es) / len(Es),
        "E_min": min(Es),
        "accept_mean": sum(acc) / len(acc),
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
    beta = args.beta if args.beta is not None else float(L)
    logger.info("Target: %s/%s  L=%d  β=%.1f", pdb_id, chain_id, L, beta)

    # ── Gate ladder ─────────────────────────────────────────────────────────
    from calphaebm.simulation.hremd import LADDER_PRESETS, GateVector, HREMDSimulator

    if args.preset:
        gate_ladder = LADDER_PRESETS[args.preset]
        logger.info("Preset '%s': %d replicas", args.preset, len(gate_ladder))
    else:
        n_hot = args.n_replicas - 1
        gate_ladder = GateVector.ladder(
            n_hot=n_hot,
            ss_min=args.ss_min,
            pack_min=args.pack_min,
        )
        logger.info(
            "Custom ladder: %d replicas  ss_min=%.3f  pack_min=%.3f", len(gate_ladder), args.ss_min, args.pack_min
        )

    n_reps = len(gate_ladder)
    lengths = torch.tensor([L])

    # Compute native contacts for Q calculation
    from calphaebm.evaluation.metrics import native_contact_set

    native_coords = R_native[:L].detach().numpy() if R_native.dim() == 2 else R_native[0, :L].detach().numpy()
    ni, nj, d0 = native_contact_set(native_coords)

    # ── Generate per-replica starting configs ───────────────────────────────
    # Each replica gets an independent random structure (different seed).
    # For native start, all replicas start from native.
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
                    f"    Replica {i}: minimized in {min_steps} steps  "
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
                    f"    Replica {i}: minimized in {min_steps} steps  "
                    f"E={E_min:.3f}  Q={q:.3f}  dRMSD_nat={dr_nat:.2f}  ΔE={min_dE:.3f}"
                )
            else:
                dr_nat = drmsd_to_native(R_rand, R_native, L)
                q = compute_q(R_rand, ni, nj, d0, L)
                print(f"    Replica {i}: random (seed={seed})  Q={q:.3f}  dRMSD_nat={dr_nat:.2f}")
            R_replicas.append(R_rand)
        print()

    elif args.start_mode == "extended":
        # Extended chain — same geometry, different noise per replica
        from calphaebm.geometry.reconstruct import nerf_reconstruct

        for i in range(n_reps):
            torch.manual_seed(42 + i * 1000)
            anchor = torch.zeros(1, 3, 3)
            anchor[0, 0] = torch.tensor([0.0, 0.0, 0.0])
            anchor[0, 1] = torch.tensor([3.8, 0.0, 0.0])
            anchor[0, 2] = torch.tensor([3.8 + 3.8 * math.cos(2.09), 3.8 * math.sin(2.09), 0.0])
            # Near-extended: theta ~ 2.0 ± 0.05, phi ~ π ± 0.1
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
                    f"    Replica {i}: extended+minimized in {min_steps} steps  "
                    f"E={E_min:.3f}  Q={q:.3f}  dRMSD_nat={dr_nat:.2f}"
                )
            R_replicas.append(R_ext)
    else:
        raise ValueError(f"Unknown start_mode: {args.start_mode}")

    # ── Build HREMD ─────────────────────────────────────────────────────────
    hremd = HREMDSimulator(
        model=model,
        seq=seq.unsqueeze(0),
        lengths=lengths,
        beta=beta,
        gate_ladder=gate_ladder,
        step_size=args.step_size,
        force_cap=args.force_cap,
        n_steps_per_swap=args.steps_per_swap,
        swap_scheme=args.swap_scheme,
    )

    # Initialize with per-replica coordinates.
    # R_replicas is a list of (1, L, 3) tensors, one per replica.
    hremd.initialize(
        start_mode=args.start_mode,
        R_native=R_native.unsqueeze(0) if R_native is not None else None,
        R_replicas=R_replicas,
    )

    # ── Header ──────────────────────────────────────────────────────────────
    total_mala = args.n_swaps * args.steps_per_swap * n_reps
    min_tag = " + minimize" if args.minimize else ""
    print()
    print("=" * 70)
    print(f"  HREMD TEST: {pdb_id} (L={L})")
    print(f"  Start: {args.start_mode}{min_tag}  |  β={beta:.1f}  |  {n_reps} replicas")
    print(
        f"  {args.n_swaps} swap rounds × {args.steps_per_swap} steps × {n_reps} replicas  "
        f"=  {total_mala:,} total MALA steps"
    )
    print(f"  Gate ladder:")
    for i, g in enumerate(gate_ladder):
        tag = "  ← TARGET" if i == 0 else ""
        print(f"    [{i}] ss={g.secondary:.4f}  pack={g.packing:.4f}{tag}")
    print("=" * 70)
    print()

    # ── Run ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    hremd.run(n_swaps=args.n_swaps, log_every=args.log_every)
    elapsed = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────────────
    swap_rates = hremd.swap_acceptance_rates()
    traj = hremd.target_trajectory()
    stats = assess_target(traj)

    print()
    print("=" * 70)
    print(f"  HREMD RESULTS: {pdb_id}  ({elapsed/60:.1f} min)")
    print("-" * 70)
    print(f"  Target replica (idx=0, all gates=1.0):")
    print(
        f"    E_mean={stats.get('E_mean', 0):.3f}  "
        f"E_min={stats.get('E_min', 0):.3f}  "
        f"MALA_accept={stats.get('accept_mean', 0):.1%}"
    )
    print("-" * 70)
    print("  Swap acceptance (target 20-40% per pair):")
    for k, rate in enumerate(swap_rates):
        gi, gj = gate_ladder[k], gate_ladder[k + 1]
        flag = ""
        if rate < 0.10:
            flag = "  ← TOO LOW (ladder too spread)"
        if rate > 0.50:
            flag = "  ← TOO HIGH (ladder too compressed)"
        print(
            f"    [{k}]↔[{k+1}]  ss: {gi.secondary:.4f}→{gj.secondary:.4f}  "
            f"pack: {gi.packing:.4f}→{gj.packing:.4f}  "
            f"rate={rate:.1%}{flag}"
        )
    print("=" * 70)

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / pdb_id / args.start_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "L": L,
        "beta": beta,
        "start_mode": args.start_mode,
        "minimize": args.minimize,
        "n_replicas": n_reps,
        "n_swaps": args.n_swaps,
        "steps_per_swap": args.steps_per_swap,
        "gate_ladder": [g.as_dict() for g in gate_ladder],
        "swap_acceptance_rates": swap_rates,
        "target_stats": stats,
        "elapsed_min": elapsed / 60,
    }
    out_path = out_dir / "hremd_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved to %s", out_path)

    # Save per-replica FES data
    fes_dir = str(out_dir / "fes")
    hremd.save_fes(fes_dir, pdb_id=pdb_id)


if __name__ == "__main__":
    main()
