#!/usr/bin/env python3
"""Model test: basin stability and ab initio folding for CalphaEBM.

Three modes via --start-mode:
  native:   Start from PDB coordinates. Tests basin stability.
  extended: Start from fully extended chain. Tests folding via annealing.
  random:   Start from random torsion angles. Tests folding via annealing.

Usage:
    # Basin stability test (from native, fixed beta, no annealing)
    python scripts/model_test.py \
        --model round002.pt \
        --pdb-id 1uao \
        --start-mode native \
        --n-steps 100000 \
        --equil-beta 1000 \
        --n-trials 1

    # Multi-beta basin scan
    python scripts/model_test.py \
        --model round002.pt \
        --pdb-id 1uao \
        --start-mode native \
        --n-steps 100000 \
        --multi-beta 100 500 1000 2000

    # Ab initio folding (simulated annealing)
    python scripts/model_test.py \
        --model round002.pt \
        --pdb-id 1uao \
        --start-mode random \
        --n-trials 10 \
        --n-steps 1000000 \
        --anneal-steps 500000 \
        --beta-start 1 --beta-end 1000 --equil-beta 1000

    # Quick folding test
    python scripts/model_test.py \
        --model round002.pt \
        --pdb-id 1uao \
        --start-mode extended \
        --n-trials 3 \
        --n-steps 100000 \
        --anneal-steps 50000
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def get_native_structure(pdb_id, cache_dir="pdb_cache"):
    """Load native structure from PDB."""
    from calphaebm.data.pdb_chain_dataset import PDBChainDataset

    dataset = PDBChainDataset(
        pdb_ids=[pdb_id],
        cache_dir=cache_dir,
        min_len=5,
        max_len=1000,
        require_monomeric=False,
        require_complete=False,
    )
    if len(dataset) == 0:
        raise ValueError(f"Could not load PDB {pdb_id}")
    coords, seq, pid, cid = dataset[0]
    return coords.numpy(), seq.numpy(), pid, cid


def generate_extended_chain(L, seq):
    """Generate a fully extended chain via NeRF (all torsions ~180°)."""
    from calphaebm.geometry.reconstruct import nerf_reconstruct

    anchor = torch.zeros(1, 3, 3)
    anchor[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    anchor[0, 1] = torch.tensor([3.8, 0.0, 0.0])
    anchor[0, 2] = torch.tensor([3.8 + 3.8 * math.cos(2.09), 3.8 * math.sin(2.09), 0.0])

    theta = torch.full((1, L - 2), 2.09, dtype=torch.float32)
    phi = torch.full((1, L - 3), math.pi, dtype=torch.float32)

    R = nerf_reconstruct(theta, phi, anchor, bond=3.8)
    R = R - R.mean(dim=1, keepdim=True)
    return R


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


def compute_metrics(coords, native_coords, ni, nj, d0, rg_native):
    """Compute Q, RMSD, Rg, dRMSD for current structure."""
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


def run_single_trial(
    model,
    seq_tensor,
    R_native,
    native_np,
    ni,
    nj,
    d0,
    rg_native,
    n_steps,
    anneal_steps,
    beta_start,
    beta_end,
    equil_beta,
    step_size,
    force_cap,
    start_mode,
    trial_idx,
    log_every,
    minimize=False,
):
    """Run one trial (basin stability or folding)."""
    from calphaebm.simulation.backends import get_simulator

    L = native_np.shape[0]
    lengths = torch.tensor([L])

    # Generate starting structure
    if start_mode == "native":
        R_init = R_native.clone()
    elif start_mode == "extended":
        R_init = generate_extended_chain(L, seq_tensor)
    elif start_mode == "random":
        R_init = generate_random_chain(L, seq_tensor, seed=42 + trial_idx * 1000)
    else:
        raise ValueError(f"Unknown start_mode: {start_mode}")

    # Minimize before dynamics
    if minimize:
        R_init, E_min, min_steps, min_drmsd, min_dE = minimize_structure(model, R_init, seq_tensor, lengths)
        print(f"    Minimized in {min_steps} steps: ΔE={min_dE:.3f}  dRMSD={min_drmsd:.2f}  E_min={E_min:.3f}")

    # Initial metrics
    init_coords = R_init[0, :L].detach().numpy()
    init_metrics = compute_metrics(init_coords, native_np, ni, nj, d0, rg_native)

    print(
        f"\n  Trial {trial_idx + 1}: start_mode={start_mode}  "
        f"Q={init_metrics['q']:.3f}  RMSD={init_metrics['rmsd']:.1f}  "
        f"dRMSD={init_metrics['drmsd']:.1f}  Rg_ratio={init_metrics['rg_ratio']:.2f}"
    )

    # Trajectory storage
    trajectory = {
        "steps": [],
        "q": [],
        "rmsd": [],
        "drmsd": [],
        "rg_ratio": [],
        "energy": [],
        "beta": [],
        "accept_rate": [],
    }

    # Best structure found
    best_q = init_metrics["q"]
    best_step = 0
    best_coords = init_coords.copy()

    # For native mode: skip annealing, go straight to equilibration
    if start_mode == "native":
        anneal_steps = 0

    # Annealing schedule: log-linear in beta
    if anneal_steps > 0:
        log_beta_start = math.log(beta_start)
        log_beta_end = math.log(beta_end)
    else:
        log_beta_start = math.log(equil_beta)
        log_beta_end = math.log(equil_beta)

    R_current = R_init.clone()
    sim = None
    current_beta = equil_beta if anneal_steps == 0 else beta_start

    total_accept = 0
    total_steps = 0

    for step in range(1, n_steps + 1):
        # Determine current beta
        if anneal_steps > 0 and step <= anneal_steps:
            frac = step / anneal_steps
            current_beta = math.exp(log_beta_start + frac * (log_beta_end - log_beta_start))
        else:
            current_beta = equil_beta

        # Re-create simulator when beta changes significantly
        if sim is None or (anneal_steps > 0 and step <= anneal_steps and step % 1000 == 1):
            sim = get_simulator(
                name="mala",
                model=model,
                seq=seq_tensor,
                R_init=R_current.detach(),
                step_size=step_size,
                beta=current_beta,
                force_cap=force_cap,
                lengths=lengths,
            )

        R_current, _, info = sim.step()
        total_steps += 1

        # Log at intervals
        if step % log_every == 0:
            coords = R_current[0, :L].detach().numpy()
            with torch.no_grad():
                energy = float(model(R_current, seq_tensor, lengths=lengths).item())

            metrics = compute_metrics(coords, native_np, ni, nj, d0, rg_native)

            accept_rate = 0.0
            if hasattr(sim, "acceptance_rate"):
                accept_rate = sim.acceptance_rate

            trajectory["steps"].append(step)
            trajectory["q"].append(metrics["q"])
            trajectory["rmsd"].append(metrics["rmsd"])
            trajectory["drmsd"].append(metrics["drmsd"])
            trajectory["rg_ratio"].append(metrics["rg_ratio"])
            trajectory["energy"].append(energy)
            trajectory["beta"].append(current_beta)
            trajectory["accept_rate"].append(accept_rate)

            if metrics["q"] > best_q:
                best_q = metrics["q"]
                best_step = step
                best_coords = coords.copy()

            phase = "anneal" if anneal_steps > 0 and step <= anneal_steps else "equil"
            print(
                f"    [{phase}] step {step//1000}K  β={current_beta:.0f}  "
                f"Q={metrics['q']:.3f}  RMSD={metrics['rmsd']:.1f}  "
                f"dRMSD={metrics['drmsd']:.1f}  Rg%={metrics['rg_ratio']*100:.0f}%  "
                f"E={energy:.3f}  accept={accept_rate*100:.1f}%"
            )

            # Early success for folding modes
            if start_mode != "native" and metrics["q"] > 0.9:
                print(f"    *** FOLDED at step {step}! Q={metrics['q']:.3f} ***")

    # Final metrics
    final_coords = R_current[0, :L].detach().numpy()
    final_metrics = compute_metrics(final_coords, native_np, ni, nj, d0, rg_native)

    result = {
        "trial": trial_idx,
        "start_mode": start_mode,
        "n_steps": n_steps,
        "anneal_steps": anneal_steps,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "equil_beta": equil_beta,
        "init_q": init_metrics["q"],
        "final_q": final_metrics["q"],
        "best_q": best_q,
        "best_step": best_step,
        "final_rmsd": final_metrics["rmsd"],
        "final_drmsd": final_metrics["drmsd"],
        "folded": best_q > 0.8,
        "stable": final_metrics["q"] > 0.9 if start_mode == "native" else best_q > 0.8,
        "trajectory": trajectory,
    }

    status = "STABLE" if result["stable"] else "UNSTABLE"
    if start_mode != "native":
        status = "FOLDED" if result["folded"] else "FAILED"

    print(
        f"\n  Trial {trial_idx + 1} DONE: "
        f"final_Q={final_metrics['q']:.3f}  best_Q={best_q:.3f}@step{best_step}  "
        f"{status}"
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Model test for CalphaEBM — basin stability and folding")
    parser.add_argument("--model", required=True, help="Trained model path (.pt file)")
    parser.add_argument("--pdb-id", type=str, default=None, help="PDB ID to test (e.g., 1uao)")
    parser.add_argument("--pdb-file", type=str, default=None, help="PDB file path")
    parser.add_argument("--cache-dir", default="pdb_cache", help="PDB cache directory")
    parser.add_argument("--n-trials", type=int, default=1, help="Number of independent trials")
    parser.add_argument("--n-steps", type=int, default=100_000, help="Total steps per trial")
    parser.add_argument(
        "--anneal-steps",
        type=int,
        default=0,
        help="Steps for annealing phase (default: 0 = no annealing). " "Ignored for --start-mode native.",
    )
    parser.add_argument("--beta-start", type=float, default=1.0, help="Starting beta for annealing")
    parser.add_argument("--beta-end", type=float, default=1000.0, help="End of annealing beta")
    parser.add_argument("--equil-beta", type=float, default=1000.0, help="Equilibration beta")
    parser.add_argument(
        "--multi-beta",
        type=float,
        nargs="+",
        default=None,
        help="Run separate trials at each beta (e.g., --multi-beta 100 500 1000 2000). "
        "Overrides --equil-beta and --n-trials.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1e-3,
        help="MALA/Langevin step size η (default: 1e-3). " "Target ~57%% acceptance. Increase for larger proteins.",
    )
    parser.add_argument("--force-cap", type=float, default=100.0, help="Force clipping")
    parser.add_argument(
        "--start-mode",
        choices=["native", "extended", "random"],
        default="native",
        help="Initial conformation (default: native)",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        default=False,
        help="Energy-minimize the starting structure before dynamics. "
        "Puts the structure in the model's energy minimum.",
    )
    parser.add_argument("--log-every", type=int, default=10000, help="Log interval (steps)")
    parser.add_argument("--output-dir", type=str, default="results/model_test", help="Output directory")
    args = parser.parse_args()

    # Load model via core_evaluation.load_model — identical to training eval.
    # build_model + _v6_model_args loads all data dirs (packing, coord, secondary)
    # exactly as training does. No manual JSON parsing, no silent fallbacks.
    print(f"Loading model from {args.model}")
    from calphaebm.evaluation.core_evaluation import load_model

    model = load_model(Path(args.model), device=torch.device("cpu"))
    model.eval()

    # Load native structure
    if args.pdb_id:
        native_coords, seq_np, pdb_id, chain_id = get_native_structure(args.pdb_id, args.cache_dir)
    elif args.pdb_file:
        raise NotImplementedError("Direct PDB file loading not yet implemented")
    else:
        parser.error("Must provide --pdb-id or --pdb-file")

    L = len(native_coords)
    print(f"Target: {pdb_id}/{chain_id}  L={L}")

    # Prepare reference data
    from calphaebm.evaluation.metrics import native_contact_set

    ni, nj, d0 = native_contact_set(native_coords)
    rg_native = float(np.sqrt(((native_coords - native_coords.mean(0)) ** 2).sum(1).mean()))

    seq_tensor = torch.tensor(seq_np, dtype=torch.long).unsqueeze(0)
    R_native = torch.tensor(native_coords, dtype=torch.float32).unsqueeze(0)

    # Native energy
    with torch.no_grad():
        E_native = float(model(R_native, seq_tensor, lengths=torch.tensor([L])).item())
    print(f"Native: E={E_native:.4f}  Rg={rg_native:.1f}Å  contacts={len(ni)}")

    # Output directory
    mode_str = args.start_mode
    out_dir = Path(args.output_dir) / pdb_id / mode_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build list of (beta, trial_idx) pairs
    if args.multi_beta:
        beta_trials = [(beta, i) for i, beta in enumerate(args.multi_beta)]
        mode_desc = f"multi-β scan: {args.multi_beta}"
    else:
        beta_trials = [(args.equil_beta, i) for i in range(args.n_trials)]
        mode_desc = f"{args.n_trials} trials at β={args.equil_beta}"

    # Header
    print(f"\n{'='*66}")
    print(f"  MODEL TEST: {pdb_id} (L={L})")
    print(f"  Mode: {args.start_mode}  |  {mode_desc}")
    print(f"  Steps: {args.n_steps//1000}K", end="")
    if args.anneal_steps > 0 and args.start_mode != "native":
        print(f"  (anneal {args.anneal_steps//1000}K: β {args.beta_start}→{args.beta_end})", end="")
    print()
    print(f"{'='*66}")

    results = []
    t_start = time.time()

    for equil_beta, trial_idx in beta_trials:
        result = run_single_trial(
            model=model,
            seq_tensor=seq_tensor,
            R_native=R_native,
            native_np=native_coords,
            ni=ni,
            nj=nj,
            d0=d0,
            rg_native=rg_native,
            n_steps=args.n_steps,
            anneal_steps=args.anneal_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            equil_beta=equil_beta,
            step_size=args.step_size,
            force_cap=args.force_cap,
            start_mode=args.start_mode,
            trial_idx=trial_idx,
            log_every=args.log_every,
            minimize=args.minimize,
        )
        results.append(result)

    # Summary
    elapsed = time.time() - t_start

    if args.start_mode == "native":
        n_stable = sum(1 for r in results if r["stable"])
        print(f"\n{'='*66}")
        print(f"  BASIN STABILITY RESULTS: {pdb_id} (L={L})")
        print(f"{'─'*66}")
        for r in results:
            status = "✓ STABLE" if r["stable"] else "✗ UNSTABLE"
            print(f"  β={r['equil_beta']:<8.0f}  final_Q={r['final_q']:.3f}  " f"RMSD={r['final_rmsd']:.1f}  {status}")
        print(f"{'─'*66}")
        print(f"  Stable: {n_stable}/{len(results)}")
    else:
        n_folded = sum(1 for r in results if r["folded"])
        best_qs = [r["best_q"] for r in results]
        print(f"\n{'='*66}")
        print(f"  FOLDING RESULTS: {pdb_id} (L={L})")
        print(f"{'─'*66}")
        print(f"  Folded: {n_folded}/{len(results)} ({100*n_folded/len(results):.0f}%)")
        print(f"  Best Q per trial: {', '.join(f'{q:.3f}' for q in best_qs)}")
        print(f"  Mean best Q: {np.mean(best_qs):.3f} ± {np.std(best_qs):.3f}")
        print(f"  Max Q reached: {max(best_qs):.3f}")

    print(f"  Time: {elapsed/60:.1f} min ({elapsed/len(results)/60:.1f} min/trial)")
    print(f"{'='*66}")

    # Save results
    summary = {
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "L": L,
        "start_mode": args.start_mode,
        "n_trials": len(results),
        "n_steps": args.n_steps,
        "anneal_steps": args.anneal_steps,
        "step_size": args.step_size,
        "E_native": E_native,
        "elapsed_sec": elapsed,
        "trials": [{k: v for k, v in r.items() if k != "trajectory"} for r in results],
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {summary_path}")

    # Save trajectories
    for i, r in enumerate(results):
        traj = r["trajectory"]
        beta_str = f"_beta{r['equil_beta']:.0f}" if args.multi_beta else ""
        traj_path = out_dir / f"trajectory_trial{i:02d}{beta_str}.npz"
        np.savez(
            traj_path,
            steps=np.array(traj["steps"]),
            q=np.array(traj["q"]),
            rmsd=np.array(traj["rmsd"]),
            drmsd=np.array(traj["drmsd"]),
            rg_ratio=np.array(traj["rg_ratio"]),
            energy=np.array(traj["energy"]),
            beta=np.array(traj["beta"]),
            accept_rate=np.array(traj["accept_rate"]),
        )
    print(f"  Saved {len(results)} trajectories to {out_dir}")


if __name__ == "__main__":
    main()
