"""Parallel basin stability evaluation — subprocess worker pool.

Self-contained: no imports from full_stage or self_consistent.
Launched by training_evaluation.eval_round() via subprocess.run() from a
fresh process with no CUDA/autograd state — fork-based multiprocessing is safe.

Each worker evaluates one protein independently:
  L-BFGS minimization → MALA trajectory → metrics (RMSD, Q, dRMSD, af%)

Usage (called internally by training_evaluation.py):
    python -m calphaebm.evaluation.eval_subprocess \
        --model-path /tmp/eval_model.pt \
        --structures-path /tmp/eval_structures.pt \
        --results-path /tmp/eval_results.pt \
        --n-workers 32 \
        --beta 100.0 \
        --n-steps 10000 \
        --sampler mala
"""
import argparse
import multiprocessing as mp

import numpy as np
import torch


def _eval_single_structure(args):
    """Worker: run short IC Langevin/MALA for one protein and return metrics.
    Args (unpacked from tuple):
        model:     TotalEnergy model (CPU)
        R:         (1, L, 3) initial Ca coordinates
        seq:       (1, L) amino acid sequence
        lengths:   (1,) chain length
        L:         int chain length
        step_size: Langevin step size
        beta:      inverse temperature
        force_cap: max force magnitude
        n_steps:   number of Langevin steps
        sampler:   "langevin" or "mala"
        pdb_id:    str protein identifier (for progress logging)
    Returns dict with:
        L, rmsd, q, rg_ratio, rmsf, e_delta, q_af, rg_af, theta, phi, error
    """
    torch.set_num_threads(1)
    model, R, seq, lengths, L, _step_size_unused, _beta_unused, force_cap, n_steps, sampler, pdb_id = args
    from calphaebm.evaluation.metrics import native_contact_set, q_smooth, rmsd_kabsch
    from calphaebm.simulation.backends import get_simulator

    try:
        R = R.detach().float()
        R_nat_np = R[0, :L].numpy()
        ni, nj, d0 = native_contact_set(R_nat_np)
        rg_nat = float(np.sqrt(((R_nat_np - R_nat_np.mean(0)) ** 2).sum(1).mean()))

        # L-dependent eval parameters — co-designed for constant exploration:
        #   beta      = L              (deeper well for longer chains)
        #   step_size = 1/(10·L²)     (stable MALA across all lengths)
        #   n_steps   = L²            (n_steps × step_size = const → same exploration)
        #                              capped at 50000 to bound wall time
        beta = float(L)
        step_size = 1.0 / (10.0 * float(L) ** 2)
        n_steps = min(50000, int(float(L) ** 2))
        print(
            f"  [{pdb_id}] L={L}  β={beta:.1f}  step_size={step_size:.2e}  " f"n_steps={n_steps}  sampler={sampler}",
            flush=True,
        )

        # Welford online RMSF accumulator
        _w_n = 0
        _w_mean = np.zeros((L, 3), dtype=np.float64)
        _w_M2 = np.zeros((L, 3), dtype=np.float64)

        # Snapshot accumulator for af% computation
        # Save (E, Q, Rg_ratio) at evenly spaced snapshots
        snapshot_every = max(n_steps // 50, 10)  # ~50 snapshots
        snapshots = []  # list of (E, Q, Rg_ratio)

        # Energy at raw PDB coordinates (before minimization)
        with torch.no_grad():
            E_pdb = float(model(R, seq, lengths=lengths).item())

        # ── Minimize PDB structure via L-BFGS in IC space ──────────
        from calphaebm.simulation.minimize import lbfgs_minimize

        _min_result = lbfgs_minimize(model, R, seq, lengths=lengths)
        R_min = _min_result["R_min"]
        E_minimized = _min_result["E_minimized"]
        E_relax = _min_result["E_relax"]
        min_steps_taken = _min_result["min_steps"]
        max_force = _min_result["max_force"]
        R_min_np = R_min[0, :L].detach().numpy()
        _d_nat = np.sqrt(((R_nat_np[:, None] - R_nat_np[None, :]) ** 2).sum(-1))
        _d_min = np.sqrt(((R_min_np[:, None] - R_min_np[None, :]) ** 2).sum(-1))
        _triu = np.triu_indices(L, k=1)
        drmsd_min = float(np.sqrt(np.mean((_d_nat[_triu] - _d_min[_triu]) ** 2)))

        def _drmsd(coords: np.ndarray) -> float:
            """Full-pairwise dRMSD to native (upper triangle, separation ≥ 1)."""
            _d = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
            return float(np.sqrt(np.mean((_d_nat[_triu] - _d[_triu]) ** 2)))

        q_min = q_smooth(R_min_np, ni, nj, d0)

        # Use minimized energy as reference for e_delta
        E_init = E_minimized

        q_init = q_smooth(R_nat_np, ni, nj, d0)
        rg_ratio_init = 1.0
        snapshots.append((E_init, q_init, rg_ratio_init, 0.0, drmsd_min))

        # Start trajectory from minimized structure (model's own minimum)
        # Re-create the main simulator from minimized coordinates
        sim = get_simulator(
            name=sampler,
            model=model,
            seq=seq,
            R_init=R_min.clone(),
            step_size=step_size,
            beta=beta,
            force_cap=force_cap,
            lengths=lengths,
        )
        R_current = R_min.clone()
        print_every = max(n_steps // 5, 500)  # ~5 progress lines per protein
        for step in range(1, n_steps + 1):
            R_current, _, _ = sim.step()
            if step % 10 == 0:
                _coords = R_current[0, :L].detach().numpy().astype(np.float64)
                _w_n += 1
                _delta = _coords - _w_mean
                _w_mean += _delta / _w_n
                _delta2 = _coords - _w_mean
                _w_M2 += _delta * _delta2

            # Save snapshot for af% computation + trajectory trend
            if step % snapshot_every == 0:
                _snap_coords = R_current[0, :L].detach().numpy()
                with torch.no_grad():
                    _snap_E = float(model(R_current, seq, lengths=lengths).item())
                _snap_q = q_smooth(_snap_coords, ni, nj, d0)
                _snap_rg = float(np.sqrt(((_snap_coords - _snap_coords.mean(0)) ** 2).sum(1).mean()))
                _snap_rg_ratio = _snap_rg / max(rg_nat, 1.0)
                _snap_rmsd = rmsd_kabsch(_snap_coords, R_nat_np)
                _snap_drmsd = _drmsd(_snap_coords)
                snapshots.append((_snap_E, _snap_q, _snap_rg_ratio, _snap_rmsd, _snap_drmsd))

            # Periodic progress during MALA — mirrors negative collection output
            if step % print_every == 0:
                _q_now = q_smooth(R_current[0, :L].detach().numpy(), ni, nj, d0)
                _acc_now = sim.acceptance_rate * 100 if hasattr(sim, "acceptance_rate") else 0.0
                _e_now = snapshots[-1][0] if snapshots else E_init
                print(
                    f"    [{pdb_id}] step {step:>5}/{n_steps}  "
                    f"Q={_q_now:.3f}  accept={_acc_now:.1f}%  E={_e_now:+.3f}",
                    flush=True,
                )

        coords_final = R_current[0, :L].detach().numpy()
        with torch.no_grad():
            E_final = float(model(R_current, seq, lengths=lengths).item())
        rmsd_val = rmsd_kabsch(coords_final, R_nat_np)
        q_val = q_smooth(coords_final, ni, nj, d0)
        rg = float(np.sqrt(((coords_final - coords_final.mean(0)) ** 2).sum(1).mean()))
        rmsf_val = 0.0
        if _w_n > 1:
            variance = _w_M2 / (_w_n - 1)
            rmsf_val = float(np.sqrt(variance.sum(axis=1)).mean())

        # ── Compute af% from Langevin snapshots ──────────────
        # Matches diagnostics.py logic: pairwise slopes with threshold
        n_qf_pairs = 0
        n_qf_anti = 0
        n_dr_pairs = 0
        n_dr_anti = 0
        n_snap = len(snapshots)
        for i in range(n_snap):
            for j in range(i + 1, n_snap):
                Ei, Qi, Rgi, RMSDi, dRi = snapshots[i]
                Ej, Qj, Rgj, RMSDj, dRj = snapshots[j]
                # Q-funnel: require |dQ| > 0.05 to avoid noise
                dQ = Qi - Qj
                dE = Ei - Ej
                if abs(dQ) > 0.05:
                    n_qf_pairs += 1
                    if dQ > 0 and dE > 0:  # i more native, but i has higher E
                        n_qf_anti += 1
                    elif dQ < 0 and dE < 0:  # j more native, but j has higher E
                        n_qf_anti += 1
                # dRMSD-funnel: lower dRMSD (more native) should have lower E
                d_delta = dRi - dRj  # positive = i has higher dRMSD (less native)
                if abs(d_delta) > 0.5:
                    n_dr_pairs += 1
                    if d_delta > 0 and dE <= 0:  # i less native but lower/equal E
                        n_dr_anti += 1
                    elif d_delta < 0 and dE >= 0:  # j less native but higher/equal E
                        n_dr_anti += 1

        q_af_pct = 100.0 * n_qf_anti / max(n_qf_pairs, 1)
        drmsd_af_pct = 100.0 * n_dr_anti / max(n_dr_pairs, 1)

        # Extract final ICs for Rama/dphi correlation
        from calphaebm.geometry.internal import bond_angles, torsions

        R_final_t = torch.tensor(coords_final, dtype=torch.float32).unsqueeze(0)
        theta_final = bond_angles(R_final_t).squeeze(0).numpy()
        phi_final = torsions(R_final_t).squeeze(0).numpy()

        # ── Per-subterm energy decomposition (#19) ───────────────
        subterms = {}
        try:
            with torch.no_grad():
                seq_1 = seq[:, :L]
                lens_1 = torch.tensor([L])
                R_final_1 = torch.tensor(coords_final, dtype=torch.float32).unsqueeze(0)
                R_nat_1 = torch.tensor(R_nat_np, dtype=torch.float32).unsqueeze(0)
                R_min_1 = R_min[:, :L].detach().float()
                for term_name in ("local", "repulsion", "secondary", "packing"):
                    term = getattr(model, term_name, None)
                    if term is not None:
                        e_final = float(term(R_final_1, seq_1, lengths=lens_1).item())
                        e_native = float(term(R_nat_1, seq_1, lengths=lens_1).item())
                        e_min = float(term(R_min_1, seq_1, lengths=lens_1).item())
                        subterms[f"e_{term_name}"] = e_final
                        subterms[f"e_{term_name}_native"] = e_native
                        subterms[f"e_{term_name}_minimized"] = e_min
                        subterms[f"e_{term_name}_delta"] = e_final - e_min
        except Exception:
            pass  # subterms remain empty — non-fatal

        # ── k64dRMSD (#6) ────────────────────────────────────────
        k64drmsd_val = 0.0
        try:
            from calphaebm.evaluation.metrics.rmsd import k_drmsd

            k64drmsd_val = k_drmsd(coords_final, R_nat_np, K=64, exclude=3)
        except Exception:
            pass

        # ── Contact order ────────────────────────────────────────
        co_rel, co_abs = 0.0, 0.0
        try:
            from calphaebm.evaluation.metrics.contacts import contact_order

            co_rel, co_abs, _ = contact_order(R_nat_np, cutoff=8.0, exclude=3)
        except Exception:
            pass

        # ── Trajectory trend detection ────────────────────────────
        # Split trajectory into thirds, check if RMSD is growing or plateaued
        traj_rmsds = [s[3] for s in snapshots]  # Kabsch RMSD — for trend detection
        n_traj = len(traj_rmsds)
        trend = "plateau"
        rmsd_early, rmsd_mid, rmsd_late = 0.0, 0.0, 0.0
        if n_traj >= 6:
            q1 = n_traj // 3
            q2 = 2 * n_traj // 3
            rmsd_early = float(np.mean(traj_rmsds[:q1]))
            rmsd_mid = float(np.mean(traj_rmsds[q1:q2]))
            rmsd_late = float(np.mean(traj_rmsds[q2:]))
            if rmsd_late > rmsd_mid * 1.05:
                trend = "growing"
            elif rmsd_late < rmsd_early * 0.95:
                trend = "contracting"
            else:
                trend = "plateau"

        # Final completion line
        _accept_pct = float(sim.acceptance_rate * 100) if hasattr(sim, "acceptance_rate") else 0.0
        print(
            f"  [{pdb_id}] L={L}  β={beta:.1f}  step_size={step_size:.2e}  "
            f"Q={q_val:.3f}  RMSD={rmsd_val:.2f}  "
            f"Rg%={rg/max(rg_nat,1)*100:.0f}%  ΔE={E_final-E_init:+.3f}  "
            f"accept={_accept_pct:.1f}%  min={min_steps_taken}",
            flush=True,
        )

        return {
            "L": L,
            "rmsd": rmsd_val,
            "q": q_val,
            "rg_ratio": rg / max(rg_nat, 1.0),
            "rmsf": rmsf_val,
            "e_delta": E_final - E_init,
            "error": None,
            "e_pdb": E_pdb,
            "e_minimized": E_init,
            "e_relax": E_relax,
            "drmsd_min": drmsd_min,
            "q_min": q_min,
            "min_steps": min_steps_taken,
            "max_force": max_force,
            "q_af": q_af_pct,
            "drmsd_af": drmsd_af_pct,
            "accept_pct": _accept_pct,
            "theta": theta_final,
            "phi": phi_final,
            # Trajectory trend
            "trend": trend,
            "rmsd_early": rmsd_early,
            "rmsd_mid": rmsd_mid,
            "rmsd_late": rmsd_late,
            # Per-subterm energies (#19)
            **subterms,
            # k64dRMSD (#6)
            "k64drmsd": k64drmsd_val,
            # Contact order
            "contact_order": co_rel,
            "abs_contact_order": co_abs,
            # Trajectory snapshots for 2D FES F(Q, RMSD)
            "traj_q": [s[1] for s in snapshots],
            "traj_rmsd": [s[3] for s in snapshots],
            "traj_drmsd": [s[4] for s in snapshots],
        }
    except Exception as e:
        import sys
        import traceback

        tb = traceback.format_exc()
        # Surface the traceback to stderr immediately so watcher logs show it
        pdb_tag = pdb_id if "pdb_id" in locals() else "?"
        sys.stderr.write(f"[eval_subprocess] [{pdb_tag}] WORKER ERROR:\n{tb}\n")
        sys.stderr.flush()
        return {
            "L": L,
            "rmsd": 99.0,
            "q": 0.0,
            "rg_ratio": 0.0,
            "rmsf": 0.0,
            "e_delta": 0.0,
            "error": f"{e}\n{tb}",
            "q_af": 50.0,
            "drmsd_af": 50.0,
            "theta": None,
            "phi": None,
        }


def main():
    parser = argparse.ArgumentParser(description="Basin stability evaluation subprocess")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--structures-path", required=True)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument(
        "--sampler",
        type=str,
        default="langevin",
        choices=["langevin", "mala"],
        help="Sampling algorithm (default: langevin)",
    )
    args = parser.parse_args()
    # Load model on CPU — never touches CUDA
    model_cpu = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model_cpu.eval()
    # Load structures
    data = torch.load(args.structures_path, map_location="cpu", weights_only=False)
    structures = data["structures"]
    # Build worker args — support both tuple formats
    worker_args = []
    for item in structures:
        if len(item) == 5:
            R, seq, pdb_id, chain_id, L = item
        elif len(item) == 3:
            R, seq, L = item
        else:
            raise ValueError(f"Unknown structure format: {len(item)} elements")
        worker_args.append(
            (
                model_cpu,
                R.unsqueeze(0),
                seq.unsqueeze(0),
                torch.tensor([L]),
                L,
                1e-3,
                args.beta,
                100.0,
                args.n_steps,
                args.sampler,
                pdb_id if len(item) >= 4 else f"protein_{len(worker_args)}",
            )
        )
    n_workers = min(len(worker_args), args.n_workers)
    print(
        f"[eval_subprocess] {len(worker_args)} structures, {n_workers} workers, "
        f"beta={args.beta}, steps={args.n_steps}, sampler={args.sampler}",
        flush=True,
    )
    # Fork is safe — fresh process, no CUDA, no autograd threads
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_eval_single_structure, worker_args)
    # Save results
    torch.save(results, args.results_path)
    n_ok = sum(1 for r in results if not r.get("error"))
    print(f"[eval_subprocess] Done. {n_ok}/{len(results)} OK", flush=True)

    # If any failed, print a summary with first error so watcher can see what's wrong
    if n_ok < len(results):
        import sys

        n_failed = len(results) - n_ok
        sys.stderr.write(f"[eval_subprocess] {n_failed}/{len(results)} workers FAILED\n")
        # Print first error — usually they're all the same pattern
        for i, r in enumerate(results):
            if r.get("error"):
                sys.stderr.write(f"[eval_subprocess] First failing worker (idx={i}):\n" f"{r['error']}\n")
                break
        sys.stderr.flush()


if __name__ == "__main__":
    main()
