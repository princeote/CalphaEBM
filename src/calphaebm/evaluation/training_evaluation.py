# src/calphaebm/evaluation/training_evaluation.py
"""Training evaluation watcher — independent basin eval during full-stage training.

Entry point:
    run_watch(args)   called by evaluate.py --mode watch

Polls for round checkpoints saved by full_stage.py, runs basin stability eval
using BasinStabilityEvaluator, writes structured JSON results and a 2D free
energy surface F(Q, RMSD) derived from trajectory snapshots.

Imports shared utilities from basin_evaluation.py — no code duplication.

Output per round (in {ckpt_dir}/{prefix}/eval_results/):
    round_{N:03d}.json       — BetaResult scalars + per-structure summary
    round_{N:03d}_fes.npz    — 2D FES: F(Q,RMSD) = -log P, min shifted to 0
    round_{N:03d}.done       — empty sentinel file (training loop can poll)
    watcher.log              — timestamped watcher log
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from calphaebm.evaluation.core_evaluation import load_model, load_structures, structures_to_loader
from calphaebm.utils.logging import get_logger

logger = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────


def find_round_checkpoint(ckpt_dir: Path, prefix: str, round_num: int) -> Optional[Path]:
    """Find checkpoint for a completed training round.

    Searches in order (new standard first, legacy fallback):
      1. {prefix}/full-stage/full_round{N:03d}/  — full-stage (new)
      2. {prefix}/self-consistent/sc_round{N:03d}.pt  — SC (new)
      3. {prefix}/stage1_round{N:03d}/        — full-stage (legacy)
      4. {prefix}/self-consistent/round{N:03d}.pt     — SC (legacy)
    """
    base = ckpt_dir / prefix

    # New: full-stage
    round_dir = base / "full-stage" / f"full_round{round_num:03d}"
    if round_dir.exists():
        bests = sorted(round_dir.glob("*_best.pt"))
        if bests:
            return bests[-1]
        all_pts = sorted(round_dir.glob("*.pt"))
        if all_pts:
            return all_pts[-1]

    # New: SC
    sc_ckpt = base / "self-consistent" / f"sc_round{round_num:03d}.pt"
    if sc_ckpt.exists():
        return sc_ckpt

    # Legacy: full-stage
    round_dir = base / f"stage1_round{round_num:03d}"
    if round_dir.exists():
        bests = sorted(round_dir.glob("*_best.pt"))
        if bests:
            return bests[-1]
        all_pts = sorted(round_dir.glob("step*.pt"))
        if all_pts:
            return all_pts[-1]

    # Legacy: SC
    sc_ckpt = base / "self-consistent" / f"round{round_num:03d}.pt"
    if sc_ckpt.exists():
        return sc_ckpt

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Result paths
# ─────────────────────────────────────────────────────────────────────────────


def _json_path(results_dir: Path, n: int) -> Path:
    return results_dir / f"round_{n:03d}.json"


def _fes_path(results_dir: Path, n: int) -> Path:
    return results_dir / f"round_{n:03d}_fes.npz"


def _done_path(results_dir: Path, n: int) -> Path:
    return results_dir / f"round_{n:03d}.done"


# ─────────────────────────────────────────────────────────────────────────────
# 2D Free Energy Surface  F(Q, RMSD) = -log P(Q, RMSD)
# ─────────────────────────────────────────────────────────────────────────────


def compute_fes_from_results(
    raw_results: list,
    n_q: int = 50,
    n_rmsd: int = 50,
    q_range: Tuple[float, float] = (0.0, 1.05),
    rmsd_range: Tuple[float, float] = (0.0, 15.0),
) -> Optional[dict]:
    """Compute 2D FES from eval_subprocess result dicts.

    Each result dict contains traj_q and traj_rmsd (lists of trajectory
    snapshots added by eval_subprocess). Pools all snapshots across
    proteins into a single 2D histogram.

    F(Q, RMSD) = -log P(Q, RMSD), shifted so minimum = 0.
    """
    q_pts, r_pts = [], []
    for r in raw_results:
        if r.get("error"):
            continue
        tq = r.get("traj_q", [])
        tr = r.get("traj_rmsd", [])
        if len(tq) > 0 and len(tr) == len(tq):
            q_pts.append(np.asarray(tq))
            r_pts.append(np.asarray(tr))

    if not q_pts:
        return None

    q_all = np.concatenate(q_pts)
    r_all = np.concatenate(r_pts)

    q_edges = np.linspace(q_range[0], q_range[1], n_q + 1)
    r_edges = np.linspace(rmsd_range[0], rmsd_range[1], n_rmsd + 1)

    counts, _, _ = np.histogram2d(q_all, r_all, bins=[q_edges, r_edges])
    P = counts + 1.0  # pseudocount — avoids log(0)
    P /= P.sum()
    F = -np.log(P)
    F -= F.min()  # shift so native basin = 0

    return {
        "Q_edges": q_edges,
        "RMSD_edges": r_edges,
        "F": F,
        "n_samples": int(len(q_all)),
    }


# ─────────────────────────────────────────────────────────────────────────────


def _beta_result_to_dict(br, round_num: int, elapsed_min: float) -> dict:
    per = [
        {
            "pdb_id": sr.pdb_id,
            "chain_id": sr.chain_id,
            "L": sr.length,
            "E_delta": sr.E_delta,
            "rmsd": sr.rmsd,
            "drmsd": sr.drmsd,
            "q": sr.q,
            "rmsf": sr.rmsf,
            "rmsd_min": sr.rmsd_min,
            "q_start": sr.q_start,
            "rmsd_to_native": sr.rmsd_to_native,
            "recovered": sr.recovered,
            "min_converged": sr.min_converged,
            "min_steps": sr.min_steps_used,
        }
        for sr in br.per_structure
    ]

    return {
        "round": round_num,
        "beta": br.beta,
        "n_structures": br.n_structures,
        "E_delta_mean": br.E_delta_mean,
        "E_delta_median": br.E_delta_median,
        "E_delta_p05": br.E_delta_p05,
        "E_delta_p95": br.E_delta_p95,
        "E_delta_neg_frac": br.E_delta_neg_frac,
        "rmsd_mean": br.rmsd_mean,
        "rmsd_p05": br.rmsd_p05,
        "rmsd_p95": br.rmsd_p95,
        "drmsd_mean": br.drmsd_mean,
        "q_mean": br.q_mean,
        "q_p05": br.q_p05,
        "rmsf_mean": br.rmsf_mean,
        "is_stable": br.is_stable,
        "summary": br.summary_line(),
        "elapsed_min": round(elapsed_min, 2),
        "per_structure": per,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-round eval
# ─────────────────────────────────────────────────────────────────────────────


def eval_round(
    ckpt_path: Path,
    structures: list,
    round_num: int,
    beta: float,
    n_steps: int,
    n_workers: int,
    results_dir: Path,
    log_fn,
) -> bool:
    """Evaluate one round in parallel via eval_subprocess worker pool.

    Launches calphaebm.evaluation.eval_subprocess in a fresh subprocess.
    Each of n_workers CPU cores evaluates one protein simultaneously.
    Wall-clock ≈ single-protein time regardless of how many are evaluated.
    """
    import copy
    import gc
    import subprocess
    import sys
    import tempfile

    log_fn(
        f"Round {round_num}: {ckpt_path.name}  " f"({len(structures)} proteins × {n_steps} steps, {n_workers} workers)"
    )
    t0 = time.time()

    try:
        with tempfile.TemporaryDirectory(prefix="eval_round_") as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            structures_path = Path(tmpdir) / "structures.pt"
            raw_results_path = Path(tmpdir) / "results.pt"

            # Load model onto CPU, save for subprocess
            model = load_model(ckpt_path, device=torch.device("cpu"))
            model.eval()
            torch.save(model, str(model_path))
            del model
            gc.collect()

            torch.save({"structures": structures}, str(structures_path))

            cmd = [
                sys.executable,
                "-m",
                "calphaebm.evaluation.eval_subprocess",
                "--model-path",
                str(model_path),
                "--structures-path",
                str(structures_path),
                "--results-path",
                str(raw_results_path),
                "--n-workers",
                str(n_workers),
                "--beta",
                str(beta),
                "--n-steps",
                str(n_steps),
                "--sampler",
                "mala",
            ]
            proc = subprocess.run(cmd, capture_output=False, text=True)

            if proc.returncode != 0:
                log_fn(f"Round {round_num}: eval_subprocess FAILED (rc={proc.returncode})")
                return False
            if not raw_results_path.exists():
                log_fn(f"Round {round_num}: results file missing")
                return False

            raw_results = torch.load(str(raw_results_path), map_location="cpu", weights_only=False)

        elapsed = (time.time() - t0) / 60
        ok = [r for r in raw_results if not r.get("error")]
        log_fn(f"Round {round_num}: done in {elapsed:.1f} min  {len(ok)}/{len(raw_results)} OK")

        if not ok:
            log_fn(f"Round {round_num}: all structures failed")
            # Log the first error so we can actually see what's wrong
            for i, r in enumerate(raw_results):
                err = r.get("error")
                if err:
                    # Keep it to the last ~30 lines so log stays readable
                    err_tail = "\n".join(str(err).splitlines()[-30:])
                    log_fn(f"Round {round_num}: first worker error (idx={i}):\n{err_tail}")
                    break
            return False

        # ── Aggregate ────────────────────────────────────────────────────
        def _mean(key, default=0.0):
            v = [r.get(key, default) for r in ok if r.get(key) is not None]
            return float(np.mean(v)) if v else default

        # ── Backbone geometry validation ─────────────────────────────────
        # Jensen-Shannon divergence of simulated θ/φ distributions vs PDB reference
        # Aggregated over all val proteins per round.

        def _energy_to_prob(E):
            """Convert energy surface to probability: P ∝ exp(-E), normalized."""
            E = np.asarray(E, dtype=np.float64)
            E_shifted = E - E.min()
            P = np.exp(-E_shifted)
            P /= P.sum()
            return P

        def _js_divergence(p, q):
            """Jensen-Shannon divergence between two discrete distributions."""
            p = np.asarray(p, dtype=np.float64).ravel()
            q = np.asarray(q, dtype=np.float64).ravel()
            # Ensure valid distributions
            p = np.clip(p, 1e-12, None)
            q = np.clip(q, 1e-12, None)
            p /= p.sum()
            q /= q.sum()
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * np.log(p / m))
            kl_qm = np.sum(q * np.log(q / m))
            return float(0.5 * (kl_pm + kl_qm))

        def _load_ref_distributions(backbone_data_dir):
            """Load PDB reference distributions from analysis/backbone_geometry/data."""
            d = Path(backbone_data_dir)
            refs = {}
            try:
                # θ-φ Ramachandran reference
                theta_edges = np.load(d / "theta_edges_deg.npy")
                phi_edges = np.load(d / "phi_edges_deg.npy")
                # Build from theta_phi_energy if available, else from histogram
                tp_energy_path = d / "theta_phi_energy.npy"
                if tp_energy_path.exists():
                    rama_E = np.load(tp_energy_path)
                    refs["rama_P"] = _energy_to_prob(rama_E)
                else:
                    # Fallback: try raw histogram
                    tp_hist = np.load(d / "theta_phi_hist.npy")
                    if tp_hist.sum() > 0:
                        refs["rama_P"] = (tp_hist + 1e-10) / (tp_hist.sum() + 1e-10 * tp_hist.size)
                if "rama_P" in refs:
                    refs["rama_theta_edges"] = theta_edges
                    refs["rama_phi_edges"] = phi_edges

                # φ_i vs φ_{i+1} correlation
                phi_phi_E = np.load(d / "phi_phi_energy.npy")
                refs["phiphi_P"] = _energy_to_prob(phi_phi_E)
                refs["phiphi_phi_i_edges"] = np.load(d / "phi_i_edges_deg.npy")
                refs["phiphi_phi_ip1_edges"] = np.load(d / "phi_ip1_edges_deg.npy")

                # θ_i vs θ_{i+1} correlation
                theta_theta_E = np.load(d / "theta_theta_energy.npy")
                refs["thetatheta_P"] = _energy_to_prob(theta_theta_E)
                refs["thetatheta_theta_i_edges"] = np.load(d / "theta_i_edges_deg.npy")
                refs["thetatheta_theta_ip1_edges"] = np.load(d / "theta_ip1_edges_deg.npy")

                # Δφ distribution
                dphi_E = np.load(d / "delta_phi_energy.npy")
                refs["dphi_P"] = _energy_to_prob(dphi_E)
                refs["dphi_centers"] = np.load(d / "delta_phi_centers.npy")
            except Exception as exc:
                log_fn(f"  Warning: failed to load backbone reference data: {exc}")
            return refs

        def _compute_backbone_js(all_theta, all_phi, refs):
            """Compute JS divergence for Rama, φ-φ, θ-θ, Δφ distributions.

            Args:
                all_theta: list of 1D arrays (degrees), one per protein
                all_phi:   list of 1D arrays (degrees), one per protein
                refs:      dict from _load_ref_distributions
            Returns:
                dict with js_rama, js_phiphi, js_thetatheta, js_dphi
            """
            result = {}
            # Concatenate all angles across proteins
            th_all = np.concatenate([np.degrees(np.asarray(t)) for t in all_theta if t is not None])
            ph_all = np.concatenate([np.degrees(np.asarray(p)) for p in all_phi if p is not None])
            n_rama = min(len(th_all), len(ph_all))
            if n_rama < 10:
                return result
            th_rama = th_all[:n_rama]
            ph_rama = ph_all[:n_rama]

            # 1) JS(Rama): θ-φ joint distribution
            if "rama_P" in refs:
                th_edges = refs["rama_theta_edges"]
                ph_edges = refs["rama_phi_edges"]
                H_sim, _, _ = np.histogram2d(th_rama, ph_rama, bins=[th_edges, ph_edges])
                H_sim = H_sim + 1e-10  # smoothing
                P_sim = H_sim / H_sim.sum()
                result["js_rama"] = round(_js_divergence(P_sim, refs["rama_P"]), 4)

            # 2) JS(φ_i, φ_{i+1}): consecutive phi correlation
            if "phiphi_P" in refs and n_rama >= 2:
                phi_i = ph_rama[:-1]
                phi_ip1 = ph_rama[1:]
                ph_i_edges = refs["phiphi_phi_i_edges"]
                ph_ip1_edges = refs["phiphi_phi_ip1_edges"]
                H_sim, _, _ = np.histogram2d(phi_i, phi_ip1, bins=[ph_i_edges, ph_ip1_edges])
                H_sim = H_sim + 1e-10
                P_sim = H_sim / H_sim.sum()
                result["js_phiphi"] = round(_js_divergence(P_sim, refs["phiphi_P"]), 4)

            # 3) JS(θ_i, θ_{i+1}): consecutive theta correlation
            if "thetatheta_P" in refs and n_rama >= 2:
                theta_i = th_rama[:-1]
                theta_ip1 = th_rama[1:]
                th_i_edges = refs["thetatheta_theta_i_edges"]
                th_ip1_edges = refs["thetatheta_theta_ip1_edges"]
                H_sim, _, _ = np.histogram2d(theta_i, theta_ip1, bins=[th_i_edges, th_ip1_edges])
                H_sim = H_sim + 1e-10
                P_sim = H_sim / H_sim.sum()
                result["js_thetatheta"] = round(_js_divergence(P_sim, refs["thetatheta_P"]), 4)

            # 4) JS(Δφ): delta-phi distribution
            if "dphi_P" in refs and n_rama >= 2:
                dphi = ph_rama[1:] - ph_rama[:-1]
                # Wrap to [-180, 180]
                dphi = (dphi + 180) % 360 - 180
                centers = refs["dphi_centers"]
                bin_width = centers[1] - centers[0]
                edges = np.concatenate([centers - bin_width / 2, [centers[-1] + bin_width / 2]])
                H_sim, _ = np.histogram(dphi, bins=edges)
                H_sim = H_sim + 1e-10
                P_sim = H_sim / H_sim.sum()
                result["js_dphi"] = round(_js_divergence(P_sim, refs["dphi_P"]), 4)

            return result

        # Per-protein basin assignment (kept for per-protein reporting)
        def _basin_assignment(r):
            """Assign residues to Rama basins for per-protein reporting."""
            th = r.get("theta")
            ph = r.get("phi")
            if th is None or ph is None:
                return {}
            th_deg = np.degrees(np.asarray(th))
            ph_deg = np.degrees(np.asarray(ph))
            n = min(len(th_deg), len(ph_deg))
            if n < 4:
                return {}
            basin_centers = [
                (92.5, 55.0, "helix"),
                (117.5, -155.0, "sheet"),
                (92.5, 85.0, "PPII"),
                (122.5, -105.0, "turn"),
            ]
            counts = {"helix": 0, "sheet": 0, "PPII": 0, "turn": 0, "other": 0}
            for i in range(n):
                best_d2, best_b = 1e9, "other"
                for tc, pc, bname in basin_centers:
                    dth = th_deg[i] - tc
                    dph = ph_deg[i] - pc
                    dph = (dph + 180) % 360 - 180
                    d2 = dth**2 / (20**2) + dph**2 / (40**2)
                    if d2 < best_d2:
                        best_d2, best_b = d2, bname
                if best_d2 < 4.0:
                    counts[best_b] += 1
                else:
                    counts["other"] += 1
            total = max(sum(counts.values()), 1)
            return {f"rama_{k}_frac": round(v / total, 3) for k, v in counts.items()}

        # Load reference distributions (once per round)
        _bb_data_dir = Path("analysis/backbone_geometry/data")
        _refs = _load_ref_distributions(_bb_data_dir)

        # ── Build per-structure records (pass-through all scalars) ────────
        # Skip large arrays (theta, phi, traj_*) but include everything else
        _skip_keys = {"theta", "phi", "traj_q", "traj_rmsd", "traj_drmsd", "error"}
        per_structure = []
        all_theta = []  # collect for aggregated JS
        all_phi = []
        for i, r in enumerate(raw_results):
            rec = {}
            # Add pdb_id/chain_id from structures list if not in result
            rec["pdb_id"] = r.get("pdb_id", structures[i][2] if i < len(structures) else "?")
            rec["chain_id"] = r.get("chain_id", structures[i][3] if i < len(structures) else "?")
            # Pass through all scalar fields
            for k, v in r.items():
                if k in _skip_keys or k in ("pdb_id", "chain_id"):
                    continue
                # Convert numpy scalars to Python floats
                if isinstance(v, (np.floating, np.integer)):
                    rec[k] = float(v)
                elif isinstance(v, (int, float, str, bool, type(None))):
                    rec[k] = v
                # Skip arrays/lists (traj data, theta/phi)
            # Add per-protein basin assignment
            bb = _basin_assignment(r)
            rec.update(bb)
            # Collect angles for aggregated JS
            if r.get("theta") is not None:
                all_theta.append(r["theta"])
                all_phi.append(r["phi"])
            # Add error flag
            if r.get("error"):
                rec["error"] = True
            per_structure.append(rec)

        # ── Aggregated backbone JS divergence ────────────────────────────
        js_metrics = _compute_backbone_js(all_theta, all_phi, _refs) if _refs else {}

        # Helper for means over per_structure dicts (not raw_results)
        def _mean_struct(key, structs):
            v = [s.get(key) for s in structs if s.get(key) is not None]
            return round(float(np.mean(v)), 3) if v else 0.0

        summary = {
            "round": round_num,
            "beta": beta,
            "n_structures": len(ok),
            "n_total": len(raw_results),
            "elapsed_min": round(elapsed, 2),
            # Core stability metrics
            "q_mean": _mean("q"),
            "rmsd_mean": _mean("rmsd"),
            "rg_pct_mean": _mean("rg_ratio") * 100,
            "e_delta_mean": _mean("e_delta"),
            "rmsf_mean": _mean("rmsf"),
            # Funnel metrics
            "q_af_mean": _mean("q_af"),
            "drmsd_af_mean": _mean("drmsd_af"),
            # Structural quality
            "k64drmsd_mean": _mean("k64drmsd"),
            "contact_order_mean": _mean("contact_order"),
            "accept_pct_mean": _mean("accept_pct"),
            # Energy decomposition (means over val proteins)
            "e_local_mean": _mean("e_local"),
            "e_repulsion_mean": _mean("e_repulsion"),
            "e_secondary_mean": _mean("e_secondary"),
            "e_packing_mean": _mean("e_packing"),
            "e_local_delta_mean": _mean("e_local_delta"),
            "e_repulsion_delta_mean": _mean("e_repulsion_delta"),
            "e_secondary_delta_mean": _mean("e_secondary_delta"),
            "e_packing_delta_mean": _mean("e_packing_delta"),
            # Backbone geometry (means over per-protein basins)
            "rama_helix_frac_mean": _mean_struct("rama_helix_frac", per_structure),
            "rama_sheet_frac_mean": _mean_struct("rama_sheet_frac", per_structure),
            "rama_other_frac_mean": _mean_struct("rama_other_frac", per_structure),
            # Jensen-Shannon divergences vs PDB reference (aggregated over all val proteins)
            **js_metrics,
            # Stability flags
            "e_delta_neg_frac": float(np.mean([r.get("e_delta", 0) < 0 for r in ok])),
            "is_stable": _mean("rmsd") < 5.0 and abs(_mean("e_delta")) < 0.3,
            "per_structure": per_structure,
        }

        log_fn(
            f"  Q={summary['q_mean']:.3f}  RMSD={summary['rmsd_mean']:.2f}  "
            f"ΔE={summary['e_delta_mean']:+.3f}  "
            f"Q_af={summary['q_af_mean']:.1f}%  dRMSD_af={summary['drmsd_af_mean']:.1f}%"
        )
        log_fn(
            f"  accept={summary['accept_pct_mean']:.1f}%  "
            f"Rg%={summary['rg_pct_mean']:.0f}%  "
            f"RMSF={summary['rmsf_mean']:.2f}  "
            f"k64dRMSD={summary['k64drmsd_mean']:.2f}"
        )
        if summary.get("e_local_mean") is not None:
            log_fn(
                f"  E/res: local={summary['e_local_mean']:.3f}  "
                f"rep={summary['e_repulsion_mean']:.3f}  "
                f"ss={summary['e_secondary_mean']:.3f}  "
                f"pack={summary['e_packing_mean']:.3f}"
            )
        if summary.get("rama_helix_frac_mean"):
            log_fn(
                f"  Rama: helix={summary['rama_helix_frac_mean']:.1%}  "
                f"sheet={summary['rama_sheet_frac_mean']:.1%}  "
                f"other={summary['rama_other_frac_mean']:.1%}"
            )
        if js_metrics:
            parts = []
            for k in ("js_rama", "js_phiphi", "js_thetatheta", "js_dphi"):
                if k in js_metrics:
                    parts.append(f"{k}={js_metrics[k]:.4f}")
            if parts:
                log_fn(f"  JS vs PDB: {', '.join(parts)}")

        with open(_json_path(results_dir, round_num), "w") as f:
            json.dump(summary, f, indent=2)

        fes = compute_fes_from_results(raw_results)
        if fes:
            np.savez(
                _fes_path(results_dir, round_num),
                Q_edges=fes["Q_edges"],
                RMSD_edges=fes["RMSD_edges"],
                F=fes["F"],
                Q_native=1.0,
                RMSD_native=0.0,
                n_samples=fes["n_samples"],
                round_num=round_num,
                beta=beta,
            )
            log_fn(f"  FES saved ({fes['n_samples']} trajectory snapshots)")

        _done_path(results_dir, round_num).touch()
        log_fn(f"  round_{round_num:03d}.done written")
        return True

    except Exception as exc:
        import traceback

        log_fn(f"Round {round_num}: FAILED — {exc}")
        log_fn(traceback.format_exc())
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_watch(args) -> int:
    """Run detached training eval watcher (--mode watch).

    Polls for new round checkpoints, evaluates each on CPUs independently
    of the GPU training process.
    """
    if not args.ckpt_dir:
        logger.error("--ckpt-dir is required for --mode watch")
        return 1
    if not args.ckpt_prefix:
        logger.error("--ckpt-prefix is required for --mode watch")
        return 1
    if not args.pdb:
        logger.error("--pdb is required for --mode watch (pass val_hq.txt)")
        return 1

    ckpt_dir = Path(args.ckpt_dir)
    results_dir = ckpt_dir / args.ckpt_prefix / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "watcher.log"

    # Timestamped logger that writes to both stdout and watcher.log
    def _wlog(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} | WATCHER | {msg}"
        logger.info(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    beta = args.beta[0] if isinstance(args.beta, list) else args.beta

    _wlog("=" * 60)
    _wlog("calphaebm evaluate --mode watch")
    _wlog(f"  ckpt:     {ckpt_dir}/{args.ckpt_prefix}")
    _wlog(f"  pdb:      {args.pdb[0] if isinstance(args.pdb, list) else args.pdb}")
    _wlog(f"  eval:     {args.n_samples} proteins × {args.n_steps} steps × β={beta}")
    _wlog(f"  rounds:   {args.start_round}..{args.max_rounds}  poll={args.poll_interval}s")
    _wlog("=" * 60)

    # Resolve pdb source — watch mode always gets a file path
    pdb_source = args.pdb[0] if isinstance(args.pdb, list) else args.pdb

    _wlog(f"Loading val structures from {pdb_source}")
    structures = load_structures(
        pdb_source=pdb_source,
        cache_dir=args.cache_dir,
        n_samples=args.n_samples,
        max_len=args.max_len,
    )
    _wlog(f"Loaded {len(structures)} val structures")

    if not structures:
        _wlog("ERROR: no structures loaded — check --pdb and --cache-dir")
        return 1

    current = args.start_round
    n_done = 0

    while current <= args.max_rounds:
        done = _done_path(results_dir, current)

        if done.exists():
            _wlog(f"Round {current}: already evaluated — skipping")
            current += 1
            n_done += 1
            continue

        ckpt = find_round_checkpoint(ckpt_dir, args.ckpt_prefix, current)
        if ckpt is None:
            _wlog(f"Round {current}: waiting for checkpoint " f"(poll every {args.poll_interval}s) ...")
            time.sleep(args.poll_interval)
            continue

        ok = eval_round(
            ckpt_path=ckpt,
            structures=structures,
            round_num=current,
            beta=beta,
            n_steps=args.n_steps,
            n_workers=args.n_samples,  # one worker per protein
            results_dir=results_dir,
            log_fn=_wlog,
        )

        if ok:
            n_done += 1
            current += 1
        else:
            time.sleep(args.poll_interval)

    _wlog(f"Watcher complete — {n_done} rounds evaluated")
    _wlog(f"Results in: {results_dir}")
    return 0
