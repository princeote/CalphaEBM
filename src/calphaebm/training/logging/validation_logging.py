"""
Validation logging utilities for SC/full-stage evaluation.

Mirrors the formatting style of DiagnosticLogger (diagnostics.py) — structured
blocks with box-drawing characters, aligned columns, and consistent widths.

Called by self_consistent.py and full_stage.py after each round's basin eval.

Usage:
    vlog = ValidationLogger()
    vlog.log_eval_block(
        round_num=6, beta=100.0, n_steps=5000,
        results=results,           # list of per-structure dicts from eval_subprocess
        structures=structures,     # list of (R, seq, pdb_id, chain_id, L)
        summary=summary_dict,      # aggregated stats
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from calphaebm.utils.logging import get_logger

logger = get_logger()

W = 100  # block width — wider than diagnostics to fit all columns


class ValidationLogger:
    """Structured logging for basin stability evaluation results."""

    def log_eval_block(
        self,
        round_num: int,
        beta: float,
        n_steps: int,
        results: List[Dict[str, Any]],
        structures: List[Any],
        summary: Optional[Dict[str, Any]] = None,
        rama_corr: float = 0.0,
        dphi_corr: float = 0.0,
    ) -> Dict[str, Any]:
        """Emit one structured evaluation block.

        Args:
            round_num:  SC round number
            beta:       inverse temperature used for eval
            n_steps:    Langevin steps per structure
            results:    list of per-structure dicts from eval_subprocess
            structures: list of (R, seq, pdb_id, chain_id, L) tuples
            summary:    pre-computed summary dict (if None, computed here)
            rama_corr:  Ramachandran correlation
            dphi_corr:  Δφ correlation

        Returns:
            summary dict with all aggregated metrics
        """
        n_structs = len(results)
        n_errors = sum(1 for r in results if r.get("error"))
        n_ok = n_structs - n_errors

        # ── Header ────────────────────────────────────────────────────
        header = f" ROUND {round_num} EVALUATION — β={beta:.0f}, {n_steps} steps, {n_structs} proteins "
        logger.info("═" * W)
        logger.info(header.center(W))
        logger.info("─" * W)

        # ── Per-structure table ───────────────────────────────────────
        # Column header
        logger.info(
            "  %-10s  %3s  %7s  %6s  %5s  %4s  %6s  %7s  %5s  %7s",
            "structure",
            "L",
            "ΔE",
            "RMSD",
            "Q",
            "Rg%",
            "RMSF",
            "k64dR",
            "CO",
            "accept%",
        )
        logger.info("  " + "·" * (W - 4))

        # Collect arrays for aggregation
        e_deltas, rmsds, q_vals, rmsfs = [], [], [], []
        rg_ratios, q_afs, drmsd_afs = [], [], []
        k64drmsds, contact_orders = [], []
        accept_pcts = []
        e_relaxes = []  # E_minimized - E_pdb (negative = energy decreased during minimization)
        # Per-subterm energy (E/res for native, minimized, delta)
        _subterm_names = ("local", "secondary", "repulsion", "packing")
        _subterm_native = {t: [] for t in _subterm_names}
        _subterm_minimized = {t: [] for t in _subterm_names}
        _subterm_delta = {t: [] for t in _subterm_names}

        for i, res in enumerate(results):
            pdb_id = structures[i][2] if i < len(structures) and len(structures[i]) > 2 else "?"
            chain_id = structures[i][3] if i < len(structures) and len(structures[i]) > 3 else "?"
            tag = f"{pdb_id}/{chain_id}"

            if res.get("error"):
                logger.info("  %-10s  %3d  %s", tag, res.get("L", 0), "ERROR: " + str(res["error"]).split("\n")[0][:60])
                continue

            L = res.get("L", 0)
            ed = res.get("e_delta", 0.0)
            rm = res.get("rmsd", 99.0)
            q = res.get("q", 0.0)
            rg = res.get("rg_ratio", 1.0) * 100
            rf = res.get("rmsf", 0.0)
            kd = res.get("k64drmsd", 0.0)
            co = res.get("contact_order", 0.0)
            ac = res.get("accept_pct", 0.0)

            logger.info(
                "  %-10s  %3d  %+7.3f  %6.2f  %5.3f  %3.0f%%  %6.3f  %7.2f  %5.3f  %6.1f%%",
                tag,
                L,
                ed,
                rm,
                q,
                rg,
                rf,
                kd,
                co,
                ac,
            )

            # Minimization info (if available)
            min_steps = res.get("min_steps", 0)
            e_rlx = res.get("e_relax", 0.0)
            dr_min = res.get("drmsd_min", 0.0)
            mf = res.get("max_force", 0.0)
            if min_steps > 0:
                logger.info(
                    "  %-10s       minimized in %d steps: ΔE=%+.3f  dRMSD=%.2f  maxF=%.1f",
                    "",
                    min_steps,
                    e_rlx,
                    dr_min,
                    mf,
                )

            e_deltas.append(ed)
            rmsds.append(rm)
            q_vals.append(q)
            rmsfs.append(rf)
            rg_ratios.append(res.get("rg_ratio", 1.0))
            q_afs.append(res.get("q_af", 50.0))
            drmsd_afs.append(res.get("drmsd_af", 50.0))
            k64drmsds.append(kd)
            contact_orders.append(co)
            accept_pcts.append(res.get("accept_pct", 0.0))
            e_relaxes.append(res.get("e_relax", 0.0))

            # Per-subterm energies (model already returns E/res)
            for t in _subterm_names:
                e_nat = res.get(f"e_{t}_native", None)
                e_min = res.get(f"e_{t}_minimized", None)
                e_dlt = res.get(f"e_{t}_delta", None)
                if e_nat is not None:
                    _subterm_native[t].append(e_nat)
                    _subterm_minimized[t].append(e_min if e_min is not None else e_nat)
                    _subterm_delta[t].append(e_dlt if e_dlt is not None else 0.0)

        # ── Aggregation ───────────────────────────────────────────────
        if not e_deltas:
            logger.info("  No valid results")
            logger.info("═" * W)
            return {"error": "no valid results"}

        e_deltas = np.array(e_deltas)
        rmsds = np.array(rmsds)
        q_vals = np.array(q_vals)
        rmsfs = np.array(rmsfs)
        rg_ratios = np.array(rg_ratios)
        k64drmsds = np.array(k64drmsds)
        contact_orders = np.array(contact_orders)

        # ── Mean/Std rows ─────────────────────────────────────────────
        logger.info("  " + "·" * (W - 4))
        logger.info(
            "  %-10s  %3s  %+7.3f  %6.2f  %5.3f  %3.0f%%  %6.3f  %7.2f  %5.3f  %6.1f%%",
            "MEAN",
            "",
            float(e_deltas.mean()),
            float(rmsds.mean()),
            float(q_vals.mean()),
            float(rg_ratios.mean()) * 100,
            float(rmsfs.mean()),
            float(k64drmsds.mean()),
            float(contact_orders.mean()),
            float(np.mean(accept_pcts)) if accept_pcts else 0.0,
        )
        logger.info(
            "  %-10s  %3s  ±%6.3f  ±%5.2f  ±%4.3f  ±%2.0f%%  ±%5.3f  ±%6.2f  ±%4.3f  ±%5.1f%%",
            "STD",
            "",
            float(e_deltas.std()),
            float(rmsds.std()),
            float(q_vals.std()),
            float(rg_ratios.std()) * 100,
            float(rmsfs.std()),
            float(k64drmsds.std()),
            float(contact_orders.std()),
            float(np.std(accept_pcts)) if accept_pcts else 0.0,
        )

        # ── Summary block ─────────────────────────────────────────────
        logger.info("─" * W)

        e_delta_mean = float(e_deltas.mean())
        rmsd_mean = float(rmsds.mean())
        q_mean = float(q_vals.mean())
        rmsf_mean = float(rmsfs.mean())
        rg_pct_mean = float(rg_ratios.mean()) * 100
        k64drmsd_mean = float(k64drmsds.mean())
        co_mean = float(contact_orders.mean())
        rg_dev = abs(rg_pct_mean - 100.0)
        q_af_mean = float(np.mean(q_afs))
        drmsd_af_mean = float(np.mean(drmsd_afs))

        # Composite score (lower is better)
        # dRMSD_af replaces rg_af — topology-sensitive anti-funnel fraction
        composite = -q_mean + k64drmsd_mean / 2.0 + rg_dev / 2.0 + q_af_mean / 20.0 + drmsd_af_mean / 20.0

        accept_mean = float(np.mean(accept_pcts)) if accept_pcts else 0.0

        logger.info(
            "  Metrics:   Q=%.3f  RMSD=%.2f  Rg%%=%.0f%%  ΔE=%+.3f  RMSF=%.3f  accept=%.1f%%",
            q_mean,
            rmsd_mean,
            rg_pct_mean,
            e_delta_mean,
            rmsf_mean,
            accept_mean,
        )
        logger.info(
            "  Funnels:   Q_af=%.1f%%  dRMSD_af=%.1f%%",
            q_af_mean,
            drmsd_af_mean,
        )
        logger.info(
            "  Structure: k64dRMSD=%.2f  CO=%.3f",
            k64drmsd_mean,
            co_mean,
        )
        # Per-subterm energy composition (E/res native, minimized, and delta)
        _has_subterms = any(len(v) > 0 for v in _subterm_native.values())
        e_relax_mean = float(np.mean(e_relaxes)) if e_relaxes else 0.0
        if _has_subterms:
            parts_nat = []
            parts_min = []
            parts_dlt = []
            total_nat = 0.0
            total_min = 0.0
            total_dlt = 0.0
            for t in _subterm_names:
                if _subterm_native[t]:
                    mn = float(np.mean(_subterm_native[t]))
                    mm = float(np.mean(_subterm_minimized[t]))
                    md = float(np.mean(_subterm_delta[t]))
                    total_nat += mn
                    total_min += mm
                    total_dlt += md
                    label = {"local": "loc", "secondary": "ss", "repulsion": "rep", "packing": "pack"}[t]
                    parts_nat.append(f"{label}={mn:+.3f}")
                    parts_min.append(f"{label}={mm:+.3f}")
                    parts_dlt.append(f"{label}={md:+.3f}")
            logger.info("  Energy:    E/res native: total=%+.3f  %s", total_nat, "  ".join(parts_nat))
            logger.info("  Energy:    E/res minim.: total=%+.3f  %s", total_min, "  ".join(parts_min))
            logger.info("  EnergyΔ:   ΔE/res (sam−min): total=%+.3f  %s", total_dlt, "  ".join(parts_dlt))
            logger.info("  Relax:     ΔE/res (min−pdb):  total=%+.3f  (negative = energy decreased)", e_relax_mean)
            # Minimized structure metrics
            rmsd_mins = [r.get("rmsd_min", 0.0) for r in results if not r.get("error")]
            q_mins = [r.get("q_min", 1.0) for r in results if not r.get("error")]
            if rmsd_mins:
                logger.info(
                    "  Minimized: RMSD=%.2f  Q=%.3f  (model minimum vs PDB)",
                    float(np.mean(rmsd_mins)),
                    float(np.mean(q_mins)),
                )
        logger.info(
            "  Composite: %.3f  =  -Q(%.3f) + k64dR/2(%.3f) + Rg_dev/2(%.3f) + Q_af/20(%.3f) + dRMSD_af/20(%.3f)",
            composite,
            -q_mean,
            k64drmsd_mean / 2.0,
            rg_dev / 2.0,
            q_af_mean / 20.0,
            drmsd_af_mean / 20.0,
        )
        if rama_corr > 0 or dphi_corr > 0:
            logger.info("  Rama=%.3f  Δφ=%.3f", rama_corr, dphi_corr)
        logger.info(
            "  Errors:    %d/%d",
            n_errors,
            n_structs,
        )
        logger.info("═" * W)

        # Build subterm energy summary for return dict
        _energy_summary = {}
        for t in _subterm_names:
            if _subterm_native[t]:
                _energy_summary[f"e_{t}_native_mean"] = float(np.mean(_subterm_native[t]))
                _energy_summary[f"e_{t}_minimized_mean"] = float(np.mean(_subterm_minimized[t]))
                _energy_summary[f"e_{t}_delta_mean"] = float(np.mean(_subterm_delta[t]))

        return {
            "e_delta_mean": e_delta_mean,
            "e_delta_median": float(np.median(e_deltas)),
            "e_delta_neg_frac": float(np.mean(e_deltas < 0)),
            "e_relax_mean": e_relax_mean,
            "rmsd_mean": rmsd_mean,
            "q_mean": q_mean,
            "rmsf_mean": rmsf_mean,
            "rg_pct": rg_pct_mean,
            "k64drmsd_mean": k64drmsd_mean,
            "contact_order_mean": co_mean,
            "accept_mean": accept_mean,
            "composite": composite,
            "rama_corr": rama_corr,
            "dphi_corr": dphi_corr,
            "n_errors": n_errors,
            "q_af": q_af_mean,
            "drmsd_af": drmsd_af_mean,
            **_energy_summary,
        }

    def log_round_summary(
        self,
        round_num: int,
        n_negatives: int,
        neg_categories: Dict[str, int],
        eval_summary: Dict[str, Any],
        time_collect: float = 0.0,
        time_retrain: float = 0.0,
        time_eval: float = 0.0,
        best_score: float = 999.0,
        best_round: int = 0,
    ) -> None:
        """Log a compact round summary after eval completes."""
        total_min = time_collect + time_retrain + time_eval
        cat_str = ", ".join(f"{k}={v}" for k, v in sorted(neg_categories.items()))

        logger.info("─" * 66)
        logger.info("  ROUND %d SUMMARY  (%d min total)", round_num, int(total_min))
        logger.info("    Negatives: %d new (%s)", n_negatives, cat_str)
        logger.info(
            "    Basin eval: RMSD=%.2f  Q=%.3f  Rg%%=%.0f%%  ΔE=%+.3f  Score=%.3f",
            eval_summary.get("rmsd_mean", 0),
            eval_summary.get("q_mean", 0),
            eval_summary.get("rg_pct", 0),
            eval_summary.get("e_delta_mean", 0),
            eval_summary.get("composite", 999),
        )
        if time_collect > 0:
            logger.info(
                "    Time: collect=%dm  retrain=%dm  eval=%dm",
                int(time_collect),
                int(time_retrain),
                int(time_eval),
            )
        logger.info("    Best so far: Score=%.3f (round %d)", best_score, best_round)
        logger.info("─" * 66)
