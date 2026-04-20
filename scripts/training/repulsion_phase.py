"""Phase 2: Repulsion calibration on native structures.

Simple three-step process:
  1. Set λ_rep = 1.0
  2. Measure mean |E_rep/residue| on NATIVE structures (no perturbation)
  3. Set λ_rep = min(target / raw_mean, cap)

Target = 1/9 ≈ 0.111 E/res (equal contribution from each of 9 subterms).
Cap = 1.5 (Hessian safety — prevents repulsion from dominating DSM gradients).

IMPORTANT: Full-chain mode pads shorter chains to batch max length with (0,0,0).
All measurements must use `lengths` to mask out padded atoms.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from calphaebm.training.core.state import TrainingState
from calphaebm.utils.logging import get_logger

logger = get_logger()

N_SUBTERMS = 9  # θθ, Δφ, φφ, ram, hb_α, hb_β, rep, geom, contact
TARGET_PER_SUBTERM = 1.0 / N_SUBTERMS  # ≈ 0.1111


def _inv_softplus(y: float, eps: float = 1e-6) -> float:
    """Inverse of softplus: returns x such that softplus(x) = y."""
    y = max(y, eps)
    if y > 20.0:
        return y
    return math.log(math.exp(y) - 1.0)


def _measure_native_repulsion(
    model,
    train_loader,
    device,
    n_batches: int = 32,
) -> Dict[str, float]:
    """Measure repulsion statistics on native structures, padding-aware."""
    rep_mod = model.repulsion
    if rep_mod is None:
        return {"raw_mean": 0.0, "n_samples": 0}

    model.eval()
    data_iter = iter(train_loader)

    rep_abs_vals = []
    rep_raw_vals = []
    min_dists = []
    wall_count = 0
    total_pairs = 0

    with torch.no_grad():
        for _ in range(n_batches):
            try:
                R, seq, _, _, lengths = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                R, seq, _, _, lengths = next(data_iter)

            R = R.to(device)
            seq = seq.to(device)
            lengths = lengths.to(device)
            B, L, _ = R.shape

            # Repulsion energy — pass lengths so padding is excluded
            E_rep = rep_mod(R, seq, lengths=lengths)  # (B,)
            rep_abs_vals.extend([abs(float(e)) for e in E_rep])
            rep_raw_vals.extend([float(e) for e in E_rep])

            # Distance statistics — padding-aware
            diff = R.unsqueeze(2) - R.unsqueeze(1)  # (B, L, L, 3)
            dist = torch.sqrt((diff**2).sum(-1) + 1e-8)  # (B, L, L)

            idx = torch.arange(L, device=device)
            sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
            nonbond_triu = (sep > 3) & torch.triu(
                torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
            )  # (L, L)

            # Valid atom mask per sample: (B, L)
            valid = idx.unsqueeze(0) < lengths.unsqueeze(1)  # (B, L)
            # Valid pair mask: both atoms must be real (not padding)
            valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, L, L)
            # Combine: non-bonded, upper-triangle, both atoms valid
            full_mask = nonbond_triu.unsqueeze(0) & valid_pair  # (B, L, L)

            dist_nb = dist[full_mask]
            if dist_nb.numel() > 0:
                min_dists.append(float(dist_nb.min().item()))
                wall_count += int((dist_nb < 4.5).sum().item())
                total_pairs += dist_nb.numel()

    raw_mean = sum(rep_abs_vals) / len(rep_abs_vals) if rep_abs_vals else 0.0
    raw_signed = sum(rep_raw_vals) / len(rep_raw_vals) if rep_raw_vals else 0.0
    min_dist = min(min_dists) if min_dists else float("nan")
    wall_frac = wall_count / max(total_pairs, 1)

    return {
        "raw_mean": raw_mean,
        "raw_signed": raw_signed,
        "min_dist": min_dist,
        "wall_frac": wall_frac,
        "n_samples": len(rep_abs_vals),
    }


def run_repulsion_phase(
    trainer,
    config,
    train_loader,
    val_loader=None,
    native_structures=None,
    resume: Optional[str] = None,
):
    """Calibrate λ_rep: measure on native structures, set to target."""

    model = trainer.model
    n_measure_batches = 32
    lambda_cap = 1.5

    # ── Resume (load Phase 1 weights) ────────────────────────────
    if resume:
        if resume == "auto":
            resume = trainer.find_latest_checkpoint(config.name)
        if resume:
            state = trainer.load_checkpoint(resume, load_optimizer=False)
            logger.info("Loaded model weights from: %s", resume)

    # ── Check repulsion module ───────────────────────────────────
    rep_mod = getattr(model, "repulsion", None)
    if rep_mod is None:
        logger.error("No repulsion module found on model")
        return _make_state(trainer, config)

    initial_lambda = float(F.softplus(rep_mod._lambda_rep_raw).item())

    # ── Header ───────────────────────────────────────────────────
    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info("  Phase 2: Repulsion Calibration")
    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info("  Method:     Measure |E_rep/res| on native structures at λ=1,")
    logger.info("              set λ = target / raw_mean")
    logger.info("  Structures: Native (no perturbation, padding-aware)")
    logger.info("  Target:     %.4f E/res (= 1/%d subterms)", TARGET_PER_SUBTERM, N_SUBTERMS)
    logger.info("  λ cap:      %.1f (Hessian safety)", lambda_cap)
    logger.info("  Batches:    %d  (×8 chains = %d structures)", n_measure_batches, n_measure_batches * 8)
    logger.info("  Current λ:  %.4f", initial_lambda)
    logger.info("══════════════════════════════════════════════════════════════════")

    trainer.optimizer = None
    trainer.scheduler = None

    # ── Step 1: Measure at unit weight ───────────────────────────
    logger.info("")
    logger.info("  Step 1: Set λ_rep = 1.0, measure on native structures")
    logger.info("  ─────────────────────────────────────────────────────")

    rep_mod._lambda_rep_raw.data.fill_(_inv_softplus(1.0))

    stats_unit = _measure_native_repulsion(
        model,
        train_loader,
        trainer.device,
        n_batches=n_measure_batches,
    )

    raw_mean = stats_unit["raw_mean"]
    logger.info("    |E_rep/res| at λ=1.0:  %.6f  (%d structures)", raw_mean, stats_unit["n_samples"])
    logger.info("    E_rep/res (signed):     %+.6f", stats_unit["raw_signed"])
    logger.info("    Min non-bonded dist:    %.3f Å", stats_unit["min_dist"])
    logger.info("    Pairs < 4.5Å (wall):   %.3f%%", stats_unit["wall_frac"] * 100)

    # ── Step 2: Compute calibrated λ ─────────────────────────────
    logger.info("")
    logger.info("  Step 2: Compute calibrated λ_rep")
    logger.info("  ─────────────────────────────────────────────────────")

    if raw_mean < 1e-8:
        logger.warning("    Raw repulsion ≈ 0. No wall activity on native structures.")
        logger.warning("    Setting λ_rep = %.1f (cap)", lambda_cap)
        new_lambda = lambda_cap
    else:
        uncapped = TARGET_PER_SUBTERM / raw_mean
        new_lambda = min(uncapped, lambda_cap)
        logger.info("    target / raw = %.4f / %.6f = %.4f", TARGET_PER_SUBTERM, raw_mean, uncapped)
        if uncapped > lambda_cap:
            logger.info("    Capped: %.4f → %.4f (Hessian safety)", uncapped, lambda_cap)

    # Apply
    rep_mod._lambda_rep_raw.data.fill_(_inv_softplus(new_lambda))
    actual_lambda = float(F.softplus(rep_mod._lambda_rep_raw).item())
    logger.info("    λ_rep set to: %.4f (verified: %.4f)", new_lambda, actual_lambda)

    # ── Step 3: Verify ───────────────────────────────────────────
    logger.info("")
    logger.info("  Step 3: Verify at λ_rep = %.4f", actual_lambda)
    logger.info("  ─────────────────────────────────────────────────────")

    stats_final = _measure_native_repulsion(
        model,
        train_loader,
        trainer.device,
        n_batches=n_measure_batches,
    )

    expected = raw_mean * new_lambda
    logger.info("    |E_rep/res| measured:   %.6f", stats_final["raw_mean"])
    logger.info("    |E_rep/res| expected:   %.6f  (raw × λ)", expected)
    logger.info("    Target:                 %.4f", TARGET_PER_SUBTERM)
    ratio = stats_final["raw_mean"] / TARGET_PER_SUBTERM if TARGET_PER_SUBTERM > 0 else 0
    logger.info("    Ratio (measured/target): %.3f", ratio)

    # ── Summary ──────────────────────────────────────────────────
    logger.info("")
    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info("  Phase 2 Complete: Repulsion Calibrated")
    logger.info("──────────────────────────────────────────────────────────────────")
    logger.info("    λ_rep:   %.4f → %.4f", initial_lambda, actual_lambda)
    logger.info("    E_rep:   %.6f E/res  (target: %.4f)", stats_final["raw_mean"], TARGET_PER_SUBTERM)
    logger.info("    Min d:   %.3f Å  (native, non-bonded)", stats_final["min_dist"])
    logger.info("    Wall:    %.3f%% pairs < 4.5Å", stats_final["wall_frac"] * 100)
    logger.info("══════════════════════════════════════════════════════════════════")

    # ── Save ─────────────────────────────────────────────────────
    trainer.global_step += 1
    trainer.phase_step = 1
    trainer.current_loss = stats_final["raw_mean"]
    trainer.save_checkpoint(config.name, trainer.phase_step, trainer.current_loss)

    return _make_state(trainer, config)


def _make_state(trainer, config) -> TrainingState:
    return TrainingState(
        global_step=trainer.global_step,
        phase_step=getattr(trainer, "phase_step", 0),
        phase=config.name,
        losses={"loss": getattr(trainer, "current_loss", 0.0)},
        gates=trainer.model.get_gates() if hasattr(trainer.model, "get_gates") else {},
        best_composite_score=getattr(trainer, "best_composite_score", None),
        best_composite_score_initialized=getattr(trainer, "best_composite_score_initialized", False),
        best_val_step=getattr(trainer, "best_val_step", 0),
        early_stopping_counter=getattr(trainer, "early_stopping_counter", 0),
        validation_history=getattr(trainer, "validation_history", []),
        converged=False,
        convergence_step=None,
        convergence_info=None,
    )
