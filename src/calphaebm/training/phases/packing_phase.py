"""Phase 4: Packing term contrastive pre-training.

Trains two packing subterms:
  E_geom    — per-residue geometry MLP (burial depth, neighbor counts)
  E_contact — rank-1 SVD pair potential (amino acid contact preferences)

CHANGES from run32 (dynamics-focused):
  - 100% IC-perturbed negatives (same sequence, perturbed backbone).
    Sequence-shuffle removed — irrelevant for dynamics, only useful for
    sequence design which is not the current target.
  - σ range widened to [0.05, 2.0] rad (was [0.02, 0.30]) to match
    the full range that IC-space Langevin dynamics explores.
  - Log-uniform σ sampling ensures equal gradient contribution at every scale.

Both λ_geom and λ_contact are frozen at 1.0 during this phase;
the packing gate ramps from start→end over ramp_steps.
Local, repulsion, secondary are frozen.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.training.core.state import TrainingState
from calphaebm.training.losses.contrastive_losses import contrastive_logistic_loss
from calphaebm.utils.logging import ProgressBar, get_logger

logger = get_logger()

# PDB-derived ratio: std(θ) / std(φ) across training chains
THETA_PHI_RATIO = 0.161


# ── Helpers ──────────────────────────────────────────────────────────────────


def _inv_softplus(y: float, eps: float = 1e-8) -> float:
    y = float(max(y, eps))
    return float(torch.log(torch.expm1(torch.tensor(y, dtype=torch.float64))).item())


def _freeze_lambda(param, target: float, name: str = "") -> None:
    """Set a softplus-parameterised λ to *target* and freeze it."""
    raw = _inv_softplus(max(target, 1e-6))
    with torch.no_grad():
        param.fill_(raw)
        param.requires_grad_(False)
    logger.debug("  Froze %s at %.4f (raw=%.4f)", name, target, raw)


def _freeze_packing_lambdas(model, target: float = 1.0) -> None:
    """Freeze **both** λ_geom and λ_contact at *target*."""
    pack = getattr(model, "packing", None)
    if pack is None:
        return
    if hasattr(pack, "_lambda_pack_raw"):
        _freeze_lambda(pack._lambda_pack_raw, target, "λ_geom")
    burial = getattr(pack, "burial", None)
    if burial is not None and hasattr(burial, "_lambda_hp_raw"):
        _freeze_lambda(burial._lambda_hp_raw, target, "λ_contact")


def _read_packing_weights(model) -> dict:
    """Read packing lambda values."""
    w = {}
    pack = getattr(model, "packing", None)
    if pack is None:
        return w
    if hasattr(pack, "lambda_pack"):
        w["geom"] = float(pack.lambda_pack.item())
    if hasattr(pack, "burial") and hasattr(pack.burial, "lambda_hp"):
        w["contact"] = float(pack.burial.lambda_hp.item())
    return w


def _mean_real_length(lengths: torch.Tensor) -> float:
    """Mean real chain length across batch (for per-residue normalisation)."""
    return max(float(lengths.float().mean().item()), 1.0)


def _sample_sigmas(B: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    """Per-sample log-uniform σ: returns (B, 1) for broadcasting."""
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    return torch.empty(B, 1, device=device).uniform_(log_min, log_max).exp()


def _ic_perturb(
    R: torch.Tensor, sigma_min: float, sigma_max: float, lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """IC-perturb a batch: returns R_neg with bonds at 3.8 Å.

    Each structure gets its own log-uniform σ so every batch covers the
    full perturbation range.  Both θ and φ are perturbed with the PDB-derived ratio.
    Padding atoms (beyond lengths) receive zero noise.
    """
    with torch.no_grad():
        theta, phi = coords_to_internal(R)
        anchor = extract_anchor(R)
        B = R.shape[0]

        sigmas = _sample_sigmas(B, sigma_min, sigma_max, R.device)  # (B, 1)
        noise_t = THETA_PHI_RATIO * sigmas * torch.randn_like(theta)
        noise_p = sigmas * torch.randn_like(phi)

        # Mask padding
        if lengths is not None:
            Lt = theta.shape[1]
            Lp = phi.shape[1]
            idx_t = torch.arange(Lt, device=R.device).unsqueeze(0)
            idx_p = torch.arange(Lp, device=R.device).unsqueeze(0)
            valid_t = idx_t < (lengths.unsqueeze(1) - 2)
            valid_p = idx_p < (lengths.unsqueeze(1) - 3)
            noise_t = noise_t * valid_t.float()
            noise_p = noise_p * valid_p.float()

        theta_neg = (theta + noise_t).clamp(0.01, math.pi - 0.01)
        phi_neg = phi + noise_p
        phi_neg = (phi_neg + math.pi) % (2 * math.pi) - math.pi

        return nerf_reconstruct(theta_neg, phi_neg, anchor)


def _generate_negatives(
    model,
    R: torch.Tensor,
    seq: torch.Tensor,
    lengths: torch.Tensor,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
) -> Optional[torch.Tensor]:
    """100% IC-perturbed negatives (same sequence, perturbed backbone).

    Each structure gets its own per-sample log-uniform σ.
    All calls to ``model.packing`` pass *lengths* so padding is masked.
    """
    try:
        R_neg = _ic_perturb(R, sigma_min, sigma_max, lengths)
        E_neg = model.packing(R_neg, seq, lengths=lengths)
        if torch.isfinite(E_neg).all():
            return E_neg
    except Exception:
        pass

    return None


# ── Main phase runner ────────────────────────────────────────────────────────


def run_packing_phase(trainer, config, train_loader, val_loader=None, native_structures=None, resume=None):
    """Run packing phase: contrastive training with IC-perturbed negatives."""

    if getattr(config, "packing_pretrain", False):
        raise RuntimeError(
            "packing_pretrain=True not supported in this packing_phase.py. "
            "Use contrastive mode (default) for the phased pipeline."
        )

    model = trainer.model
    pack = getattr(model, "packing", None)
    if pack is None:
        raise RuntimeError("No packing term found in model.")

    # ── Configuration ────────────────────────────────────────────
    n_steps = int(config.n_steps)
    lr = float(config.lr)
    lr_final = float(getattr(config, "lr_final", lr * 0.1))
    weight_decay = float(getattr(config, "weight_decay", 0.0))

    # σ range (matching Langevin exploration)
    sigma_min = float(getattr(config, "sigma_min_rad", 0.05))
    sigma_max = float(getattr(config, "sigma_max_rad", 2.0))

    # Gate ramp
    ramp_start = getattr(config, "ramp_pack_start", None)
    ramp_end = getattr(config, "ramp_pack_end", None)
    ramp_steps = int(getattr(config, "ramp_steps", 0) or 0)
    has_ramp = ramp_start is not None and ramp_end is not None and ramp_steps > 0
    if has_ramp:
        ramp_start = float(ramp_start)
        ramp_end = float(ramp_end)

    # ── Freeze both packing lambdas equally ──────────────────────
    _freeze_packing_lambdas(model, target=1.0)

    # Trainable: packing params only (excluding both frozen lambdas)
    frozen_names = {"_lambda_pack_raw"}
    burial = getattr(pack, "burial", None)
    if burial is not None and hasattr(burial, "_lambda_hp_raw"):
        frozen_names.add("burial._lambda_hp_raw")

    trainable = [(n, p) for n, p in pack.named_parameters() if p.requires_grad and n not in frozen_names]
    trainable_params = [p for _, p in trainable]
    n_params = sum(p.numel() for p in trainable_params)

    logger.info("Trainable parameters: %d tensors, %s params", len(trainable_params), f"{n_params:,}")

    is_fresh = (not resume) or (resume == "auto" and trainer.find_latest_checkpoint(config.name) is None)
    if is_fresh:
        for name, p in trainable:
            logger.info("  - %s: %s", name, tuple(p.shape))

    if not trainable_params:
        raise RuntimeError("No trainable packing parameters")

    trainer.optimizer = torch.optim.AdamW(trainable_params, lr=lr * 0.5, weight_decay=weight_decay)

    # LR scheduler
    trainer.scheduler = None
    if getattr(config, "lr_schedule", None) == "cosine" and n_steps > 0:
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=n_steps, eta_min=lr_final * 0.5
        )

    # ── Log header ───────────────────────────────────────────────
    init_w = _read_packing_weights(model)
    logger.info("Initial lambdas: %s  (both frozen at 1.0)", "  ".join(f"{k}={v:.4f}" for k, v in init_w.items()))
    if has_ramp:
        logger.info("Gate ramp: %.4f → %.4f over %d steps", ramp_start, ramp_end, ramp_steps)
    logger.info("Negatives: 100%% IC-perturbed (dynamics-focused)")
    logger.info("σ range: [%.3f, %.3f] rad (log-uniform), θ ratio=%.3f", sigma_min, sigma_max, THETA_PHI_RATIO)

    # ── Resume ───────────────────────────────────────────────────
    start_step = 0
    if resume:
        if resume == "auto":
            resume = trainer.find_latest_checkpoint(config.name)
        if resume:
            state = trainer.load_checkpoint(resume, load_optimizer=True)
            start_step = int(state.phase_step)
            logger.info("Resumed from phase step %d (global=%d)", start_step, trainer.global_step)

    # Set initial gate
    if has_ramp:
        model.set_gates(packing=ramp_start)

    # ── Training loop ────────────────────────────────────────────
    model.train()
    trainer._init_validators()
    data_iter = iter(train_loader)
    progress = ProgressBar(n_steps, prefix=f"Phase {config.name}")

    loss = None
    best_delta = float("-inf")
    ema_delta = 0.0
    ema_correct = 0.5
    alpha = 0.05  # EMA smoothing

    for phase_step in range(start_step + 1, n_steps + 1):
        trainer.global_step += 1
        trainer.phase_step = phase_step

        # Gate ramp
        if has_ramp:
            t = min(phase_step, ramp_steps) / max(ramp_steps, 1)
            gate_val = ramp_start + t * (ramp_end - ramp_start)
            model.set_gates(packing=gate_val)

        try:
            R, seq, _, _, lengths = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            R, seq, _, _, lengths = next(data_iter)

        R = R.to(trainer.device)
        seq = seq.to(trainer.device)
        lengths = lengths.to(trainer.device)
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        mean_L = _mean_real_length(lengths)

        try:
            # Positive: native (structure, sequence) — padding-aware
            E_pos = model.packing(R, seq, lengths=lengths)

            # Negative: IC-perturbed (same sequence) — padding-aware
            E_neg = _generate_negatives(model, R, seq, lengths, sigma_min=sigma_min, sigma_max=sigma_max)
            if E_neg is None:
                continue

            # Contrastive loss (full batch — no size mismatch since 100% IC)
            loss = contrastive_logistic_loss(E_pos, E_neg)

            if not torch.isfinite(loss):
                logger.error("Non-finite loss at step %d", phase_step)
                continue

            trainer.current_loss = loss.item()

            # Track gap (already per-residue from model normalization)
            with torch.no_grad():
                delta_per_res = float((E_neg - E_pos).mean().item())
                correct = float((E_pos < E_neg).float().mean().item())
                ema_delta = alpha * delta_per_res + (1 - alpha) * ema_delta
                ema_correct = alpha * correct + (1 - alpha) * ema_correct

            # Optimize
            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            has_nan = any(torch.isnan(p.grad).any() for p in trainable_params if p.grad is not None)
            if not has_nan:
                trainer.optimizer.step()
                if trainer.scheduler:
                    trainer.scheduler.step()

        except Exception as e:
            logger.error("Error at step %d: %s", phase_step, e)
            import traceback

            traceback.print_exc()
            continue

        # ── Summary every 50 steps ───────────────────────────────
        if phase_step % 50 == 0:
            weights = _read_packing_weights(model)
            gates = model.get_gates() if hasattr(model, "get_gates") else {}
            g_pack = float(gates.get("packing", 1.0))

            logger.info(
                "[%s] step %6d/%d (global=%d) | loss=%.4f | lr=%.2e | "
                "ΔE/res=%.4f (ema=%.4f) | correct=%.0f%% (ema=%.0f%%) | "
                "g_pack=%.3f | %s",
                config.name,
                phase_step,
                n_steps,
                trainer.global_step,
                trainer.current_loss,
                current_lr,
                delta_per_res,
                ema_delta,
                correct * 100,
                ema_correct * 100,
                g_pack,
                "  ".join(f"λ_{k}={v:.3f}" for k, v in weights.items()),
            )

        # ── Diagnostic block every 200 steps ─────────────────────
        if phase_step % 200 == 0:
            with torch.no_grad():
                weights = _read_packing_weights(model)
                gates = model.get_gates() if hasattr(model, "get_gates") else {}
                g_pack = float(gates.get("packing", 1.0))

                e_pos_per_res = float(E_pos.mean().item())
                e_neg_per_res = float(E_neg.mean().item())

            logger.info("══════════════════════════════════════════════════════════════════")
            logger.info(
                "           STEP %d/%d | loss=%.4f | lr=%.2e", phase_step, n_steps, trainer.current_loss, current_lr
            )
            logger.info("──────────────────────────────────────────────────────────────────")
            logger.info("  Lambdas:  %s  (both frozen)", "  ".join(f"{k}={v:.4f}" for k, v in weights.items()))
            logger.info(
                "  Gate:     g_pack=%.4f  (effective geom=%.4f  contact=%.4f)",
                g_pack,
                g_pack * weights.get("geom", 1.0),
                g_pack * weights.get("contact", 1.0),
            )
            logger.info("  Mode:     100%% IC-perturbed (dynamics-focused)")
            logger.info("──────────────────────────────────────────────────────────────────")
            logger.info("  E_pos/res (native):           %+.4f", e_pos_per_res)
            logger.info("  E_neg/res (IC-perturbed):      %+.4f", e_neg_per_res)
            logger.info("  ΔE/res (neg − pos):           %+.4f  (EMA: %+.4f)", delta_per_res, ema_delta)
            logger.info("  Correct (pos<neg):             %.1f%%  (EMA: %.1f%%)", correct * 100, ema_correct * 100)
            logger.info("  Loss (softplus):               %.6f", trainer.current_loss)
            logger.info("  Mean chain length:             %.0f residues", mean_L)

            if hasattr(pack, "mlp"):
                mlp_params = torch.cat([p.detach().flatten() for p in pack.mlp.parameters()])
                logger.info(
                    "  Geom MLP weights: mean=%.4f  std=%.4f  |max|=%.4f",
                    mlp_params.mean().item(),
                    mlp_params.std().item(),
                    mlp_params.abs().max().item(),
                )

            if hasattr(pack, "burial") and hasattr(pack.burial, "h"):
                h = pack.burial.h.detach()
                logger.info(
                    "  Contact h vector: mean=%.4f  std=%.4f  |max|=%.4f",
                    h.mean().item(),
                    h.std().item(),
                    h.abs().max().item(),
                )

            if ema_delta > best_delta:
                best_delta = ema_delta
                logger.info("  ★ New best EMA ΔE/res: %.4f", best_delta)

            logger.info("══════════════════════════════════════════════════════════════════")

        # ── Convergence check ────────────────────────────────────
        if phase_step % 100 == 0:
            if trainer._check_convergence():
                logger.info("Training converged at step %d", phase_step)
                trainer.save_checkpoint(config.name, phase_step, trainer.current_loss, is_best=True)
                break

        # ── Checkpoint ───────────────────────────────────────────
        save_every = int(getattr(config, "save_every", 0) or 0)
        if save_every > 0 and phase_step % save_every == 0:
            trainer.save_checkpoint(config.name, phase_step, trainer.current_loss)

        progress.update(1)

    # ── Final ────────────────────────────────────────────────────
    if not getattr(trainer, "converged", False):
        final_loss = getattr(trainer, "current_loss", float("nan"))
        trainer.save_checkpoint(config.name, trainer.phase_step, final_loss)

    final_w = _read_packing_weights(model)
    logger.info("Phase %s complete: %d/%d steps", config.name, phase_step, n_steps)
    logger.info("Final lambdas: %s  (both frozen)", "  ".join(f"{k}={v:.4f}" for k, v in final_w.items()))
    logger.info("Best EMA ΔE/res: %.4f  |  Final EMA correct: %.1f%%", best_delta, ema_correct * 100)

    return TrainingState(
        global_step=trainer.global_step,
        phase_step=trainer.phase_step,
        phase=config.name,
        losses={"loss": getattr(trainer, "current_loss", float("nan"))},
        gates=model.get_gates() if hasattr(model, "get_gates") else {},
        best_composite_score=getattr(trainer, "best_composite_score", float("inf")),
        best_composite_score_initialized=getattr(trainer, "best_composite_score_initialized", False),
        best_val_step=getattr(trainer, "best_val_step", 0),
        early_stopping_counter=getattr(trainer, "early_stopping_counter", 0),
        validation_history=getattr(trainer, "validation_history", []),
        converged=getattr(trainer, "converged", False),
        convergence_step=getattr(trainer, "convergence_step", None),
        convergence_info=getattr(trainer, "convergence_info", None),
    )
