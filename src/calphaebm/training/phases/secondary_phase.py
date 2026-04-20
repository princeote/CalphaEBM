"""Phase 3: Secondary structure term training.

Contrastive loss: push E_ss(native) < E_ss(IC-perturbed).

Trains three subterms:
  E_ram     — mixture-of-basins Ramachandran (basin weights + mixing MLP)
  E_hb_α    — helical H-bond (i→i+4 Cα distance, μ=6.04Å)
  E_hb_β    — sheet H-bond (bimodal, μ₁=5.74/μ₂=10.74Å)

CHANGES from run32:
  - Negatives are IC-perturbed (θ+noise, φ+noise) at log-uniform σ ∈ [0.05, 2.0],
    matching the scale range that IC-space Langevin dynamics explores.
  - θ IS perturbed (run32 only randomized φ → θθ MLP got zero gradient → dead).
  - R_neg is reconstructed via NeRF so H-bond terms see perturbed Cα distances
    (run32 passed native R to both pos/neg → hb_β got zero signal → dead).
  - Multi-scale σ creates gradient at every distance from native, shaping the
    basin rather than just the cliff edge.

Local and repulsion are frozen during this phase.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from calphaebm.geometry.internal import bond_angles, torsions
from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.training.core.state import TrainingState
from calphaebm.utils.logging import ProgressBar, get_logger

logger = get_logger()

# PDB-derived ratio: std(θ) / std(φ) across training chains
THETA_PHI_RATIO = 0.161


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_secondary_weights(sec) -> dict:
    """Read all lambda values from the secondary module."""
    w = {}
    if hasattr(sec, "ram_weight"):
        w["ram"] = float(sec.ram_weight.item())
    if hasattr(sec, "hb_helix") and hasattr(sec.hb_helix, "lambda_hb"):
        w["hb_α"] = float(sec.hb_helix.lambda_hb.item())
    if hasattr(sec, "hb_sheet") and hasattr(sec.hb_sheet, "lambda_hb"):
        w["hb_β"] = float(sec.hb_sheet.lambda_hb.item())
    return w


def _read_hbond_params(sec) -> dict:
    """Read H-bond Gaussian parameters."""
    p = {}
    if hasattr(sec, "hb_helix"):
        hb = sec.hb_helix
        if hasattr(hb, "mu"):
            p["hb_α_μ"] = float(hb.mu.item())
        if hasattr(hb, "sigma"):
            p["hb_α_σ"] = float(hb.sigma.item())
    if hasattr(sec, "hb_sheet"):
        hb = sec.hb_sheet
        if hasattr(hb, "mu1"):
            p["hb_β_μ₁"] = float(hb.mu1.item())
        if hasattr(hb, "mu2"):
            p["hb_β_μ₂"] = float(hb.mu2.item())
    return p


def _sample_sigmas(B: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    """Per-sample log-uniform σ: returns (B, 1) for broadcasting."""
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    return torch.empty(B, 1, device=device).uniform_(log_min, log_max).exp()


def _ic_perturb(R: torch.Tensor, sigma_min: float, sigma_max: float, lengths: Optional[torch.Tensor] = None):
    """IC-perturb a batch: returns (R_neg, theta_neg, phi_neg).

    Each structure gets its own log-uniform σ so every batch covers the
    full perturbation range.  Both θ and φ are perturbed.
    R_neg is reconstructed via NeRF so H-bond terms see perturbed Cα distances.
    Bonds stay 3.8 Å.
    """
    with torch.no_grad():
        theta, phi = coords_to_internal(R)
        anchor = extract_anchor(R)
        B = R.shape[0]

        sigmas = _sample_sigmas(B, sigma_min, sigma_max, R.device)  # (B, 1)
        noise_t = THETA_PHI_RATIO * sigmas * torch.randn_like(theta)
        noise_p = sigmas * torch.randn_like(phi)

        # Mask padding if lengths provided
        if lengths is not None:
            B, Lt = theta.shape
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

        R_neg = nerf_reconstruct(theta_neg, phi_neg, anchor)

    return R_neg, theta_neg, phi_neg


# ── Main phase runner ────────────────────────────────────────────────────────


def run_secondary_phase(trainer, config, train_loader, val_loader=None, native_structures=None, resume=None):
    """Run secondary phase training with IC-perturbed contrastive loss."""

    # ── Verify ───────────────────────────────────────────────────
    if not hasattr(trainer.model, "secondary") or trainer.model.secondary is None:
        raise RuntimeError("Secondary term not found in model.")

    sec = trainer.model.secondary

    # ── σ range ──────────────────────────────────────────────────
    sigma_min = float(getattr(config, "sigma_min_rad", 0.05))
    sigma_max = float(getattr(config, "sigma_max_rad", 2.0))

    # ── Optimizer (secondary params only) ────────────────────────
    sec_named_params = [(n, p) for n, p in sec.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in sec_named_params]
    n_params = sum(p.numel() for p in trainable_params)

    logger.info("Trainable parameters: %d tensors, %s params", len(trainable_params), f"{n_params:,}")

    is_fresh_run = (not resume) or (resume == "auto" and trainer.find_latest_checkpoint(config.name) is None)
    if is_fresh_run:
        for name, p in sec_named_params:
            logger.info("  - %s: %s", name, tuple(p.shape))

    if not trainable_params or config.lr <= 0:
        raise RuntimeError("No trainable parameters or lr <= 0")

    weight_decay = getattr(config, "weight_decay", 1e-5)
    trainer.optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=weight_decay)

    # ── LR scheduler ─────────────────────────────────────────────
    trainer.scheduler = None
    if getattr(config, "lr_schedule", None):
        if config.lr_schedule == "cosine":
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer, T_max=config.n_steps, eta_min=getattr(config, "lr_final", 0.0)
            )
        elif config.lr_schedule == "linear":
            lr_final = getattr(config, "lr_final", config.lr)
            lambda_fn = lambda step: 1.0 - (step / config.n_steps) * (1 - lr_final / config.lr)
            trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda_fn)
        elif config.lr_schedule == "exponential":
            lr_final = getattr(config, "lr_final", config.lr)
            gamma = math.exp(math.log(lr_final / config.lr) / config.n_steps)
            trainer.scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer.optimizer, gamma=gamma)

    # ── Log initial state ────────────────────────────────────────
    init_w = _read_secondary_weights(sec)
    init_hb = _read_hbond_params(sec)
    logger.info("Initial lambdas: %s", "  ".join(f"{k}={v:.4f}" for k, v in init_w.items()))
    if init_hb:
        logger.info("H-bond params: %s", "  ".join(f"{k}={v:.3f}" for k, v in init_hb.items()))
    logger.info(
        "IC perturbation: σ ∈ [%.3f, %.3f] rad (log-uniform), θ ratio=%.3f", sigma_min, sigma_max, THETA_PHI_RATIO
    )
    logger.info("Negatives: IC-perturbed (θ+φ), R reconstructed for H-bonds")

    # ── Resume ───────────────────────────────────────────────────
    start_step = 0
    if resume:
        if resume == "auto":
            resume = trainer.find_latest_checkpoint(config.name)
        if resume:
            state = trainer.load_checkpoint(resume, load_optimizer=True)
            start_step = getattr(state, "phase_step", 0)
            logger.info("Resumed from phase step %d (global=%d)", start_step, trainer.global_step)

    # ── Training loop ────────────────────────────────────────────
    trainer.model.train()
    trainer._init_validators()
    data_iter = iter(train_loader)

    progress = ProgressBar(config.n_steps, prefix=f"Phase {config.name}")
    loss = None
    nan_streak = 0
    ema_gap = 0.0
    ema_correct = 0.5
    alpha = 0.05

    for phase_step in range(start_step + 1, config.n_steps + 1):
        trainer.global_step += 1
        trainer.phase_step = phase_step
        trainer._apply_gate_schedule(config)

        try:
            R, seq, _, _, lengths = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            R, seq, _, _, lengths = next(data_iter)

        R = R.to(trainer.device)
        seq = seq.to(trainer.device)
        lengths = lengths.to(trainer.device) if lengths is not None else None
        current_lr = trainer.optimizer.param_groups[0]["lr"]

        try:
            # Native internal coordinates
            theta = bond_angles(R)  # (B, L-2)
            phi = torsions(R)  # (B, L-3)

            # IC-perturbed negative — per-sample σ, both θ and φ noised, R reconstructed
            R_neg, theta_neg, phi_neg = _ic_perturb(R, sigma_min, sigma_max, lengths)

            # Energies — R / R_neg passed so H-bonds see real distances
            E_pos = sec.energy_from_thetaphi(theta, phi, seq, R=R, lengths=lengths)
            E_neg = sec.energy_from_thetaphi(theta_neg, phi_neg, seq, R=R_neg, lengths=lengths)

            # Contrastive: softplus(E_pos - E_neg) + light regularization
            reg = 1e-4 * (E_pos**2 + E_neg**2).mean()
            loss = F.softplus(E_pos - E_neg).mean() + reg

            if not torch.isfinite(loss):
                nan_streak += 1
                logger.error("Non-finite loss at step %d (streak=%d)", phase_step, nan_streak)
                if nan_streak >= 10:
                    logger.error("10 consecutive NaN losses — stopping")
                    break
                continue

            nan_streak = 0
            trainer.current_loss = loss.item()

            # Track gap
            with torch.no_grad():
                gap = float((E_neg - E_pos).mean().item())
                correct = float((E_pos < E_neg).float().mean().item())
                ema_gap = alpha * gap + (1 - alpha) * ema_gap
                ema_correct = alpha * correct + (1 - alpha) * ema_correct

            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.1)

            has_nan = any(torch.isnan(p.grad).any() for p in trainable_params if p.grad is not None)
            if not has_nan:
                trainer.optimizer.step()
                if trainer.scheduler:
                    trainer.scheduler.step()
            else:
                logger.warning("NaN gradients at step %d — skipping", phase_step)

        except Exception as e:
            logger.error("Error at step %d: %s", phase_step, e)
            import traceback

            traceback.print_exc()
            continue

        # ── Summary every 50 steps ───────────────────────────────
        if phase_step % 50 == 0:
            weights = _read_secondary_weights(sec)
            logger.info(
                "[%s] step %6d/%d (global=%d) | loss=%.4f | lr=%.2e | "
                "gap=%.4f (ema=%.4f) | correct=%.0f%% (ema=%.0f%%) | %s",
                config.name,
                phase_step,
                config.n_steps,
                trainer.global_step,
                trainer.current_loss,
                current_lr,
                gap,
                ema_gap,
                correct * 100,
                ema_correct * 100,
                "  ".join(f"{k}={v:.3f}" for k, v in weights.items()),
            )

        # ── Diagnostic block every 200 steps ─────────────────────
        if phase_step % 200 == 0:
            with torch.no_grad():
                weights = _read_secondary_weights(sec)
                hb_params = _read_hbond_params(sec)

                E_pos_mean = float(E_pos.mean().item()) if E_pos is not None else 0.0
                E_neg_mean = float(E_neg.mean().item()) if E_neg is not None else 0.0
                gap_val = E_neg_mean - E_pos_mean

            logger.info("══════════════════════════════════════════════════════════════════")
            logger.info(
                "           STEP %d/%d | loss=%.4f | lr=%.2e",
                phase_step,
                config.n_steps,
                trainer.current_loss,
                current_lr,
            )
            logger.info("──────────────────────────────────────────────────────────────────")
            logger.info("  Lambdas:  %s", "  ".join(f"{k}={v:.4f}" for k, v in weights.items()))
            if hb_params:
                logger.info("  H-bond:   %s", "  ".join(f"{k}={v:.3f}" for k, v in hb_params.items()))
            logger.info("  σ range:  [%.3f, %.3f] rad  (per-sample log-uniform)", sigma_min, sigma_max)
            logger.info("──────────────────────────────────────────────────────────────────")
            logger.info("  E_pos (native):       %+.6f", E_pos_mean)
            logger.info("  E_neg (IC-perturbed): %+.6f", E_neg_mean)
            logger.info(
                "  Gap (neg − pos):      %+.6f  (EMA: %+.6f)  %s", gap_val, ema_gap, "good" if gap_val > 0 else "BAD"
            )
            logger.info("  Correct (pos<neg):    %.1f%%  (EMA: %.1f%%)", correct * 100, ema_correct * 100)
            logger.info("  Loss (softplus):      %.6f", trainer.current_loss)

            if hasattr(sec, "A"):
                A = sec.A.detach()
                logger.info("  Basin mixing A:       |max|=%.4f  std=%.4f", A.abs().max().item(), A.std().item())

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
    if not trainer.converged:
        final_loss = getattr(trainer, "current_loss", float("nan"))
        trainer.save_checkpoint(config.name, trainer.phase_step, final_loss)

    final_w = _read_secondary_weights(sec)
    final_hb = _read_hbond_params(sec)
    logger.info("Phase %s complete: %d/%d steps", config.name, phase_step, config.n_steps)
    logger.info("Final lambdas: %s", "  ".join(f"{k}={v:.4f}" for k, v in final_w.items()))
    if final_hb:
        logger.info("Final H-bond params: %s", "  ".join(f"{k}={v:.3f}" for k, v in final_hb.items()))
    logger.info("Final EMA gap: %.4f  |  Final EMA correct: %.1f%%", ema_gap, ema_correct * 100)

    return TrainingState(
        global_step=trainer.global_step,
        phase_step=trainer.phase_step,
        phase=config.name,
        losses={"loss": getattr(trainer, "current_loss", float("nan"))},
        gates=trainer.model.get_gates() if hasattr(trainer.model, "get_gates") else {},
        best_composite_score=getattr(trainer, "best_composite_score", float("inf")),
        best_composite_score_initialized=getattr(trainer, "best_composite_score_initialized", False),
        best_val_step=getattr(trainer, "best_val_step", 0),
        early_stopping_counter=getattr(trainer, "early_stopping_counter", 0),
        validation_history=getattr(trainer, "validation_history", []),
        converged=getattr(trainer, "converged", False),
        convergence_step=getattr(trainer, "convergence_step", None),
        convergence_info=getattr(trainer, "convergence_info", None),
    )
