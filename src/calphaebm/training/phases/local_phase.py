"""Phase 1: Local term training (internal coordinate version).

Uses dsm_ic_loss — noises (θ, φ) in radians, reconstructs R via NeRF,
scores dE/dθ and dE/dφ. Bond_spring is gone.

Trains three subterms:
  E_θθ   — angular persistence MLP (sequence-conditioned)
  E_Δφ   — torsional smoothness (tabulated, only λ trains)
  E_φφ   — consecutive torsion MLP (sequence-conditioned)
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from calphaebm.training.core.state import TrainingState
from calphaebm.training.losses.dsm import dsm_ic_loss
from calphaebm.training.validation.local_validator import LocalValidator
from calphaebm.utils.logging import ProgressBar, get_logger

logger = get_logger()

MAX_ATTEMPTS_MULTIPLIER = 10


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_local_weights(local_mod) -> dict:
    """Read lambda values from the local module (supports both old and 4-mer)."""
    w = {}
    # 4-mer architecture
    if hasattr(local_mod, "theta_phi_weight"):
        w["θφ"] = float(local_mod.theta_phi_weight.item())
    # Old 3-subterm architecture
    if hasattr(local_mod, "theta_theta_weight"):
        w["θθ"] = float(local_mod.theta_theta_weight.item())
    if hasattr(local_mod, "delta_phi_weight"):
        w["Δφ"] = float(local_mod.delta_phi_weight.item())
    if hasattr(local_mod, "phi_phi_weight"):
        w["φφ"] = float(local_mod.phi_phi_weight.item())
    return w


def _read_local_energies(local_mod, R, seq, lengths=None) -> dict:
    """Read per-subterm energies (mean over batch). Padding-aware."""
    e = {}
    # 4-mer architecture
    if hasattr(local_mod, "theta_phi_energy"):
        try:
            e["θφ"] = float(local_mod.theta_phi_energy(R, seq, lengths=lengths).mean().item())
        except Exception:
            e["θφ"] = float("nan")
        e["total"] = e.get("θφ", float("nan"))
        return e
    # Old 3-subterm architecture
    try:
        e["θθ"] = float(local_mod.theta_theta_energy(R, seq, lengths=lengths).mean().item())
    except Exception:
        try:
            e["θθ"] = float(local_mod.theta_theta_energy(R).mean().item())
        except Exception:
            e["θθ"] = float("nan")
    try:
        e["Δφ"] = float(local_mod.delta_phi_energy(R, lengths=lengths).mean().item())
    except Exception:
        e["Δφ"] = float("nan")
    try:
        e["φφ"] = float(local_mod.phi_phi_energy(R, seq, lengths=lengths).mean().item())
    except Exception:
        try:
            e["φφ"] = float(local_mod.phi_phi_energy(R).mean().item())
        except Exception:
            e["φφ"] = float("nan")
    try:
        e["total"] = float(local_mod(R, seq, lengths=lengths).mean().item())
    except Exception:
        e["total"] = float("nan")
    return e


def _debug_state(local_mod) -> dict:
    """Collect lightweight debug state for checkpoint saves."""
    state = {}
    # 4-mer
    if hasattr(local_mod, "theta_phi_weight"):
        state["theta_phi_weight"] = local_mod.theta_phi_weight.detach().cpu()
    # Old
    for attr in ("theta_theta_weight", "delta_phi_weight", "phi_phi_weight"):
        if hasattr(local_mod, attr):
            state[attr] = getattr(local_mod, attr).detach().cpu()
    return state


# ── Main phase runner ────────────────────────────────────────────────────────


def run_local_phase(trainer, config, train_loader, val_loader=None, native_structures=None, resume=None):
    """Run local phase training (IC version)."""

    # Reset convergence flags for new phase
    trainer.converged = False
    trainer.convergence_step = None
    trainer.convergence_info = None

    # ── Setup optimizer — only local parameters ──────────────────────────
    local_named_params = [(n, p) for n, p in trainer.model.local.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in local_named_params]

    n_tensors = len(trainable_params)
    n_params = sum(p.numel() for p in trainable_params)
    logger.info("Trainable parameters: %d tensors, %s params", n_tensors, f"{n_params:,}")

    is_fresh_run = (not resume) or (resume == "auto" and trainer.find_latest_checkpoint(config.name) is None)
    if is_fresh_run:
        for name, p in local_named_params:
            logger.info("  - %s: %s", name, tuple(p.shape))

    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found for local phase")
    if config.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {config.lr}")

    weight_decay = getattr(config, "weight_decay", 0.0)
    trainer.optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=weight_decay)

    # ── LR scheduler ─────────────────────────────────────────────────────
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
        else:
            trainer.scheduler = None
    else:
        trainer.scheduler = None

    # ── IC DSM sigma (radians) ───────────────────────────────────────────
    sigma_rad = float(getattr(config, "sigma_rad", getattr(config, "sigma", 0.08)))
    sigma_min_raw = getattr(config, "sigma_min_rad", getattr(config, "sigma_min", None))
    sigma_max_raw = getattr(config, "sigma_max_rad", getattr(config, "sigma_max", None))
    sigma_min_rad = float(sigma_min_raw) if sigma_min_raw is not None else None
    sigma_max_rad = float(sigma_max_raw) if sigma_max_raw is not None else None

    if sigma_min_rad is not None and sigma_max_rad is not None:
        logger.info("Multi-scale IC DSM: σ ~ LogUniform(%.3f, %.3f) rad", sigma_min_rad, sigma_max_rad)
    else:
        logger.info("Fixed-sigma IC DSM: σ=%.3f rad", sigma_rad)

    # ── Log initial state ────────────────────────────────────────────────
    init_w = _read_local_weights(trainer.model.local)
    logger.info("Initial lambdas: %s", "  ".join(f"{k}={v:.4f}" for k, v in init_w.items()))

    # ── Resume ───────────────────────────────────────────────────────────
    start_opt_step = 0
    if resume:
        if resume == "auto":
            resume = trainer.find_latest_checkpoint(config.name)
        if resume:
            state = trainer.load_checkpoint(resume, load_optimizer=True)
            start_opt_step = state.phase_step
            logger.info("Resumed from optimizer step %d (global=%d)", start_opt_step, state.global_step)

    # ── Local validator ──────────────────────────────────────────────────
    local_validator = None
    validate_every = int(getattr(config, "validate_every", 0) or 0)
    if val_loader is not None and validate_every > 0:
        local_validator = LocalValidator(trainer.model, trainer.device)
        logger.info("Local validator initialized (every %d steps)", validate_every)
    elif validate_every > 0 and val_loader is None:
        logger.warning("validate_every > 0 but val_loader is None; validation disabled.")

    # ── Training loop ────────────────────────────────────────────────────
    trainer.model.train()
    trainer._init_validators()
    data_iter = iter(train_loader)

    try:
        progress = ProgressBar(config.n_steps, prefix=f"Phase {config.name}", initial=start_opt_step)
    except Exception:
        progress = ProgressBar(config.n_steps, prefix=f"Phase {config.name}")
        for _ in range(start_opt_step):
            progress.update(1)

    attempt_step = start_opt_step
    opt_step = start_opt_step
    max_attempts = config.n_steps * MAX_ATTEMPTS_MULTIPLIER
    loss = None
    last_successful_loss = None

    if not hasattr(trainer, "best_local_gap"):
        trainer.best_local_gap = float("-inf")
        trainer.best_local_gap_step = 0

    # Cache for logging on skip steps
    _last_energies = {}

    while opt_step < config.n_steps:
        attempt_step += 1

        if attempt_step > max_attempts:
            logger.error(
                "Exceeded maximum attempts (%d) with only %d/%d optimizer steps",
                max_attempts,
                opt_step,
                config.n_steps,
            )
            debug_path = (
                Path(trainer.ckpt_dir)
                / trainer.experiment_prefix
                / config.name
                / (f"debug_max_attempts_opt{opt_step:06d}_att{attempt_step:06d}.pt")
            )
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"attempt_step": attempt_step, "opt_step": opt_step, **_debug_state(trainer.model.local)}, debug_path
            )
            break

        trainer.phase_step = opt_step
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

        step_successful = False
        skip_reason = None

        try:
            loss = dsm_ic_loss(
                trainer.model,
                R,
                seq,
                sigma=sigma_rad,
                sigma_min=sigma_min_rad,
                sigma_max=sigma_max_rad,
                lengths=lengths,
            )

            if not torch.isfinite(loss):
                logger.error("Non-finite loss at attempt %d: %s", attempt_step, loss.item())
                debug_path = (
                    Path(trainer.ckpt_dir)
                    / trainer.experiment_prefix
                    / config.name
                    / (f"debug_nonfinite_loss_opt{opt_step:06d}_att{attempt_step:06d}.pt")
                )
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "R": R.detach().cpu(),
                        "seq": seq.detach().cpu(),
                        "loss": float(loss.item()),
                        **_debug_state(trainer.model.local),
                    },
                    debug_path,
                )
                skip_reason = "non-finite loss"
                trainer._last_skip_reason = skip_reason
                continue

            trainer.current_loss = loss.item()

            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            # Check gradients
            grad_ok = True
            offending_param = None
            grad_info = {}
            for name, p in local_named_params:
                g = p.grad
                grad_info[name] = {"has_grad": g is not None}
                if g is None:
                    continue
                if not torch.isfinite(g).all():
                    grad_ok = False
                    offending_param = name
                    logger.error("Non-finite gradient in parameter %s shape %s", name, tuple(p.shape))
                    continue
                grad_info[name].update(
                    {
                        "norm": float(g.norm().item()),
                        "abs_max": float(g.abs().max().item()),
                    }
                )

            if grad_ok:
                trainer.optimizer.step()
                if trainer.scheduler:
                    trainer.scheduler.step()
                opt_step += 1
                trainer.global_step += 1
                trainer.phase_step = opt_step
                step_successful = True
                trainer._last_skip_reason = None
                last_successful_loss = trainer.current_loss
            else:
                skip_reason = f"non-finite gradient in {offending_param}"
                trainer._last_skip_reason = skip_reason
                debug_path = (
                    Path(trainer.ckpt_dir)
                    / trainer.experiment_prefix
                    / config.name
                    / (f"debug_nonfinite_grad_opt{opt_step:06d}_att{attempt_step:06d}.pt")
                )
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"grad_info": grad_info, "opt_step": opt_step, **_debug_state(trainer.model.local)}, debug_path
                )

        except Exception as e:
            logger.error("Error at attempt %d: %s", attempt_step, e)
            import traceback

            traceback.print_exc()
            skip_reason = f"exception: {type(e).__name__}"
            trainer._last_skip_reason = skip_reason

        # ── (Force spike detection skipped in local phase ─────────────
        # Local-only training produces large Cartesian forces through NeRF
        # chain propagation that are not meaningful without repulsion/packing.
        # Force monitoring is done in full_phase where all terms interact.)

        # ── Logging every 50 steps ───────────────────────────────────────
        if opt_step % 50 == 0 and opt_step > 0:
            with torch.no_grad():
                weights = _read_local_weights(trainer.model.local)
                if step_successful:
                    energies = _read_local_energies(trainer.model.local, R.detach(), seq.detach(), lengths=lengths)
                    _last_energies = energies
                else:
                    energies = _last_energies or {}

            display_loss = (
                trainer.current_loss
                if step_successful
                else (
                    last_successful_loss
                    if last_successful_loss is not None
                    else getattr(trainer, "current_loss", float("nan"))
                )
            )

            # Summary line (matches full_phase style)
            logger.info(
                "[%s] step %6d/%d (global=%d) | loss=%.4f | lr=%.2e  dsm=%.4f",
                config.name,
                opt_step,
                config.n_steps,
                trainer.global_step,
                display_loss,
                current_lr,
                display_loss,
            )

        # ── Detailed diagnostic block every 200 steps ────────────────────
        if opt_step % 200 == 0 and opt_step > 0:
            with torch.no_grad():
                weights = _read_local_weights(trainer.model.local)
                energies = _read_local_energies(trainer.model.local, R.detach(), seq.detach(), lengths=lengths)
                _last_energies = energies

            logger.info("══════════════════════════════════════════════════════════════════")
            logger.info(
                "           STEP %d/%d | loss=%.4f | lr=%.2e",
                opt_step,
                config.n_steps,
                trainer.current_loss,
                current_lr,
            )
            logger.info("──────────────────────────────────────────────────────────────────")
            logger.info("  Lambdas:  %s", "  ".join(f"{k}={v:.4f}" for k, v in weights.items()))
            logger.info("──────────────────────────────────────────────────────────────────")
            logger.info("  subterm       λ          E/res       E%")
            logger.info("  ·····················································")

            # Detect architecture: 4-mer has "θφ", old has "θθ"/"Δφ"/"φφ"
            if "θφ" in energies:
                subterm_keys = ["θφ"]
            else:
                subterm_keys = ["θθ", "Δφ", "φφ"]

            total_abs = sum(abs(energies.get(k, 0.0)) for k in subterm_keys)
            total_abs = max(total_abs, 1e-8)
            for key in subterm_keys:
                w = weights.get(key, float("nan"))
                e = energies.get(key, float("nan"))
                pct = abs(e) / total_abs * 100.0 if math.isfinite(e) else 0.0
                logger.info("  %-10s  %8.4f    %+10.6f   %5.1f%%", key, w, e, pct)

            logger.info("  ·····················································")
            logger.info("  TOTAL                  %+10.6f", energies.get("total", float("nan")))

            # MLP weight stats
            local_mod = trainer.model.local
            # 4-mer architecture
            if hasattr(local_mod, "f_theta_phi"):
                mlp = local_mod.f_theta_phi
                all_w = torch.cat([p.detach().flatten() for p in mlp.parameters()])
                logger.info(
                    "  θφ 4-mer MLP weights: mean=%.4f  std=%.4f  |max|=%.4f",
                    all_w.mean().item(),
                    all_w.std().item(),
                    all_w.abs().max().item(),
                )
            # Old architecture
            for mlp_name, mlp_attr in [("θθ MLP", "f_theta_theta"), ("φφ MLP", "f_phi_phi")]:
                mlp = getattr(local_mod, mlp_attr, None)
                if mlp is not None:
                    all_w = torch.cat([p.detach().flatten() for p in mlp.parameters()])
                    logger.info(
                        "  %s weights: mean=%.4f  std=%.4f  |max|=%.4f",
                        mlp_name,
                        all_w.mean().item(),
                        all_w.std().item(),
                        all_w.abs().max().item(),
                    )

            logger.info("══════════════════════════════════════════════════════════════════")

        # ── Validation ───────────────────────────────────────────────────
        if step_successful and local_validator is not None and validate_every > 0 and (opt_step % validate_every == 0):
            try:
                val_metrics = local_validator.validate(
                    val_loader,
                    n_batches=3,
                    step=opt_step,
                    noise_scale=0.03,
                    proj_steps=20,
                    proj_lr=5e-3,
                    n_corruptions_per_batch=5,
                    warn_bond_rmsd_diff=0.03,
                )
                local_validator.log_validation(val_metrics)

                bond_rmsd_ok = val_metrics.dist_bond_rmsd < float(getattr(config, "best_gap_max_bond_rmsd", 0.08))
                gap_sr_ok = val_metrics.gap_success_rate > float(getattr(config, "best_gap_min_success_rate", 0.60))

                if bond_rmsd_ok and gap_sr_ok and (val_metrics.gap_mean > trainer.best_local_gap):
                    trainer.best_local_gap = val_metrics.gap_mean
                    trainer.best_local_gap_step = opt_step
                    trainer.save_checkpoint(config.name, opt_step, trainer.current_loss, is_best=True)
                    logger.info("  New best local gap: %.4f", val_metrics.gap_mean)
            except Exception as e:
                logger.error("Validation error at step %d: %s", opt_step, e)

        # ── Convergence check ────────────────────────────────────────────
        if step_successful and opt_step % 100 == 0:
            if trainer._check_convergence():
                logger.info("Training converged at optimizer step %d", opt_step)
                trainer.save_checkpoint(
                    config.name,
                    opt_step,
                    last_successful_loss or getattr(trainer, "current_loss", float("nan")),
                    is_best=True,
                )
                break

        # ── Checkpoint save ──────────────────────────────────────────────
        save_every = int(getattr(config, "save_every", 0) or 0)
        if step_successful and save_every > 0 and (opt_step % save_every == 0):
            trainer.save_checkpoint(
                config.name,
                opt_step,
                last_successful_loss or getattr(trainer, "current_loss", float("nan")),
            )

        if step_successful:
            progress.update(1)

        if trainer.converged:
            break

    # ── Final ────────────────────────────────────────────────────────────
    if not trainer.converged and opt_step > start_opt_step:
        final_loss = (
            last_successful_loss if last_successful_loss is not None else getattr(trainer, "current_loss", float("nan"))
        )
        if final_loss == 0.0 and loss is not None and hasattr(loss, "item") and torch.isfinite(loss):
            final_loss = float(loss.item())
        trainer.save_checkpoint(config.name, opt_step, final_loss)

    logger.info(
        "Phase %s complete: %d/%d optimizer steps, %d attempts", config.name, opt_step, config.n_steps, attempt_step
    )

    # Final lambda snapshot
    final_w = _read_local_weights(trainer.model.local)
    logger.info("Final lambdas: %s", "  ".join(f"{k}={v:.4f}" for k, v in final_w.items()))

    loss_value = getattr(trainer, "current_loss", float("nan"))
    if not math.isfinite(loss_value):
        loss_value = float("nan")

    return TrainingState(
        global_step=trainer.global_step,
        phase_step=opt_step,
        phase=config.name,
        losses={"loss": loss_value},
        gates=trainer.model.get_gates() if hasattr(trainer.model, "get_gates") else {},
        best_composite_score=getattr(trainer, "best_composite_score", float("inf")),
        best_composite_score_initialized=getattr(trainer, "best_composite_score_initialized", False),
        best_val_step=getattr(trainer, "best_val_step", 0),
        early_stopping_counter=getattr(trainer, "early_stopping_counter", 0),
        validation_history=getattr(trainer, "validation_history", []),
        converged=trainer.converged,
        convergence_step=trainer.convergence_step,
        convergence_info=getattr(trainer, "convergence_info", None),
    )
