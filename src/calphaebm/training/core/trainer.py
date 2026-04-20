"""Base trainer class for phased training."""

import math
import os
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from calphaebm.training.core.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from calphaebm.training.core.convergence import ConvergenceCriteria, ConvergenceMonitor
from calphaebm.training.core.schedules import apply_gate_schedule
from calphaebm.training.core.state import TrainingState, ValidationMetrics
from calphaebm.training.logging.diagnostics import DiagnosticLogger
from calphaebm.training.validation import DynamicsValidator
from calphaebm.training.validation.behavior import BehaviorValidator
from calphaebm.training.validation.generation import GenerationValidator
from calphaebm.utils.logging import get_logger

logger = get_logger()


class BaseTrainer:
    """Base class for phased trainers."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        ckpt_dir: str = "checkpoints",
        experiment_prefix: str = "run1",
        convergence_config: Optional[ConvergenceCriteria] = None,
    ):
        self.model = model
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.experiment_prefix = experiment_prefix
        self.convergence_config = convergence_config or ConvergenceCriteria()

        self.optimizer = None
        self.scheduler = None
        self.current_phase = None
        self.global_step = 0
        self.phase_step = 0
        self.current_loss = 0.0
        self.best_composite_score = None
        self.best_composite_score_initialized = False
        self.best_val_step = 0
        self.early_stopping_counter = 0
        self.validation_history = []
        self.config = None

        # Convergence monitoring
        self.convergence_monitor = None
        self.converged = False
        self.convergence_step = None
        self.convergence_info = None

        # Validators (lazy initialized)
        self.generation_validator = None
        self.behavior_validator = None
        self.dynamics_validator = None

        # Diagnostics (lazy initialized)
        self.diagnostic_logger = None

        # Calibrators (initialized in specific phases)
        self.repulsion_calibrator = None
        self.packing_diagnostics = None

        # Stability tracking
        self.last_max_force = 0.0
        self.last_clip_frac = 0.0
        self.last_p999_force = 0.0
        self.consecutive_bad_validations = 0

        # Warning cache to avoid log spam
        self._warned_missing_paths: Set[str] = set()

        os.makedirs(ckpt_dir, exist_ok=True)

    # ============================================================
    # Safe schedule method that doesn't touch gates
    # ============================================================
    def _apply_non_gate_schedule(self, config) -> None:
        """Apply schedules that should NOT touch gate parameters.

        This is a no-op by default. Override in subclasses if you have other schedules
        (e.g., temperature annealing, noise sigma scheduling, EMA decay, etc.) that
        should run even during gate ramp phases.

        Args:
            config: Phase configuration object
        """
        # Default implementation does nothing.
        # Subclasses can override to implement their own non-gate schedules.
        return

    # ============================================================

    def _init_validators(self) -> None:
        """Initialize validators and diagnostic logger if needed."""
        if self.generation_validator is None:
            val_langevin_steps = int(getattr(self, "val_langevin_steps", 500))
            val_langevin_beta = float(getattr(self, "val_langevin_beta", 1.0))
            self.generation_validator = GenerationValidator(
                self.model,
                self.device,
                n_steps=val_langevin_steps,
                langevin_beta=val_langevin_beta,
            )
        if self.behavior_validator is None:
            self.behavior_validator = BehaviorValidator(self.model, self.device)
        if self.diagnostic_logger is None:
            self.diagnostic_logger = DiagnosticLogger(self.model, self.device)
        if self.dynamics_validator is None:
            try:
                self.dynamics_validator = DynamicsValidator.from_pdb_ids(
                    model=self.model,
                    device=self.device,
                    pdb_ids=["1crn"],
                    beta=100.0,
                    n_steps=2000,
                    step_size=1e-4,
                    minimize_steps=200,
                    save_every=50,
                )
            except Exception as e:
                logger.warning("DynamicsValidator init failed: %s", e)
                self.dynamics_validator = None

    def _phase_path(self, phase: str) -> str:
        """Get the directory for a specific phase."""
        return os.path.join(self.ckpt_dir, self.experiment_prefix, phase)

    def _checkpoint_path(self, phase: str, step: int) -> str:
        """Get checkpoint path for a specific phase and step (phase-local step)."""
        return os.path.join(self._phase_path(phase), f"step{step:06d}.pt")

    def save_checkpoint(self, phase: str, step: int, loss: float, is_best: bool = False) -> str:
        """Save checkpoint with explicit phase and step (phase-local step)."""
        path = self._checkpoint_path(phase, step)

        # Build convergence info
        convergence_info = None
        if self.convergence_monitor:
            try:
                convergence_info = {
                    "converged": self.converged,
                    "convergence_step": self.convergence_step,
                    "metrics": self.convergence_monitor.get_summary()
                    if hasattr(self.convergence_monitor, "get_summary")
                    else None,
                }
                # Store in trainer for resume
                self.convergence_info = convergence_info
            except Exception as e:
                logger.warning(f"Could not get convergence summary: {e}")
                convergence_info = {"converged": self.converged, "convergence_step": self.convergence_step}
                self.convergence_info = convergence_info

        # Store both global and phase step
        best_score = self.best_composite_score if self.best_composite_score_initialized else None

        return save_checkpoint(
            path=path,
            global_step=self.global_step,
            phase_step=step,
            phase=phase,
            loss=loss,
            model_state=self.model.state_dict(),
            gates=self.model.get_gates() if hasattr(self.model, "get_gates") else {},
            optimizer_state=self.optimizer.state_dict() if self.optimizer else None,
            scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
            best_composite_score=best_score,
            best_composite_score_initialized=self.best_composite_score_initialized,
            best_val_step=self.best_val_step,
            early_stopping_counter=self.early_stopping_counter,
            validation_history=self.validation_history,
            convergence_info=convergence_info,
            is_best=is_best,
            model=self.model,
            optimizer=self.optimizer,
        )

    def load_checkpoint(self, path: str, load_optimizer: bool = True, strict: bool = False) -> TrainingState:
        """Load checkpoint and update trainer state."""
        state = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            load_optimizer=load_optimizer,
            strict=strict,
        )

        # CRITICAL: Update trainer state from loaded state
        self.global_step = state.global_step
        self.phase_step = state.phase_step
        self.current_phase = state.phase
        self.best_composite_score = state.best_composite_score
        self.best_composite_score_initialized = state.best_composite_score_initialized
        self.best_val_step = state.best_val_step
        self.early_stopping_counter = state.early_stopping_counter
        self.validation_history = state.validation_history
        self.converged = state.converged
        self.convergence_step = state.convergence_step
        self.convergence_info = state.convergence_info

        logger.debug(
            f"Restored trainer state: global={self.global_step}, phase={self.phase_step}, "
            f"best_score={self.best_composite_score}, best_initialized={self.best_composite_score_initialized}"
        )

        return state

    def find_latest_checkpoint(self, phase: str) -> Optional[str]:
        """Find latest checkpoint for a specific phase."""
        phase_dir = self._phase_path(phase)
        return find_latest_checkpoint(phase_dir)

    # ============================================================
    # Robust generation accounting (tolerates key drift)
    # ============================================================
    def _safe_int_cast(self, value: Any, default: int = 0) -> int:
        """Safely cast value to int."""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _get_attempted_count(self, gen_metrics: dict) -> int:
        """Extract number of attempted structures from generation metrics."""
        # Try standard keys
        for key in ("n_attempted", "n_structures_attempted"):
            if key in gen_metrics:
                return self._safe_int_cast(gen_metrics[key])

        # Fallback: generated + failures
        n_generated = self._get_generated_count(gen_metrics)
        n_fail = self._safe_int_cast(gen_metrics.get("generation_failures", 0))
        return n_generated + n_fail

    def _get_generated_count(self, gen_metrics: dict) -> int:
        """Extract number of successfully generated structures."""
        for key in ("n_generated", "n_samples", "n_success", "n_structures_success"):
            if key in gen_metrics:
                return self._safe_int_cast(gen_metrics[key])
        return 0

    # ============================================================
    # Validation
    # ============================================================
    def _validate(self, val_loader: DataLoader, step: int, **kwargs) -> ValidationMetrics:
        """Run validation by generating structures and testing model behavior."""
        self.model.eval()
        self._init_validators()

        val_max_samples = int(getattr(self, "val_max_samples", 256))
        val_step_size = getattr(self, "val_step_size", None)  # None = use constructor default
        gen_metrics = self.generation_validator.validate(
            val_loader, max_samples=val_max_samples, step_size=val_step_size
        )

        # Extract counts
        n_generated = self._get_generated_count(gen_metrics)
        n_attempted = self._get_attempted_count(gen_metrics)
        n_fail = self._safe_int_cast(gen_metrics.get("generation_failures", 0))

        # Handle validation failure
        if not gen_metrics.get("valid", True):
            logger.warning(f"Validation at step {step} failed to generate any valid structures")
            self.consecutive_bad_validations += 1
            if self.consecutive_bad_validations > 3:
                logger.warning(
                    f"  {self.consecutive_bad_validations} consecutive validation failures - check model stability"
                )

            return ValidationMetrics(
                step=step,
                composite_score=float("inf"),
                bond_length_mean=float("inf"),
                bond_length_std=float("inf"),
                bond_length_rmsd=float("inf"),
                train_loss=self.current_loss,
                native_vs_distorted_gap=float("inf"),
                helix_vs_random_gap=float("inf"),
                mean_energy=float("inf"),
                energy_std=float("inf"),
                ramachandran_corr=0.0,
                delta_phi_corr=0.0,
                failure_reason=gen_metrics.get("failure_reason", "no_valid_structures"),
                additional_metrics={
                    "n_generated": n_generated,
                    "n_attempted": n_attempted,
                    "generation_failures": n_fail,
                },
            )

        # Reset consecutive failures counter on successful validation
        self.consecutive_bad_validations = 0

        behavior_metrics = self.behavior_validator.validate(val_loader)

        mean_energy = float(behavior_metrics.get("energy_mean", 0.0) or 0.0)
        if abs(mean_energy) > 1e6:
            logger.warning(f"Energy explosion detected: {mean_energy:.3f}")

        # Safely get phase name
        phase_name = getattr(getattr(self, "config", None), "name", "full")

        # Dynamics validation (β=100 Langevin on crambin, 2K steps)
        dynamics_metrics = {}
        if self.dynamics_validator is not None and phase_name == "full":
            try:
                dynamics_metrics = self.dynamics_validator.validate(step=step)
            except Exception as e:
                logger.warning("[dynamics] Validation failed: %s", e)

        # Get metrics
        rama = float(gen_metrics.get("ramachandran_corr", 0.0) or 0.0)
        dphi = float(gen_metrics.get("delta_phi_corr", 0.0) or 0.0)

        # Clamp correlations to valid range
        rama = max(0.0, min(1.0, rama))
        dphi = max(0.0, min(1.0, dphi))

        # Composite score based on phase
        if phase_name == "secondary":
            # Secondary phase: Ramachandran and Δφ correlations matter
            composite_score = (1.0 - rama) * 10.0 + (1.0 - dphi) * 10.0

        elif phase_name == "local":
            # Local phase: bond RMSD + Δφ correlation (from local term)
            bond_rmsd = float(gen_metrics.get("bond_rmsd", 0.1) or 0.1)
            # Δφ correlation is meaningful for local phase (from Δφ persistence)
            dphi_score = (1.0 - dphi) * 5.0  # Weight Δφ appropriately
            composite_score = bond_rmsd * 100.0 + dphi_score + self.current_loss * 0.01

            # Log note about Ramachandran if it's non-zero (for debugging)
            if rama > 0.01:
                logger.debug(f"  Note: Ramachandran correlation ({rama:.4f}) is from secondary term (currently off)")

        else:
            # Full phase: stability-aware composite (lower = better)
            # Components:
            #   bond_score:  bond RMSD (IC sim guarantees ~0, but kept as safety)
            #   dphi_score:  delta-phi correlation quality
            #   rmsd_score:  structural drift from native (Langevin RMSD)
            #   delta_score: energy delta — dominates until STABLE (delta < 0)
            #   gap_score:   mean landscape gap — deeper = better (negative contribution)
            bond_rmsd = float(gen_metrics.get("bond_rmsd", 0.0) or 0.0)
            bond_score = bond_rmsd * 100.0 if bond_rmsd > 0 else 10.0
            dphi_score = (1.0 - dphi) * 2.0

            mean_rmsd = float(gen_metrics.get("mean_rmsd", 0.0) or 0.0)
            energy_delta = gen_metrics.get("mean_energy_delta", None)
            if energy_delta is None or not math.isfinite(energy_delta):
                rmsd_score = mean_rmsd * 0.5 if math.isfinite(mean_rmsd) else 10.0
                delta_score = 10.0
            else:
                rmsd_score = mean_rmsd * 0.5 if math.isfinite(mean_rmsd) else 10.0
                delta_score = max(float(energy_delta), 0.0) * 2.0

            # Mean gap across σ=[0.3, 0.5, 1.0, 2.0] rad — deeper landscape = lower composite
            # Capped contribution: -0.5 * mean_gap, so a gap of 4.0 gives -2.0 bonus
            mean_gap = float(behavior_metrics.get("mean_gap", 0.0) or 0.0)
            gap_score = -0.5 * max(mean_gap, 0.0)

            composite_score = bond_score + dphi_score + rmsd_score + delta_score + gap_score

            logger.info(
                "  Composite breakdown: bond=%.2f + dphi=%.2f + rmsd=%.2f + delta=%.2f + gap=%.2f = %.2f",
                bond_score,
                dphi_score,
                rmsd_score,
                delta_score,
                gap_score,
                composite_score,
            )

        if not math.isfinite(composite_score):
            logger.warning(f"Non-finite composite score {composite_score} at step {step}, using fallback")
            composite_score = 1e9

        metrics = ValidationMetrics(
            step=step,
            composite_score=composite_score,
            bond_length_mean=float(gen_metrics.get("bond_mean", 0.0) or 0.0),
            bond_length_std=float(gen_metrics.get("bond_std", 0.0) or 0.0),
            bond_length_rmsd=float(gen_metrics.get("bond_rmsd", 0.0) or 0.0),
            train_loss=self.current_loss,
            native_vs_distorted_gap=float(behavior_metrics.get("native_vs_distorted_gap", 0.0) or 0.0),
            helix_vs_random_gap=float(behavior_metrics.get("helix_vs_random_gap", 0.0) or 0.0),
            mean_energy=mean_energy,
            energy_std=float(behavior_metrics.get("energy_std", 0.0) or 0.0),
            ramachandran_corr=rama,
            delta_phi_corr=dphi,
            failure_reason=None,
            additional_metrics={
                "secondary_helix_gap": float(behavior_metrics.get("secondary_helix_gap", 0.0) or 0.0),
                "energy_consistency": float(behavior_metrics.get("energy_consistency", 0.0) or 0.0),
                "generation_failures": n_fail,
                "n_generated": n_generated,
                "n_attempted": n_attempted,
                "mean_rmsd": float(gen_metrics.get("mean_rmsd", 0.0) or 0.0),
                "mean_drmsd": float(gen_metrics.get("mean_drmsd", 0.0) or 0.0),
                "mean_q": float(gen_metrics.get("mean_q", 0.0) or 0.0),
                "mean_energy_delta": float(gen_metrics.get("mean_energy_delta", 0.0) or 0.0),
                "energy_delta_neg_frac": float(gen_metrics.get("energy_delta_neg_frac", 0.0) or 0.0),
                "is_stable": bool(gen_metrics.get("is_stable", False)),
                "mean_gap": float(behavior_metrics.get("mean_gap", 0.0) or 0.0),
                "gap_profile": behavior_metrics.get("gap_profile", {}),
                "native_gap_gate_normalized": (
                    float(behavior_metrics.get("native_vs_distorted_gap", 0.0) or 0.0)
                    / max(
                        sum(
                            float(v)
                            for v in (
                                self.model.get_gates().values() if hasattr(self.model, "get_gates") else {}.values()
                            )
                        ),
                        1e-6,
                    )
                ),
                **{k: v for k, v in dynamics_metrics.items() if k != "valid"},
            },
        )

        self.validation_history.append(metrics)

        # Feed validation-only metrics to EMA
        _ed_neg_frac = float(gen_metrics.get("energy_delta_neg_frac", 0.0) or 0.0)
        if hasattr(self, "diagnostic_logger") and self.diagnostic_logger is not None:
            self.diagnostic_logger.update_ema(frac_negative=_ed_neg_frac)

        # Customize log message based on phase
        logger.info(f"\n{'=' * 60}")
        logger.info(f"VALIDATION at step {step} (global={self.global_step}) - Phase: {phase_name}")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Bond lengths: {metrics.bond_length_mean:.3f} ± {metrics.bond_length_std:.3f} Å")
        logger.info(f"  Bond RMSD: {metrics.bond_length_rmsd:.4f}")

        if phase_name == "local":
            # Local phase: Δφ is meaningful, Ramachandran is not
            logger.info(f"  Delta phi corr: {metrics.delta_phi_corr:.4f} (from local Δφ persistence)")
            if metrics.ramachandran_corr > 0.01:
                logger.info(f"  Ramachandran corr: {metrics.ramachandran_corr:.4f} (N/A - secondary term inactive)")
            else:
                logger.info(f"  Ramachandran corr: N/A (secondary term inactive)")

        elif phase_name == "secondary":
            # Secondary phase: both Ramachandran and Δφ are meaningful
            logger.info(f"  Ramachandran corr: {metrics.ramachandran_corr:.4f}")
            logger.info(f"  Delta phi corr: {metrics.delta_phi_corr:.4f}")

        else:
            # Full phase: all metrics matter
            logger.info(f"  Ramachandran corr: {metrics.ramachandran_corr:.4f}")
            logger.info(f"  Delta phi corr: {metrics.delta_phi_corr:.4f}")
            # Stability metrics from Langevin generation
            _am = metrics.additional_metrics or {}
            _m_rmsd = _am.get("mean_rmsd", 0.0)
            _m_delta = _am.get("mean_energy_delta", 0.0)
            _m_q = _am.get("mean_q", 0.0)
            _stable = _am.get("is_stable", False)
            _neg_frac = _am.get("energy_delta_neg_frac", 0.0)
            _stable_str = "STABLE" if _stable else "UNSTABLE"
            logger.info(
                f"  RMSD: {_m_rmsd:.3f} Å  Q: {_m_q:.3f}  E_delta: {_m_delta:+.4f}  neg%={100*_neg_frac:.0f}%  [{_stable_str}]"
            )
            _m_gap = _am.get("mean_gap", 0.0)
            _gap_prof = _am.get("gap_profile", {})
            if _gap_prof:
                _gp_str = "  ".join(f"@{s:.1f}r={g:+.3f}" for s, g in sorted(_gap_prof.items()))
                logger.info(f"  Gap profile: {_gp_str}  mean={_m_gap:+.3f}")

            # Dynamics validator summary (β=100 Langevin on crambin)
            if dynamics_metrics.get("valid"):
                _d_rmsd = dynamics_metrics.get("dynamics_rmsd", 0.0)
                _d_q = dynamics_metrics.get("dynamics_q", 0.0)
                _d_rmsf = dynamics_metrics.get("dynamics_rmsf", 0.0)
                _d_rg_pct = 100 * dynamics_metrics.get("dynamics_rg_ratio", 1.0)
                _d_edelta = dynamics_metrics.get("dynamics_E_delta", 0.0)
                _d_pack = 100 * dynamics_metrics.get("dynamics_frac_packing", 0.0)
                logger.info(
                    f"  β=100 dynamics: RMSD={_d_rmsd:.2f}  Q={_d_q:.3f}  RMSF={_d_rmsf:.2f}  "
                    f"Rg={_d_rg_pct:.0f}%%  E_delta={_d_edelta:+.3f}  pack={_d_pack:.0f}%%"
                )

        logger.info(f"  Native gap: {metrics.native_vs_distorted_gap:.3f}")
        logger.info(f"  Helix gap: {metrics.helix_vs_random_gap:.3f}")

        # Gate-normalized gap: divides by sum of active gates so runs with
        # different gate_local scales are directly comparable.
        try:
            gates = self.model.get_gates() if hasattr(self.model, "get_gates") else {}
            gate_sum = sum(float(v) for v in gates.values() if v is not None)
            if gate_sum > 1e-6:
                norm_gap = metrics.native_vs_distorted_gap / gate_sum
                logger.info(f"  Native gap (gate-normalized, /Σg={gate_sum:.2f}): {norm_gap:.3f}")
        except Exception:
            pass
        logger.info(f"  Composite: {composite_score:.4f}")
        logger.info(f"  Generated samples: {n_generated} / attempted: {n_attempted} (failures: {n_fail})")
        logger.info(f"{'=' * 60}\n")

        return metrics

    # ============================================================
    # Early stopping
    # ============================================================
    def _check_early_stopping(self, composite_score: float, config) -> bool:
        """Check early stopping, ignoring invalid/failed validations."""
        if getattr(config, "early_stopping_patience", None) is None:
            return False

        # Skip best model update if validation failed or score is invalid
        if not math.isfinite(composite_score) or composite_score > 1e30:
            logger.info(f"  Skipping best model update - validation failed/invalid (score={composite_score})")
            return False

        # Initialize best if this is the first valid score
        if not self.best_composite_score_initialized:
            self.best_composite_score = composite_score
            self.best_composite_score_initialized = True
            self.best_val_step = self.phase_step
            logger.info(f"  Initializing best composite score: {composite_score:.4f}")
            return False

        min_delta = float(getattr(config, "early_stopping_min_delta", 0.0) or 0.0)
        patience = int(getattr(config, "early_stopping_patience", 0) or 0)

        if composite_score < self.best_composite_score - min_delta:
            self.best_composite_score = composite_score
            self.best_val_step = self.phase_step
            self.early_stopping_counter = 0
            logger.info(f"  New best composite score: {composite_score:.4f}")
            return False

        self.early_stopping_counter += 1
        logger.info(f"  No improvement for {self.early_stopping_counter} checks")
        if self.early_stopping_counter >= patience:
            logger.info("  Early stopping triggered")
            return True

        return False

    # ============================================================
    # Schedules / monitoring utilities
    # ============================================================
    def _apply_gate_schedule(self, config) -> None:
        """Apply gate scheduling if configured."""
        apply_gate_schedule(self.model, self.phase_step, config.n_steps, config.gate_schedule)

    def _count_trainable_params(self) -> Dict[str, int]:
        """Count trainable parameters per module."""
        counts = {"total": sum(p.numel() for p in self.model.parameters() if p.requires_grad)}

        for mod in ("local", "secondary", "repulsion", "packing"):
            if hasattr(self.model, mod) and getattr(self.model, mod) is not None:
                counts[mod] = sum(p.numel() for p in getattr(self.model, mod).parameters() if p.requires_grad)

        return counts

    def _setup_convergence_monitor(self, config) -> None:
        """Setup convergence monitoring based on phase."""
        if config.name == "secondary":
            weight_names = [
                "secondary.lambda_ram",
                "secondary.lambda_theta_phi",
                "secondary.lambda_phi_phi",
            ]
            self.convergence_monitor = ConvergenceMonitor(
                config=self.convergence_config,
                phase=config.name,
                weight_names=weight_names,
            )
        elif config.name == "local":
            weight_names = [
                "local.bond_spring",
                "local.theta_theta_weight",
                "local.delta_phi_weight",
            ]
            self.convergence_monitor = ConvergenceMonitor(
                config=self.convergence_config,
                phase=config.name,
                weight_names=weight_names,
            )
        else:
            self.convergence_monitor = ConvergenceMonitor(
                config=self.convergence_config,
                phase=config.name,
            )

    def _get_monitored_weight(self, path: str) -> Optional[float]:
        """Safely get a monitored weight value."""
        try:
            obj = self.model
            parts = path.split(".")
            for i, part in enumerate(parts):
                if not hasattr(obj, part):
                    # Log missing paths only once to avoid spam
                    full_path = ".".join(parts[: i + 1])
                    if full_path not in self._warned_missing_paths:
                        logger.warning(f"Monitored weight path {full_path} not found")
                        self._warned_missing_paths.add(full_path)
                    return None
                obj = getattr(obj, part)

            if isinstance(obj, (torch.Tensor, nn.Parameter)):
                return obj.item()
            elif isinstance(obj, (int, float)):
                return float(obj)
            else:
                return None
        except Exception as e:
            full_path = path
            if full_path not in self._warned_missing_paths:
                logger.warning(f"Failed to get monitored weight {path}: {e}")
                self._warned_missing_paths.add(full_path)
            return None

    def _check_convergence(self) -> bool:
        """Check if training has converged."""
        if self.convergence_monitor is None:
            return False

        metrics = {
            "step": self.phase_step,
            "loss": self.current_loss,
        }

        # Add weight values if being monitored
        if getattr(self.convergence_monitor, "weight_names", None):
            for name in self.convergence_monitor.weight_names:
                value = self._get_monitored_weight(name)
                if value is not None:
                    metrics[name] = value

        self.converged, self.convergence_step = self.convergence_monitor.update(metrics)

        if self.converged and self.convergence_step == self.phase_step:
            logger.info(f"Convergence detected at step {self.phase_step} (global={self.global_step})!")
            if hasattr(self.convergence_monitor, "convergence_reason"):
                logger.info(f"  Reason: {self.convergence_monitor.convergence_reason}")

        return self.converged
