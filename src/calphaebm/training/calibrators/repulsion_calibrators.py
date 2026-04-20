"""Repulsion calibration utilities with wall fraction control and EMA smoothing.

Early warning multiplier (linear ramp):
  Let m = global minimum nonbonded distance (Å)
      E = early_warning_min (Å)
      C = catastrophic_min (Å)
      M = early_warning_max_mult (unitless)

  t = clip((E - m) / (E - C), 0, 1)
  mult = 1 + t * (M - 1)

Catastrophic override:
  If m <= C, apply mult = 1.5 and reset cooldown.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, Optional, Tuple

import torch

from calphaebm.utils.logging import get_logger
from calphaebm.utils.neighbors import pairwise_distances

logger = get_logger()

# Version stamp and path fingerprint
__version__ = "2.0.0-early-warning-ramp"
__date__ = "2024-12-08"


def _is_dataloader_worker() -> bool:
    """Best-effort detection of DataLoader workers."""
    try:
        from torch.utils.data import get_worker_info

        return get_worker_info() is not None
    except Exception:
        return False


def _rank_is_zero() -> bool:
    """torchrun-style rank check; defaults to 0 in single-process runs."""
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    except Exception:
        return True


def _print_module_load_banner_once_global() -> None:
    """Print the module 'LOADED ...' banner at most once across *all* processes.

    Why: DataLoader workers / multiprocessing import modules in separate OS processes.
    A module-level boolean only suppresses within a single process.

    Approach: atomic lockfile creation in the system temp dir.
    """
    if _is_dataloader_worker():
        return
    if not _rank_is_zero():
        return

    # Unique per-user/per-path/per-version, so different checkouts don't stomp each other.
    # (hash() is salted per-process; don't use it. Use a stable string + sanitize.)
    safe_path = os.path.abspath(__file__).replace(os.sep, "_").replace(":", "_")
    fname = f"calphaebm_repulsion_calibrators_loaded_{__version__}_{safe_path}.lock"
    lock_path = os.path.join(tempfile.gettempdir(), fname)

    try:
        # Atomic create: only one process wins.
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
        finally:
            os.close(fd)

        logger.info(
            "LOADED repulsion_calibrators.py v%s from: %s (pid=%d)",
            __version__,
            os.path.abspath(__file__),
            os.getpid(),
        )
    except FileExistsError:
        # Someone else already printed it.
        return
    except Exception:
        # If anything goes wrong, fail open (don't crash training).
        return


# Print banner once total (across processes)
_print_module_load_banner_once_global()


class RepulsionCalibrator:
    """Repulsion calibrator using log-space proportional control on wall fraction.

    Control signal:
      frac = fraction of samples with min_nonbonded < wall_threshold

    Goal:
      Drive frac toward target_frac using multiplicative updates:
        log(lambda_new) = log(lambda_old) + eta * (frac_ctrl - target_frac)

      where frac_ctrl is an exponentially-smoothed version of frac
      to reduce quantization noise from batch size.

    Safety:
      If global nonbonded min <= catastrophic_min, apply an emergency 1.5x jump
      (bypasses max_step_ratio but still obeys absolute lambda bounds) and
      trigger cooldown.

    Early warning:
      If catastrophic_min < min < early_warning_min, apply a linear ramp:
        t = clip((early_warning_min - min) / (early_warning_min - catastrophic_min), 0, 1)
        mult = 1 + t * (early_warning_max_mult - 1)
      No cooldown reset, but cooldown continues counting down.

    Cooldown:
      After a catastrophic event, block decreases for a set number of steps
      (increases are still allowed). Cooldown counts down on any non-catastrophic
      step (early warning or control).
    """

    def __init__(
        self,
        # Control parameters
        target_frac: float = 0.03,  # e.g., 1/32 ≈ 0.03125
        eta: float = 1.5,  # gain (dimensionless)
        # EMA smoothing (to reduce quantization noise)
        ema_alpha: float = 0.3,  # (0, 1] ; 1 = instant, smaller = smoother
        # Deadband (ignore small errors)
        deadband: float = 0.01,  # absolute fraction, e.g. 0.01 = ±1.0%
        # Safety thresholds (Å)
        catastrophic_min: float = 2.8,  # <= triggers catastrophic multiplier + cooldown
        early_warning_min: float = 3.2,  # must be > catastrophic_min
        # Early warning max multiplier at catastrophic_min (capped by max_step_ratio)
        early_warning_max_mult: float = 1.2,
        # Cooldown after catastrophic events
        catastrophic_cooldown_steps: int = 25,
        # Update limits
        max_step_ratio: float = 1.5,
        max_lambda: float = 1000.0,
        min_lambda: float = 0.01,
        # Wall threshold (Å)
        wall_threshold: float = 4.5,
        # Nonbonded settings
        exclude: int = 3,
        max_dist_for_near: float = 8.0,
        # Debug
        debug_min_pair: bool = False,
        debug_once: bool = True,
    ):
        if not (0.0 < target_frac < 1.0):
            raise ValueError("target_frac must be between 0 and 1")
        if eta <= 0:
            raise ValueError("eta must be > 0")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1]")
        if deadband < 0:
            raise ValueError("deadband must be >= 0")
        if catastrophic_min <= 0:
            raise ValueError("catastrophic_min must be > 0")
        if early_warning_min <= catastrophic_min:
            raise ValueError("early_warning_min must be > catastrophic_min")
        if early_warning_max_mult < 1.0:
            raise ValueError("early_warning_max_mult must be >= 1.0")
        if max_step_ratio <= 1.0:
            raise ValueError("max_step_ratio must be > 1.0")
        if not (0 < min_lambda < max_lambda):
            raise ValueError("Require 0 < min_lambda < max_lambda")
        if wall_threshold <= 0:
            raise ValueError("wall_threshold must be > 0")

        self.target_frac = float(target_frac)
        self.eta = float(eta)
        self.ema_alpha = float(ema_alpha)
        self.deadband = float(deadband)

        self.catastrophic_min = float(catastrophic_min)
        self.early_warning_min = float(early_warning_min)

        # Cap early-warning maximum by max_step_ratio (so EW never exceeds cap).
        self.max_step_ratio = float(max_step_ratio)
        self.early_warning_max_mult = min(float(early_warning_max_mult), self.max_step_ratio)

        self.catastrophic_cooldown_steps = int(catastrophic_cooldown_steps)

        self.max_lambda = float(max_lambda)
        self.min_lambda = float(min_lambda)

        self.wall_threshold = float(wall_threshold)

        self.exclude = int(exclude)
        self.max_dist_for_near = float(max_dist_for_near)

        self.debug_min_pair = bool(debug_min_pair)
        self.debug_once = bool(debug_once)
        self._has_logged_close_contact = False
        self._has_logged_catastrophic_details = False
        self._has_logged_early_warning = False

        # Cooldown counter (reset on catastrophic events)
        self._cooldown = 0

        # EMA state
        self._frac_ema: Optional[float] = None

        # Cache for mask and pair indices
        self._mask_cache: Dict[Tuple[int, str, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._max_cache_size = 5

        logger.info(
            "RepulsionCalibrator v%s (%s) init | target_frac=%.2f%% eta=%.2f ema_alpha=%.2f deadband=%.3f "
            "early_warning=%.2fA (max_mult=%.2f) catastrophic_min=%.2fA (cat_mult=1.50) cooldown=%d "
            "max_step_ratio=%.2fx lambda=[%.3g, %.3g] wall_threshold=%.2fA exclude=%d",
            __version__,
            __date__,
            100.0 * self.target_frac,
            self.eta,
            self.ema_alpha,
            self.deadband,
            self.early_warning_min,
            self.early_warning_max_mult,
            self.catastrophic_min,
            self.catastrophic_cooldown_steps,
            self.max_step_ratio,
            self.min_lambda,
            self.max_lambda,
            self.wall_threshold,
            self.exclude,
        )

    def _get_mask_and_indices(self, L: int, device: torch.device, exclude: int):
        """Get cached nonbonded mask and flat pair indices."""
        cache_key = (L, str(device), exclude)
        if cache_key not in self._mask_cache:
            idx = torch.arange(L, device=device)
            sep = (idx[:, None] - idx[None, :]).abs()
            triu_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
            nonbonded_mask = (sep > exclude) & triu_mask
            mask_flat = nonbonded_mask.reshape(-1)
            flat_pair_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)

            self._mask_cache[cache_key] = (nonbonded_mask, mask_flat, flat_pair_idx)

            if len(self._mask_cache) > self._max_cache_size:
                oldest_key = next(iter(self._mask_cache))
                del self._mask_cache[oldest_key]

        return self._mask_cache[cache_key]

    def compute_distance_statistics(self, R: torch.Tensor, step: Optional[int] = None) -> dict:
        """Compute nonbonded distance statistics."""
        D = pairwise_distances(R)  # (B, L, L)
        batch_size, L = D.shape[:2]

        _, mask_flat, flat_pair_idx = self._get_mask_and_indices(L, R.device, self.exclude)
        n_total = int(mask_flat.sum().item())

        D_flat = D.reshape(batch_size, -1)  # (B, L*L)
        all_distances = D_flat[:, mask_flat]  # (B, N_pairs)

        if all_distances.numel() == 0:
            return {
                "min_nonbonded": float("inf"),
                "min_nonbonded_q10": float("inf"),
                "min_nonbonded_q01": float("inf"),
                "min_per_sample": [],
                "min_per_batch": [],
                "frac_samples_below_40": 0.0,
                "frac_samples_below_45": 0.0,
                "frac_samples_below_wall": 0.0,
                "q01_near": float("nan"),
                "n_total": n_total,
                "n_near": 0,
                "debug_min_pair": None,
            }

        min_nonbonded_global = float(all_distances.min().item())
        min_per_sample = all_distances.amin(dim=1)  # (B,)

        min_nonbonded_q10 = float(torch.quantile(min_per_sample, 0.10).item())
        min_nonbonded_q01 = float(torch.quantile(min_per_sample, 0.01).item())

        frac_samples_below_40 = float((min_per_sample < 4.0).float().mean().item())
        frac_samples_below_wall = float((min_per_sample < self.wall_threshold).float().mean().item())

        debug_info = None
        should_debug_close = (
            self.debug_min_pair
            and (min_nonbonded_global < self.catastrophic_min * 1.2)
            and (not self.debug_once or not self._has_logged_close_contact)
        )

        periodic = (step is not None) and (int(step) % 50 == 0)

        if (
            should_debug_close
            or (min_nonbonded_global <= self.catastrophic_min)
            or (min_nonbonded_global < self.early_warning_min)
        ):
            flat_min_idx = int(all_distances.argmin().item())
            batch_idx = flat_min_idx // all_distances.shape[1]
            pair_idx_in_batch = flat_min_idx % all_distances.shape[1]

            pair_flat = int(flat_pair_idx[pair_idx_in_batch].item())
            i = pair_flat // L
            j = pair_flat % L
            seq_sep = abs(i - j)

            debug_info = {
                "min_value": min_nonbonded_global,
                "batch_idx": batch_idx,
                "i": i,
                "j": j,
                "seq_sep": seq_sep,
                "note": f"Nonbonded pair (|i-j|={seq_sep} > {self.exclude})",
            }

            if min_nonbonded_global <= self.catastrophic_min:
                if (not self._has_logged_catastrophic_details) or periodic:
                    logger.error("CATASTROPHIC min distance: %.3fA | %s", min_nonbonded_global, debug_info)
                    self._has_logged_catastrophic_details = True
                else:
                    logger.error("CATASTROPHIC min distance: %.3fA (details suppressed)", min_nonbonded_global)

            elif min_nonbonded_global < self.early_warning_min:
                if (not self._has_logged_early_warning) or periodic:
                    logger.warning("EARLY WARNING min distance: %.3fA | %s", min_nonbonded_global, debug_info)
                    self._has_logged_early_warning = True
                else:
                    logger.warning("EARLY WARNING min distance: %.3fA (details suppressed)", min_nonbonded_global)

            elif should_debug_close:
                logger.warning("Close-contact debug: %s", debug_info)
                self._has_logged_close_contact = True

            if seq_sep <= self.exclude:
                raise RuntimeError(f"BUG: Found pair with |i-j|={seq_sep} <= exclude={self.exclude} in nonbonded mask!")

        near_distances = all_distances[all_distances < self.max_dist_for_near]
        if near_distances.numel() > 0:
            q01_near = float(torch.quantile(near_distances, 0.01).item())
            n_near = int(near_distances.numel())
        else:
            q01_near = float("nan")
            n_near = 0

        return {
            "min_nonbonded": min_nonbonded_global,
            "min_nonbonded_q10": min_nonbonded_q10,
            "min_nonbonded_q01": min_nonbonded_q01,
            "min_per_sample": min_per_sample.tolist(),
            "min_per_batch": min_per_sample.tolist(),
            "frac_samples_below_40": frac_samples_below_40,
            "frac_samples_below_wall": frac_samples_below_wall,
            "frac_samples_below_45": frac_samples_below_wall,
            "q01_near": q01_near,
            "n_total": n_total,
            "n_near": n_near,
            "debug_min_pair": debug_info,
        }

    def _clip_abs_bounds(self, x: float) -> float:
        return float(max(self.min_lambda, min(self.max_lambda, x)))

    def compute_update(self, current_gate: float, stats: dict) -> Tuple[float, str, dict]:
        """Compute lambda update using wall fraction control with EMA smoothing."""
        current_gate = self._clip_abs_bounds(float(current_gate))

        raw_frac = float(stats.get("frac_samples_below_wall", stats.get("frac_samples_below_45", 0.0)))
        glob_min = float(stats.get("min_nonbonded", float("inf")))
        q10 = float(stats.get("min_nonbonded_q10", stats.get("min_nonbonded", float("inf"))))

        cooldown_before = self._cooldown

        info = {
            "raw_frac": raw_frac,
            "target_frac": self.target_frac,
            "glob_min": glob_min,
            "q10": q10,
            "eta": self.eta,
            "ema_alpha": self.ema_alpha,
            "deadband": self.deadband,
            "max_step_ratio": self.max_step_ratio,
            "cooldown_before": cooldown_before,
        }

        # --- Catastrophic override (1.5x jump + cooldown) ---
        if glob_min <= self.catastrophic_min:
            self._cooldown = self.catastrophic_cooldown_steps
            new_gate = self._clip_abs_bounds(current_gate * 1.5)
            mult = (new_gate / current_gate) if current_gate > 0 else float("inf")

            info.update(
                {
                    "cooldown_after": self._cooldown,
                    "multiplier": mult,
                    "mode": "catastrophic",
                    "severity": 1.0,
                    "frac": None,
                    "error": None,
                    "raw_log_step": None,
                    "clipped_log_step": None,
                    "deadband_active": False,
                    "held": False,
                }
            )

            reason = f"CATASTROPHIC (min={glob_min:.3f} <= {self.catastrophic_min:.3f}) + cooldown={self._cooldown}"
            return new_gate, reason, info

        # --- Early warning (linear ramp, no cooldown reset) ---
        if glob_min < self.early_warning_min:
            # cooldown counts down on any non-catastrophic step
            if self._cooldown > 0:
                self._cooldown -= 1

            denom = max(1e-6, (self.early_warning_min - self.catastrophic_min))
            t = (self.early_warning_min - glob_min) / denom
            t = max(0.0, min(1.0, t))

            early_mult = 1.0 + t * (self.early_warning_max_mult - 1.0)

            new_gate = self._clip_abs_bounds(current_gate * early_mult)
            mult = (new_gate / current_gate) if current_gate > 0 else float("inf")

            info.update(
                {
                    "cooldown_after": self._cooldown,
                    "multiplier": mult,
                    "mode": "early_warning",
                    "severity": t,
                    "frac": None,
                    "error": None,
                    "raw_log_step": None,
                    "clipped_log_step": None,
                    "deadband_active": False,
                    "held": False,
                }
            )

            reason = (
                f"EARLY WARNING (min={glob_min:.3f} < {self.early_warning_min:.3f}) "
                f"t={t:.3f} ew_mult={early_mult:.3f} -> x{mult:.3f}"
            )
            return new_gate, reason, info

        # --- Control path: Update EMA and apply control law ---
        if self._frac_ema is None:
            self._frac_ema = raw_frac
        else:
            self._frac_ema = (1.0 - self.ema_alpha) * self._frac_ema + self.ema_alpha * raw_frac

        frac_ctrl = float(self._frac_ema)
        info["frac"] = frac_ctrl

        error = frac_ctrl - self.target_frac
        max_log_step = math.log(self.max_step_ratio)

        deadband_active = abs(error) < self.deadband
        if deadband_active:
            raw_log_step = 0.0
            clipped_log_step = 0.0
            capped = False
        else:
            raw_log_step = self.eta * error
            clipped_log_step = max(-max_log_step, min(max_log_step, raw_log_step))
            capped = abs(clipped_log_step - raw_log_step) > 1e-12

        proposed_gate = self._clip_abs_bounds(current_gate * math.exp(clipped_log_step))

        held = False
        if cooldown_before > 0 and proposed_gate < current_gate:
            proposed_gate = current_gate
            held = True

        if self._cooldown > 0:
            self._cooldown -= 1

        mult = (proposed_gate / current_gate) if current_gate > 0 else float("inf")

        reason = (
            f"CTRL wall_ema={frac_ctrl:.2%} (raw={raw_frac:.2%}) target={self.target_frac:.2%} "
            f"err={error:+.4f} log_step={clipped_log_step:+.3f}"
            + (" (capped)" if capped else "")
            + (" (deadband)" if deadband_active else "")
        )
        if held:
            reason = f"HOLD (cooldown={cooldown_before}) | " + reason

        info.update(
            {
                "error": error,
                "raw_log_step": raw_log_step,
                "clipped_log_step": clipped_log_step,
                "multiplier": mult,
                "held": held,
                "deadband_active": deadband_active,
                "cooldown_after": self._cooldown,
                "mode": "control",
                "severity": None,
            }
        )

        return proposed_gate, reason, info

    def get_loss(self, stats: dict) -> float:
        """Loss-like scalar for logging/plots: |frac - target| * 100 (percentage points)."""
        raw_frac = float(stats.get("frac_samples_below_wall", stats.get("frac_samples_below_45", 0.0)))
        if self._frac_ema is not None:
            return abs(float(self._frac_ema) - self.target_frac) * 100.0
        return abs(raw_frac - self.target_frac) * 100.0

    def log_calibration(self, step: int, current: float, new: float, reason: str, info: dict):
        """Log calibration step with wall fraction and cooldown info."""
        mode = info.get("mode", "unknown")

        if mode == "control":
            logger.info(
                "  Repulsion step %d [%s] | lambda=%.4f -> %.4f (x%.3f) | wall_ema=%.2f%% (raw=%.2f%%) target=%.2f%% | "
                "min=%.3f | q10=%.3f | cooldown=%d->%d | %s",
                step,
                mode,
                float(current),
                float(new),
                float(info.get("multiplier", float("nan"))),
                100.0 * float(info.get("frac", 0.0)),
                100.0 * float(info.get("raw_frac", 0.0)),
                100.0 * float(info.get("target_frac", 0.0)),
                float(info.get("glob_min", float("nan"))),
                float(info.get("q10", float("nan"))),
                int(info.get("cooldown_before", 0)),
                int(info.get("cooldown_after", 0)),
                reason,
            )
        elif mode == "early_warning":
            logger.info(
                "  Repulsion step %d [%s] | lambda=%.4f -> %.4f (x%.3f) | severity=%.2f | wall_raw=%.2f%% target=%.2f%% | "
                "min=%.3f | q10=%.3f | cooldown=%d->%d | %s",
                step,
                mode,
                float(current),
                float(new),
                float(info.get("multiplier", float("nan"))),
                float(info.get("severity", 0.0)),
                100.0 * float(info.get("raw_frac", 0.0)),
                100.0 * float(info.get("target_frac", 0.0)),
                float(info.get("glob_min", float("nan"))),
                float(info.get("q10", float("nan"))),
                int(info.get("cooldown_before", 0)),
                int(info.get("cooldown_after", 0)),
                reason,
            )
        else:  # catastrophic
            logger.info(
                "  Repulsion step %d [%s] | lambda=%.4f -> %.4f (x%.3f) | wall_raw=%.2f%% target=%.2f%% | "
                "min=%.3f | q10=%.3f | cooldown=%d->%d | %s",
                step,
                mode,
                float(current),
                float(new),
                float(info.get("multiplier", float("nan"))),
                100.0 * float(info.get("raw_frac", 0.0)),
                100.0 * float(info.get("target_frac", 0.0)),
                float(info.get("glob_min", float("nan"))),
                float(info.get("q10", float("nan"))),
                int(info.get("cooldown_before", 0)),
                int(info.get("cooldown_after", 0)),
                reason,
            )
