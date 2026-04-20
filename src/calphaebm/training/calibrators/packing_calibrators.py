"""Packing diagnostics and convergence detection for packing phase calibration."""

import math
from collections import deque
from typing import Optional, Tuple

import torch

from calphaebm.training.logging.diagnostics import ExponentialMovingAverage
from calphaebm.utils.logging import get_logger

logger = get_logger()


class PackingConvergenceDetector:
    """Sophisticated convergence detection for packing phase."""

    def __init__(
        self,
        patience_steps: int = 500,
        min_steps: int = 500,
        window: int = 10,
        delta_plateau: float = 0.01,
        correct_plateau: float = 0.02,
        target_delta: float = -0.02,
        target_correct: float = 0.60,
    ):
        """
        Args:
            patience_steps: Steps to wait after plateau before stopping
            min_steps: Minimum steps before considering convergence
            window: Number of history points for trend analysis
            delta_plateau: Max ΔE change over window to consider plateaued
            correct_plateau: Max correct rate change over window to consider plateaued
            target_delta: Minimum ΔE magnitude needed (more negative is better)
            target_correct: Minimum correct rate needed
        """
        self.patience_steps = patience_steps
        self.min_steps = min_steps
        self.window = window

        self.delta_plateau = delta_plateau
        self.correct_plateau = correct_plateau
        self.target_delta = target_delta
        self.target_correct = target_correct

        self.best_ema_delta = float("inf")  # minimizing ΔE (more negative is better)
        self.best_step: Optional[int] = None
        self.steps_since_improvement = 0

        self.history = deque(maxlen=max(window, 20))
        self._last_step = None

    def update(self, step: int, ema_delta: Optional[float], ema_correct: Optional[float]) -> Tuple[bool, str]:
        """Check if packing has converged.

        Returns:
            (converged, reason)
        """
        # Step delta for patience accounting
        if self._last_step is None:
            step_delta = 0
        else:
            step_delta = max(0, step - self._last_step)
        self._last_step = step

        # Only store history if both EMAs are finite
        if (
            ema_delta is not None
            and math.isfinite(ema_delta)
            and ema_correct is not None
            and math.isfinite(ema_correct)
        ):
            self.history.append((step, ema_delta, ema_correct))

        # Best update (minimizing ΔE) - only if we have valid delta
        if ema_delta is not None and math.isfinite(ema_delta):
            old_best = self.best_ema_delta
            if ema_delta < self.best_ema_delta:
                self.best_ema_delta = ema_delta
                self.best_step = step
                self.steps_since_improvement = 0

                # Log significant improvements (avoid inf in first log)
                if math.isfinite(old_best):
                    improvement = old_best - ema_delta
                    if improvement > 0.001:
                        logger.info(f"  New best ΔE: {ema_delta:.4f} (improved by {improvement:.4f})")
                else:
                    logger.info(f"  New best ΔE: {ema_delta:.4f}")
            else:
                self.steps_since_improvement += step_delta

        # Don't check convergence too early
        if step < self.min_steps:
            return False, f"below minimum steps ({step}/{self.min_steps})"

        if len(self.history) < self.window:
            return False, f"insufficient history ({len(self.history)}/{self.window})"

        # Need valid current values for plateau check
        if ema_delta is None or not math.isfinite(ema_delta) or ema_correct is None or not math.isfinite(ema_correct):
            return False, "invalid current metrics"

        # Plateau check once patience exceeded
        if self.steps_since_improvement >= self.patience_steps:
            # Get window endpoints
            recent = list(self.history)[-self.window :]
            _, d0, c0 = recent[0]
            _, d1, c1 = recent[-1]

            delta_trend = d1 - d0
            correct_trend = c1 - c0

            if (
                abs(delta_trend) < self.delta_plateau
                and abs(correct_trend) < self.correct_plateau
                and ema_delta < self.target_delta
                and ema_correct > self.target_correct
            ):
                reason = (
                    f"plateau: ΔE trend={delta_trend:+.4f}, "
                    f"correct trend={correct_trend:+.2%}, "
                    f"emaΔE={ema_delta:.4f}, emaCorrect={ema_correct:.2%}"
                )
                return True, reason

        return False, "training"


class PackingDiagnostics:
    """Diagnostics for packing term learning."""

    def __init__(self, window_size=10):
        self.E_pos_history = deque(maxlen=window_size)
        self.E_neg_history = deque(maxlen=window_size)
        self.delta_E_history = deque(maxlen=window_size)
        self.correct_order_history = deque(maxlen=window_size)

        # EMAs for tracking trends
        self.ema_delta = ExponentialMovingAverage(0.95)
        self.ema_correct = ExponentialMovingAverage(0.95)

        # Convergence detector
        self.convergence_detector = PackingConvergenceDetector(
            patience_steps=500,
            min_steps=500,
            window=10,
            delta_plateau=0.01,
            correct_plateau=0.02,
            target_delta=-0.02,
            target_correct=0.60,
        )

        # Store last convergence result
        self._last_converged = False
        self._last_reason = ""

    def log(self, E_pos: torch.Tensor, E_neg: torch.Tensor, step: int, n_pairs: int):
        """Log packing diagnostics."""
        with torch.no_grad():
            E_pos_mean = E_pos.mean().item()
            E_neg_mean = E_neg.mean().item()
            delta_E = (E_pos - E_neg).mean().item()
            correct_order = (E_pos < E_neg).float().mean().item()

            self.E_pos_history.append((step, E_pos_mean))
            self.E_neg_history.append((step, E_neg_mean))
            self.delta_E_history.append((step, delta_E))
            self.correct_order_history.append((step, correct_order))

            # Update EMAs
            ema_delta = self.ema_delta.update(delta_E)
            ema_correct = self.ema_correct.update(correct_order)

            # Check convergence and store result
            self._last_converged, self._last_reason = self.convergence_detector.update(step, ema_delta, ema_correct)

            # Format values for logging (handle None)
            delta_str = f"{delta_E:.4f}" if math.isfinite(delta_E) else "NaN"
            ema_delta_str = f"{ema_delta:.4f}" if ema_delta is not None else "N/A"
            correct_str = f"{correct_order:.2%}" if math.isfinite(correct_order) else "NaN"
            ema_correct_str = f"{ema_correct:.2%}" if ema_correct is not None else "N/A"

            # Log with convergence status
            status = "CONVERGED" if self._last_converged else "training"
            logger.info(
                f"  PACKING DIAG step {step}: "
                f"E_pos={E_pos_mean:.4f} | E_neg={E_neg_mean:.4f} | "
                f"ΔE={delta_str} (EMA={ema_delta_str}) | "
                f"correct={correct_str} (EMA={ema_correct_str}) | "
                f"n_pairs={n_pairs} | {status}"
            )

            return self._last_converged, self._last_reason

    def has_converged(self) -> bool:
        """Check if packing has converged based on last detector result."""
        return self._last_converged

    def summary(self):
        """Print summary of packing diagnostics."""
        if not self.delta_E_history:
            return

        # Get current values (handle None)
        current_delta = self.ema_delta.value
        current_correct = self.ema_correct.value

        delta_str = f"{current_delta:.4f}" if current_delta is not None else "N/A"
        correct_str = f"{current_correct:.2%}" if current_correct is not None else "N/A"

        # Handle best value (might still be inf if no finite EMA observed)
        best_delta = self.convergence_detector.best_ema_delta
        if math.isfinite(best_delta):
            best_str = f"{best_delta:.4f}"
            best_step_str = str(self.convergence_detector.best_step)
        else:
            best_str = "N/A (no finite EMA observed)"
            best_step_str = "N/A"

        logger.info(f"\n{'='*60}")
        logger.info(f"Packing Learning Summary")
        logger.info(f"{'='*60}")
        logger.info(f"  Best EMA ΔE = {best_str} at step {best_step_str}")
        logger.info(f"  Current EMA ΔE = {delta_str}")
        logger.info(f"  Current EMA correct = {correct_str}")
        logger.info(f"  Steps since improvement = {self.convergence_detector.steps_since_improvement}")
        logger.info(f"  Converged: {'Yes' if self.has_converged() else 'No'}")
        if self.has_converged():
            logger.info(f"  Reason: {self._last_reason}")
        logger.info(f"{'='*60}\n")
