"""Convergence monitoring for phased training."""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from calphaebm.utils.logging import get_logger

logger = get_logger()


@dataclass
class ConvergenceCriteria:
    """Criteria for determining training convergence."""

    # Loss stability
    loss_window: int = 1000
    loss_tolerance: float = 0.1  # Relative change

    # Weight changes
    weight_window: int = 500
    weight_tolerance: float = 0.005  # Absolute change

    # Energy gap (for contrastive phases)
    gap_window: int = 500
    gap_tolerance: float = 0.5

    # Required consecutive checks
    min_consecutive: int = 3

    # Minimum steps before checking
    min_steps: int = 5000

    def __post_init__(self):
        self.loss_tolerance = float(self.loss_tolerance)
        self.weight_tolerance = float(self.weight_tolerance)
        self.gap_tolerance = float(self.gap_tolerance)


class ConvergenceMonitor:
    """Monitor training metrics for convergence."""

    def __init__(
        self,
        config: ConvergenceCriteria,
        phase: str,
        weight_names: Optional[List[str]] = None,
    ):
        self.config = config
        self.phase = phase
        self.weight_names = weight_names or []

        # History buffers
        self.loss_history = deque(maxlen=config.loss_window)
        self.weight_history = {name: deque(maxlen=config.weight_window) for name in self.weight_names}
        self.gap_history = deque(maxlen=config.gap_window)

        # Convergence state
        self.consecutive_passes = 0
        self.converged = False
        self.convergence_step = None
        self.convergence_reason = ""

        logger.debug(f"ConvergenceMonitor initialized for phase {phase}")
        logger.debug(f"  Loss window: {config.loss_window}, tolerance: {config.loss_tolerance*100:.1f}%")
        if weight_names:
            logger.debug(f"  Monitoring {len(weight_names)} weights")

    def update(self, metrics: Dict) -> Tuple[bool, Optional[int]]:
        """Update with new metrics and check convergence.

        Args:
            metrics: Dictionary with keys:
                - 'step': current step
                - 'loss': current loss value
                - (optional) weight names from self.weight_names
                - (optional) 'gap': energy gap for contrastive phases

        Returns:
            (converged, convergence_step)
        """
        step = metrics["step"]

        # Don't check before minimum steps
        if step < self.config.min_steps:
            return False, None

        # Update histories
        self.loss_history.append(metrics["loss"])

        for name in self.weight_names:
            if name in metrics:
                self.weight_history[name].append(metrics[name])

        if "gap" in metrics:
            self.gap_history.append(metrics["gap"])

        # Check convergence if we have enough data
        if len(self.loss_history) < self.config.loss_window:
            return False, None

        checks_passed = []

        # 1. Check loss stability
        loss_stable = self._check_loss_stability()
        checks_passed.append(loss_stable)
        if not loss_stable:
            logger.debug(f"Step {step}: Loss not stable")

        # 2. Check weight stability (if monitoring weights)
        if self.weight_names:
            weights_stable = self._check_weight_stability()
            checks_passed.append(weights_stable)
            if not weights_stable:
                logger.debug(f"Step {step}: Weights not stable")

        # 3. Check gap stability (if monitoring gap)
        if len(self.gap_history) == self.config.gap_window:
            gap_stable = self._check_gap_stability()
            checks_passed.append(gap_stable)
            if not gap_stable:
                logger.debug(f"Step {step}: Gap not stable")

        # All checks must pass
        all_passed = all(checks_passed)

        if all_passed:
            self.consecutive_passes += 1
            logger.debug(
                f"Step {step}: Convergence check passed ({self.consecutive_passes}/{self.config.min_consecutive})"
            )
        else:
            self.consecutive_passes = 0

        # Converged if we've had enough consecutive passes
        if self.consecutive_passes >= self.config.min_consecutive:
            self.converged = True
            self.convergence_step = step
            self.convergence_reason = self._get_reason()
            logger.info(f"🎯 Convergence achieved at step {step}")
            logger.info(f"  Reason: {self.convergence_reason}")
            return True, step

        return False, None

    def _check_loss_stability(self) -> bool:
        """Check if loss has stabilized."""
        loss_array = np.array(self.loss_history)
        mean_loss = np.mean(loss_array)

        # Check relative change over window
        first_half = loss_array[: len(loss_array) // 2]
        second_half = loss_array[len(loss_array) // 2 :]

        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)

        if mean_first == 0:
            return False

        rel_change = abs(mean_second - mean_first) / abs(mean_first)

        # Also check variance
        std_second = np.std(second_half)
        cv_second = std_second / (abs(mean_second) + 1e-8)  # Coefficient of variation

        stable = (rel_change < self.config.loss_tolerance) and (cv_second < self.config.loss_tolerance)

        logger.debug(f"Loss stability: rel_change={rel_change:.4f}, cv={cv_second:.4f}, stable={stable}")
        return stable

    def _check_weight_stability(self) -> bool:
        """Check if weights have stopped changing."""
        if not self.weight_names:
            return True

        stable_counts = []
        for name in self.weight_names:
            history = list(self.weight_history[name])
            if len(history) < 2:
                stable_counts.append(False)
                continue

            # Check max change over window
            first_val = history[0]
            last_val = history[-1]
            abs_change = abs(last_val - first_val)

            # Also check if changes are decreasing
            diffs = np.diff(history)
            trend_decreasing = np.mean(np.abs(diffs[-10:])) < np.mean(np.abs(diffs[:10])) if len(diffs) > 20 else True

            stable = (abs_change < self.config.weight_tolerance) and trend_decreasing
            stable_counts.append(stable)

            logger.debug(f"  {name}: change={abs_change:.6f}, stable={stable}")

        return all(stable_counts)

    def _check_gap_stability(self) -> bool:
        """Check if energy gap has stabilized."""
        gap_array = np.array(self.gap_history)

        first_half = gap_array[: len(gap_array) // 2]
        second_half = gap_array[len(gap_array) // 2 :]

        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)

        abs_change = abs(mean_second - mean_first)

        # Also check trend (should be increasing for contrastive loss)
        diffs = np.diff(gap_array)
        mean_slope = np.mean(diffs[-50:]) if len(diffs) > 50 else 0

        # Gap should be stable (not changing much)
        stable = abs_change < self.config.gap_tolerance

        logger.debug(f"Gap stability: change={abs_change:.4f}, slope={mean_slope:.4f}, stable={stable}")
        return stable

    def _get_reason(self) -> str:
        """Get human-readable convergence reason."""
        reasons = []

        # Convert deque to list for slicing
        loss_list = list(self.loss_history)
        if len(loss_list) >= 100:
            recent_losses = loss_list[-100:]
            reasons.append(f"Loss stable at {np.mean(recent_losses):.4f} ± {np.std(recent_losses):.4f}")

        if self.weight_names:
            reasons.append("Weights stabilized")

        if len(self.gap_history) == self.config.gap_window:
            gap_list = list(self.gap_history)
            if len(gap_list) >= 100:
                recent_gaps = gap_list[-100:]
                reasons.append(f"Gap stable at {np.mean(recent_gaps):.3f}")

        return " | ".join(reasons) if reasons else "No convergence criteria met"

    def get_summary(self) -> Dict:
        """Get convergence summary."""
        # Convert deque to list for safe slicing
        loss_list = list(self.loss_history) if self.loss_history else []

        # Calculate final loss stats if we have enough history
        final_loss = None
        loss_std = None
        if len(loss_list) >= 100:
            recent_losses = loss_list[-100:]
            final_loss = float(np.mean(recent_losses))
            loss_std = float(np.std(recent_losses))
        elif len(loss_list) > 0:
            # Use whatever we have
            final_loss = float(np.mean(loss_list))
            loss_std = float(np.std(loss_list))

        return {
            "converged": self.converged,
            "convergence_step": self.convergence_step,
            "reason": self.convergence_reason,
            "final_loss": final_loss,
            "loss_std": loss_std,
        }
