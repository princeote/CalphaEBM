"""PyTorch-based Langevin dynamics simulator."""

from __future__ import annotations

from typing import List, Optional

import torch

from calphaebm.simulation.base import SimulationResult, Simulator
from calphaebm.utils.logging import get_logger
from calphaebm.utils.math import safe_norm

logger = get_logger()


def langevin_sample(model, R0, seq, n_steps=50, step_size=1e-3, force_cap=50.0, log_every=None):
    """Simple Langevin dynamics for quick sampling.

    This function is designed for quick sampling during calibration and validation.
    It ensures gradients are properly enabled for each step.

    Args:
        model: Energy model
        R0: (B, L, 3) initial coordinates
        seq: (B, L) amino acid indices
        n_steps: Number of steps
        step_size: Step size
        force_cap: Maximum force magnitude
        log_every: If not None, save trajectory every log_every steps

    Returns:
        List of trajectory frames
    """
    R = R0.clone()
    trajectories = [R.clone()]

    for step in range(n_steps):
        # CRITICAL: Ensure gradients are enabled for this step
        # This is necessary even if we're in a no_grad context outside
        with torch.enable_grad():
            # Create a fresh tensor that requires gradients
            R_grad = R.clone().detach().requires_grad_(True)

            # Compute energy and gradients
            E = model(R_grad, seq).sum()

            # Debug: Check if energy requires grad
            if not E.requires_grad:
                print(f"WARNING: E.requires_grad = {E.requires_grad} at step {step}")
                print(f"E.grad_fn = {E.grad_fn}")

            grad = torch.autograd.grad(E, R_grad, create_graph=False, retain_graph=False)[0]

        # Force clipping (do this outside the gradient context)
        grad_norm = torch.norm(grad, dim=-1, keepdim=True)
        scale = torch.clamp(force_cap / (grad_norm + 1e-8), max=1.0)
        grad = grad * scale

        # Update WITHOUT gradients - create new tensor
        with torch.no_grad():
            noise = torch.sqrt(torch.tensor(2 * step_size, device=R_grad.device)) * torch.randn_like(R_grad)
            R_new = R_grad - step_size * grad + noise

        # Replace R with the new tensor
        R = R_new

        if log_every and step % log_every == 0:
            trajectories.append(R.clone())

    # Always add final frame
    trajectories.append(R.clone())

    return trajectories


class PyTorchSimulator(Simulator):
    """Overdamped Langevin simulator using PyTorch autograd.

    Implements:
        R_{t+1} = R_t + step_size * F(R_t) + sqrt(2 * step_size / beta) * noise

    with optional force clipping and NaN detection.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        beta: float = 1.0,
        force_cap: Optional[float] = 50.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model)
        self.beta = beta
        self.force_cap = force_cap
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_forces(self, R: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Compute forces = -grad(E)."""
        # Ensure R requires gradients
        if not R.requires_grad:
            R = R.detach().requires_grad_(True)

        E = self.model(R, seq).sum()

        # Debug check for gradient flow
        if not E.requires_grad:
            logger.warning("E.requires_grad is False in _compute_forces")

        F = -torch.autograd.grad(E, R, create_graph=False)[0]
        return F

    def _clip_forces(self, F: torch.Tensor) -> tuple:
        """Clip forces to force_cap, return clipped forces and clip fraction."""
        if self.force_cap is None or self.force_cap <= 0:
            return F, 0.0

        norms = safe_norm(F, dim=-1, keepdim=True)
        clip_mask = norms > self.force_cap
        clip_frac = clip_mask.float().mean().item()

        scale = torch.clamp(self.force_cap / (norms + 1e-12), max=1.0)
        F_clipped = F * scale

        return F_clipped, clip_frac

    def run(
        self,
        R0: torch.Tensor,
        seq: torch.Tensor,
        n_steps: int,
        step_size: float,
        log_every: int = 50,
        silent_progress: bool = False,
        **kwargs,
    ) -> SimulationResult:
        """Run Langevin dynamics with NaN detection.

        Args:
            R0: (B, L, 3) initial coordinates.
            seq: (B, L) amino acid indices.
            n_steps: Number of steps.
            step_size: Integration step size.
            log_every: Logging frequency.
            **kwargs: Additional arguments passed to observers.

        Returns:
            SimulationResult containing trajectory and metadata.
        """
        R = R0.to(self.device)
        seq = seq.to(self.device)

        # Clear observers and add trajectory observer if none present
        if not self.observers:
            from calphaebm.simulation.observers import TrajectoryObserver

            self.add_observer(TrajectoryObserver(save_every=log_every))

        # Reset all observers
        for obs in self.observers:
            obs.reset()

        # Noise scale from fluctuation-dissipation
        noise_scale = (2.0 * step_size / max(self.beta, 1e-12)) ** 0.5

        logger.info(f"Starting Langevin simulation for {n_steps} steps")
        logger.info(f"  step_size = {step_size:.2e}, beta = {self.beta}")
        logger.info(f"  force_cap = {self.force_cap}")

        # Initial frame
        self._notify_observers(0, R, seq=seq, **kwargs)

        completed_steps = n_steps
        for step in range(1, n_steps + 1):
            # Compute forces (with gradients enabled)
            F = self._compute_forces(R, seq)

            # ===== NAN CHECK =====
            if torch.isnan(F).any():
                logger.error(f"NaN detected in forces at step {step}! Stopping simulation.")
                logger.error(f"  R stats: min={R.min():.3f}, max={R.max():.3f}, mean={R.mean():.3f}")
                logger.error(f"  F stats: min={F.min():.3f}, max={F.max():.3f}, mean={F.mean():.3f}")

                # Save the last valid frame for debugging
                torch.save(R, f"debug_last_valid_step_{step-1}.pt")
                logger.info(f"Saved last valid frame to debug_last_valid_step_{step-1}.pt")
                completed_steps = step - 1
                break

            # Check for extreme values (early warning)
            if torch.abs(F).max() > 1000:
                logger.warning(f"Large forces detected at step {step}: max|F|={torch.abs(F).max():.3f}")
            # ====================

            # Apply force clipping
            if self.force_cap:
                F, clip_frac = self._clip_forces(F)
            else:
                clip_frac = 0.0

            # Update positions (no gradients needed for update)
            with torch.no_grad():
                noise = noise_scale * torch.randn_like(R)
                R = R + step_size * F + noise

            # Check positions for NaN
            if torch.isnan(R).any():
                logger.error(f"NaN detected in positions at step {step}! Stopping simulation.")
                completed_steps = step - 1
                break

            # Notify observers
            self._notify_observers(
                step,
                R,
                seq=seq,
                forces=F,
                force_cap=self.force_cap,
                clip_frac=clip_frac,
                **kwargs,
            )

            # Log progress (suppressed when an observer owns the line)
            if not silent_progress and step % log_every == 0:
                with torch.no_grad():
                    E = self.model(R, seq).mean().item()
                    max_force = safe_norm(F, dim=-1).max().item()

                log_msg = f"step {step:6d}/{n_steps} | E={E:.3f} | max|F|={max_force:.3f}"
                if self.force_cap:
                    log_msg += f" | clip={clip_frac:.3f}"
                logger.info(log_msg)

        # Gather results
        data = self._gather_observer_data()

        # Create result object
        result = SimulationResult(
            trajectories=data.get("trajectory", []),
            energies=data.get("total_energy"),
            min_distances=data.get("min_distance"),
            clip_fractions=data.get("clip_fraction"),
            metadata={
                "n_steps": completed_steps,
                "target_steps": n_steps,
                "completed": completed_steps >= n_steps,
                "step_size": step_size,
                "beta": self.beta,
                "force_cap": self.force_cap,
                "device": str(self.device),
                **kwargs,
            },
        )

        logger.info(f"Simulation complete: {len(result.trajectories)} frames saved")
        if completed_steps < n_steps:
            logger.warning(f"Simulation stopped early at step {completed_steps}/{n_steps} due to NaN")

        return result

    def run_with_snapshots(
        self,
        R0: torch.Tensor,
        seq: torch.Tensor,
        n_steps: int,
        step_size: float,
        save_every: int = 50,
        **kwargs,
    ) -> List[torch.Tensor]:
        """Simplified interface returning only trajectory snapshots."""
        from calphaebm.simulation.observers import TrajectoryObserver

        self.clear_observers()
        self.add_observer(TrajectoryObserver(save_every=save_every))

        result = self.run(R0, seq, n_steps, step_size, **kwargs)

        return result.trajectories
