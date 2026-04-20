"""Internal coordinate Langevin integrator.

src/calphaebm/simulation/ic_langevin.py

Runs Langevin dynamics in (θ, φ) space with bond lengths fixed at exactly
3.8 Å by NeRF reconstruction. This permanently solves the bond length drift
problem that occurs in Cartesian space simulation.

How it works
------------
At each step:

  1. Reconstruct R from (θ, φ, anchor) via NeRF — bonds are exactly 3.8 Å
  2. Evaluate E(R, seq) using all energy terms unchanged
  3. Compute dE/dθ and dE/dφ via autograd (chain rule through NeRF)
  4. Cap forces in (θ, φ) space
  5. Apply Langevin update: θ ← θ - dt·dE/dθ + √(2dt/β)·ξ_θ
                            φ ← φ - dt·dE/dφ + √(2dt/β)·ξ_φ
  6. Wrap φ to [-π, π] (angles are periodic)
  7. Clamp θ to (0, π) (bond angles must be positive and < 180°)

Bond lengths are never touched — they are exactly 3.8 Å by construction
at every step, not as a constraint but as a geometric identity.

Noise scaling
-------------
The noise magnitude for θ and φ updates uses the same dt and β as the
Cartesian integrator. The effective step size in Cartesian space depends
on the Jacobian of the NeRF reconstruction, but for sampling purposes
(not dynamics) this is acceptable — we want correct Boltzmann statistics,
and the Langevin dynamics converges to the correct distribution regardless
of the coordinate system provided the noise is isotropic in the DOF space.

Anchor
------
The anchor (first 3 atoms) is fixed throughout simulation. It sets global
translation and orientation, which are irrelevant to all energy terms
(distances and angles are translation/rotation invariant).

Usage
-----
    from calphaebm.simulation.ic_langevin import ICLangevinSimulator

    sim = ICLangevinSimulator(
        model=total_energy,
        seq=seq,
        R_init=R_native,
        step_size=1e-5,
        beta=1.0,
        force_cap=50.0,
    )

    for step in range(n_steps):
        R, E, info = sim.step()
        # R has exact 3.8 Å bonds at every step
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.simulation.fixman import fixman_potential
from calphaebm.utils.logging import get_logger
from calphaebm.utils.math import wrap_to_pi

logger = get_logger()


@dataclass
class ICStepInfo:
    """Diagnostic info returned at each simulation step."""

    step: int
    energy: float
    theta_grad_norm: float
    phi_grad_norm: float
    theta_std: float
    phi_std: float
    bond_length_mean: float  # should be ~3.8 always
    bond_length_std: float  # should be ~0.000 always
    bond_length_min: float  # should be ~3.8 always
    bond_length_max: float  # should be ~3.8 always


class ICLangevinSimulator:
    """Langevin dynamics in internal coordinate space.

    Maintains state as (theta, phi) — bond angles and torsion angles.
    Bond lengths are fixed at exactly 3.8 Å by NeRF reconstruction.

    Args:
        model:      TotalEnergy model (unchanged from Cartesian version)
        seq:        (1, L) amino acid sequence tensor
        R_init:     (1, L, 3) or (L, 3) initial Cα coordinates (native structure)
        step_size:  Langevin step size dt
        beta:       Inverse temperature (1/kT)
        force_cap:  Maximum force magnitude in (θ, φ) space
        bond:       Fixed Cα-Cα bond length (default 3.8 Å)
        lengths:    (1,) or scalar — real chain length for padding-aware energy.
                    CRITICAL: without this, padding atoms corrupt every force computation.
        device:     Torch device
    """

    def __init__(
        self,
        model: nn.Module,
        seq: torch.Tensor,
        R_init: torch.Tensor,
        step_size: float = 1e-5,
        beta: float = 1.0,
        force_cap: float = 50.0,
        bond: float = 3.8,
        lengths: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.step_size = float(step_size)
        self.beta = float(beta)
        self.force_cap = float(force_cap)
        self.bond = float(bond)
        self.step_num = 0

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        # Ensure (1, L, 3)
        if R_init.dim() == 2:
            R_init = R_init.unsqueeze(0)
        R_init = R_init.float().to(device)

        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        self.seq = seq.to(device)

        # Lengths for padding-aware energy evaluation
        if lengths is not None:
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)
            self.lengths = lengths.to(device)
        else:
            self.lengths = None

        # Fixed anchor — first 3 atoms never move
        self.anchor = extract_anchor(R_init)  # (1, 3, 3)

        # Initial internal coordinates
        theta_init, phi_init = coords_to_internal(R_init)  # (1, L-2), (1, L-3)

        # State: θ and φ are the only degrees of freedom
        # Requires grad so autograd can compute dE/dθ and dE/dφ
        self.theta = theta_init.clone().detach().requires_grad_(True)
        self.phi = phi_init.clone().detach().requires_grad_(True)

        L = R_init.shape[1]
        logger.debug("ICLangevinSimulator initialized:")
        logger.debug("  L=%d  theta shape=%s  phi shape=%s", L, tuple(self.theta.shape), tuple(self.phi.shape))
        logger.debug("  step_size=%.2e  beta=%.1f  force_cap=%.1f  bond=%.2fÅ", step_size, beta, force_cap, bond)
        logger.debug("  Anchor fixed: first 3 atoms set from R_init")
        logger.debug("  Bond lengths: EXACTLY %.2fÅ by NeRF construction (not a penalty)", bond)

    @property
    def L(self) -> int:
        return int(self.theta.shape[1]) + 2

    def _get_R(self) -> torch.Tensor:
        """Reconstruct Cartesian coordinates from current (θ, φ).

        Returns (1, L, 3) with bonds exactly = self.bond Å.
        """
        return nerf_reconstruct(self.theta, self.phi, self.anchor, bond=self.bond)

    def _compute_gradients(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass + autograd to get dE/dθ and dE/dφ.

        Includes the Fixman correction U_Fixman = -(1/β)·Σ log sin(θᵢ)
        so that the stationary distribution is exactly P(R) ∝ exp(-β·E(R))
        in Cartesian space rather than the IC-biased distribution.

        Returns:
            E:          scalar model energy (1,) — without Fixman term
            grad_theta: (1, L-2) gradient of U_eff = E + U_Fixman w.r.t. θ
            grad_phi:   (1, L-3) gradient of U_eff w.r.t. φ (Fixman is θ-only)
        """
        theta = self.theta.requires_grad_(True)
        phi = self.phi.requires_grad_(True)

        R = nerf_reconstruct(theta, phi, self.anchor, bond=self.bond)
        E = self.model(R, self.seq, lengths=self.lengths)  # (1,)

        # Fixman correction: U_Fixman(θ) = -(1/β)·Σ log sin(θᵢ)
        # Added before backward so autograd includes ∇_θ U_Fixman = -(1/β)·cot(θᵢ)
        U_fixman = fixman_potential(theta, self.beta)
        E_eff = E.sum() + U_fixman

        grads = torch.autograd.grad(
            E_eff,
            [theta, phi],
            create_graph=False,
            retain_graph=False,
        )

        grad_theta = torch.nan_to_num(grads[0].detach(), nan=0.0, posinf=0.0, neginf=0.0)
        grad_phi = torch.nan_to_num(grads[1].detach(), nan=0.0, posinf=0.0, neginf=0.0)

        return E.detach(), grad_theta, grad_phi

    def _cap_forces(
        self,
        grad_theta: torch.Tensor,
        grad_phi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cap force magnitude per DOF to prevent runaway steps."""
        grad_theta = grad_theta.clamp(-self.force_cap, self.force_cap)
        grad_phi = grad_phi.clamp(-self.force_cap, self.force_cap)
        return grad_theta, grad_phi

    def step(self) -> Tuple[torch.Tensor, torch.Tensor, ICStepInfo]:
        """Take one Langevin step in (θ, φ) space.

        Returns:
            R:    (1, L, 3) reconstructed coordinates (bonds exactly 3.8 Å)
            E:    (1,) energy at current step
            info: ICStepInfo diagnostic struct
        """
        dt = self.step_size
        beta = self.beta
        noise_scale = (2.0 * dt / beta) ** 0.5

        # 1) Compute gradients
        E, grad_theta, grad_phi = self._compute_gradients()

        # 2) Cap forces
        grad_theta, grad_phi = self._cap_forces(grad_theta, grad_phi)

        # 3) Langevin update (no_grad — we're updating the values, not building a graph)
        with torch.no_grad():
            noise_theta = torch.randn_like(self.theta) * noise_scale
            noise_phi = torch.randn_like(self.phi) * noise_scale

            new_theta = self.theta - dt * grad_theta + noise_theta
            new_phi = self.phi - dt * grad_phi + noise_phi

            # θ must stay in (0, π) — bond angles are physically bounded
            new_theta = new_theta.clamp(0.01, torch.pi - 0.01)

            # φ is periodic — wrap to [-π, π]
            new_phi = wrap_to_pi(new_phi)

            # Update state
            self.theta = new_theta.requires_grad_(True)
            self.phi = new_phi.requires_grad_(True)

        # 4) Reconstruct R for return / logging
        with torch.no_grad():
            R = self._get_R()

            # Verify bond lengths (cheap diagnostic)
            diffs = R[:, 1:, :] - R[:, :-1, :]
            bond_lengths = torch.sqrt((diffs * diffs).sum(dim=-1))  # (1, L-1)
            bl_mean = bond_lengths.mean().item()
            bl_std = bond_lengths.std().item()
            bl_min = bond_lengths.min().item()
            bl_max = bond_lengths.max().item()

        self.step_num += 1

        info = ICStepInfo(
            step=self.step_num,
            energy=float(E.mean().item()),
            theta_grad_norm=float(grad_theta.norm().item()),
            phi_grad_norm=float(grad_phi.norm().item()),
            theta_std=float(self.theta.std().item()),
            phi_std=float(self.phi.std().item()),
            bond_length_mean=bl_mean,
            bond_length_std=bl_std,
            bond_length_min=bl_min,
            bond_length_max=bl_max,
        )

        return R, E, info

    def run(
        self,
        n_steps: int,
        save_every: int = 500,
        log_every: int = 1000,
    ) -> Tuple[list[torch.Tensor], list[float]]:
        """Run simulation for n_steps, returning saved frames and energies.

        Args:
            n_steps:    Total number of Langevin steps
            save_every: Save coordinates every N steps
            log_every:  Log progress every N steps

        Returns:
            frames:   List of (1, L, 3) coordinate tensors
            energies: List of scalar energy values
        """
        frames = []
        energies = []

        for i in range(n_steps):
            R, E, info = self.step()

            if (i + 1) % save_every == 0:
                frames.append(R.detach().cpu())
                energies.append(info.energy)

            if (i + 1) % log_every == 0:
                logger.info(
                    "step %6d/%d | E=%+.3f | " "bond=%.4f±%.4fÅ [%.4f,%.4f] | " "|∇θ|=%.3f |∇φ|=%.3f",
                    i + 1,
                    n_steps,
                    info.energy,
                    info.bond_length_mean,
                    info.bond_length_std,
                    info.bond_length_min,
                    info.bond_length_max,
                    info.theta_grad_norm,
                    info.phi_grad_norm,
                )

                # Bond length should be ~3.800 ± 0.000 at every step
                if abs(info.bond_length_mean - self.bond) > 0.01:
                    logger.warning(
                        "Unexpected bond length drift: mean=%.4fÅ (expected %.3fÅ)", info.bond_length_mean, self.bond
                    )

        return frames, energies

    @torch.no_grad()
    def get_current_R(self) -> torch.Tensor:
        """Return current Cartesian coordinates without stepping."""
        return self._get_R()

    @torch.no_grad()
    def get_current_angles(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current (theta, phi) tensors."""
        return self.theta.detach(), self.phi.detach()

    def reset(self, R_init: torch.Tensor) -> None:
        """Reset simulation to a new starting structure."""
        if R_init.dim() == 2:
            R_init = R_init.unsqueeze(0)
        R_init = R_init.float().to(self.device)
        self.anchor = extract_anchor(R_init)
        theta_init, phi_init = coords_to_internal(R_init)
        self.theta = theta_init.clone().detach().requires_grad_(True)
        self.phi = phi_init.clone().detach().requires_grad_(True)
        self.step_num = 0
        logger.info("ICLangevinSimulator reset to new starting structure.")
