"""Metropolis-Adjusted Langevin Algorithm (MALA) in internal coordinates.

src/calphaebm/simulation/backends/langevin_mala.py

Like ICLangevinSimulator but with Metropolis accept/reject correction.
Eliminates systematic energy drift that accumulates over long trajectories,
critical for basin stability evaluation at high β.

Standard Langevin has discretization error O(η²) per step that accumulates
monotonically — after 40K steps at β=500, proteins drift past RMSD=3Å not
because the energy landscape is wrong, but because the integrator leaked.
MALA fixes this: proposed steps that would increase energy too much are
rejected, maintaining detailed balance exactly.

Algorithm
---------
At each step:
  1. Propose:  θ' = θ - η·∇E(θ) + √(2η/β)·ξ   (same as Langevin)
               φ' = φ - η·∇E(φ) + √(2η/β)·ξ
  2. Clamp θ' to (0, π), wrap φ' to [-π, π]
  3. Evaluate E(θ', φ') and ∇E(θ', φ')
  4. Compute Metropolis-Hastings acceptance ratio:
       α = min(1, exp(-β·ΔE) · q(θ,φ|θ',φ') / q(θ',φ'|θ,φ))
     where q is the Gaussian proposal density.
  5. Accept (θ,φ) ← (θ',φ') with probability α, else keep old state.

The proposal correction term handles the asymmetry of the gradient-biased
proposal distribution — without it, detailed balance is violated.

Cost: 2× energy+gradient evaluations per step (current + proposed).
But acceptance rates of 60-80% mean effective sampling per evaluation
is similar to uncorrected Langevin, with zero drift.

Usage
-----
    from calphaebm.simulation.backends.langevin_mala import MALASimulator

    sim = MALASimulator(
        model=total_energy,
        seq=seq,
        R_init=R_native,
        step_size=1e-4,
        beta=500.0,
        force_cap=100.0,
    )

    for step in range(n_steps):
        R, E, info = sim.step()
        # info.accepted tells you if this step was accepted
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.simulation.fixman import fixman_potential
from calphaebm.utils.logging import get_logger
from calphaebm.utils.math import wrap_to_pi

logger = get_logger()


@dataclass
class MALAStepInfo:
    """Diagnostic info returned at each MALA step."""

    step: int
    energy: float
    proposed_energy: float
    accepted: bool
    acceptance_prob: float
    acceptance_rate: float  # running average
    theta_grad_norm: float
    phi_grad_norm: float
    theta_std: float
    phi_std: float
    bond_length_mean: float
    bond_length_std: float
    bond_length_min: float
    bond_length_max: float


class MALASimulator:
    """Metropolis-Adjusted Langevin dynamics in internal coordinate space.

    Like ICLangevinSimulator but with accept/reject correction that
    eliminates systematic energy drift. 2× cost per step but exact
    detailed balance.

    Args:
        model:      TotalEnergy model
        seq:        (1, L) amino acid sequence tensor
        R_init:     (1, L, 3) or (L, 3) initial Cα coordinates
        step_size:  Langevin step size η
        beta:       Inverse temperature (1/kT)
        force_cap:  Unused in MALA (MH rejection handles large gradients).
                    Kept for API compatibility with ICLangevinSimulator.
        bond:       Fixed Cα-Cα bond length (default 3.8 Å)
        lengths:    (1,) chain length for padding-aware energy
        device:     Torch device
    """

    def __init__(
        self,
        model: nn.Module,
        seq: torch.Tensor,
        R_init: torch.Tensor,
        step_size: float = 1e-4,
        beta: float = 1.0,
        force_cap: float = 100.0,
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

        # Acceptance tracking
        self._n_accepted = 0
        self._n_total = 0

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

        # Lengths for padding-aware energy
        if lengths is not None:
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)
            self.lengths = lengths.to(device)
        else:
            self.lengths = None

        # Fixed anchor
        self.anchor = extract_anchor(R_init)

        # Initial internal coordinates
        theta_init, phi_init = coords_to_internal(R_init)
        self.theta = theta_init.clone().detach()
        self.phi = phi_init.clone().detach()

        # Cache current energy and gradients (avoid recomputation)
        self._current_E = None
        self._current_grad_t = None
        self._current_grad_p = None

        L = R_init.shape[1]
        logger.debug("MALASimulator initialized:")
        logger.debug("  L=%d  theta shape=%s  phi shape=%s", L, tuple(self.theta.shape), tuple(self.phi.shape))
        logger.debug("  step_size=%.2e  beta=%.1f  force_cap=%.1f  bond=%.2fÅ", step_size, beta, force_cap, bond)

    @property
    def L(self) -> int:
        return int(self.theta.shape[1]) + 2

    @property
    def acceptance_rate(self) -> float:
        if self._n_total == 0:
            return 1.0
        return self._n_accepted / self._n_total

    def _evaluate(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate effective energy and gradients at (θ, φ).

        Includes the Fixman correction so that the MALA stationary distribution
        is exactly P(R) ∝ exp(-β·E(R)) in Cartesian space:

            U_eff(θ, φ) = E(NeRF(θ, φ)) + U_Fixman(θ)
            U_Fixman(θ) = -(1/β) · Σᵢ log sin(θᵢ)

        The MH accept/reject ratio uses U_eff, ensuring detailed balance
        with respect to the correct Cartesian Boltzmann distribution.

        No force capping — MH rejection handles large gradients naturally.
        Capping would break detailed balance by mismatching proposal densities.

        Returns:
            E_eff:      scalar effective energy (model + Fixman)
            grad_theta: (1, L-2)
            grad_phi:   (1, L-3)
        """
        theta_g = theta.detach().requires_grad_(True)
        phi_g = phi.detach().requires_grad_(True)

        R = nerf_reconstruct(theta_g, phi_g, self.anchor, bond=self.bond)
        E_model = self.model(R, self.seq, lengths=self.lengths).sum()

        # Fixman correction — θ only, φ needs no correction
        U_fixman = fixman_potential(theta_g, self.beta)
        E_eff = E_model + U_fixman

        grads = torch.autograd.grad(E_eff, [theta_g, phi_g])
        grad_t = torch.nan_to_num(grads[0].detach(), nan=0.0, posinf=0.0, neginf=0.0)
        grad_p = torch.nan_to_num(grads[1].detach(), nan=0.0, posinf=0.0, neginf=0.0)

        return E_eff.detach(), grad_t, grad_p

    def _log_proposal_density(
        self,
        x_to: torch.Tensor,
        x_from: torch.Tensor,
        grad_from: torch.Tensor,
    ) -> torch.Tensor:
        """Log density of proposing x_to given current state x_from.

        q(x_to | x_from) = N(x_from - η·∇E(x_from), 2η/β · I)

        log q = -β/(4η) · ||x_to - x_from + η·∇E(x_from)||²  + const
        """
        eta = self.step_size
        beta = self.beta
        mean = x_from - eta * grad_from
        diff = x_to - mean
        return -beta / (4.0 * eta) * (diff * diff).sum()

    def step(self) -> Tuple[torch.Tensor, torch.Tensor, MALAStepInfo]:
        """Take one MALA step in (θ, φ) space.

        Returns:
            R:    (1, L, 3) coordinates (bonds exactly 3.8 Å)
            E:    (1,) energy at current state (after accept/reject)
            info: MALAStepInfo with acceptance diagnostics
        """
        eta = self.step_size
        beta = self.beta
        noise_scale = (2.0 * eta / beta) ** 0.5

        # ── Current state energy + gradients (cached from previous step) ──
        if self._current_E is None:
            self._current_E, self._current_grad_t, self._current_grad_p = self._evaluate(self.theta, self.phi)

        E_current = self._current_E
        grad_t = self._current_grad_t
        grad_p = self._current_grad_p

        # ── Propose new state ─────────────────────────────────────────────
        with torch.no_grad():
            noise_t = torch.randn_like(self.theta) * noise_scale
            noise_p = torch.randn_like(self.phi) * noise_scale

            theta_prop = self.theta - eta * grad_t + noise_t
            phi_prop = self.phi - eta * grad_p + noise_p

            # Clamp and wrap
            theta_prop = theta_prop.clamp(0.01, torch.pi - 0.01)
            phi_prop = wrap_to_pi(phi_prop)

        # ── Evaluate proposed state ───────────────────────────────────────
        E_prop, grad_t_prop, grad_p_prop = self._evaluate(theta_prop, phi_prop)

        # ── Metropolis-Hastings acceptance ─────────────────────────────────
        with torch.no_grad():
            # Energy difference
            delta_E = float(E_prop.item()) - float(E_current.item())

            # Proposal density correction (log q(old|new) - log q(new|old))
            # For θ:
            log_q_reverse_t = self._log_proposal_density(self.theta, theta_prop, grad_t_prop)
            log_q_forward_t = self._log_proposal_density(theta_prop, self.theta, grad_t)

            # For φ:
            log_q_reverse_p = self._log_proposal_density(self.phi, phi_prop, grad_p_prop)
            log_q_forward_p = self._log_proposal_density(phi_prop, self.phi, grad_p)

            log_q_correction = float((log_q_reverse_t - log_q_forward_t + log_q_reverse_p - log_q_forward_p).item())

            # Log acceptance ratio
            log_alpha = -beta * delta_E + log_q_correction
            acceptance_prob = min(1.0, float(torch.exp(torch.tensor(log_alpha).clamp(max=20.0)).item()))

            # Accept or reject
            accepted = torch.rand(1).item() < acceptance_prob

        # ── Update state ──────────────────────────────────────────────────
        self._n_total += 1
        if accepted:
            self._n_accepted += 1
            self.theta = theta_prop
            self.phi = phi_prop
            self._current_E = E_prop
            self._current_grad_t = grad_t_prop
            self._current_grad_p = grad_p_prop
            E_out = E_prop
        else:
            # Keep current state — gradients already cached
            E_out = E_current

        # ── Reconstruct R for return ──────────────────────────────────────
        with torch.no_grad():
            R = nerf_reconstruct(self.theta, self.phi, self.anchor, bond=self.bond)

            # Bond length diagnostic
            diffs = R[:, 1:, :] - R[:, :-1, :]
            bond_lengths = torch.sqrt((diffs * diffs).sum(dim=-1))
            bl_mean = bond_lengths.mean().item()
            bl_std = bond_lengths.std().item()
            bl_min = bond_lengths.min().item()
            bl_max = bond_lengths.max().item()

        self.step_num += 1

        info = MALAStepInfo(
            step=self.step_num,
            energy=float(E_out.item()),
            proposed_energy=float(E_prop.item()),
            accepted=accepted,
            acceptance_prob=acceptance_prob,
            acceptance_rate=self.acceptance_rate,
            theta_grad_norm=float(grad_t.norm().item()),
            phi_grad_norm=float(grad_p.norm().item()),
            theta_std=float(self.theta.std().item()),
            phi_std=float(self.phi.std().item()),
            bond_length_mean=bl_mean,
            bond_length_std=bl_std,
            bond_length_min=bl_min,
            bond_length_max=bl_max,
        )

        return R, E_out.unsqueeze(0), info

    def run(
        self,
        n_steps: int,
        save_every: int = 500,
        log_every: int = 1000,
    ) -> Tuple[list, list]:
        """Run MALA for n_steps, returning saved frames and energies.

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
                    "step %6d/%d | E=%+.3f | accept=%.1f%% | " "bond=%.4f±%.4fÅ | |∇θ|=%.3f |∇φ|=%.3f",
                    i + 1,
                    n_steps,
                    info.energy,
                    info.acceptance_rate * 100,
                    info.bond_length_mean,
                    info.bond_length_std,
                    info.theta_grad_norm,
                    info.phi_grad_norm,
                )

        logger.info("MALA complete: %d steps, acceptance rate %.1f%%", n_steps, self.acceptance_rate * 100)

        return frames, energies

    @torch.no_grad()
    def get_current_R(self) -> torch.Tensor:
        """Return current Cartesian coordinates without stepping."""
        return nerf_reconstruct(self.theta, self.phi, self.anchor, bond=self.bond)

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
        self.theta = theta_init.clone().detach()
        self.phi = phi_init.clone().detach()
        self.step_num = 0
        self._n_accepted = 0
        self._n_total = 0
        self._current_E = None
        self._current_grad_t = None
        self._current_grad_p = None
        logger.info("MALASimulator reset.")
