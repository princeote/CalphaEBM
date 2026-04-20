# src/calphaebm/simulation/observers.py

"""Observers for collecting data during simulation."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from calphaebm.utils.math import safe_norm
from calphaebm.utils.neighbors import pairwise_distances


class Observer(ABC):
    """Base class for simulation observers."""

    @abstractmethod
    def update(self, step: int, R: torch.Tensor, **kwargs):
        """Update observer with current state."""

    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """Return collected data."""

    def reset(self):
        """Reset observer state."""


class EnergyObserver(Observer):
    """Observer for tracking energies."""

    def __init__(self, model, log_every: int = 1):
        self.model = model
        self.log_every = log_every
        self.steps = []
        self.energies = []
        self.term_energies = {k: [] for k in ["local", "repulsion", "secondary", "packing"]}

    def update(self, step: int, R: torch.Tensor, seq: torch.Tensor, **kwargs):
        if step % self.log_every == 0:
            with torch.no_grad():
                if hasattr(self.model, "term_energies"):
                    terms = self.model.term_energies(R, seq)
                    total = sum(terms.values())

                    self.steps.append(step)
                    self.energies.append(total.item())

                    for k, v in terms.items():
                        if k in self.term_energies:
                            self.term_energies[k].append(v.item())

    def get_results(self) -> Dict[str, Any]:
        return {
            "energy_steps": self.steps,
            "total_energy": self.energies,
            "term_energies": self.term_energies,
        }

    def reset(self):
        self.steps = []
        self.energies = []
        self.term_energies = {k: [] for k in self.term_energies}


class MinDistanceObserver(Observer):
    """Observer for tracking minimum nonbonded distances."""

    def __init__(self, exclude: int = 2, log_every: int = 1):
        self.exclude = exclude
        self.log_every = log_every
        self.steps = []
        self.min_distances = []
        self.median_distances = []

    def _min_nonbonded(self, R: torch.Tensor) -> tuple:
        """Compute min and median nonbonded distances.

        FIX: replaced roll()-based diagonal masking (which wraps around and
        incorrectly masks distant pairs in short sequences) with an explicit
        sequence-separation mask using index arithmetic.
        """
        B, L, _ = R.shape
        D = pairwise_distances(R)  # (B, L, L)

        # Build correct exclusion mask: True where |i-j| > exclude
        idx = torch.arange(L, device=R.device)
        sep = (idx[:, None] - idx[None, :]).abs()  # (L, L)
        allowed = sep > self.exclude  # (L, L) — no wrap-around
        # Upper triangle only (count each pair once)
        triu = torch.triu(torch.ones(L, L, dtype=torch.bool, device=R.device), diagonal=1)
        allowed = allowed & triu  # (L, L)

        min_vals = []
        med_vals = []
        for b in range(B):
            vals = D[b][allowed]
            if vals.numel() > 0:
                min_vals.append(vals.min().item())
                med_vals.append(vals.median().item())
            else:
                min_vals.append(float("inf"))
                med_vals.append(float("inf"))

        return (
            torch.tensor(min_vals, device=R.device),
            torch.tensor(med_vals, device=R.device),
        )

    def update(self, step: int, R: torch.Tensor, **kwargs):
        if step % self.log_every == 0:
            with torch.no_grad():
                min_vals, med_vals = self._min_nonbonded(R)

            self.steps.append(step)
            self.min_distances.append(min_vals.min().item())
            self.median_distances.append(med_vals.median().item())

    def get_results(self) -> Dict[str, Any]:
        return {
            "min_dist_steps": self.steps,
            "min_distance": self.min_distances,
            "median_distance": self.median_distances,
        }

    def reset(self):
        self.steps = []
        self.min_distances = []
        self.median_distances = []


class ClippingObserver(Observer):
    """Observer for tracking force clipping statistics."""

    def __init__(self, log_every: int = 1):
        self.log_every = log_every
        self.steps = []
        self.clip_fractions = []
        self.max_forces = []

    def update(self, step: int, R: torch.Tensor, force_cap: float | None = None, **kwargs):
        if step % self.log_every == 0 and "forces" in kwargs:
            forces = kwargs["forces"]
            force_norms = safe_norm(forces, dim=-1)

            max_force = force_norms.max().item()
            clip_frac = (force_norms > force_cap).float().mean().item() if force_cap else 0.0

            self.steps.append(step)
            self.max_forces.append(max_force)
            self.clip_fractions.append(clip_frac)

    def get_results(self) -> Dict[str, Any]:
        return {
            "clip_steps": self.steps,
            "max_force": self.max_forces,
            "clip_fraction": self.clip_fractions,
        }

    def reset(self):
        self.steps = []
        self.max_forces = []
        self.clip_fractions = []


class TrajectoryObserver(Observer):
    """Observer for saving trajectory frames."""

    def __init__(self, save_every: int = 50):
        self.save_every = save_every
        self.frames = []
        self.steps = []

    def update(self, step: int, R: torch.Tensor, **kwargs):
        if step % self.save_every == 0:
            self.frames.append(R.detach().clone())
            self.steps.append(step)

    def get_results(self) -> Dict[str, Any]:
        return {
            "trajectory_steps": self.steps,
            "trajectory": self.frames,
        }

    def reset(self):
        self.frames = []
        self.steps = []
