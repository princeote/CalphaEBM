# src/calphaebm/simulation/backends/__init__.py
"""Simulation backends — Langevin and MALA integrators."""

from __future__ import annotations

from typing import List, Optional

import torch

# IC integrators
from calphaebm.simulation.backends.langevin import ICLangevinSimulator
from calphaebm.simulation.backends.langevin_mala import MALASimulator

# Cartesian integrator (kept for DSM training and legacy comparison)
try:
    from calphaebm.simulation.backends.pytorch import PyTorchSimulator

    _pytorch_available = True
except ImportError:
    _pytorch_available = False


def get_simulator(
    name: str,
    model,
    seq: torch.Tensor,
    R_init: torch.Tensor,
    step_size: float = 1e-4,
    beta: float = 1.0,
    force_cap: float = 100.0,
    bond: float = 3.8,
    lengths: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
):
    """Factory: return the requested simulator with a unified API.

    Both simulators have identical .step() signature:
        R, E, info = sim.step()
    where R is (1,L,3), E is (1,), info has .energy attribute.

    Args:
        name:  "langevin" or "mala"
        (remaining args forwarded to the simulator constructor)

    Returns:
        Simulator instance with .step() -> (R, E, info)
    """
    name = name.lower().strip()
    if name == "langevin":
        return ICLangevinSimulator(
            model=model,
            seq=seq,
            R_init=R_init,
            step_size=step_size,
            beta=beta,
            force_cap=force_cap,
            bond=bond,
            lengths=lengths,
            device=device,
        )
    elif name == "mala":
        return MALASimulator(
            model=model,
            seq=seq,
            R_init=R_init,
            step_size=step_size,
            beta=beta,
            force_cap=force_cap,
            bond=bond,
            lengths=lengths,
            device=device,
        )
    else:
        raise ValueError(f"Unknown sampler: {name!r}. Use 'langevin' or 'mala'.")


def langevin_sample(
    model,
    R0: torch.Tensor,
    seq: torch.Tensor,
    n_steps: int = 1000,
    step_size: float = 2e-5,
    force_cap: float = 50.0,
    beta: float = 1.0,
    log_every: int = 1000,
    save_every: Optional[int] = None,
    sampler: str = "langevin",
) -> List[torch.Tensor]:
    """Thin compatibility shim — runs the requested integrator and returns
    a list of coordinate snapshots, matching the old pytorch.langevin_sample API.

    Args:
        sampler:  "langevin" (default) or "mala"

    Returns:
        List of R tensors (one per save_every interval, plus the final frame).
        The last element is the final configuration.
    """
    if save_every is None:
        save_every = max(1, n_steps // 10)

    sim = get_simulator(
        name=sampler,
        model=model,
        seq=seq,
        R_init=R0,
        step_size=step_size,
        beta=beta,
        force_cap=force_cap,
    )

    snapshots: List[torch.Tensor] = []
    for i in range(n_steps):
        R, _E, _info = sim.step()
        if (i + 1) % save_every == 0:
            snapshots.append(R.detach().clone())

    # Ensure the final frame is always present
    if not snapshots or not torch.allclose(snapshots[-1], R.detach()):
        snapshots.append(R.detach().clone())

    return snapshots


__all__ = [
    "ICLangevinSimulator",
    "MALASimulator",
    "get_simulator",
    "langevin_sample",
]

if _pytorch_available:
    __all__.append("PyTorchSimulator")
