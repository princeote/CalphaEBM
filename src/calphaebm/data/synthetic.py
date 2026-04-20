# src/calphaebm/data/synthetic.py

"""Synthetic data generators for testing and debugging."""

from __future__ import annotations

import torch

from calphaebm.utils.constants import NUM_AA


def make_extended_chain(
    batch: int,
    length: int,
    bond: float = 3.8,
    noise: float = 0.2,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate an extended Cα chain along x-axis with noise."""
    if device is None:
        device = torch.device("cpu")

    x = torch.arange(length, device=device, dtype=torch.float32) * float(bond)
    R = torch.zeros((batch, length, 3), device=device, dtype=torch.float32)
    R[..., 0] = x.unsqueeze(0).expand(batch, -1)

    if noise and noise > 0:
        R = R + float(noise) * torch.randn_like(R)

    return R


def make_helix(
    batch: int,
    length: int,
    radius: float = 2.3,
    rise: float = 1.5,
    twist: float = 100.0,
    noise: float = 0.1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate an ideal alpha helix (approx) in Cα space."""
    if device is None:
        device = torch.device("cpu")

    twist_rad = torch.deg2rad(torch.tensor(float(twist), device=device, dtype=torch.float32))

    i = torch.arange(length, device=device, dtype=torch.float32)
    x = float(radius) * torch.cos(i * twist_rad)
    y = float(radius) * torch.sin(i * twist_rad)
    z = i * float(rise)

    R = torch.stack([x, y, z], dim=-1)  # (L, 3)
    R = R.unsqueeze(0).expand(batch, -1, -1)  # (B, L, 3)

    if noise and noise > 0:
        R = R + float(noise) * torch.randn_like(R)

    return R


def random_sequence(
    batch: int,
    length: int,
    num_aa: int = NUM_AA,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate random amino acid sequences."""
    if device is None:
        device = torch.device("cpu")

    return torch.randint(
        low=0,
        high=int(num_aa),
        size=(batch, length),
        device=device,
        dtype=torch.long,
    )


def random_protein_like(
    batch: int,
    length: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random protein-like coordinates and sequence."""
    R = make_extended_chain(batch, length, bond=3.8, noise=0.2, device=device)
    seq = random_sequence(batch, length, device=device)
    return R, seq
