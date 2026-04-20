"""Ramachandran validity gate — shared by packing, local, and secondary.

src/calphaebm/models/rama_gate.py

  v_i = max_k exp(-(θ_i - θ_k*)² / (2σ_θ²) - wrap(φ_i - φ_k*)² / (2σ_φ²))

Answers "is this residue in ANY allowed backbone region?"
  Native ≈ 0.85, forbidden ≈ 0.02.

No hardcoded defaults.  Peaks always extracted from basin energy surfaces.
Construct via:
  RamaValidityGate.from_data_dir("analysis/secondary_analysis/data")
  RamaValidityGate.from_basin_surfaces(basin_potentials_module_list)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from calphaebm.models.learnable_buffers import reg
from calphaebm.utils.logging import get_logger

logger = get_logger()

_SIGMA_THETA = 0.35  # rad (~20°)
_SIGMA_PHI = 0.70  # rad (~40°)


class RamaValidityGate(nn.Module):
    """Per-residue Ramachandran validity gate.

    No trainable parameters.  No defaults — peaks must come from data.
    """

    def __init__(
        self,
        basin_theta_rad: list[float],
        basin_phi_rad: list[float],
        sigma_theta: float = _SIGMA_THETA,
        sigma_phi: float = _SIGMA_PHI,
        learn_geometry: bool = False,
    ):
        super().__init__()
        reg(self, "_gate_basin_theta", torch.tensor(basin_theta_rad, dtype=torch.float32), learnable=learn_geometry)
        reg(self, "_gate_basin_phi", torch.tensor(basin_phi_rad, dtype=torch.float32), learnable=learn_geometry)
        reg(
            self,
            "_gate_inv_2sig2_theta",
            torch.tensor(1.0 / (2.0 * sigma_theta**2), dtype=torch.float32),
            learnable=learn_geometry,
        )
        reg(
            self,
            "_gate_inv_2sig2_phi",
            torch.tensor(1.0 / (2.0 * sigma_phi**2), dtype=torch.float32),
            learnable=learn_geometry,
        )
        if learn_geometry:
            logger.info("  RamaValidityGate: geometry LEARNABLE (10 params)")

    @classmethod
    def from_data_dir(
        cls, data_dir: str, smooth_sigma: float = 2.0, pseudocount: float = 1e-6, learn_geometry: bool = False, **kwargs
    ) -> "RamaValidityGate":
        """Load basin energy surfaces from disk and extract peaks.

        Applies the same Gaussian smoothing as BasinPotential so that peaks
        found here match peaks found by from_basin_surfaces().
        """
        from pathlib import Path

        d = Path(data_dir)
        basin_paths = sorted(d.glob("basin_*_energy.npy"))
        if len(basin_paths) < 4:
            raise FileNotFoundError(
                f"Need 4 basin_*_energy.npy in {d}, found {len(basin_paths)}. " f"Run: calphaebm analyze basins"
            )

        # Resolve edge files
        theta_ep = d / "theta_edges_deg.npy"
        phi_ep = d / "phi_edges_deg.npy"
        if not theta_ep.exists():
            theta_ep = d / "figure_3a_xedges.npy"
        if not phi_ep.exists():
            phi_ep = d / "figure_3a_yedges.npy"

        theta_edges = np.load(theta_ep).astype(np.float32).reshape(-1)
        phi_edges = np.load(phi_ep).astype(np.float32).reshape(-1)
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

        # Apply same smoothing as BasinPotential — peaks must match.
        try:
            from scipy.ndimage import gaussian_filter

            have_scipy = True
        except ImportError:
            have_scipy = False
            logger.warning("scipy not available — peaks may differ from BasinPotential")

        peaks_theta = []
        peaks_phi = []
        labels = {0: "SHEET", 1: "HELIX", 2: "PPII", 3: "TURN"}
        for k in range(4):
            raw = np.load(basin_paths[k]).astype(np.float32)

            # Detect histogram vs energy (same logic as BasinPotential)
            is_prob_like = float(np.nanmax(raw)) <= 1.5 and float(np.nanmin(raw)) >= 0.0
            if is_prob_like:
                if have_scipy:
                    smoothed = gaussian_filter(
                        raw, sigma=(smooth_sigma, smooth_sigma), mode=("nearest", "wrap")
                    ).astype(np.float32)
                else:
                    smoothed = raw
                S = -np.log(smoothed + float(pseudocount))
                S = S - float(np.nanmin(S))
            else:
                # Already energy — smooth directly
                if have_scipy:
                    S = gaussian_filter(raw, sigma=(smooth_sigma, smooth_sigma), mode=("nearest", "wrap")).astype(
                        np.float32
                    )
                else:
                    S = raw

            S = np.nan_to_num(S, nan=0.0, posinf=50.0, neginf=0.0)
            it, ip = np.unravel_index(S.argmin(), S.shape)
            t_deg = float(theta_centers[it])
            p_deg = float(phi_centers[ip])
            peaks_theta.append(float(np.radians(t_deg)))
            peaks_phi.append(float(np.radians(p_deg)))
            logger.info(
                "  RamaValidityGate basin %d (%s): theta=%.1f deg, phi=%.1f deg", k, labels.get(k, "?"), t_deg, p_deg
            )

        return cls(basin_theta_rad=peaks_theta, basin_phi_rad=peaks_phi, learn_geometry=learn_geometry, **kwargs)

    @classmethod
    def from_basin_surfaces(cls, basin_potentials: nn.ModuleList, **kwargs) -> "RamaValidityGate":
        """Extract peaks from already-loaded BasinPotential modules."""
        peaks_theta = []
        peaks_phi = []
        labels = {0: "SHEET", 1: "HELIX", 2: "PPII", 3: "TURN"}
        for k, bp in enumerate(basin_potentials):
            grid = bp.energy_grid
            min_idx = grid.argmin()
            it = min_idx // bp.n_phi
            ip = min_idx % bp.n_phi
            t_rad = float(np.radians(float(bp.theta_centers[it])))
            p_rad = float(np.radians(float(bp.phi_centers[ip])))
            peaks_theta.append(t_rad)
            peaks_phi.append(p_rad)
            logger.info(
                "  RamaValidityGate basin %d (%s): theta=%.1f deg, phi=%.1f deg",
                k,
                labels.get(k, "?"),
                np.degrees(t_rad),
                np.degrees(p_rad),
            )
        return cls(basin_theta_rad=peaks_theta, basin_phi_rad=peaks_phi, **kwargs)

    def forward(self, theta_rad: torch.Tensor, phi_rad: torch.Tensor) -> torch.Tensor:
        """Per-position validity.

        Args:
            theta_rad: (B, N) bond angles in radians
            phi_rad:   (B, N) dihedrals in radians (same N)
        Returns:
            (B, N) validity in [0, 1], detached.
        """
        dth = theta_rad.unsqueeze(-1) - self._gate_basin_theta  # (B, N, K)
        dph_raw = phi_rad.unsqueeze(-1) - self._gate_basin_phi  # (B, N, K)
        dph = dph_raw - 2.0 * 3.14159 * torch.round(dph_raw / (2.0 * 3.14159))

        d2 = dth**2 * self._gate_inv_2sig2_theta + dph**2 * self._gate_inv_2sig2_phi  # (B, N, K)
        v = d2.neg().exp().max(dim=-1).values  # (B, N)

        return v.detach()
