"""Secondary structure basin loss.

Purpose
-------
Enforces that the secondary structure term assigns lower energy to helical
residues than to extended (beta-strand) residues, using the real backbone
angles from the training batch — no synthetic geometry.

Why this is needed for run20
----------------------------
In run19, DSM alone gave no signal about *which* clean geometry is preferred
— helix and extended are equally "clean" from DSM's perspective. The secondary
MLPs (f_theta_phi, f_phi_phi) drifted to assign low energy broadly over the
training distribution, which contains many extended/loop structures, producing
an inverted helix gap.

Approach
--------
For each training batch (R, seq):

  1. Extract real backbone angles: theta, phi = coords_to_internal(R).
     These are detached — gradients flow only through the secondary term.

  2. Convert to degrees:
       theta_deg = rad2deg(theta)      — always in (0°, 180°)
       phi_deg   = rad2deg(phi)        — standard convention, helix ≈ +50°

  3. Classify each residue as helix or extended by (theta_deg, phi_deg)
     position relative to basin peak positions, which are read at construction
     time from the actual basin energy grids. No hardcoded thresholds.

  4. Compute per-residue secondary energy:
       E_per_residue = secondary._compute_energy_components(theta, phi, seq)
       (sum of E_ram + E_theta_phi + E_phi_phi per position)

  5. Enforce gap with margin:
       loss = relu(margin - (mean E[extended] - mean E[helix]))²

  Loss is zero when helix residues have lower secondary energy than extended
  residues by at least `margin`. Active and penalising when inverted.

Basin-derived thresholds
------------------------
Helix and extended regions are defined by the peaks of the basin energy grids
loaded in SecondaryStructureEnergy. At construction time, this module reads
theta_centers and phi_centers from each basin_potential and finds the minimum-
energy (most probable) position of each basin. Basins whose peak theta is below
the median peak theta are labelled "helix" (compact geometry, lower bond angle);
basins above the median are labelled "extended" (stretched geometry, higher
bond angle). Classification margins are set to ±half the inter-basin distance
so the masks are neither too tight (few residues masked) nor too loose
(mixing helix and extended).

This ensures the classifier is permanently consistent with the basins the
secondary term was trained on.

Usage
-----
    from calphaebm.training.losses.secondary_basin_loss import (
        SecondaryBasinLoss, secondary_basin_diagnostics,
    )

    # Construct once — reads basin peaks from model.secondary
    basin_loss_fn = SecondaryBasinLoss(model.secondary, margin=0.5)

    # Each training step
    loss_basin, basin_diag = basin_loss_fn(model, R, seq)
    loss = loss + lambda_basin * loss_basin

CLI flags:
    --lambda-basin  1.0     weight on the loss (0 = disabled)
    --basin-margin  0.5     required E_extended - E_helix gap per residue
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from calphaebm.geometry.internal import bond_angles, torsions
from calphaebm.utils.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Basin peak analysis
# ---------------------------------------------------------------------------


def _find_basin_peaks(secondary: torch.nn.Module) -> list[dict]:
    """Find the peak (minimum energy = most probable) position of each basin.

    Reads theta_centers, phi_centers, energy_grid from each BasinPotential
    in secondary.basin_potentials. Returns a list of dicts, one per basin:
        {
            "theta_peak_deg": float,   # bond angle at energy minimum
            "phi_peak_deg":   float,   # torsion at energy minimum (codebase convention)
            "basin_idx":      int,
        }
    """
    peaks = []
    for k, basin in enumerate(secondary.basin_potentials):
        E = basin.energy_grid  # (n_theta, n_phi)
        tc = basin.theta_centers  # (n_theta,)  degrees
        pc = basin.phi_centers  # (n_phi,)    degrees

        # argmin of the 2D energy grid
        flat_idx = E.argmin().item()
        i_theta = flat_idx // E.shape[1]
        i_phi = flat_idx % E.shape[1]

        peaks.append(
            {
                "basin_idx": k,
                "theta_peak_deg": float(tc[i_theta].item()),
                "phi_peak_deg": float(pc[i_phi].item()),
            }
        )
        logger.info(
            "  Basin %d peak: theta=%.1f°  phi=%.1f° (codebase convention)",
            k,
            peaks[-1]["theta_peak_deg"],
            peaks[-1]["phi_peak_deg"],
        )
    return peaks


def _derive_masks(peaks: list[dict]) -> dict:
    """Derive helix and extended (theta_deg, phi_deg) classification windows.

    Strategy (v2): Use single basin surfaces, not averaged groups.
      - Helix: α-helix basin (θ≈92°, φ≈+50° in standard convention)
      - Extended: β-sheet basin (θ≈120°, φ≈-170° in standard convention)

    IMPORTANT: After fixing torsions() to standard convention, basins must be
    regenerated and basin indices re-verified. Basin numbering from GMM/K-means
    is not guaranteed to be stable across regenerations.

    Window half-width:
      - theta: half the gap between helix and sheet peak thetas, clamped [5°, 25°]
      - phi: 60° (wide enough to capture the basin, narrow enough to exclude neighbors)

    Returns dict with:
        helix_theta_center, helix_phi_center,
        extended_theta_center, extended_phi_center,
        theta_half_width, phi_half_width
    """
    # Single-surface indices — MUST be verified after each basins regeneration
    HELIX_IDX = 1
    SHEET_IDX = 0

    if len(peaks) <= max(HELIX_IDX, SHEET_IDX):
        # Fallback: old median-based grouping if not enough basins
        sorted_peaks = sorted(peaks, key=lambda p: p["theta_peak_deg"])
        helix_theta = sorted_peaks[0]["theta_peak_deg"]
        helix_phi = sorted_peaks[0]["phi_peak_deg"]
        extended_theta = sorted_peaks[-1]["theta_peak_deg"]
        extended_phi = sorted_peaks[-1]["phi_peak_deg"]
    else:
        helix_theta = peaks[HELIX_IDX]["theta_peak_deg"]
        helix_phi = peaks[HELIX_IDX]["phi_peak_deg"]
        extended_theta = peaks[SHEET_IDX]["theta_peak_deg"]
        extended_phi = peaks[SHEET_IDX]["phi_peak_deg"]

    # Window half-widths
    theta_gap = abs(extended_theta - helix_theta)
    theta_hw = float(max(5.0, min(25.0, theta_gap / 2.0)))
    phi_hw = 60.0  # wide enough for the basin

    logger.info(
        "  Helix region:    theta=%.1f±%.1f°  phi=%.1f±%.1f°  (basin %d)",
        helix_theta,
        theta_hw,
        helix_phi,
        phi_hw,
        HELIX_IDX,
    )
    logger.info(
        "  Extended region: theta=%.1f±%.1f°  phi=%.1f±%.1f°  (basin %d)",
        extended_theta,
        theta_hw,
        extended_phi,
        phi_hw,
        SHEET_IDX,
    )

    return {
        "helix_theta": helix_theta,
        "helix_phi": helix_phi,
        "extended_theta": extended_theta,
        "extended_phi": extended_phi,
        "theta_hw": theta_hw,
        "phi_hw": phi_hw,
    }


# ---------------------------------------------------------------------------
# Loss class
# ---------------------------------------------------------------------------


class SecondaryBasinLoss:
    """Secondary structure basin loss using real batch geometry.

    Constructed once per training run by reading basin peak positions from
    model.secondary. Thereafter called each training step with (model, R, seq).

    Args:
        secondary : SecondaryStructureEnergy module (model.secondary).
        margin    : Required energy gap E_extended − E_helix per residue.
                    Default 0.5. Loss is zero when gap >= margin.
    """

    def __init__(
        self,
        secondary: torch.nn.Module,
        margin: float = 0.5,
        mode: str = "continuous",
        T_base: float = 2.0,
    ):
        self.margin = float(margin)
        self.mode = mode
        self.T_base = float(T_base)

        logger.info("SecondaryBasinLoss: reading basin peak positions...")
        peaks = _find_basin_peaks(secondary)
        self._regions = _derive_masks(peaks)

        logger.info(
            "SecondaryBasinLoss ready: margin=%.2f  mode=%s  T_base=%.2f  "
            "helix_theta=%.1f±%.1f°  extended_theta=%.1f±%.1f°",
            self.margin,
            self.mode,
            self.T_base,
            self._regions["helix_theta"],
            self._regions["theta_hw"],
            self._regions["extended_theta"],
            self._regions["theta_hw"],
        )

    # ------------------------------------------------------------------

    def _classify(
        self,
        theta_deg: torch.Tensor,  # (B, N)  bond angles in degrees
        phi_deg: torch.Tensor,  # (B, N)  torsions in degrees (codebase convention)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return boolean masks (B, N) for helix and extended residues.

        A residue is helix/extended when both its theta and phi fall within
        the respective window derived from the basin peaks.
        """
        r = self._regions

        helix_mask = (
            (theta_deg >= r["helix_theta"] - r["theta_hw"])
            & (theta_deg <= r["helix_theta"] + r["theta_hw"])
            & (phi_deg >= r["helix_phi"] - r["phi_hw"])
            & (phi_deg <= r["helix_phi"] + r["phi_hw"])
        )
        extended_mask = (
            (theta_deg >= r["extended_theta"] - r["theta_hw"])
            & (theta_deg <= r["extended_theta"] + r["theta_hw"])
            & (phi_deg >= r["extended_phi"] - r["phi_hw"])
            & (phi_deg <= r["extended_phi"] + r["phi_hw"])
        )
        return helix_mask, extended_mask

    # ------------------------------------------------------------------

    def __call__(
        self,
        model: torch.nn.Module,
        R: torch.Tensor,
        seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute basin loss on the real training batch.

        Args:
            model   : TotalEnergy model with .secondary and .gate_secondary.
            R       : (B, L, 3) Cα coordinates from training batch.
            seq     : (B, L) amino acid indices.
            lengths : (B,) actual chain lengths. If provided, padding positions
                      are excluded from basin classification.

        Returns:
            (loss, diag)
              loss : scalar tensor, differentiable through model.secondary.
                     Zero when E_extended − E_helix >= margin.
              diag : dict with E_helix, E_extended, gap, n_helix, n_extended,
                     ok (bool), margin.
        """
        device = R.device
        zero = torch.zeros((), device=device, dtype=R.dtype)
        empty = {
            "E_helix": 0.0,
            "E_extended": 0.0,
            "gap": 0.0,
            "n_helix": 0,
            "n_extended": 0,
            "ok": True,
            "margin": self.margin,
            "active": False,
        }

        if not hasattr(model, "secondary") or model.secondary is None:
            return zero, empty

        # ── extract real backbone angles (detached — no grad through geometry)
        with torch.no_grad():
            theta_raw = bond_angles(R)  # (B, L-2)  radians
            phi_raw = torsions(R)  # (B, L-3)  radians

        # secondary.py alignment: phi aligns with theta[:, :L-3]
        # theta_for_phi = theta[:, :L-3],  phi_aligned = phi
        # both have shape (B, L-3) — use this N for classification
        N = phi_raw.shape[1]
        theta = theta_raw[:, :N]  # (B, N)

        # Degree conversion — standard Cα pseudo-torsion (Oldfield & Hubbard 1994)
        theta_deg = torch.rad2deg(theta)  # (B, N)  always positive
        phi_deg = torch.rad2deg(phi_raw)  # (B, N)  helix ≈ +50°

        # ── classify residues from basin-derived windows
        helix_mask, extended_mask = self._classify(theta_deg.detach(), phi_deg.detach())

        # ── exclude padding positions from classification
        if lengths is not None:
            idx = torch.arange(N, device=device)
            valid_ic = idx.unsqueeze(0) < (lengths.unsqueeze(1) - 3)  # (B, N)
            helix_mask = helix_mask & valid_ic
            extended_mask = extended_mask & valid_ic

        n_helix = int(helix_mask.sum().item())
        n_extended = int(extended_mask.sum().item())

        # Need at least a few residues of each type in the batch to be meaningful
        min_residues = 4
        if n_helix < min_residues or n_extended < min_residues:
            return zero, {
                **empty,
                "n_helix": n_helix,
                "n_extended": n_extended,
                "active": False,
                "skip_reason": "too_few_residues",
            }

        # ── per-position secondary energy (with gradients through secondary weights)
        gate = getattr(model, "gate_secondary", torch.ones(1, device=device))

        E_per_pos = _compute_per_position(model.secondary, theta_raw, phi_raw, seq)  # (B, N)
        E_per_pos = gate * E_per_pos

        # ── gap loss
        E_helix = E_per_pos[helix_mask].mean()
        E_extended = E_per_pos[extended_mask].mean()
        gap = E_extended - E_helix  # positive = helix preferred

        if self.mode == "continuous":
            loss = torch.exp(-gap / self.T_base)
        else:
            loss = F.relu(self.margin - gap) ** 2
        ok = float(gap.detach().item()) >= self.margin

        diag = {
            "E_helix": float(E_helix.detach().item()),
            "E_extended": float(E_extended.detach().item()),
            "gap": float(gap.detach().item()),
            "n_helix": n_helix,
            "n_extended": n_extended,
            "ok": ok,
            "margin": self.margin,
            "mode": self.mode,
            "active": True,
        }
        return loss, diag


# ---------------------------------------------------------------------------
# Per-position energy helper
# The internal _compute_energy_components returns (B,) sums.
# We need (B, N) per-position values to apply residue masks.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-position energy helper
# The internal _compute_energy_components returns (B,) sums.
# We need (B, N) per-position values to apply residue masks.
# We achieve this by patching normalize_by_length=False and intercepting
# the sum — but _compute_energy_components still sums over positions.
#
# Cleanest solution: call the secondary term's own basin + MLP sub-components
# directly at the position level, mirroring what _compute_energy_components
# does internally but stopping before the sum(dim=1).
# ---------------------------------------------------------------------------


def _compute_per_position(
    secondary: torch.nn.Module,
    theta: torch.Tensor,  # (B, L-2) radians
    phi: torch.Tensor,  # (B, L-3) radians
    seq: torch.Tensor,  # (B, L)
) -> torch.Tensor:
    """Compute total secondary energy per residue position: (B, N).

    N = L-3 (shortest aligned dimension — matches phi).

    Mirrors secondary._mixture_ramachandran but returns per-position
    energy before summation, so residue masks can be applied.

    With the v2 architecture (H-bond terms replace θφ/φφ MLPs), only E_ram
    is computed per-position. H-bond terms are distance-based and not
    meaningful for per-position helix/extended classification.

    Gradients flow through secondary.A, secondary.a, and lambda_ram.
    """
    from calphaebm.models.secondary import _cat

    _, L = seq.shape
    N = phi.shape[1]  # L-3

    theta = torch.nan_to_num(theta, nan=0.0)
    phi = torch.nan_to_num(phi, nan=0.0)

    e = secondary.emb(seq)  # (B, L, emb_dim)

    # Degree conversion — standard Cα pseudo-torsion (Oldfield & Hubbard 1994)
    theta_deg = torch.nan_to_num(torch.rad2deg(theta), nan=0.0)
    phi_deg = torch.nan_to_num(torch.rad2deg(phi), nan=0.0)

    theta_for_phi = theta_deg[:, :N]  # (B, N)
    phi_aligned = phi_deg  # (B, N)

    ctx = _cat(e[:, :N], e[:, 1 : N + 1], e[:, 2 : N + 2], e[:, 3 : N + 3])
    ctx = torch.nan_to_num(ctx, nan=0.0)

    # Basin Ramachandran per position (B, N)
    with torch.no_grad():
        U = (
            torch.stack(
                [U_k(theta_for_phi, phi_aligned) for U_k in secondary.basin_potentials],
                dim=-1,
            )
            .detach()
            .clamp(max=50.0)
        )
    U = torch.nan_to_num(U, nan=0.0)

    logits = (torch.einsum("kf,bnf->bnk", secondary.A, ctx) + secondary.a).clamp(-10.0, 10.0)
    E_ram_pos = secondary.ram_weight * torch.clamp(-torch.logsumexp(logits - U, dim=-1), -50.0, 50.0)

    return E_ram_pos  # (B, N)


# ---------------------------------------------------------------------------
# Diagnostics (no_grad, called from diagnostic block)
# ---------------------------------------------------------------------------


def secondary_basin_diagnostics(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    loss_fn: "SecondaryBasinLoss",
    lengths: torch.Tensor | None = None,
) -> dict:
    """Diagnostic-only evaluation — no gradients, logs E_helix/E_extended/gap.

    Args:
        model   : TotalEnergy model.
        R       : (B, L, 3) batch coordinates.
        seq     : (B, L) amino acid indices.
        loss_fn : The SecondaryBasinLoss instance constructed at training start.
        lengths : (B,) actual chain lengths.

    Returns the same diag dict as SecondaryBasinLoss.__call__ but detached.
    """
    with torch.no_grad():
        _, diag = loss_fn(model, R, seq, lengths=lengths)

    if diag.get("active", False):
        status = "OK" if diag["ok"] else "FAIL"
        logger.info(
            "  Basin:    E_helix=%.4f  E_extended=%.4f  gap=%.4f  " "margin=%.2f  n_helix=%d  n_extended=%d  %s",
            diag["E_helix"],
            diag["E_extended"],
            diag["gap"],
            diag["margin"],
            diag["n_helix"],
            diag["n_extended"],
            status,
        )
    else:
        reason = diag.get("skip_reason", "no secondary term")
        logger.info("  Basin:    skipped (%s)", reason)

    return diag
