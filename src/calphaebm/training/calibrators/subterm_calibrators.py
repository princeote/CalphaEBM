"""Sub-term scale calibration for energy model initialization.

Purpose
-------
Each energy sub-term has a different raw output magnitude by construction.
Without calibration, learnable weights must compensate for these raw magnitude
differences rather than learning physics. This module measures each sub-term's
mean raw output under the training perturbation distribution and computes init
weights such that every sub-term contributes `target` energy per residue.

Architecture (run31+):
  local     (3): theta_theta MLP*, delta_phi table, phi_phi MLP*
  secondary (3): ram basins, hb_alpha Gaussian, hb_beta 2×Gaussian
  repulsion (1): tabulated wall
  packing   (2): geom MLP, contact h·h·g(r)

  * MLP terms at random init output ≈ 0 → skip calibration (use --calibrate-mlp-terms
    only after training when MLPs produce meaningful output)

Usage
-----
    calphaebm calibrate \\
        --pdb train_entities.no_test_entries.txt \\
        --backbone-data-dir analysis/backbone_geometry/data \\
        --secondary-data-dir analysis/secondary_analysis/data \\
        --repulsion-data-dir analysis/repulsion_analysis/data \\
        --apply-to-ckpt checkpoints/run28/run1/full/step008000.pt \\
        --out-ckpt checkpoints/run31_calibrated.pt
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from calphaebm.geometry.internal import bond_angles, torsions
from calphaebm.utils.logging import get_logger

logger = get_logger()


# ── perturbation ──────────────────────────────────────────────────────────────


def _perturb(R: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Perturb in IC space (radians), matching DSM training perturbation.

    sigma ~ LogUniform(sigma_min, sigma_max) in RADIANS.
    Perturbs bond angles (θ) and torsions (φ), then reconstructs via NeRF.
    This is the same perturbation DSM training uses, so calibrated force
    scales match what the model actually sees during training.

    Args:
        R: (B, L, 3) clean Cα coordinates.
        sigma_min: lower bound in radians (e.g. 0.05 rad ≈ 3°).
        sigma_max: upper bound in radians (e.g. 2.0 rad ≈ 115°).

    Returns:
        (B, L, 3) perturbed coordinates.
    """
    from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct

    B = R.shape[0]
    log_sigmas = torch.empty(B).uniform_(math.log(sigma_min), math.log(sigma_max))
    sigmas = torch.exp(log_sigmas)  # (B,)

    theta, phi = coords_to_internal(R)  # (B, L-2), (B, L-3) in radians
    anchor = extract_anchor(R)  # (B, 3, 3)

    # Perturb angles and torsions with per-sample sigma
    theta_pert = theta + torch.randn_like(theta) * sigmas[:, None]
    phi_pert = phi + torch.randn_like(phi) * sigmas[:, None]

    # Clamp theta to valid range [0.3, π-0.3] to avoid degenerate geometry
    theta_pert = theta_pert.clamp(0.3, math.pi - 0.3)

    # Reconstruct
    R_pert = nerf_reconstruct(theta_pert, phi_pert, anchor)
    return R_pert


# ── raw sub-term outputs ──────────────────────────────────────────────────────
# Each helper temporarily sets the relevant weight to 1.0 and calls the model
# method, then restores the original value.

_INV_SP_1 = math.log(math.exp(1.0) - 1.0)  # softplus(x) = 1.0


@torch.no_grad()
def _raw_theta_theta(R: torch.Tensor, seq: torch.Tensor, local: torch.nn.Module) -> torch.Tensor:
    """Theta-theta energy with weight=1. Handles both v1 (quadratic) and v2 (MLP)."""
    # Find the weight parameter: v2 uses _theta_theta_mlp_w, v1 uses _theta_theta_weight_raw
    param = getattr(local, "_theta_theta_mlp_w", getattr(local, "_theta_theta_weight_raw", None))
    if param is None:
        return torch.zeros(1, device=R.device)
    orig = param.data.clone()
    try:
        param.data.fill_(_INV_SP_1)
        # v2 accepts seq, v1 does not
        try:
            return local.theta_theta_energy(R, seq)
        except TypeError:
            return local.theta_theta_energy(R)
    finally:
        param.data.copy_(orig)


@torch.no_grad()
def _raw_delta_phi(R: torch.Tensor, local: torch.nn.Module) -> torch.Tensor:
    """Delta-phi energy with delta_phi_weight=1."""
    orig = local._delta_phi_weight_raw.data.clone()
    try:
        local._delta_phi_weight_raw.data.fill_(_INV_SP_1)
        return local.delta_phi_energy(R)
    finally:
        local._delta_phi_weight_raw.data.copy_(orig)


@torch.no_grad()
def _raw_phi_phi(R: torch.Tensor, seq: torch.Tensor, local: torch.nn.Module) -> torch.Tensor:
    """Phi-phi MLP energy with weight=1. Returns zeros if not present (v1)."""
    param = getattr(local, "_phi_phi_mlp_w", getattr(local, "_phi_phi_weight_raw", None))
    if param is None or not hasattr(local, "phi_phi_energy"):
        return torch.zeros(1, device=R.device)
    orig = param.data.clone()
    try:
        param.data.fill_(_INV_SP_1)
        try:
            return local.phi_phi_energy(R, seq)
        except TypeError:
            return local.phi_phi_energy(R)
    finally:
        param.data.copy_(orig)


@torch.no_grad()
def _raw_theta_phi(R: torch.Tensor, seq: torch.Tensor, local: torch.nn.Module) -> torch.Tensor:
    """4-mer theta-phi energy with weight=1."""
    param = getattr(local, "_lambda_raw", None)
    if param is None or not hasattr(local, "theta_phi_energy"):
        return torch.zeros(1, device=R.device)
    orig = param.data.clone()
    try:
        param.data.fill_(_INV_SP_1)
        return local.theta_phi_energy(R, seq)
    finally:
        param.data.copy_(orig)


@torch.no_grad()
def _raw_repulsion(R: torch.Tensor, repulsion: torch.nn.Module) -> torch.Tensor:
    """Repulsion energy with lambda_rep=1, measured on perturbed structures."""
    params = dict(repulsion.named_parameters())
    key = next((k for k in ["_lambda_rep_raw", "lambda_rep_raw"] if k in params), None)
    if key is None:
        raise AttributeError(f"Cannot find lambda_rep_raw. Available: {list(params.keys())}")
    orig = params[key].data.clone()
    try:
        params[key].data.fill_(_INV_SP_1)
        seq_dummy = torch.zeros(R.shape[0], R.shape[1], dtype=torch.long)
        return repulsion(R, seq_dummy)
    finally:
        params[key].data.copy_(orig)


@torch.no_grad()
def _raw_secondary_components(
    R: torch.Tensor,
    seq: torch.Tensor,
    secondary: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-residue mean of each secondary sub-term at weight = 1.

    v2 architecture: returns (E_ram, |E_hb_alpha|, |E_hb_beta|)
    Falls back to v1 (theta_phi, phi_phi MLPs) if subterm_energies fails.

    Uses abs() on H-bond terms because they are negative (attractive).
    """
    # Temporarily set ram_weight to 1.0
    param_names = dict(secondary.named_parameters())

    def _find_param(candidates):
        for name in candidates:
            if name in param_names:
                return name
        for cand in candidates:
            key = cand.split(".")[-1]
            for name in param_names:
                if name.endswith(key):
                    return name
        return None

    ram_key = _find_param(["_lambda_ram_raw", "_ram_weight_raw", "lambda_ram_raw", "ram_weight_raw", "lambda_ram"])

    def _get(key):
        return param_names[key] if key else None

    # Save and set ram weight
    orig_ram = None
    if ram_key:
        orig_ram = _get(ram_key).data.clone()
        _get(ram_key).data.fill_(_INV_SP_1)

    try:
        # Try v2 interface: subterm_energies returns (E_ram, E_hb_alpha, E_hb_beta)
        E_ram, E_hb_a, E_hb_b = secondary.subterm_energies(R, seq)
        return E_ram, E_hb_a.abs(), E_hb_b.abs()
    except Exception:
        # Fall back to v1 interface
        theta = bond_angles(R)
        phi = torsions(R)
        try:
            E_ram, E_tp, E_pp = secondary._compute_energy_components(theta, phi, seq)
            return E_ram, E_tp.abs(), E_pp.abs()
        except Exception:
            logger.warning("Could not measure secondary components")
            B = R.shape[0]
            z = torch.zeros(B, device=R.device)
            return z, z, z
    finally:
        if ram_key and orig_ram is not None:
            _get(ram_key).data.copy_(orig_ram)


@torch.no_grad()
def _raw_packing_components(
    R: torch.Tensor,
    seq: torch.Tensor,
    packing: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packing energy split into (E_geom, E_contact) at lambda_pack=1.

    Measured on NATIVE (unperturbed) structures because contact energy
    collapses on perturbed structures.
    """
    params = dict(packing.named_parameters())
    key = next((k for k in ["_lambda_pack_raw", "lambda_pack_raw"] if k in params), None)
    if key is None:
        raise AttributeError(f"Cannot find lambda_pack_raw. Available: {list(params.keys())}")
    orig = params[key].data.clone()
    try:
        params[key].data.fill_(_INV_SP_1)
        # Try split interface
        try:
            E_geom, E_hp = packing.subterm_energies(R, seq)
            return E_geom, E_hp
        except Exception:
            # Fallback: combined
            E = packing(R, seq)
            return E, torch.zeros_like(E)
    finally:
        params[key].data.copy_(orig)


# ── results container ─────────────────────────────────────────────────────────


@dataclass
class CalibrationResults:
    """Results from SubtermScaleCalibrator.run().

    Supports both old (9 subterms) and 4-mer (7 subterms) architectures.
    """

    # Recommended init weights (target / mean_raw)
    # 4-mer architecture
    init_theta_phi_weight: Optional[float] = None  # None = old architecture
    # Old 3-subterm architecture
    init_theta_theta_weight: float = 1.0
    init_delta_phi_weight: float = 1.0
    init_phi_phi_weight: Optional[float] = None  # None = MLP at init, skip
    init_ram_weight: float = 1.0
    init_hb_alpha_weight: Optional[float] = None  # None if not measured
    init_hb_beta_weight: Optional[float] = None  # None if not measured
    init_rep_weight: float = 1.0
    init_pack_geom_weight: float = 1.0  # from native measurement
    init_pack_contact_weight: float = 1.0  # from native measurement

    # Raw mean outputs (weight=1) for inspection
    means: Dict[str, Optional[float]] = field(default_factory=dict)

    # Actual lambda values read from the model before calibration
    current_lambdas: Dict[str, Optional[float]] = field(default_factory=dict)

    # Run metadata
    target: float = 0.1111
    sigma_min: float = 0.05
    sigma_max: float = 2.0
    n_samples: int = 0

    def log_summary(self) -> None:
        is_4mer = self.init_theta_phi_weight is not None
        n_subterms = 7 if is_4mer else 9
        logger.info("=" * 72)
        logger.info("SUB-TERM SCALE CALIBRATION (%d subterms)", n_subterms)
        logger.info("=" * 72)
        logger.info(
            "  sigma ~ LogUniform(%.3f, %.3f)  |  target=%.4f/subterm  |  n=%d",
            self.sigma_min,
            self.sigma_max,
            self.target,
            self.n_samples,
        )
        logger.info("")
        logger.info("  %-18s  %12s  %12s  %12s  %s", "sub-term", "current_λ", "raw_out(λ=1)", "new_λ", "notes")
        logger.info("  " + "-" * 72)

        if is_4mer:
            rows = [
                ("theta_phi", "theta_phi", "theta_phi", self.init_theta_phi_weight, "4-mer"),
            ]
        else:
            rows = [
                (
                    "theta_theta",
                    "theta_theta",
                    "theta_theta",
                    self.init_theta_theta_weight,
                    "" if self.init_theta_theta_weight != 1.0 else "MLP ~0, skipped",
                ),
                ("delta_phi", "delta_phi", "delta_phi", self.init_delta_phi_weight, ""),
                (
                    "phi_phi",
                    "phi_phi",
                    "phi_phi",
                    self.init_phi_phi_weight,
                    "" if self.init_phi_phi_weight is not None else "MLP ~0, skipped",
                ),
            ]
        rows += [
            ("ram", "ram", "ram", self.init_ram_weight, ""),
            (
                "hb_alpha",
                "hb_alpha",
                "hb_alpha",
                self.init_hb_alpha_weight,
                "" if self.init_hb_alpha_weight is not None else "not measured",
            ),
            (
                "hb_beta",
                "hb_beta",
                "hb_beta",
                self.init_hb_beta_weight,
                "" if self.init_hb_beta_weight is not None else "not measured",
            ),
            ("rep", "rep", "rep", self.init_rep_weight, "PRESERVED — safety constraint"),
            ("pack_geom", "pack_geom", "pack_geom", self.init_pack_geom_weight, "native"),
            ("pack_contact", "pack_contact", "pack_contact", self.init_pack_contact_weight, "native"),
        ]
        for label, lam_key, mean_key, new_lam, note in rows:
            cur = self.current_lambdas.get(lam_key)
            raw = self.means.get(mean_key)
            cur_s = f"{cur:.4f}" if cur is not None else "  n/a  "
            raw_s = f"{raw:.4f}" if raw is not None else "  n/a  "
            new_s = f"{new_lam:.4f}" if new_lam is not None else "preserved"
            logger.info("  %-18s  %12s  %12s  %12s  %s", label, cur_s, raw_s, new_s, note)

        logger.info("=" * 72)

    def apply_to_model(self, model: torch.nn.Module) -> None:
        """Write calibrated init weights directly into model parameters."""

        def _set_softplus_param(param: torch.nn.Parameter, target_val: float, eps: float = 1e-6) -> None:
            y = max(target_val - eps, 1e-8)
            raw = y if y > 20.0 else math.log(math.exp(y) - 1.0)
            param.data.fill_(raw)

        def _find_and_set(module, candidates, target_val, label):
            params = dict(module.named_parameters())
            for name in candidates:
                if name in params:
                    _set_softplus_param(params[name], target_val)
                    logger.info("  Set %s (via %s) = %.4f", label, name, target_val)
                    return True
            logger.warning("  Could not find parameter for %s (tried %s)", label, candidates)
            return False

        local = model.local

        if self.init_theta_phi_weight is not None:
            # 4-mer architecture: single lambda
            _find_and_set(local, ["_lambda_raw"], self.init_theta_phi_weight, "theta_phi_weight")
        else:
            # Old 3-subterm architecture
            # θθ weight
            _find_and_set(
                local,
                ["_theta_theta_mlp_w", "_theta_theta_weight_raw"],
                self.init_theta_theta_weight,
                "theta_theta_weight",
            )

            # Δφ weight
            _find_and_set(local, ["_delta_phi_weight_raw"], self.init_delta_phi_weight, "delta_phi_weight")

            # φφ weight (if calibrated)
            if self.init_phi_phi_weight is not None:
                _find_and_set(
                    local, ["_phi_phi_mlp_w", "_phi_phi_weight_raw"], self.init_phi_phi_weight, "phi_phi_weight"
                )
            else:
                logger.info("  Skipping phi_phi_weight — MLP at init, not calibrated")

        # Secondary
        if model.secondary is not None:
            ss = model.secondary
            _find_and_set(
                ss,
                ["_lambda_ram_raw", "_ram_weight_raw", "lambda_ram_raw", "lambda_ram"],
                self.init_ram_weight,
                "ram_weight",
            )

            # H-bond lambdas live inside hb_helix/hb_sheet modules
            if self.init_hb_alpha_weight is not None and hasattr(ss, "hb_helix"):
                _find_and_set(ss.hb_helix, ["_lambda_raw"], self.init_hb_alpha_weight, "hb_alpha_lambda")

            if self.init_hb_beta_weight is not None and hasattr(ss, "hb_sheet"):
                _find_and_set(ss.hb_sheet, ["_lambda_raw"], self.init_hb_beta_weight, "hb_beta_lambda")

        # Repulsion is a SAFETY constraint, not a physics energy contributor.
        # It must stay strong enough to prevent atom clashes that cause Hessian
        # blow-ups in other terms. NEVER reduce it below checkpoint value.
        # Skip calibration — preserve from checkpoint.
        if model.repulsion is not None:
            logger.info("  Skipping lambda_rep — safety constraint, preserved from checkpoint")

        # Packing — lambda_pack only scales E_geom. E_contact has its own
        # internal lambda inside _HydrophobicPairs.
        if model.packing is not None:
            _find_and_set(
                model.packing,
                ["_lambda_pack_raw", "lambda_pack_raw"],
                self.init_pack_geom_weight,
                "lambda_pack (from geom measurement)",
            )

            # Contact HP lambda is inside the burial sub-module
            if hasattr(model.packing, "burial"):
                _find_and_set(
                    model.packing.burial,
                    ["_lambda_hp_raw", "_lambda_raw"],
                    self.init_pack_contact_weight,
                    "lambda_hp (contact)",
                )
            else:
                logger.info("  No burial sub-module — skipping contact lambda")

        logger.info("Applied calibrated init weights to model parameters.")

    def to_dict(self) -> dict:
        return {
            "init_theta_phi_weight": self.init_theta_phi_weight,
            "init_theta_theta_weight": self.init_theta_theta_weight,
            "init_delta_phi_weight": self.init_delta_phi_weight,
            "init_phi_phi_weight": self.init_phi_phi_weight,
            "init_ram_weight": self.init_ram_weight,
            "init_hb_alpha_weight": self.init_hb_alpha_weight,
            "init_hb_beta_weight": self.init_hb_beta_weight,
            "init_rep_weight": self.init_rep_weight,
            "init_pack_geom_weight": self.init_pack_geom_weight,
            "init_pack_contact_weight": self.init_pack_contact_weight,
            "means": self.means,
            "current_lambdas": self.current_lambdas,
            "target": self.target,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "n_samples": self.n_samples,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved calibration results to %s", path)

    @classmethod
    def load(cls, path: str) -> "CalibrationResults":
        with open(path) as f:
            d = json.load(f)
        # Back-compat defaults
        d.setdefault("init_theta_phi_weight", None)
        d.setdefault("init_phi_phi_weight", None)
        d.setdefault("init_hb_alpha_weight", None)
        d.setdefault("init_hb_beta_weight", None)
        d.setdefault("init_pack_geom_weight", d.pop("init_pack_weight", 1.0))
        d.setdefault("init_pack_contact_weight", 1.0)
        d.setdefault("current_lambdas", {})
        # Remove old keys
        d.pop("init_bond_spring", None)
        return cls(**d)


# ── calibrator ────────────────────────────────────────────────────────────────


class SubtermScaleCalibrator:
    """Calibrate energy sub-term init weights from PDB structures.

    9 subterms calibrated to the same per-subterm target:
        local     (3): theta_theta*, delta_phi, phi_phi*
        secondary (3): ram, hb_alpha, hb_beta
        repulsion (1): measured on PERTURBED structures
        packing   (2): geom, contact — measured on NATIVE structures

    * MLP terms (theta_theta, phi_phi) at random init output ≈ 0.
      Skip with default; use --calibrate-mlp-terms after training.
    """

    def __init__(
        self,
        sigma_min: float = 0.05,
        sigma_max: float = 8.0,
        target: float = 1.0,
        calibrate_mlp: bool = False,
    ):
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be > 0, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(f"sigma_max must be > sigma_min, got {sigma_max} <= {sigma_min}")
        if target <= 0:
            raise ValueError(f"target must be > 0, got {target}")

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.target = float(target)
        self.calibrate_mlp = bool(calibrate_mlp)

        mlp_note = " (MLP sub-terms will be calibrated)" if calibrate_mlp else ""
        logger.info(
            "SubtermScaleCalibrator: sigma~LogUniform(%.3f, %.3f), " "target=%.4f/subterm%s",
            self.sigma_min,
            self.sigma_max,
            self.target,
            mlp_note,
        )

    @staticmethod
    def _read_current_lambdas(model: torch.nn.Module) -> Dict[str, Optional[float]]:
        """Read the actual lambda values currently in the model."""
        import torch.nn.functional as _F

        lams: Dict[str, Optional[float]] = {}
        local = getattr(model, "local", None)
        if local is not None:
            # 4-mer architecture
            if hasattr(local, "theta_phi_weight"):
                lams["theta_phi"] = float(local.theta_phi_weight.item())
            # Old 3-subterm architecture
            if hasattr(local, "theta_theta_weight"):
                lams["theta_theta"] = float(local.theta_theta_weight.item())
            if hasattr(local, "delta_phi_weight"):
                lams["delta_phi"] = float(local.delta_phi_weight.item())
            if hasattr(local, "phi_phi_weight"):
                lams["phi_phi"] = float(local.phi_phi_weight.item())
        ss = getattr(model, "secondary", None)
        if ss is not None:
            lams["ram"] = float(ss.ram_weight.item()) if hasattr(ss, "ram_weight") else None
            if hasattr(ss, "hb_helix"):
                lams["hb_alpha"] = float(ss.hb_helix.lambda_hb.item())
            if hasattr(ss, "hb_sheet"):
                lams["hb_beta"] = float(ss.hb_sheet.lambda_hb.item())
        rep = getattr(model, "repulsion", None)
        if rep is not None:
            lams["rep"] = float(rep.lambda_rep.item())
        pack = getattr(model, "packing", None)
        if pack is not None:
            lams["pack_geom"] = float(pack.lambda_pack.item()) if hasattr(pack, "lambda_pack") else None
            lams["pack_contact"] = lams["pack_geom"]  # shared lambda
        return lams

    def run(
        self,
        segments: List[dict],
        model: torch.nn.Module,
        n_samples: int = 512,
        batch_size: int = 16,
    ) -> CalibrationResults:
        """Run calibration over PDB segments."""
        current_lambdas = self._read_current_lambdas(model)

        segs = list(segments)
        random.shuffle(segs)
        segs = segs[:n_samples]

        accum: Dict[str, List[float]] = {
            k: []
            for k in [
                "theta_phi",
                "theta_theta",
                "delta_phi",
                "phi_phi",
                "ram",
                "hb_alpha",
                "hb_beta",
                "rep",
                "pack_geom",
                "pack_contact",
            ]
        }

        model.eval()
        local = model.local
        is_4mer = hasattr(local, "theta_phi_energy")
        n_batches = math.ceil(len(segs) / batch_size)

        for batch_idx in range(n_batches):
            batch = segs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            if not batch:
                break

            R_clean, seq_t = self._collate(batch)
            R_pert = _perturb(R_clean, self.sigma_min, self.sigma_max)

            # Local: measured on perturbed structures
            if is_4mer:
                accum["theta_phi"].append(_raw_theta_phi(R_pert, seq_t, local).mean().item())
            else:
                if self.calibrate_mlp:
                    accum["theta_theta"].append(_raw_theta_theta(R_pert, seq_t, local).mean().item())
                    accum["phi_phi"].append(_raw_phi_phi(R_pert, seq_t, local).mean().item())
                accum["delta_phi"].append(_raw_delta_phi(R_pert, local).mean().item())

            # Secondary: ram on perturbed, H-bonds on perturbed (they use basin probs + distances)
            if model.secondary is not None:
                E_ram, E_hb_a, E_hb_b = _raw_secondary_components(R_pert, seq_t, model.secondary)
                accum["ram"].append(E_ram.mean().item())
                accum["hb_alpha"].append(E_hb_a.mean().item())
                accum["hb_beta"].append(E_hb_b.mean().item())

            # Repulsion: measured on perturbed
            if model.repulsion is not None:
                accum["rep"].append(_raw_repulsion(R_pert, model.repulsion).mean().item())

            # Packing: measured on NATIVE (contact energy collapses on perturbed)
            if model.packing is not None:
                E_geom, E_hp = _raw_packing_components(R_clean, seq_t, model.packing)
                accum["pack_geom"].append(abs(E_geom.mean().item()))
                accum["pack_contact"].append(abs(E_hp.mean().item()))

            if (batch_idx + 1) % 10 == 0:
                logger.info("  Calibration: %d / %d batches", batch_idx + 1, n_batches)

        import numpy as np

        means = {k: float(np.mean(v)) if v else None for k, v in accum.items()}

        if is_4mer:
            return CalibrationResults(
                init_theta_phi_weight=self._safe_weight(means["theta_phi"], "theta_phi"),
                init_ram_weight=self._safe_weight(means["ram"], "ram"),
                init_hb_alpha_weight=self._safe_weight(means["hb_alpha"], "hb_alpha"),
                init_hb_beta_weight=self._safe_weight(means["hb_beta"], "hb_beta"),
                init_rep_weight=self._safe_weight(means["rep"], "rep"),
                init_pack_geom_weight=self._safe_weight(means["pack_geom"], "pack_geom"),
                init_pack_contact_weight=self._safe_weight(means["pack_contact"], "pack_contact"),
                means=means,
                current_lambdas=current_lambdas,
                target=self.target,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                n_samples=len(segs),
            )
        else:
            return CalibrationResults(
                init_theta_theta_weight=self._safe_weight(means["theta_theta"], "theta_theta")
                if self.calibrate_mlp
                else 1.0,
                init_delta_phi_weight=self._safe_weight(means["delta_phi"], "delta_phi"),
                init_phi_phi_weight=self._safe_weight(means["phi_phi"], "phi_phi") if self.calibrate_mlp else None,
                init_ram_weight=self._safe_weight(means["ram"], "ram"),
                init_hb_alpha_weight=self._safe_weight(means["hb_alpha"], "hb_alpha"),
                init_hb_beta_weight=self._safe_weight(means["hb_beta"], "hb_beta"),
                init_rep_weight=self._safe_weight(means["rep"], "rep"),
                init_pack_geom_weight=self._safe_weight(means["pack_geom"], "pack_geom"),
                init_pack_contact_weight=self._safe_weight(means["pack_contact"], "pack_contact"),
                means=means,
                current_lambdas=current_lambdas,
                target=self.target,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                n_samples=len(segs),
            )

    def _safe_weight(self, mean_val: Optional[float], name: str) -> float:
        """Compute target / |mean_raw|, with validation and clamping.

        Caps at 10.0 to prevent blow-ups: terms with tiny raw output (e.g.
        sparse H-bonds) would get λ=50+ which amplifies rapidly as training
        grows their output. Better to start moderate and let balance loss adjust.
        """
        if mean_val is None or not math.isfinite(mean_val) or mean_val == 0:
            logger.warning(
                "  Could not compute init weight for '%s' (mean=%.6g) — defaulting to 1.0",
                name,
                mean_val or 0.0,
            )
            return 1.0
        abs_mean = abs(mean_val)
        weight = self.target / abs_mean
        weight = float(max(1e-3, min(1.5, weight)))  # cap at 1.5 — H-bond Hessians scale with λ
        if self.target / abs_mean > 1.5:
            logger.info(
                "  %-20s mean_raw=%.6f  target=%.4f  init_weight=%.4f (CAPPED from %.1f)",
                name,
                mean_val,
                self.target,
                weight,
                self.target / abs_mean,
            )
        else:
            logger.info("  %-20s mean_raw=%.6f  target=%.4f  init_weight=%.4f", name, mean_val, self.target, weight)
        return weight

    @staticmethod
    def _collate(batch: List[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad variable-length segments to a uniform batch."""

        def _coords(s: dict) -> torch.Tensor:
            return s["coords"] if "coords" in s else s["R"]

        max_L = max(_coords(s).shape[0] for s in batch)
        Rs, seqs = [], []
        for s in batch:
            R = _coords(s)
            seq = s["seq"]
            pad = max_L - R.shape[0]
            if pad > 0:
                R = torch.cat([R, R.new_zeros(pad, 3)])
                seq = torch.cat([seq, seq.new_zeros(pad)])
            Rs.append(R)
            seqs.append(seq)
        return torch.stack(Rs), torch.stack(seqs)
