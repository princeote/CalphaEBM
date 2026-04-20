# src/calphaebm/models/energy.py
# src/calphaebm/models/energy.py
"""Complete energy model combining all terms with learnable gates.

E_total = gate_local * E_local + gate_repulsion * E_rep + gate_secondary * E_ss + gate_packing * E_pack

All terms are initialized from analysis data:
- Local: Backbone connectivity + torsional preferences
- Repulsion: Monotonic wall from RDF analysis
- Secondary: Mixture-of-basins + MLP correlations
- Packing: Pair preferences from contact enrichment analysis

NOTE:
This file wires packing debug flags (packing_debug_scale / packing_debug_every)
so they are consumed only by PackingEnergy and do not leak as unexpected kwargs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from calphaebm.models.local import LocalEnergy
from calphaebm.models.packing import PackingEnergy
from calphaebm.models.repulsion import RepulsionEnergy
from calphaebm.models.secondary import SecondaryStructureEnergy
from calphaebm.utils.logging import get_logger

logger = get_logger()


class TotalEnergy(nn.Module):
    """Complete energy model with up to four terms and learnable gates."""

    def __init__(
        self,
        # Paths to analysis data
        backbone_data_dir: str = "analysis/backbone_geometry/data",
        secondary_data_dir: str = "analysis/secondary_analysis/data",
        repulsion_data_dir: str = "analysis/repulsion_analysis/data",
        packing_data_dir: Optional[str] = None,
        # Model architecture
        emb_dim: int = 16,
        hidden_dims: Tuple[int, ...] = (128, 128),
        # Nonbonded parameters
        K_neighbors: int = 64,
        exclude_nonbonded: int = 3,
        repulsion_r_on: float = 8.0,
        repulsion_r_cut: float = 10.0,
        packing_r_on: float = 8.0,
        packing_r_cut: float = 10.0,
        # Packing specific parameters
        packing_short_gate_on: float = 4.5,
        packing_short_gate_off: float = 5.0,
        packing_rbf_centers: Tuple[float, ...] = (5.5, 7.0, 9.0),
        packing_rbf_width: float = 1.0,
        packing_max_dist: float = 10.0,
        packing_init_from: str = "log_oe",
        packing_normalize_by_length: bool = True,
        # Geometry calibration — kept for backward compat, no longer used (geom MLP removed)
        packing_geom_calibration: Optional[str] = None,
        # Packing diagnostics — kept for backward compat, no longer used
        packing_debug_scale: bool = False,
        packing_debug_every: int = 200,
        # Rg Flory size restraint (part of packing energy)
        packing_rg_lambda: float = 0.1,
        packing_rg_r0: float = 2.0,
        packing_rg_nu: float = 0.38,
        # Coordination penalty (part of packing energy)
        coord_lambda: float = 0.01,
        coord_n_lo: Optional[list] = None,
        coord_n_hi: Optional[list] = None,
        coord_n_mean: Optional[list] = None,  # per-AA mean coordination (suppresses _Hydrophobic warning)
        coord_n_std: Optional[list] = None,  # per-AA std  coordination
        # Local term parameters
        init_bond_spring: float = 50.0,
        init_theta_theta_weight: float = 1.0,
        init_delta_phi_weight: float = 1.0,
        local_window_size: int = 8,
        # Secondary basin params (kept explicit for compatibility)
        num_basins: int = 4,
        use_cos_theta: bool = True,
        normalize_secondary_by_length: bool = True,
        # Debug
        debug_mode: bool = False,
        # Internal lambda inits — calibrated via `calphaebm calibrate`
        init_lambda_pack: float = 1.0,
        # Which terms to include
        include_repulsion: bool = True,
        include_secondary: bool = True,
        include_packing: bool = True,
        # Physics prior mode — zeros learnable logits in secondary,
        # zeros local MLP contribution, for untrained physics-only evaluation.
        physics_prior: bool = False,
        # Packing extra kwargs (Run5: ρ params, constraint params, coord stats)
        packing_extra: Optional[Dict] = None,
        # Learnable buffer flags (all default False = backwards compat)
        learn_packing_coords: bool = False,
        learn_packing_density: bool = False,
        learn_penalty_shapes: bool = False,
        learn_packing_bounds: bool = False,
        learn_penalty_strengths: bool = False,
        learn_gate_geometry: bool = False,
        learn_hbond_geometry: bool = False,
        # Device
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Default packing data dir to repulsion data dir if omitted
        if packing_data_dir is None:
            packing_data_dir = repulsion_data_dir
            logger.debug("packing_data_dir not provided, using repulsion_data_dir: %s", repulsion_data_dir)
        else:
            logger.debug("Using packing_data_dir: %s", packing_data_dir)

        # 1) Local term (always included)
        logger.debug("Initializing LocalEnergy term...")
        self.local = LocalEnergy(
            window_size=local_window_size,
            data_dir=backbone_data_dir,
            init_bond_spring=init_bond_spring,
            init_theta_theta_weight=init_theta_theta_weight,
            init_delta_phi_weight=init_delta_phi_weight,
            secondary_data_dir=secondary_data_dir,
        )

        # 2) Repulsion term (optional)
        self.repulsion: Optional[RepulsionEnergy] = None
        if include_repulsion:
            logger.debug("Initializing RepulsionEnergy term...")
            # IMPORTANT: Repulsion scaling is handled by the outer gate; RepulsionEnergy should not need init_lambda.
            self.repulsion = RepulsionEnergy(
                data_dir=repulsion_data_dir,
                K=K_neighbors,
                exclude=exclude_nonbonded,
                r_on=repulsion_r_on,
                r_cut=repulsion_r_cut,
            )

        # 3) Secondary term (optional)
        self.secondary: Optional[SecondaryStructureEnergy] = None
        if include_secondary:
            logger.debug("Initializing SecondaryStructureEnergy term...")
            self.secondary = SecondaryStructureEnergy(
                num_aa=20,
                emb_dim=emb_dim,
                hidden_dims=hidden_dims,
                num_basins=num_basins,
                use_cos_theta=use_cos_theta,
                data_dir=secondary_data_dir,
                debug_mode=debug_mode,
                physics_prior=physics_prior,
                learn_hbond_geometry=learn_hbond_geometry,
            )

        # 4) Packing term (optional)
        self.packing: Optional[PackingEnergy] = None
        if include_packing:
            logger.debug("Initializing PackingEnergy term...")
            _packing_kw = dict(
                num_aa=20,
                topk=K_neighbors,
                exclude=exclude_nonbonded,
                max_dist=packing_max_dist,
                normalize_by_length=packing_normalize_by_length,
                rg_lambda=packing_rg_lambda,
                rg_r0=packing_rg_r0,
                rg_nu=packing_rg_nu,
                coord_lambda=coord_lambda,
                coord_n_lo=coord_n_lo,
                coord_n_hi=coord_n_hi,
                coord_n_mean=coord_n_mean,
                coord_n_std=coord_n_std,
                secondary_data_dir=secondary_data_dir,
                learn_packing_coords=learn_packing_coords,
                learn_packing_density=learn_packing_density,
                learn_penalty_shapes=learn_penalty_shapes,
                learn_packing_bounds=learn_packing_bounds,
                learn_penalty_strengths=learn_penalty_strengths,
                learn_gate_geometry=learn_gate_geometry,
                learn_hbond_geometry=learn_hbond_geometry,
            )
            if packing_extra:
                _packing_kw.update(packing_extra)
            self.packing = PackingEnergy(**_packing_kw)

        # Outer gates — frozen at 1.0 (not trained; internal lambdas handle scaling)
        # In physics_prior mode, local term is zeroed (random MLP → pure noise).
        _gate_local_init = 0.0 if physics_prior else 1.0
        self.register_buffer("gate_local", torch.tensor(_gate_local_init, dtype=torch.float32))
        self.register_buffer("gate_repulsion", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("gate_secondary", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("gate_packing", torch.tensor(1.0, dtype=torch.float32))

        if physics_prior:
            logger.info("TotalEnergy: physics_prior=True — gate_local=0.0, " "secondary logits forced to 0")

        # Summary is logged post-load by train_main calling _log_summary()
        logger.debug(
            "TotalEnergy model constructed (%d params)", sum(p.numel() for p in self.parameters() if p.requires_grad)
        )

        if device is not None:
            self.to(device)

    def _log_summary(self) -> None:
        """Consolidated architecture summary — one block instead of 80 scattered lines."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        sep = "═" * 66

        # Gather per-term param counts and descriptions
        lines = []
        lines.append(sep)
        lines.append("  ARCHITECTURE (%d trainable params)" % total_params)
        lines.append("─" * 66)

        # Local
        ls = getattr(self.local, "_init_summary", {})
        lp = ls.get("total_params", "?")
        lines.append(
            "  local       %5sp  %d-mer θφ MLP(in=%s, h=%s)"
            % (
                lp,
                ls.get("window_size", "?"),
                ls.get("input_dim", "?"),
                ls.get("hidden_dims", "?"),
            )
        )

        # Secondary
        if self.secondary is not None:
            ss = getattr(self.secondary, "_init_summary", {})
            sp = ss.get("total_params", "?")
            lines.append(
                "  secondary   %5sp  E_ram(%d basins) + E_hb_α(μ=%.1fÅ) + E_hb_β(μ1=%.1f,μ2=%.1fÅ)"
                % (
                    sp,
                    ss.get("num_basins", 4),
                    ss.get("hb_helix_mu", 0),
                    ss.get("hb_sheet_mu1", 0),
                    ss.get("hb_sheet_mu2", 0),
                )
            )

        # Packing
        if self.packing is not None:
            ps = getattr(self.packing, "_init_summary", {})
            pp = ps.get("total_params", "?")
            lines.append(
                "  packing     %5sp  E_geom MLP(%dp) + E_contact(%dp, SVD init)"
                % (
                    pp,
                    ps.get("geom_params", 0),
                    ps.get("contact_params", 0),
                )
            )

        # Repulsion
        if self.repulsion is not None:
            rs = getattr(self.repulsion, "_init_summary", {})
            rp = rs.get("total_params", 1)
            r_range = rs.get("r_range", (0, 0))
            lines.append(
                "  repulsion   %5sp  tabulated wall [%.1f, %.1f]Å"
                % (
                    rp,
                    r_range[0],
                    r_range[1],
                )
            )

        lines.append("─" * 66)

        # Lambdas
        lam_parts = []
        lam_parts.append("θφ=%.3f" % self.local.weight.item())
        if self.secondary is not None:
            lam_parts.append("ram=%.3f" % self.secondary.ram_weight.item())
            if hasattr(self.secondary, "hb_helix"):
                lam_parts.append("hbα=%.3f" % self.secondary.hb_helix.lambda_hb.item())
            if hasattr(self.secondary, "hb_sheet"):
                lam_parts.append("hbβ=%.3f" % self.secondary.hb_sheet.lambda_hb.item())
        if self.repulsion is not None:
            lam_parts.append("rep=%.3f" % self.repulsion.lambda_rep.item())
        if self.packing is not None:
            if hasattr(self.packing, "burial") and hasattr(self.packing.burial, "lambda_hp"):
                lam_parts.append("cont=%.3f" % self.packing.burial.lambda_hp.item())
        lines.append("  Lambdas:  " + "  ".join(lam_parts))

        # Gates
        lines.append(
            "  Gates:    local=%.1f  secondary=%.1f  repulsion=%.1f  packing=%.1f"
            % (
                self.gate_local.item(),
                self.gate_secondary.item(),
                self.gate_repulsion.item(),
                self.gate_packing.item(),
            )
        )
        lines.append(sep)

        logger.info("\n".join(lines))

    def log_packing_norm_state(self, context: str = "") -> None:
        """Log packing term state. Geom MLP removed in v3; kept for API compat."""
        if self.packing is None:
            logger.info("log_packing_norm_state: no packing term present")
            return
        prefix = f"[{context}] " if context else ""
        logger.info("%sPacking v3: contact-only (geom MLP removed)", prefix)

    def forward_dsm(self, R: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Energy for DSM training — excludes bond_spring from local term.

        Bond_spring is a fixed physical buffer (k=750 at gate=5). Its forces
        (~450/residue at sigma=0.3Å) are ~100× larger than the learned terms
        and dominate the DSM score residual, preventing gradient signal from
        reaching secondary, packing, and repulsion. Bond geometry is already
        encoded in the data distribution — DSM learns it implicitly.

        All other terms are identical to forward().
        """
        E = self.gate_local * self.local.forward_learned(R, seq, lengths=lengths)

        if self.repulsion is not None:
            E = E + self.gate_repulsion * self.repulsion(R, seq, lengths=lengths)

        if self.secondary is not None:
            E = E + self.gate_secondary * self.secondary(R, seq, lengths=lengths)

        if self.packing is not None:
            E = E + self.gate_packing * self.packing(R, seq, lengths=lengths)

        return E

    def forward(self, R: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Full energy including fixed bond_spring.

        Used for diagnostics, Langevin generation, geogap, and energy gap
        reporting. NOT called by DSM training loss — use forward_dsm() there.
        """
        E = self.gate_local * self.local(R, seq, lengths=lengths)

        if self.repulsion is not None:
            E = E + self.gate_repulsion * self.repulsion(R, seq, lengths=lengths)

        if self.secondary is not None:
            E = E + self.gate_secondary * self.secondary(R, seq, lengths=lengths)

        if self.packing is not None:
            E = E + self.gate_packing * self.packing(R, seq, lengths=lengths)

        return E

    @torch.no_grad()
    def term_energies(
        self, R: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {"local": self.local(R, seq, lengths=lengths)}
        if self.repulsion is not None:
            out["repulsion"] = self.repulsion(R, seq, lengths=lengths)
        if self.secondary is not None:
            out["secondary"] = self.secondary(R, seq, lengths=lengths)
        if self.packing is not None:
            out["packing"] = self.packing(R, seq, lengths=lengths)
        return out

    @torch.no_grad()
    def set_gates(self, **kwargs) -> None:
        for name, value in kwargs.items():
            v = float(value)
            if name == "local":
                self.gate_local.fill_(v)
            elif name == "repulsion":
                self.gate_repulsion.fill_(v)
            elif name == "secondary":
                self.gate_secondary.fill_(v)
            elif name == "packing":
                self.gate_packing.fill_(v)
            else:
                raise ValueError(f"Unknown gate: {name}")

    def get_gates(self) -> Dict[str, float]:
        return {
            "local": float(self.gate_local.item()),
            "repulsion": float(self.gate_repulsion.item()),
            "secondary": float(self.gate_secondary.item()),
            "packing": float(self.gate_packing.item()),
        }

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_total_energy(
    backbone_data_dir: str = "analysis/backbone_geometry/data",
    secondary_data_dir: str = "analysis/secondary_analysis/data",
    repulsion_data_dir: str = "analysis/repulsion_analysis/data",
    packing_data_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> TotalEnergy:
    """Factory function to create a total energy model.

    IMPORTANT:
    This function must not forward unexpected kwargs into TotalEnergy.__init__.
    Any extra args should be handled here (or in the caller).
    """
    backbone_path = Path(backbone_data_dir)
    repulsion_path = Path(repulsion_data_dir)

    if not backbone_path.exists():
        logger.warning("Backbone data directory not found: %s", backbone_data_dir)

    if not repulsion_path.exists():
        raise FileNotFoundError(
            f"Repulsion data directory not found: {repulsion_data_dir}. " "Run 'calphaebm analyze repulsion' first."
        )

    if packing_data_dir is not None:
        packing_path = Path(packing_data_dir)
        if not packing_path.exists():
            logger.warning(
                "Packing data directory not found: %s; falling back to %s",
                packing_data_dir,
                repulsion_data_dir,
            )
            packing_data_dir = repulsion_data_dir

    # If someone passes these via **kwargs but TotalEnergy signature changes, this prevents leakage.
    # (Safe even if kwargs already matches TotalEnergy; pop defaults and re-insert below.)
    packing_debug_scale = kwargs.pop("packing_debug_scale", False)
    packing_debug_every = kwargs.pop("packing_debug_every", 200)
    packing_rg_lambda = kwargs.pop("packing_rg_lambda", 0.1)
    packing_rg_r0 = kwargs.pop("packing_rg_r0", 2.0)
    packing_rg_nu = kwargs.pop("packing_rg_nu", 0.38)
    coord_lambda = kwargs.pop("coord_lambda", 0.01)
    coord_n_lo = kwargs.pop("coord_n_lo", None)
    coord_n_hi = kwargs.pop("coord_n_hi", None)
    coord_n_mean = kwargs.pop("coord_n_mean", None)
    coord_n_std = kwargs.pop("coord_n_std", None)
    local_window_size = kwargs.pop("local_window_size", 8)
    packing_extra = kwargs.pop("packing_extra", None)

    # Backward-compat: older CLI / configs may pass rg_gate_sigma, init_gate_*, or packing cap options.
    # These are silently consumed.
    kwargs.pop("rg_gate_sigma", None)
    kwargs.pop("init_gate_local", None)
    kwargs.pop("init_gate_repulsion", None)
    kwargs.pop("init_gate_secondary", None)
    kwargs.pop("init_gate_packing", None)
    kwargs.pop("packing_cap_cc", None)
    kwargs.pop("packing_cap_value", None)

    return TotalEnergy(
        backbone_data_dir=backbone_data_dir,
        secondary_data_dir=secondary_data_dir,
        repulsion_data_dir=repulsion_data_dir,
        packing_data_dir=packing_data_dir,
        device=device,
        packing_debug_scale=packing_debug_scale,
        packing_debug_every=packing_debug_every,
        packing_rg_lambda=packing_rg_lambda,
        packing_rg_r0=packing_rg_r0,
        packing_rg_nu=packing_rg_nu,
        coord_lambda=coord_lambda,
        coord_n_lo=coord_n_lo,
        coord_n_hi=coord_n_hi,
        coord_n_mean=coord_n_mean,
        coord_n_std=coord_n_std,
        local_window_size=local_window_size,
        packing_extra=packing_extra,
        **kwargs,
    )
