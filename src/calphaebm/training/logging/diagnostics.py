"""
Diagnostic logging utilities for training.

Normalization policy (IMPORTANT):
- Each energy term may *internally* normalize by sequence length (or by its own
  natural count, e.g. bonds L-1, torsions L-4). This is controlled per-term by
  `normalize_by_length`.

- Therefore, diagnostics must NOT blindly divide by L again, or you will
  double-normalize and shrink energies artificially.

This module reports:
- Energy values in the term's own units (already per-residue-like when
  normalize_by_length=True).
- A single structured block every N steps consolidating: lambdas, gates,
  term contributions + gaps, safety metrics, and geogap status.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from calphaebm.models.learnable_buffers import buffer_drift_report
from calphaebm.utils.logging import get_logger
from calphaebm.utils.neighbors import pairwise_distances

logger = get_logger()


class ExponentialMovingAverage:
    """Simple EMA for tracking metrics."""

    def __init__(self, alpha: float = 0.99):
        self.alpha = float(alpha)
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * x
        return self.value

    def reset(self) -> None:
        self.value = None


def _is_internally_normalized(module) -> bool:
    """Return True if module advertises internal length normalization."""
    return bool(getattr(module, "normalize_by_length", False))


def _infer_nonbonded_exclude(model, default: int = 3) -> int:
    for name in ("packing", "repulsion"):
        mod = getattr(model, name, None)
        if mod is not None and hasattr(mod, "exclude"):
            try:
                return int(getattr(mod, "exclude"))
            except Exception:
                pass
    return int(default)


def _read_lambdas(model) -> Dict[str, Optional[float]]:
    """Read all current lambda values from the model."""
    import torch.nn.functional as F

    lams: Dict[str, Optional[float]] = {}
    local = getattr(model, "local", None)
    if local is not None:
        # 4-mer architecture
        lams["theta_phi"] = float(local.theta_phi_weight.item()) if hasattr(local, "theta_phi_weight") else None
        # Old 3-subterm architecture
        lams["theta"] = float(local.theta_theta_weight.item()) if hasattr(local, "theta_theta_weight") else None
        lams["delta_phi"] = (
            float(F.softplus(local._delta_phi_weight_raw).item()) if hasattr(local, "_delta_phi_weight_raw") else None
        )
        lams["phi_phi"] = float(local.phi_phi_weight.item()) if hasattr(local, "phi_phi_weight") else None
    ss = getattr(model, "secondary", None)
    if ss is not None:
        lams["ram"] = (
            float(ss.ram_weight.item())
            if hasattr(ss, "ram_weight")
            else (float(F.softplus(ss.lambda_ram).item()) if hasattr(ss, "lambda_ram") else None)
        )
        lams["hb_a"] = float(ss.hb_helix.lambda_hb.item()) if hasattr(ss, "hb_helix") else None
        lams["hb_b"] = float(ss.hb_sheet.lambda_hb.item()) if hasattr(ss, "hb_sheet") else None
    rep = getattr(model, "repulsion", None)
    if rep is not None:
        lams["rep"] = float(F.softplus(rep._lambda_rep_raw).item()) if hasattr(rep, "_lambda_rep_raw") else None
    pack = getattr(model, "packing", None)
    if pack is not None:
        # v5: hp_rew (from burial) + rho_rew
        if hasattr(pack, "burial") and hasattr(pack.burial, "_lambda_hp_raw"):
            lams["hp_rew"] = float(F.softplus(pack.burial._lambda_hp_raw).item())
        if hasattr(pack, "_lambda_rho_raw"):
            lams["rho_rew"] = float(F.softplus(pack._lambda_rho_raw).item())
        # v4 fallback
        lams["geom"] = float(F.softplus(pack._lambda_pack_raw).item()) if hasattr(pack, "_lambda_pack_raw") else None
        if "hp_rew" not in lams and hasattr(pack, "burial") and hasattr(pack.burial, "_lambda_hp_raw"):
            lams["cont"] = float(F.softplus(pack.burial._lambda_hp_raw).item())
    return lams


class DiagnosticLogger:
    """Helper class for logging comprehensive training diagnostics."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.clip_fractions = []
        self.max_forces = []
        self.last_mean_gap = 0.0

        # EMA tracker for smooth metrics (single-batch diagnostics are noisy)
        # alpha=0.2 → 20% weight on new data, half-life ≈ 3 steps, responsive to recent batches
        from calphaebm.training.core.ema_tracker import EMATracker

        self.ema = EMATracker(alpha=0.2)

    def update_ema(self, **kwargs: float) -> None:
        """Update EMA metrics. Call every training step from full_phase.py.

        Typical keys: dsm, z_score, balance, basin_gap, anti_funnel,
        hb_alpha, hb_beta, total_loss, max_force
        """
        self.ema.update(**kwargs)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _repulsion_metrics(self, R: torch.Tensor, lengths: torch.Tensor | None = None) -> dict:
        B, L = R.shape[:2]
        exclude = _infer_nonbonded_exclude(self.model, default=3)
        idx = torch.arange(L, device=R.device)
        sep = (idx[:, None] - idx[None, :]).abs()
        triu_mask = torch.triu(torch.ones(L, L, device=R.device, dtype=torch.bool), diagonal=1)
        nonbonded_mask = (sep > exclude) & triu_mask  # (L, L)

        # Mask out pairs involving padding atoms
        if lengths is not None:
            valid = torch.arange(L, device=R.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, L)
            pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, L, L)
        else:
            pair_valid = None

        D = pairwise_distances(R)  # (B, L, L)

        # Apply both masks
        if pair_valid is not None:
            # Set padding pair distances to inf so they don't affect min/stats
            D = D.masked_fill(~pair_valid, float("inf"))

        D_flat = D.reshape(B, -1)
        mask_flat = nonbonded_mask.reshape(-1)
        nonbonded_dists = D_flat[:, mask_flat]
        if nonbonded_dists.numel() == 0:
            return {"exclude": exclude, "min_dist": float("inf"), "frac_below_40": 0.0, "frac_below_45": 0.0}
        return {
            "exclude": int(exclude),
            "min_dist": float(nonbonded_dists.amin(dim=1).median().item()),
            "frac_below_40": float((nonbonded_dists < 4.0).float().mean(dim=1).median().item()),
            "frac_below_45": float((nonbonded_dists < 4.5).float().mean(dim=1).median().item()),
        }

    def _term_energies(
        self, R: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> Dict[str, float]:
        """Compute per-term energies, pulling subterm splits where available.

        Uses subterm_energies() for secondary and packing so each term does
        ONE forward pass that returns both the total and the split.
        """
        out = {}

        # ── local ────────────────────────────────────────────────────────
        local_mod = getattr(self.model, "local", None)
        if local_mod is not None:
            out["local"] = float(local_mod(R, seq, lengths=lengths).mean().item())
            # 4-mer architecture
            if hasattr(local_mod, "theta_phi_energy"):
                out["local_thetaphi"] = float(local_mod.theta_phi_energy(R, seq, lengths=lengths).mean().item())
            # Old 3-subterm architecture
            if hasattr(local_mod, "theta_theta_energy"):
                out["local_thetatheta"] = float(local_mod.theta_theta_energy(R, seq, lengths=lengths).mean().item())
            if hasattr(local_mod, "delta_phi_energy"):
                out["local_deltaphi"] = float(local_mod.delta_phi_energy(R, lengths=lengths).mean().item())
            if hasattr(local_mod, "phi_phi_energy"):
                out["local_phiphi"] = float(local_mod.phi_phi_energy(R, seq, lengths=lengths).mean().item())

        # ── repulsion ─────────────────────────────────────────────────────
        rep_mod = getattr(self.model, "repulsion", None)
        if rep_mod is not None:
            out["repulsion"] = float(rep_mod(R, seq, lengths=lengths).mean().item())

        # ── secondary (one call → ram + hb_α + hb_β) ─────────────────────
        ss_mod = getattr(self.model, "secondary", None)
        if ss_mod is not None:
            if hasattr(ss_mod, "subterm_energies"):
                E_ram, E_tp, E_pp = ss_mod.subterm_energies(R, seq, lengths=lengths)
                out["secondary_ram"] = float(E_ram.mean().item())
                out["secondary_hb_alpha"] = float(E_tp.mean().item())
                out["secondary_hb_beta"] = float(E_pp.mean().item())
                out["secondary"] = out["secondary_ram"] + out["secondary_hb_alpha"] + out["secondary_hb_beta"]
            else:
                out["secondary"] = float(ss_mod(R, seq, lengths=lengths).mean().item())

        # ── packing (learned: hp_reward + rho_reward) + constraint (analytical)
        pack_mod = getattr(self.model, "packing", None)
        if pack_mod is not None:
            if hasattr(pack_mod, "subterm_energies"):
                subterms = pack_mod.subterm_energies(R, seq, lengths=lengths)
                if len(subterms) == 5:
                    # v5: (E_hp_reward, E_hp_penalty, E_rho_reward, E_rho_penalty, E_rg_penalty)
                    E_hp_rew, E_hp_pen, E_rho_rew, E_rho_pen, E_rg_pen = subterms
                    out["packing_hp_reward"] = float(E_hp_rew.mean().item())
                    out["packing_hp_penalty"] = float(E_hp_pen.mean().item())
                    out["packing_rho_reward"] = float(E_rho_rew.mean().item())
                    out["packing_rho_penalty"] = float(E_rho_pen.mean().item())
                    out["packing_rg"] = float(E_rg_pen.mean().item())
                    # Packing = learned subterms (in E-balance)
                    out["packing_contact"] = out["packing_hp_reward"]
                    out["packing"] = out["packing_hp_reward"] + out["packing_rho_reward"]
                    # Constraints = analytical guardrails, 0 trainable params
                    out["packing_coord"] = out["packing_hp_penalty"]
                    out["constraint"] = out["packing_hp_penalty"] + out["packing_rho_penalty"] + out["packing_rg"]
                else:
                    # v4 fallback: (E_hp, E_coord, E_rg)
                    E_hp, E_coord, E_rg = subterms
                    out["packing_contact"] = float(E_hp.mean().item())
                    out["packing_coord"] = float(E_coord.mean().item())
                    out["packing_rg"] = float(E_rg.mean().item())
                    out["packing"] = out["packing_contact"]
                    out["constraint"] = out["packing_coord"] + out["packing_rg"]
            else:
                out["packing"] = float(pack_mod(R, seq, lengths=lengths).mean().item())

        return out

    def _gates(self) -> Dict[str, float]:
        g = {}
        for name in ("local", "repulsion", "secondary", "packing"):
            attr = f"gate_{name}"
            if hasattr(self.model, attr):
                g[name] = float(getattr(self.model, attr).item())
        return g

    # ── public API ────────────────────────────────────────────────────────────

    def log_step_block(
        self,
        phase_step: int,
        n_steps: int,
        loss: float,
        lr: float,
        R: torch.Tensor,
        seq: torch.Tensor,
        forces: Optional[torch.Tensor] = None,
        force_cap: float = 50.0,
        perturb_sigma: float = 0.30,
        geogap_diag: Optional[Dict[str, Any]] = None,
        geogap_margin: float = 0.20,
        loss_pack_c: Optional[float] = None,
        lambda_pack_contrastive: float = 0.0,
        pack_contrastive_margin: float = 0.5,
        loss_balance: Optional[float] = None,
        lambda_balance: float = 0.0,
        term_absmeans: Optional[Dict[str, float]] = None,
        term_absmeans_agg: Optional[Dict[str, float]] = None,
        balance_r_term: float = 4.0,
        loss_basin: Optional[float] = None,
        lambda_basin: float = 0.0,
        basin_margin: float = 0.5,
        loss_native: Optional[float] = None,
        lambda_native: float = 0.0,
        native_margin: float = 0.5,
        native_diag: Optional[Dict[str, Any]] = None,
        lengths: Optional[torch.Tensor] = None,
        # ELT diagnostics
        elt_diag: Optional[Dict[str, Any]] = None,
        loss_funnel: Optional[float] = None,
        lambda_funnel: float = 0.0,
        loss_zscore: Optional[float] = None,
        lambda_zscore: float = 0.0,
        target_zscore: float = 3.0,
        loss_elt_gap: Optional[float] = None,
        lambda_gap_elt: float = 0.0,
        loss_frust: Optional[float] = None,
        lambda_frustration: float = 0.0,
        # Native depth loss
        loss_native_depth: Optional[float] = None,
        lambda_native_depth: float = 0.0,
        target_native_depth: float = -1.0,
        e_native_depth: Optional[float] = None,
        # Pre-computed from training loop (avoids double computation)
        precomputed: Optional[Dict[str, Any]] = None,
        # DSM loss (for display in diagnostic block)
        loss_dsm: Optional[float] = None,
        dsm_diag: Optional[Dict] = None,
    ) -> None:
        """
        Emit one structured diagnostic block combining all metrics.

        When precomputed dict is provided, uses those values instead of
        recomputing. Keys: "term_energies", "safety", "max_force", "clip_frac".
        Gap profiling (σ perturbation study) is diagnostic-only and always
        computed here.
        """
        W = 66  # block width
        precomputed = precomputed or {}

        # Multiple sigma levels for gap profiling
        gap_sigmas = [0.30, 0.50, 1.00, 2.00]

        with torch.no_grad():
            gates = self._gates()
            lams = _read_lambdas(self.model)

            # Always compute term energies fresh — includes subterm splits
            # (precomputed only has the 4 main terms, not subterms like
            #  local_thetatheta, secondary_ram, packing_contact, etc.)
            E_clean = self._term_energies(R, seq, lengths=lengths)

            # Gap profiling: IC-space perturbations (consistent with ELT and Langevin)
            # Sigmas are in radians — 0.3≈17°, 0.5≈29°, 1.0≈57°, 2.0≈115°
            # Structures with min nonbonded dist < 1.0Å are rejected and resampled.
            import math as _math

            from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct

            E_perts = {}
            try:
                theta_gap, phi_gap = coords_to_internal(R)
                anchor_gap = extract_anchor(R)

                # Pre-compute padding masks
                if lengths is not None:
                    idx_t = torch.arange(theta_gap.shape[1], device=R.device)
                    idx_p = torch.arange(phi_gap.shape[1], device=R.device)
                    vt = idx_t.unsqueeze(0) < (lengths.unsqueeze(1) - 2)
                    vp = idx_p.unsqueeze(0) < (lengths.unsqueeze(1) - 3)
                else:
                    vt = vp = None

                for sig in gap_sigmas:
                    noise_t = sig * 0.161 * torch.randn_like(theta_gap)
                    noise_p = sig * torch.randn_like(phi_gap)
                    if vt is not None:
                        noise_t = noise_t * vt.float()
                        noise_p = noise_p * vp.float()
                    theta_p = (theta_gap + noise_t).clamp(0.01, _math.pi - 0.01)
                    phi_p = phi_gap + noise_p
                    phi_p = (phi_p + _math.pi) % (2 * _math.pi) - _math.pi
                    R_p = nerf_reconstruct(theta_p, phi_p, anchor_gap)
                    E_perts[sig] = self._term_energies(R_p, seq, lengths=lengths)
            except Exception as _gap_err:
                logger.warning("Gap profiling failed: %s", _gap_err)
                for sig in gap_sigmas:
                    E_perts[sig] = dict(E_clean)

        # ── header ────────────────────────────────────────────────────────────
        header = f" STEP {phase_step}/{n_steps} | loss={loss:.4f} | lr={lr:.2e} "
        logger.info("═" * W)
        logger.info(header.center(W))
        logger.info("─" * W)

        # ── lambdas ───────────────────────────────────────────────────────────
        def _lv(key, fix=False):
            v = lams.get(key)
            if v is None:
                return "n/a"
            suffix = "(fix)" if fix else ""
            return f"{v:.3f}{suffix}"

        if lams.get("theta_phi") is not None:
            if lams.get("hp_rew") is not None:
                # v5: hp_rew + rho_rew
                logger.info(
                    "  Lambdas:  θφ=%-7s ram=%-7s hbα=%-7s hbβ=%-7s rep=%-7s hp=%-7s ρ=%s",
                    _lv("theta_phi"),
                    _lv("ram"),
                    _lv("hb_a"),
                    _lv("hb_b"),
                    _lv("rep"),
                    _lv("hp_rew"),
                    _lv("rho_rew"),
                )
            else:
                # v4: geom + cont
                logger.info(
                    "  Lambdas:  θφ=%-7s ram=%-7s hbα=%-7s hbβ=%-7s rep=%-7s geom=%-7s cont=%s",
                    _lv("theta_phi"),
                    _lv("ram"),
                    _lv("hb_a"),
                    _lv("hb_b"),
                    _lv("rep"),
                    _lv("geom"),
                    _lv("cont"),
                )
        else:
            logger.info(
                "  Lambdas:  θθ=%-7s Δφ=%-7s φφ=%-7s ram=%-7s hbα=%-7s hbβ=%-7s rep=%-7s geom=%-7s cont=%s",
                _lv("theta"),
                _lv("delta_phi"),
                _lv("phi_phi"),
                _lv("ram"),
                _lv("hb_a"),
                _lv("hb_b"),
                _lv("rep"),
                _lv("geom"),
                _lv("cont"),
            )

        # ── gates ─────────────────────────────────────────────────────────────
        g_local = gates.get("local", 1.0)
        g_rep = gates.get("repulsion", 1.0)
        g_ss = gates.get("secondary", 1.0)
        g_pack = gates.get("packing", 1.0)
        logger.info(
            "  Gates:    local=%.3f  rep=%.3f  ss=%.3f  pack=%.3f",
            g_local,
            g_rep,
            g_ss,
            g_pack,
        )
        logger.info("─" * W)

        # ── term table ────────────────────────────────────────────────────────
        terms = [
            ("local", "local", g_local),
            ("repulsion", "repulsion", g_rep),
            ("secondary", "secondary", g_ss),
            ("packing", "packing", g_pack),
        ]
        # Constraints (coord + Rg) use packing gate — they live in PackingEnergy
        has_constraint = "constraint" in E_clean
        if has_constraint:
            terms.append(("constraint", "constraint", g_pack))

        weighted = {n: g * E_clean.get(n, 0.0) for n, _, g in terms}
        E_total_w = sum(weighted.values())

        # |E|% source: subterm-level from energy_balance_loss
        if term_absmeans:
            abs_local = term_absmeans.get("local_thetaphi", 0.0)
            if abs_local == 0.0:
                abs_local = (
                    term_absmeans.get("local_thetatheta", 0.0)
                    + term_absmeans.get("local_deltaphi", 0.0)
                    + term_absmeans.get("local_phiphi", 0.0)
                )
            abs_ss = term_absmeans.get("secondary", 0.0)
            if abs_ss == 0.0:
                abs_ss = (
                    term_absmeans.get("secondary_ram", 0.0)
                    + term_absmeans.get("secondary_hb_alpha", 0.0)
                    + term_absmeans.get("secondary_hb_beta", 0.0)
                )
                if abs_ss == 0.0:
                    # Legacy fallback
                    abs_ss = term_absmeans.get("secondary_thetaphi", 0.0) + term_absmeans.get("secondary_phiphi", 0.0)
            abs_rep = term_absmeans.get("repulsion", 0.0)
            abs_pack = term_absmeans.get("packing", 0.0)
            if abs_pack == 0.0:
                # Packing = learned subterm (contact only)
                abs_pack = term_absmeans.get("packing_contact", 0.0)
            abs_constr = (
                term_absmeans.get("packing_hp_penalty", term_absmeans.get("packing_coord", 0.0))
                + term_absmeans.get("packing_rho_penalty", 0.0)
                + term_absmeans.get("packing_rg", 0.0)
            )
            pct_sub = {
                "local": abs_local,
                "secondary": abs_ss,
                "repulsion": abs_rep,
                "packing": abs_pack,
            }
            if has_constraint:
                pct_sub["constraint"] = abs_constr
        else:
            pct_sub = {n: abs(v) for n, v in weighted.items()}
        total_abs_sub = max(sum(pct_sub.values()), 1e-12)

        # E% source: percentage of gate×E (signed energy contribution)
        # Uses abs(gate×E) for each term / sum(abs(gate×E)) — reflects the E/res column
        pct_term = {n: abs(v) for n, v in weighted.items()}
        total_abs_term = max(sum(pct_term.values()), 1e-12)

        # Compute per-term gaps at each sigma
        gaps_by_sigma: dict[float, dict[str, float]] = {}
        for sig in gap_sigmas:
            Ep = E_perts[sig]
            gaps_by_sigma[sig] = {n: g * (Ep.get(n, 0.0) - E_clean.get(n, 0.0)) for n, _, g in terms}

        # Total gap at each sigma + mean across sigmas
        total_gaps = {sig: sum(gd.values()) for sig, gd in gaps_by_sigma.items()}
        mean_gap_total = sum(total_gaps.values()) / max(len(total_gaps), 1)

        # Per-term mean gap across sigmas
        mean_gaps = {}
        for name, _, _ in terms:
            vals = [gaps_by_sigma[sig].get(name, 0.0) for sig in gap_sigmas]
            mean_gaps[name] = sum(vals) / max(len(vals), 1)

        # Subterm definitions: parent → [(display_name, key_in_E_clean), ...]
        subterms = {}
        if "local_thetaphi" in E_clean:
            subterms["local"] = [("  └ θφ", "local_thetaphi")]
        elif "local_thetatheta" in E_clean:
            subs = [("  ├ θθ", "local_thetatheta"), ("  ├ Δφ", "local_deltaphi")]
            if "local_phiphi" in E_clean:
                subs.append(("  └ φφ", "local_phiphi"))
            else:
                subs[-1] = ("  └ Δφ", "local_deltaphi")
            subterms["local"] = subs
        if "secondary_ram" in E_clean:
            subterms["secondary"] = [
                ("  ├ ram", "secondary_ram"),
                ("  ├ hb_α", "secondary_hb_alpha"),
                ("  └ hb_β", "secondary_hb_beta"),
            ]
        if "packing_contact" in E_clean:
            # v5: 2 balance subterms (hp_reward, rho_reward)
            if "packing_rho_reward" in E_clean:
                subterms["packing"] = [
                    ("  ├ hp_rew", "packing_hp_reward"),
                    ("  └ rho_rew", "packing_rho_reward"),
                ]
            else:
                # v4 fallback
                subterms["packing"] = [
                    ("  └ contact", "packing_contact"),
                ]
        if has_constraint:
            # v5: 3 constraints (hp_penalty, rho_penalty, rg_penalty)
            if "packing_rho_penalty" in E_clean:
                subterms["constraint"] = [
                    ("  ├ hp_pen", "packing_hp_penalty"),
                    ("  ├ rho_pen", "packing_rho_penalty"),
                    ("  └ Rg", "packing_rg"),
                ]
            else:
                # v4 fallback: 2 constraints (coord, Rg)
                subterms["constraint"] = [
                    ("  ├ coord", "packing_coord"),
                    ("  └ Rg", "packing_rg"),
                ]

        # Header with all sigma columns
        sig_hdr = "  ".join(f"@{s:.1f}r" for s in gap_sigmas)
        logger.info(
            "  %-14s  %9s  %9s  %5s  %5s  %s  mean",
            "term",
            "E/res",
            "gate×E",
            "E%",
            "|E|%",
            sig_hdr,
        )
        logger.info("  " + "·" * (62 + 7 * len(gap_sigmas)))

        def _subterm_gaps(key, gate, E_base):
            """Compute gap columns and mean for a subterm key."""
            cols = []
            for sig in gap_sigmas:
                Ep = E_perts[sig]
                cols.append(gate * (Ep.get(key, 0.0) - E_base))
            mean_val = sum(cols) / max(len(cols), 1)
            col_str = "  ".join(f"{v:+.3f}" for v in cols)
            return col_str, mean_val

        for name, key, g in terms:
            E = E_clean.get(key, 0.0)
            Ew = weighted[name]
            ps = 100.0 * pct_sub.get(name, abs(Ew)) / total_abs_sub
            pt = 100.0 * pct_term.get(name, abs(Ew)) / total_abs_term
            gap_cols = "  ".join(f"{gaps_by_sigma[sig].get(name, 0.0):+.3f}" for sig in gap_sigmas)
            mg = mean_gaps.get(name, 0.0)

            # Parent term row — E% then |E|%
            logger.info(
                "  %-14s  %+9.3f  %+9.3f  %4.1f%%  %4.1f%%  %s  %+.3f",
                name,
                E,
                Ew,
                pt,
                ps,
                gap_cols,
                mg,
            )

            # Subterm rows (if available)
            if name in subterms:
                for sub_label, sub_key in subterms[name]:
                    E_sub = E_clean.get(sub_key, 0.0)
                    sub_gap_str, sub_mg = _subterm_gaps(sub_key, g, E_sub)
                    logger.info(
                        "  %-14s  %+9.3f  %+9.3f  %5s  %5s  %s  %+.3f",
                        sub_label,
                        E_sub,
                        g * E_sub,
                        "",
                        "",
                        sub_gap_str,
                        sub_mg,
                    )

        logger.info("  " + "·" * (62 + 7 * len(gap_sigmas)))
        total_gap_cols = "  ".join(f"{total_gaps[sig]:+.3f}" for sig in gap_sigmas)
        logger.info(
            "  %-14s  %9s  %+9.3f  %6s  %6s  %s  %+.3f",
            "TOTAL",
            "",
            E_total_w,
            "",
            "",
            total_gap_cols,
            mean_gap_total,
        )

        # Store mean_gap_total on the logger for external access
        self.last_mean_gap = mean_gap_total

        # ── balance bounds ─────────────────────────────────────────────────────
        # 7 learned subterms in balance equalization (constraints excluded — analytical):
        #   local(1: θφ) + repulsion(1) + secondary(3: ram,hb_α,hb_β) + packing(2: hp_rew,rho_rew)
        if term_absmeans:
            N_sub = 7
            r_sub = 7.0

            def bounds_sub(k):
                mn = 100.0 * k / (k + (N_sub - k) * r_sub)
                mx = 100.0 * k * r_sub / (k * r_sub + (N_sub - k))
                return mn, mx

            bsub = {
                "local": bounds_sub(1),
                "secondary": bounds_sub(3),
                "repulsion": bounds_sub(1),
                "packing": bounds_sub(2),
            }
            parts_sub = "  ".join(f"{n}=[{mn:.0f}%,{mx:.0f}%]" for n, (mn, mx) in bsub.items())
            logger.info("  |E|%% bounds (subterm r=%.0f): %s", r_sub, parts_sub)

        # Term bounds (N=4 terms, r_term)
        r_t = balance_r_term
        N_term = 4

        def bounds_term(r_v):
            mn = 100.0 / (1 + (N_term - 1) * r_v)
            mx = 100.0 * r_v / (r_v + (N_term - 1))
            return mn, mx

        bt_mn, bt_mx = bounds_term(r_t)
        logger.info(
            "  E%%%% bounds (term r=%.0f): each in [%.1f%%,%.1f%%]",
            r_t,
            bt_mn,
            bt_mx,
        )

        logger.info("─" * W)

        # ── safety metrics ────────────────────────────────────────────────────
        if "safety" in precomputed:
            rep_m = precomputed["safety"]
        else:
            rep_m = self._repulsion_metrics(R, lengths=lengths)
        safety_msg = (
            f"  Safety:   min_dist={rep_m['min_dist']:.3f}Å"
            f"  <4.0={rep_m['frac_below_40']:.2%}"
            f"  <4.5={rep_m['frac_below_45']:.2%}"
        )
        if "max_force" in precomputed:
            max_f = precomputed["max_force"]
            clip_fr = precomputed.get("clip_frac", 0.0)
            safety_msg += f"  max|F|={max_f:.2f}  clip={clip_fr:.2%}"
            self.clip_fractions.append(clip_fr)
            self.max_forces.append(max_f)
        elif forces is not None:
            force_norms = torch.norm(forces, dim=-1)
            max_f = float(force_norms.max().item())
            clip_fr = float((force_norms > force_cap).float().mean().item())
            safety_msg += f"  max|F|={max_f:.2f}  clip={clip_fr:.2%}"
            self.clip_fractions.append(clip_fr)
            self.max_forces.append(max_f)
        logger.info(safety_msg)

        # ── geogap ────────────────────────────────────────────────────────────
        if geogap_diag is not None:
            gap_v = geogap_diag.get("gap", 0.0)
            e_clean = geogap_diag.get("E_clean", 0.0)
            e_pert = geogap_diag.get("E_perturbed", 0.0)
            status = "OK" if not geogap_diag.get("gap_active", False) else "FAIL"
            logger.info(
                "  Geogap:   gap=%.4f  E_clean=%.4f  E_perturb=%.4f  margin=%.2f  %s",
                gap_v,
                e_clean,
                e_pert,
                geogap_margin,
                status,
            )

        # ── balance (before Pack-C for readability) ─────────────────────────
        if loss_balance is not None and lambda_balance > 0.0:
            _absmeans = term_absmeans or {}
            if _absmeans:
                parts = "  ".join(f"{n}={v:.3f}" for n, v in _absmeans.items())
                logger.info(
                    "  Balance:  loss=%.4f  weighted=%.4f(x%.1f)  |E|: %s",
                    loss_balance,
                    lambda_balance * loss_balance,
                    lambda_balance,
                    parts,
                )
            else:
                logger.info(
                    "  Balance:  loss=%.4f  weighted=%.4f(x%.1f)",
                    loss_balance,
                    lambda_balance * loss_balance,
                    lambda_balance,
                )

        # ── total loss + DSM ───────────────────────────────────────────────
        if loss_dsm is not None:
            if dsm_diag and dsm_diag.get("n_samples", 1) > 1:
                _ds = dsm_diag.get("dsm_std", 0)
                _da = dsm_diag.get("dsm_alpha")
                _dm = dsm_diag.get("dsm_mixed")
                _da_s = f"{_da:.4f}" if _da is not None else "skip"
                _dm_s = f"{_dm:.4f}" if _dm is not None else "skip"
                logger.info(
                    "  Loss:     total=%.4f  dsm=%.4f (std=%.4f  α=%s  mix=%s)",
                    loss,
                    loss_dsm,
                    _ds,
                    _da_s,
                    _dm_s,
                )
            else:
                logger.info(
                    "  Loss:     total=%.4f  dsm=%.4f",
                    loss,
                    loss_dsm,
                )
        else:
            logger.info("  Loss:     total=%.4f", loss)

        # ── packing contrastive ───────────────────────────────────────────────
        if loss_pack_c is not None and lambda_pack_contrastive > 0.0:
            pack_mod = getattr(self.model, "packing", None)
            g_pack_diag = gates.get("packing", 1.0)
            if pack_mod is not None:
                try:
                    import math as _math
                    import random as _random

                    from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct

                    with torch.no_grad():
                        B_diag = R.shape[0]
                        t1 = B_diag // 3
                        t2 = (2 * B_diag) // 3
                        R1, R2, R3 = R[:t1], R[t1:t2], R[t2:]
                        seq1, seq2, seq3 = seq[:t1], seq[t1:t2], seq[t2:]
                        len1 = lengths[:t1] if lengths is not None else None
                        len2 = lengths[t1:t2] if lengths is not None else None
                        len3 = lengths[t2:] if lengths is not None else None

                        e_clean1 = float((g_pack_diag * pack_mod(R1, seq1, lengths=len1)).mean().item())
                        e_clean2 = float((g_pack_diag * pack_mod(R2, seq2, lengths=len2)).mean().item())
                        e_clean3 = float((g_pack_diag * pack_mod(R3, seq3, lengths=len3)).mean().item())
                        e_clean = (e_clean1 + e_clean2 + e_clean3) / 3.0

                        # Negative 1: sequence shuffle (first third)
                        shuffled_idx = torch.randperm(seq1.shape[0], device=seq1.device)
                        e_shuf = float((g_pack_diag * pack_mod(R1, seq1[shuffled_idx], lengths=len1)).mean().item())
                        gap_shuf = e_shuf - e_clean1
                        ok_shuf = "OK" if gap_shuf >= pack_contrastive_margin else "FAIL"

                        # Negative 2: IC noise (second third)
                        sigma = _math.exp(_random.uniform(_math.log(0.02), _math.log(0.30)))
                        theta, phi = coords_to_internal(R2)
                        anchor = extract_anchor(R2)
                        theta_n = (theta + 0.161 * sigma * torch.randn_like(theta)).clamp(0.01, _math.pi - 0.01)
                        phi_n = phi + sigma * torch.randn_like(phi)
                        phi_n = (phi_n + _math.pi) % (2 * _math.pi) - _math.pi
                        R_neg = nerf_reconstruct(theta_n, phi_n, anchor)
                        e_ic = float((g_pack_diag * pack_mod(R_neg, seq2, lengths=len2)).mean().item())
                        gap_ic = e_ic - e_clean2
                        ok_ic = "OK" if gap_ic >= pack_contrastive_margin else "FAIL"

                        # Negative 3: geometry shuffle (third third)
                        perm_idx = torch.randperm(R3.shape[0], device=R3.device)
                        e_pert = float(
                            (
                                g_pack_diag
                                * pack_mod(R3[perm_idx], seq3, lengths=len3[perm_idx] if len3 is not None else None)
                            )
                            .mean()
                            .item()
                        )
                        gap_pert = e_pert - e_clean3
                        ok_pert = "OK" if gap_pert >= pack_contrastive_margin else "FAIL"

                    logger.info(
                        "  Pack-C:   <E_clean>=%.4f  E_shuf=%.4f(gap=%.3f %s)  "
                        "E_ic=%.4f(gap=%.3f %s)  E_pert=%.4f(gap=%.3f %s)  margin=%.2f",
                        e_clean,
                        e_shuf,
                        gap_shuf,
                        ok_shuf,
                        e_ic,
                        gap_ic,
                        ok_ic,
                        e_pert,
                        gap_pert,
                        ok_pert,
                        pack_contrastive_margin,
                    )
                except Exception as e_diag:
                    logger.warning("Pack-C diagnostic error: %s", e_diag)

        # ── secondary basin ───────────────────────────────────────────────────
        if loss_basin is not None and lambda_basin > 0.0:
            logger.info(
                "  Basin:    loss=%.4f  weighted=%.4f(x%.1f)  margin=%.2f",
                loss_basin,
                lambda_basin * loss_basin,
                lambda_basin,
                basin_margin,
            )

        # ── native gap ────────────────────────────────────────────────────────
        if loss_native is not None and lambda_native > 0.0:
            if native_diag:
                diag_mode = native_diag.get("mode", "margin")
                if diag_mode == "continuous":
                    logger.info(
                        "  Native:   loss=%.4f  weighted=%.4f(x%.1f)  "
                        "E_native=%.4f  E_pert=%.4f  gap=%.3f(min=%.3f)  "
                        "sigma=%.3f  T=%.2f  [continuous]",
                        loss_native,
                        lambda_native * loss_native,
                        lambda_native,
                        native_diag.get("E_native", 0.0),
                        native_diag.get("E_pert", 0.0),
                        native_diag.get("gap_mean", 0.0),
                        native_diag.get("gap_min", 0.0),
                        native_diag.get("sigma", 0.0),
                        native_diag.get("T_mean", 0.0),
                    )
                else:
                    ok_str = "OK" if native_diag.get("ok", False) else "FAIL"
                    logger.info(
                        "  Native:   loss=%.4f  weighted=%.4f(x%.1f)  "
                        "E_native=%.4f  E_pert=%.4f  gap=%.3f(min=%.3f)  sigma=%.3f  margin=%.2f  %s",
                        loss_native,
                        lambda_native * loss_native,
                        lambda_native,
                        native_diag.get("E_native", 0.0),
                        native_diag.get("E_pert", 0.0),
                        native_diag.get("gap_mean", 0.0),
                        native_diag.get("gap_min", 0.0),
                        native_diag.get("sigma", 0.0),
                        native_margin,
                        ok_str,
                    )
            else:
                logger.info(
                    "  Native:   loss=%.4f  weighted=%.4f(x%.1f)",
                    loss_native,
                    lambda_native * loss_native,
                    lambda_native,
                )

        # ── ELT losses (Q-funnel + Z-score + Frustration) ────────────────────
        _any_elt = (
            loss_funnel is not None or loss_zscore is not None or loss_elt_gap is not None or loss_frust is not None
        )
        if _any_elt:
            parts_elt = []
            if loss_funnel is not None:
                parts_elt.append(f"funnel={loss_funnel:.4f}(x{lambda_funnel}={lambda_funnel*loss_funnel:.4f})")
            if loss_zscore is not None:
                parts_elt.append(f"zscore={loss_zscore:.4f}(x{lambda_zscore}={lambda_zscore*loss_zscore:.4f})")
            if loss_elt_gap is not None:
                parts_elt.append(f"gap={loss_elt_gap:.4f}(x{lambda_gap_elt}={lambda_gap_elt*loss_elt_gap:.4f})")
            if loss_frust is not None:
                parts_elt.append(f"frust={loss_frust:.4f}(x{lambda_frustration}={lambda_frustration*loss_frust:.4f})")
            logger.info("  ELT:      %s", "  ".join(parts_elt))

            if elt_diag:
                # Q-funnel diagnostics
                if "Q_native_mean" in elt_diag:
                    n_anti = elt_diag.get("n_anti_funnel", 0)
                    n_pairs = elt_diag.get("n_pairs", 1)
                    af_pct = 100.0 * n_anti / max(n_pairs, 1)
                    logger.info(
                        "  Q-funnel: Q_native=%.3f  Q_decoy=[%.3f,%.3f](mean=%.3f)  "
                        "slope_mean=%.3f  anti_funnel=%.0f%% (%d/%d)",
                        elt_diag.get("Q_native_mean", 0.0),
                        elt_diag.get("Q_decoy_min", 0.0),
                        elt_diag.get("Q_decoy_max", 0.0),
                        elt_diag.get("Q_decoy_mean", 0.0),
                        elt_diag.get("mean_slope", 0.0),
                        af_pct,
                        n_anti,
                        n_pairs,
                    )
                # Z-score diagnostics (only when loss is active)
                if "Z_mean" in elt_diag:
                    logger.info(
                        "  Z-score:  Z=%.2f [%.2f,%.2f]  target=%.1f  E_native/res=%.4f  E_decoy/res=%.4f",
                        elt_diag.get("Z_mean", 0.0),
                        elt_diag.get("Z_min", 0.0),
                        elt_diag.get("Z_max", 0.0),
                        target_zscore,
                        elt_diag.get("E_native_pr", 0.0),
                        elt_diag.get("E_decoy_pr", 0.0),
                    )
                # Frustration diagnostics (only when loss is active)
                if "f_mean" in elt_diag and lambda_frustration > 0:
                    logger.info(
                        "  Frust:    f=%.2f [%.2f,%.2f]  frac_frustrated=%.1f%%  " "E_native/res=%.4f  E_perm/res=%.4f",
                        elt_diag.get("f_mean", 0.0),
                        elt_diag.get("f_min", 0.0),
                        elt_diag.get("f_max", 0.0),
                        100 * elt_diag.get("frac_frustrated", 0.0),
                        elt_diag.get("E_native_pr", 0.0),
                        elt_diag.get("E_perm_pr", 0.0),
                    )

        # ── Native depth loss ──────────────────────────────────────────────
        if loss_native_depth is not None:
            _e_nat = e_native_depth if e_native_depth is not None else 0.0
            logger.info(
                "  Depth:    loss=%.4f(x%.1f=%.4f)  E_native/res=%.4f  target=%.2f",
                loss_native_depth,
                lambda_native_depth,
                lambda_native_depth * loss_native_depth,
                _e_nat,
                target_native_depth,
            )

        # ── EMA summary (smoothed trend — more meaningful than single-batch) ──
        ema = self.ema.get()
        if ema:
            # Line 1: core metrics
            line1 = []
            _core = [
                ("total", "loss", ".2f"),
                ("dsm", "DSM", ".3f"),
                ("dsm_std", "DSM_s", ".3f"),
                ("dsm_alpha", "DSM_α", ".3f"),
                ("dsm_mixed", "DSM_m", ".3f"),
                ("e_native", "E_nat", ".3f"),
                ("e_decoy", "E_dec", ".3f"),
                ("z_score", "Z̄", ".2f"),
                ("elt_gap", "gap", ".3f"),
                ("slope", "slope", ".3f"),
                ("anti_funnel", "af%", ".0f"),
                ("frac_negative", "neg%", ".0f"),
                ("basin_gap", "bgap", ".2f"),
                ("frust", "f̄", ".1f"),
                ("frac_frust", "ff%", ".1f"),
            ]
            for key, label, fmt in _core:
                if key in ema:
                    line1.append(f"{label}={ema[key]:{fmt}}")
            if line1:
                logger.info("  EMA(α=%.2f): %s", self.ema.alpha, "  ".join(line1))

            # Line 2: energy mix
            line2 = []
            _mix = [
                ("E%local", "E%loc", ".0f"),
                ("E%repulsion", "E%rep", ".0f"),
                ("E%secondary", "E%ss", ".0f"),
                ("E%packing", "E%pack", ".0f"),
            ]
            for key, label, fmt in _mix:
                if key in ema:
                    line2.append(f"{label}={ema[key]:{fmt}}")
            if line2:
                logger.info("             : %s", "  ".join(line2))

        # ── discrimination gaps (from decoys, passed via precomputed) ────────
        if "disc_gaps" in precomputed:
            disc = precomputed["disc_gaps"]
            disc_parts = [f"{k}={v:+.3f}" for k, v in disc.items() if k not in ("coord", "hp_pen", "rho_pen", "rg")]
            logger.info("  Disc:     %s", "  ".join(disc_parts))
            constr_parts = []
            # v5: separate hp_pen, rho_pen, rg
            for cn in ("hp_pen", "rho_pen", "rg"):
                if cn in disc:
                    constr_parts.append(f"{cn}={disc[cn]:+.4f}")
            # v4 fallback: coord, rg
            if not constr_parts:
                if "coord" in disc:
                    constr_parts.append(f"coord={disc['coord']:+.4f}")
                if "rg" in disc:
                    constr_parts.append(f"rg={disc['rg']:+.4f}")
            if constr_parts:
                logger.info("  DiscC:    %s", "  ".join(constr_parts))

        # ── funnel diagnostics (from decoys or training loss) ────────────────
        if "funnel" in precomputed:
            fn = precomputed["funnel"]
            logger.info(
                "  Funnel:   slope=%.3f  Q_af=%.1f%% (%d/%d)  dRMSD_af=%.1f%% (%d/%d)",
                fn.get("mean_slope", 0.0),
                fn.get("q_af", 0.0),
                fn.get("n_qf_anti", 0),
                fn.get("n_qf_pairs", 0),
                fn.get("drmsd_af", 0.0),
                fn.get("n_dr_anti", 0),
                fn.get("n_dr_pairs", 0),
            )

        # ── learnable buffer drift (only if any are active) ──────────────────
        drift_lines = buffer_drift_report(self.model)
        if len(drift_lines) > 1:  # more than "none active"
            logger.info("──────────────────────────────────────────────────────────────────")
            for dl in drift_lines:
                logger.info(dl)

        logger.info("═" * W)

    # ── legacy methods kept for any residual callers ───────────────────────────

    def log_diagnostics(self, phase_step, R, seq, forces=None, force_cap=50.0):
        """Deprecated one-liner — use log_step_block instead."""
        pass  # silenced; log_step_block now owns all diagnostic output

    def log_term_contributions(self, phase_step, R, seq, perturb_sigma=0.3):
        """Deprecated block logger — use log_step_block instead."""
        pass  # silenced; log_step_block now owns all diagnostic output

    # ── compute_energy_metrics kept for external callers ──────────────────────
    def compute_energy_metrics(self, R, seq, lengths=None):
        L = int(R.shape[1])
        metrics = {}
        for name in ("local", "repulsion", "secondary", "packing"):
            mod = getattr(self.model, name, None)
            if mod is not None:
                E = mod(R, seq, lengths=lengths)
                already = _is_internally_normalized(mod)
                metrics[f"{name}_per_res_like"] = float((E if already else E / float(max(L, 1))).mean().item())
                metrics[f"{name}_mean"] = float(E.mean().item())
        E_total = self.model(R, seq, lengths=lengths)
        metrics["total_model_units_mean"] = float(E_total.mean().item())
        return metrics

    # ── log_full_stage_diag removed (#32) ──────────────────────────────────
    # Both full_stage.py and self_consistent.py now use log_step_block
    # with disc/funnel data passed via precomputed dict.
