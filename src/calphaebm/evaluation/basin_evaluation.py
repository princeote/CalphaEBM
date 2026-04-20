"""Basin stability evaluation — engine, data classes, and CLI entry point.

Merged from basin.py (BasinStabilityEvaluator, StructureResult, BetaResult)
with the CLI runner (run_basin) and result saving (save_basin_results).
Shared model/data utilities imported from core_evaluation.py.

Python API:
    from calphaebm.evaluation.basin_evaluation import BasinStabilityEvaluator
    evaluator = BasinStabilityEvaluator(model, device)
    results = evaluator.sweep(val_loader, betas=[100.0])

CLI entry point:
    calphaebm evaluate --mode basin \
        --checkpoint checkpoints/run6/run6/stage1_round003/step_best.pt \
        --pdb val_hq.txt \
        --beta 100.0 --n-steps 10000 --n-samples 64 --sampler mala
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from calphaebm.evaluation.core_evaluation import load_model, load_structures, structures_to_loader
from calphaebm.evaluation.metrics.contacts import native_contact_set, q_smooth
from calphaebm.evaluation.metrics.rg import radius_of_gyration
from calphaebm.evaluation.metrics.rmsd import drmsd, rmsd_kabsch
from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.simulation.backends.langevin import ICLangevinSimulator
from calphaebm.utils.logging import get_logger

logger = get_logger()


@dataclass
class StructureResult:
    """Per-structure Langevin result with trajectory time series."""

    pdb_id: str
    chain_id: str
    length: int
    E_init: float  # energy at PDB coordinates
    E_minimized: float  # energy after minimization (= E_init if no minimization)
    E_final: float  # energy after Langevin
    E_delta: float  # E_final - E_minimized
    rmsd_min: float  # RMSD from PDB to minimized structure
    rmsd: float  # RMSD from minimized to final (Langevin drift)
    drmsd: float
    q: float
    rmsf: float = 0.0  # per-residue RMSF (mean over residues, Welford online)
    min_converged: bool = False  # did minimization converge?
    min_steps_used: int = 0  # steps actually used

    # Perturbation recovery tracking
    rmsd_start: float = 0.0  # RMSD of starting point from native (0 if no perturbation)
    q_start: float = 1.0  # Q at starting point (1.0 if native start)
    rmsd_to_native: float = 0.0  # final RMSD to native (for perturbed: did it recover?)
    q_to_native: float = 1.0  # final Q to native
    recovered: bool = False  # did RMSD to native decrease over the run?
    langevin_converged: bool = False  # did Langevin equilibrate early?
    langevin_steps_used: int = 0  # actual steps run (< n_steps if converged)

    # Trajectory time series (sampled every save_every steps)
    traj_steps: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_energy: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_rmsd: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_rmsd_to_native: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_drmsd: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_q: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_rg: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_grad_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_mean_rmsf: np.ndarray = field(default_factory=lambda: np.array([]))  # running mean RMSF at each snapshot

    # Per-component energy trajectories (sampled at same points as traj_energy)
    traj_E_local: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_E_repulsion: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_E_secondary: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_E_ram: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_E_hb_alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_E_hb_beta: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_E_packing: np.ndarray = field(default_factory=lambda: np.array([]))
    # Packing subterms
    traj_E_geom: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_E_contact: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_E_coord: np.ndarray = field(default_factory=lambda: np.array([]))
    traj_E_rg: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class BetaResult:
    """Aggregated results for a single beta value."""

    beta: float
    n_structures: int
    per_structure: List[StructureResult]

    # Energy delta stats
    E_delta_mean: float = 0.0
    E_delta_median: float = 0.0
    E_delta_p05: float = 0.0
    E_delta_p95: float = 0.0
    E_delta_neg_frac: float = 0.0

    # Structural stats
    rmsd_mean: float = 0.0
    rmsd_p05: float = 0.0
    rmsd_p95: float = 0.0
    drmsd_mean: float = 0.0
    q_mean: float = 0.0
    q_p05: float = 0.0
    rmsf_mean: float = 0.0

    is_stable: bool = False

    def compute_stats(self):
        """Compute aggregate stats from per-structure results."""
        if not self.per_structure:
            return
        deltas = np.array([s.E_delta for s in self.per_structure])
        rmsds = np.array([s.rmsd for s in self.per_structure])
        drmsds = np.array([s.drmsd for s in self.per_structure])
        qs = np.array([s.q for s in self.per_structure])
        rmsfs = np.array([s.rmsf for s in self.per_structure])

        self.E_delta_mean = float(np.mean(deltas))
        self.E_delta_median = float(np.median(deltas))
        self.E_delta_p05 = float(np.quantile(deltas, 0.05))
        self.E_delta_p95 = float(np.quantile(deltas, 0.95))
        self.E_delta_neg_frac = float(np.mean(deltas < 0))

        self.rmsd_mean = float(np.mean(rmsds))
        self.rmsd_p05 = float(np.quantile(rmsds, 0.05))
        self.rmsd_p95 = float(np.quantile(rmsds, 0.95))
        self.drmsd_mean = float(np.mean(drmsds))
        self.q_mean = float(np.mean(qs))
        self.q_p05 = float(np.quantile(qs, 0.05))
        self.rmsf_mean = float(np.mean(rmsfs))

        # Stability criteria:
        #   |E_delta| < 0.3  → thermal fluctuation around basin (STABLE)
        #   E_delta < -0.3   → found deeper basin, possibly non-native compaction
        #   E_delta > +0.3   → escaped basin (UNSTABLE)
        self.is_stable = (self.rmsd_mean < 5.0) and (abs(self.E_delta_mean) < 0.3)

    def summary_line(self) -> str:
        stable_str = "STABLE" if self.is_stable else "UNSTABLE"
        return (
            f"beta={self.beta:<5.1f}  E_delta={self.E_delta_mean:+.3f} "
            f"(med={self.E_delta_median:+.3f} neg={100*self.E_delta_neg_frac:.0f}%)  "
            f"RMSD={self.rmsd_mean:.2f}  dRMSD={self.drmsd_mean:.2f}  "
            f"Q={self.q_mean:.3f}  RMSF={self.rmsf_mean:.2f}  [{stable_str}]"
        )


class BasinStabilityEvaluator:
    """Evaluate basin stability by running IC Langevin at multiple temperatures.

    Reuses evaluation.metrics for structural analysis — no duplication.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        n_steps: int = 500,
        minimize_steps: int = 0,
        step_size: float = 1e-4,
        force_cap: float = 100.0,
        contact_cutoff: float = 8.0,
        exclude: int = 2,
        save_every: int = 50,
        perturb_sigma: float = 0.0,
        no_early_stop: bool = False,
        log_every: int = 1000,
    ):
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.minimize_steps = minimize_steps
        self.step_size = step_size
        self.force_cap = force_cap
        self.contact_cutoff = contact_cutoff
        self.exclude = exclude
        self.save_every = save_every
        self.perturb_sigma = perturb_sigma
        self.no_early_stop = no_early_stop
        self.log_every = log_every

    def _compute_components(
        self,
        R: torch.Tensor,
        seq: torch.Tensor,
        lengths: "torch.Tensor | None" = None,
    ) -> Dict[str, float]:
        """Compute per-term energy components (mirrors diagnostics._term_energies).

        Returns dict with keys: local, repulsion, secondary, packing,
        and packing subterms (packing_geom, packing_contact, packing_coord, packing_rg).
        All values are per-residue (E/res) floats.
        """
        out = {}

        local_mod = getattr(self.model, "local", None)
        if local_mod is not None:
            out["local"] = float(local_mod(R, seq, lengths=lengths).mean().item())

        rep_mod = getattr(self.model, "repulsion", None)
        if rep_mod is not None:
            out["repulsion"] = float(rep_mod(R, seq, lengths=lengths).mean().item())

        ss_mod = getattr(self.model, "secondary", None)
        if ss_mod is not None:
            if hasattr(ss_mod, "subterm_energies"):
                E_ram, E_hba, E_hbb = ss_mod.subterm_energies(R, seq, lengths=lengths)
                out["secondary_ram"] = float(E_ram.mean().item())
                out["secondary_hb_alpha"] = float(E_hba.mean().item())
                out["secondary_hb_beta"] = float(E_hbb.mean().item())
                out["secondary"] = out["secondary_ram"] + out["secondary_hb_alpha"] + out["secondary_hb_beta"]
            else:
                out["secondary"] = float(ss_mod(R, seq, lengths=lengths).mean().item())

        pack_mod = getattr(self.model, "packing", None)
        if pack_mod is not None:
            if hasattr(pack_mod, "subterm_energies"):
                E_geom, E_hp, E_coord, E_rg = pack_mod.subterm_energies(R, seq, lengths=lengths)
                out["packing_geom"] = float(E_geom.mean().item())
                out["packing_contact"] = float(E_hp.mean().item())
                out["packing_coord"] = float(E_coord.mean().item())
                out["packing_rg"] = float(E_rg.mean().item())
                out["packing"] = out["packing_geom"] + out["packing_contact"] + out["packing_coord"] + out["packing_rg"]
            else:
                out["packing"] = float(pack_mod(R, seq, lengths=lengths).mean().item())

        return out

    def _run_single(
        self,
        R_native: torch.Tensor,
        seq: torch.Tensor,
        beta: float,
        pdb_id: str = "?",
        chain_id: str = "?",
        length: int = 0,
    ) -> Optional[StructureResult]:
        """Run Langevin on one structure, optionally from a perturbed start.

        When perturb_sigma > 0, applies IC noise before dynamics. All RMSD/Q
        metrics are tracked relative to BOTH the starting point (perturbed or
        native) and the original native structure, enabling recovery analysis.
        """
        THETA_PHI_RATIO = 0.161

        R_b = R_native.unsqueeze(0).to(self.device)
        seq_b = seq.unsqueeze(0).to(self.device)
        L = length if length > 0 else int(R_native.shape[0])
        len_b = torch.tensor([L], dtype=torch.long, device=self.device)

        # Native reference (always the PDB structure)
        R_nat_np = R_native[:L].cpu().numpy()
        ni_nat, nj_nat, d0_nat = native_contact_set(R_nat_np, cutoff=self.contact_cutoff, exclude=self.exclude)

        with torch.no_grad():
            E_init = float(self.model(R_b, seq_b, lengths=len_b).item())

        # --- IC perturbation (if requested) ---
        R_start = R_b.clone()
        rmsd_start = 0.0
        q_start = 1.0
        if self.perturb_sigma > 0:
            try:
                with torch.no_grad():
                    theta, phi = coords_to_internal(R_b)
                    anchor = extract_anchor(R_b)
                    # Scale sigma so all chain lengths get similar Cartesian RMSD
                    # IC noise compounds as √L, so divide by √(L/50) to normalize
                    sigma = self.perturb_sigma / math.sqrt(L / 50.0)
                    noise_t = THETA_PHI_RATIO * sigma * torch.randn_like(theta)
                    noise_p = sigma * torch.randn_like(phi)
                    # Mask padding
                    if L < theta.shape[1] + 2:
                        idx_t = torch.arange(theta.shape[1], device=R_b.device)
                        idx_p = torch.arange(phi.shape[1], device=R_b.device)
                        noise_t[:, L - 2 :] = 0
                        noise_p[:, L - 3 :] = 0
                    theta_p = (theta + noise_t).clamp(0.01, math.pi - 0.01)
                    phi_p = phi + noise_p
                    phi_p = (phi_p + math.pi) % (2 * math.pi) - math.pi
                    R_start = nerf_reconstruct(theta_p, phi_p, anchor)
                R_start_np = R_start[0, :L].detach().cpu().numpy()
                rmsd_start = rmsd_kabsch(R_start_np, R_nat_np)
                q_start = q_smooth(R_start_np, ni_nat, nj_nat, d0_nat)
            except Exception as e:
                logger.debug("Perturbation failed for %s: %s", pdb_id, e)
                R_start = R_b.clone()

        # --- Energy minimization ---
        R_min = R_start.clone()
        E_minimized = E_init
        min_converged = False
        min_steps_used = 0
        if self.minimize_steps > 0:
            try:
                with torch.no_grad():
                    E_minimized = float(self.model(R_min, seq_b, lengths=len_b).item())
                min_sim = ICLangevinSimulator(
                    model=self.model,
                    seq=seq_b,
                    R_init=R_min,
                    step_size=self.step_size,
                    beta=1e6,
                    force_cap=self.force_cap,
                    lengths=len_b,
                )
                check_every = 20
                tol = 1e-4
                E_prev_check = E_minimized

                for step_i in range(1, self.minimize_steps + 1):
                    R_min, _, _ = min_sim.step()
                    min_steps_used = step_i
                    if step_i % check_every == 0:
                        R_min_det = R_min.detach()
                        with torch.no_grad():
                            E_now = float(self.model(R_min_det, seq_b, lengths=len_b).item())
                        rel_change = abs(E_now - E_prev_check) / (abs(E_prev_check) + 1e-10)
                        if rel_change < tol:
                            min_converged = True
                            break
                        E_prev_check = E_now

                R_min = R_min.detach()
                with torch.no_grad():
                    E_minimized = float(self.model(R_min, seq_b, lengths=len_b).item())
            except Exception as e:
                logger.debug("Minimization failed for %s (L=%d): %s", pdb_id, L, e)
                R_min = R_start.clone()

        # Reference for drift metrics: the structure we start Langevin from
        R_ref_np = R_min[0, :L].detach().cpu().numpy()
        rmsd_min = rmsd_kabsch(R_ref_np, R_nat_np)

        # Native contacts from NATIVE (not minimized/perturbed) for Q-to-native
        # Also compute contacts from the Langevin start for drift Q
        ni_ref, nj_ref, d0_ref = native_contact_set(R_ref_np, cutoff=self.contact_cutoff, exclude=self.exclude)

        # --- Langevin dynamics ---
        traj_steps = []
        traj_energy = []
        traj_rmsd = []  # RMSD to Langevin start (drift)
        traj_rmsd_to_native = []  # RMSD to native PDB (recovery)
        traj_drmsd = []
        traj_q = []  # Q relative to Langevin start
        traj_rg = []
        traj_grad_norm = []

        # Component energy trajectories
        traj_E_local = []
        traj_E_repulsion = []
        traj_E_secondary = []
        traj_E_ram = []
        traj_E_hb_alpha = []
        traj_E_hb_beta = []
        traj_E_packing = []
        traj_E_geom = []
        traj_E_contact = []
        traj_E_coord = []
        traj_E_rg_comp = []

        # Welford online RMSF computation (no superposition — valid for small drift)
        n_rmsf = 0
        mean_pos = np.zeros((L, 3), dtype=np.float64)
        M2_pos = np.zeros((L, 3), dtype=np.float64)
        traj_mean_rmsf_list = []

        try:
            sim = ICLangevinSimulator(
                model=self.model,
                seq=seq_b,
                R_init=R_min.detach(),
                step_size=self.step_size,
                beta=beta,
                force_cap=self.force_cap,
                lengths=len_b,
            )
            R_current = R_min.detach()
            converged_step = 0
            convergence_window = 5  # check over last 5 snapshots
            convergence_tol = 0.05  # RMSD change < 5% over window

            # Progress logging interval
            _rg_native = radius_of_gyration(R_nat_np)
            _log_interval = max(self.log_every, self.save_every)

            for step_i in range(1, self.n_steps + 1):
                R_current, E_step, info = sim.step()

                if step_i % self.save_every == 0 or step_i == self.n_steps:
                    R_snap = R_current[0, :L].detach().cpu().numpy()
                    traj_steps.append(step_i)
                    traj_energy.append(info.energy)
                    traj_rmsd.append(rmsd_kabsch(R_snap, R_ref_np))
                    traj_rmsd_to_native.append(rmsd_kabsch(R_snap, R_nat_np))
                    traj_drmsd.append(drmsd(R_snap, R_ref_np, mode="nonlocal", exclude=self.exclude))
                    traj_q.append(q_smooth(R_snap, ni_ref, nj_ref, d0_ref))
                    traj_rg.append(radius_of_gyration(R_snap))
                    traj_grad_norm.append(info.theta_grad_norm + info.phi_grad_norm)

                    # Component energies (no grad — diagnostic only)
                    with torch.no_grad():
                        _comps = self._compute_components(R_current, seq_b, lengths=len_b)
                    traj_E_local.append(_comps.get("local", 0.0))
                    traj_E_repulsion.append(_comps.get("repulsion", 0.0))
                    traj_E_secondary.append(_comps.get("secondary", 0.0))
                    traj_E_ram.append(_comps.get("secondary_ram", 0.0))
                    traj_E_hb_alpha.append(_comps.get("secondary_hb_alpha", 0.0))
                    traj_E_hb_beta.append(_comps.get("secondary_hb_beta", 0.0))
                    traj_E_packing.append(_comps.get("packing", 0.0))
                    traj_E_geom.append(_comps.get("packing_geom", 0.0))
                    traj_E_contact.append(_comps.get("packing_contact", 0.0))
                    traj_E_coord.append(_comps.get("packing_coord", 0.0))
                    traj_E_rg_comp.append(_comps.get("packing_rg", 0.0))

                    # Progress log
                    if step_i % _log_interval == 0 or step_i == self.n_steps:
                        _rg_pct = 100.0 * traj_rg[-1] / _rg_native if _rg_native > 0 else 0.0
                        _rmsf_val = traj_mean_rmsf_list[-1] if traj_mean_rmsf_list else 0.0

                        # Safety: min nonbonded distance and clash fraction
                        _dists = torch.cdist(R_current[0, :L], R_current[0, :L])
                        _mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=4)
                        _nb_dists = _dists[_mask]
                        _min_dist = float(_nb_dists.min().item()) if _nb_dists.numel() > 0 else 99.0
                        _n_nb = float(_nb_dists.numel())
                        _clash_4p0 = float((_nb_dists < 4.0).sum().item()) / max(_n_nb, 1) * 100
                        _clash_4p5 = float((_nb_dists < 4.5).sum().item()) / max(_n_nb, 1) * 100

                        logger.info(
                            "    [%s] step %6d/%d  E=%.3f  RMSD=%.2f  dRMSD=%.2f  "
                            "Q=%.3f  Rg=%.1f/%.1f(%d%%)  drift=%.2f  |g|=%.2f  RMSF=%.2f",
                            pdb_id,
                            step_i,
                            self.n_steps,
                            traj_energy[-1],
                            traj_rmsd_to_native[-1],
                            traj_drmsd[-1],
                            traj_q[-1],
                            traj_rg[-1],
                            _rg_native,
                            _rg_pct,
                            traj_rmsd[-1],
                            traj_grad_norm[-1],
                            _rmsf_val,
                        )
                        logger.info(
                            "           loc=%.3f rep=%.3f "
                            "ss=%.3f(ram=%.3f hba=%.3f hbb=%.3f) "
                            "pack=%.3f(g=%.3f c=%.3f co=%.3f rg=%.3f)",
                            traj_E_local[-1],
                            traj_E_repulsion[-1],
                            traj_E_secondary[-1],
                            traj_E_ram[-1],
                            traj_E_hb_alpha[-1],
                            traj_E_hb_beta[-1],
                            traj_E_packing[-1],
                            traj_E_geom[-1],
                            traj_E_contact[-1],
                            traj_E_coord[-1],
                            traj_E_rg_comp[-1],
                        )
                        # Safety: forces (from Langevin info if available)
                        _max_f = getattr(info, "max_force", None)
                        _clip_fr = getattr(info, "clip_frac", None) or getattr(info, "clip_fraction", None)
                        _safety_parts = [
                            f"min_dist={_min_dist:.3f}Å",
                            f"<4.0={_clash_4p0:.2f}%",
                            f"<4.5={_clash_4p5:.2f}%",
                        ]
                        if _max_f is not None:
                            _safety_parts.append(f"max|F|={_max_f:.2f}")
                        if _clip_fr is not None:
                            _safety_parts.append(f"clip={_clip_fr:.2f}%")
                        logger.info("           %s", "  ".join(_safety_parts))

                    # Welford RMSF update
                    n_rmsf += 1
                    delta_w = R_snap.astype(np.float64) - mean_pos
                    mean_pos += delta_w / n_rmsf
                    delta_w2 = R_snap.astype(np.float64) - mean_pos
                    M2_pos += delta_w * delta_w2

                    # Running mean RMSF at this snapshot
                    if n_rmsf > 1:
                        _var = M2_pos / (n_rmsf - 1)
                        _rmsf_per_res = np.sqrt(_var.sum(axis=1))
                        traj_mean_rmsf_list.append(float(_rmsf_per_res.mean()))
                    else:
                        traj_mean_rmsf_list.append(0.0)

                    # Convergence check: RMSD-to-native stable over window
                    n_snaps = len(traj_rmsd_to_native)
                    if n_snaps >= convergence_window + 1:
                        recent = traj_rmsd_to_native[-convergence_window:]
                        rmsd_range = max(recent) - min(recent)
                        mean_rmsd = np.mean(recent)
                        rel_change = rmsd_range / (mean_rmsd + 1e-10)
                        if rel_change < convergence_tol and n_snaps >= 10 and not self.no_early_stop:
                            converged_step = step_i
                            break

            R_final = R_current.detach()
        except Exception as e:
            logger.debug("Langevin failed for %s (L=%d): %s", pdb_id, L, e)
            return None

        with torch.no_grad():
            E_final = float(self.model(R_final, seq_b, lengths=len_b).item())

        # Final metrics
        R_fin_np = R_final[0, :L].cpu().numpy()
        rmsd_val = rmsd_kabsch(R_fin_np, R_ref_np)
        drmsd_val = drmsd(R_fin_np, R_ref_np, mode="nonlocal", exclude=self.exclude)
        q_val = q_smooth(R_fin_np, ni_ref, nj_ref, d0_ref)
        rmsd_to_native = rmsd_kabsch(R_fin_np, R_nat_np)
        q_to_native = q_smooth(R_fin_np, ni_nat, nj_nat, d0_nat)

        # Recovery check: did RMSD to native decrease?
        # Works for both Langevin trajectories and minimize-only mode
        recovered = False
        if len(traj_rmsd_to_native) >= 2:
            # Langevin mode: compare first and last trajectory snapshots
            recovered = traj_rmsd_to_native[-1] < traj_rmsd_to_native[0]
        elif rmsd_start > 0 and self.n_steps == 0:
            # Minimize-only mode: compare perturbed start to minimized result
            recovered = rmsd_to_native < rmsd_start

        # RMSF from Welford accumulators
        if n_rmsf > 1:
            variance = M2_pos / (n_rmsf - 1)  # (L, 3) per-residue per-axis variance
            per_residue_rmsf = np.sqrt(variance.sum(axis=1))  # sqrt(var_x + var_y + var_z)
            rmsf_val = float(per_residue_rmsf.mean())
        else:
            rmsf_val = 0.0

        return StructureResult(
            pdb_id=pdb_id,
            chain_id=chain_id,
            length=L,
            E_init=E_init,
            E_minimized=E_minimized,
            E_final=E_final,
            E_delta=E_final - E_minimized,
            rmsd_min=rmsd_min,
            rmsd=rmsd_val,
            drmsd=drmsd_val,
            q=q_val,
            rmsf=rmsf_val,
            min_converged=min_converged,
            min_steps_used=min_steps_used,
            rmsd_start=rmsd_start,
            q_start=q_start,
            rmsd_to_native=rmsd_to_native,
            q_to_native=q_to_native,
            recovered=recovered,
            langevin_converged=converged_step > 0,
            langevin_steps_used=converged_step if converged_step > 0 else self.n_steps,
            traj_steps=np.array(traj_steps),
            traj_energy=np.array(traj_energy),
            traj_rmsd=np.array(traj_rmsd),
            traj_rmsd_to_native=np.array(traj_rmsd_to_native),
            traj_drmsd=np.array(traj_drmsd),
            traj_q=np.array(traj_q),
            traj_rg=np.array(traj_rg),
            traj_grad_norm=np.array(traj_grad_norm),
            traj_mean_rmsf=np.array(traj_mean_rmsf_list),
            # Component energies
            traj_E_local=np.array(traj_E_local),
            traj_E_repulsion=np.array(traj_E_repulsion),
            traj_E_secondary=np.array(traj_E_secondary),
            traj_E_ram=np.array(traj_E_ram),
            traj_E_hb_alpha=np.array(traj_E_hb_alpha),
            traj_E_hb_beta=np.array(traj_E_hb_beta),
            traj_E_packing=np.array(traj_E_packing),
            traj_E_geom=np.array(traj_E_geom),
            traj_E_contact=np.array(traj_E_contact),
            traj_E_coord=np.array(traj_E_coord),
            traj_E_rg=np.array(traj_E_rg_comp),
        )

    def run_beta(
        self,
        structures: List[Tuple[torch.Tensor, torch.Tensor, str, str, int]],
        beta: float,
    ) -> BetaResult:
        """Run all structures at a single beta."""
        logger.info("  Running %d structures at beta=%.1f (%d Langevin steps)...", len(structures), beta, self.n_steps)

        results = []
        for idx, (R_nat, seq, pdb_id, chain_id, length) in enumerate(structures):
            r = self._run_single(R_nat, seq, beta, pdb_id=pdb_id, chain_id=chain_id, length=length)
            if r is not None:
                results.append(r)
            if (idx + 1) % 8 == 0 or idx == len(structures) - 1:
                logger.info("    %d/%d done (ok=%d)", idx + 1, len(structures), len(results))

        br = BetaResult(beta=beta, n_structures=len(results), per_structure=results)
        br.compute_stats()

        # Per-structure table
        if results:
            is_perturbed = self.perturb_sigma > 0
            has_min = self.minimize_steps > 0

            if is_perturbed:
                # Perturbed start: show recovery metrics
                logger.info("  " + "-" * 105)
                logger.info(
                    "  %-10s %3s  %4s  %7s  %7s  %7s  %6s  %6s  %8s  %s",
                    "pdb",
                    "ch",
                    "L",
                    "RMSD_0",
                    "RMSD_f",
                    "dRMSD",
                    "Q_0",
                    "Q_f",
                    "E_delta",
                    "recovery",
                )
                logger.info("  " + "-" * 105)
                for sr in sorted(results, key=lambda x: x.rmsd_to_native):
                    delta_rmsd = sr.rmsd_to_native - sr.rmsd_start
                    status = "RECOVERED" if sr.recovered else "drifted"
                    logger.info(
                        "  %-10s %3s  %4d  %7.2f  %7.2f  %7.2f  %6.3f  %6.3f  %+8.3f  %s (%+.1f)",
                        sr.pdb_id,
                        sr.chain_id,
                        sr.length,
                        sr.rmsd_start,
                        sr.rmsd_to_native,
                        sr.drmsd,
                        sr.q_start,
                        sr.q_to_native,
                        sr.E_delta,
                        status,
                        delta_rmsd,
                    )
                n_recovered = sum(1 for r in results if r.recovered)
                mean_start = np.mean([r.rmsd_start for r in results])
                mean_final = np.mean([r.rmsd_to_native for r in results])
                logger.info(
                    "  Recovery: %d/%d (%.0f%%)  mean RMSD: %.2f → %.2f (%+.2f)",
                    n_recovered,
                    len(results),
                    100 * n_recovered / len(results),
                    mean_start,
                    mean_final,
                    mean_final - mean_start,
                )

            elif has_min:
                logger.info("  " + "-" * 110)
                logger.info(
                    "  %-10s %3s  %4s  %8s  %7s  %8s  %7s  %7s  %6s  %6s  %s",
                    "pdb",
                    "ch",
                    "L",
                    "E_delta",
                    "E_drop",
                    "RMSD_min",
                    "RMSD",
                    "dRMSD",
                    "Q",
                    "RMSF",
                    "status",
                )
                logger.info("  " + "-" * 110)
                for sr in sorted(results, key=lambda x: x.E_delta):
                    if sr.E_delta < -0.3:
                        status = "DEEP_BASIN"
                    elif sr.E_delta > 0.3:
                        status = "unstable"
                    else:
                        status = "STABLE"
                    e_drop = sr.E_init - sr.E_minimized
                    logger.info(
                        "  %-10s %3s  %4d  %+8.3f  %+7.3f  %8.2f  %7.2f  %7.2f  %6.3f  %6.2f  %s",
                        sr.pdb_id,
                        sr.chain_id,
                        sr.length,
                        sr.E_delta,
                        e_drop,
                        sr.rmsd_min,
                        sr.rmsd,
                        sr.drmsd,
                        sr.q,
                        sr.rmsf,
                        status,
                    )
                drops = [sr.E_init - sr.E_minimized for sr in results]
                rmins = [sr.rmsd_min for sr in results]
                n_conv = sum(1 for sr in results if sr.min_converged)
                mean_steps = np.mean([sr.min_steps_used for sr in results])
                logger.info(
                    "  Minimization: mean E_drop=%.3f  mean RMSD_to_PDB=%.2fÅ  " "converged=%d/%d  mean_steps=%.0f",
                    np.mean(drops),
                    np.mean(rmins),
                    n_conv,
                    len(results),
                    mean_steps,
                )
            else:
                logger.info("  " + "-" * 90)
                logger.info(
                    "  %-10s %3s  %4s  %8s  %7s  %7s  %6s  %6s  %s",
                    "pdb",
                    "ch",
                    "L",
                    "E_delta",
                    "RMSD",
                    "dRMSD",
                    "Q",
                    "RMSF",
                    "status",
                )
                logger.info("  " + "-" * 90)
                for sr in sorted(results, key=lambda x: x.E_delta):
                    if sr.E_delta < -0.3:
                        status = "DEEP_BASIN"
                    elif sr.E_delta > 0.3:
                        status = "unstable"
                    else:
                        status = "STABLE"
                    logger.info(
                        "  %-10s %3s  %4d  %+8.3f  %7.2f  %7.2f  %6.3f  %6.2f  %s",
                        sr.pdb_id,
                        sr.chain_id,
                        sr.length,
                        sr.E_delta,
                        sr.rmsd,
                        sr.drmsd,
                        sr.q,
                        sr.rmsf,
                        status,
                    )

            # Length vs E_delta correlation
            if len(results) >= 4:
                lens = np.array([r.length for r in results], dtype=np.float64)
                eds = np.array([r.E_delta for r in results], dtype=np.float64)
                if np.std(lens) > 0 and np.std(eds) > 0:
                    corr = float(np.corrcoef(lens, eds)[0, 1])
                    logger.info(
                        "  Length vs E_delta: r=%.3f (%s)",
                        corr,
                        "longer → less stable"
                        if corr > 0.3
                        else "weak/none"
                        if abs(corr) <= 0.3
                        else "shorter → less stable",
                    )

            # --- Trajectory evolution summary ---
            traj_results = [r for r in results if len(r.traj_steps) > 0]
            if traj_results:
                steps = traj_results[0].traj_steps
                n_snaps = len(steps)

                all_E = np.array([r.traj_energy[:n_snaps] for r in traj_results if len(r.traj_energy) >= n_snaps])
                all_rmsd = np.array([r.traj_rmsd[:n_snaps] for r in traj_results if len(r.traj_rmsd) >= n_snaps])
                all_q = np.array([r.traj_q[:n_snaps] for r in traj_results if len(r.traj_q) >= n_snaps])
                all_rg = np.array([r.traj_rg[:n_snaps] for r in traj_results if len(r.traj_rg) >= n_snaps])
                all_rmsd_nat = np.array(
                    [r.traj_rmsd_to_native[:n_snaps] for r in traj_results if len(r.traj_rmsd_to_native) >= n_snaps]
                )

                if len(all_E) > 0:
                    milestones = [0, n_snaps // 4, n_snaps // 2, 3 * n_snaps // 4, n_snaps - 1]
                    milestones = sorted(set(max(0, min(m, n_snaps - 1)) for m in milestones))

                    if is_perturbed and len(all_rmsd_nat) > 0:
                        # Show both drift RMSD and RMSD-to-native (recovery)
                        logger.info("  " + "-" * 95)
                        logger.info(
                            "  Trajectory evolution (mean ± std over %d structures, σ=%.2f):",
                            len(all_E),
                            self.perturb_sigma,
                        )
                        logger.info(
                            "  %8s  %10s  %12s  %12s  %10s  %10s",
                            "step",
                            "Energy",
                            "RMSD_drift",
                            "RMSD_native",
                            "Q",
                            "Rg(Å)",
                        )
                        logger.info("  " + "-" * 95)
                        for mi in milestones:
                            logger.info(
                                "  %8d  %+7.3f±%.2f  %6.2f±%.2f  %6.2f±%.2f  %6.3f±%.3f  %6.2f±%.2f",
                                int(steps[mi]),
                                float(all_E[:, mi].mean()),
                                float(all_E[:, mi].std()),
                                float(all_rmsd[:, mi].mean()),
                                float(all_rmsd[:, mi].std()),
                                float(all_rmsd_nat[:, mi].mean()),
                                float(all_rmsd_nat[:, mi].std()),
                                float(all_q[:, mi].mean()),
                                float(all_q[:, mi].std()),
                                float(all_rg[:, mi].mean()),
                                float(all_rg[:, mi].std()),
                            )
                        logger.info("  " + "-" * 95)

                        # Recovery check: does RMSD to native decrease?
                        rmsd_nat_start = float(all_rmsd_nat[:, 0].mean())
                        rmsd_nat_end = float(all_rmsd_nat[:, -1].mean())
                        if rmsd_nat_end < rmsd_nat_start * 0.9:
                            logger.info(
                                "  ✓ BASIN RECOVERY: RMSD to native decreased (%.2f → %.2f) — landscape pulls toward native",
                                rmsd_nat_start,
                                rmsd_nat_end,
                            )
                        elif rmsd_nat_end < rmsd_nat_start * 1.1:
                            logger.info(
                                "  ~ RMSD to native stable (%.2f → %.2f) — near basin edge",
                                rmsd_nat_start,
                                rmsd_nat_end,
                            )
                        else:
                            logger.info(
                                "  ✗ RMSD to native increased (%.2f → %.2f) — no basin recovery",
                                rmsd_nat_start,
                                rmsd_nat_end,
                            )
                    else:
                        # Standard trajectory table (start from native)
                        logger.info("  " + "-" * 80)
                        logger.info("  Trajectory evolution (mean ± std over %d structures):", len(all_E))
                        logger.info("  %8s  %10s  %10s  %10s  %10s", "step", "Energy", "RMSD(Å)", "Q", "Rg(Å)")
                        logger.info("  " + "-" * 80)
                        for mi in milestones:
                            logger.info(
                                "  %8d  %+7.3f±%.2f  %6.2f±%.2f  %6.3f±%.3f  %6.2f±%.2f",
                                int(steps[mi]),
                                float(all_E[:, mi].mean()),
                                float(all_E[:, mi].std()),
                                float(all_rmsd[:, mi].mean()),
                                float(all_rmsd[:, mi].std()),
                                float(all_q[:, mi].mean()),
                                float(all_q[:, mi].std()),
                                float(all_rg[:, mi].mean()),
                                float(all_rg[:, mi].std()),
                            )
                        logger.info("  " + "-" * 80)

                        mid_idx = n_snaps // 2
                        q3_idx = 3 * n_snaps // 4
                        rmsd_mid = float(all_rmsd[:, mid_idx].mean())
                        rmsd_q3 = float(all_rmsd[:, q3_idx].mean())
                        rmsd_end = float(all_rmsd[:, -1].mean())
                        if rmsd_end > rmsd_mid * 1.1:
                            logger.info(
                                "  ⚠ RMSD still growing (mid=%.2f → q3=%.2f → end=%.2f) — not equilibrated",
                                rmsd_mid,
                                rmsd_q3,
                                rmsd_end,
                            )
                        else:
                            last_quarter = all_rmsd[:, q3_idx:]
                            lq_std = float(last_quarter.std())
                            logger.info(
                                "  ✓ RMSD plateaued (mid=%.2f → q3=%.2f → end=%.2f, last-quarter std=%.3f) — equilibrated",
                                rmsd_mid,
                                rmsd_q3,
                                rmsd_end,
                                lq_std,
                            )

                        # Early stopping status
                        if self.no_early_stop:
                            logger.info("  Early stopping: DISABLED (ran all %d steps)", self.n_steps)

            # --- Convergence summary ---
            n_converged = sum(1 for r in results if r.langevin_converged)
            if n_converged > 0:
                conv_steps = [r.langevin_steps_used for r in results if r.langevin_converged]
                logger.info(
                    "  Convergence: %d/%d equilibrated early (mean step=%d, saved %.0f%% compute)",
                    n_converged,
                    len(results),
                    int(np.mean(conv_steps)),
                    100 * (1 - np.mean(conv_steps) / self.n_steps),
                )

        return br

    def sweep(
        self,
        val_loader: DataLoader,
        betas: List[float],
        max_samples: int = 64,
    ) -> Dict[float, BetaResult]:
        """Run basin stability at multiple betas.

        Args:
            val_loader: DataLoader providing (R, seq, ...) batches.
            betas: List of inverse temperature values to test.
            max_samples: Max structures to evaluate.

        Returns:
            Dict mapping beta -> BetaResult.
        """
        # Collect structures with metadata
        structures = []
        for batch in val_loader:
            R_batch, seq_batch = batch[0], batch[1]
            pdb_ids = batch[2] if len(batch) > 2 else ["?"] * R_batch.shape[0]
            chain_ids = batch[3] if len(batch) > 3 else ["?"] * R_batch.shape[0]
            lengths_batch = batch[4] if len(batch) > 4 else None
            for i in range(R_batch.shape[0]):
                L = int(lengths_batch[i]) if lengths_batch is not None else int(R_batch.shape[1])
                structures.append((R_batch[i], seq_batch[i], pdb_ids[i], chain_ids[i], L))
                if len(structures) >= max_samples:
                    break
            if len(structures) >= max_samples:
                break

        real_lengths = [s[4] for s in structures]
        logger.info(
            "Collected %d structures for basin stability (L=%d-%d, mean=%d)",
            len(structures),
            min(real_lengths) if real_lengths else 0,
            max(real_lengths) if real_lengths else 0,
            int(np.mean(real_lengths)) if real_lengths else 0,
        )

        self.model.eval()

        if self.minimize_steps > 0:
            logger.info("Energy minimization: %d steps (β=1e6) before each Langevin run", self.minimize_steps)

        if self.perturb_sigma > 0:
            logger.info("IC perturbation: σ_base=%.2f, scaled by 1/√(L/50) per structure", self.perturb_sigma)
            logger.info(
                "  → L=50: σ=%.3f  L=100: σ=%.3f  L=200: σ=%.3f  (targets ~3-5Å RMSD for all lengths)",
                self.perturb_sigma,
                self.perturb_sigma / math.sqrt(100 / 50),
                self.perturb_sigma / math.sqrt(200 / 50),
            )

        results = {}
        for beta in sorted(betas):
            logger.info("\n" + "=" * 60)
            br = self.run_beta(structures, beta)
            results[beta] = br
            logger.info("  %s", br.summary_line())

        # Summary table
        logger.info("\n" + "=" * 80)
        logger.info("  BASIN STABILITY SWEEP")
        logger.info("=" * 80)
        logger.info(
            "  %-6s  %8s  %8s  %6s  %7s  %7s  %6s  %6s  %7s",
            "beta",
            "E_delta",
            "median",
            "neg%",
            "RMSD",
            "dRMSD",
            "Q",
            "RMSF",
            "stable",
        )
        logger.info("  " + "-" * 76)
        for beta in sorted(results.keys()):
            r = results[beta]
            logger.info(
                "  %-6.1f  %+8.3f  %+8.3f  %5.1f%%  %7.2f  %7.2f  %6.3f  %6.2f  %7s",
                beta,
                r.E_delta_mean,
                r.E_delta_median,
                100 * r.E_delta_neg_frac,
                r.rmsd_mean,
                r.drmsd_mean,
                r.q_mean,
                r.rmsf_mean,
                "YES" if r.is_stable else "no",
            )
        logger.info("=" * 80)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Result saving
# ─────────────────────────────────────────────────────────────────────────────


def save_basin_results(results: Dict[float, BetaResult], out_dir: Path) -> None:
    """Save BetaResult objects to JSON scalars + per-structure trajectory CSVs."""
    import csv as _csv

    from calphaebm.evaluation.io.writers import save_metrics_json

    out_dir.mkdir(parents=True, exist_ok=True)

    for beta, br in results.items():
        scalars = {
            "beta": beta,
            "n_structures": br.n_structures,
            "E_delta_mean": br.E_delta_mean,
            "E_delta_median": br.E_delta_median,
            "E_delta_p05": br.E_delta_p05,
            "E_delta_p95": br.E_delta_p95,
            "E_delta_neg_frac": br.E_delta_neg_frac,
            "rmsd_mean": br.rmsd_mean,
            "rmsd_p05": br.rmsd_p05,
            "rmsd_p95": br.rmsd_p95,
            "drmsd_mean": br.drmsd_mean,
            "q_mean": br.q_mean,
            "q_p05": br.q_p05,
            "rmsf_mean": br.rmsf_mean,
            "is_stable": br.is_stable,
        }
        save_metrics_json(scalars, out_dir / f"beta_{beta:.1f}.json")

        traj_dir = out_dir / f"trajectories_beta_{beta:.1f}"
        traj_dir.mkdir(parents=True, exist_ok=True)

        for sr in br.per_structure:
            if len(sr.traj_steps) == 0:
                continue
            traj_path = traj_dir / f"{sr.pdb_id}_{sr.chain_id}.csv"
            with open(traj_path, "w", newline="") as f:
                writer = _csv.writer(f)
                writer.writerow(
                    [
                        "step",
                        "energy",
                        "rmsd",
                        "rmsd_to_native",
                        "drmsd",
                        "q",
                        "rg",
                        "grad_norm",
                        "mean_rmsf",
                        "E_local",
                        "E_repulsion",
                        "E_secondary",
                        "E_ram",
                        "E_hb_alpha",
                        "E_hb_beta",
                        "E_packing",
                        "E_geom",
                        "E_contact",
                        "E_coord",
                        "E_rg",
                    ]
                )
                has_comp = len(sr.traj_E_local) == len(sr.traj_steps)
                for i in range(len(sr.traj_steps)):
                    rn = sr.traj_rmsd_to_native[i] if i < len(sr.traj_rmsd_to_native) else ""
                    mrf = sr.traj_mean_rmsf[i] if i < len(sr.traj_mean_rmsf) else ""
                    row = [
                        int(sr.traj_steps[i]),
                        f"{sr.traj_energy[i]:.4f}",
                        f"{sr.traj_rmsd[i]:.4f}",
                        f"{rn:.4f}" if isinstance(rn, float) else "",
                        f"{sr.traj_drmsd[i]:.4f}",
                        f"{sr.traj_q[i]:.4f}",
                        f"{sr.traj_rg[i]:.4f}",
                        f"{sr.traj_grad_norm[i]:.4f}",
                        f"{mrf:.4f}" if isinstance(mrf, float) else "",
                    ]
                    if has_comp:
                        row.extend(
                            [
                                f"{sr.traj_E_local[i]:.4f}",
                                f"{sr.traj_E_repulsion[i]:.4f}",
                                f"{sr.traj_E_secondary[i]:.4f}",
                                f"{sr.traj_E_ram[i]:.4f}",
                                f"{sr.traj_E_hb_alpha[i]:.4f}",
                                f"{sr.traj_E_hb_beta[i]:.4f}",
                                f"{sr.traj_E_packing[i]:.4f}",
                                f"{sr.traj_E_geom[i]:.4f}",
                                f"{sr.traj_E_contact[i]:.4f}",
                                f"{sr.traj_E_coord[i]:.4f}",
                                f"{sr.traj_E_rg[i]:.4f}",
                            ]
                        )
                    else:
                        row.extend([""] * 11)
                    writer.writerow(row)

            rmsf_per_res = getattr(sr, "rmsf_per_residue", [])
            if len(rmsf_per_res) > 0:
                rp = traj_dir / f"{sr.pdb_id}_{sr.chain_id}_rmsf.csv"
                with open(rp, "w", newline="") as f:
                    writer = _csv.writer(f)
                    writer.writerow(["residue_index", "rmsf_A"])
                    for ri, v in enumerate(rmsf_per_res):
                        writer.writerow([ri, f"{v:.4f}"])


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_basin(args) -> int:
    """Run basin stability evaluation (calphaebm evaluate --mode basin)."""
    import random as _random

    from calphaebm.cli.commands.train.data_utils import parse_pdb_arg
    from calphaebm.utils.seed import seed_all

    if not args.checkpoint:
        logger.error("--checkpoint is required for --mode basin")
        return 1
    if not args.pdb:
        logger.error("--pdb is required for --mode basin")
        return 1

    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Parse PDB IDs ─────────────────────────────────────────────────────────
    all_ids = parse_pdb_arg(args.pdb)
    logger.info("Parsed %d unique PDB entry IDs", len(all_ids))

    if len(all_ids) <= 20:
        eval_ids = all_ids
    else:
        ckpt_dir = Path(args.checkpoint).parent
        val_ids = None
        for sd in [ckpt_dir, ckpt_dir.parent, ckpt_dir.parent.parent]:
            vp = sd / "val_ids.txt"
            if vp.exists():
                raw = [l.strip() for l in vp.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
                if raw:
                    val_ids = list(dict.fromkeys(v.split("_")[0].upper() for v in raw if v))
                    logger.info("Loaded %d val IDs from %s", len(val_ids), vp)
                    break
        if not val_ids:
            logger.warning("No val_ids.txt — using 80/20 split (seed=42)")
            rng = _random.Random(42)
            shuffled = list(all_ids)
            rng.shuffle(shuffled)
            val_ids = shuffled[int(0.8 * len(shuffled)) :]
        eval_ids = val_ids

    # ── Load structures ───────────────────────────────────────────────────────
    structures = load_structures(
        pdb_source=eval_ids,
        cache_dir=args.cache_dir,
        n_samples=args.n_samples,
        max_len=args.max_len,
        min_len=args.min_len,
        processed_cache_dir=getattr(args, "processed_cache_dir", None),
    )
    logger.info("Loaded %d structures", len(structures))
    for i, (_, _, pid, cid, L) in enumerate(structures[: args.n_samples]):
        logger.info("  %d: %s chain %s  L=%d", i, pid, cid, L)

    val_loader = structures_to_loader(structures)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(Path(args.checkpoint), device)
    logger.info("Loaded checkpoint: %s", args.checkpoint)
    if hasattr(model, "get_gates"):
        g = model.get_gates()
        logger.info(
            "Gates: local=%.3f rep=%.3f ss=%.3f pack=%.3f",
            g.get("local", 1.0),
            g.get("repulsion", 1.0),
            g.get("secondary", 1.0),
            g.get("packing", 1.0),
        )

    # ── Run sweep ─────────────────────────────────────────────────────────────
    betas = sorted(args.beta) if isinstance(args.beta, list) else [args.beta]
    evaluator = BasinStabilityEvaluator(
        model=model,
        device=device,
        n_steps=args.n_steps,
        minimize_steps=args.minimize_steps,
        step_size=args.step_size,
        force_cap=args.force_cap,
        contact_cutoff=args.contact_cutoff,
        exclude=args.exclude,
        save_every=args.save_every,
        perturb_sigma=args.perturb_sigma,
        no_early_stop=args.no_early_stop,
        log_every=args.log_every,
    )
    results = evaluator.sweep(val_loader, betas=betas, max_samples=args.n_samples)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir or "basin_results")
    save_basin_results(results, out_dir)
    logger.info("Results saved to %s", out_dir)
    return 0
