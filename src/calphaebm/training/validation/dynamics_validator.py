"""Inline dynamics validator for monitoring landscape quality during training.

Runs short Langevin trajectories at β=100 on 1-3 small proteins and reports
basin stability metrics: RMSD, Q, RMSF, Rg, E_delta, and energy subterm
fractions. Designed to run every ~500 training steps in ~30-60 seconds.

The β=10 validation used previously was in the "melted" regime and could not
distinguish good from bad landscapes. β=100 is in the thermodynamic regime
where basin tightness, compaction, and subterm balance are all visible.

Key output: a single compact log line per validation, plus a trajectory
summary showing equilibration behavior.

Usage in training loop:
    dyn_val = DynamicsValidator(model, device, val_structures)
    # Every 500 steps:
    metrics = dyn_val.validate(step=global_step)
"""

from __future__ import annotations

import math

import numpy as np
import torch

from calphaebm.evaluation.metrics.contacts import native_contact_set, q_smooth
from calphaebm.evaluation.metrics.rmsd import kabsch_rotate, rmsd_kabsch
from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct
from calphaebm.simulation.backends.langevin import ICLangevinSimulator
from calphaebm.utils.logging import get_logger

from .base import BaseValidator

logger = get_logger()


def _radius_of_gyration(R: np.ndarray) -> float:
    """Radius of gyration from (L, 3) coordinates."""
    center = R.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((R - center) ** 2, axis=1))))


class DynamicsValidator(BaseValidator):
    """Validates landscape quality by running Langevin dynamics at β=100.

    Unlike GenerationValidator (which tests generation from native at β~1),
    this tests whether the energy landscape holds proteins near native at
    physiological temperature — the core requirement for a useful potential.

    Reports per-structure and aggregate metrics including RMSF (via Welford's
    online algorithm), Rg tracking, and energy subterm fractions.

    Args:
        model:       The energy model (must expose .local, .repulsion, .secondary, .packing).
        device:      Torch device.
        structures:  List of dicts with keys 'pdb_id', 'chain_id', 'coords' (L,3 numpy),
                     'seq' (L, numpy int). Typically 1-3 small proteins (e.g. crambin).
        beta:        Inverse temperature for Langevin dynamics. Default 100.
        n_steps:     Langevin steps per validation. Default 2000.
        step_size:   Langevin step size. Default 1e-4.
        minimize_steps: Energy minimization steps before dynamics. Default 200.
        force_cap:   Force clipping threshold. Default 100.0.
        save_every:  Snapshot interval for trajectory tracking. Default 50.
    """

    def __init__(
        self,
        model,
        device,
        structures: list,
        beta: float = 100.0,
        n_steps: int = 2000,
        step_size: float = 1e-4,
        minimize_steps: int = 200,
        force_cap: float = 100.0,
        save_every: int = 50,
    ):
        super().__init__(model, device)
        self.structures = structures
        self.beta = beta
        self.n_steps = n_steps
        self.step_size = step_size
        self.minimize_steps = minimize_steps
        self.force_cap = force_cap
        self.save_every = save_every

    # ------------------------------------------------------------------
    # Energy subterm fractions
    # ------------------------------------------------------------------

    def _energy_fractions(self, R, seq, lengths=None):
        """Compute fractional contribution of each energy subterm.

        Returns dict like {'local': 0.27, 'repulsion': 0.04, 'secondary': 0.24, 'packing': 0.45}.
        """
        fracs = {}
        abs_vals = {}
        total_abs = 0.0

        for name in ("local", "repulsion", "secondary", "packing"):
            term = getattr(self.model, name, None)
            if term is None:
                continue
            try:
                with torch.no_grad():
                    e = term(R, seq, lengths=lengths).mean().item()
                    ae = abs(e)
                    abs_vals[name] = ae
                    total_abs += ae
            except Exception:
                pass

        if total_abs > 0:
            for name, ae in abs_vals.items():
                fracs[name] = ae / total_abs

        return fracs

    # ------------------------------------------------------------------
    # Single-structure Langevin run
    # ------------------------------------------------------------------

    def _run_one(self, coords_np, seq_np, pdb_id, chain_id):
        """Run minimization + Langevin on a single structure. Returns metrics dict or None."""
        L = len(coords_np)

        # Prepare tensors
        R = torch.tensor(coords_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        seq = torch.tensor(seq_np, dtype=torch.long, device=self.device).unsqueeze(0)
        lengths = torch.tensor([L], dtype=torch.long, device=self.device)

        # Center
        R = R - R.mean(dim=1, keepdim=True)

        # Native reference
        R_nat_np = R[0, :L].detach().cpu().numpy()
        ni, nj, d0 = native_contact_set(R_nat_np)
        rg_native = _radius_of_gyration(R_nat_np)

        # Initial energy
        with torch.no_grad():
            E_init = float(self.model(R, seq, lengths=lengths).item())

        # Minimize
        R_min = R.detach().clone()
        if self.minimize_steps > 0:
            try:
                sim_min = ICLangevinSimulator(
                    model=self.model,
                    seq=seq,
                    R_init=R_min,
                    step_size=self.step_size,
                    beta=1e6,
                    force_cap=self.force_cap,
                    lengths=lengths,
                )
                for _ in range(self.minimize_steps):
                    R_min, _, _ = sim_min.step()
            except Exception as e:
                logger.debug("Minimization failed for %s: %s", pdb_id, e)
                return None

        R_min = R_min.detach()
        with torch.no_grad():
            E_min = float(self.model(R_min, seq, lengths=lengths).item())

        R_ref_np = R_min[0, :L].cpu().numpy()

        # Welford accumulators for RMSF
        welf_n = 0
        welf_mean = np.zeros((L, 3), dtype=np.float64)
        welf_M2 = np.zeros((L, 3), dtype=np.float64)

        # Trajectory tracking
        traj_rmsd = []
        traj_q = []
        traj_rg = []
        traj_energy = []

        # Langevin dynamics
        try:
            sim = ICLangevinSimulator(
                model=self.model,
                seq=seq,
                R_init=R_min,
                step_size=self.step_size,
                beta=self.beta,
                force_cap=self.force_cap,
                lengths=lengths,
            )
            R_current = R_min.detach()

            for step_i in range(1, self.n_steps + 1):
                R_current, _, info = sim.step()

                if step_i % self.save_every == 0 or step_i == self.n_steps:
                    R_snap = R_current[0, :L].detach().cpu().numpy()

                    traj_rmsd.append(rmsd_kabsch(R_snap, R_nat_np))
                    traj_q.append(q_smooth(R_snap, ni, nj, d0))
                    traj_rg.append(_radius_of_gyration(R_snap))
                    traj_energy.append(info.energy)

                    # Welford update (align to minimized reference)
                    R_aligned, _ = kabsch_rotate(R_snap, R_ref_np)
                    welf_n += 1
                    delta = R_aligned - welf_mean
                    welf_mean += delta / welf_n
                    delta2 = R_aligned - welf_mean
                    welf_M2 += delta * delta2

        except Exception as e:
            logger.debug("Langevin failed for %s: %s", pdb_id, e)
            return None

        # Final metrics
        R_fin_np = R_current[0, :L].detach().cpu().numpy()
        with torch.no_grad():
            E_final = float(self.model(R_current, seq, lengths=lengths).item())

        # RMSF from Welford
        if welf_n >= 2:
            variance = welf_M2 / (welf_n - 1)
            rmsf_per_res = np.sqrt(variance.sum(axis=1))
            mean_rmsf = float(np.mean(rmsf_per_res))
        else:
            mean_rmsf = 0.0

        # Energy fractions at final position
        e_fracs = self._energy_fractions(R_current, seq, lengths)

        return {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "L": L,
            "E_init": E_init,
            "E_min": E_min,
            "E_final": E_final,
            "E_delta": E_final - E_min,
            "rmsd": rmsd_kabsch(R_fin_np, R_nat_np),
            "q": q_smooth(R_fin_np, ni, nj, d0),
            "rg": _radius_of_gyration(R_fin_np),
            "rg_native": rg_native,
            "rg_ratio": _radius_of_gyration(R_fin_np) / max(rg_native, 1e-6),
            "mean_rmsf": mean_rmsf,
            "e_fracs": e_fracs,
            # Trajectory for trend detection
            "traj_rmsd": traj_rmsd,
            "traj_q": traj_q,
            "traj_rg": traj_rg,
            "traj_energy": traj_energy,
        }

    # ------------------------------------------------------------------
    # Main validate
    # ------------------------------------------------------------------

    def validate(self, step=None) -> dict:
        """Run dynamics validation on all structures. Returns aggregate metrics dict."""
        self.model.eval()
        results = []

        for struct in self.structures:
            r = self._run_one(
                struct["coords"],
                struct["seq"],
                struct.get("pdb_id", "?"),
                struct.get("chain_id", "?"),
            )
            if r is not None:
                results.append(r)

        if not results:
            logger.warning("[dynamics] No structures completed at step %s", step)
            return {"valid": False}

        # Aggregate
        n = len(results)
        mean_rmsd = float(np.mean([r["rmsd"] for r in results]))
        mean_q = float(np.mean([r["q"] for r in results]))
        mean_rmsf = float(np.mean([r["mean_rmsf"] for r in results]))
        mean_rg = float(np.mean([r["rg"] for r in results]))
        mean_rg_native = float(np.mean([r["rg_native"] for r in results]))
        mean_rg_ratio = float(np.mean([r["rg_ratio"] for r in results]))
        mean_edelta = float(np.mean([r["E_delta"] for r in results]))
        n_stable = sum(1 for r in results if r["E_delta"] < 0)

        # Mean energy fractions
        frac_keys = set()
        for r in results:
            frac_keys.update(r["e_fracs"].keys())
        mean_fracs = {}
        for k in sorted(frac_keys):
            vals = [r["e_fracs"].get(k, 0.0) for r in results]
            mean_fracs[k] = float(np.mean(vals))

        # Compact one-line summary
        stable_str = "STABLE" if n_stable == n else f"{n_stable}/{n}"
        frac_str = "  ".join(f"{k}={100*v:.0f}%" for k, v in mean_fracs.items())

        logger.info(
            "[dynamics β=%g step=%s] RMSD=%.2f  Q=%.3f  RMSF=%.2f  " "Rg=%.1f/%.1f(%.0f%%)  E_delta=%+.3f  [%s]  %s",
            self.beta,
            step if step is not None else "?",
            mean_rmsd,
            mean_q,
            mean_rmsf,
            mean_rg,
            mean_rg_native,
            100 * mean_rg_ratio,
            mean_edelta,
            stable_str,
            frac_str,
        )

        # Per-structure detail (compact)
        for r in results:
            status = "ok" if r["E_delta"] < 0 else "BAD"
            logger.info(
                "  %s/%s L=%d: RMSD=%.2f Q=%.3f RMSF=%.2f Rg=%.1f(%.0f%%) E_delta=%+.3f [%s]",
                r["pdb_id"],
                r["chain_id"],
                r["L"],
                r["rmsd"],
                r["q"],
                r["mean_rmsf"],
                r["rg"],
                100 * r["rg_ratio"],
                r["E_delta"],
                status,
            )

        # Trajectory trend (first structure only, compact)
        r0 = results[0]
        n_traj = len(r0["traj_rmsd"])
        if n_traj >= 4:
            q1 = n_traj // 4
            q3 = 3 * n_traj // 4
            rmsd_early = np.mean(r0["traj_rmsd"][:q1])
            rmsd_mid = np.mean(r0["traj_rmsd"][q1:q3])
            rmsd_late = np.mean(r0["traj_rmsd"][q3:])
            rg_late = np.mean(r0["traj_rg"][q3:])
            growing = "↑growing" if rmsd_late > rmsd_mid * 1.05 else "→plateau"
            logger.info(
                "  trajectory(%s): RMSD %.1f→%.1f→%.1f %s  Rg_late=%.1f",
                r0["pdb_id"],
                rmsd_early,
                rmsd_mid,
                rmsd_late,
                growing,
                rg_late,
            )

        metrics = {
            "valid": True,
            "dynamics_rmsd": mean_rmsd,
            "dynamics_q": mean_q,
            "dynamics_rmsf": mean_rmsf,
            "dynamics_rg": mean_rg,
            "dynamics_rg_native": mean_rg_native,
            "dynamics_rg_ratio": mean_rg_ratio,
            "dynamics_E_delta": mean_edelta,
            "dynamics_n_stable": n_stable,
            "dynamics_n_total": n,
            "dynamics_beta": self.beta,
        }
        # Add subterm fractions
        for k, v in mean_fracs.items():
            metrics[f"dynamics_frac_{k}"] = v

        self.model.train()
        return metrics

    # ------------------------------------------------------------------
    # Factory: create from PDB IDs
    # ------------------------------------------------------------------

    @classmethod
    def from_pdb_ids(
        cls,
        model,
        device,
        pdb_ids: list,
        cache_dir: str = "./pdb_cache",
        min_len: int = 40,
        max_len: int = 100,
        **kwargs,
    ):
        """Create DynamicsValidator by loading structures from PDB IDs.

        Args:
            model:     The energy model.
            device:    Torch device.
            pdb_ids:   List of PDB IDs (e.g. ['1crn', '2ci2']).
            cache_dir: PDB cache directory.
            min_len/max_len: Length filters.
            **kwargs:  Passed to DynamicsValidator.__init__.
        """
        from calphaebm.data.pdb_chain_dataset import PDBChainDataset

        dataset = PDBChainDataset(
            pdb_ids=pdb_ids,
            cache_dir=cache_dir,
            min_len=min_len,
            max_len=max_len,
        )

        structures = []
        for i in range(len(dataset)):
            coords, seq, pdb_id, chain_id = dataset[i]
            structures.append(
                {
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "coords": coords.numpy(),
                    "seq": seq.numpy(),
                }
            )

        if not structures:
            logger.warning("DynamicsValidator: no structures loaded from %s", pdb_ids)

        return cls(model=model, device=device, structures=structures, **kwargs)
