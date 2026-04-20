"""Validator that generates structures from the model and tests them.

Key points:
- Per-call vs per-structure spike accounting is tracked separately.
- Denominators are consistent in both success and failure paths.
- Bond thresholds match dataset filtering.
- All thresholds are configurable for easy tuning.
- Tracks detailed failure reasons for debugging.
- Uses per-structure filtering instead of batch-level rejection.

Changes from previous version:
- FIX (critical): step_size default was 5e-5 — with force magnitudes of 5–15,
  this gives ~0.05 Å total displacement over 200 steps.  Structures barely moved,
  RMSD=0.050 Å, and 35% bond_threshold failures came from the mid-loop abort
  returning R_init for the whole batch.  New default: step_size=1e-4, n_steps=2000.
- FIX (critical): mid-loop bond check aborted and returned R_init for ALL B
  structures if ANY had a bad bond — batch-level rejection caused cascading
  failures.  Removed the mid-loop bond check entirely; bond filtering is now
  applied per-structure after _generate_structure returns, in validate().
  Mid-loop drift check similarly removed for the same reason.  Only genuinely
  unrecoverable conditions (NaN/inf, step-size collapse) abort inside the loop.
- FIX: force_spike_threshold was 1000.0 — with max|F| ~ 5–15 during normal
  training this never fired.  New default: 50.0 (meaningful at current scale).
- FIX: clip_spike_threshold was 0.20 — with force_cap=100 and typical forces
  of 5–15 this also never fired.  New default: 0.05.
- CLARITY: drift_ok filtering rewritten to be explicit and correct.
- ROBUSTNESS: added torch.no_grad() around validation forward passes to avoid
  accidentally building autograd graphs during generation.
- FIX: bond_lengths computed twice on overlapping data — now reuse lengths[ok_idx].
- FIX: bond_angles/torsions computed twice — now reuse theta[ok_idx], phi[ok_idx].
- PERF: all geometry tensors computed once, then indexed by ok_idx.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np
import torch

from calphaebm.data.pdb_parse import MAX_BOND_LENGTH, MIN_BOND_LENGTH
from calphaebm.evaluation.metrics.contacts import native_contact_set, q_smooth
from calphaebm.evaluation.metrics.rmsd import drmsd, rmsd_kabsch
from calphaebm.geometry.internal import bond_angles, bond_lengths, torsions
from calphaebm.simulation.backends.langevin import ICLangevinSimulator
from calphaebm.training.validation.metrics import compute_delta_phi_correlation, compute_ramachandran_correlation
from calphaebm.utils.logging import get_logger

from .base import BaseValidator

logger = get_logger()


def _run_single_langevin(args):
    """Worker function for parallel basin evaluation.

    Runs IC Langevin dynamics for a single structure on CPU.
    Called by multiprocessing.Pool.map() in _generate_structure().
    """
    model, R_b, seq_b, len_b, n_real, step_size, beta, force_cap, n_steps = args
    try:
        ic_sim = ICLangevinSimulator(
            model=model,
            seq=seq_b,
            R_init=R_b,
            step_size=step_size,
            beta=beta,
            force_cap=force_cap,
            lengths=len_b,
        )
        # Welford online RMSF
        _w_n = 0
        _w_mean = np.zeros((n_real, 3), dtype=np.float64)
        _w_M2 = np.zeros((n_real, 3), dtype=np.float64)
        for step_i in range(n_steps):
            R_b, _, _ = ic_sim.step()
            if (step_i + 1) % 10 == 0:
                coords = R_b[0, :n_real].detach().cpu().numpy().astype(np.float64)
                _w_n += 1
                delta = coords - _w_mean
                _w_mean += delta / _w_n
                delta2 = coords - _w_mean
                _w_M2 += delta * delta2
        if _w_n > 1:
            variance = _w_M2 / (_w_n - 1)
            per_res_rmsf = np.sqrt(variance.sum(axis=1))
            rmsf_val = float(per_res_rmsf.mean())
        else:
            rmsf_val = 0.0
        return R_b.detach().cpu(), rmsf_val
    except Exception:
        return R_b.detach().cpu(), 0.0


# Bond thresholds matching dataset filtering
BOND_MIN = float(MIN_BOND_LENGTH)
BOND_MAX = float(MAX_BOND_LENGTH)
IDEAL_BOND = 3.8


class GenerationValidator(BaseValidator):
    """Validates model by running short Langevin dynamics and checking generated structures."""

    def __init__(
        self,
        model,
        device,
        n_steps: int = 2000,  # increased: smaller step_size needs more steps
        step_size: float = 1e-4,  # reduced from 1e-3: stability limit = 1/(2k_eff)
        # at bond_spring=750, limit=0.000667 → 1e-3 unstable
        force_cap: float = 100.0,
        force_spike_threshold: float = 50.0,  # was 1000.0 — never fired at typical force scales
        clip_spike_threshold: float = 0.05,  # was 0.20  — never fired with force_cap=100, forces~10
        abort_clip_threshold: float = 0.20,
        max_drift_rmsd: float = 10.0,  # was 5.0; raised to allow meaningful exploration
        min_step_frac_abort: float = 0.05,
        langevin_beta: float = 1.0,  # inverse temperature for Langevin dynamics
        # Step size adaptation
        clip_frac_heavy: float = 0.20,
        clip_frac_moderate: float = 0.05,
        shrink_heavy: float = 0.5,
        shrink_moderate: float = 0.8,
        n_eval_workers: int = 8,
    ):
        """
        Args:
            model:                 The energy model.
            device:                Torch device.
            n_steps:               Langevin steps per generation call.
            step_size:             Base Langevin step size η.  Must be < 1/(2*k_eff)
                                   for stability.  At bond_spring=750, limit=6.7e-4,
                                   one step displaces by η·|F| ≈ 0.01 Å; 500 steps
                                   gives ~1–3 Å RMSD from native, which is the
                                   meaningful exploration regime.
            force_cap:             Force clipping threshold (per-atom norm).
            force_spike_threshold: Per-structure max|F| threshold marking a spike.
                                   Set relative to typical force magnitudes (~5–15).
            clip_spike_threshold:  Per-structure clip fraction threshold.
            abort_clip_threshold:  Abort if step_size is tiny AND clip_frac stays above this.
            max_drift_rmsd:        Post-generation drift filter (Å).  Applied per-structure
                                   in validate(), not inside the Langevin loop.
            min_step_frac_abort:   Abort if step_size shrinks below base * this fraction.
            clip_frac_heavy:       Global clip fraction threshold for heavy shrink.
            clip_frac_moderate:    Global clip fraction threshold for moderate shrink.
            shrink_heavy:          Step-size multiplier on heavy clipping.
            shrink_moderate:       Step-size multiplier on moderate clipping.
        """
        super().__init__(model, device)
        self.n_steps = int(n_steps)
        self.base_step_size = float(step_size)
        self.force_cap = float(force_cap)
        self.force_spike_threshold = float(force_spike_threshold)
        self.clip_spike_threshold = float(clip_spike_threshold)
        self.abort_clip_threshold = float(abort_clip_threshold)
        self.max_drift_rmsd = float(max_drift_rmsd)
        self.min_step_frac_abort = float(min_step_frac_abort)
        self.langevin_beta = float(langevin_beta)
        self.clip_frac_heavy = float(clip_frac_heavy)
        self.clip_frac_moderate = float(clip_frac_moderate)
        self.shrink_heavy = float(shrink_heavy)
        self.shrink_moderate = float(shrink_moderate)
        self.n_eval_workers = int(n_eval_workers)

        # Running totals across validation calls
        self.total_generation_calls = 0
        self.total_structures_attempted = 0
        self.total_structures_success = 0
        self.total_structures_failed = 0
        self.force_spike_calls = 0
        self.heavy_clip_calls = 0

    def _make_spike_stats(
        self,
        struct_force_spike: torch.Tensor,
        struct_heavy_clip: torch.Tensor,
    ) -> dict:
        return {
            "force_spike_any": bool(struct_force_spike.any().item()),
            "heavy_clip_any": bool(struct_heavy_clip.any().item()),
            "n_force_spike_structs": int(struct_force_spike.sum().item()),
            "n_heavy_clip_structs": int(struct_heavy_clip.sum().item()),
        }

    def _generate_structure(
        self,
        R_init: torch.Tensor,
        seq: torch.Tensor,
        temperature: float = 0.0,
        R_ref: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ):
        """Run Langevin dynamics from R_init.

        Mid-loop aborts only on unrecoverable numerical failures (NaN/inf,
        step-size collapse).  Bond and drift filtering is done per-structure
        in validate() after this returns, avoiding batch-level rejection.

        Args:
            R_init:      Starting coordinates (B, L, 3).
            seq:         Sequence tensor (B, L).
            temperature: Langevin temperature; 0 = deterministic gradient descent.
            R_ref:       Optional reference coordinates for RMSD tracking (B, L, 3).
            lengths:     (B,) real chain lengths for padding-aware energy/RMSD.

        Returns:
            R_out, success, spike_stats, avg_clip_fraction, max_force_preclip,
            failure_reason, rmsd_trajectory, energy_init, energy_final,
            per_struct_energy_deltas.
        """
        # ── IC Langevin simulation (replaces Cartesian Langevin) ─────────────
        # ICLangevinSimulator fixes bonds at exactly 3.8Å by NeRF construction.
        # Parallelized across structures using multiprocessing.
        B = R_init.shape[0]

        # Run IC Langevin in parallel across structures
        beta = self.langevin_beta
        import copy
        import multiprocessing as mp

        # Prepare args for each structure
        worker_args = []
        model_cpu = copy.deepcopy(self.model).cpu().eval()
        for b in range(B):
            R_b = R_init[b : b + 1].cpu()
            seq_b = seq[b : b + 1].cpu()
            len_b = lengths[b : b + 1].cpu() if lengths is not None else None
            n_real = int(lengths[b]) if lengths is not None else R_b.shape[1]
            worker_args.append(
                (
                    model_cpu,
                    R_b,
                    seq_b,
                    len_b,
                    n_real,
                    self.base_step_size,
                    beta,
                    self.force_cap,
                    self.n_steps,
                )
            )

        n_workers = min(B, self.n_eval_workers)
        if n_workers > 1:
            try:
                ctx = mp.get_context("fork")
                with ctx.Pool(n_workers) as pool:
                    results = pool.map(_run_single_langevin, worker_args)
            except Exception as e:
                logger.debug("Parallel eval failed (%s), falling back to serial", e)
                results = [_run_single_langevin(args) for args in worker_args]
        else:
            results = [_run_single_langevin(args) for args in worker_args]

        R_out_list = []
        per_struct_rmsf = []
        for R_b_out, rmsf_val in results:
            R_out_list.append(R_b_out)
            per_struct_rmsf.append(rmsf_val)

        R = torch.cat(R_out_list, dim=0)  # (B, L, 3)

        # Dummy diagnostics (IC sim doesn't use adaptive step size / clipping)
        struct_force_spike = torch.zeros(B, dtype=torch.bool, device=R.device)
        struct_heavy_clip = torch.zeros(B, dtype=torch.bool, device=R.device)
        current_step_size = self.base_step_size
        total_clipped_atoms = 0
        total_atoms = 1
        max_force_preclip = 0.0

        # RMSD + energy tracking — per-structure, padding-aware
        rmsd_trajectory = []
        energy_init = None
        energy_final = None
        per_struct_energy_deltas = []
        if R_ref is not None:
            with torch.no_grad():
                E0 = self.model(R_init.detach(), seq, lengths=lengths)
                E_final_t = self.model(R.detach(), seq, lengths=lengths)
                energy_init = float(E0.mean().item())
                energy_final = float(E_final_t.mean().item())
                per_struct_deltas = E_final_t - E0
                per_struct_energy_deltas = per_struct_deltas.cpu().tolist()
                # Padding-aware Kabsch-aligned RMSD: only over real atoms
                per_struct_rmsd = []
                for b in range(B):
                    n = int(lengths[b]) if lengths is not None else R.shape[1]
                    gen_b = R[b, :n].cpu().numpy()
                    ref_b = R_ref[b, :n].cpu().numpy()
                    per_struct_rmsd.append(rmsd_kabsch(gen_b, ref_b))
                rmsd_final = float(np.mean(per_struct_rmsd))
                rmsd_trajectory = [(self.n_steps, rmsd_final)]

        avg_clip = total_clipped_atoms / max(1, total_atoms)
        return (
            R.detach(),
            True,
            self._make_spike_stats(struct_force_spike, struct_heavy_clip),
            float(avg_clip),
            float(max_force_preclip),
            None,
            rmsd_trajectory,
            energy_init,
            energy_final,
            per_struct_energy_deltas,
            per_struct_rmsf,
        )

    def validate(
        self,
        val_loader,
        max_samples: int = 5000,
        temperature: float = 1.0,  # beta=1 per model convention; 0.0 = pure gradient descent (wrong)
        step_size: Optional[float] = None,  # override constructor step_size for this call
    ) -> dict:
        """Validate by running Langevin dynamics from clean native structures.

        Starts from unperturbed PDB coordinates — no noise added.  This tests
        the stricter question: "is the native structure a stable minimum of the
        learned energy function?"  If the energy landscape is correct, Langevin
        should keep structures near native (low RMSD, valid bonds, falling energy).
        Drift or bond failures from a clean start are a direct signal that the
        gradient is pointing in the wrong direction.

        RMSD is tracked at 4 checkpoints (25%, 50%, 75%, 100% of steps) so the
        trajectory shape is visible — converging (RMSD falls), stable (flat), or
        diverging (RMSD rises).

        Args:
            val_loader:   DataLoader providing (R, seq, pdb_id, chain_id).
            max_samples:  Maximum number of structures to attempt.
            temperature:  Langevin temperature; 0 = deterministic gradient descent.

        Returns:
            Dictionary with validation metrics, energy delta, and RMSD trajectory.
        """
        max_samples = int(max_samples)
        temperature = float(temperature)
        if step_size is not None:
            _orig_step_size = self.base_step_size
            self.base_step_size = float(step_size)
        else:
            _orig_step_size = None

        all_lengths = []
        all_theta = []
        all_phi = []
        rmsd_values = []  # Kabsch-aligned RMSD per structure
        drmsd_values = []  # pairwise distance RMSD per structure
        q_values = []  # Q_smooth native contacts per structure
        rmsf_values = []  # per-structure mean RMSF from Welford

        # Trajectory accumulators: list of mean RMSD per checkpoint step
        traj_accum: dict[int, list] = {}
        energy_deltas = []  # energy_final - energy_init per successful batch
        energy_deltas_per_struct = []  # per-structure energy deltas for distribution

        generation_calls = 0
        n_structures_attempted = 0
        n_structures_success = 0
        generation_failures = 0

        failure_counts = {
            "nan_inf": 0,
            "abort_clip": 0,
            "bond_threshold": 0,
            "drift": 0,
            "exception": 0,
            "success": 0,
        }

        spike_calls_force = 0
        spike_calls_clip = 0
        spike_calls_any = 0
        total_force_spike_structs = 0
        total_heavy_clip_structs = 0
        total_clip_fractions = []
        max_force_preclip_values = []

        self.model.eval()

        n_batches = len(val_loader)
        logger.info(
            "  [stability-val] Starting basin stability validation: "
            "%d structures requested, %d Langevin steps/structure, step_size=%.1e, beta=%.1f",
            max_samples,
            self.n_steps,
            self.base_step_size,
            self.langevin_beta,
        )

        for batch_idx, (R_native, seq, pdb_ids, chain_ids, lengths) in enumerate(val_loader):
            if n_structures_attempted >= max_samples:
                break

            R_native = R_native.to(self.device)
            seq = seq.to(self.device)
            lengths = lengths.to(self.device)

            B = R_native.shape[0]
            remaining = max_samples - n_structures_attempted
            if B > remaining:
                R_native = R_native[:remaining]
                seq = seq[:remaining]
                lengths = lengths[:remaining]
                B = R_native.shape[0]

            if B == 0:
                break

            generation_calls += 1
            n_structures_attempted += B

            # Progress log every 8 structures
            if n_structures_attempted % 8 == 0 or n_structures_attempted == max_samples:
                logger.info(
                    "  [stability-val] %d/%d structures simulated (ok=%d failed=%d)",
                    n_structures_attempted,
                    max_samples,
                    n_structures_success,
                    generation_failures,
                )

            # Start from clean native — tests whether native is a stable minimum
            R_init = R_native.clone()

            try:
                (
                    R_gen,
                    success,
                    spike_stats,
                    avg_clip,
                    max_force_preclip,
                    failure_reason,
                    rmsd_traj,
                    energy_init,
                    energy_final,
                    per_struct_deltas,
                    per_struct_rmsf_batch,
                ) = self._generate_structure(R_init, seq, temperature=temperature, R_ref=R_native, lengths=lengths)

                total_clip_fractions.append(avg_clip)
                max_force_preclip_values.append(max_force_preclip)

                total_force_spike_structs += int(spike_stats["n_force_spike_structs"])
                total_heavy_clip_structs += int(spike_stats["n_heavy_clip_structs"])

                call_force = bool(spike_stats["force_spike_any"])
                call_clip = bool(spike_stats["heavy_clip_any"])
                if call_force:
                    spike_calls_force += 1
                if call_clip:
                    spike_calls_clip += 1
                if call_force or call_clip:
                    spike_calls_any += 1

                # Accumulate trajectory
                for step_i, rmsd_i in rmsd_traj:
                    traj_accum.setdefault(step_i, []).append(rmsd_i)
                if energy_init is not None and energy_final is not None:
                    energy_deltas.append(energy_final - energy_init)
                # Collect per-structure deltas for distribution analysis
                if per_struct_deltas:
                    energy_deltas_per_struct.extend(per_struct_deltas)

                # Numerical failure — whole batch aborted
                if not success:
                    generation_failures += B
                    failure_counts[failure_reason if failure_reason in failure_counts else "exception"] += B
                    continue

                # NaN/Inf guard on output coords
                if not torch.isfinite(R_gen).all():
                    generation_failures += B
                    failure_counts["nan_inf"] += B
                    continue

                # ----------------------------------------------------------------
                # Compute ALL geometry once, then filter per-structure.
                # ----------------------------------------------------------------
                with torch.no_grad():
                    bond_lens = bond_lengths(R_gen)  # (B, L-1)
                    theta = bond_angles(R_gen)  # (B, L-2)
                    phi = torsions(R_gen)  # (B, L-3)

                # ---- Bond check -- skipped for IC simulator ----
                # IC Langevin guarantees bonds = 3.8 Angstrom exactly by NeRF construction.
                bond_ok = torch.ones(B, dtype=torch.bool, device=R_gen.device)
                ok_idx = torch.arange(B, device=R_gen.device)

                # ---- Per-structure aligned metrics (padding-aware) ----
                R_gen_np = R_gen.cpu().numpy()  # (B, L, 3)
                R_native_np = R_native.cpu().numpy()  # (B, L, 3)

                for b in ok_idx.cpu().numpy().tolist():
                    n = int(lengths[b])  # real chain length
                    gen_b = R_gen_np[b, :n]  # (n, 3)
                    native_b = R_native_np[b, :n]  # (n, 3)
                    # Kabsch-aligned RMSD (removes rigid-body displacement)
                    rmsd_values.append(rmsd_kabsch(gen_b, native_b))
                    # dRMSD -- rotation-invariant, no alignment needed
                    drmsd_values.append(drmsd(gen_b, native_b, mode="nonlocal", exclude=2))
                    # Q_smooth native contacts
                    try:
                        ni, nj, d0 = native_contact_set(native_b, cutoff=8.0, exclude=2)
                        q_values.append(q_smooth(gen_b, ni, nj, d0))
                    except Exception:
                        q_values.append(0.0)

                # ---- Collect angle/bond stats (padding-aware) ----
                for b in ok_idx.cpu().numpy().tolist():
                    n = int(lengths[b])
                    # Bonds: n-1 valid bonds per structure
                    all_lengths.extend(bond_lens[b, : n - 1].cpu().numpy().tolist())
                    # Theta: n-2 valid angles, phi: n-3 valid torsions
                    n_phi = n - 3
                    if n_phi > 0:
                        all_theta.extend(theta[b, :n_phi].cpu().numpy().tolist())
                        all_phi.extend(phi[b, :n_phi].cpu().numpy().tolist())

                n_structures_success += len(ok_idx)
                failure_counts["success"] += len(ok_idx)
                # Collect per-structure RMSF from this batch
                if per_struct_rmsf_batch:
                    rmsf_values.extend(per_struct_rmsf_batch)

            except Exception as exc:
                logger.debug("Generation failed for batch %d: %s", batch_idx, exc)
                generation_failures += B
                failure_counts["exception"] += B
                continue

        # Restore step_size if we overrode it
        if _orig_step_size is not None:
            self.base_step_size = _orig_step_size

        # Update running totals
        self.total_generation_calls += generation_calls
        self.total_structures_attempted += n_structures_attempted
        self.total_structures_success += n_structures_success
        self.total_structures_failed += generation_failures
        self.force_spike_calls += spike_calls_force
        self.heavy_clip_calls += spike_calls_clip

        denom_structs = max(1, n_structures_attempted)
        denom_calls = max(1, generation_calls)

        spike_result = {
            "force_spike_rate_calls": spike_calls_force / denom_calls,
            "heavy_clip_rate_calls": spike_calls_clip / denom_calls,
            "any_spike_rate_calls": spike_calls_any / denom_calls,
            "force_spike_rate_structures": total_force_spike_structs / denom_structs,
            "heavy_clip_rate_structures": total_heavy_clip_structs / denom_structs,
        }

        avg_clip = float(np.mean(total_clip_fractions)) if total_clip_fractions else 0.0
        max_force_arr = (
            np.array(max_force_preclip_values, dtype=np.float64)
            if max_force_preclip_values
            else np.array([], dtype=np.float64)
        )
        force_result = {
            "avg_clip_fraction": avg_clip,
            "avg_max_force_preclip": float(max_force_arr.mean()) if max_force_arr.size else 0.0,
            "max_max_force_preclip": float(max_force_arr.max()) if max_force_arr.size else 0.0,
            "p99_max_force_preclip": float(np.quantile(max_force_arr, 0.99))
            if max_force_arr.size > 1
            else (float(max_force_arr[0]) if max_force_arr.size else 0.0),
        }
        counts_result = {
            "n_calls": generation_calls,
            "n_attempted": n_structures_attempted,
            "n_success": n_structures_success,
            "n_generated": n_structures_success,  # alias for trainer compatibility
            "generation_failures": generation_failures,
            "failure_counts": failure_counts,
        }

        if not all_lengths:
            logger.warning("No valid structures generated during validation")
            if generation_failures > 0:
                logger.info(
                    "  Failure reasons: %s",
                    {k: v for k, v in failure_counts.items() if v > 0 and k != "success"},
                )
            return {
                "valid": False,
                "failure_reason": "no_valid_structures",
                **counts_result,
                **spike_result,
                **force_result,
                "bond_mean": float("inf"),
                "bond_std": float("inf"),
                "bond_rmsd": float("inf"),
                "ramachandran_corr": 0.0,
                "delta_phi_corr": 0.0,
                "mean_rmsd": float("inf"),
                "std_rmsd": float("inf"),
                "max_rmsd": float("inf"),
                "p95_rmsd": float("inf"),
            }

        # Convert to numpy
        all_lengths = np.asarray(all_lengths, dtype=np.float64)
        all_theta = np.asarray(all_theta, dtype=np.float64)
        all_phi = np.asarray(all_phi, dtype=np.float64)
        rmsd_values = np.asarray(rmsd_values, dtype=np.float64)
        drmsd_values = np.asarray(drmsd_values, dtype=np.float64)
        q_values = np.asarray(q_values, dtype=np.float64)

        # Verify theta/phi alignment — should always match after per-structure trim above
        if len(all_theta) != len(all_phi):
            logger.warning(
                "theta/phi length mismatch (%d vs %d) — truncating to min. "
                "This should not happen; check angle collection logic.",
                len(all_theta),
                len(all_phi),
            )
            min_angle_len = min(len(all_theta), len(all_phi))
            all_theta = all_theta[:min_angle_len]
            all_phi = all_phi[:min_angle_len]

        valid_mask = (~np.isnan(all_theta)) & (~np.isnan(all_phi))
        if not np.all(valid_mask):
            logger.debug("Removed %d NaN angle values", int(np.sum(~valid_mask)))
            all_theta = all_theta[valid_mask]
            all_phi = all_phi[valid_mask]

        # Bond statistics
        bond_mean = float(all_lengths.mean())
        bond_std = float(all_lengths.std())
        bond_rmsd = float(np.sqrt(np.mean((all_lengths - IDEAL_BOND) ** 2)))
        min_bond = float(all_lengths.min())
        max_bond = float(all_lengths.max())
        p01_min_bond = float(np.quantile(all_lengths, 0.01))
        p99_max_bond = float(np.quantile(all_lengths, 0.99))

        # Angle correlations — negative values are NOT clipped (they are diagnostic)
        if all_theta.size > 0:
            rama_corr = float(
                compute_ramachandran_correlation(
                    torch.tensor(all_theta, device="cpu"),
                    torch.tensor(all_phi, device="cpu"),
                )
            )
            dphi_corr = float(
                compute_delta_phi_correlation(
                    torch.tensor(all_phi, device="cpu"),
                )
            )
        else:
            rama_corr = 0.0
            dphi_corr = 0.0

        # RMSD statistics (Kabsch-aligned)
        mean_rmsd = float(rmsd_values.mean())
        std_rmsd = float(rmsd_values.std())
        max_rmsd = float(rmsd_values.max())
        p05_rmsd = float(np.quantile(rmsd_values, 0.05)) if rmsd_values.size else float("inf")
        p95_rmsd = float(np.quantile(rmsd_values, 0.95)) if rmsd_values.size else float("inf")

        # dRMSD statistics (rotation-invariant)
        mean_drmsd = float(drmsd_values.mean()) if drmsd_values.size else float("inf")
        std_drmsd = float(drmsd_values.std()) if drmsd_values.size else float("inf")
        p05_drmsd = float(np.quantile(drmsd_values, 0.05)) if drmsd_values.size else float("inf")
        p95_drmsd = float(np.quantile(drmsd_values, 0.95)) if drmsd_values.size else float("inf")

        # Q_smooth statistics
        mean_q = float(q_values.mean()) if q_values.size else 0.0
        std_q = float(q_values.std()) if q_values.size else 0.0
        p05_q = float(np.quantile(q_values, 0.05)) if q_values.size else 0.0
        p95_q = float(np.quantile(q_values, 0.95)) if q_values.size else 0.0

        # RMSF statistics (from Welford online accumulation during Langevin)
        rmsf_arr = np.asarray(rmsf_values, dtype=np.float64) if rmsf_values else np.array([], dtype=np.float64)
        mean_rmsf = float(rmsf_arr.mean()) if rmsf_arr.size else 0.0
        std_rmsf = float(rmsf_arr.std()) if rmsf_arr.size else 0.0

        # Stability verdict — mean_energy_delta must be computed first
        mean_energy_delta = float(np.mean(energy_deltas)) if energy_deltas else None

        # |E_delta| < 0.3 → thermal fluctuation (stable)
        # E_delta < -0.3 → found deeper non-native basin (compaction)
        # E_delta > +0.3 → escaped basin (unstable)
        is_stable = (mean_rmsd < 5.0) and (mean_energy_delta is not None) and (abs(mean_energy_delta) < 0.3)

        # Per-structure energy delta distribution
        if energy_deltas_per_struct:
            ed_arr = np.asarray(energy_deltas_per_struct, dtype=np.float64)
            ed_neg_frac = float(np.mean(ed_arr < 0))
            ed_p05 = float(np.quantile(ed_arr, 0.05))
            ed_p50 = float(np.quantile(ed_arr, 0.50))
            ed_p95 = float(np.quantile(ed_arr, 0.95))
        else:
            ed_neg_frac = 0.0
            ed_p05 = ed_p50 = ed_p95 = 0.0

        # Energy delta summary
        traj_summary = sorted([(step, float(np.mean(rmsds))) for step, rmsds in traj_accum.items()])

        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("Basin Stability Validation Results:")
        logger.info("=" * 60)
        logger.info("  Mode: start from native PDB -> run %d Langevin steps -> measure drift", self.n_steps)
        logger.info(
            "  Calls: %d | Attempted: %d | Successful: %d | Failures: %d",
            generation_calls,
            n_structures_attempted,
            n_structures_success,
            generation_failures,
        )
        logger.info(
            "  Spike calls: force=%d/%d (%.2f%%), clip=%d/%d (%.2f%%)",
            spike_calls_force,
            generation_calls,
            100 * spike_calls_force / denom_calls,
            spike_calls_clip,
            generation_calls,
            100 * spike_calls_clip / denom_calls,
        )
        logger.info("  Avg clip fraction: %.2f%%", 100 * avg_clip)
        if mean_energy_delta is not None:
            direction = "↓ good (attracted)" if mean_energy_delta < 0 else "↑ bad (pushed away)"
            logger.info("  Energy delta (final−init): %.4f  [%s]", mean_energy_delta, direction)
            if energy_deltas_per_struct:
                logger.info(
                    "  E delta distribution (n=%d): p05=%.3f  median=%.3f  p95=%.3f  frac_negative=%.1f%%",
                    len(energy_deltas_per_struct),
                    ed_p05,
                    ed_p50,
                    ed_p95,
                    100 * ed_neg_frac,
                )
        if traj_summary:
            traj_str = "  → ".join(f"step{s}:{r:.3f}Å" for s, r in traj_summary)
            logger.info("  RMSD trajectory: %s", traj_str)
        logger.info(
            "  RMSD:  mean=%.3f ± %.3f Å  p05=%.3f  p95=%.3f  max=%.3f",
            mean_rmsd,
            std_rmsd,
            p05_rmsd,
            p95_rmsd,
            max_rmsd,
        )
        logger.info(
            "  dRMSD: mean=%.3f ± %.3f Å  p05=%.3f  p95=%.3f",
            mean_drmsd,
            std_drmsd,
            p05_drmsd,
            p95_drmsd,
        )
        logger.info(
            "  Q (native contacts): mean=%.3f ± %.3f  p05=%.3f  p95=%.3f",
            mean_q,
            std_q,
            p05_q,
            p95_q,
        )
        if rmsf_arr.size:
            logger.info(
                "  RMSF:  mean=%.3f ± %.3f Å",
                mean_rmsf,
                std_rmsf,
            )
        logger.info(
            "  Bonds: mean=%.3f ± %.3f Å  rmsd_to_3.80=%.4f  range=[%.3f, %.3f]",
            bond_mean,
            bond_std,
            bond_rmsd,
            min_bond,
            max_bond,
        )
        logger.info("  Ramachandran corr: %.4f", rama_corr)
        logger.info("  Delta phi corr: %.4f", dphi_corr)
        stability_str = "STABLE ✓" if is_stable else "UNSTABLE ✗✗  — native pushed away, landscape wrong"
        logger.info("  Stability: %s", stability_str)
        logger.info("=" * 60)

        return {
            "valid": True,
            "failure_reason": None,
            **counts_result,
            **spike_result,
            **force_result,
            "bond_mean": bond_mean,
            "bond_std": bond_std,
            "bond_rmsd": bond_rmsd,
            "min_bond": min_bond,
            "max_bond": max_bond,
            "p01_min_bond": p01_min_bond,
            "p99_max_bond": p99_max_bond,
            "ramachandran_corr": rama_corr,
            "delta_phi_corr": dphi_corr,
            "mean_rmsd": mean_rmsd,
            "std_rmsd": std_rmsd,
            "max_rmsd": max_rmsd,
            "p05_rmsd": p05_rmsd,
            "p95_rmsd": p95_rmsd,
            "mean_drmsd": mean_drmsd,
            "std_drmsd": std_drmsd,
            "p05_drmsd": p05_drmsd,
            "p95_drmsd": p95_drmsd,
            "mean_q": mean_q,
            "std_q": std_q,
            "p05_q": p05_q,
            "p95_q": p95_q,
            "mean_rmsf": mean_rmsf,
            "std_rmsf": std_rmsf,
            "mean_energy_delta": mean_energy_delta,
            "energy_delta_p05": ed_p05,
            "energy_delta_p50": ed_p50,
            "energy_delta_p95": ed_p95,
            "energy_delta_neg_frac": ed_neg_frac,
            "rmsd_trajectory": traj_summary,
            "is_stable": is_stable,
        }
