"""Negative configuration collector for self-consistent training.

src/calphaebm/training/negative_collector.py

Runs CG Langevin dynamics from native structures with the current energy
model, collects configurations where the model fails (compaction, false
basins, drift, etc.), and returns them as contrastive training pairs.

This is Stage 2a of the self-consistent training loop:
  Stage 1: DSM + ELT on PDB (standard training)
  Stage 2a: CG Langevin → collect failure modes → retrain  ← THIS
  Stage 2b: Generative ensemble → thermodynamic negatives (future)

Usage:
    from calphaebm.training.negative_collector import NegativeCollector

    collector = NegativeCollector(model, device)
    negatives = collector.collect(
        dataloader,
        n_proteins=100,
        n_steps=50000,
        beta=1000.0,
    )
    # negatives: list of NegativeExample with R_native, R_negative, seq, category
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from calphaebm.evaluation.metrics.contacts import native_contact_set, q_smooth
from calphaebm.evaluation.metrics.rg import radius_of_gyration
from calphaebm.evaluation.metrics.rmsd import rmsd_kabsch
from calphaebm.simulation.backends import get_simulator
from calphaebm.utils.logging import get_logger

logger = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Parallel worker infrastructure (fork-based, CPU only)
# ─────────────────────────────────────────────────────────────────────────────

# Global reference set before fork — workers inherit via copy-on-write
_WORKER_COLLECTOR = None


def _worker_fn(task: tuple) -> list:
    """Worker function for parallel collection.
    Called in forked child processes. Accesses _WORKER_COLLECTOR (set
    before fork) via copy-on-write memory sharing.
    """
    torch.set_num_threads(1)  # Prevent thread oversubscription
    R_i, seq_i, beta, n_steps, pdb_id, chain_id = task
    if _WORKER_COLLECTOR is None:
        return []
    try:
        return _WORKER_COLLECTOR._collect_single(
            R_native=R_i,
            seq=seq_i,
            beta=beta,
            n_steps=n_steps,
            pdb_id=pdb_id,
            chain_id=chain_id,
        )
    except Exception as e:
        # print() reaches stdout in forked workers; logger may not
        print(f"  [WORKER ERROR] {pdb_id} crashed: {e}", flush=True)
        logger.warning("Worker failed for %s: %s", pdb_id, e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Failure categories
# ─────────────────────────────────────────────────────────────────────────────


class FailureCategory(str, Enum):
    """Categories of null-space failure modes discovered during Langevin."""

    COMPACTED = "compacted"  # Rg dropped below threshold
    SWOLLEN = "swollen"  # Rg expanded beyond threshold
    FALSE_BASIN = "false_basin"  # low Q, lower energy than native
    DRIFT_PRESERVED = "drift_preserved"  # high RMSD but contacts preserved
    FROZEN = "frozen"  # RMSF too low (over-stabilized)
    SS_LOSS = "ss_loss"  # secondary structure content changed
    MISFOLD = "misfold"  # high RMSD + low Q (global unfolding/misfold)


@dataclass
class NegativeExample:
    """A single negative (failure) configuration paired with its native."""

    pdb_id: str
    chain_id: str
    seq: torch.Tensor  # (L,) int
    R_native: torch.Tensor  # (L, 3) native coordinates
    R_negative: torch.Tensor  # (L, 3) failure configuration
    category: FailureCategory
    # Diagnostics
    E_native: float  # energy at native
    E_negative: float  # energy at failure config
    rg_ratio: float  # Rg(negative) / Rg_native
    q: float  # Q(negative, native)
    rmsd: float  # RMSD to native
    rmsf: float  # running RMSF at collection point
    step: int  # Langevin step where collected
    drmsd: float = 0.0  # full pairwise dRMSD to native (Å)


@dataclass
class CollectionStats:
    """Summary statistics from a collection run."""

    n_proteins: int = 0
    n_steps: int = 0
    n_negatives: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)
    wall_time_sec: float = 0.0
    proteins_with_failures: int = 0

    def summary(self) -> str:
        lines = [
            f"Collection: {self.n_proteins} proteins, {self.n_steps} steps/protein",
            f"  Total negatives: {self.n_negatives} from {self.proteins_with_failures}/{self.n_proteins} proteins",
            f"  Wall time: {self.wall_time_sec:.0f}s ({self.wall_time_sec/60:.1f} min)",
        ]
        if self.category_counts:
            lines.append("  Categories:")
            for cat, count in sorted(self.category_counts.items()):
                lines.append(f"    {cat}: {count}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Secondary structure assignment from Cα geometry
# ─────────────────────────────────────────────────────────────────────────────


def _assign_ss_from_ca(R: np.ndarray) -> np.ndarray:
    """Assign secondary structure from Cα coordinates using virtual angles.

    Uses Cα-Cα-Cα bond angles (θ) and Cα-Cα-Cα-Cα dihedrals (φ) to
    classify each residue as helix (H), sheet (E), or coil (C).

    Helix:  θ ∈ [80°, 105°] — tight turn
    Sheet:  θ ∈ [105°, 150°] — extended
    Coil:   everything else

    Args:
        R: (L, 3) Cα coordinates

    Returns:
        ss: (L,) array of 0=coil, 1=helix, 2=sheet
    """
    L = R.shape[0]
    ss = np.zeros(L, dtype=np.int32)  # default coil

    if L < 4:
        return ss

    # Compute virtual bond angles θ_i = angle(Cα_{i-1}, Cα_i, Cα_{i+1})
    for i in range(1, L - 1):
        v1 = R[i - 1] - R[i]
        v2 = R[i + 1] - R[i]
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

        if 80 <= theta_deg <= 105:
            ss[i] = 1  # helix
        elif 105 < theta_deg <= 150:
            ss[i] = 2  # sheet

    return ss


def _ss_change(R_native: np.ndarray, R_other: np.ndarray) -> float:
    """Compute fractional secondary structure change between two conformations.

    Returns fraction of residues that changed SS category (0.0 to 1.0).
    """
    ss_nat = _assign_ss_from_ca(R_native)
    ss_oth = _assign_ss_from_ca(R_other)
    return float((ss_nat != ss_oth).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Main collector
# ─────────────────────────────────────────────────────────────────────────────


class NegativeCollector:
    """Collect failure-mode configurations from CG Langevin dynamics.

    Runs the current energy model's Langevin dynamics from native structures,
    monitors for failure modes (compaction, false basins, etc.), and collects
    the failing configurations as contrastive training pairs.

    Args:
        model:          TotalEnergy model (current training state)
        device:         torch device
        step_size:      Langevin step size
        force_cap:      force clipping threshold
        save_every:     check for failures every N steps
        contact_cutoff: cutoff for Q contact calculation
        exclude:        sequence separation for contacts

        Failure thresholds:
        rg_compact:     Rg ratio below this → compacted (default 0.85)
        rg_swollen:     Rg ratio above this → swollen (default 1.20)
        q_false_basin:  Q below this + E < E_minimized → false basin (default 0.70)
        rmsd_drift:     RMSD above this + Q > 0.9 → drift with contacts (default 5.0)
        rmsf_frozen:    RMSF below this → over-stabilized (default 0.3)
        ss_change_thr:  SS change above this → secondary structure loss (default 0.3)
        rmsd_misfold:   RMSD above this + Q below q_misfold → misfold (default 6.0)
        q_misfold:      Q below this (with high RMSD) → misfold (default 0.80)
        max_negatives_per_protein: cap to prevent flooding from one protein
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        step_size: float = 1e-3,
        force_cap: float = 100.0,
        save_every: int = 50,
        contact_cutoff: float = 8.0,
        exclude: int = 2,
        # Failure thresholds
        rg_compact: float = 0.85,
        rg_swollen: float = 1.20,
        q_false_basin: float = 0.70,
        rmsd_drift: float = 5.0,
        rmsf_frozen: float = 0.3,
        ss_change_thr: float = 0.3,
        rmsd_misfold: float = 6.0,
        q_misfold: float = 0.80,
        max_negatives_per_protein: int = 10,
        sampler: str = "langevin",
    ):
        self.model = model
        self.device = device
        self.step_size = step_size
        self.force_cap = force_cap
        self.save_every = save_every
        self.contact_cutoff = contact_cutoff
        self.exclude = exclude
        self.sampler = sampler

        # Thresholds
        self.rg_compact = rg_compact
        self.rg_swollen = rg_swollen
        self.q_false_basin = q_false_basin
        self.rmsd_drift = rmsd_drift
        self.rmsf_frozen = rmsf_frozen
        self.ss_change_thr = ss_change_thr
        self.rmsd_misfold = rmsd_misfold
        self.q_misfold = q_misfold
        self.max_negatives_per_protein = max_negatives_per_protein

    def _flory_rg(self, L: int, r0: float = 2.0, nu: float = 0.38) -> float:
        """Expected Rg from Flory scaling."""
        return r0 * (L**nu)

    def _collect_single(
        self,
        R_native: torch.Tensor,  # (1, L, 3)
        seq: torch.Tensor,  # (1, L)
        beta: float,
        n_steps: int,
        pdb_id: str = "?",
        chain_id: str = "?",
    ) -> List[NegativeExample]:
        """Run Langevin on one structure, collect failure configurations."""

        L = R_native.shape[1]
        R_nat_np = R_native[0, :L].cpu().numpy()
        rg_native = radius_of_gyration(R_nat_np)
        rg_flory = self._flory_rg(L)

        # Native contacts for Q
        ni_nat, nj_nat, d0_nat = native_contact_set(R_nat_np, cutoff=self.contact_cutoff, exclude=self.exclude)

        # Native energy (raw PDB)
        len_b = torch.tensor([L], device=self.device)
        with torch.no_grad():
            E_native = float(self.model(R_native, seq, lengths=len_b).item())

        # Safety: catch NaN/Inf or extreme energies from broken model
        if not math.isfinite(E_native):
            print(f"    [{pdb_id}] E_native={E_native} (NaN/Inf) — skipping", flush=True)
            return []
        if abs(E_native) > 50.0:
            print(f"    [{pdb_id}] Extreme E_native={E_native:.1f} — model may be broken", flush=True)

        # ── Minimize PDB structure via L-BFGS in IC space ──────────
        from calphaebm.simulation.minimize import lbfgs_minimize

        _min_result = lbfgs_minimize(self.model, R_native, seq, lengths=len_b)
        R_min = _min_result["R_min"]
        E_minimized = _min_result["E_minimized"]
        min_steps_taken = _min_result["min_steps"]
        max_force = _min_result["max_force"]
        R_min_np = R_min[0, :L].detach().cpu().numpy()
        _d_nat = np.sqrt(((R_nat_np[:, None] - R_nat_np[None, :]) ** 2).sum(-1))
        _d_min = np.sqrt(((R_min_np[:, None] - R_min_np[None, :]) ** 2).sum(-1))
        _triu = np.triu_indices(L, k=1)
        drmsd_min = float(np.sqrt(np.mean((_d_nat[_triu] - _d_min[_triu]) ** 2)))
        logger.info(
            "    [%s] minimized in %d steps: ΔE=%.3f  dRMSD=%.2f  maxF=%.1f",
            pdb_id,
            min_steps_taken,
            E_minimized - E_native,
            drmsd_min,
            max_force,
        )

        negatives = []

        # Welford RMSF
        n_rmsf = 0
        mean_pos = np.zeros((L, 3), dtype=np.float64)
        M2_pos = np.zeros((L, 3), dtype=np.float64)

        sim = None  # initialized here so MALA stats check works after try/except
        try:
            # Start dynamics from minimized structure (model's own minimum)
            sim = get_simulator(
                name=self.sampler,
                model=self.model,
                seq=seq,
                R_init=R_min.detach(),
                step_size=1.0 / (10.0 * float(L) ** 2),
                beta=beta,
                force_cap=self.force_cap,
                lengths=len_b,
            )
            R_current = R_min.detach()

            for step_i in range(1, n_steps + 1):
                R_current, E_step, info = sim.step()

                if step_i % self.save_every == 0:
                    # Already collected enough? Stop entirely.
                    if len(negatives) >= self.max_negatives_per_protein:
                        break

                    # MALA early stop: if acceptance < 10% after 10K steps,
                    # protein is frozen — no point running 90K more steps
                    if (
                        self.sampler == "mala"
                        and step_i == 10000
                        and hasattr(sim, "acceptance_rate")
                        and sim.acceptance_rate < 0.10
                    ):
                        logger.info(
                            "    [%s] MALA early stop: %.1f%% acceptance at step 10K — skipping",
                            pdb_id,
                            sim.acceptance_rate * 100,
                        )
                        break

                    # Progress log every 10K steps
                    if step_i % 10000 == 0 and step_i < n_steps:
                        _acc_str = ""
                        if hasattr(sim, "acceptance_rate"):
                            _acc_str = f"  accept={sim.acceptance_rate*100:.1f}%"
                        logger.info(
                            "    [%s] step %dK/%dK  neg=%d/8  β=%.0f%s",
                            pdb_id,
                            step_i // 1000,
                            n_steps // 1000,
                            len(negatives),
                            beta,
                            _acc_str,
                        )

                    R_snap = R_current[0, :L].detach().cpu().numpy()

                    # Welford RMSF update
                    n_rmsf += 1
                    delta_w = R_snap.astype(np.float64) - mean_pos
                    mean_pos += delta_w / n_rmsf
                    delta_w2 = R_snap.astype(np.float64) - mean_pos
                    M2_pos += delta_w * delta_w2

                    if n_rmsf > 1:
                        _var = M2_pos / (n_rmsf - 1)
                        rmsf_val = float(np.sqrt(_var.sum(axis=1)).mean())
                    else:
                        rmsf_val = 0.0

                    # Compute metrics
                    rg_snap = radius_of_gyration(R_snap)
                    rg_ratio = rg_snap / rg_native if rg_native > 0 else 1.0
                    q_val = q_smooth(R_snap, ni_nat, nj_nat, d0_nat)
                    rmsd_val = rmsd_kabsch(R_snap, R_nat_np)
                    E_snap = info.energy
                    ss_change_val = _ss_change(R_nat_np, R_snap)

                    # ── Classify failure mode ─────────────────────────────
                    category = None

                    if rg_ratio < self.rg_compact:
                        category = FailureCategory.COMPACTED
                    elif rg_ratio > self.rg_swollen:
                        category = FailureCategory.SWOLLEN
                    elif q_val < self.q_false_basin and E_snap < E_minimized:
                        category = FailureCategory.FALSE_BASIN
                    elif rmsd_val > self.rmsd_drift and q_val > 0.95:
                        category = FailureCategory.DRIFT_PRESERVED
                    elif ss_change_val > self.ss_change_thr:
                        category = FailureCategory.SS_LOSS
                    elif rmsd_val > self.rmsd_misfold or q_val < self.q_misfold:
                        category = FailureCategory.MISFOLD

                    if category is not None:
                        # Full dRMSD — reuse R_nat_np and R_current already available
                        _R_neg_np = R_current[0, :L].detach().cpu().numpy()
                        _D_nat = np.sqrt(((R_nat_np[:, None] - R_nat_np[None, :]) ** 2).sum(-1))
                        _D_neg = np.sqrt(((_R_neg_np[:, None] - _R_neg_np[None, :]) ** 2).sum(-1))
                        _triu = np.triu_indices(L, k=4)
                        _drmsd_val = (
                            float(np.sqrt(np.mean((_D_nat[_triu] - _D_neg[_triu]) ** 2))) if len(_triu[0]) > 0 else 0.0
                        )

                        neg = NegativeExample(
                            pdb_id=pdb_id,
                            chain_id=chain_id,
                            seq=seq[0].cpu(),
                            R_native=R_native[0].cpu(),
                            R_negative=R_current[0, :L].detach().cpu(),
                            category=category,
                            E_native=E_minimized,
                            E_negative=E_snap,
                            rg_ratio=rg_ratio,
                            q=q_val,
                            rmsd=rmsd_val,
                            rmsf=rmsf_val,
                            step=step_i,
                            drmsd=_drmsd_val,
                        )
                        negatives.append(neg)

                        # Log every failure
                        logger.info(
                            "    [%s] %s at step %d (%d/%d): "
                            "Rg=%.0f%%  Q=%.3f  RMSD=%.1f  RMSF=%.2f  E=%.3f (min=%.3f)",
                            pdb_id,
                            category.value,
                            step_i,
                            len(negatives),
                            self.max_negatives_per_protein,
                            rg_ratio * 100,
                            q_val,
                            rmsd_val,
                            rmsf_val,
                            E_snap,
                            E_minimized,
                        )

        except Exception as e:
            # CRITICAL: print() so errors are visible in forked workers
            print(f"    [{pdb_id}] {self.sampler} failed: {e}", flush=True)
            logger.warning("%s failed for %s: %s", self.sampler, pdb_id, e)

        # ── Final structural summary for every protein ──────────────
        try:
            if sim is not None and "R_current" in dir():
                _R_final = R_current[0, :L].detach().cpu().numpy()
                _q_fin = q_smooth(_R_final, ni_nat, nj_nat, d0_nat)
                _rm_fin = rmsd_kabsch(_R_final, R_nat_np)
                _rg_fin = radius_of_gyration(_R_final)
                _rg_pct = _rg_fin / rg_native * 100 if rg_native > 0 else 0.0
                _D_nat = np.sqrt(((R_nat_np[:, None] - R_nat_np[None, :]) ** 2).sum(-1))
                _D_fin = np.sqrt(((_R_final[:, None] - _R_final[None, :]) ** 2).sum(-1))
                _triu_k4 = np.triu_indices(L, k=4)
                _dr_fin = (
                    float(np.sqrt(np.mean((_D_nat[_triu_k4] - _D_fin[_triu_k4]) ** 2))) if len(_triu_k4[0]) > 0 else 0.0
                )
                _steps_run = sim._n_total if hasattr(sim, "_n_total") else n_steps
                logger.info(
                    "    [%s] DONE  β=%.0f  L=%d  steps=%d  "
                    "Q=%.3f  RMSD=%.2fA  dRMSD=%.2fA  Rg=%.0f%%  "
                    "failures=%d/%d",
                    pdb_id,
                    beta,
                    L,
                    _steps_run,
                    _q_fin,
                    _rm_fin,
                    _dr_fin,
                    _rg_pct,
                    len(negatives),
                    self.max_negatives_per_protein,
                )
        except Exception as _e:
            logger.debug("Final summary failed for %s: %s", pdb_id, _e)

        # Log MALA stats for every protein
        if self.sampler == "mala" and sim is not None and hasattr(sim, "acceptance_rate"):
            acc = sim.acceptance_rate * 100
            n_acc = sim._n_accepted
            n_tot = sim._n_total
            n_rej = n_tot - n_acc
            logger.info(
                "    [%s] MALA: accept=%d/%d (%.1f%%)  reject=%d  step_size=%.1e  beta=%.0f",
                pdb_id,
                n_acc,
                n_tot,
                acc,
                n_rej,
                1.0 / (10.0 * float(L) ** 2),
                beta,
            )
            if acc < 20:
                logger.warning(
                    "    [%s] MALA acceptance %.1f%% — step_size may be too large",
                    pdb_id,
                    acc,
                )

        return negatives

    def collect(
        self,
        dataloader: DataLoader,
        n_proteins: int = 100,
        n_steps: int = 50000,
        beta: float = 1000.0,
        seed: int | None = None,
        n_workers: int = 1,
        max_collect_len: int = 200,
    ) -> Tuple[List[NegativeExample], CollectionStats]:
        """Collect negative configurations from randomly sampled proteins.

        Each call samples a DIFFERENT random subset of proteins from the
        dataloader.  Pass different seeds each round to get diverse
        coverage of the training set across rounds.

        When n_workers > 1, uses fork-based multiprocessing for parallel
        Langevin dynamics (CPU only). Each worker inherits the model via
        copy-on-write memory sharing.

        Args:
            dataloader:  DataLoader yielding (R, seq, pdb_id, chain_id, lengths) batches
            n_proteins:  number of proteins to sample and run
            n_steps:     Langevin steps per protein
            beta:        inverse temperature
            seed:        random seed for protein selection (None = time-based)
            n_workers:   parallel workers (default 1 = serial, 8 = 8 cores)

        Returns:
            negatives:  list of NegativeExample
            stats:      CollectionStats summary
        """
        import random as _rng

        self.model.eval()

        all_negatives = []
        stats = CollectionStats(n_steps=n_steps)
        t0 = time.time()

        # Gather all available structures from the dataloader
        all_structures = []
        for batch in dataloader:
            # PDBChainDataset.collate returns (R, seq, pdb_ids, chain_ids, lengths)
            if len(batch) == 5:
                R_batch, seq_batch, pdb_ids, chain_ids, lengths_batch = batch
            elif len(batch) == 4:
                R_batch, seq_batch, pdb_ids, chain_ids = batch
                lengths_batch = None
            else:
                R_batch, seq_batch = batch[0], batch[1]
                pdb_ids = [f"unk_{i}" for i in range(R_batch.shape[0])]
                chain_ids = ["A"] * R_batch.shape[0]
                lengths_batch = None

            B = R_batch.shape[0]
            for i in range(B):
                pdb_id = pdb_ids[i] if isinstance(pdb_ids, (list, tuple)) else str(pdb_ids)
                chain_id = chain_ids[i] if isinstance(chain_ids, (list, tuple)) else str(chain_ids)
                if lengths_batch is not None:
                    L = int(lengths_batch[i].item())
                elif seq_batch[i].min() < 0:
                    L = int(seq_batch[i].ne(-1).sum())
                else:
                    L = seq_batch[i].shape[0]
                # Filter by max length — long chains dominate collection time
                if L > max_collect_len:
                    continue
                all_structures.append((R_batch[i : i + 1, :L], seq_batch[i : i + 1, :L], pdb_id, chain_id, L))

        # Randomly sample n_proteins (different set each round)
        rng = _rng.Random(seed)
        n_sample = min(n_proteins, len(all_structures))
        selected = rng.sample(all_structures, n_sample)

        logger.info(
            "  Randomly selected %d/%d proteins (seed=%s, workers=%d)", n_sample, len(all_structures), seed, n_workers
        )

        # Build tasks: (R_i, seq_i, beta, n_steps, pdb_id, chain_id)
        tasks = [
            (R_i.cpu(), seq_i.cpu(), beta, n_steps, pdb_id, chain_id) for (R_i, seq_i, pdb_id, chain_id, L) in selected
        ]

        if n_workers > 1:
            # ── Parallel collection via fork ──────────────────────────
            all_negatives = self._collect_parallel(tasks, n_workers)
        else:
            # ── Serial collection ─────────────────────────────────────
            for idx, task in enumerate(tasks):
                R_i, seq_i, beta_t, n_steps_t, pdb_id, chain_id = task
                L = R_i.shape[1]
                logger.info("  [%d/%d] %s chain %s (L=%d)...", idx + 1, n_sample, pdb_id, chain_id, L)

                negs = self._collect_single(
                    R_native=R_i,
                    seq=seq_i,
                    beta=beta_t,
                    n_steps=n_steps_t,
                    pdb_id=pdb_id,
                    chain_id=chain_id,
                )
                all_negatives.extend(negs)
                logger.info("    → %d negatives collected", len(negs))

        # Compute stats
        for neg in all_negatives:
            cat = neg.category.value
            stats.category_counts[cat] = stats.category_counts.get(cat, 0) + 1
        stats.proteins_with_failures = len(set((n.pdb_id, n.chain_id) for n in all_negatives))
        stats.n_proteins = n_sample
        stats.n_negatives = len(all_negatives)
        stats.wall_time_sec = time.time() - t0

        logger.info("\n%s", stats.summary())

        return all_negatives, stats

    def _collect_parallel(
        self,
        tasks: List[tuple],
        n_workers: int,
    ) -> List[NegativeExample]:
        """Run collection tasks in parallel using fork-based multiprocessing.

        Sets a module-level global to self (the collector), then forks.
        Worker processes inherit the model and all state via copy-on-write.
        CPU only — CUDA tensors cannot be shared across forked processes.

        After retraining, .backward() leaves autograd engine threads that
        can't survive fork(). We deep-copy the model first — the copy has
        identical weights but no autograd history, making fork safe.
        """
        import copy

        import torch.multiprocessing as mp

        # Deep-copy model to shed autograd thread state from backward().
        # Without this, fork() crashes after any retraining round.
        original_device = self.device
        self.device = torch.device("cpu")
        original_model = self.model
        original_model.cpu()
        self.model = copy.deepcopy(original_model)
        original_model.to(original_device)
        self.model.eval()

        global _WORKER_COLLECTOR
        _WORKER_COLLECTOR = self

        logger.info("  Starting parallel collection: %d tasks on %d workers", len(tasks), n_workers)

        # Use fork context (workers inherit model via COW)
        ctx = mp.get_context("fork")
        all_negatives = []

        try:
            with ctx.Pool(processes=n_workers) as pool:
                results = pool.map(_worker_fn, tasks)

            for negs in results:
                if negs:
                    all_negatives.extend(negs)

            n_empty = sum(1 for r in results if len(r) == 0)
            n_with = sum(1 for r in results if len(r) > 0)
            logger.info(
                "  Parallel collection complete: %d negatives from %d proteins "
                "(%d with failures, %d without/crashed)",
                len(all_negatives),
                len(tasks),
                n_with,
                n_empty,
            )
            if len(all_negatives) == 0 and len(tasks) > 0:
                logger.warning(
                    "  All %d workers returned empty — check thresholds or model state",
                    n_empty,
                )
        except Exception as e:
            logger.warning("Parallel collection failed: %s — falling back to serial", e)
            for task in tasks:
                R_i, seq_i, beta, n_steps, pdb_id, chain_id = task
                negs = self._collect_single(
                    R_native=R_i,
                    seq=seq_i,
                    beta=beta,
                    n_steps=n_steps,
                    pdb_id=pdb_id,
                    chain_id=chain_id,
                )
                all_negatives.extend(negs)
        finally:
            _WORKER_COLLECTOR = None  # clean up global ref
            self.model = original_model  # restore original (with grad history)
            self.device = original_device

        return all_negatives
