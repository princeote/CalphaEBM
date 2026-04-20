"""Self-consistent CG training loop.

src/calphaebm/training/self_consistent.py

Orchestrates the Stage 2a self-consistent training loop:

    for round in range(n_rounds):
        1. Run CG Langevin from native structures (100 proteins, 50K steps)
        2. Classify failure configurations (compaction, false basin, etc.)
        3. Retrain: PDB(depth+bal+discrim) + Sampled(hsm+qf+gap)
        4. Evaluate via basin stability on held-out proteins
        5. If improved, save checkpoint.  If converged, stop.

Cost: ~4 hours/round on 8-core Mac, 3-5 rounds = ~1 day total.
Compare: BioEmu's Stage 2 requires 200+ ms of all-atom MD (months of supercomputer).

Usage (standalone):
    python -m calphaebm.training.self_consistent \\
        --checkpoint checkpoints/run58/run1/full/step003000_best.pt \\
        --pdb train_entities.no_test_entries.txt \\
        --n-rounds 5 \\
        --collect-proteins 100 \\
        --collect-steps 50000 \\
        --retrain-steps 3000 \\
        --out-dir checkpoints/run58_sc

Usage (programmatic):
    from calphaebm.training.self_consistent import SelfConsistentTrainer

    sc = SelfConsistentTrainer(model, train_loader, val_loader, device, args)
    sc.run(n_rounds=5)
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from calphaebm.training.logging.diagnostics import DiagnosticLogger
from calphaebm.training.logging.validation_logging import ValidationLogger
from calphaebm.training.losses.balance_loss import energy_balance_loss
from calphaebm.training.negative_collector import CollectionStats, FailureCategory, NegativeCollector, NegativeExample
from calphaebm.training.sc_defaults import SC_DEFAULTS as D
from calphaebm.utils.logging import get_logger

logger = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Round result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RoundResult:
    """Result of one self-consistent round."""

    round_num: int
    # Collection stats
    n_negatives: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)
    collect_time_sec: float = 0.0
    # Training stats
    retrain_steps: int = 0
    final_loss: float = 0.0
    final_sampled_gap_loss: float = 0.0
    retrain_time_sec: float = 0.0
    # Evaluation stats
    basin_rg_pct: float = 0.0
    basin_e_delta: float = 0.0
    basin_rmsd: float = 0.0
    basin_q: float = 0.0
    basin_q_af: float = 100.0
    basin_rg_af: float = 100.0
    basin_rmsf: float = 0.0
    basin_composite: float = 0.0
    basin_k64drmsd: float = 0.0
    basin_contact_order: float = 0.0
    eval_time_sec: float = 0.0
    # Improvement tracking
    improved: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────────────────────


class SelfConsistentTrainer:
    """Self-consistent CG training loop.

    Alternates between collecting failure-mode negatives via CG Langevin
    and retraining the energy function with sampled losses on the
    collected negatives.

    Args:
        model:            TotalEnergy model (pre-trained via full stage)
        train_loader:     DataLoader for training data (PDB chains)
        val_loader:       DataLoader for validation data (for basin eval)
        device:           torch device

        Collection parameters:
        collect_proteins: number of proteins to run Langevin on per round
        collect_steps:    Langevin steps per protein
        collect_beta:     inverse temperature for collection dynamics
        collect_save_every: check for failures every N steps

        Retraining parameters:
        retrain_steps:    training steps per round
        retrain_lr:       learning rate for retraining
        lambda_sampled_gap:  weight for sampled gap loss (E_native < E_neg)
        lambda_elt:       weight for ELT funnel loss
        lambda_discrim:   weight for discrimination maintenance loss
        sc_margin:        margin for sampled gap loss (E/res)

        Evaluation parameters:
        eval_steps:       Langevin steps for basin eval (shorter than collection)
        eval_beta:        inverse temperature for basin eval
        eval_proteins:    number of proteins for basin eval

        Output:
        out_dir:          directory for saving checkpoints and logs
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        # Collection
        collect_proteins: int = D["collect_proteins"],
        collect_steps: int = D["collect_steps"],
        collect_beta: float = D["collect_beta"],
        collect_betas: list = D.get("collect_betas", None),
        collect_beta_min: float = D["collect_beta_min"],
        collect_beta_max: float = D["collect_beta_max"],
        collect_save_every: int = D["collect_save_every"],
        collect_step_size: float = D["collect_step_size"],
        collect_force_cap: float = D["collect_force_cap"],
        collect_n_workers: int = D["collect_n_workers"],
        collect_max_len: int = D["collect_max_len"],
        # Retraining
        retrain_steps: int = D["retrain_steps"],
        retrain_lr: float = D["retrain_lr"],
        # PDB batch losses (depth, balance, discrim)
        lambda_elt: float = D["lambda_elt"],
        lambda_gap: float = D["lambda_gap"],
        lambda_discrim: float = D["lambda_discrim"],
        lambda_depth: float = D["lambda_depth"],
        target_depth: float = D["target_depth"],
        lambda_balance: float = D["lambda_balance"],
        balance_r: float = D["balance_r"],
        balance_r_term: float = D["balance_r_term"],
        # Sampled negative losses (Cartesian, fully batched)
        lambda_sampled_hsm: float = D["lambda_sampled_hsm"],
        lambda_sampled_qf: float = D["lambda_sampled_qf"],
        lambda_sampled_drmsd_funnel: float = D.get(
            "lambda_sampled_drmsd_funnel", D.get("lambda_sampled_rg_funnel", 2.0)
        ),
        lambda_sampled_gap: float = D["lambda_sampled_gap"],
        sc_margin: float = D["sc_margin"],
        # Saturating exponential margins (Run5)
        funnel_m: float = D.get("funnel_m", 5.0),
        funnel_alpha: float = D.get("funnel_alpha", 5.0),
        gap_m: float = D.get("gap_m", 5.0),
        gap_alpha: float = D.get("gap_alpha", 5.0),
        # Evaluation
        eval_steps: int = D["eval_steps"],
        eval_beta: float = D["eval_beta"],
        eval_proteins: int = D["eval_proteins"],
        # Collector thresholds
        rg_compact: float = D["rg_compact"],
        rg_swollen: float = D["rg_swollen"],
        q_false_basin: float = D["q_false_basin"],
        rmsd_drift: float = D["rmsd_drift"],
        rmsf_frozen: float = D["rmsf_frozen"],
        ss_change_thr: float = D["ss_change_thr"],
        rmsd_misfold: float = D.get("rmsd_misfold", 6.0),
        q_misfold: float = D.get("q_misfold", 0.80),
        max_negatives_per_protein: int = D["max_negatives_per_protein"],
        # Sampler
        sampler: str = D["sampler"],
        # Disabled subterms
        disable_subterms: list | None = None,
        # Output
        out_dir: str = "checkpoints/self_consistent",
        # Accept additional kwargs (future-proof, like PhaseConfig)
        **kwargs,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Collection
        self.collect_proteins = collect_proteins
        self.collect_steps = collect_steps
        self.collect_beta = collect_beta
        self.collect_betas = collect_betas  # None = use LogUniform(beta_min, beta_max)
        self.collect_beta_min = collect_beta_min
        self.collect_beta_max = collect_beta_max
        self.collect_save_every = collect_save_every
        self.collect_step_size = collect_step_size
        self.collect_force_cap = collect_force_cap
        self.collect_n_workers = collect_n_workers
        self.collect_max_len = collect_max_len
        self.sampler = sampler

        # Retraining — PDB batch losses
        self.retrain_steps = retrain_steps
        self.retrain_lr = retrain_lr
        self.lambda_elt = lambda_elt
        self.lambda_gap = lambda_gap
        self.lambda_discrim = lambda_discrim
        self.lambda_depth = lambda_depth
        self.target_depth = target_depth
        self.lambda_balance = lambda_balance
        self.balance_r = balance_r
        self.balance_r_term = balance_r_term

        # Retraining — model-sampled negative losses
        self.lambda_sampled_hsm = lambda_sampled_hsm
        self.lambda_sampled_qf = lambda_sampled_qf
        self.lambda_sampled_drmsd_funnel = lambda_sampled_drmsd_funnel
        self.lambda_sampled_gap = lambda_sampled_gap
        self.sc_margin = sc_margin
        self.funnel_m = funnel_m
        self.funnel_alpha = funnel_alpha
        self.gap_m = gap_m
        self.gap_alpha = gap_alpha

        # Evaluation
        self.eval_steps = eval_steps
        self.eval_beta = eval_beta
        self.eval_proteins = eval_proteins

        # Collector thresholds
        self.rg_compact = rg_compact
        self.rg_swollen = rg_swollen
        self.q_false_basin = q_false_basin
        self.rmsd_drift = rmsd_drift
        self.rmsf_frozen = rmsf_frozen
        self.ss_change_thr = ss_change_thr
        self.rmsd_misfold = rmsd_misfold
        self.q_misfold = q_misfold
        self.max_negatives_per_protein = max_negatives_per_protein

        # Disabled subterms (lambda clamped to zero after each step)
        self.disable_subterms = disable_subterms or []

        # Output
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Absorb any extra kwargs as attributes (future-proof)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

        # Track best result across rounds — composite score (lower is better)
        self.best_composite = float("inf")
        self.best_round = -1

        # Convergence criteria — consistent with full_stage
        self.converge_q = D.get("converge_q", 0.98)
        self.converge_rmsd = D.get("converge_rmsd", 2.0)
        self.converge_rg_lo = D.get("converge_rg_lo", 95.0)
        self.converge_rg_hi = D.get("converge_rg_hi", 105.0)

    def _clamp_disabled_subterms(self):
        """Clamp lambda of disabled subterms to zero after each optimizer step.

        Maps subterm names to their lambda parameters in the model.
        Same logic as full_phase.py's disable_subterms post-step clamp.
        """
        import math as _math

        _RAW_FLOOR = _math.log(_math.exp(1e-6) - 1)  # softplus^{-1}(~0)

        _SUBTERM_MAP = {
            "geom": ("packing", "lambda_geom_raw"),
            "contact": ("packing", "lambda_hp_raw"),
            "ram": ("secondary", "lambda_ram_raw"),
            "hb_alpha": ("secondary", "lambda_hb_alpha_raw"),
            "hb_beta": ("secondary", "lambda_hb_beta_raw"),
        }

        with torch.no_grad():
            for subterm in self.disable_subterms:
                if subterm in _SUBTERM_MAP:
                    module_name, param_name = _SUBTERM_MAP[subterm]
                    module = getattr(self.model, module_name, None)
                    if module is not None and hasattr(module, param_name):
                        getattr(module, param_name).data.clamp_(max=_RAW_FLOOR)

    def _hsm_on_negatives(
        self,
        negatives_batch: List[NegativeExample],
    ) -> torch.Tensor:
        """Cartesian-space score matching — grouped by protein, no padding.

        Target: ∇_R E(R_neg) ≈ R_neg - R_native  (harmonic restoring force)
        Loss:   ||∇_R E(R_neg) - (R_neg - R_native)||² / (3L)

        Groups negatives by protein (same chain length L), so each batch
        has uniform size — no padding, no NaN from zero-distance atoms.

        Args:
            negatives_batch: sampled NegativeExample list

        Returns:
            Scalar loss tensor (mean over all negatives)
        """
        if not negatives_batch:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Group by protein (same L, same seq)
        by_protein: Dict[str, List[NegativeExample]] = {}
        for neg in negatives_batch:
            key = f"{neg.pdb_id}_{neg.chain_id}"
            if key not in by_protein:
                by_protein[key] = []
            by_protein[key].append(neg)

        total_loss = torch.tensor(0.0, device=self.device)
        n_valid = 0

        for key, negs in by_protein.items():
            L = negs[0].R_native.shape[0]
            N = len(negs)

            # Stack all negatives + natives — same L, no padding needed
            R_neg_t = torch.stack([n.R_negative for n in negs]).to(self.device)  # (N, L, 3)
            R_nat_t = torch.stack([n.R_native for n in negs]).to(self.device)  # (N, L, 3)
            seq_t = negs[0].seq.unsqueeze(0).expand(N, -1).to(self.device)  # (N, L)
            lengths_t = torch.tensor([L] * N, device=self.device)

            R_neg_t = R_neg_t.detach().requires_grad_(True)

            try:
                # Single batched forward pass — all negs for this protein
                E = self.model(R_neg_t, seq_t, lengths=lengths_t)  # (N,)
                if isinstance(E, tuple):
                    E = E[0]

                # Single batched backward
                grad_R = torch.autograd.grad(
                    E.sum(),
                    R_neg_t,
                    create_graph=True,
                )[
                    0
                ]  # (N, L, 3)

                # Target: Cartesian displacement
                delta = R_neg_t.detach() - R_nat_t  # (N, L, 3)

                # Per-sample loss — no masking needed (no padding)
                sq_err = (grad_R - delta).pow(2)  # (N, L, 3)
                per_sample = sq_err.sum(dim=(1, 2)) / (3.0 * L)  # (N,)

                valid = torch.isfinite(per_sample)
                if valid.any():
                    total_loss = total_loss + per_sample[valid].sum()
                    n_valid += int(valid.sum().item())

            except Exception as e:
                logger.debug("Cartesian HSM error for %s: %s", key, e)
                continue

        if n_valid > 0:
            return total_loss / n_valid
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _funnel_gap_on_negatives(
        self,
        negatives_batch: List[NegativeExample],
        funnel_margin: float = 0.5,
        min_dq: float = 0.05,
        min_ddelta: float = 0.05,
        gap_margin: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Combined QF + dRMSD-funnel + Gap on sampled negatives.

        ONE batched model forward per protein group (all negs + native, same L).
        Uses pairwise energy ordering with margin (no slope division).

        Returns dict with losses AND diagnostic counts (single source of truth):
            qf, rg, gap:           loss tensors
            n_qf_pairs, n_qf_anti: Q-funnel pair/violation counts
            n_rg_pairs, n_rg_anti: dRMSD-funnel pair/violation counts
            mean_slope:            mean dE/dQ slope for Q-funnel pairs
        """
        # Group negatives by protein
        by_protein: Dict[str, List[NegativeExample]] = {}
        for neg in negatives_batch:
            key = f"{neg.pdb_id}_{neg.chain_id}"
            if key not in by_protein:
                by_protein[key] = []
            by_protein[key].append(neg)

        total_qf = torch.tensor(0.0, device=self.device)
        total_rg = torch.tensor(0.0, device=self.device)
        total_gap = torch.tensor(0.0, device=self.device)
        n_qf, n_rg, n_gap = 0, 0, 0

        # Diagnostic counters (computed alongside loss — no separate recomputation)
        _total_qf_anti = 0
        _total_rg_anti = 0
        _total_qf_pairs = 0
        _total_rg_pairs = 0
        _slopes = []

        for key, negs in by_protein.items():
            L = negs[0].R_native.shape[0]
            N = len(negs) + 1  # native + negatives

            # Batch all structures — same L, no padding needed
            R_batch = torch.zeros(N, L, 3, device=self.device)
            R_batch[0] = negs[0].R_native.to(self.device)
            q_vals = [1.0]
            for i, neg in enumerate(negs):
                R_batch[i + 1] = neg.R_negative.to(self.device)
                q_vals.append(neg.q)

            seq_batch = negs[0].seq.unsqueeze(0).expand(N, -1).to(self.device)
            lengths_batch = torch.tensor([L] * N, device=self.device)

            if N < 2:
                continue

            # ── ONE batched model forward ────────────────────────────
            with torch.enable_grad():
                E_all = self.model(R_batch, seq_batch, lengths=lengths_batch)
                if isinstance(E_all, tuple):
                    E_all = E_all[0]

            Q_all = torch.tensor(q_vals, device=self.device)

            # ── Q-funnel (pairwise energy ordering) ──────────────────
            from calphaebm.training.losses.elt_losses import q_funnel_loss

            _qf_loss, _qf_n, _ = q_funnel_loss(
                E_all, Q_all, m=self.funnel_m, alpha=self.funnel_alpha, threshold=min_dq, clamp_max=5.0
            )
            if _qf_n > 0:
                total_qf = total_qf + _qf_loss * _qf_n
                n_qf += _qf_n

            # ── Diagnostic counts for Q-funnel (detached, no grad) ───
            with torch.no_grad():
                dQ = Q_all.unsqueeze(1) - Q_all.unsqueeze(0)
                dE = E_all.detach().unsqueeze(1) - E_all.detach().unsqueeze(0)
                valid_q = dQ > min_dq
                n_valid_q = int(valid_q.sum().item())
                _total_qf_pairs += n_valid_q
                if n_valid_q > 0:
                    # Anti-funnel: higher-Q has higher energy (dE > 0 when dQ > 0)
                    _total_qf_anti += int(((dE > 0) & valid_q).sum().item())
                    # Slopes for mean_slope
                    sl = (dE / dQ.clamp(min=min_dq))[valid_q]
                    _slopes.extend(sl.tolist())

            # ── dRMSD-funnel ──────────────────────────────────────────
            # Full dRMSD computed from stored R_native / R_negative.
            # D_native materialised once per protein group; D_neg reuses
            # R_batch already on device — no extra cdist for R_native.
            R_native_dev = R_batch[0]  # (L, 3)
            D_native = torch.cdist(R_native_dev.unsqueeze(0), R_native_dev.unsqueeze(0)).squeeze(0)  # (L, L)
            idx_r = torch.arange(L, device=self.device)
            triu_mask = ((idx_r.unsqueeze(0) - idx_r.unsqueeze(1)).abs() >= 4) & (
                idx_r.unsqueeze(0) > idx_r.unsqueeze(1)
            )  # upper triangle
            d_nat_flat = D_native[triu_mask]  # (n_pairs,)

            drmsd_vals = [0.0]  # native
            for i in range(1, N):
                D_i = torch.cdist(R_batch[i].unsqueeze(0), R_batch[i].unsqueeze(0)).squeeze(0)
                d_i_flat = D_i[triu_mask]
                drmsd_i = (
                    float(torch.sqrt(((d_i_flat - d_nat_flat) ** 2).mean()).item()) if d_nat_flat.numel() > 0 else 0.0
                )
                drmsd_vals.append(drmsd_i)

            drmsd_all_t = torch.tensor(drmsd_vals, device=self.device)

            from calphaebm.training.losses.elt_losses import drmsd_funnel_loss

            _dr_loss, _dr_n, _ = drmsd_funnel_loss(
                E_all, drmsd_all_t, m=self.funnel_m, alpha=self.funnel_alpha, threshold=0.5, clamp_max=5.0
            )
            if _dr_n > 0:
                total_rg = total_rg + _dr_loss * _dr_n
                n_rg += _dr_n

            # ── Diagnostic counts for dRMSD-funnel (detached) ─────────
            with torch.no_grad():
                dd = drmsd_all_t.unsqueeze(1) - drmsd_all_t.unsqueeze(0)
                valid_dr = dd > 0.5
                n_valid_dr = int(valid_dr.sum().item())
                _total_rg_pairs += n_valid_dr
                if n_valid_dr > 0:
                    dE_dr = E_all.detach().unsqueeze(1) - E_all.detach().unsqueeze(0)
                    # Anti-funnel: i has higher dRMSD but lower/equal energy
                    _total_rg_anti += int(((dE_dr <= 0) & valid_dr).sum().item())

            # ── Gap ──────────────────────────────────────────────────
            # Q-scaled saturating margin: near-native negs get small margin
            Q_neg_t = torch.tensor([neg.q for neg in negs], device=self.device, dtype=torch.float32)
            delta_Q = (1.0 - Q_neg_t).clamp(min=0.0)
            from calphaebm.training.losses.elt_losses import _saturating_margin

            required_gap = _saturating_margin(delta_Q, self.gap_m, self.gap_alpha)
            gaps = E_all[1:] - E_all[0] - required_gap
            gap_loss = torch.exp((-gaps).clamp(max=5.0))
            total_gap = total_gap + gap_loss.sum()
            n_gap += len(negs)

        return {
            "qf": total_qf / max(n_qf, 1),
            "drmsd": total_rg / max(n_rg, 1),
            "gap": total_gap / max(n_gap, 1),
            # Diagnostic counts — single source of truth (#26)
            "n_qf_pairs": _total_qf_pairs,
            "n_qf_anti": _total_qf_anti,
            "n_rg_pairs": _total_rg_pairs,
            "n_rg_anti": _total_rg_anti,
            "mean_slope": sum(_slopes) / max(len(_slopes), 1) if _slopes else 0.0,
        }

    def _collect_negatives(
        self,
        round_num: int,
    ) -> Tuple[List[NegativeExample], CollectionStats]:
        """Run CG Langevin and collect failure configurations."""

        logger.info("=" * 66)
        logger.info("  ROUND %d — COLLECTING NEGATIVES", round_num)

        # Multi-β collection: either discrete betas list or LogUniform per protein
        import math as _math
        import random as _rng

        if self.collect_betas is not None and isinstance(self.collect_betas, list):
            # Legacy: discrete betas, split proteins across them
            betas = self.collect_betas
            n_per_beta = max(1, self.collect_proteins // len(betas))
            total_proteins = n_per_beta * len(betas)
            logger.info(
                "  %d proteins/β × %d betas %s = %d total (sampler=%s, η=%.0e)",
                n_per_beta,
                len(betas),
                betas,
                total_proteins,
                self.sampler,
                self.collect_step_size,
            )
        else:
            # LogUniform β per protein: each protein gets its own temperature
            total_proteins = self.collect_proteins
            betas = None  # will sample per-protein below
            logger.info(
                "  %d proteins × β=L×LogU[%.2f,%.1f] (sampler=%s, η=%.0e)",
                total_proteins,
                self.collect_beta_min,
                self.collect_beta_max,
                self.sampler,
                self.collect_step_size,
            )

        logger.info("  %d steps/protein, %d parallel workers", self.collect_steps, self.collect_n_workers)
        logger.info("=" * 66)

        collector = NegativeCollector(
            model=self.model,
            device=self.device,
            step_size=self.collect_step_size,
            force_cap=self.collect_force_cap,
            save_every=self.collect_save_every,
            rg_compact=self.rg_compact,
            rg_swollen=self.rg_swollen,
            q_false_basin=self.q_false_basin,
            rmsd_drift=self.rmsd_drift,
            rmsf_frozen=self.rmsf_frozen,
            ss_change_thr=self.ss_change_thr,
            rmsd_misfold=self.rmsd_misfold,
            q_misfold=self.q_misfold,
            max_negatives_per_protein=self.max_negatives_per_protein,
            sampler=self.sampler,
        )

        # Build tasks for ALL betas, then run in ONE parallel pool

        # Gather all available structures
        all_structures = []
        for batch in self.train_loader:
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
                if L > self.collect_max_len:
                    continue
                all_structures.append((R_batch[i : i + 1, :L], seq_batch[i : i + 1, :L], pdb_id, chain_id, L))

        # Build tasks
        all_tasks = []
        if betas is not None:
            # Discrete betas: sample different proteins for each beta
            for beta in betas:
                rng = _rng.Random(42 + round_num * 1000 + int(beta))
                n_sample = min(n_per_beta, len(all_structures))
                selected = rng.sample(all_structures, n_sample)
                logger.info("  β=%6.0f: %d proteins selected", beta, n_sample)
                for R_i, seq_i, pdb_id, chain_id, L in selected:
                    all_tasks.append((R_i.cpu(), seq_i.cpu(), beta, self.collect_steps, pdb_id, chain_id))
        else:
            # β = L × s, s ~ LogU[beta_min, beta_max]: scales with chain length, varied across rounds.
            rng = _rng.Random(42 + round_num * 1000)
            n_sample = min(total_proteins, len(all_structures))
            selected = rng.sample(all_structures, n_sample)
            _log_smin = _math.log(self.collect_beta_min)
            _log_smax = _math.log(self.collect_beta_max)
            for R_i, seq_i, pdb_id, chain_id, L in selected:
                scale = _math.exp(rng.uniform(_log_smin, _log_smax))
                beta_i = float(L) * scale
                all_tasks.append((R_i.cpu(), seq_i.cpu(), beta_i, self.collect_steps, pdb_id, chain_id))
            sampled_betas = [t[2] for t in all_tasks]
            logger.info(
                "  β=L×LogU[%.2f,%.1f] per protein: min=%.0f  max=%.0f  mean=%.0f",
                self.collect_beta_min,
                self.collect_beta_max,
                min(sampled_betas),
                max(sampled_betas),
                sum(sampled_betas) / len(sampled_betas),
            )

        logger.info("  Total tasks: %d, workers: %d", len(all_tasks), self.collect_n_workers)

        # Run all tasks in one parallel pool
        all_negatives = collector._collect_parallel(all_tasks, self.collect_n_workers)

        # Build stats
        total_stats = CollectionStats(n_steps=self.collect_steps)
        total_stats.n_proteins = total_proteins
        total_stats.n_negatives = len(all_negatives)
        total_stats.proteins_with_failures = len(set((n.pdb_id, n.chain_id) for n in all_negatives))
        for neg in all_negatives:
            cat = neg.category.value
            total_stats.category_counts[cat] = total_stats.category_counts.get(cat, 0) + 1

        import time

        total_stats.wall_time_sec = 0  # set by _collect_parallel timing

        logger.info(
            "  Multi-β total: %d negatives from %d/%d proteins (%s)",
            total_stats.n_negatives,
            total_stats.proteins_with_failures,
            total_stats.n_proteins,
            ", ".join(f"{k}={v}" for k, v in sorted(total_stats.category_counts.items())),
        )

        return all_negatives, total_stats

    def _retrain(
        self,
        negatives: List[NegativeExample],
        round_num: int,
    ) -> Dict[str, float]:
        """Retrain: PDB(depth+bal+discrim) + Sampled(hsm+qf+drmsd+gap).

        PDB batch losses anchor native energy, maintain subterm balance,
        and ensure subterm discrimination — all without NeRF.
        Sampled HSM teaches ∇E at failure configs to point toward native.
        Sampled gap teaches E(native) < E(failure) — energy ordering.
        Sampled Q-funnel teaches dE/dQ slopes from real dynamics.
        Sampled dRMSD-funnel: lower full-pairwise dRMSD → lower energy.
          dRMSD catches topology errors (strand-swaps,
          register shifts) that satisfy the Rg constraint.
        """

        logger.info("=" * 66)
        logger.info("  ROUND %d — RETRAINING (%d steps, %d negatives)", round_num, self.retrain_steps, len(negatives))
        logger.info("  PDB batch: depth + balance + discrim (Cartesian noise, no NeRF)")
        logger.info("  Sampled:   HSM + Q-funnel + dRMSD-funnel + gap (Cartesian, fully batched)")
        logger.info(
            "  PDB:    λ_depth=%.2f  λ_bal=%.2e→%.3f (round %d/10)  λ_disc=%.2f",
            self.lambda_depth,
            1e-6,
            self.lambda_balance,
            round_num,
            self.lambda_discrim,
        )
        logger.info(
            "  Sampled: λ_hsm=%.2f  λ_qf=%.2f  λ_drmsd=%.2f  λ_gap=%.2f  margin=%.2f  lr=%.1e",
            self.lambda_sampled_hsm,
            self.lambda_sampled_qf,
            self.lambda_sampled_drmsd_funnel,
            self.lambda_sampled_gap,
            self.sc_margin,
            self.retrain_lr,
        )
        logger.info("=" * 66)

        self.model.train()
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.retrain_lr,
        )

        # Cosine schedule: lr → lr/10 over retrain_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.retrain_steps,
            eta_min=self.retrain_lr / 10,
        )

        # Sample negatives into mini-batches for training
        # Each step uses a random subset of negatives + a fresh PDB batch
        train_iter = iter(self.train_loader)
        log_every = D["log_every"]
        save_every = D["save_every"]
        diag_logger = DiagnosticLogger(self.model, self.device)

        running_loss = 0.0
        running_s_hsm = 0.0
        running_s_qf = 0.0
        running_s_drmsd = 0.0
        running_sampled_gap = 0.0
        running_discrim = 0.0
        running_qf = 0.0
        running_gap = 0.0
        running_depth = 0.0
        running_balance = 0.0
        # Funnel diagnostic counts from training loss (#26 single source of truth)
        _latest_funnel_counts = {
            "mean_slope": 0.0,
            "n_qf_pairs": 0,
            "n_qf_anti": 0,
            "n_rg_pairs": 0,
            "n_rg_anti": 0,
        }
        _lambda_bal_eff = self.lambda_balance  # updated per step by ramp
        _last_avg_depth = 0.0  # preserved across log resets for diagnostic block

        for step in range(1, self.retrain_steps + 1):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)

            # ── PDB batch losses (depth, balance, ELT) ──────────────────────────
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            R, seq = batch[0], batch[1]
            lengths = batch[4] if len(batch) >= 5 else None
            R = R.to(self.device)
            seq = seq.to(self.device)
            B, L_max, _ = R.shape
            if lengths is not None:
                lengths = lengths.to(self.device)
            else:
                lengths = torch.tensor(
                    [int(seq[b].ne(-1).sum()) if seq[b].min() < 0 else L_max for b in range(B)],
                    device=self.device,
                )

            # ══ SC retrain: PDB batch losses (no NeRF) ══════════════════
            # Depth + balance + discrim from PDB batches.
            # are replaced by Cartesian-space sampled losses below.
            # Only depth + balance remain from PDB batches (no NeRF needed).

            _loss_discrim_val = None
            _discrim_diag_step = None
            _elt_diag_step = None
            _loss_funnel_val = None
            _loss_elt_gap_val = None

            # ── Native depth loss (no NeRF — just model forward) ─────
            _e_native_val = None
            if self.lambda_depth > 0:
                try:
                    E_nat = self.model(R, seq, lengths=lengths).mean()
                    _e_native_val = float(E_nat.item())
                    exponent = (E_nat - self.target_depth).clamp(max=5.0)
                    loss_depth = torch.exp(exponent)
                    if torch.isfinite(loss_depth):
                        total_loss = total_loss + self.lambda_depth * loss_depth
                        running_depth += loss_depth.item()
                except Exception as e:
                    logger.debug("Depth error at step %d: %s", step, e)

            # ── Energy balance loss (no NeRF — just subterm measurement) ─
            _balance_absmeans = None
            _balance_term_absmeans = None
            if self.lambda_balance > 0:
                try:
                    # Ramp lambda_balance from 1e-6 → lambda_balance over 10 rounds
                    # mirrors full-stage ramp over round 1 = 5000 steps
                    _bal_t = min(1.0, (round_num - 1) / 10.0)
                    _lambda_bal_eff = 1e-6 + (self.lambda_balance - 1e-6) * _bal_t
                    loss_bal, _balance_absmeans, _balance_term_absmeans = energy_balance_loss(
                        self.model,
                        R,
                        seq,
                        r=float(getattr(self, "balance_r", 7.0)),
                        r_term=float(getattr(self, "balance_r_term", 4.0)),
                        lengths=lengths,
                        exclude_subterms=set(self.disable_subterms),
                    )
                    if torch.isfinite(loss_bal):
                        total_loss = total_loss + _lambda_bal_eff * loss_bal
                        running_balance += loss_bal.item()
                except Exception as e:
                    logger.debug("Balance error at step %d: %s", step, e)

            # ── Cartesian discrimination loss (batched, no NeRF) ────────
            # Stack native + perturbed into (2B, L_max, 3), ONE call per subterm
            if self.lambda_discrim > 0 and step % 2 == 0:
                try:
                    sigma = math.exp(random.uniform(math.log(0.3), math.log(3.0)))
                    noise = sigma * torch.randn_like(R)
                    arange_d = torch.arange(R.shape[1], device=self.device)
                    mask_d = (arange_d.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(2)
                    R_pert = R + noise * mask_d.float()

                    # Stack: even=native, odd=perturbed
                    R_disc = torch.cat([R, R_pert], dim=0)  # (2B, L_max, 3)
                    seq_disc = torch.cat([seq, seq], dim=0)  # (2B, L_max)
                    lens_disc = torch.cat([lengths, lengths], dim=0)  # (2B,)

                    discrim_loss = torch.tensor(0.0, device=self.device)
                    n_terms = 0
                    for term_name in ("local", "repulsion", "secondary", "packing"):
                        term = getattr(self.model, term_name, None)
                        if term is None or term_name in self.disable_subterms:
                            continue
                        E_disc = term(R_disc, seq_disc, lengths=lens_disc)  # (2B,)
                        E_nat_sub = E_disc[:B].mean()
                        E_pert_sub = E_disc[B:].mean()
                        gap_sub = E_pert_sub - E_nat_sub
                        discrim_loss = discrim_loss + torch.exp((-gap_sub / 2.0).clamp(max=5.0))
                        n_terms += 1
                    if n_terms > 0:
                        discrim_loss = discrim_loss / n_terms
                        if torch.isfinite(discrim_loss):
                            total_loss = total_loss + self.lambda_discrim * discrim_loss
                            running_discrim += discrim_loss.item()
                            _loss_discrim_val = discrim_loss.item()
                except Exception as e:
                    logger.debug("Discrim error at step %d: %s", step, e)

            # ── Self-consistent losses on model-sampled negatives ────
            # Sample B proteins from negatives pool,
            # all negatives per protein (up to 10, = max_negatives_per_protein).
            # Each protein contributes all its negatives (up to 10)
            # Sampled:   B proteins × up to 10 negatives each
            _loss_s_gap_val = None
            _loss_s_hsm_val = None
            _loss_s_qf_val = None
            _loss_s_drmsd_val = None
            if self.lambda_sampled_gap > 0 and negatives:
                # Group negatives by protein
                neg_by_protein: Dict[str, List[NegativeExample]] = {}
                for neg in negatives:
                    key = f"{neg.pdb_id}_{neg.chain_id}"
                    if key not in neg_by_protein:
                        neg_by_protein[key] = []
                    neg_by_protein[key].append(neg)

                # Sample B proteins (same count as PDB batch)
                protein_keys = list(neg_by_protein.keys())
                n_proteins = min(B, len(protein_keys))
                selected_keys = random.sample(protein_keys, n_proteins)

                # Collect ALL negatives for selected proteins
                neg_batch = []
                for key in selected_keys:
                    neg_batch.extend(neg_by_protein[key])

                # ── Sampled HSM: score at failure should point toward native ──
                try:
                    loss_hsm_neg = self._hsm_on_negatives(neg_batch)
                    if torch.isfinite(loss_hsm_neg):
                        total_loss = total_loss + self.lambda_sampled_hsm * loss_hsm_neg
                        running_s_hsm += loss_hsm_neg.item()
                        _loss_s_hsm_val = loss_hsm_neg.item()
                except Exception as e:
                    if step <= 3:
                        logger.warning("Sampled HSM error at step %d: %s", step, e)

                # ── Combined QF + Rg + Gap (ONE forward per protein group) ──
                try:
                    # Fixed thresholds for Q and Rg funnel pairing
                    _fixed_dq = 0.05
                    _fixed_dd = 0.05
                    fg = self._funnel_gap_on_negatives(
                        neg_batch,
                        gap_margin=self.sc_margin,
                        min_dq=_fixed_dq,
                        min_ddelta=_fixed_dd,
                    )
                    # Gap
                    if torch.isfinite(fg["gap"]):
                        total_loss = total_loss + self.lambda_sampled_gap * fg["gap"]
                        running_sampled_gap += fg["gap"].item()
                        _loss_s_gap_val = fg["gap"].item()
                    # QF
                    if self.lambda_sampled_qf > 0 and torch.isfinite(fg["qf"]):
                        total_loss = total_loss + self.lambda_sampled_qf * fg["qf"]
                        running_s_qf += fg["qf"].item()
                        _loss_s_qf_val = fg["qf"].item()
                    # Rg
                    if self.lambda_sampled_drmsd_funnel > 0 and torch.isfinite(fg["drmsd"]):
                        total_loss = total_loss + self.lambda_sampled_drmsd_funnel * fg["drmsd"]
                        running_s_drmsd += fg["drmsd"].item()
                        _loss_s_drmsd_val = fg["drmsd"].item()
                    # Store diagnostic counts from training loss (#26 single source of truth)
                    _latest_funnel_counts = {
                        "mean_slope": fg.get("mean_slope", 0.0),
                        "n_qf_pairs": fg.get("n_qf_pairs", 0),
                        "n_qf_anti": fg.get("n_qf_anti", 0),
                        "n_rg_pairs": fg.get("n_rg_pairs", 0),
                        "n_rg_anti": fg.get("n_rg_anti", 0),
                    }
                except Exception as e:
                    if step <= 3:
                        logger.warning("Funnel+Gap error at step %d: %s", step, e)

            # ── Backprop ─────────────────────────────────────────────
            if torch.isfinite(total_loss) and total_loss.requires_grad:
                total_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                optimizer.step()

                # Clamp disabled subterms to zero (prevents drift during retraining)
                if self.disable_subterms:
                    self._clamp_disabled_subterms()
            scheduler.step()

            running_loss += total_loss.item()

            # ── EMA update (every step) ──────────────────────────────
            _ema_kw = {"total": total_loss.item()}
            if _loss_funnel_val is not None:
                _ema_kw["qf"] = _loss_funnel_val
            if _loss_elt_gap_val is not None:
                _ema_kw["gap"] = _loss_elt_gap_val
            if _e_native_val is not None:
                _ema_kw["e_native"] = _e_native_val
            if _loss_discrim_val is not None:
                _ema_kw["discrim"] = _loss_discrim_val
            if _loss_s_gap_val is not None:
                _ema_kw["s_gap"] = _loss_s_gap_val
            if _loss_s_hsm_val is not None:
                _ema_kw["s_hsm"] = _loss_s_hsm_val
            if _loss_s_qf_val is not None:
                _ema_kw["s_qf"] = _loss_s_qf_val
            if _loss_s_drmsd_val is not None:
                _ema_kw["s_drmsd"] = _loss_s_drmsd_val
            # Subterm |E| from balance
            if _balance_absmeans:
                for _bk, _bv in _balance_absmeans.items():
                    _ema_kw[f"|{_bk}|"] = _bv
                if "secondary_hb_alpha" in _balance_absmeans:
                    _ema_kw["hb_alpha"] = _balance_absmeans["secondary_hb_alpha"]
                if "secondary_hb_beta" in _balance_absmeans:
                    _ema_kw["hb_beta"] = _balance_absmeans["secondary_hb_beta"]
            # Term-level E%
            if _balance_term_absmeans:
                _abs_total = sum(_balance_term_absmeans.values()) or 1.0
                for _tk in ["local", "secondary", "repulsion", "packing"]:
                    if _tk in _balance_term_absmeans:
                        _ema_kw[f"E%{_tk}"] = 100.0 * _balance_term_absmeans[_tk] / _abs_total
            # ELT diag (af %, slope)
            if _elt_diag_step:
                if "n_anti_funnel" in _elt_diag_step and "n_pairs" in _elt_diag_step:
                    n_total = _elt_diag_step["n_pairs"]
                    if n_total > 0:
                        _ema_kw["anti_funnel"] = 100.0 * _elt_diag_step["n_anti_funnel"] / n_total
                if "mean_slope" in _elt_diag_step:
                    _ema_kw["slope"] = _elt_diag_step["mean_slope"]
                if "gap_mean" in _elt_diag_step:
                    _ema_kw["elt_gap"] = _elt_diag_step["gap_mean"]
            diag_logger.update_ema(**_ema_kw)

            # ── Logging ──────────────────────────────────────────────
            if step % log_every == 0 or step == self.retrain_steps:
                avg_loss = running_loss / log_every
                avg_s_hsm = running_s_hsm / log_every
                avg_s_qf = running_s_qf / log_every
                avg_s_drmsd = running_s_drmsd / log_every
                avg_sampled_gap = running_sampled_gap / log_every
                avg_depth = running_depth / log_every
                avg_bal = running_balance / log_every
                avg_disc = running_discrim / max(log_every // 2, 1)
                lr = optimizer.param_groups[0]["lr"]

                # Effective (λ × raw) format matching full_stage
                parts = [f"loss={avg_loss:.3f}"]
                parts.append(f"depth={self.lambda_depth * avg_depth:.3f}({self.lambda_depth:.1e}x{avg_depth:.1f})")
                parts.append(f"bal={_lambda_bal_eff * avg_bal:.3f}({_lambda_bal_eff:.1e}x{avg_bal:.0f})")
                parts.append(f"disc={self.lambda_discrim * avg_disc:.3f}({self.lambda_discrim:.1e}x{avg_disc:.1f})")
                parts.append(
                    f"s_hsm={self.lambda_sampled_hsm * avg_s_hsm:.3f}({self.lambda_sampled_hsm:.1e}x{avg_s_hsm:.1f})"
                )
                parts.append(
                    f"s_qf={self.lambda_sampled_qf * avg_s_qf:.3f}({self.lambda_sampled_qf:.1e}x{avg_s_qf:.1f})"
                )
                parts.append(
                    f"s_drmsd={self.lambda_sampled_drmsd_funnel * avg_s_drmsd:.3f}({self.lambda_sampled_drmsd_funnel:.1e}x{avg_s_drmsd:.1f})"
                )
                parts.append(
                    f"s_gap={self.lambda_sampled_gap * avg_sampled_gap:.3f}({self.lambda_sampled_gap:.1e}x{avg_sampled_gap:.1f})"
                )
                logger.info(
                    "  [SC round %d] step %d/%d | lr=%.1e | %s",
                    round_num,
                    step,
                    self.retrain_steps,
                    lr,
                    "  ".join(parts),
                )
                running_loss = 0.0
                running_s_hsm = 0.0
                running_s_qf = 0.0
                running_s_drmsd = 0.0
                running_sampled_gap = 0.0
                running_discrim = 0.0
                running_qf = 0.0
                running_gap = 0.0
                _last_avg_depth = avg_depth  # preserve for diagnostic block
                running_depth = 0.0
                running_balance = 0.0

            # ── Periodic checkpoint save ─────────────────────────────
            if step % save_every == 0 and step < self.retrain_steps:
                self._save_checkpoint(round_num, tag=f"step{step:04d}")

            # ── Detailed diagnostics (same cadence as checkpoint) ─────
            if step % save_every == 0:
                try:
                    R_diag = R.detach()
                    seq_diag = seq.detach()
                    lr_now = optimizer.param_groups[0]["lr"]

                    # Build precomputed with funnel + disc data
                    _precomputed_sc = {}
                    # Funnel counts from training loss (#26)
                    _n_qf_pairs = _latest_funnel_counts["n_qf_pairs"]
                    _n_qf_anti = _latest_funnel_counts["n_qf_anti"]
                    _n_dr_pairs = _latest_funnel_counts["n_rg_pairs"]
                    _n_dr_anti = _latest_funnel_counts["n_rg_anti"]
                    _mean_slope = _latest_funnel_counts["mean_slope"]
                    _q_af_pct = 100.0 * _n_qf_anti / max(_n_qf_pairs, 1)
                    _drmsd_af_pct = 100.0 * _n_dr_anti / max(_n_dr_pairs, 1)
                    _precomputed_sc["funnel"] = {
                        "mean_slope": _mean_slope,
                        "q_af": _q_af_pct,
                        "drmsd_af": _drmsd_af_pct,
                        "n_qf_pairs": _n_qf_pairs,
                        "n_qf_anti": _n_qf_anti,
                        "n_dr_pairs": _n_dr_pairs,
                        "n_dr_anti": _n_dr_anti,
                    }
                    # Discrimination gaps from training step
                    if _discrim_diag_step:
                        _precomputed_sc["disc_gaps"] = _discrim_diag_step

                    diag_logger.log_step_block(
                        phase_step=step,
                        n_steps=self.retrain_steps,
                        loss=total_loss.item(),
                        lr=lr_now,
                        R=R_diag,
                        seq=seq_diag,
                        lengths=lengths,
                        # Balance
                        loss_balance=running_balance / max(step, 1),
                        lambda_balance=self.lambda_balance,
                        term_absmeans=_balance_absmeans,
                        term_absmeans_agg=_balance_term_absmeans,
                        # ELT
                        elt_diag=_elt_diag_step,
                        loss_funnel=_loss_funnel_val,
                        lambda_funnel=self.lambda_elt,
                        loss_elt_gap=_loss_elt_gap_val,
                        lambda_gap_elt=self.lambda_gap,
                        # Depth
                        loss_native_depth=_last_avg_depth,
                        lambda_native_depth=self.lambda_depth,
                        target_native_depth=self.target_depth,
                        e_native_depth=_e_native_val,
                        # Disc + Funnel inside the block (#32)
                        precomputed=_precomputed_sc,
                    )

                    # SC-specific: sampled loss summary
                    sc_parts = []
                    if _loss_s_hsm_val is not None:
                        sc_parts.append(
                            f"s_hsm={self.lambda_sampled_hsm * _loss_s_hsm_val:.3f}({self.lambda_sampled_hsm:.1e}x{_loss_s_hsm_val:.1f})"
                        )
                    if _loss_s_qf_val is not None:
                        sc_parts.append(
                            f"s_qf={self.lambda_sampled_qf * _loss_s_qf_val:.3f}({self.lambda_sampled_qf:.1e}x{_loss_s_qf_val:.1f})"
                        )
                    if _loss_s_drmsd_val is not None:
                        sc_parts.append(
                            f"s_drmsd={self.lambda_sampled_drmsd_funnel * _loss_s_drmsd_val:.3f}({self.lambda_sampled_drmsd_funnel:.1e}x{_loss_s_drmsd_val:.1f})"
                        )
                    if _loss_s_gap_val is not None:
                        sc_parts.append(
                            f"s_gap={self.lambda_sampled_gap * _loss_s_gap_val:.3f}({self.lambda_sampled_gap:.1e}x{_loss_s_gap_val:.1f})"
                        )
                    if sc_parts:
                        logger.info("  Sampled:  %s", "  ".join(sc_parts))

                except Exception as e:
                    logger.debug("SC diagnostic error at step %d: %s", step, e)

        self.model.eval()
        return {
            "final_loss": total_loss.item(),
            "final_sc": running_sampled_gap,
        }

    def _evaluate(self, round_num: int) -> Dict[str, float]:
        """Run basin stability evaluation on validation proteins — parallelized.

        Uses fork context with autograd multithreading disabled — same
        pattern as negative collection, which works reliably after GPU retrain.
        """
        import copy
        import gc
        import multiprocessing as mp

        logger.info("=" * 66)
        logger.info("  ROUND %d — EVALUATING (basin stability)", round_num)
        logger.info("=" * 66)

        # Collect structures from val_loader — filter to L ≤ collect_max_len
        max_eval_len = self.collect_max_len  # eval full range, same as collection
        structures = []
        for batch in self.val_loader:
            R_batch, seq_batch = batch[0], batch[1]
            pdb_ids = batch[2] if len(batch) > 2 else [f"unk_{i}" for i in range(R_batch.shape[0])]
            chain_ids = batch[3] if len(batch) > 3 else ["A"] * R_batch.shape[0]
            lengths_batch = batch[4] if len(batch) > 4 else None
            B = R_batch.shape[0]
            for i in range(B):
                if len(structures) >= self.eval_proteins:
                    break
                if lengths_batch is not None:
                    L = int(lengths_batch[i].item())
                elif seq_batch[i].min() < 0:
                    L = int(seq_batch[i].ne(-1).sum())
                else:
                    L = seq_batch[i].shape[0]
                if L > max_eval_len:
                    continue  # skip proteins that are too long
                structures.append(
                    (
                        R_batch[i, :L].cpu().detach(),
                        seq_batch[i, :L].cpu().detach(),
                        pdb_ids[i] if isinstance(pdb_ids, (list, tuple)) else str(pdb_ids),
                        chain_ids[i] if isinstance(chain_ids, (list, tuple)) else str(chain_ids),
                        L,
                    )
                )
            if len(structures) >= self.eval_proteins:
                break

        n_structs = len(structures)
        logger.info(
            "  Running %d structures (L≤%d) at beta=%.1f (%d Langevin steps)...",
            n_structs,
            max_eval_len,
            self.eval_beta,
            self.eval_steps,
        )

        # ── Prepare model for fork — same pattern as negative collection ──
        # 1. Save original device/model refs
        # 2. Move model to CPU BEFORE deepcopy (avoid CUDA tensor in copy)
        # 3. Disable autograd multithreading (prevents fork crash)
        # 4. Fork workers inherit CPU model via COW
        # 5. Restore everything in finally block
        original_device = self.device if hasattr(self, "device") else next(self.model.parameters()).device
        original_model = self.model

        # Move to CPU, deepcopy — keep on CPU until AFTER fork completes.
        # If self.model is on CUDA when fork() happens, children inherit
        # CUDA state and crash. Restore to CUDA in the finally block.
        self.model.cpu()
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        model_cpu = copy.deepcopy(self.model)
        model_cpu.eval()
        # NOTE: self.model stays on CPU — restored in finally block below

        # Build worker args — all tensors CPU
        worker_args = []
        for R, seq_t, pdb_id, chain_id, L in structures:
            worker_args.append(
                (
                    model_cpu,
                    R.unsqueeze(0).cpu(),
                    seq_t.unsqueeze(0).cpu(),
                    torch.tensor([L]),
                    L,
                    1e-4,
                    self.eval_beta,
                    100.0,
                    self.eval_steps,
                )
            )

        # ── Parallel eval via SUBPROCESS — avoids CUDA/autograd fork crash ──
        # After 3000 GPU backward passes, autograd's thread pool persists and
        # corrupts fork.  Solution: launch a fresh Python process that has never
        # touched CUDA, which can fork safely.
        import os
        import subprocess
        import tempfile

        n_workers = min(n_structs, self.eval_proteins)
        logger.info("  Using subprocess wrapper -> %d fork workers (clean process)", n_workers)

        # Save model and structures to temp files
        tmp_dir = tempfile.mkdtemp(prefix="eval_")
        model_path = os.path.join(tmp_dir, "model.pt")
        struct_path = os.path.join(tmp_dir, "structures.pt")
        result_path = os.path.join(tmp_dir, "results.pt")

        torch.save(model_cpu, model_path)
        torch.save({"structures": [(R, seq, pid, cid, L) for R, seq, pid, cid, L in structures]}, struct_path)

        try:
            cmd = [
                sys.executable,
                "-m",
                "calphaebm.evaluation.eval_subprocess",
                "--model-path",
                model_path,
                "--structures-path",
                struct_path,
                "--results-path",
                result_path,
                "--n-workers",
                str(n_workers),
                "--beta",
                str(self.eval_beta),
                "--n-steps",
                str(self.eval_steps),
                "--sampler",
                self.sampler,
            ]
            logger.info("  Launching: %s", " ".join(cmd[-6:]))
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        logger.info("  [subprocess] %s", line)
                proc.wait(timeout=int(D.get("eval_timeout", 86400)))
            except subprocess.TimeoutExpired:
                proc.kill()
                raise RuntimeError("Eval subprocess timed out")
            if proc.returncode != 0:
                stderr_out = proc.stderr.read() if proc.stderr else ""
                logger.warning(
                    "  Subprocess failed (rc=%d): %s", proc.returncode, stderr_out[-500:] if stderr_out else "no stderr"
                )
                raise RuntimeError(f"Eval subprocess failed: rc={proc.returncode}")

            results = torch.load(result_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.error("Subprocess eval failed: %s", e)
            results = [
                {
                    "L": s[4],
                    "e_delta": 0.0,
                    "rmsd": 99.0,
                    "q": 0.0,
                    "rg_ratio": 0.0,
                    "rmsf": 0.0,
                    "error": str(e),
                    "theta": None,
                    "phi": None,
                }
                for s in structures
            ]
        finally:
            # Clean up temp files
            for f in [model_path, struct_path, result_path]:
                if os.path.exists(f):
                    os.remove(f)
            if os.path.exists(tmp_dir):
                os.rmdir(tmp_dir)
            # Restore model to CUDA
            self.model.to(original_device)
            self.model.train()

        # ── Compute Rama/dphi correlations ─────────────────────────────
        rama_corr = 0.0
        dphi_corr = 0.0
        try:
            from calphaebm.training.validation.metrics import (
                compute_delta_phi_correlation,
                compute_ramachandran_correlation,
            )

            ok = [r for r in results if not r.get("error")]
            all_theta = [r["theta"] for r in ok if r.get("theta") is not None]
            all_phi = [r["phi"] for r in ok if r.get("phi") is not None]
            if all_theta and all_phi and len(all_theta) == len(all_phi):
                paired_theta = []
                paired_phi = []
                for th, ph in zip(all_theta, all_phi):
                    n = min(len(th), len(ph))
                    paired_theta.append(th[:n])
                    paired_phi.append(ph[:n])
                theta_cat = torch.tensor(np.concatenate(paired_theta))
                phi_cat = torch.tensor(np.concatenate(paired_phi))
                rama_corr = compute_ramachandran_correlation(theta_cat, phi_cat)
                dphi_corr = compute_delta_phi_correlation(torch.tensor(np.concatenate(all_phi)))
        except Exception as e:
            logger.debug("Rama/dphi correlation error: %s", e)

        # ── Structured eval logging via ValidationLogger ──────────────
        vlog = ValidationLogger()
        return vlog.log_eval_block(
            round_num=round_num,
            beta=self.eval_beta,
            n_steps=self.eval_steps,
            results=results,
            structures=structures,
            rama_corr=rama_corr,
            dphi_corr=dphi_corr,
        )

    def _save_checkpoint(self, round_num: int, tag: str = "") -> Path:
        """Save model checkpoint (round weights + training config)."""
        suffix = f"_{tag}" if tag else ""
        ckpt_path = self.out_dir / f"sc_round{round_num:03d}{suffix}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "round_num": round_num,
                "stage": "self-consistent",
                "training": {
                    "funnel_m": self.funnel_m,
                    "funnel_alpha": self.funnel_alpha,
                    "gap_m": self.gap_m,
                    "gap_alpha": self.gap_alpha,
                    "sc_margin": self.sc_margin,
                    "retrain_lr": self.retrain_lr,
                    "collect_beta_min": self.collect_beta_min,
                    "collect_beta_max": self.collect_beta_max,
                },
            },
            ckpt_path,
        )
        logger.info("  Saved checkpoint: %s", ckpt_path)
        return ckpt_path

    def _load_negatives(self, round_num: int) -> List[NegativeExample]:
        """Load saved negatives from a previous round."""
        neg_dir = self.out_dir / f"negatives_round{round_num:03d}"
        if not neg_dir.exists():
            logger.warning("Negatives dir not found: %s", neg_dir)
            return []

        loaded = []
        for pt_file in sorted(neg_dir.glob("*.pt")):
            data = torch.load(pt_file, map_location=self.device, weights_only=False)
            cat_name = pt_file.stem  # e.g. "drift_preserved"
            try:
                cat = FailureCategory(cat_name)
            except ValueError:
                logger.warning("Unknown category %s in %s — skipping", cat_name, pt_file)
                continue

            n = len(data["pdb_ids"])
            for i in range(n):
                neg = NegativeExample(
                    pdb_id=data["pdb_ids"][i],
                    chain_id=data["chain_ids"][i],
                    seq=data["seqs"][i].to(self.device),
                    R_native=data["R_natives"][i].to(self.device),
                    R_negative=data["R_negatives"][i].to(self.device),
                    category=cat,
                    E_native=data["E_natives"][i],
                    E_negative=data["E_negatives"][i],
                    rg_ratio=data["rg_ratios"][i],
                    q=data["q_values"][i],
                    rmsd=data["rmsd_values"][i],
                    rmsf=0.0,
                    step=data["steps"][i],
                )
                loaded.append(neg)

        logger.info("Loaded %d negatives from round %d (%s)", len(loaded), round_num, neg_dir)
        return loaded

    def _save_negatives(
        self,
        negatives: List[NegativeExample],
        round_num: int,
    ) -> Path:
        """Save collected negatives to disk for analysis/reuse."""
        neg_dir = self.out_dir / f"negatives_round{round_num:03d}"
        neg_dir.mkdir(parents=True, exist_ok=True)

        # Save as individual .pt files grouped by category
        by_cat: Dict[str, List[NegativeExample]] = {}
        for neg in negatives:
            cat = neg.category.value
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(neg)

        for cat, negs in by_cat.items():
            data = {
                "pdb_ids": [n.pdb_id for n in negs],
                "chain_ids": [n.chain_id for n in negs],
                "seqs": [n.seq for n in negs],
                "R_natives": [n.R_native for n in negs],
                "R_negatives": [n.R_negative for n in negs],
                "E_natives": [n.E_native for n in negs],
                "E_negatives": [n.E_negative for n in negs],
                "rg_ratios": [n.rg_ratio for n in negs],
                "q_values": [n.q for n in negs],
                "rmsd_values": [n.rmsd for n in negs],
                "steps": [n.step for n in negs],
            }
            torch.save(data, neg_dir / f"{cat}.pt")

        logger.info("Saved %d negatives to %s", len(negatives), neg_dir)
        return neg_dir

    def run(
        self,
        n_rounds: int = 5,
        convergence_threshold: float = 0.05,
        min_negatives: int = 10,
        resume_round: int = 0,
    ) -> List[RoundResult]:
        """Run the self-consistent training loop.

        Args:
            n_rounds:               maximum number of rounds
            convergence_threshold:  stop if E_delta improvement < this between rounds
            min_negatives:          skip retraining if fewer negatives collected
            resume_round:           resume from this round (0 = start fresh).
                                    Loads model from round{N}.pt and negatives
                                    from rounds 1..N, then continues from N+1.

        Returns:
            List of RoundResult for each completed round
        """
        logger.info("=" * 66)
        logger.info("  SELF-CONSISTENT CG TRAINING")
        logger.info(
            "  %d rounds  |  %d proteins/round  |  %dK steps/protein",
            n_rounds,
            self.collect_proteins,
            self.collect_steps // 1000,
        )
        logger.info(
            "  Retrain: %d steps  |  λ_sampled_gap=%.2f  margin=%.2f",
            self.retrain_steps,
            self.lambda_sampled_gap,
            self.sc_margin,
        )
        if resume_round > 0:
            logger.info("  RESUMING from round %d", resume_round)
        logger.info("=" * 66)

        results: List[RoundResult] = []
        all_negatives: List[NegativeExample] = []  # cumulative across rounds
        prev_e_delta = float("-inf")
        start_round = 1

        # Early stop: read consecutive increases from file (persists across bash invocations)
        _consec_file = self.out_dir / "consecutive_increases.txt"
        _consecutive_increases = 0
        if _consec_file.exists():
            try:
                _consecutive_increases = int(_consec_file.read_text().strip())
                logger.info("  Restored consecutive_increases=%d from %s", _consecutive_increases, _consec_file)
            except Exception:
                _consecutive_increases = 0
        # Track previous round's composite for "consecutive increases" logic
        _prev_score_file = self.out_dir / "prev_round_score.txt"
        _prev_round_score = None
        if _prev_score_file.exists():
            try:
                _prev_round_score = float(_prev_score_file.read_text().strip())
                logger.info("  Restored prev_round_score=%.3f from %s", _prev_round_score, _prev_score_file)
            except Exception:
                _prev_round_score = None
        _conv_consec_file = self.out_dir / "consecutive_converged.txt"
        _consecutive_converged = 0
        if _conv_consec_file.exists():
            try:
                _consecutive_converged = int(_conv_consec_file.read_text().strip())
                logger.info("  Restored consecutive_converged=%d from %s", _consecutive_converged, _conv_consec_file)
            except Exception:
                _consecutive_converged = 0
                _conv_consec_file.write_text("0\n")

        # ── Resume from previous round ────────────────────────────────
        if resume_round > 0:
            ckpt_path = self.out_dir / f"sc_round{resume_round:03d}.pt"
            if ckpt_path.exists():
                from calphaebm.defaults import MODEL as _M
                from calphaebm.utils.checkpoint import apply_config_overrides

                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                self.model.load_state_dict(state_dict, strict=False)
                ckpt_config = ckpt.get("config", ckpt.get("training", {}))
                logger.info("  Loaded model from %s (round=%s)", ckpt_path, ckpt.get("round_num", "?"))

                # Apply current intended values from defaults (replaces manual overrides)
                apply_config_overrides(
                    self.model,
                    ckpt_config,
                    {
                        "packing.rg_lambda": _M["rg_lambda"],
                        "packing.coord_lambda": _M["coord_lambda"],
                        "packing.rg_dead_zone": _M["rg_dead_zone"],
                        "packing.rg_m": _M["rg_m"],
                        "packing.rg_alpha": _M["rg_alpha"],
                        "packing.coord_m": _M["coord_m"],
                        "packing.coord_alpha": _M["coord_alpha"],
                    },
                )

                # Restore best_composite from ALL eval JSONs across rounds
                # Fixes bug: best_composite resets to inf each bash invocation
                import glob as _glob

                _eval_jsons = sorted(_glob.glob(str(self.out_dir / "eval_round*.json")))
                for _ef in _eval_jsons:
                    try:
                        with open(_ef) as _fh:
                            _ed = json.load(_fh)
                        _comp = _ed.get("composite", float("inf"))
                        _rnum = int(Path(_ef).stem.replace("eval_round", ""))
                        if _comp < self.best_composite:
                            self.best_composite = _comp
                            self.best_round = _rnum
                    except Exception:
                        pass
                if self.best_round >= 0:
                    logger.info("  Restored best: Composite=%.3f from round %d", self.best_composite, self.best_round)

                # Eval is decoupled — skip deferred eval, watcher handles it.
                # Just load cached results if they exist for best_composite tracking.
                eval_json_path = self.out_dir / f"eval_round{resume_round:03d}.json"
                if eval_json_path.exists():
                    with open(eval_json_path) as f:
                        eval_stats = json.load(f)
                    logger.info("  Loaded cached eval results from %s", eval_json_path)
                    logger.info(
                        "  Cached eval: E_delta=%.3f  RMSD=%.2f  Q=%.3f  Composite=%.3f",
                        eval_stats.get("e_delta_mean", 0),
                        eval_stats.get("rmsd_mean", 0),
                        eval_stats.get("q_mean", 0),
                        eval_stats.get("composite", 0),
                    )
                else:
                    logger.info("  No cached eval for round %d — skipping (eval decoupled)", resume_round)

                # Model already retrained — skip to next round
                start_round = resume_round + 1
            else:
                logger.info("  No checkpoint for round %d — will redo retraining", resume_round)
                # Model not retrained yet — redo this round's retraining
                start_round = resume_round

            # Load all negatives from rounds 1..resume_round
            for r in range(1, resume_round + 1):
                loaded = self._load_negatives(r)
                all_negatives.extend(loaded)
            logger.info("  Loaded %d cumulative negatives from rounds 1-%d", len(all_negatives), resume_round)

        for round_num in range(start_round, n_rounds + 1):
            # Check if previous run already fired early stop (#37)
            if _consecutive_increases >= 3:
                logger.info(
                    "  SKIPPING round %d: early stop already fired (%d consecutive increases)",
                    round_num,
                    _consecutive_increases,
                )
                break

            t_round = time.time()
            result = RoundResult(round_num=round_num)

            # ── Phase 1: Collect negatives ────────────────────────────
            # Skip collection only if this round's negatives exist on disk
            neg_dir = self.out_dir / f"negatives_round{round_num:03d}"
            if round_num <= resume_round and neg_dir.exists() and len(all_negatives) > 0:
                logger.info("  Round %d: negatives already on disk (%s), skipping collection", round_num, neg_dir)
                result.n_negatives = len(all_negatives)
                new_negatives = []  # no new ones — all pre-loaded
            else:
                t0 = time.time()
                # Disable autograd multithreading before fork-based collection.
                # After GPU retrain, autograd's internal thread pool corrupts
                # fork — disabling it allows the fork to proceed safely.
                _mt_was_enabled = (
                    torch.autograd.is_multithreading_enabled()
                    if hasattr(torch.autograd, "is_multithreading_enabled")
                    else True
                )
                try:
                    if hasattr(torch.autograd, "set_multithreading_enabled"):
                        torch.autograd.set_multithreading_enabled(False)
                    new_negatives, stats = self._collect_negatives(round_num)
                finally:
                    if hasattr(torch.autograd, "set_multithreading_enabled"):
                        torch.autograd.set_multithreading_enabled(_mt_was_enabled)
                result.n_negatives = stats.n_negatives
                result.category_counts = dict(stats.category_counts)
                result.collect_time_sec = time.time() - t0

                # Accumulate negatives across rounds (old failures still relevant)
                all_negatives.extend(new_negatives)

                # Save negatives to disk IMMEDIATELY — before retraining.
                # Retraining takes hours; if killed, negatives are lost otherwise.
                if new_negatives:
                    self._save_negatives(new_negatives, round_num)

                logger.info(
                    "  Round %d: %d new negatives, %d cumulative", round_num, len(new_negatives), len(all_negatives)
                )

                if len(new_negatives) < min_negatives:
                    logger.info(
                        "  Too few negatives (%d < %d) — model may be converged.", len(new_negatives), min_negatives
                    )
                    if round_num > 1:
                        logger.info("  Stopping early (convergence).")
                        results.append(result)
                        break

            # ── Phase 2: Retrain ──────────────────────────────────────
            t0 = time.time()
            train_stats = self._retrain(all_negatives, round_num)
            result.retrain_steps = self.retrain_steps
            result.final_loss = train_stats.get("final_loss", 0.0)
            result.retrain_time_sec = time.time() - t0

            # Save checkpoint
            self._save_checkpoint(round_num)

            # Always write resume marker so external loop can advance
            resume_file = self.out_dir / "resume_next_round.txt"
            resume_file.write_text(str(round_num))
            logger.info("  Wrote resume marker: round %d -> %s", round_num, resume_file)

            # Eval is decoupled — run7_eval_watcher.sh handles it independently.
            # Always exit cleanly for the bash loop to advance to next round.
            logger.info("  Exiting for clean restart (fork safety)...")
            return results

            total_time = time.time() - t_round

            logger.info("─" * 66)
            logger.info("  ROUND %d SUMMARY  (%.0f min total)", round_num, total_time / 60)
            logger.info(
                "    Negatives: %d new (%s)",
                len(new_negatives),
                ", ".join(f"{k}={v}" for k, v in sorted(result.category_counts.items())),
            )
            logger.info(
                "    Basin eval: RMSD=%.2f  Q=%.3f  Rg%%=%.0f%%  ΔE=%.3f  k64dR=%.2f  CO=%.3f  Score=%.3f  %s",
                result.basin_rmsd,
                result.basin_q,
                result.basin_rg_pct,
                result.basin_e_delta,
                result.basin_k64drmsd,
                result.basin_contact_order,
                result.basin_composite,
                "✓ IMPROVED" if result.improved else "",
            )
            logger.info(
                "    Time: collect=%.0fm  retrain=%.0fm  eval=%.0fm",
                result.collect_time_sec / 60,
                result.retrain_time_sec / 60,
                result.eval_time_sec / 60,
            )
            logger.info("    Best so far: Score=%.3f (round %d)", self.best_composite, self.best_round)
            logger.info("─" * 66)

            results.append(result)

            # ── Early stop: 3 consecutive score increases (#37) ─────
            # "Increase" = current score > previous round score (worsening).
            # NOT "didn't beat global best" — that's too aggressive.
            if _prev_round_score is not None and composite > _prev_round_score:
                _consecutive_increases += 1
            else:
                _consecutive_increases = 0
            _prev_round_score = composite
            _consec_file.write_text(f"{_consecutive_increases}\n")
            _prev_score_file.write_text(f"{composite}\n")
            if _consecutive_increases >= 3:
                logger.info(
                    "  EARLY STOP: %d consecutive rounds with increasing composite (worsening)", _consecutive_increases
                )
                # Save best checkpoint as stage_best
                best_ckpt = self.out_dir / f"round{self.best_round:03d}_best.pt"
                stage_best = self.out_dir / "stage_best.pt"
                if best_ckpt.exists():
                    import shutil

                    shutil.copy2(best_ckpt, stage_best)
                    logger.info(
                        "  Saved stage_best.pt from round %d (Composite=%.3f)", self.best_round, self.best_composite
                    )
                break

            # ── Convergence check — Q/RMSD/Rg% criteria ────────────────
            rg_in_band = self.converge_rg_lo <= result.basin_rg_pct <= self.converge_rg_hi
            _converged_now = (
                result.basin_q >= self.converge_q
                and result.basin_rmsd <= self.converge_rmsd
                and result.basin_rmsd > 0
                and rg_in_band
                and result.basin_q_af <= 2.0
                and result.basin_rg_af <= 2.0
            )
            if _converged_now:
                _consecutive_converged += 1
                _conv_consec_file.write_text(f"{_consecutive_converged}\n")
                logger.info(
                    "  CONVERGED (%d/2): Q=%.3f≥%.3f  RMSD=%.2f≤%.1f  Rg%%=%.0f%% in [%.0f,%.0f]  Q_af=%.1f%%≤2%%  Rg_af=%.1f%%≤2%%",
                    _consecutive_converged,
                    result.basin_q,
                    self.converge_q,
                    result.basin_rmsd,
                    self.converge_rmsd,
                    result.basin_rg_pct,
                    self.converge_rg_lo,
                    self.converge_rg_hi,
                    result.basin_q_af,
                    result.basin_rg_af,
                )
                if _consecutive_converged >= 2:
                    _conv_path = self.out_dir / "converged.txt"
                    _conv_path.write_text(f"round {round_num}\n")
                    logger.info("  Wrote convergence marker (2 consecutive): %s", _conv_path)
                    break
            else:
                _consecutive_converged = 0

        # ── Save stage_best.pt from the round with lowest composite ──
        import glob as _glob
        import shutil as _shutil

        _best_comp = float("inf")
        _best_rnum = -1
        for _ef in sorted(_glob.glob(str(self.out_dir / "eval_round*.json"))):
            try:
                with open(_ef) as _fh:
                    _ed = json.load(_fh)
                _comp = _ed.get("composite", float("inf"))
                _rnum = int(Path(_ef).stem.replace("eval_round", ""))
                if _comp < _best_comp:
                    _best_comp = _comp
                    _best_rnum = _rnum
            except Exception:
                pass
        if _best_rnum > 0:
            _src = self.out_dir / f"sc_round{_best_rnum:03d}.pt"
            _dst = self.out_dir / "stage_best.pt"
            if _src.exists():
                _shutil.copy2(str(_src), str(_dst))
                logger.info("  Saved stage_best.pt from round %d (Composite=%.3f)", _best_rnum, _best_comp)
            else:
                logger.warning("  Best round %d checkpoint missing: %s", _best_rnum, _src)

        # ── Final summary ─────────────────────────────────────────────
        logger.info("=" * 66)
        logger.info("  SELF-CONSISTENT TRAINING COMPLETE")
        logger.info("  Rounds: %d  |  Best Score: %.3f (round %d)", len(results), self.best_composite, self.best_round)
        logger.info("  Total negatives collected: %d", len(all_negatives))
        total_time = sum(r.collect_time_sec + r.retrain_time_sec + r.eval_time_sec for r in results)
        logger.info("  Total wall time: %.1f hours", total_time / 3600)
        logger.info("=" * 66)

        return results
