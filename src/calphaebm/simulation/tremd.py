"""
TREMDSimulator — Temperature Replica Exchange for CalphaEBM.

Unlike HREMD which modifies the Hamiltonian (gate scaling), TREMD keeps
the energy function identical across all replicas and varies only β
(inverse temperature). This preserves the landscape shape — the native
basin remains the global minimum at every temperature; only the relative
well depths change. Hot replicas (low β) cross barriers easily; cold
replicas (high β) resolve fine structure.

Swap criterion (same Hamiltonian H, different temperatures β_i, β_j):
    Δ = (β_i - β_j) · (E(x_i) - E(x_j))
    P_swap = min(1, exp(Δ))

When the hot replica finds a low-energy config (E_j < E_i), Δ > 0 and
the swap is accepted — pushing the good config to the cold replica.

β ladder: geometric spacing between β_target (cold) and β_min (hot):
    β_k = β_target · (β_min / β_target)^(k / (N-1))
    k=0: target (coldest),  k=N-1: hottest
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# β ladder utilities
# ─────────────────────────────────────────────────────────────────────────────


def geometric_beta_ladder(beta_target: float, beta_min: float, n_replicas: int) -> List[float]:
    """Geometric β ladder from β_target (cold, idx=0) to β_min (hot, idx=N-1).

    Geometric spacing ensures roughly equal overlap of energy distributions
    between adjacent replicas, giving uniform swap acceptance rates.
    """
    if n_replicas == 1:
        return [beta_target]
    ratio = beta_min / beta_target
    return [beta_target * (ratio ** (k / (n_replicas - 1))) for k in range(n_replicas)]


# ─────────────────────────────────────────────────────────────────────────────
# TREMDSimulator
# ─────────────────────────────────────────────────────────────────────────────


class TREMDSimulator:
    """
    Temperature REMD using CalphaEBM's MALA sampler.

    All replicas share the same energy function (no gate modification).
    Each replica runs MALA at a different β (inverse temperature).
    Swaps exchange coordinates between adjacent-temperature replicas.

    Quick start:
        betas = geometric_beta_ladder(beta_target=100.0, beta_min=5.0, n_replicas=4)
        tremd = TREMDSimulator(
            model=model, seq=seq, lengths=lengths,
            beta_ladder=betas, step_size=3e-5, n_steps_per_swap=200,
        )
        tremd.initialize(start_mode='random', R_native=R_native, R_replicas=R_list)
        tremd.run(n_swaps=2000, log_every=100)
    """

    def __init__(
        self,
        model: nn.Module,
        seq: torch.Tensor,
        lengths: torch.Tensor,
        beta_ladder: List[float],
        step_size: float = 3e-5,
        force_cap: float = 100.0,
        n_steps_per_swap: int = 200,
        swap_scheme: str = "adjacent",
        scale_step_size: bool = True,
        device: str = "cpu",
    ):
        """
        Args:
            beta_ladder:     List of β values, idx=0 is target (coldest).
            scale_step_size: If True, scale step_size by sqrt(β_target/β_k) for
                             each replica so hot replicas take proportionally
                             larger steps. Default True.
        """
        self.model = model.eval()
        self.seq = seq
        self.lengths = lengths
        self.betas = beta_ladder
        self.n = len(beta_ladder)
        self.n_steps_per_swap = n_steps_per_swap
        self.swap_scheme = swap_scheme
        self.device = device

        # Step size per replica (optionally scaled by temperature)
        self._base_step_size = step_size
        if scale_step_size:
            beta_target = beta_ladder[0]
            self._step_sizes = [step_size * math.sqrt(beta_target / max(b, 1e-6)) for b in beta_ladder]
        else:
            self._step_sizes = [step_size] * self.n

        self._force_cap = force_cap
        self.mala: List = []

        # Swap stats: [attempts, accepts] per adjacent pair
        self.swap_stats: List[List[int]] = [[0, 0] for _ in range(self.n - 1)]

        self._traj: List[Dict] = []
        self.global_step = 0
        self.swap_round = 0

        logger.info(
            "TREMDSimulator: %d replicas  β_target=%.1f  β_min=%.1f  " "steps/swap=%d  scheme=%s  scale_step=%s",
            self.n,
            beta_ladder[0],
            beta_ladder[-1],
            n_steps_per_swap,
            swap_scheme,
            scale_step_size,
        )
        logger.info("  β ladder:")
        for i, b in enumerate(beta_ladder):
            tag = "  ← TARGET" if i == 0 else ""
            logger.info("    [%d] β=%7.2f  step_size=%.2e%s", i, b, self._step_sizes[i], tag)

    # ── Public interface ────────────────────────────────────────────────────

    def initialize(
        self,
        start_mode: str = "random",
        R_native: Optional[torch.Tensor] = None,
        R_replicas: Optional[List[torch.Tensor]] = None,
    ):
        """Build and initialize all MALA replicas.

        If R_replicas is provided (list of (1, L, 3) tensors, one per replica),
        these are used directly as starting coords, overriding the start_mode
        generation. This allows the caller to pre-generate and optionally
        minimize independent starting structures per replica.
        """
        from calphaebm.geometry.reconstruct import extract_anchor, nerf_reconstruct
        from calphaebm.simulation.backends.langevin_mala import MALASimulator

        if R_replicas is not None:
            assert len(R_replicas) == self.n, f"R_replicas has {len(R_replicas)} entries but need {self.n} replicas"
            R_init_fn = lambda i: R_replicas[i]

        elif start_mode == "native":
            if R_native is None:
                raise ValueError("R_native required for start_mode=native")
            R_init_fn = lambda _: R_native

        elif start_mode == "extended":
            L = self.lengths[0].item() if self.lengths is not None else R_native.shape[-2]
            theta_ext = torch.full((1, L - 2), 2.0)
            phi_ext = torch.full((1, L - 3), math.pi - 0.01)
            if R_native is not None:
                anch = extract_anchor(R_native)
            else:
                anch = torch.zeros(1, 3, 3)
            R_ext = nerf_reconstruct(theta_ext, phi_ext, anch, bond=3.8)
            R_init_fn = lambda _: R_ext

        else:  # random
            if R_native is None:
                raise ValueError("R_native required to extract anchor for random start")
            L = R_native.shape[-2]
            anch = extract_anchor(R_native)

            def _rand_R(_):
                theta_r = torch.rand(1, L - 2) * (math.pi - 0.2) + 0.1
                phi_r = (torch.rand(1, L - 3) * 2 - 1) * math.pi
                return nerf_reconstruct(theta_r, phi_r, anch, bond=3.8)

            R_init_fn = _rand_R

        self._R_native = R_native

        # All replicas use the SAME model (no gate wrappers) but different β
        self.mala = []
        for i in range(self.n):
            R_init = R_init_fn(i)
            sim = MALASimulator(
                model=self.model,
                seq=self.seq,
                R_init=R_init,
                step_size=self._step_sizes[i],
                beta=self.betas[i],
                force_cap=self._force_cap,
                lengths=self.lengths,
            )
            self.mala.append(sim)

        # Per-replica trajectory buffers
        self._replica_traj: List[List[Dict]] = [[] for _ in range(self.n)]
        self._folded: List[bool] = [False] * self.n
        self._best_Q: List[float] = [0.0] * self.n
        self._best_dRMSD: List[float] = [999.0] * self.n
        self._best_RMSD: List[float] = [999.0] * self.n
        self._best_step: List[int] = [-1] * self.n

        # Native energy (same for all replicas since same Hamiltonian)
        self._E_native = float("nan")
        if R_native is not None:
            with torch.no_grad():
                self._E_native = self.model(R_native, self.seq, self.lengths).item()
            logger.info("  Native energy: %.3f", self._E_native)

        logger.info("Initialized %d replicas  start_mode=%s", self.n, start_mode)

    def run(self, n_swaps: int, log_every: int = 50):
        """
        Main TREMD loop. Each round:
          1. Advance each replica n_steps_per_swap MALA steps.
          2. Compute total energy for each replica (one forward pass each).
          3. Attempt swaps between adjacent-temperature pairs.
          4. Record trajectories and check for folding.
        """
        t0 = time.time()

        for _ in range(n_swaps):
            # 1. MALA steps
            for sim in self.mala:
                for _ in range(self.n_steps_per_swap):
                    sim.step()
            self.global_step += self.n_steps_per_swap

            # 2. Total energies — O(n_replicas) forward passes
            energies = self._compute_energies()

            # 3. Swap attempts
            self._attempt_swaps(energies)

            self.swap_round += 1

            # 4. Structural metrics
            structs = []
            for sim in self.mala:
                structs.append(self._structural_metrics(sim.get_current_R()))

            # 5. Record per-replica snapshots
            for i, (sim, st, E) in enumerate(zip(self.mala, structs, energies)):
                snap = {
                    "step": self.global_step,
                    "E": E,
                    "beta": self.betas[i],
                    "accept": sim.acceptance_rate,
                    "Q": st["Q"],
                    "RMSD": st["RMSD"],
                    "dRMSD": st["dRMSD"],
                    "Rg": st["Rg"],
                    "Rg_pct": st["Rg_pct"],
                    "coords": sim.get_current_R().cpu(),
                }
                self._replica_traj[i].append(snap)

            # 6. Folding detection
            for i, st in enumerate(structs):
                Q = st["Q"]
                dR = st["dRMSD"]
                RM = st["RMSD"]
                if Q == Q and Q > self._best_Q[i]:
                    self._best_Q[i] = Q
                    self._best_step[i] = self.global_step
                if dR == dR and dR < self._best_dRMSD[i]:
                    self._best_dRMSD[i] = dR
                if RM == RM and RM < self._best_RMSD[i]:
                    self._best_RMSD[i] = RM
                if (Q == Q) and (dR == dR):
                    was_folded = self._folded[i]
                    is_folded = (Q >= 0.95) and (dR < 3.0)
                    if is_folded and not was_folded:
                        tag = " *** TARGET ***" if i == 0 else ""
                        logger.info(
                            "  *** FOLDED [replica %d, β=%.1f] at step %d! " "Q=%.3f dRMSD=%.2fÅ RMSD=%.2fÅ%s ***",
                            i,
                            self.betas[i],
                            self.global_step,
                            Q,
                            dR,
                            RM,
                            tag,
                        )
                    self._folded[i] = is_folded

            # 7. Log
            if self.swap_round % log_every == 0:
                self._log(energies, structs, elapsed=time.time() - t0)

    # ── Internal ────────────────────────────────────────────────────────────

    def _compute_energies(self) -> List[float]:
        """Total energy for each replica. Same Hamiltonian, no gates."""
        energies = []
        for sim in self.mala:
            R = sim.get_current_R()
            with torch.no_grad():
                E = self.model(R, self.seq, self.lengths).item()
            energies.append(E)
        return energies

    def _attempt_swaps(self, energies: List[float]):
        pairs = self._swap_pairs()
        for i, j in pairs:
            self._swap(i, j, energies)

    def _swap(self, i: int, j: int, energies: List[float]):
        """
        TREMD swap criterion between replicas i (colder) and j (hotter).

        Δ = (β_i - β_j) · (E_i - E_j)

        When the hot replica found a lower-energy config (E_j < E_i),
        Δ > 0 → swap accepted, pushing the good config to the cold replica.
        """
        beta_i, beta_j = self.betas[i], self.betas[j]
        E_i, E_j = energies[i], energies[j]

        delta = (beta_i - beta_j) * (E_i - E_j)

        pair_idx = min(i, j)
        self.swap_stats[pair_idx][0] += 1

        if math.log(random.random()) < delta:
            self.swap_stats[pair_idx][1] += 1
            self._exchange_coords(i, j)
            # Swap cached energies to stay aligned with configs
            energies[i], energies[j] = energies[j], energies[i]

    def _exchange_coords(self, i: int, j: int):
        """Swap IC coordinates between two MALA replicas."""
        si, sj = self.mala[i], self.mala[j]
        si.theta, sj.theta = sj.theta.clone(), si.theta.clone()
        si.phi, sj.phi = sj.phi.clone(), si.phi.clone()
        if hasattr(si, "anchor") and si.anchor is not None:
            si.anchor, sj.anchor = sj.anchor.clone(), si.anchor.clone()
        si._current_E = None
        sj._current_E = None

    def _swap_pairs(self) -> List[Tuple[int, int]]:
        if self.swap_scheme == "adjacent":
            start = self.swap_round % 2
            return [(k, k + 1) for k in range(start, self.n - 1, 2)]
        elif self.swap_scheme == "all_pairs":
            return [(k, k + 1) for k in range(self.n - 1)]
        else:
            raise ValueError(f"Unknown swap_scheme: {self.swap_scheme}")

    # ── Public queries ──────────────────────────────────────────────────────

    def swap_acceptance_rates(self) -> List[float]:
        """Per-pair acceptance rates. Target: 20-40%."""
        return [acc / max(1, att) for att, acc in self.swap_stats]

    def target_trajectory(self) -> List[Dict]:
        return self._replica_traj[0]

    def replica_trajectories(self) -> List[List[Dict]]:
        return self._replica_traj

    def save_fes(self, output_dir: str, pdb_id: str = "protein"):
        """Save per-replica FES data as .npz files."""
        import os

        import numpy as np

        os.makedirs(output_dir, exist_ok=True)
        for i, traj in enumerate(self._replica_traj):
            if not traj:
                continue
            tag = "target" if i == 0 else f"rep{i:02d}"
            fname = os.path.join(output_dir, f"{pdb_id}_tremd_{tag}.npz")
            np.savez(
                fname,
                Q=[t["Q"] for t in traj],
                dRMSD=[t["dRMSD"] for t in traj],
                RMSD=[t["RMSD"] for t in traj],
                E=[t["E"] for t in traj],
                Rg_pct=[t["Rg_pct"] for t in traj],
                step=[t["step"] for t in traj],
                beta=self.betas[i],
                replica_idx=i,
            )
            logger.info("Saved FES data: %s  (%d snapshots)", fname, len(traj))

    # ── Structural metrics ──────────────────────────────────────────────────

    @torch.no_grad()
    def _structural_metrics(self, R: torch.Tensor) -> Dict[str, float]:
        """Q, RMSD, dRMSD, Rg vs native."""
        nan = float("nan")
        out = {"Q": nan, "RMSD": nan, "dRMSD": nan, "Rg": nan, "Rg_pct": nan}
        if self._R_native is None:
            return out

        R_nat = self._R_native
        L = R.shape[1]
        rc = R[0]  # (L, 3) current
        rn = R_nat[0]  # (L, 3) native

        # Q: fraction native contacts (cutoff 8Å, |i-j|>3)
        def _contacts(coords, cutoff=8.0):
            d = (coords.unsqueeze(0) - coords.unsqueeze(1)).norm(dim=-1)
            mask = torch.ones(L, L, dtype=torch.bool, device=coords.device)
            for k in range(min(4, L)):
                mask.diagonal(k).fill_(False)
                mask.diagonal(-k).fill_(False)
            return (d < cutoff) & mask

        nc = _contacts(rn)
        cc = _contacts(rc)
        n_nat = nc.sum().float()
        out["Q"] = ((cc & nc).sum().float() / n_nat).item() if n_nat > 0 else nan

        # RMSD: Kabsch
        c1 = rc - rc.mean(0)
        c2 = rn - rn.mean(0)
        H = c1.T @ c2
        U, S, Vh = torch.linalg.svd(H)
        d = torch.linalg.det(Vh.T @ U.T)
        D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=R.dtype, device=R.device))
        rot = Vh.T @ D @ U.T
        out["RMSD"] = ((c1 @ rot.T - c2).pow(2).sum(-1).mean()).sqrt().item()

        # dRMSD
        def _pdist(coords):
            d = (coords.unsqueeze(0) - coords.unsqueeze(1)).norm(dim=-1)
            idx = torch.triu_indices(L, L, offset=4)
            return d[idx[0], idx[1]]

        out["dRMSD"] = ((_pdist(rc) - _pdist(rn)).pow(2).mean()).sqrt().item()

        # Rg
        rg = ((rc - rc.mean(0)).pow(2).sum(-1).mean()).sqrt().item()
        rg_flory = 2.0 * (L**0.38)
        out["Rg"] = rg
        out["Rg_pct"] = int(round(100 * rg / rg_flory))

        return out

    def _log(self, energies: List[float], structs: List[Dict], elapsed: float):
        rates = self.swap_acceptance_rates()
        logger.info("  TREMD | step=%7d | round=%4d | elapsed=%.0fs", self.global_step, self.swap_round, elapsed)
        logger.info(
            "  %-5s %7s %10s %+9s %6s %6s %6s %5s %8s %7s",
            "Rep",
            "β",
            "E",
            "ΔE_nat",
            "Q",
            "RMSD",
            "dRMSD",
            "Rg%",
            "swap→",
            "MALA%",
        )
        for i, (E, st) in enumerate(zip(energies, structs)):
            swap = f"{rates[i]:.0%}" if i < self.n - 1 else "—"
            tag = "←" if i == 0 else " "

            def _f(v, fmt):
                return fmt % v if v == v else "n/a"

            dE = E - self._E_native if self._E_native == self._E_native else float("nan")
            logger.info(
                "  [%d]%s %7.1f %10.3f %+9.3f %6s %6s %6s %5s %8s %6.1f%%",
                i,
                tag,
                self.betas[i],
                E,
                dE,
                _f(st["Q"], "%.3f"),
                _f(st["RMSD"], "%.2f"),
                _f(st["dRMSD"], "%.2f"),
                _f(st["Rg_pct"], "%d%%"),
                swap,
                self.mala[i].acceptance_rate * 100,
            )
        # Best-ever per replica
        logger.info("  Best-ever Q per replica:")
        parts = []
        for i in range(self.n):
            bq = self._best_Q[i]
            bd = self._best_dRMSD[i]
            bs = self._best_step[i]
            folded = bq >= 0.95 and bd < 3.0
            mark = " ★FOLDED" if folded else ""
            parts.append(f"  [{i}] β={self.betas[i]:.1f} Q={bq:.3f} dR={bd:.2f}Å @step{bs}{mark}")
        logger.info("\n".join(parts))
