"""
HREMDSimulator — Hamiltonian Replica Exchange for CalphaEBM.

Uses TotalEnergy's existing gate architecture:
  - model.set_gates(local=, repulsion=, secondary=, packing=)
  - model.get_gates() -> dict
  - model.term_energies(R, seq, lengths) -> dict of UNSCALED per-term energies

Each replica runs at the same β=L with a different gate vector:
    E_H(x) = g_local·E_local + g_rep·E_rep + g_ss·E_ss + g_pack·E_pack

Target replica (idx=0): all gates = 1.0
Hot replicas: g_ss and g_pack scaled toward 0; g_rep always 1.0.

Swap criterion (same β, different Hamiltonians i and j):
    ΔU = β · Σ_t (g_j_t - g_i_t) · (E_t(x_i) - E_t(x_j))
    P_swap = min(1, exp(-ΔU))

term_energies() returns unscaled components → swap requires zero extra
forward passes, just a dot product of gate differences × term energy differences.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_TERMS = ("local", "repulsion", "secondary", "packing")


# ─────────────────────────────────────────────────────────────────────────────
# GateVector
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GateVector:
    """
    Outer gate multipliers for one HREMD replica.
    Repulsion is always 1.0 — never soften it (prevents chain crossing).
    Local is always 1.0 — backbone geometry doesn't cause topological trapping.
    The key scaling axes are secondary and packing.
    """

    local: float = 1.0
    repulsion: float = 1.0
    secondary: float = 1.0
    packing: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "local": self.local,
            "repulsion": self.repulsion,
            "secondary": self.secondary,
            "packing": self.packing,
        }

    def gated_energy(self, terms: Dict[str, float]) -> float:
        """Dot this gate vector against a dict of unscaled term energies."""
        g = self.as_dict()
        return sum(g.get(t, 1.0) * terms.get(t, 0.0) for t in _TERMS)

    @staticmethod
    def target() -> "GateVector":
        return GateVector(1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def ladder(n_hot: int, ss_min: float = 0.08, pack_min: float = 0.05) -> List["GateVector"]:
        """
        Geometric ladder: [target, ..., hottest].
        n_hot: number of non-target replicas.
        ss_min / pack_min: gate values at the hottest replica.
        """
        gates = [GateVector.target()]
        for k in range(1, n_hot + 1):
            frac = k / n_hot
            gates.append(
                GateVector(
                    local=1.0,
                    repulsion=1.0,
                    secondary=round(ss_min**frac, 5),
                    packing=round(pack_min**frac, 5),
                )
            )
        return gates

    def __repr__(self):
        return (
            f"GateVector(local={self.local:.2f}, rep={self.repulsion:.2f}, "
            f"ss={self.secondary:.4f}, pack={self.packing:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gated model wrapper — uses TotalEnergy.set_gates() / get_gates() directly
# ─────────────────────────────────────────────────────────────────────────────


class _GatedModelWrapper(nn.Module):
    """
    Wraps TotalEnergy to apply this replica's gate vector during forward().

    CRITICAL: does NOT use set_gates() / fill_() because those are in-place
    operations on register_buffer tensors that participate in the autograd
    graph. In-place modification between forward() and backward() breaks
    PyTorch's version counter and raises RuntimeError.

    Instead: call each sub-module directly with Python float multipliers.
    Python floats never enter the computation graph — safe for autograd.
    """

    def __init__(self, model: nn.Module, gates: GateVector):
        super().__init__()
        self._model = model
        self._gates = gates

    def forward(self, R: torch.Tensor, seq: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        m = self._model
        g = self._gates  # GateVector with Python float fields

        # g.local etc. are plain Python floats — scalar multiplication
        # never creates autograd nodes, so gradient flow is unaffected.
        E = float(g.local) * m.local(R, seq, lengths=lengths)

        if m.repulsion is not None:
            E = E + float(g.repulsion) * m.repulsion(R, seq, lengths=lengths)

        if m.secondary is not None:
            E = E + float(g.secondary) * m.secondary(R, seq, lengths=lengths)

        if m.packing is not None:
            E = E + float(g.packing) * m.packing(R, seq, lengths=lengths)

        return E

    # Proxy everything else to the base model (named_parameters, state_dict, etc.)
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


# ─────────────────────────────────────────────────────────────────────────────
# HREMDSimulator
# ─────────────────────────────────────────────────────────────────────────────


class HREMDSimulator:
    """
    Hamiltonian REMD using CalphaEBM's outer gate architecture.

    Quick start:
        gate_ladder = GateVector.ladder(n_hot=4, ss_min=0.08, pack_min=0.05)
        hremd = HREMDSimulator(
            model=model, seq=seq, lengths=lengths, beta=56.0,
            gate_ladder=gate_ladder, step_size=3e-5, n_steps_per_swap=200,
        )
        hremd.initialize(start_mode='random', R_native=R_native)
        hremd.run(n_swaps=2000, log_every=100)
    """

    def __init__(
        self,
        model: nn.Module,
        seq: torch.Tensor,
        lengths: torch.Tensor,
        beta: float,
        gate_ladder: List[GateVector],
        step_size: float = 3e-5,
        force_cap: float = 100.0,
        n_steps_per_swap: int = 200,
        swap_scheme: str = "adjacent",  # 'adjacent' | 'all_pairs'
        device: str = "cpu",
    ):
        self.model = model.eval()
        self.seq = seq
        self.lengths = lengths
        self.beta = beta
        self.gates = gate_ladder
        self.n = len(gate_ladder)
        self.n_steps_per_swap = n_steps_per_swap
        self.swap_scheme = swap_scheme
        self.device = device

        # Store build params — MALASimulators created lazily in initialize()
        self._step_size = step_size
        self._force_cap = force_cap
        self.mala: List = []

        # Swap stats: [attempts, accepts] per adjacent pair (indexed by lower replica)
        self.swap_stats: List[List[int]] = [[0, 0] for _ in range(self.n - 1)]

        # Target replica trajectory
        self._traj: List[Dict] = []
        self.global_step = 0
        self.swap_round = 0

        logger.info(
            "HREMDSimulator: %d replicas  β=%.1f  steps/swap=%d  scheme=%s", self.n, beta, n_steps_per_swap, swap_scheme
        )
        logger.info("  Gate ladder:")
        for i, g in enumerate(gate_ladder):
            tag = "  ← TARGET" if i == 0 else ""
            logger.info(
                "    [%d] local=%.2f  rep=%.2f  ss=%.4f  pack=%.4f%s",
                i,
                g.local,
                g.repulsion,
                g.secondary,
                g.packing,
                tag,
            )

    # ── Public interface ────────────────────────────────────────────────────

    def initialize(
        self,
        start_mode: str = "random",
        R_native: Optional[torch.Tensor] = None,
        R_replicas: Optional[List[torch.Tensor]] = None,
    ):
        """Build and initialize all MALA replicas.

        MALASimulator(model, seq, R_init, ...) — no start_mode param.
        We generate appropriate R_init here based on start_mode:
          native:   R_init = R_native (same for all replicas)
          random:   each replica gets independent random IC angles → R
          extended: β-sheet extended conformation for all replicas

        If R_replicas is provided (list of (1, L, 3) tensors, one per replica),
        these are used directly as starting coords, overriding the start_mode
        generation. This allows the caller to pre-generate and optionally
        minimize independent starting structures per replica.
        """
        from calphaebm.geometry.reconstruct import extract_anchor, nerf_reconstruct
        from calphaebm.simulation.backends.langevin_mala import MALASimulator

        if R_replicas is not None:
            # Pre-built per-replica coords (possibly minimized by caller)
            assert len(R_replicas) == self.n, f"R_replicas has {len(R_replicas)} entries but need {self.n} replicas"
            R_init_fn = lambda i: R_replicas[i]

        elif start_mode == "native":
            if R_native is None:
                raise ValueError("R_native required for start_mode=native")
            R_init_fn = lambda _: R_native  # same for every replica

        elif start_mode == "extended":
            L = self.lengths[0].item() if self.lengths is not None else R_native.shape[-2]
            import math

            # All-extended: theta≈2.0 rad, phi≈π (anti-parallel)
            theta_ext = torch.full((1, L - 2), 2.0)
            phi_ext = torch.full((1, L - 3), math.pi - 0.01)
            if R_native is not None:
                anch = extract_anchor(R_native)
            else:
                anch = torch.zeros(1, 3, 3)
            R_ext = nerf_reconstruct(theta_ext, phi_ext, anch, bond=3.8)
            R_init_fn = lambda _: R_ext

        else:  # random — each replica gets independent random start
            if R_native is None:
                raise ValueError("R_native required to extract anchor for random start")
            L = R_native.shape[-2]
            anch = extract_anchor(R_native)

            def _rand_R(_):
                import math

                theta_r = torch.rand(1, L - 2) * (math.pi - 0.2) + 0.1
                phi_r = (torch.rand(1, L - 3) * 2 - 1) * math.pi
                return nerf_reconstruct(theta_r, phi_r, anch, bond=3.8)

            R_init_fn = _rand_R

        self._R_native = R_native

        self.mala = []
        for i, g in enumerate(self.gates):
            wrapped = _GatedModelWrapper(self.model, g)
            R_init = R_init_fn(i)
            sim = MALASimulator(
                model=wrapped,
                seq=self.seq,
                R_init=R_init,
                step_size=self._step_size,
                beta=self.beta,
                force_cap=self._force_cap,
                lengths=self.lengths,
            )
            self.mala.append(sim)

        # Per-replica trajectory buffers and folded state trackers
        self._replica_traj: List[List[Dict]] = [[] for _ in range(self.n)]
        self._folded: List[bool] = [False] * self.n
        # Best-ever structural metrics per replica
        self._best_Q: List[float] = [0.0] * self.n
        self._best_dRMSD: List[float] = [999.0] * self.n
        self._best_RMSD: List[float] = [999.0] * self.n
        self._best_step: List[int] = [-1] * self.n
        # Compute native energy under each replica's gated Hamiltonian
        # E_native_i = gates_i · term_energies(R_native)
        self._E_native_per_replica: List[float] = []
        if R_native is not None:
            with torch.no_grad():
                nat_terms = self.model.term_energies(R_native, self.seq, self.lengths)
                nat_terms_f = {k: v.item() for k, v in nat_terms.items()}
            for g in self.gates:
                self._E_native_per_replica.append(g.gated_energy(nat_terms_f))
            logger.info(
                "  Native energies per replica: %s",
                "  ".join(f"[{i}]={e:.3f}" for i, e in enumerate(self._E_native_per_replica)),
            )
        else:
            self._E_native_per_replica = [float("nan")] * self.n

        logger.info("Initialized %d replicas  start_mode=%s", self.n, start_mode)

    def run(self, n_swaps: int, log_every: int = 50):
        """
        Main HREMD loop. Each round:
          1. Advance each replica n_steps_per_swap MALA steps.
          2. Compute unscaled term energies (one forward pass per replica via term_energies()).
          3. Attempt swaps between adjacent pairs.
          4. Record target trajectory.
        """
        t0 = time.time()

        for _ in range(n_swaps):
            # 1. MALA steps
            for sim in self.mala:
                for _ in range(self.n_steps_per_swap):
                    sim.step()
            self.global_step += self.n_steps_per_swap

            # 2. Unscaled component energies — O(n_replicas) forward passes
            term_cache = self._compute_term_energies()

            # 3. Swap attempts — O(n_replicas) dot products, no extra forward passes
            self._attempt_swaps(term_cache)

            self.swap_round += 1

            # 4. Compute structural metrics for all replicas
            structs = []
            for sim in self.mala:
                structs.append(self._structural_metrics(sim.get_current_R()))

            # 5. Save snapshots to per-replica trajectory buffers
            for i, (sim, st, terms) in enumerate(zip(self.mala, structs, term_cache)):
                E_cur = self.gates[i].gated_energy(terms)
                Q_cur = st["Q"]
                dR_cur = st["dRMSD"]
                snap = {
                    "step": self.global_step,
                    "E": E_cur,
                    "E_local": terms.get("local", 0.0),
                    "E_ss": terms.get("secondary", 0.0),
                    "E_pack": terms.get("packing", 0.0),
                    "E_rep": terms.get("repulsion", 0.0),
                    "accept": sim.acceptance_rate,
                    "Q": Q_cur,
                    "RMSD": st["RMSD"],
                    "dRMSD": dR_cur,
                    "Rg": st["Rg"],
                    "Rg_pct": st["Rg_pct"],
                    "coords": sim.get_current_R().cpu(),
                }
                self._replica_traj[i].append(snap)

            # 6. Check for folding events in each replica
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
                # folding detection uses already-computed Q and dR
                if (Q == Q) and (dR == dR):  # not nan
                    was_folded = self._folded[i]
                    is_folded = (Q >= 0.95) and (dR < 3.0)
                    if is_folded and not was_folded:
                        tag = " *** TARGET ***" if i == 0 else ""
                        logger.info(
                            "  *** FOLDED [replica %d] at step %d! " "Q=%.3f dRMSD=%.2fÅ RMSD=%.2fÅ%s ***",
                            i,
                            self.global_step,
                            Q,
                            dR,
                            st["RMSD"],
                            tag,
                        )
                    self._folded[i] = is_folded

            # 7. Log
            if self.swap_round % log_every == 0:
                self._log(term_cache, structs, elapsed=time.time() - t0)

    # ── Internal ────────────────────────────────────────────────────────────

    def _compute_term_energies(self) -> List[Dict[str, float]]:
        """
        Get unscaled term energies for every replica via model.term_energies().
        Gates are NOT applied here — raw per-term values only.
        """
        results = []
        for sim in self.mala:
            R = sim.get_current_R()  # (1, L, 3)
            with torch.no_grad():
                raw = self.model.term_energies(R, self.seq, self.lengths)
            results.append({k: v.item() for k, v in raw.items()})
        return results

    def _attempt_swaps(self, term_cache: List[Dict[str, float]]):
        pairs = self._swap_pairs()
        for i, j in pairs:
            if self._swap(i, j, term_cache):
                # Swap component cache to stay aligned with replica coords
                term_cache[i], term_cache[j] = term_cache[j], term_cache[i]

    def _swap(self, i: int, j: int, term_cache: List[Dict]) -> bool:
        """
        HREMD swap criterion between replicas i (colder) and j (hotter).

        ΔU = β · Σ_t (g_j_t - g_i_t) · (E_t(x_i) - E_t(x_j))

        For scaled terms, g_j_t < g_i_t (j is hotter).
        Compact native-like configs have lower E_t, so moving them from
        hot→cold is thermodynamically favoured → positive swap probability.
        """
        gi, gj = self.gates[i].as_dict(), self.gates[j].as_dict()
        ti, tj = term_cache[i], term_cache[j]

        delta_U = self.beta * sum((gj[t] - gi[t]) * (ti.get(t, 0.0) - tj.get(t, 0.0)) for t in _TERMS)

        self.swap_stats[min(i, j)][0] += 1
        if math.log(random.random()) < -delta_U:
            self.swap_stats[min(i, j)][1] += 1
            self._exchange_coords(i, j)
            return True
        return False

    def _exchange_coords(self, i: int, j: int):
        """Swap IC coordinates (theta, phi, anchor) between two MALA replicas.
        Also invalidates cached energy so next step recomputes correctly."""
        si, sj = self.mala[i], self.mala[j]
        si.theta, sj.theta = sj.theta.clone(), si.theta.clone()
        si.phi, sj.phi = sj.phi.clone(), si.phi.clone()
        if hasattr(si, "anchor") and si.anchor is not None:
            si.anchor, sj.anchor = sj.anchor.clone(), si.anchor.clone()
        # Invalidate cached energies — must recompute after coordinate swap
        si._current_E = None
        sj._current_E = None

    def _swap_pairs(self) -> List[Tuple[int, int]]:
        if self.swap_scheme == "adjacent":
            # Alternate even/odd offsets for ergodicity
            start = self.swap_round % 2
            return [(k, k + 1) for k in range(start, self.n - 1, 2)]
        elif self.swap_scheme == "all_pairs":
            return [(k, k + 1) for k in range(self.n - 1)]
        else:
            raise ValueError(f"Unknown swap_scheme: {self.swap_scheme}")

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def swap_acceptance_rates(self) -> List[float]:
        """Target: 20-40% per adjacent pair. Adjust ladder spacing if outside this."""
        return [acc / max(1, att) for att, acc in self.swap_stats]

    def target_trajectory(self) -> List[Dict]:
        return self._replica_traj[0]

    def replica_trajectories(self) -> List[List[Dict]]:
        """All per-replica snapshot lists. Index 0 = target."""
        return self._replica_traj

    def save_fes(self, output_dir: str, pdb_id: str = "protein"):
        """Save per-replica FES data (Q, dRMSD, E snapshots) as .npz files."""
        import os

        import numpy as np

        out = output_dir
        os.makedirs(out, exist_ok=True)
        for i, traj in enumerate(self._replica_traj):
            if not traj:
                continue
            tag = "target" if i == 0 else f"rep{i:02d}"
            fname = os.path.join(out, f"{pdb_id}_hremd_{tag}.npz")
            Qs = [t["Q"] for t in traj]
            dRs = [t["dRMSD"] for t in traj]
            RMSDs = [t["RMSD"] for t in traj]
            Es = [t["E"] for t in traj]
            Rgs = [t["Rg_pct"] for t in traj]
            steps = [t["step"] for t in traj]
            np.savez(
                fname,
                Q=Qs,
                dRMSD=dRs,
                RMSD=RMSDs,
                E=Es,
                Rg_pct=Rgs,
                step=steps,
                gate_ss=self.gates[i].secondary,
                gate_pack=self.gates[i].packing,
                replica_idx=i,
            )
            logger.info("Saved FES data: %s  (%d snapshots)", fname, len(traj))

    @torch.no_grad()
    def _structural_metrics(self, R: torch.Tensor) -> Dict[str, float]:
        """Full structural metrics vs native: Q, RMSD, dRMSD, Rg, Q_af, dRMSD_af."""
        nan = float("nan")
        out = {"Q": nan, "RMSD": nan, "dRMSD": nan, "Rg": nan, "Rg_pct": nan}
        if self._R_native is None:
            return out

        R_nat = self._R_native  # (1, L, 3)
        L = R.shape[1]
        rc = R[0]  # (L, 3)  current
        rn = R_nat[0]  # (L, 3)  native

        # ── Q: fraction native contacts (cutoff 8Å, |i-j|>3) ──
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

        # ── RMSD: Kabsch superposition ──
        c1 = rc - rc.mean(0)
        c2 = rn - rn.mean(0)
        H = c1.T @ c2
        U, S, Vh = torch.linalg.svd(H)
        d = torch.linalg.det(Vh.T @ U.T)
        D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=R.dtype, device=R.device))
        rot = Vh.T @ D @ U.T
        out["RMSD"] = ((c1 @ rot.T - c2).pow(2).sum(-1).mean()).sqrt().item()

        # ── dRMSD: pairwise distance RMSD (no superposition needed) ──
        def _pdist(coords):
            d = (coords.unsqueeze(0) - coords.unsqueeze(1)).norm(dim=-1)
            idx = torch.triu_indices(L, L, offset=4)
            return d[idx[0], idx[1]]

        pd_cur = _pdist(rc)
        pd_nat = _pdist(rn)
        out["dRMSD"] = ((pd_cur - pd_nat).pow(2).mean()).sqrt().item()

        # ── Rg and Rg% vs Flory scaling ──
        rg = ((rc - rc.mean(0)).pow(2).sum(-1).mean()).sqrt().item()
        rg_flory = 2.0 * (L**0.38)
        out["Rg"] = rg
        out["Rg_pct"] = int(round(100 * rg / rg_flory))

        return out

    def _log(self, term_cache: List[Dict], structs: List[Dict], elapsed: float):
        rates = self.swap_acceptance_rates()
        logger.info("  HREMD | step=%7d | round=%4d | elapsed=%.0fs", self.global_step, self.swap_round, elapsed)
        # header
        logger.info(
            "  %-5s %-10s %-8s %-8s %-8s %-8s %-8s %-6s %-6s %-6s %-5s %-7s %-7s %-8s %-7s",
            "Rep",
            "E_gated",
            "ΔE",
            "local",
            "rep",
            "ss",
            "pack",
            "Q",
            "RMSD",
            "dRMSD",
            "Rg%",
            "g_ss",
            "g_pk",
            "swap→",
            "MALA%",
        )
        for i, (terms, g, st) in enumerate(zip(term_cache, self.gates, structs)):
            swap = f"{rates[i]:.0%}" if i < self.n - 1 else "—"
            tag = "←" if i == 0 else " "

            def _f(v, fmt):
                return fmt % v if v == v else "n/a"

            E_cur = g.gated_energy(terms)
            E_nat = self._E_native_per_replica[i]
            dE = E_cur - E_nat if (E_nat == E_nat) else float("nan")
            logger.info(
                "  [%d]%s %10.3f %+8.3f %8.3f %8.3f %8.3f %8.3f %6s %6s %6s %5s %7.4f %7.4f %8s %6.1f%%",
                i,
                tag,
                E_cur,
                dE,
                terms.get("local", 0.0),
                terms.get("repulsion", 0.0),
                terms.get("secondary", 0.0),
                terms.get("packing", 0.0),
                _f(st["Q"], "%.3f"),
                _f(st["RMSD"], "%.2f"),
                _f(st["dRMSD"], "%.2f"),
                _f(st["Rg_pct"], "%d%%"),
                g.secondary,
                g.packing,
                swap,
                self.mala[i].acceptance_rate * 100,
            )
        # Best-ever summary line
        logger.info("  Best-ever Q per replica:")
        parts = []
        for i in range(self.n):
            tag = "TARGET" if i == 0 else f"rep{i}"
            bq = self._best_Q[i]
            bd = self._best_dRMSD[i]
            bs = self._best_step[i]
            folded = bq >= 0.95 and bd < 3.0
            mark = " ★FOLDED" if folded else ""
            parts.append(f"  [{i}] Q={bq:.3f} dR={bd:.2f}Å @step{bs}{mark}")
        logger.info("\n".join(parts))


# ─────────────────────────────────────────────────────────────────────────────
# Preset gate ladders
# ─────────────────────────────────────────────────────────────────────────────

LADDER_PRESETS: Dict[str, List[GateVector]] = {
    # L=35-70, 4 replicas. Hottest: pack≈0, ss≈0 → near-pure backbone
    "small": [
        GateVector(1.0, 1.0, 1.0000, 1.0000),  # [0] target
        GateVector(1.0, 1.0, 0.5000, 0.3500),  # [1] warm
        GateVector(1.0, 1.0, 0.1500, 0.0800),  # [2] hot
        GateVector(1.0, 1.0, 0.0200, 0.0080),  # [3] very hot
    ],
    # L=70-200, 5 replicas
    "medium": [
        GateVector(1.0, 1.0, 1.0000, 1.0000),
        GateVector(1.0, 1.0, 0.6000, 0.4500),
        GateVector(1.0, 1.0, 0.2500, 0.1500),
        GateVector(1.0, 1.0, 0.0700, 0.0350),
        GateVector(1.0, 1.0, 0.0120, 0.0040),
    ],
    # L=200-512, 6 replicas
    "large": [
        GateVector(1.0, 1.0, 1.0000, 1.0000),
        GateVector(1.0, 1.0, 0.7000, 0.6000),
        GateVector(1.0, 1.0, 0.3800, 0.2800),
        GateVector(1.0, 1.0, 0.1600, 0.0900),
        GateVector(1.0, 1.0, 0.0500, 0.0200),
        GateVector(1.0, 1.0, 0.0100, 0.0030),
    ],
    # OOD (L<40): eliminate packing in hot replica to bypass rho_lo mismatch
    "ood_small": [
        GateVector(1.0, 1.0, 1.0000, 1.0000),
        GateVector(1.0, 1.0, 0.5000, 0.1500),
        GateVector(1.0, 1.0, 0.1500, 0.0150),
        GateVector(1.0, 1.0, 0.0300, 0.0010),  # pack≈0
    ],
    # Diagnostic: only scale secondary (is ss the barrier?)
    "ss_only": [
        GateVector(1.0, 1.0, 1.0000, 1.0000),
        GateVector(1.0, 1.0, 0.3500, 1.0000),
        GateVector(1.0, 1.0, 0.0700, 1.0000),
        GateVector(1.0, 1.0, 0.0080, 1.0000),
    ],
    # Diagnostic: only scale packing (is packing the barrier?)
    "pack_only": [
        GateVector(1.0, 1.0, 1.0000, 1.0000),
        GateVector(1.0, 1.0, 1.0000, 0.3000),
        GateVector(1.0, 1.0, 1.0000, 0.0500),
        GateVector(1.0, 1.0, 1.0000, 0.0060),
    ],
}
