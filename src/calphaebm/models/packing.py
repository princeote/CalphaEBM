# src/calphaebm/models/packing.py
"""
Packing energy and constraint terms for CalphaEBM (v6).

Architecture
============
  E_pack = E_hp_reward + E_rho_reward          (landscape — in energy balance)
         + E_hp_penalty + E_rho_penalty         (constraints — guardrails)
         + E_rg_penalty                         (size guardrail)

5-group physicochemical scheme
-------------------------------
  k=0  core_hydrophobic       PHE, ILE, LEU, MET, VAL
  k=1  amphipathic_hydrophobic ALA, PRO, TRP, TYR
  k=2  positive               HIS, LYS, ARG
  k=3  negative               ASP, GLU
  k=4  polar                  CYS, GLY, ASN, GLN, SER, THR

Per-residue group-conditional coordination
------------------------------------------
  n_i^(k) = Σ_{j:|i-j|>3, G(s_j)=k}  g(r_ij)
  n_i     = Σ_k n_i^(k)               (partition, exact)

Per-chain group-conditional contact density
-------------------------------------------
  ρ^(k) = (1/L) Σ_i n_i^(k)
  ρ     = Σ_k ρ^(k)                   (partition, exact)

E_hp_reward — per-residue product Gaussian (22 trainable params)
-----------------------------------------------------------------
  E_hp = -(λ_hp/L) Σ_i  w_{s_i} · [Π_k exp(-(n_i^(k) - n*^(k)_{s_i})² / 2σ^(k)²_{s_i})] · v_i

  Each factor is a gate: any wrong group suppresses the full reward.
  Trainable: 20 per-AA weights w (softplus, init Kyte-Doolittle) + 1 λ_hp.
  Fixed buffers: n*^(k)_{s_i} [20×5], σ^(k)_{s_i} [20×5].

E_rho_reward — per-chain product Gaussian (1 trainable param)
--------------------------------------------------------------
  E_rho = -λ_ρ · Π_k exp(-(ρ^(k) - ρ*^(k)(L))² / 2σ²_{ρ^(k)})

  ρ*^(k)(L) = a_k - b_k·exp(-L/c_k)  (five saturating-exponential fits from PDB)
  Trainable: 1 λ_ρ.
  Fixed buffers: a_k, b_k, c_k [5 each], σ_{ρ^(k)} [5].

E_hp_penalty — per-residue, per-group saturating exponential (0 trainable)
---------------------------------------------------------------------------
  E_hp_pen = (λ_coord/L) Σ_i Σ_k  P(n_i^(k) outside [n^(k)_lo_{s_i}, n^(k)_hi_{s_i}])
  P(x) = m·(1 - exp(-α·max(0,x)))     (bounded at m, zero inside band)
  Fixed buffers: n^(k)_lo_{s_i}, n^(k)_hi_{s_i} [20×5 each].

E_rho_penalty — per-chain, per-group saturating exponential (0 trainable)
--------------------------------------------------------------------------
  E_rho_pen = λ_{ρ,pen} Σ_k  P(ρ^(k) outside [ρ^(k)_lo(L), ρ^(k)_hi(L)])
  Bounds derived from ρ*^(k)(L) ± 1.35·σ_{ρ^(k)}.

E_rg_penalty — per-chain Flory size restraint (0 trainable)
------------------------------------------------------------
  E_rg_pen = λ_Rg · P(|Rg/Rg* - 1| - δ)
  Rg*(L) = r0·L^ν  (r0=2.0 Å, ν=0.38, dead zone δ=0.30)

Total trainable parameters: 22  (20 w + 1 λ_hp + 1 λ_ρ)
All other quantities are fixed buffers from coordination analysis output.

Sigmoid: Best et al. (2013) Cα-adapted — r_half=8.0 Å, τ=0.2 Å (β=5.0 Å⁻¹)
Unified across Q, n_i^(k), and ρ^(k) computations.

History
-------
  v4 (Run4): r_half=7.0, τ=1.0, scalar n_i, quadratic penalties, no ρ term.
  v5 (Run5): Best sigmoid, scalar n_i Gaussian, scalar ρ Gaussian, exp penalties.
  v6 (Run6): 5-group decomposition, product Gaussian for both E_hp and E_rho,
             per-group penalties for E_hp_pen and E_rho_pen.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calphaebm.geometry.internal import bond_angles, torsions
from calphaebm.geometry.pairs import topk_nonbonded_pairs
from calphaebm.models.learnable_buffers import reg
from calphaebm.models.rama_gate import RamaValidityGate
from calphaebm.utils.logging import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Best et al. (2013) Cα-adapted sigmoid constants
# ---------------------------------------------------------------------------
BEST_R_HALF = 8.0  # Å — sigmoid midpoint
BEST_TAU = 0.2  # Å — steepness (β = 5.0 Å⁻¹)

# ---------------------------------------------------------------------------
# 5-group physicochemical scheme
# AA index order matches AA_ORDER in coordination/cli.py:
#   ALA(0) CYS(1) ASP(2) GLU(3) PHE(4) GLY(5) HIS(6) ILE(7) LYS(8) LEU(9)
#   MET(10) ASN(11) PRO(12) GLN(13) ARG(14) SER(15) THR(16) VAL(17) TRP(18) TYR(19)
# ---------------------------------------------------------------------------
NUM_GROUPS = 5
GROUP_NAMES = [
    "core_hydrophobic",  # 0
    "amphipathic_hydrophobic",  # 1
    "positive",  # 2
    "negative",  # 3
    "polar",  # 4
]

# Maps AA index (0-19) -> group index (0-4)
_AA_GROUP_DEFAULT: list[int] = [
    1,  # ALA  amphipathic
    4,  # CYS  polar
    3,  # ASP  negative
    3,  # GLU  negative
    0,  # PHE  core_hydrophobic
    4,  # GLY  polar
    2,  # HIS  positive
    0,  # ILE  core_hydrophobic
    2,  # LYS  positive
    0,  # LEU  core_hydrophobic
    0,  # MET  core_hydrophobic
    4,  # ASN  polar
    1,  # PRO  amphipathic
    4,  # GLN  polar
    2,  # ARG  positive
    4,  # SER  polar
    4,  # THR  polar
    0,  # VAL  core_hydrophobic
    1,  # TRP  amphipathic
    1,  # TYR  amphipathic
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batch_rg(R: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Differentiable, padding-aware radius of gyration."""
    B, L, _ = R.shape
    if lengths is not None:
        mask = torch.arange(L, device=R.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.float().unsqueeze(2)
        n_atoms = lengths.float().clamp(min=1.0)
        com = (R * mask_f).sum(dim=1) / n_atoms.unsqueeze(1)
        diff = R - com.unsqueeze(1)
        sq = (diff**2).sum(dim=2) * mask.float()
        rg = (sq.sum(dim=1) / n_atoms).sqrt()
    else:
        com = R.mean(dim=1)
        diff = R - com.unsqueeze(1)
        rg = (diff**2).sum(dim=2).mean(dim=1).sqrt()
    return rg


_MISSING_LENGTHS_WARNED = False


def _check_lengths(R: torch.Tensor, lengths, caller: str) -> None:
    global _MISSING_LENGTHS_WARNED
    if lengths is None and R.shape[0] > 1 and not _MISSING_LENGTHS_WARNED:
        logger.warning(
            "[%s] lengths=None with batch_size=%d — padding atoms will corrupt "
            "energy. Pass lengths to all model calls during training.",
            caller,
            R.shape[0],
        )
        _MISSING_LENGTHS_WARNED = True


def _inv_softplus(y: float, eps: float = 1e-8) -> float:
    y = float(max(y, eps))
    return float(np.log(np.expm1(y)))


# Kyte-Doolittle hydrophobicity scale (AA_ORDER order)
KYTE_DOOLITTLE_20 = [
    +1.8,
    +2.5,
    -3.5,
    -3.5,
    +2.8,
    -0.4,
    -3.2,
    +4.5,
    -3.9,
    +3.8,
    +1.9,
    -3.5,
    -1.6,
    -3.5,
    -4.5,
    -0.8,
    -0.7,
    +4.2,
    -0.9,
    -1.3,
]

# ---------------------------------------------------------------------------
# Hydrophobic burial module — product Gaussian over 5 groups  (E_hp_reward)
# ---------------------------------------------------------------------------


class _Hydrophobic(nn.Module):
    """Per-residue grouped coordination packing energy.

    E_hp_i = w_{s_i} · Π_k exp(-(n_i^(k) - n*^(k)_{s_i})² / 2σ^(k)²_{s_i})

    The product form makes each group a gate: any wrong group composition
    suppresses the full reward even if all other groups are correct.

    Parameters (trainable): 20 per-AA weights w + 1 λ_hp = 21
    Fixed buffers:          n_star_group [20,5], sigma_group [20,5]
    """

    COORD_R_HALF = BEST_R_HALF
    COORD_TAU = BEST_TAU

    def __init__(
        self,
        num_aa: int = 20,
        init_lambda: float = 1.0,
        # Group-conditional statistics [num_aa × NUM_GROUPS]
        n_group_mean: Optional[list] = None,  # [[m0..m4], ...] length num_aa
        n_group_std: Optional[list] = None,  # [[s0..s4], ...] length num_aa
        # Group assignment [num_aa]
        group_assignment: Optional[list] = None,
        # Learnable buffer flag
        learn_coords: bool = False,
    ):
        super().__init__()
        self._learn_coords = learn_coords

        # --- per-AA hydrophobicity weights (20 params) ---
        kd = torch.tensor(KYTE_DOOLITTLE_20[:num_aa], dtype=torch.float32)
        kd_norm = 0.1 + 0.9 * (kd - kd.min()) / (kd.max() - kd.min())
        self._w_raw = nn.Parameter(torch.tensor([_inv_softplus(float(v)) for v in kd_norm], dtype=torch.float32))

        # --- λ_hp (1 param) ---
        self._lambda_hp_raw = nn.Parameter(torch.tensor(_inv_softplus(init_lambda), dtype=torch.float32))

        # --- group assignment buffer [num_aa] ---
        ga = group_assignment if group_assignment is not None else _AA_GROUP_DEFAULT
        self.register_buffer(
            "group_assignment",
            torch.tensor(ga[:num_aa], dtype=torch.long),
        )

        # --- n*^(k)_{s_i}: group-conditional coordination centers [num_aa, 5] ---
        if n_group_mean is not None:
            n_star = torch.tensor(n_group_mean[:num_aa], dtype=torch.float32)  # [num_aa, 5]
        else:
            logger.warning("_Hydrophobic: no n_group_mean provided, using defaults 4.0/5")
            n_star = torch.ones(num_aa, NUM_GROUPS, dtype=torch.float32) * (4.0 / NUM_GROUPS)
        assert n_star.shape == (
            num_aa,
            NUM_GROUPS,
        ), f"n_group_mean must be [{num_aa}, {NUM_GROUPS}], got {list(n_star.shape)}"
        reg(self, "n_star_group", n_star, learnable=learn_coords)

        # --- σ^(k)_{s_i}: group-conditional coordination widths [num_aa, 5] ---
        if n_group_std is not None:
            sigma = torch.tensor(n_group_std[:num_aa], dtype=torch.float32).clamp(min=0.8)
        else:
            logger.warning("_Hydrophobic: no n_group_std provided, using sigma=0.8")
            sigma = torch.ones(num_aa, NUM_GROUPS, dtype=torch.float32) * 0.8
        assert sigma.shape == (
            num_aa,
            NUM_GROUPS,
        ), f"n_group_std must be [{num_aa}, {NUM_GROUPS}], got {list(sigma.shape)}"
        reg(self, "sigma_group", sigma, learnable=learn_coords)
        # NOTE: sigma_group is clamped at min=0.8 per group.
        # The product of 5 Gaussians underflows below ~1e-13 for sigma < 0.8
        # in typical misfolded configurations, making gradients effectively dead.
        # PDB-fitted sigma values from the analysis rerun are expected to be
        # 0.8-2.0 naturally; the clamp only activates for sparse AA-group
        # combinations where the estimated std is artificially tight.

        logger.info(
            "_Hydrophobic v6 (grouped product Gaussian): %d AA, %d groups, λ_init=%.2f",
            num_aa,
            NUM_GROUPS,
            init_lambda,
        )
        logger.info(
            "  sigmoid: r_half=%.1f Å, τ=%.2f Å (Best et al. 2013)",
            self.COORD_R_HALF,
            self.COORD_TAU,
        )
        logger.info(
            "  n*_group range: [%.2f, %.2f]  σ_group range: [%.2f, %.2f]",
            self.n_star_group.min().item(),
            self.n_star_group.max().item(),
            self.sigma_group.min().item(),
            self.sigma_group.max().item(),
        )
        if learn_coords:
            logger.info(
                "  n*_group, σ_group: LEARNABLE (%d params)", self.n_star_group.numel() + self.sigma_group.numel()
            )

    @property
    def w(self) -> torch.Tensor:
        return F.softplus(self._w_raw)

    @property
    def lambda_hp(self) -> torch.Tensor:
        return F.softplus(self._lambda_hp_raw) + 1e-6

    def forward(
        self,
        seq: torch.Tensor,  # (B, L) int64 AA indices
        r: torch.Tensor,  # (B, L, K) distances from topk_nonbonded_pairs
        j_idx: torch.Tensor,  # (B, L, K) partner indices
        max_dist: float = 12.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-residue grouped packing energy and group coordination vectors.

        Returns:
            E_hp_i:   (B, L)    per-residue Gaussian product value (positive = favorable)
            n_grouped:(B, L, 5) group-conditional coordination vectors
        """
        B, L, K = r.shape

        # --- sigmoid weights for all k=64 neighbor slots ---
        valid = (r < max_dist - 1e-4).float()
        g_ij = torch.sigmoid((self.COORD_R_HALF - r.clamp(max=max_dist)) / self.COORD_TAU) * valid  # (B, L, K)

        # --- group membership of each neighbor j ---
        # j_idx: (B, L, K) indices into [0, L)
        # seq:   (B, L)    AA index for each position
        # group_assignment: [num_aa] -> group index
        seq_j = seq.gather(
            dim=1,
            index=j_idx.clamp(0, L - 1).view(B, -1),
        ).view(
            B, L, K
        )  # (B, L, K) AA index of neighbor j

        group_j = self.group_assignment[seq_j.clamp(0, len(self.group_assignment) - 1)]
        # group_j: (B, L, K)  group index 0..4 of each neighbor

        # --- n_i^(k): group-conditional coordination [B, L, 5] ---
        # For each group k: n_i^(k) = Σ_{j: group_j == k} g_ij
        group_one_hot = (group_j.unsqueeze(-1) == torch.arange(NUM_GROUPS, device=r.device)).float()  # (B, L, K, 5)
        n_grouped = (g_ij.unsqueeze(-1) * group_one_hot).sum(dim=2)  # (B, L, 5)

        # --- product Gaussian over groups ---
        # E_hp_i = w_{s_i} · Π_k exp(-(n_i^(k) - n*^(k)_{s_i})² / 2σ^(k)²_{s_i})
        n_star_i = self.n_star_group[seq]  # (B, L, 5)
        sigma_i = self.sigma_group[seq]  # (B, L, 5)
        w_i = self.w[seq]  # (B, L)

        log_gauss = -((n_grouped - n_star_i) ** 2) / (2.0 * sigma_i**2)  # (B, L, 5)
        product_gauss = log_gauss.sum(dim=-1).exp()  # (B, L)

        E_hp_i = w_i * product_gauss  # (B, L)

        return E_hp_i, n_grouped


# ---------------------------------------------------------------------------
# Main PackingEnergy module (v6)
# ---------------------------------------------------------------------------


class PackingEnergy(nn.Module):
    """Packing energy v6: grouped coordination, product Gaussians.

    E_pack = E_hp_reward + E_rho_reward   (landscape, in energy balance)
           + E_hp_penalty + E_rho_penalty + E_rg_penalty  (guardrails)

    Trainable parameters: 22  (20 w + 1 λ_hp + 1 λ_ρ)
    Fixed buffers: group statistics from coordination analysis (v6 output).
    """

    def __init__(
        self,
        num_aa: int = 20,
        # --- Neighbour graph ---
        topk: int = 64,
        exclude: int = 3,
        max_dist: float = 12.0,
        # --- Normalisation ---
        normalize_by_length: bool = True,
        # --- Group assignment ---
        group_assignment: Optional[list] = None,
        # --- E_hp_reward buffers [num_aa × 5] ---
        n_group_mean: Optional[list] = None,
        n_group_std: Optional[list] = None,
        # --- E_hp_penalty buffers [num_aa × 5] ---
        n_group_lo: Optional[list] = None,
        n_group_hi: Optional[list] = None,
        coord_lambda: float = 0.10,
        coord_m: float = 1.0,
        coord_alpha: float = 2.0,
        # --- E_rho_reward buffers ---
        # rho_group_fits: list of 5 dicts, each with fit_a, fit_b, fit_c
        rho_group_fits: Optional[list] = None,  # [5] dicts {fit_a, fit_b, fit_c}
        rho_group_sigma: Optional[list] = None,  # [5] floats
        rho_lambda: float = 0.1,
        # --- E_rho_penalty buffers ---
        rho_group_lo: Optional[list] = None,  # [5] global p5  (or length-derived)
        rho_group_hi: Optional[list] = None,  # [5] global p95
        rho_penalty_lambda: float = 0.1,
        rho_m: float = 1.0,
        rho_alpha: float = 2.0,
        # --- E_rg_penalty ---
        rg_lambda: float = 1.0,
        rg_r0: float = 2.0,
        rg_nu: float = 0.38,
        rg_dead_zone: float = 0.30,
        rg_m: float = 1.0,
        rg_alpha: float = 3.0,
        # --- Learnable buffer flags (all default False = backwards compat) ---
        learn_packing_coords: bool = False,
        learn_packing_density: bool = False,
        learn_penalty_shapes: bool = False,
        learn_packing_bounds: bool = False,
        learn_penalty_strengths: bool = False,
        learn_gate_geometry: bool = False,
        learn_hbond_geometry: bool = False,
        # --- Legacy / compat kwargs (accepted, silently ignored) ---
        seq_dim: int = 16,
        hidden1: int = 32,
        hidden2: int = 16,
        init_lambda: float = 1.0,
        r_on: float = 8.0,
        r_cut: float = 10.0,
        # --- Rama validity gate data ---
        secondary_data_dir: str = "analysis/secondary_analysis/data",
        **kwargs,
    ):
        super().__init__()

        self.num_aa = int(num_aa)
        self.topk = int(topk)
        self.exclude = int(exclude)
        self.max_dist = float(max_dist)
        self.normalize_by_length = bool(normalize_by_length)

        # --- Rg penalty buffers ---
        reg(self, "rg_lambda", torch.tensor(float(rg_lambda)), learnable=learn_penalty_strengths)
        reg(self, "rg_r0", torch.tensor(float(rg_r0)), learnable=learn_penalty_shapes)
        reg(self, "rg_nu", torch.tensor(float(rg_nu)), learnable=learn_penalty_shapes)
        reg(self, "rg_dead_zone", torch.tensor(float(rg_dead_zone)), learnable=learn_penalty_shapes)
        reg(self, "rg_m", torch.tensor(float(rg_m)), learnable=learn_penalty_shapes)
        reg(self, "rg_alpha", torch.tensor(float(rg_alpha)), learnable=learn_penalty_shapes)

        # --- Coordination penalty buffers [num_aa, 5] ---
        reg(self, "coord_lambda", torch.tensor(float(coord_lambda)), learnable=learn_penalty_strengths)
        reg(self, "coord_m", torch.tensor(float(coord_m)), learnable=learn_penalty_shapes)
        reg(self, "coord_alpha", torch.tensor(float(coord_alpha)), learnable=learn_penalty_shapes)

        def _make_group_buf(data, shape, default_val, name):
            if data is not None:
                t = torch.tensor(data, dtype=torch.float32)
                assert t.shape == torch.Size(shape), f"{name}: expected shape {shape}, got {list(t.shape)}"
            else:
                logger.warning("PackingEnergy: %s not provided, using %.2f", name, default_val)
                t = torch.full(shape, default_val, dtype=torch.float32)
            return t

        reg(
            self,
            "coord_n_group_lo",
            _make_group_buf(n_group_lo, (num_aa, NUM_GROUPS), 0.0, "n_group_lo"),
            learnable=learn_packing_bounds,
        )
        reg(
            self,
            "coord_n_group_hi",
            _make_group_buf(n_group_hi, (num_aa, NUM_GROUPS), 99.0, "n_group_hi"),
            learnable=learn_packing_bounds,
        )

        # --- E_rho_reward: 5 ρ*(k)(L) curves ---
        # Each curve: ρ*(k)(L) = a_k - b_k·exp(-L/c_k)
        if rho_group_fits is not None:
            fits = rho_group_fits
        else:
            logger.warning("PackingEnergy: rho_group_fits not provided, using scalar defaults")
            fits = [{"fit_a": 1.0, "fit_b": 0.4, "fit_c": 112.0}] * NUM_GROUPS

        rho_fit_a = torch.tensor([f["fit_a"] for f in fits], dtype=torch.float32)
        rho_fit_b = torch.tensor([f["fit_b"] for f in fits], dtype=torch.float32)
        rho_fit_c = torch.tensor([f["fit_c"] for f in fits], dtype=torch.float32)
        reg(self, "rho_fit_a", rho_fit_a, learnable=learn_packing_density)  # (5,)
        reg(self, "rho_fit_b", rho_fit_b, learnable=learn_packing_density)  # (5,)
        reg(self, "rho_fit_c", rho_fit_c, learnable=learn_packing_density)  # (5,)

        if rho_group_sigma is not None:
            rho_sig = torch.tensor(rho_group_sigma, dtype=torch.float32).clamp(min=1e-3)
        else:
            logger.warning("PackingEnergy: rho_group_sigma not provided, using 0.3")
            rho_sig = torch.full((NUM_GROUPS,), 0.3, dtype=torch.float32)
        reg(self, "rho_sigma_group", rho_sig, learnable=learn_packing_density)  # (5,)

        # λ_ρ — 1 trainable parameter
        self._lambda_rho_raw = nn.Parameter(torch.tensor(_inv_softplus(float(rho_lambda)), dtype=torch.float32))

        # --- E_rho_penalty: group lo/hi bands ---
        reg(self, "rho_penalty_lambda", torch.tensor(float(rho_penalty_lambda)), learnable=learn_penalty_strengths)
        reg(self, "rho_m", torch.tensor(float(rho_m)), learnable=learn_penalty_shapes)
        reg(self, "rho_alpha", torch.tensor(float(rho_alpha)), learnable=learn_penalty_shapes)

        if rho_group_lo is not None:
            rho_lo_t = torch.tensor(rho_group_lo, dtype=torch.float32)
        else:
            logger.warning("PackingEnergy: rho_group_lo not provided, using rho_fit_a - 1.35*sigma")
            rho_lo_t = rho_fit_a - 1.35 * rho_sig

        if rho_group_hi is not None:
            rho_hi_t = torch.tensor(rho_group_hi, dtype=torch.float32)
        else:
            rho_hi_t = rho_fit_a + 1.35 * rho_sig

        reg(self, "rho_group_lo_buf", rho_lo_t, learnable=learn_packing_bounds)  # (5,)
        reg(self, "rho_group_hi_buf", rho_hi_t, learnable=learn_packing_bounds)  # (5,)

        # --- E_hp_reward module ---
        self.burial = _Hydrophobic(
            num_aa=num_aa,
            init_lambda=1.0,
            n_group_mean=n_group_mean,
            n_group_std=n_group_std,
            group_assignment=group_assignment,
            learn_coords=learn_packing_coords,
        )

        # --- One-hot buffer (for potential extensions) ---
        self.register_buffer("_aa_eye", torch.eye(self.num_aa, dtype=torch.float32))

        # --- Ramachandran validity gate — peaks from basin surface files ---
        self.rama_gate = RamaValidityGate.from_data_dir(secondary_data_dir)

        # --- Parameter count summary ---
        n_hp = sum(p.numel() for p in self.burial.parameters())
        n_rho = 1
        logger.info(
            "PackingEnergy v6: %d trainable params (hp=%d, λ_ρ=%d)",
            n_hp + n_rho,
            n_hp,
            n_rho,
        )
        logger.info("  Groups: %s", " | ".join(f"{k}:{GROUP_NAMES[k]}" for k in range(NUM_GROUPS)))
        logger.info(
            "  topk=%d  max_dist=%.1f Å  Rg=%.1f·L^%.2f (dead_zone=%.2f)",
            topk,
            max_dist,
            rg_r0,
            rg_nu,
            rg_dead_zone,
        )
        # --- Learnable buffer summary ---
        _lb = []
        if learn_packing_coords:
            _lb.append("coords(n*,σ)")
        if learn_packing_density:
            _lb.append("density(ρ_fit,ρ_σ)")
        if learn_penalty_shapes:
            _lb.append("shapes(m,α,dz)")
        if learn_packing_bounds:
            _lb.append("bounds(lo,hi)")
        if learn_penalty_strengths:
            _lb.append("strengths(λ)")
        if learn_gate_geometry:
            _lb.append("gate(θ,φ)")
        if _lb:
            logger.info("  Learnable buffers: %s", ", ".join(_lb))
        else:
            logger.info("  Learnable buffers: none (all fixed)")

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def lambda_rho(self) -> torch.Tensor:
        return F.softplus(self._lambda_rho_raw) + 1e-6

    # -----------------------------------------------------------------------
    # ρ*(k)(L) curves — per-group saturating exponential
    # -----------------------------------------------------------------------

    def _rho_star_group(self, L_real: torch.Tensor) -> torch.Tensor:
        """ρ*(k)(L) = a_k - b_k·exp(-L/c_k) for each of 5 groups.

        Args:
            L_real: (B,) chain lengths (float)
        Returns:
            rho_star: (B, 5) expected group-k contact density per chain
        """
        # L_real: (B,) -> (B, 1) for broadcasting with (5,)
        L = L_real.unsqueeze(1)  # (B, 1)
        return self.rho_fit_a - self.rho_fit_b * torch.exp(-L / self.rho_fit_c)  # (B, 5)

    def _rho_bounds_group(self, L_real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-group ρ bounds: rho_star ± 1.35*sigma_group.

        Returns:
            lo: (B, 5), hi: (B, 5)
        """
        rho_star = self._rho_star_group(L_real)  # (B, 5)
        lo = rho_star - 1.35 * self.rho_sigma_group  # (B, 5)  broadcast (5,)
        hi = rho_star + 1.35 * self.rho_sigma_group  # (B, 5)
        return lo, hi

    # -----------------------------------------------------------------------
    # Ramachandran validity gate
    # -----------------------------------------------------------------------

    def _backbone_validity_gate(self, R: torch.Tensor, lengths=None) -> torch.Tensor:
        """Soft gate in [0,1] per residue based on backbone angle validity.

        Returns (B, L) — boundary residues (missing θ or φ) get gate=1.0.
        """
        B, L, _ = R.shape
        if L < 4:
            return torch.ones(B, L, device=R.device, dtype=R.dtype)

        theta = bond_angles(R)  # (B, L-2)
        phi = torsions(R)  # (B, L-3)
        n_pos = min(theta.shape[1], phi.shape[1])

        v_pos = self.rama_gate(theta[:, :n_pos], phi[:, :n_pos])  # (B, n_pos)

        # Pad to full chain length: boundary residues get 1.0 (ungated)
        gate = torch.ones(B, L, device=R.device, dtype=R.dtype)
        n = min(n_pos, L - 2)
        gate[:, 1 : 1 + n] = v_pos[:, :n]

        if lengths is not None:
            valid = torch.arange(L, device=R.device).unsqueeze(0) < lengths.unsqueeze(1)
            gate = gate * valid.float()

        return gate.detach()

    # -----------------------------------------------------------------------
    # Saturating exponential penalty helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _exp_penalty(violation: torch.Tensor, m: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """P(x) = m·(1 - exp(-α·max(0,x))).  Saturates at m, zero for x≤0."""
        return m * (1.0 - torch.exp(-alpha * violation.clamp(min=0.0)))

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        R: torch.Tensor,
        seq: torch.Tensor,
        return_components: bool = False,
        lengths: Optional[torch.Tensor] = None,
    ):
        if seq.min().item() < 0 or seq.max().item() >= self.num_aa:
            raise ValueError(f"seq must be in [0, {self.num_aa - 1}]")

        _check_lengths(R, lengths, "PackingEnergy")
        B, L, _ = R.shape

        # Padding mask
        R_safe = R
        valid_atom = None
        if lengths is not None:
            valid_atom = torch.arange(L, device=R.device).unsqueeze(0) < lengths.unsqueeze(1)
            R_safe = R.masked_fill(~valid_atom.unsqueeze(2), 1e6)

        # k=64 nearest nonbonded pairs — same as v5
        r, j_idx = topk_nonbonded_pairs(
            R_safe,
            k=self.topk,
            exclude=self.exclude,
            cutoff=self.max_dist,
        )

        # ==================================================================
        # E_hp_reward: per-residue product Gaussian over 5 groups
        # ==================================================================
        E_hp_i, n_grouped = self.burial(seq, r, j_idx, max_dist=self.max_dist)
        # E_hp_i:   (B, L)     Gaussian product value (positive)
        # n_grouped:(B, L, 5)  group-conditional coordination vectors

        if valid_atom is not None:
            E_hp_i = torch.where(valid_atom, E_hp_i, torch.zeros_like(E_hp_i))
            n_grouped = n_grouped * valid_atom.unsqueeze(-1).float()

        ram_gate = self._backbone_validity_gate(R, lengths=lengths)
        E_hp_i = E_hp_i * ram_gate

        lam_hp = self.burial.lambda_hp
        E_hp_reward = -lam_hp * E_hp_i.sum(dim=1)  # (B,)
        if self.normalize_by_length:
            L_real_f = lengths.float().clamp(min=1.0) if lengths is not None else float(L)
            E_hp_reward = E_hp_reward / L_real_f

        # Scalar n_i = Σ_k n_i^(k)  (for diagnostics/backward compat)
        n_i = n_grouped.sum(dim=-1)  # (B, L)

        # ==================================================================
        # E_rho_reward: per-chain product Gaussian over 5 group densities
        # ==================================================================
        L_real = lengths.float() if lengths is not None else torch.full((B,), float(L), device=R.device)

        # ρ^(k) = (1/L) Σ_i n_i^(k)
        rho_grouped = n_grouped.sum(dim=1) / L_real.unsqueeze(1).clamp(min=1.0)  # (B, 5)
        rho = rho_grouped.sum(dim=1)  # (B,)  scalar

        rho_star_g = self._rho_star_group(L_real)  # (B, 5)
        log_gauss_rho = -((rho_grouped - rho_star_g) ** 2) / (2.0 * self.rho_sigma_group**2)  # (B, 5)
        E_rho_reward = -self.lambda_rho * log_gauss_rho.sum(dim=-1).exp()  # (B,)

        # ==================================================================
        # E_packing = E_hp_reward + E_rho_reward  (landscape, in balance)
        # ==================================================================
        E_packing = E_hp_reward + E_rho_reward

        # ==================================================================
        # CONSTRAINTS (guardrails — NOT in energy balance)
        # ==================================================================
        E_hp_penalty = self._compute_coord_penalty(seq, n_grouped, valid_atom, lengths, B, L)
        E_rho_penalty = self._compute_rho_penalty(rho_grouped, L_real, B)
        E_rg_penalty = self._compute_rg_penalty(R_safe, lengths, B, L)

        E_constraint = E_hp_penalty + E_rho_penalty + E_rg_penalty

        # ==================================================================
        # Total
        # ==================================================================
        E = E_packing + E_constraint

        if return_components:
            rg_current = _batch_rg(R_safe, lengths)
            rg_expected = self.rg_r0 * L_real**self.rg_nu
            rho_star_g_detach = rho_star_g.detach()
            return E, {
                "E_hp_reward": E_hp_reward.detach(),
                "E_rho_reward": E_rho_reward.detach(),
                "E_hp_penalty": E_hp_penalty.detach(),
                "E_rho_penalty": E_rho_penalty.detach(),
                "E_rg_penalty": E_rg_penalty.detach(),
                "E_packing": E_packing.detach(),
                "E_constraint": E_constraint.detach(),
                # Scalar diagnostics
                "rho": rho.detach(),
                "n_i": n_i.detach(),
                # Group diagnostics (B, 5)
                "rho_grouped": rho_grouped.detach(),
                "rho_star_grouped": rho_star_g_detach,
                "n_grouped": n_grouped.detach(),
                # Rg diagnostics
                "rg": rg_current.detach(),
                "rg_ratio": (rg_current / rg_expected.clamp(min=1.0)).detach(),
                # Raw pair distances (B, L, K)
                "r": r.detach(),
            }
        return E

    # -----------------------------------------------------------------------
    # Constraint sub-terms
    # -----------------------------------------------------------------------

    def _compute_coord_penalty(
        self,
        seq: torch.Tensor,
        n_grouped: torch.Tensor,
        valid_atom,
        lengths,
        B: int,
        L: int,
    ) -> torch.Tensor:
        """E_hp_penalty: per-residue, per-group exponential penalty.

        Σ_i Σ_k  P(n_i^(k) outside [n^(k)_lo_{s_i}, n^(k)_hi_{s_i}])
        """
        if self.coord_lambda <= 0:
            return torch.zeros(B, device=seq.device)

        # Gather per-residue group bands: (B, L, 5)
        n_lo_i = self.coord_n_group_lo[seq]  # (B, L, 5)
        n_hi_i = self.coord_n_group_hi[seq]  # (B, L, 5)

        under = n_lo_i - n_grouped  # positive when n_i^(k) < n_lo
        over = n_grouped - n_hi_i  # positive when n_i^(k) > n_hi

        # Sum over groups, then over residues
        penalty_ik = self._exp_penalty(under, self.coord_m, self.coord_alpha) + self._exp_penalty(
            over, self.coord_m, self.coord_alpha
        )  # (B, L, 5)
        penalty_i = penalty_ik.sum(dim=-1)  # (B, L)   sum over groups

        if valid_atom is not None:
            penalty_i = torch.where(valid_atom, penalty_i, torch.zeros_like(penalty_i))

        E_coord = self.coord_lambda * penalty_i.sum(dim=1)  # (B,)
        if self.normalize_by_length:
            L_real_f = lengths.float().clamp(min=1.0) if lengths is not None else float(L)
            E_coord = E_coord / L_real_f
        return E_coord

    def _compute_rho_penalty(
        self,
        rho_grouped: torch.Tensor,  # (B, 5)
        L_real: torch.Tensor,  # (B,)
        B: int,
    ) -> torch.Tensor:
        """E_rho_penalty: per-group exponential penalty on chain-level ρ^(k).

        Σ_k  P(ρ^(k) outside [ρ^(k)_lo(L), ρ^(k)_hi(L)])
        """
        if self.rho_penalty_lambda <= 0:
            return torch.zeros(B, device=rho_grouped.device)

        rho_lo, rho_hi = self._rho_bounds_group(L_real)  # (B, 5) each

        under = rho_lo - rho_grouped  # (B, 5)
        over = rho_grouped - rho_hi  # (B, 5)

        penalty_k = self._exp_penalty(under, self.rho_m, self.rho_alpha) + self._exp_penalty(
            over, self.rho_m, self.rho_alpha
        )  # (B, 5)
        return self.rho_penalty_lambda * penalty_k.sum(dim=-1)  # (B,)

    def _compute_rg_penalty(
        self,
        R_safe: torch.Tensor,
        lengths,
        B: int,
        L: int,
    ) -> torch.Tensor:
        """E_rg_penalty: saturating exponential with dead zone on Rg/Rg*(L)."""
        if self.rg_lambda <= 0:
            return torch.zeros(B, device=R_safe.device)

        rg_current = _batch_rg(R_safe, lengths)
        L_real = lengths.float() if lengths is not None else torch.full((B,), float(L), device=R_safe.device)
        rg_expected = self.rg_r0 * L_real**self.rg_nu
        rg_ratio = rg_current / rg_expected.clamp(min=1.0)
        deviation = (rg_ratio - 1.0).abs()
        violation = (deviation - self.rg_dead_zone).clamp(min=0.0)
        return self.rg_lambda * self._exp_penalty(violation, self.rg_m, self.rg_alpha)

    # -----------------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------------

    def packing_energy(self, R, seq, lengths=None):
        """Return E_packing only (for energy balance monitoring)."""
        _, comps = self.forward(R, seq, return_components=True, lengths=lengths)
        return comps["E_packing"]

    def constraint_energy(self, R, seq, lengths=None):
        """Return E_constraint only (guardrails, not in balance)."""
        _, comps = self.forward(R, seq, return_components=True, lengths=lengths)
        return comps["E_constraint"]

    def subterm_energies(self, R, seq, lengths=None):
        """Return all 5 sub-terms as separate (B,) tensors.

        Order matches v5 convention consumed by diagnostics.py:
            (E_hp_reward, E_hp_penalty, E_rho_reward, E_rho_penalty, E_rg_penalty)
        """
        _, comps = self.forward(R, seq, return_components=True, lengths=lengths)
        return (
            comps["E_hp_reward"],
            comps["E_hp_penalty"],
            comps["E_rho_reward"],
            comps["E_rho_penalty"],
            comps["E_rg_penalty"],
        )

    def group_diagnostics(self, R, seq, lengths=None) -> dict:
        """Return per-group coordination and density diagnostics.

        Returns dict with:
            n_grouped:       (B, L, 5) per-residue group coordination
            rho_grouped:     (B, 5)    per-chain group density
            rho_star_grouped:(B, 5)    expected per-chain group density
        """
        _, comps = self.forward(R, seq, return_components=True, lengths=lengths)
        return {
            "n_grouped": comps["n_grouped"],
            "rho_grouped": comps["rho_grouped"],
            "rho_star_grouped": comps["rho_star_grouped"],
        }


class SimplePackingEnergy(PackingEnergy):
    """Backward-compatible alias."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


__all__ = ["PackingEnergy", "SimplePackingEnergy"]
