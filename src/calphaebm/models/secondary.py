"""Secondary structure energy term.

src/calphaebm/models/secondary.py

Three subterms:
  1) E_ram      : mixture-of-basins Ramachandran potential
  2) E_hb_alpha : helical H-bond energy (local, i→i+4)
  3) E_hb_beta  : sheet H-bond energy (nonlocal)

H-bond gating uses responsibilities × validity:
  responsibilities (softmax) → basin selectivity (helix vs sheet)
  validity gate (RamaValidityGate) → forbidden region rejection
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calphaebm.geometry.internal import bond_angles, torsions
from calphaebm.geometry.pairs import topk_nonbonded_pairs
from calphaebm.models.embeddings import AAEmbedding
from calphaebm.models.hbond import HELIX_BASIN_IDX, SHEET_BASIN_IDX, HBondHelix, HBondSheet
from calphaebm.models.rama_gate import RamaValidityGate
from calphaebm.utils.logging import get_logger

logger = get_logger()


def _cat(*xs: torch.Tensor) -> torch.Tensor:
    return torch.cat(xs, dim=-1)


def _inv_softplus(y: float, eps: float = 1e-8) -> float:
    y = float(max(y, eps))
    return float(np.log(np.expm1(y)))


def _resolve_edge_paths(data_dir: Path) -> tuple[Path, Path]:
    canonical_theta = data_dir / "theta_edges_deg.npy"
    canonical_phi = data_dir / "phi_edges_deg.npy"
    legacy_theta = data_dir / "figure_3a_xedges.npy"
    legacy_phi = data_dir / "figure_3a_yedges.npy"
    theta_edges = canonical_theta if canonical_theta.exists() else legacy_theta
    phi_edges = canonical_phi if canonical_phi.exists() else legacy_phi
    return theta_edges, phi_edges


class BasinPotential(nn.Module):
    """Tabulated basin potential with periodic phi and bilinear interpolation."""

    _FULL_CIRCLE_THRESHOLD_DEG = 300.0
    _PHI_PERIOD_DEG = 360.0

    def __init__(
        self,
        hist_path: Path,
        theta_edges_path: Path,
        phi_edges_path: Path,
        smooth_sigma: float = 2.0,
        pseudocount: float = 1e-6,
        smooth: bool = True,
    ):
        super().__init__()
        hist = np.load(hist_path).astype(np.float32)
        theta_edges = np.load(theta_edges_path).astype(np.float32).reshape(-1)
        phi_edges = np.load(phi_edges_path).astype(np.float32).reshape(-1)
        if hist.ndim != 2:
            raise ValueError(f"Expected 2D grid, got shape {hist.shape}")
        is_prob_like = float(np.nanmax(hist)) <= 1.5 and float(np.nanmin(hist)) >= 0.0
        if is_prob_like:
            if smooth:
                try:
                    from scipy.ndimage import gaussian_filter

                    hist = gaussian_filter(hist, sigma=(smooth_sigma, smooth_sigma), mode=("nearest", "wrap")).astype(
                        np.float32
                    )
                except ImportError:
                    pass
            energy = -np.log(hist + float(pseudocount))
            energy = energy - float(np.nanmin(energy))
        else:
            energy = hist
            if smooth:
                try:
                    from scipy.ndimage import gaussian_filter

                    energy = gaussian_filter(
                        energy, sigma=(smooth_sigma, smooth_sigma), mode=("nearest", "wrap")
                    ).astype(np.float32)
                except ImportError:
                    pass
        energy = np.nan_to_num(energy, nan=0.0, posinf=50.0, neginf=0.0).astype(np.float32)
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        n_theta = len(theta_centers)
        n_phi = len(phi_centers)
        if energy.shape != (n_theta, n_phi):
            if energy.shape == (n_phi, n_theta):
                energy = energy.T
            else:
                raise ValueError(f"Energy grid shape {energy.shape} incompatible with ({n_theta},{n_phi})")
        phi_span = float(phi_centers[-1] - phi_centers[0])
        self.phi_periodic = phi_span >= self._FULL_CIRCLE_THRESHOLD_DEG
        self.phi_period = self._PHI_PERIOD_DEG if self.phi_periodic else None
        self.register_buffer("energy_grid", torch.tensor(energy, dtype=torch.float32), persistent=False)
        self.register_buffer("theta_centers", torch.tensor(theta_centers, dtype=torch.float32), persistent=False)
        self.register_buffer("phi_centers", torch.tensor(phi_centers, dtype=torch.float32), persistent=False)
        self.n_theta = n_theta
        self.n_phi = n_phi

    @classmethod
    def from_energy_edges(cls, energy, theta_edges, phi_edges, phi_period=360.0):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        energy_np = energy.float().numpy()
        theta_centers = (0.5 * (theta_edges[:-1] + theta_edges[1:])).float().numpy()
        phi_centers = (0.5 * (phi_edges[:-1] + phi_edges[1:])).float().numpy()
        obj.register_buffer("energy_grid", torch.tensor(energy_np, dtype=torch.float32), persistent=False)
        obj.register_buffer("theta_centers", torch.tensor(theta_centers, dtype=torch.float32), persistent=False)
        obj.register_buffer("phi_centers", torch.tensor(phi_centers, dtype=torch.float32), persistent=False)
        obj.n_theta = len(theta_centers)
        obj.n_phi = len(phi_centers)
        obj.phi_periodic = True
        obj.phi_period = float(phi_period)
        return obj

    def forward(self, theta_deg: torch.Tensor, phi_deg: torch.Tensor) -> torch.Tensor:
        orig_shape = theta_deg.shape
        theta_flat = theta_deg.reshape(-1)
        phi_flat = phi_deg.reshape(-1)
        tc, pc, E = self.theta_centers, self.phi_centers, self.energy_grid
        theta_clamped = theta_flat.clamp(float(tc[0]), float(tc[-1]))
        if self.phi_periodic and self.phi_period is not None:
            phi_min = float(pc[0])
            phi_wrapped = torch.remainder(phi_flat - phi_min, self.phi_period) + phi_min
            phi_clamped = phi_wrapped.clamp(float(pc[0]), float(pc[-1]))
        else:
            phi_clamped = phi_flat.clamp(float(pc[0]), float(pc[-1]))
        dt = float(tc[1] - tc[0]) if self.n_theta > 1 else 1.0
        ut = (theta_clamped - float(tc[0])) / dt
        it0 = torch.floor(ut).long().clamp(0, self.n_theta - 2)
        tt = (ut - it0.float()).clamp(0.0, 1.0)
        it1 = (it0 + 1).clamp(0, self.n_theta - 1)
        dp = float(pc[1] - pc[0]) if self.n_phi > 1 else 1.0
        up = (phi_clamped - float(pc[0])) / dp
        ip0 = torch.floor(up).long().clamp(0, self.n_phi - 2)
        tp = (up - ip0.float()).clamp(0.0, 1.0)
        ip1 = (ip0 + 1).clamp(0, self.n_phi - 1)
        val = (
            (1 - tt) * (1 - tp) * E[it0, ip0]
            + (1 - tt) * tp * E[it0, ip1]
            + tt * (1 - tp) * E[it1, ip0]
            + tt * tp * E[it1, ip1]
        )
        return val.reshape(orig_shape)


class SecondaryStructureEnergy(nn.Module):
    """Secondary structure energy: E_ram + E_hb(alpha) + E_hb(beta)."""

    def __init__(
        self,
        num_aa: int = 20,
        emb_dim: int = 16,
        num_basins: int = 4,
        data_dir: str = "analysis/secondary_analysis/data",
        normalize_by_length: bool = True,
        debug_mode: bool = False,
        hb_topk: int = 32,
        physics_prior: bool = False,
        learn_hbond_geometry: bool = False,
        **legacy_kwargs,
    ):
        super().__init__()
        for k in ["hidden_dims", "use_cos_theta"]:
            legacy_kwargs.pop(k, None)
        if legacy_kwargs:
            logger.warning("SecondaryStructureEnergy: unexpected kwargs ignored: %s", list(legacy_kwargs.keys()))

        self.physics_prior = bool(physics_prior)
        if getattr(self, "physics_prior", False):
            logger.info(
                "SecondaryStructureEnergy: physics_prior=True — " "logits forced to 0, responsibilities = softmax(-U_k)"
            )

        self.num_basins = int(num_basins)
        self.normalize_by_length = bool(normalize_by_length)
        self.debug_mode = bool(debug_mode)
        self.hb_topk = int(hb_topk)

        self.emb = AAEmbedding(num_aa=num_aa, dim=emb_dim)
        ctx_dim = 4 * emb_dim

        # ── Basin potentials ──────────────────────────────────────────────
        data_path = Path(data_dir)
        theta_edges_path, phi_edges_path = _resolve_edge_paths(data_path)
        basin_paths = sorted(data_path.glob("basin_*_energy.npy"))
        if len(basin_paths) < num_basins:
            raise FileNotFoundError(
                f"Need {num_basins} basin files in {data_path}, found {len(basin_paths)}.\n"
                f"Run: calphaebm analyze basins --force-reextract"
            )

        self.basin_potentials = nn.ModuleList(
            [
                BasinPotential(
                    hist_path=basin_paths[k], theta_edges_path=theta_edges_path, phi_edges_path=phi_edges_path
                )
                for k in range(num_basins)
            ]
        )

        # ── Mixture weights ───────────────────────────────────────────────
        self.A = nn.Parameter(torch.randn(num_basins, ctx_dim) * 0.01)
        self.a = nn.Parameter(torch.zeros(num_basins))
        self.A.register_hook(lambda g: torch.nan_to_num(g.clamp(-1.0, 1.0)))

        # ── H-bond terms ─────────────────────────────────────────────────
        self.hb_helix = HBondHelix(learn_geometry=learn_hbond_geometry)
        self.hb_sheet = HBondSheet(learn_geometry=learn_hbond_geometry)

        # ── E_ram weight ──────────────────────────────────────────────────
        self.lambda_ram = nn.Parameter(torch.tensor(_inv_softplus(1.0), dtype=torch.float32))

        # ── Rama validity gate — peaks from basin surfaces (no defaults) ─
        self.rama_gate = RamaValidityGate.from_basin_surfaces(self.basin_potentials)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._init_summary = {
            "total_params": total_params,
            "num_basins": num_basins,
            "hb_helix_mu": self.hb_helix.mu.item(),
            "hb_helix_sigma": self.hb_helix.sigma.item(),
            "hb_sheet_mu1": self.hb_sheet.mu1.item(),
            "hb_sheet_mu2": self.hb_sheet.mu2.item(),
        }

    @property
    def ram_weight(self) -> torch.Tensor:
        return F.softplus(self.lambda_ram)

    def _mixture_ramachandran(self, theta_deg, phi_deg, context):
        theta_deg = torch.nan_to_num(theta_deg, nan=0.0, posinf=0.0, neginf=0.0)
        phi_deg = torch.nan_to_num(phi_deg, nan=0.0, posinf=0.0, neginf=0.0)
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)

        if getattr(self, "physics_prior", False):
            # Pure tabulated potential: no sequence logits.
            # E_ram = -logsumexp(-U_k) — tabulated multi-well Boltzmann inversion.
            # responsibilities = softmax(-U_k) — geometry-only, no sequence bias.
            logits = torch.zeros(
                context.shape[0],
                context.shape[1],
                self.num_basins,
                device=context.device,
                dtype=context.dtype,
            )
        else:
            logits = torch.einsum("kf,bnf->bnk", self.A, context) + self.a
            logits = torch.clamp(logits, -10.0, 10.0)

        with torch.no_grad():
            U = torch.stack([U_k(theta_deg, phi_deg) for U_k in self.basin_potentials], dim=-1)
        U = U.detach()
        U = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
        U = torch.clamp(U, max=50.0)
        s = logits - U
        log_Z = torch.logsumexp(s, dim=-1)
        energy = torch.clamp(-log_Z, min=-50.0, max=50.0)
        responsibilities = torch.softmax(s, dim=-1)
        return energy, responsibilities

    def _compute_energy_components(self, theta, phi, seq, R, debug=False):
        _, L = seq.shape
        if L < 5:
            raise ValueError(f"Need L>=5, got L={L}")
        theta = torch.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
        phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

        e = self.emb(seq)
        theta_deg = torch.nan_to_num(torch.rad2deg(theta), nan=0.0, posinf=0.0, neginf=0.0)
        phi_deg = torch.nan_to_num(torch.rad2deg(phi), nan=0.0, posinf=0.0, neginf=0.0)
        theta_for_phi = theta_deg[:, : L - 3]
        phi_aligned = phi_deg
        ctx = _cat(e[:, : L - 3], e[:, 1 : L - 2], e[:, 2 : L - 1], e[:, 3:L])
        ctx = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)

        # 1) E_ram
        E_ram_per_pos, responsibilities = self._mixture_ramachandran(theta_for_phi, phi_aligned, ctx)
        if debug or (self.debug_mode and self.training and torch.rand(1).item() < 0.001):
            avg_resp = responsibilities.mean(dim=(0, 1)).detach().cpu().numpy()
            logger.debug("Avg basin responsibilities: %s", avg_resp)
        if self.normalize_by_length:
            E_ram = self.ram_weight * (E_ram_per_pos.sum(dim=1) / float(max(E_ram_per_pos.shape[1], 1)))
        else:
            E_ram = self.ram_weight * E_ram_per_pos.sum(dim=1)

        # 2) Basin probabilities × validity gate
        p_helix = responsibilities[:, :, HELIX_BASIN_IDX].detach()
        p_ext = responsibilities[:, :, SHEET_BASIN_IDX].detach()

        validity = self.rama_gate(theta[:, : L - 3], phi)  # (B, N)
        p_helix = p_helix * validity
        p_ext = p_ext * validity

        # 3) E_hb_alpha
        E_hb_alpha = self.hb_helix(p_helix, R, normalize_by_length=self.normalize_by_length)

        # 4) E_hb_beta
        r, j_idx = topk_nonbonded_pairs(R, k=self.hb_topk, exclude=3, cutoff=12.0)
        E_hb_beta = self.hb_sheet(p_ext, R, r, j_idx, normalize_by_length=self.normalize_by_length)

        return E_ram, E_hb_alpha, E_hb_beta

    def forward(self, R, seq, lengths=None):
        _, L, _ = R.shape
        if L < 5:
            raise ValueError(f"Need L>=5, got L={L}")
        theta = bond_angles(R)
        phi = torsions(R)
        E_ram, E_hb_a, E_hb_b = self._compute_energy_components(theta, phi, seq, R)
        return E_ram + E_hb_a + E_hb_b

    def subterm_energies(self, R, seq, lengths=None):
        theta = bond_angles(R)
        phi = torsions(R)
        return self._compute_energy_components(theta, phi, seq, R)

    def energy_from_thetaphi(self, theta, phi, seq, R=None, debug=False, lengths=None):
        if R is not None:
            return sum(self._compute_energy_components(theta, phi, seq, R, debug=debug))
        else:
            B, L = seq.shape
            theta = torch.nan_to_num(theta, nan=0.0)
            phi = torch.nan_to_num(phi, nan=0.0)
            e = self.emb(seq)
            theta_deg = torch.nan_to_num(torch.rad2deg(theta), nan=0.0)
            phi_deg = torch.nan_to_num(torch.rad2deg(phi), nan=0.0)
            ctx = _cat(e[:, : L - 3], e[:, 1 : L - 2], e[:, 2 : L - 1], e[:, 3:L])
            ctx = torch.nan_to_num(ctx, nan=0.0)
            E_ram_per_pos, _ = self._mixture_ramachandran(theta_deg[:, : L - 3], phi_deg, ctx)
            if self.normalize_by_length:
                return self.ram_weight * (E_ram_per_pos.sum(dim=1) / float(max(E_ram_per_pos.shape[1], 1)))
            return self.ram_weight * E_ram_per_pos.sum(dim=1)


SecondaryStructureEnergy.__module__ = __name__
