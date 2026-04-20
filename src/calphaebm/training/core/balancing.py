# src/calphaebm/training/core/balancing.py

"""Force-scale balancing to recommend lambda weights.

Key design decisions
--------------------
1. Forces are computed with gates applied (g * E_term), not raw E_term.
   This means the report reflects the actual force each term exerts during
   simulation, not just its raw output magnitude.

2. Forces are measured at SIMULATION-RELEVANT geometry, not perfect native
   geometry. At native geometry, repulsion is essentially zero (no clashes),
   which gives a misleading lambda recommendation of ~1000x for repulsion.
   Instead we perturb the structure in two ways:

   a) Thermal noise (sigma_thermal): small Gaussian displacement representing
      normal thermal fluctuations during Langevin dynamics. This activates
      local and secondary forces but not repulsion much.

   b) Clash perturbations (clash_frac, clash_sigma): a random fraction of
      residue pairs are pushed toward each other, creating mild clashes.
      This activates the repulsion term in a realistic way.

   The result is a force measurement that reflects what each term actually
   does during a typical Langevin step, not at the idealized native structure.

3. Multiple perturbation samples are averaged to reduce variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from calphaebm.models.energy import TotalEnergy
from calphaebm.utils.math import safe_norm


@dataclass
class BalanceReport:
    """Report from lambda balancing."""

    force_scales: Dict[str, float]  # Median gated force norm per term
    recommended_lambdas: Dict[str, float]  # Recommended gate values
    geometry: str = "native"  # Description of geometry used
    n_samples: int = 1  # Number of perturbation samples averaged


def _median_force_scale(F: torch.Tensor) -> float:
    """Compute median force norm across all atoms in batch."""
    norms = safe_norm(F, dim=-1).reshape(-1)
    return float(torch.median(norms).item())


def _perturb_with_clashes(
    R: torch.Tensor,
    sigma_thermal: float = 0.3,
    clash_frac: float = 0.05,
    clash_sigma: float = 0.5,
    clash_min_seq_sep: int = 4,
) -> torch.Tensor:
    """Perturb coordinates to create simulation-relevant geometry.

    Combines thermal noise (always active) with clash perturbations
    (pushes a random fraction of residue pairs closer together).

    Args:
        R: (B, L, 3) native coordinates.
        sigma_thermal: Std of Gaussian noise in Angstroms. 0.3 Å is typical
                       for Cα fluctuations at ~300K.
        clash_frac: Fraction of residue pairs to perturb toward each other.
                    0.05 = 5% of pairs, giving realistic occasional clashes.
        clash_sigma: How much to push clashing pairs toward each other (Å).
                     0.5 Å creates mild clashes that activate repulsion wall.
        clash_min_seq_sep: Minimum sequence separation for clash perturbation.
                           Must be > exclude_nonbonded (3) to be visible to
                           repulsion term.

    Returns:
        (B, L, 3) perturbed coordinates.
    """
    B, L, _ = R.shape

    # Step 1: thermal noise on all atoms
    R_perturbed = R + sigma_thermal * torch.randn_like(R)

    # Step 2: clash perturbations on random non-bonded pairs
    if clash_frac > 0 and L > clash_min_seq_sep + 1:
        for b in range(B):
            # Sample random pairs with minimum sequence separation
            n_pairs = max(1, int(clash_frac * L))
            for _ in range(n_pairs):
                i = torch.randint(0, L - clash_min_seq_sep, (1,)).item()
                j = torch.randint(int(i) + clash_min_seq_sep, L, (1,)).item()

                # Push j toward i by clash_sigma Å
                direction = R_perturbed[b, i] - R_perturbed[b, j]
                dist = direction.norm().clamp(min=0.1)
                unit = direction / dist
                R_perturbed[b, j] = R_perturbed[b, j] + clash_sigma * unit

    return R_perturbed


def term_forces_gated(
    model: TotalEnergy,
    R: torch.Tensor,
    seq: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute GATED forces from each energy term separately.

    Uses the actual gate values (g_local, g_rep, etc.) so forces reflect
    what the model actually exerts during simulation.

    Args:
        model: TotalEnergy model.
        R: (B, L, 3) coordinates (should be simulation-relevant geometry).
        seq: (B, L) amino acid indices.

    Returns:
        Dictionary mapping term names to force tensors (B, L, 3).
    """
    forces: Dict[str, torch.Tensor] = {}

    # Local term — gated
    Rg = R.detach().requires_grad_(True)
    E_local = (model.gate_local * model.local(Rg, seq)).sum()
    F_local = -torch.autograd.grad(E_local, Rg, create_graph=False)[0]
    forces["local"] = F_local.detach()

    # Repulsion — gated
    if model.repulsion is not None:
        Rg = R.detach().requires_grad_(True)
        E_rep = (model.gate_repulsion * model.repulsion(Rg, seq)).sum()
        F_rep = -torch.autograd.grad(E_rep, Rg, create_graph=False)[0]
        forces["repulsion"] = F_rep.detach()

    # Secondary — gated
    if model.secondary is not None:
        Rg = R.detach().requires_grad_(True)
        E_ss = (model.gate_secondary * model.secondary(Rg, seq)).sum()
        F_ss = -torch.autograd.grad(E_ss, Rg, create_graph=False)[0]
        forces["secondary"] = F_ss.detach()

    # Packing — gated
    if model.packing is not None:
        Rg = R.detach().requires_grad_(True)
        E_pack = (model.gate_packing * model.packing(Rg, seq)).sum()
        F_pack = -torch.autograd.grad(E_pack, Rg, create_graph=False)[0]
        forces["packing"] = F_pack.detach()

    return forces


def term_forces(
    model: TotalEnergy,
    R: torch.Tensor,
    seq: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute UNGATED forces (legacy interface, kept for compatibility).

    Prefer term_forces_gated() for simulation-relevant balancing.
    """
    forces: Dict[str, torch.Tensor] = {}

    Rg = R.detach().requires_grad_(True)
    E_local = model.local(Rg, seq).sum()
    forces["local"] = -torch.autograd.grad(E_local, Rg, create_graph=False)[0].detach()

    if model.repulsion is not None:
        Rg = R.detach().requires_grad_(True)
        E_rep = model.repulsion(Rg, seq).sum()
        forces["repulsion"] = -torch.autograd.grad(E_rep, Rg, create_graph=False)[0].detach()

    if model.secondary is not None:
        Rg = R.detach().requires_grad_(True)
        E_ss = model.secondary(Rg, seq).sum()
        forces["secondary"] = -torch.autograd.grad(E_ss, Rg, create_graph=False)[0].detach()

    if model.packing is not None:
        Rg = R.detach().requires_grad_(True)
        E_pack = model.packing(Rg, seq).sum()
        forces["packing"] = -torch.autograd.grad(E_pack, Rg, create_graph=False)[0].detach()

    return forces


def recommend_lambdas(
    model: TotalEnergy,
    R: torch.Tensor,
    seq: torch.Tensor,
    reference: str = "local",
    current_lambdas: Optional[Dict[str, float]] = None,
    clip: Tuple[float, float] = (1e-3, 1e3),
    # Perturbation parameters
    n_samples: int = 5,
    sigma_thermal: float = 0.3,
    clash_frac: float = 0.05,
    clash_sigma: float = 0.5,
    use_native: bool = False,
) -> BalanceReport:
    """Recommend lambda values to balance force scales at simulation geometry.

    Measures gated forces at perturbed (simulation-relevant) geometry by
    default, averaged over multiple perturbation samples to reduce variance.

    Args:
        model: TotalEnergy model.
        R: (B, L, 3) native coordinates.
        seq: (B, L) amino acid indices.
        reference: Term to use as reference scale (default: local).
        current_lambdas: Current lambda values (unused, kept for compatibility).
        clip: Min/max allowed lambda values.
        n_samples: Number of perturbation samples to average over.
        sigma_thermal: Thermal noise std in Angstroms (default: 0.3 Å).
        clash_frac: Fraction of pairs to push toward each other (default: 0.05).
        clash_sigma: Clash perturbation magnitude in Angstroms (default: 0.5 Å).
        use_native: If True, measure at native geometry only (legacy behavior).

    Returns:
        BalanceReport with gated force scales and recommended lambdas.
    """
    model.eval()

    all_scales: Dict[str, List[float]] = {}

    with torch.no_grad():
        pass  # context for clarity — grad computed inside term_forces_gated

    for sample_idx in range(n_samples):
        if use_native:
            R_eval = R
            geom_desc = "native"
        else:
            R_eval = _perturb_with_clashes(
                R,
                sigma_thermal=sigma_thermal,
                clash_frac=clash_frac,
                clash_sigma=clash_sigma,
            )
            geom_desc = f"perturbed(thermal={sigma_thermal}Å, clash_frac={clash_frac}, clash_sigma={clash_sigma}Å)"

        forces = term_forces_gated(model, R_eval, seq)

        for k, F in forces.items():
            scale = _median_force_scale(F)
            all_scales.setdefault(k, []).append(scale)

    # Average across samples
    avg_scales = {k: float(sum(v) / len(v)) for k, v in all_scales.items()}

    if reference not in avg_scales or avg_scales[reference] <= 0:
        raise ValueError(f"Reference term '{reference}' missing or zero. Available: {list(avg_scales)}")

    ref_scale = avg_scales[reference]
    lo, hi = clip

    rec: Dict[str, float] = {}
    for k, s in avg_scales.items():
        if s <= 0:
            rec[k] = 1.0
        else:
            rec[k] = float(max(lo, min(hi, ref_scale / s)))

    return BalanceReport(
        force_scales=avg_scales,
        recommended_lambdas=rec,
        geometry=geom_desc,
        n_samples=n_samples,
    )


def balance_across_batch(
    model: TotalEnergy,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_batches: int = 10,
    reference: str = "local",
    sigma_thermal: float = 0.3,
    clash_frac: float = 0.05,
    clash_sigma: float = 0.5,
    use_native: bool = False,
) -> BalanceReport:
    """Average gated force scales across multiple batches.

    Args:
        model: TotalEnergy model.
        dataloader: DataLoader providing (R, seq, _, _) tuples.
        device: Torch device.
        n_batches: Number of batches to average over.
        reference: Reference term.
        sigma_thermal: Thermal noise std (Å).
        clash_frac: Fraction of pairs to clash-perturb.
        clash_sigma: Clash perturbation magnitude (Å).
        use_native: If True, use native geometry only (legacy).

    Returns:
        BalanceReport with averaged gated force scales.
    """
    model.eval()

    all_scales: Dict[str, List[float]] = {}

    for i, (R, seq, _, _) in enumerate(dataloader):
        if i >= n_batches:
            break

        R = R.to(device)
        seq = seq.to(device)

        if use_native:
            R_eval = R
        else:
            R_eval = _perturb_with_clashes(
                R,
                sigma_thermal=sigma_thermal,
                clash_frac=clash_frac,
                clash_sigma=clash_sigma,
            )

        forces = term_forces_gated(model, R_eval, seq)

        for k, F in forces.items():
            scale = _median_force_scale(F)
            all_scales.setdefault(k, []).append(scale)

    avg_scales = {k: float(sum(v) / len(v)) for k, v in all_scales.items()}

    ref_scale = avg_scales.get(reference, 0.0)
    if ref_scale <= 0:
        raise ValueError(f"Reference term '{reference}' has zero scale")

    rec = {k: float(ref_scale / s) if s > 0 else 1.0 for k, s in avg_scales.items()}

    geom_desc = "native" if use_native else f"perturbed(thermal={sigma_thermal}Å, clash_frac={clash_frac})"

    return BalanceReport(
        force_scales=avg_scales,
        recommended_lambdas=rec,
        geometry=geom_desc,
        n_batches=n_batches,
    )
