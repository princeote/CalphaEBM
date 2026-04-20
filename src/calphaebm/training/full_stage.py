# src/calphaebm/training/full_stage.py
"""Stage 1: Full PDB-only training with all losses active.

Replaces the multi-phase approach (phases 1-5) with a single training stage
that runs in rounds. Eval is fully decoupled — run eval_watcher.py
independently to evaluate each round checkpoint as it is saved.

PDB batch losses (native structures only):
  - Depth:  exp(clamp(E_nat - target, max=5))
  - Balance: subterm magnitude balance (ramped 1e-6 -> lambda_balance in round 1)

Decoy-based losses (pre-generated IC-noised decoys with correct 3.8A bonds):
  - DSM:     Denoising Score Matching — target = (R_native - R_decoy) / sigma^2
  - Discrimination: per-subterm, E(decoy) > E(native)
  - Q-funnel:    monotonic dE/dQ slope (lower Q -> higher E)
  - dRMSD-funnel: full pairwise dRMSD ordering (higher dRMSD -> higher E)
  - Gap:         exp(clamp(-(E_pert - E_nat - margin), max=5))

Round structure:
  Phase A: Pre-generate decoys for 1024 proteins (parallel, ~30s on 64 CPUs)
  Phase B: Train N steps sampling batches of 32 from stored proteins
  Phase C: Save round checkpoint — eval_watcher.py picks this up for basin eval
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from calphaebm.defaults import TRAIN as _T
from calphaebm.training.core.state import TrainingState
from calphaebm.training.logging.diagnostics import DiagnosticLogger
from calphaebm.training.logging.validation_logging import ValidationLogger
from calphaebm.training.losses.balance_loss import energy_balance_loss
from calphaebm.utils.logging import get_logger

logger = get_logger()

BOND_LENGTH = 3.8
THETA_PHI_RATIO = 0.161  # theta noise = 0.161 * phi noise (narrower distribution)


# =====================================================================
#  Round metrics
# =====================================================================


@dataclass
class RoundMetrics:
    round_num: int = 0
    train_loss: float = 0.0
    eval_rmsd: float = 0.0
    eval_q: float = 0.0
    eval_rg_pct: float = 0.0
    eval_rmsf: float = 0.0
    eval_e_delta: float = 0.0
    eval_rama_corr: float = 0.0
    eval_dphi_corr: float = 0.0
    eval_k64drmsd: float = 0.0
    eval_contact_order: float = 0.0
    composite: float = 0.0
    q_af: float = 100.0
    rg_af: float = 100.0
    n_ok: int = 0
    n_total: int = 0
    ema: Dict[str, float] = field(default_factory=dict)


# =====================================================================
#  IC extraction and NeRF reconstruction — delegates to geometry modules
# =====================================================================

from calphaebm.geometry.internal import bond_angles, torsions
from calphaebm.geometry.reconstruct import nerf_reconstruct as _nerf_reconstruct_batched


def _extract_ic(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract bond angles and dihedrals from Ca coordinates.
    Args:
        R: (L, 3) single-chain Ca coordinates.
    Returns:
        theta: (L-2,) bond angles (radians).
        phi:   (L-3,) dihedral angles (radians, in [-pi, pi]).

    Convention: Oldfield & Hubbard (Proteins 1994).
    Delegates to calphaebm.geometry.internal.
    """
    R_b = R.unsqueeze(0)  # (1, L, 3)
    theta = bond_angles(R_b).squeeze(0)  # (L-2,)
    phi = torsions(R_b).squeeze(0)  # (L-3,)
    return theta, phi


def _nerf_reconstruct(R_init, theta, phi, bond_length=BOND_LENGTH):
    """Reconstruct Ca chain from ICs via NeRF. Bonds always 3.8 A.
    Delegates to calphaebm.geometry.reconstruct.

    Args:
        R_init: (3, 3) anchor atoms OR (N, 3, 3) batched anchors.
        theta:  (L-2,) or (N, L-2) bond angles.
        phi:    (L-3,) or (N, L-3) dihedral angles.

    Returns:
        (L, 3) or (N, L, 3) reconstructed coordinates.
    """
    batched = theta.dim() == 2
    if not batched:
        anchor = R_init.unsqueeze(0)  # (1, 3, 3)
        theta_b = theta.unsqueeze(0)  # (1, L-2)
        phi_b = phi.unsqueeze(0)  # (1, L-3)
        R = _nerf_reconstruct_batched(theta_b, phi_b, anchor, bond=bond_length)
        return R.squeeze(0)  # (L, 3)
    else:
        anchor = R_init  # already (N, 3, 3)
        return _nerf_reconstruct_batched(theta, phi, anchor, bond=bond_length)


def _wrap_to_pi(x):
    """Wrap angles to [-pi, pi]."""
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


# =====================================================================
#  Decoy generation: parallel across CPUs
# =====================================================================


def _generate_one_decoy(args):
    """Worker: generate ONE IC-noised decoy on one CPU core.

    Called by ProcessPoolExecutor — each of B*n_decoys tasks
    runs independently on its own CPU.
    """
    torch.set_num_threads(1)
    anchor, theta, phi, sigma, ni, nj, d0, rg_nat, Lb = args

    theta_n = theta + THETA_PHI_RATIO * sigma * torch.randn_like(theta)
    phi_n = _wrap_to_pi(phi + sigma * torch.randn_like(phi))
    R_new = _nerf_reconstruct(anchor, theta_n, phi_n)

    D_new = torch.cdist(R_new.unsqueeze(0), R_new.unsqueeze(0)).squeeze(0)
    d_ij = D_new[ni, nj]
    q_val = float((1.0 / (1.0 + torch.exp(2.0 * (d_ij - d0)))).mean().item())
    com_new = R_new.mean(dim=0)
    rg_new = float(torch.sqrt(((R_new - com_new) ** 2).sum(-1).mean()).item())

    return {
        "R": R_new.numpy(),
        "sigma": sigma,
        "q": q_val,
        "rg_ratio": rg_new / rg_nat,
    }


def _pre_generate_round_data(train_loader, n_proteins, n_decoys, sigma_min, sigma_max, device):
    """Pre-generate decoys for n_proteins at round start.

    Phase A of each round:
      1. Collect n_proteins from shuffled train_loader
      2. Pre-compute contacts for each
      3. Submit n_proteins * n_decoys tasks to ProcessPoolExecutor (64 CPUs)
      4. Return list of protein dicts with seq, R_native, decoys (all on CPU)
    """
    t0 = time.time()

    # Step 1: Collect proteins from loader
    proteins = []
    for batch in train_loader:
        R = batch[0]
        seq_batch = batch[1]
        lengths = batch[4] if len(batch) >= 5 else None
        B = R.shape[0]
        if lengths is None:
            lengths = torch.full((B,), R.shape[1])

        for b in range(B):
            Lb = int(lengths[b].item())
            if Lb < 4:
                continue
            proteins.append(
                {
                    "R_native": R[b, :Lb].detach().cpu(),
                    "seq": seq_batch[b, :Lb].detach().cpu(),
                    "Lb": Lb,
                }
            )
            if len(proteins) >= n_proteins:
                break
        if len(proteins) >= n_proteins:
            break

    if not proteins:
        return []

    # Step 2: Pre-compute contacts
    for prot in proteins:
        Rb = prot["R_native"]
        Lb = prot["Lb"]
        com = Rb.mean(dim=0)
        prot["rg_nat"] = float(torch.sqrt(((Rb - com) ** 2).sum(-1).mean()).item())
        D_nat = torch.cdist(Rb.unsqueeze(0), Rb.unsqueeze(0)).squeeze(0)
        idx = torch.arange(Lb)
        sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        contact_mask = (sep > 3) & (D_nat < 9.5)
        ni, nj = contact_mask.nonzero(as_tuple=True)
        prot["ni"] = ni
        prot["nj"] = nj
        prot["d0"] = D_nat[ni, nj] if len(ni) > 0 else torch.tensor([])

    # Step 3: Build tasks — one per decoy, across all proteins
    tasks = []
    task_map = []  # (protein_idx, decoy_idx)
    for pi, prot in enumerate(proteins):
        if len(prot["ni"]) == 0:
            continue
        theta, phi = _extract_ic(prot["R_native"])
        for di in range(n_decoys):
            sigma = math.exp(random.uniform(math.log(sigma_min), math.log(sigma_max)))
            tasks.append(
                (
                    prot["R_native"][:3].clone(),
                    theta,
                    phi,
                    sigma,
                    prot["ni"],
                    prot["nj"],
                    prot["d0"],
                    prot["rg_nat"],
                    prot["Lb"],
                )
            )
            task_map.append((pi, di))

    # Step 4: Parallel NeRF — n_proteins * n_decoys tasks across 64 CPUs
    # Note: zero_grad(set_to_none=True) is called before _pre_generate_round_data
    # to clear .grad tensors that would cause double-free on fork exit.
    n_workers = min(len(tasks), int(os.environ.get("CALPHAEBM_WORKERS", min(os.cpu_count() or 4, 64))))
    logger.info(
        "  Generating %d decoys (%d proteins x %d) on %d CPUs...", len(tasks), len(proteins), n_decoys, n_workers
    )
    sys.stdout.flush()

    # Fix: use submit+as_completed instead of pool.map to avoid pipe deadlock.
    # pool.map with 32K tasks and large args (18-40KB each) can fill the 64KB
    # Linux pipe buffer during submission, blocking main in pipe_write while
    # workers block in pipe_read — a classic producer-consumer deadlock.
    # submit+as_completed decouples submission from result gathering.
    decoy_results = [None] * len(tasks)
    t_submit = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_generate_one_decoy, task): idx for idx, task in enumerate(tasks)}
        logger.info("  Decoy-gen: submitted %d tasks in %.1fs", len(futures), time.time() - t_submit)
        sys.stdout.flush()

        completed = 0
        log_every = max(1, len(tasks) // 10)
        for fut in as_completed(futures):
            idx = futures[fut]
            decoy_results[idx] = fut.result()
            completed += 1
            if completed % log_every == 0 or completed == len(tasks):
                logger.info("  Decoy-gen: %d/%d completed (%.1fs)", completed, len(tasks), time.time() - t_submit)
                sys.stdout.flush()

    # Assign decoys back to proteins
    for prot in proteins:
        prot["decoys"] = []
    for (pi, di), dec_result in zip(task_map, decoy_results):
        # Convert numpy back to tensor (was numpy in worker to avoid shared memory)
        dec_result["R"] = torch.from_numpy(dec_result["R"])
        proteins[pi]["decoys"].append(dec_result)

    # Compute full dRMSD for each decoy sequentially — cheap O(L²) op,
    # avoids passing R_native through the ProcessPoolExecutor pickle pipe.
    for prot in proteins:
        Rb = prot["R_native"]
        Lb = prot["Lb"]
        D_nat = torch.cdist(Rb.unsqueeze(0), Rb.unsqueeze(0)).squeeze(0)
        idx = torch.arange(Lb)
        triu_mask = ((idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= 4) & (idx.unsqueeze(0) > idx.unsqueeze(1))
        d_nat_flat = D_nat[:Lb, :Lb][triu_mask]
        for dec in prot["decoys"]:
            R_new = dec["R"]
            D_new = torch.cdist(R_new.unsqueeze(0), R_new.unsqueeze(0)).squeeze(0)
            d_new_flat = D_new[:Lb, :Lb][triu_mask]
            dec["drmsd"] = (
                float(torch.sqrt(((d_new_flat - d_nat_flat) ** 2).mean()).item()) if d_nat_flat.numel() > 0 else 0.0
            )

    # Filter: only keep proteins with all decoys and valid contacts
    valid = [p for p in proteins if len(p.get("decoys", [])) == n_decoys and len(p["ni"]) > 0]

    elapsed = time.time() - t0
    logger.info(
        "  Pre-generated %d proteins x %d decoys = %d total (%.1fs)",
        len(valid),
        n_decoys,
        len(valid) * n_decoys,
        elapsed,
    )
    return valid


def _collate_protein_batch(proteins, device):
    """Pad a list of protein dicts into batched tensors on device.
    Padded positions use far-away coords (999A+) beyond energy cutoffs.
    Returns: R (B, L_max, 3), seq (B, L_max), lengths (B,)
    """
    B = len(proteins)
    L_max = max(p["Lb"] for p in proteins)
    FAR = 999.0
    R = torch.zeros(B, L_max, 3, device=device)
    seq = torch.zeros(B, L_max, dtype=torch.long, device=device)
    lengths = torch.zeros(B, dtype=torch.long, device=device)
    for i, prot in enumerate(proteins):
        Lb = prot["Lb"]
        R[i, :Lb] = prot["R_native"].to(device)
        if Lb < L_max:
            base_x = FAR + i * 200.0
            R[i, Lb:, 0] = torch.arange(L_max - Lb, device=device).float() * 50.0 + base_x
        seq[i, :Lb] = prot["seq"].to(device)
        lengths[i] = Lb
    return R, seq, lengths


# =====================================================================
#  Loss functions
# =====================================================================


def _discrimination_loss(model, decoy_data, device, disc_T=2.0, disable_subterms=frozenset()):
    """Batched per-subterm discrimination using pre-generated decoys.

    Pads all (native, decoy) pairs into one (2B, L_max, 3) tensor per
    subterm call. ONE forward per subterm instead of B separate calls.

    Loss = (1/K) sum_k mean_b exp(clamp(-(E_k(decoy_b) - E_k(native_b)) / T))
    """
    if not decoy_data:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Collect valid pairs
    pairs = []
    for pdata in decoy_data:
        decoys = pdata["decoys"]
        if not decoys:
            continue
        dec = random.choice(decoys)
        pairs.append((pdata["Lb"], pdata["R_native"], pdata["seq"], dec["R"]))

    if not pairs:
        return torch.tensor(0.0, device=device, requires_grad=True)

    B = len(pairs)
    L_max = max(p[0] for p in pairs)

    # Build (2B, L_max, 3): native at even indices, decoy at odd
    # Pad with far-away atoms beyond energy cutoffs
    FAR = 999.0
    R_all = torch.zeros(2 * B, L_max, 3, device=device)
    seq_all = torch.zeros(2 * B, L_max, dtype=torch.long, device=device)
    lens_all = torch.zeros(2 * B, dtype=torch.long, device=device)

    for i, (Lb, R_nat, seq_p, R_dec) in enumerate(pairs):
        R_all[2 * i, :Lb] = R_nat.to(device)
        R_all[2 * i + 1, :Lb] = R_dec.to(device)
        if Lb < L_max:
            base_x = FAR + i * 200.0
            offsets_pad = torch.arange(L_max - Lb, device=device).float() * 50.0 + base_x
            R_all[2 * i, Lb:, 0] = offsets_pad
            R_all[2 * i + 1, Lb:, 0] = offsets_pad
        seq_all[2 * i, :Lb] = seq_p.to(device)
        seq_all[2 * i + 1, :Lb] = seq_p.to(device)
        lens_all[2 * i] = Lb
        lens_all[2 * i + 1] = Lb

    # ONE forward per subterm
    loss = torch.tensor(0.0, device=device)
    n_terms = 0
    for term_name in ("local", "repulsion", "secondary", "packing"):
        term = getattr(model, term_name, None)
        if term is None or term_name in disable_subterms:
            continue
        E_all = term(R_all, seq_all, lengths=lens_all)  # (2B,)
        E_nat = E_all[0::2]  # (B,)
        E_dec = E_all[1::2]  # (B,)
        gaps = E_dec - E_nat  # (B,) — positive = correct
        loss = loss + torch.exp((-gaps / disc_T).clamp(max=5.0)).mean()
        n_terms += 1

    if n_terms > 0:
        loss = loss / n_terms
    return loss


def _dsm_on_decoys(model, decoy_data, device):
    """Denoising Score Matching using ALL pre-generated decoys per protein.

    For each protein, batches all n_decoys into (D, Lb, 3) — same length,
    no padding needed. ONE autograd.grad call per protein for all decoys.

    Per decoy:
      delta       = R_native - R_decoy             (displacement)
      sigma_eff^2 = (1/L) sum_i ||delta_i||^2      (effective noise)
      target      = delta / sigma_eff^2             (denoising direction)
      score       = -dE/dR                          (model force)
      loss        = sigma_eff^2 * ||score - target||^2 / L

    8 proteins × 8 decoys = 64 score targets per step,
    computed via 8 autograd.grad calls (one per protein).
    """
    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for pdata in decoy_data:
        Lb = pdata["Lb"]
        decoys = pdata["decoys"]
        if not decoys:
            continue

        D = len(decoys)
        R_nat_cpu = pdata["R_native"]  # (Lb, 3) on CPU

        # Pre-compute targets and sigma_eff for all decoys (detached)
        with torch.no_grad():
            R_nat_dev = R_nat_cpu.to(device)  # (Lb, 3)
            targets = []
            sigma_sq_vals = []
            valid_indices = []
            for di, dec in enumerate(decoys):
                delta = R_nat_dev - dec["R"].to(device)  # (Lb, 3)
                s2 = float((delta**2).sum(-1).mean().item())
                if s2 < 1e-8:
                    continue
                targets.append(delta / s2)  # (Lb, 3)
                sigma_sq_vals.append(s2)
                valid_indices.append(di)

        if not valid_indices:
            continue

        D_valid = len(valid_indices)

        # Stack valid decoys into (D_valid, Lb, 3) — same Lb, no padding
        R_dec_batch = (
            torch.stack([decoys[di]["R"].to(device) for di in valid_indices]).detach().requires_grad_(True)
        )  # (D_valid, Lb, 3)

        target_batch = torch.stack(targets)  # (D_valid, Lb, 3) detached
        seq_batch = pdata["seq"].to(device).unsqueeze(0).expand(D_valid, -1)  # (D_valid, Lb)
        lens_batch = torch.full((D_valid,), Lb, device=device)

        # ONE forward + ONE autograd.grad for all decoys of this protein
        E_all = model(R_dec_batch, seq_batch, lengths=lens_batch)  # (D_valid,)
        grad_E = torch.autograd.grad(E_all.sum(), R_dec_batch, create_graph=True)[0]
        score = -grad_E  # (D_valid, Lb, 3)

        # Per-decoy loss with sigma_eff^2 weighting
        for k in range(D_valid):
            diff = (score[k] - target_batch[k]) ** 2  # (Lb, 3)
            loss_k = sigma_sq_vals[k] * diff.sum() / Lb
            if torch.isfinite(loss_k):
                total_loss = total_loss + loss_k
                n_valid += 1

    if n_valid > 0:
        return total_loss / n_valid
    return torch.tensor(0.0, device=device, requires_grad=True)


def _funnel_and_gap_losses(
    model,
    decoy_data,
    device,
    lambda_qf=1.0,
    lambda_drmsd=2.0,
    lambda_gap=1.0,
    T_funnel=2.0,
    gap_margin=0.5,
    min_dq=0.05,
    min_ddrmsd=0.5,
    slope_clamp=10.0,
    funnel_m=5.0,
    funnel_alpha=5.0,
    gap_m=5.0,
    gap_alpha=5.0,
):
    """Batched Q-funnel + dRMSD-funnel + Gap losses.

    ONE batched model forward for all proteins x (1 native + n_decoys).
    Then per-protein pairwise slope computation (pure tensor ops, no model calls).

    Returns dict with loss tensors: {"qf": ..., "drmsd": ..., "gap": ...}
    Only includes losses with lambda > 0.
    """
    if not decoy_data:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return {"qf": zero, "drmsd": zero, "gap": zero}

    n_per = [1 + len(pdata["decoys"]) for pdata in decoy_data]
    total_N = sum(n_per)
    L_max = max(pdata["Lb"] for pdata in decoy_data)

    FAR = 999.0
    R_all = torch.zeros(total_N, L_max, 3, device=device)
    seq_all = torch.zeros(total_N, L_max, dtype=torch.long, device=device)
    lens_all = torch.zeros(total_N, dtype=torch.long, device=device)

    q_per_protein = []
    drmsd_per_protein = []  # full dRMSD per structure (0 for native)
    offsets = []

    idx = 0
    for pi, pdata in enumerate(decoy_data):
        Lb = pdata["Lb"]
        decoys = pdata["decoys"]
        N_i = 1 + len(decoys)

        offsets.append(idx)
        base_x = FAR + pi * 200.0
        offsets_pad = None
        if Lb < L_max:
            offsets_pad = torch.arange(L_max - Lb, device=device).float() * 50.0 + base_x

        R_all[idx, :Lb] = pdata["R_native"].to(device)
        if offsets_pad is not None:
            R_all[idx, Lb:, 0] = offsets_pad
        seq_all[idx, :Lb] = pdata["seq"].to(device)
        lens_all[idx] = Lb

        q_vals = [1.0]
        drmsd_vals = [0.0]  # native has dRMSD = 0 by definition

        for di, dec in enumerate(decoys):
            R_all[idx + 1 + di, :Lb] = dec["R"].to(device)
            if offsets_pad is not None:
                R_all[idx + 1 + di, Lb:, 0] = offsets_pad
            seq_all[idx + 1 + di, :Lb] = pdata["seq"].to(device)
            lens_all[idx + 1 + di] = Lb
            q_vals.append(dec["q"])
            drmsd_vals.append(dec["drmsd"])  # pre-computed in worker, free

        q_per_protein.append(torch.tensor(q_vals, device=device))
        drmsd_per_protein.append(torch.tensor(drmsd_vals, device=device))
        idx += N_i

    # ── ONE batched model forward ────────────────────────────────────────
    E_all = model(R_all, seq_all, lengths=lens_all)  # (total_N,)

    # ── Compute losses per protein ───────────────────────────────────────
    results = {}

    # Q-funnel
    if lambda_qf > 0:
        total_qf = torch.tensor(0.0, device=device)
        n_qf = 0
        for pi, pdata in enumerate(decoy_data):
            N_i = n_per[pi]
            E_p = E_all[offsets[pi] : offsets[pi] + N_i]
            Q_p = q_per_protein[pi]
            from calphaebm.training.losses.elt_losses import q_funnel_loss

            _qf_loss, _qf_n, _ = q_funnel_loss(
                E_p, Q_p, m=funnel_m, alpha=funnel_alpha, threshold=min_dq, clamp_max=5.0
            )
            if _qf_n > 0:
                total_qf = total_qf + _qf_loss * _qf_n
                n_qf += _qf_n
        results["qf"] = total_qf / max(n_qf, 1)

    # dRMSD-funnel (replaces Rg-funnel)
    if lambda_drmsd > 0:
        total_drmsd = torch.tensor(0.0, device=device)
        n_drmsd = 0
        for pi, pdata in enumerate(decoy_data):
            N_i = n_per[pi]
            E_p = E_all[offsets[pi] : offsets[pi] + N_i]
            D_p = drmsd_per_protein[pi]
            from calphaebm.training.losses.elt_losses import drmsd_funnel_loss

            _dr_loss, _dr_n, _ = drmsd_funnel_loss(
                E_p, D_p, m=funnel_m, alpha=funnel_alpha, threshold=min_ddrmsd, clamp_max=5.0
            )
            if _dr_n > 0:
                total_drmsd = total_drmsd + _dr_loss * _dr_n
                n_drmsd += _dr_n
        results["drmsd"] = total_drmsd / max(n_drmsd, 1)

    # Gap
    if lambda_gap > 0:
        total_gap = torch.tensor(0.0, device=device)
        n_gap = 0
        for pi, pdata in enumerate(decoy_data):
            N_i = n_per[pi]
            E_p = E_all[offsets[pi] : offsets[pi] + N_i]
            Q_decoys = q_per_protein[pi][1:]
            delta_Q = (1.0 - Q_decoys).clamp(min=0.0)
            from calphaebm.training.losses.elt_losses import _saturating_margin

            required_gap = _saturating_margin(delta_Q, gap_m, gap_alpha)
            gaps = E_p[1:] - E_p[0] - required_gap
            pair_loss = torch.exp((-gaps).clamp(max=5.0))
            total_gap = total_gap + pair_loss.sum()
            n_gap += len(E_p) - 1
        results["gap"] = total_gap / max(n_gap, 1)

    return results


# =====================================================================
#  Basin evaluation (quick Langevin probe)
# =====================================================================


def _run_basin_eval(
    model,
    val_loader,
    n_proteins=64,
    n_steps=5000,
    beta=100.0,
    step_size=0.005,
    force_cap=50.0,
    n_workers=64,
    max_len=128,
    round_num=0,
    sampler="langevin",
):
    """Run quick basin eval on val set via subprocess.

    Same approach as SC training: save model + structures to temp files,
    launch a fresh Python subprocess (never touched CUDA), which forks
    workers safely. Avoids CUDA/autograd fork crash after GPU training.
    """
    import copy
    import gc
    import os
    import subprocess
    import sys
    import tempfile

    model.eval()
    model.zero_grad(set_to_none=True)
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    model_cpu = copy.deepcopy(model)
    model_cpu.eval()

    structures = []
    for batch in val_loader:
        R, seq = batch[0], batch[1]
        pdb_ids = batch[2] if len(batch) > 2 else [f"unk_{i}" for i in range(R.shape[0])]
        chain_ids = batch[3] if len(batch) > 3 else ["A"] * R.shape[0]
        lengths = batch[4] if len(batch) >= 5 else None
        B = R.shape[0]
        if lengths is None:
            lengths = torch.full((B,), R.shape[1])
        for b in range(B):
            Lb = int(lengths[b].item())
            if Lb < 10 or Lb > max_len:
                continue
            structures.append(
                (
                    R[b, :Lb].detach().cpu().clone(),
                    seq[b, :Lb].detach().cpu().clone(),
                    pdb_ids[b] if b < len(pdb_ids) else "unk",
                    chain_ids[b] if b < len(chain_ids) else "A",
                    Lb,
                )
            )
            if len(structures) >= n_proteins:
                break
        if len(structures) >= n_proteins:
            break

    if not structures:
        logger.warning("No val structures found for basin eval")
        return {}

    n_structs = len(structures)
    logger.info("  Eval: %d structures (L≤%d) at beta=%.1f (%d steps)", n_structs, max_len, beta, n_steps)

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
            "calphaebm.training.eval_subprocess",
            "--model-path",
            model_path,
            "--structures-path",
            struct_path,
            "--results-path",
            result_path,
            "--n-workers",
            str(min(n_workers, n_structs)),
            "--beta",
            str(beta),
            "--n-steps",
            str(n_steps),
            "--sampler",
            sampler,
        ]
        logger.info("  Launching subprocess eval (%d workers)...", min(n_workers, n_structs))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout in real-time so eval progress is visible
        stdout_lines = []
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    stdout_lines.append(line)
                    logger.info("  [subprocess] %s", line)
            proc.wait(timeout=int(_T["eval_timeout"]))
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.error("  Subprocess eval timed out after %d seconds", int(_T["eval_timeout"]))
            return {"n_ok": 0, "n_total": n_structs}
        if proc.returncode != 0:
            stderr_out = proc.stderr.read() if proc.stderr else ""
            logger.error(
                "  Subprocess failed (rc=%d):\n%s", proc.returncode, stderr_out[-1000:] if stderr_out else "no stderr"
            )
            return {"n_ok": 0, "n_total": n_structs}

        results = torch.load(result_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.error("  Subprocess eval error: %s", e)
        return {"n_ok": 0, "n_total": n_structs}
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

    ok = [r for r in results if r.get("error") is None]
    if not ok:
        errors = [r.get("error", "unknown") for r in results if r.get("error")]
        for i, err in enumerate(errors[:3]):
            logger.warning("  Basin eval structure %d error: %s", i, err)
        return {"n_ok": 0, "n_total": len(results)}

    # ── Compute Rama/dphi correlations ─────────────────────────────
    rama_corr = 0.0
    dphi_corr = 0.0
    try:
        from calphaebm.training.validation.metrics import (
            compute_delta_phi_correlation,
            compute_ramachandran_correlation,
        )

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
    summary = vlog.log_eval_block(
        round_num=round_num,
        beta=beta,
        n_steps=n_steps,
        results=results,
        structures=structures,
        rama_corr=rama_corr,
        dphi_corr=dphi_corr,
    )
    # Add n_ok/n_total for backward compat with caller
    summary["n_ok"] = len(ok)
    summary["n_total"] = len(results)
    # Short key aliases for full_stage caller compatibility
    summary["rmsd"] = summary.get("rmsd_mean", 0.0)
    summary["q"] = summary.get("q_mean", 0.0)
    summary["rmsf"] = summary.get("rmsf_mean", 0.0)
    summary["e_delta"] = summary.get("e_delta_mean", 0.0)
    return summary


# =====================================================================
#  Single training round (pre-generate + train)
# =====================================================================


def _train_round(
    trainer,
    train_loader,
    n_steps,
    lr,
    lr_final,
    lambda_depth,
    target_depth,
    lambda_balance,
    lambda_dsm,
    dsm_sigma_min,
    dsm_sigma_max,
    lambda_discrim,
    disc_T,
    lambda_qf,
    lambda_drmsd,
    lambda_gap,
    gap_margin,
    sigma_min,
    sigma_max,
    n_decoys,
    T_funnel,
    decoy_every,
    discrim_every,
    disable_subterms,
    log_every,
    collect_proteins=1024,
    funnel_m=5.0,
    funnel_alpha=5.0,
    gap_m=5.0,
    gap_alpha=5.0,
    round_num=1,
    max_rounds=10,
):
    """Run one training round in two phases:

    Phase A: Pre-generate decoys for collect_proteins (parallel, ~30s)
    Phase B: Train n_steps sampling batches of 8 from stored proteins (fast)

    All losses per step use the SAME batch of proteins — native R, seq,
    and decoys are always correctly paired.
    """
    device = trainer.device
    batch_size = 32

    # ── Phase A: Pre-generate decoys ─────────────────────────────────────
    # Clear grad tensors before fork — learnable buffer Parameters accumulate
    # .grad during training; forked workers share the parent's memory and
    # trigger "double free or corruption" when their GC frees the same grads.
    trainer.model.zero_grad(set_to_none=True)
    stored_proteins = _pre_generate_round_data(
        train_loader,
        n_proteins=collect_proteins,
        n_decoys=n_decoys,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
    )
    if not stored_proteins:
        logger.warning("No proteins collected — skipping round")
        return {}

    # ── Phase B: Train ───────────────────────────────────────────────────
    params = [p for p in trainer.model.parameters() if p.requires_grad]

    # Cosine-with-warm-restarts: peak LR decays linearly across rounds.
    #   Round 1: peak=lr, Round max_rounds: peak=lr_final
    #   Each round does cosine from peak → lr_final.
    if max_rounds > 1:
        peak_lr = lr_final + (lr - lr_final) * (max_rounds - round_num) / (max_rounds - 1)
    else:
        peak_lr = lr
    logger.info("  LR schedule: peak=%.2e → floor=%.2e (round %d/%d)", peak_lr, lr_final, round_num, max_rounds)

    optimizer = torch.optim.Adam(params, lr=peak_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_steps,
        eta_min=lr_final,
    )

    trainer.model.train()
    n_stored = len(stored_proteins)

    ema = {}
    ema_beta = 0.95

    def _eu(key, val):
        if val is None:
            return
        ema[key] = ema_beta * ema.get(key, val) + (1 - ema_beta) * val

    # Gradient clipping stats
    _clip_count = 0
    _clip_total = 0
    _max_grad_norm = 0.0

    # Diagnostic logger (uses model's non-bonded exclude, correct safety metrics)
    diag_logger = DiagnosticLogger(trainer.model, device)

    for step in range(1, n_steps + 1):
        trainer.global_step += 1
        trainer.phase_step = step

        # Sample batch of 8 proteins from stored set
        indices = random.sample(range(n_stored), min(batch_size, n_stored))
        batch_proteins = [stored_proteins[i] for i in indices]

        # Collate into padded tensors (same proteins for ALL losses)
        R, seq, lengths = _collate_protein_batch(batch_proteins, device)

        total_loss = torch.tensor(0.0, device=device)
        _v = {k: None for k in ("depth", "bal", "dsm", "disc", "qf", "drmsd", "gap")}
        _e_native = None

        try:
            # -- Depth --
            if lambda_depth > 0:
                E_nat = trainer.model(R, seq, lengths=lengths).mean()
                _e_native = float(E_nat.item())
                loss_depth = torch.exp((E_nat - target_depth).clamp(max=5.0))
                if torch.isfinite(loss_depth):
                    total_loss = total_loss + lambda_depth * loss_depth
                    _v["depth"] = (float(loss_depth.item()), float((lambda_depth * loss_depth).item()))

            # -- Balance (ramp 1e-6 -> lambda_balance in round 1 only) --
            if lambda_balance > 0:
                if round_num == 1:  # only ramp in the very first round
                    frac = min(step / max(n_steps, 1), 1.0)
                    bal_w = 1e-6 * (lambda_balance / 1e-6) ** frac
                else:
                    bal_w = lambda_balance
                loss_bal, _, _ = energy_balance_loss(
                    trainer.model,
                    R,
                    seq,
                    r=float(_T["balance_r"]),
                    r_term=float(_T["balance_r_term"]),
                    lengths=lengths,
                    exclude_subterms=disable_subterms,
                )
                if torch.isfinite(loss_bal):
                    total_loss = total_loss + bal_w * loss_bal
                    _v["bal"] = (float(loss_bal.item()), float((bal_w * loss_bal).item()))

            # -- Build decoy_data for this batch (all decoy-based losses use this) --
            decoy_batch = []
            for prot in batch_proteins:
                decoy_batch.append(
                    {
                        "R_native": prot["R_native"],
                        "seq": prot["seq"],
                        "Lb": prot["Lb"],
                        "rg_nat": prot["rg_nat"],
                        "ni": prot["ni"],
                        "nj": prot["nj"],
                        "d0": prot["d0"],
                        "decoys": prot["decoys"],
                    }
                )

            # -- DSM on pre-generated decoys (no NeRF) --
            if lambda_dsm > 0 and decoy_batch:
                loss_dsm = _dsm_on_decoys(
                    trainer.model,
                    decoy_batch,
                    device,
                )
                if loss_dsm is not None and torch.isfinite(loss_dsm):
                    total_loss = total_loss + lambda_dsm * loss_dsm
                    _v["dsm"] = (float(loss_dsm.item()), float((lambda_dsm * loss_dsm).item()))

            # -- Discrimination on pre-generated decoys (no NeRF) --
            if lambda_discrim > 0 and step % discrim_every == 0 and decoy_batch:
                loss_disc = _discrimination_loss(
                    trainer.model,
                    decoy_batch,
                    device,
                    disc_T=disc_T,
                    disable_subterms=disable_subterms,
                )
                if torch.isfinite(loss_disc):
                    total_loss = total_loss + lambda_discrim * loss_disc
                    _v["disc"] = (float(loss_disc.item()), float((lambda_discrim * loss_disc).item()))

            # -- Funnel + Gap losses (ONE batched forward for all) --
            if decoy_batch and (lambda_qf > 0 or lambda_drmsd > 0 or lambda_gap > 0):
                fg = _funnel_and_gap_losses(
                    trainer.model,
                    decoy_batch,
                    device,
                    lambda_qf=lambda_qf,
                    lambda_drmsd=lambda_drmsd,
                    lambda_gap=lambda_gap,
                    T_funnel=T_funnel,
                    gap_margin=gap_margin,
                    funnel_m=funnel_m,
                    funnel_alpha=funnel_alpha,
                    gap_m=gap_m,
                    gap_alpha=gap_alpha,
                )
                if "qf" in fg and torch.isfinite(fg["qf"]):
                    total_loss = total_loss + lambda_qf * fg["qf"]
                    _v["qf"] = (float(fg["qf"].item()), float((lambda_qf * fg["qf"]).item()))
                if "drmsd" in fg and torch.isfinite(fg["drmsd"]):
                    total_loss = total_loss + lambda_drmsd * fg["drmsd"]
                    _v["drmsd"] = (float(fg["drmsd"].item()), float((lambda_drmsd * fg["drmsd"]).item()))
                if "gap" in fg and torch.isfinite(fg["gap"]):
                    total_loss = total_loss + lambda_gap * fg["gap"]
                    _v["gap"] = (float(fg["gap"].item()), float((lambda_gap * fg["gap"]).item()))

            # -- Backward + step --
            trainer.current_loss = float(total_loss.item())
            if torch.isfinite(total_loss) and total_loss.requires_grad:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                _clip_total += 1
                if grad_norm > 10.0:
                    _clip_count += 1
                _max_grad_norm = max(_max_grad_norm, float(grad_norm))
                optimizer.step()

                # Clamp penalty multipliers to non-negative — they can go to
                # zero (model kills the term) but must never flip sign.
                with torch.no_grad():
                    for _pname in ("coord_lambda", "coord_m", "rho_penalty_lambda", "rho_m"):
                        _p = getattr(trainer.model.packing, _pname, None)
                        if _p is not None and isinstance(_p, torch.nn.Parameter):
                            _p.clamp_(min=0.0)

            scheduler.step()

        except Exception as e:
            logger.error("Error at step %d: %s", step, e)
            import traceback

            traceback.print_exc()
            continue

        # -- EMA --
        _eu("total", trainer.current_loss)
        for k, v in _v.items():
            if v is None:
                continue
            if isinstance(v, tuple):
                _eu(k + "_raw", v[0])
                _eu(k, v[1])
            else:
                _eu(k, v)
        if _e_native is not None:
            _eu("e_nat", _e_native)

        # -- Log --
        if step % log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            parts = [f"loss={ema.get('total', 0):.4f}"]
            for k in ("depth", "bal", "dsm", "disc", "qf", "drmsd", "gap"):
                if k in ema:
                    eff = ema[k]
                    raw_key = k + "_raw"
                    if raw_key in ema:
                        raw = ema[raw_key]
                        lam = eff / raw if abs(raw) > 1e-10 else 0
                        parts.append(f"{k}={eff:.3f}({lam:.1e}x{raw:.1f})")
                    else:
                        parts.append(f"{k}={eff:.3f}")
            if "e_nat" in ema:
                parts.append(f"E={ema['e_nat']:.3f}")
            logger.info("  step %5d/%d | lr=%.2e | %s", step, n_steps, current_lr, "  ".join(parts))

        # -- Detailed diagnostics every 500 steps --
        if step % 500 == 0:
            try:
                trainer.model.eval()
                with torch.no_grad():
                    # ── Compute disc/funnel from decoys for precomputed ──
                    import random as _diag_random

                    _precomputed = {
                        "clip_frac": _clip_count / max(_clip_total, 1),
                        "max_force": _max_grad_norm,
                    }

                    if decoy_batch:
                        # Discrimination gaps per term
                        _disc = {tn: [] for tn in ("local", "secondary", "repulsion", "packing")}
                        _disc_c = {"hp_pen": [], "rho_pen": [], "rg": []}
                        for prot in decoy_batch:
                            dec = _diag_random.choice(prot["decoys"])
                            Lb_d = prot["Lb"]
                            R_n = prot["R_native"].to(device).unsqueeze(0)
                            R_d = dec["R"].to(device).unsqueeze(0)
                            s_d = prot["seq"].to(device).unsqueeze(0)
                            l_d = torch.tensor([Lb_d], device=device)
                            for tn in ("local", "secondary", "repulsion", "packing"):
                                t = getattr(trainer.model, tn, None)
                                if t is not None:
                                    _disc[tn].append(t(R_d, s_d, lengths=l_d).item() - t(R_n, s_d, lengths=l_d).item())
                            pack_m = getattr(trainer.model, "packing", None)
                            if pack_m is not None and hasattr(pack_m, "subterm_energies"):
                                try:
                                    st_n = pack_m.subterm_energies(R_n, s_d, lengths=l_d)
                                    st_d = pack_m.subterm_energies(R_d, s_d, lengths=l_d)
                                    if len(st_n) == 5:
                                        # v5: (hp_rew, hp_pen, rho_rew, rho_pen, rg_pen)
                                        _disc_c["hp_pen"].append(st_d[1].item() - st_n[1].item())
                                        _disc_c["rho_pen"].append(st_d[3].item() - st_n[3].item())
                                        _disc_c["rg"].append(st_d[4].item() - st_n[4].item())
                                    else:
                                        # v4 fallback: (E_hp, E_coord, E_rg)
                                        _disc_c["hp_pen"].append(st_d[1].item() - st_n[1].item())
                                        _disc_c["rg"].append(st_d[2].item() - st_n[2].item())
                                except Exception:
                                    pass
                        disc_gaps = {}
                        for tn in ("local", "secondary", "repulsion", "packing"):
                            if _disc[tn]:
                                disc_gaps[tn] = sum(_disc[tn]) / len(_disc[tn])
                        for cn in ("hp_pen", "rho_pen", "rg"):
                            if _disc_c.get(cn):
                                disc_gaps[cn] = sum(_disc_c[cn]) / len(_disc_c[cn])
                        _precomputed["disc_gaps"] = disc_gaps

                        # Q-funnel / dRMSD-funnel from decoys
                        _nqp, _nqa, _ndrp, _ndra = 0, 0, 0, 0
                        _slopes_all = []
                        for prot in decoy_batch:
                            Lb_d = prot["Lb"]
                            decoys = prot["decoys"]
                            N_d = len(decoys) + 1
                            R_f = torch.zeros(N_d, Lb_d, 3, device=device)
                            R_f[0] = prot["R_native"].to(device)
                            q_v = [1.0]
                            dr_v = [0.0]
                            for di, dec in enumerate(decoys):
                                R_f[di + 1] = dec["R"].to(device)
                                q_v.append(dec["q"])
                                dr_v.append(dec["drmsd"])
                            q_t = torch.tensor(q_v, device=device)
                            dr_t = torch.tensor(dr_v, device=device)
                            s_f = prot["seq"].to(device).unsqueeze(0).expand(N_d, -1)
                            l_f = torch.full((N_d,), Lb_d, device=device)
                            E_f = trainer.model(R_f, s_f, lengths=l_f)
                            dQ = q_t.unsqueeze(1) - q_t.unsqueeze(0)
                            dE = E_f.unsqueeze(1) - E_f.unsqueeze(0)
                            valid_q = dQ > 0.05
                            if valid_q.any():
                                sl = (dE / dQ.clamp(min=0.05))[valid_q]
                                _slopes_all.append(sl)
                                _nqp += int(valid_q.sum().item())
                                _nqa += int((sl > 0).sum().item())
                            # dRMSD anti-funnel: i has higher dRMSD but lower/equal energy
                            dd = dr_t.unsqueeze(1) - dr_t.unsqueeze(0)  # dd[i,j]=drmsd_i-drmsd_j
                            dE_dr = E_f.unsqueeze(1) - E_f.unsqueeze(0)
                            valid_dr = dd > 0.5
                            if valid_dr.any():
                                _ndrp += int(valid_dr.sum().item())
                                _ndra += int(((dE_dr <= 0) & valid_dr).sum().item())

                        q_af = 100.0 * _nqa / max(_nqp, 1)
                        drmsd_af = 100.0 * _ndra / max(_ndrp, 1)
                        ms = float(torch.cat(_slopes_all).mean().item()) if _slopes_all else 0.0
                        _precomputed["funnel"] = {
                            "mean_slope": ms,
                            "q_af": q_af,
                            "drmsd_af": drmsd_af,
                            "n_qf_pairs": _nqp,
                            "n_qf_anti": _nqa,
                            "n_dr_pairs": _ndrp,
                            "n_dr_anti": _ndra,
                        }
                        ema["q_af"] = q_af
                        ema["drmsd_af"] = drmsd_af

                    # ── Call the beautiful log_step_block ──
                    diag_logger.log_step_block(
                        phase_step=step,
                        n_steps=n_steps,
                        loss=trainer.current_loss,
                        lr=scheduler.get_last_lr()[0],
                        R=R,
                        seq=seq,
                        lengths=lengths,
                        loss_dsm=_v.get("dsm", (None,))[0] if isinstance(_v.get("dsm"), tuple) else _v.get("dsm"),
                        lambda_native_depth=lambda_depth,
                        target_native_depth=target_depth,
                        precomputed=_precomputed,
                    )
                trainer.model.train()
            except Exception as e:
                logger.debug("Diagnostic error at step %d: %s", step, e)
                trainer.model.train()

            # Reset clip stats for next window
            _clip_count = 0
            _clip_total = 0
            _max_grad_norm = 0.0

            # Periodic checkpoint
            trainer.save_checkpoint(
                f"full-stage/full_step{step:05d}",
                step,
                trainer.current_loss,
            )

    return ema


# =====================================================================
#  Main entry point: round loop
# =====================================================================


def run_full_stage(trainer, config, train_loader, val_loader=None):
    """Stage 1: PDB-only training with all losses, running in rounds.

    Each round:
      Phase A: Pre-generate decoys for 128 proteins (parallel CPUs)
      Phase B: Train steps_per_round steps sampling from stored proteins
      Phase C: Basin eval (short Langevin)
      Phase D: Log metrics, check convergence
    """

    # -- Parse config (defaults from calphaebm.defaults.TRAIN) --
    max_rounds = int(getattr(config, "max_rounds", _T["max_rounds"]))
    steps_per_round = int(getattr(config, "steps_per_round", _T["steps_per_round"]))
    lr = float(getattr(config, "lr", _T["lr"]))
    lr_final = float(getattr(config, "lr_final", _T["lr_final"]))
    log_every = int(getattr(config, "log_every", _T["log_every"]))

    # PDB batch losses
    lambda_depth = float(getattr(config, "lambda_depth", _T["lambda_depth"]))
    target_depth = float(getattr(config, "target_depth", _T["target_depth"]))
    lambda_balance = float(getattr(config, "lambda_balance", _T["lambda_balance"]))
    lambda_dsm = float(getattr(config, "lambda_dsm", _T["lambda_dsm"]))

    # IC-noised losses
    lambda_discrim = float(getattr(config, "lambda_discrim", _T["lambda_discrim"]))
    disc_T = float(getattr(config, "disc_T", _T["disc_T"]))
    lambda_qf = float(getattr(config, "lambda_qf", _T["lambda_qf"]))
    lambda_drmsd = float(getattr(config, "lambda_drmsd", _T.get("lambda_drmsd", 2.0)))
    lambda_gap = float(getattr(config, "lambda_gap", _T["lambda_gap"]))
    gap_margin = float(getattr(config, "gap_margin", _T["gap_margin"]))

    # Saturating exponential margins (Run5)
    funnel_m = float(getattr(config, "funnel_m", _T["funnel_m"]))
    funnel_alpha = float(getattr(config, "funnel_alpha", _T["funnel_alpha"]))
    gap_m = float(getattr(config, "gap_m", _T["gap_m"]))
    gap_alpha = float(getattr(config, "gap_alpha", _T["gap_alpha"]))

    # IC noise params
    sigma_min = float(getattr(config, "sigma_min", _T["sigma_min"]))
    sigma_max = float(getattr(config, "sigma_max", _T["sigma_max"]))
    n_decoys = int(getattr(config, "n_decoys", _T["n_decoys"]))
    T_funnel = float(getattr(config, "T_funnel", _T["T_funnel"]))

    # DSM sigma
    dsm_sigma_min = float(getattr(config, "dsm_sigma_min", sigma_min))
    dsm_sigma_max = float(getattr(config, "dsm_sigma_max", sigma_max))

    # Disabled subterms
    disable_subterms = set(getattr(config, "disable_subterms", []) or [])

    # Amortise
    decoy_every = int(getattr(config, "decoy_every", _T["decoy_every"]))
    discrim_every = int(getattr(config, "discrim_every", _T["discrim_every"]))

    # Collection
    collect_proteins = int(getattr(config, "collect_proteins", 1024))

    logger.info("=" * 66)
    logger.info("  STAGE 1: FULL PDB-ONLY TRAINING (ROUND-BASED)")
    logger.info(
        "  %d rounds x %d steps = %d total, lr=%.1e -> %.1e",
        max_rounds,
        steps_per_round,
        max_rounds * steps_per_round,
        lr,
        lr_final,
    )
    logger.info("  PDB:    depth=%.2f  bal=%.3f  dsm=%.2f", lambda_depth, lambda_balance, lambda_dsm)
    logger.info(
        "  Decoys: disc=%.2f(T=%.1f)  qf=%.2f  drmsd=%.2f  gap=%.2f",
        lambda_discrim,
        disc_T,
        lambda_qf,
        lambda_drmsd,
        lambda_gap,
    )
    logger.info("  Margin: funnel(m=%.1f,α=%.1f)  gap(m=%.1f,α=%.1f)", funnel_m, funnel_alpha, gap_m, gap_alpha)
    logger.info(
        "  IC noise: sigma ~ LogU(%.3f, %.3f) rad, %d decoys, %d proteins/round",
        sigma_min,
        sigma_max,
        n_decoys,
        collect_proteins,
    )
    logger.info("  Eval:   detached — run eval_watcher.py independently")
    logger.info("=" * 66)

    round_history: List[RoundMetrics] = []

    # ── Resume: compute start round from restored global_step ─────────
    start_round = 1
    if trainer.global_step > 0 and steps_per_round > 0:
        completed_rounds = trainer.global_step // steps_per_round
        start_round = completed_rounds + 1
        if start_round > 1:
            logger.info(
                "  Resuming from round %d (global_step=%d, %d rounds completed)",
                start_round,
                trainer.global_step,
                completed_rounds,
            )

    for round_num in range(start_round, max_rounds + 1):
        logger.info("-" * 50)
        logger.info("  ROUND %d/%d", round_num, max_rounds)
        logger.info("-" * 50)

        # -- Train --
        ema = _train_round(
            trainer=trainer,
            train_loader=train_loader,
            n_steps=steps_per_round,
            lr=lr,
            lr_final=lr_final,
            lambda_depth=lambda_depth,
            target_depth=target_depth,
            lambda_balance=lambda_balance,
            lambda_dsm=lambda_dsm,
            dsm_sigma_min=dsm_sigma_min,
            dsm_sigma_max=dsm_sigma_max,
            lambda_discrim=lambda_discrim,
            disc_T=disc_T,
            lambda_qf=lambda_qf,
            lambda_drmsd=lambda_drmsd,
            lambda_gap=lambda_gap,
            gap_margin=gap_margin,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            n_decoys=n_decoys,
            T_funnel=T_funnel,
            decoy_every=decoy_every,
            discrim_every=discrim_every,
            disable_subterms=disable_subterms,
            log_every=log_every,
            collect_proteins=collect_proteins,
            funnel_m=funnel_m,
            funnel_alpha=funnel_alpha,
            gap_m=gap_m,
            gap_alpha=gap_alpha,
            round_num=round_num,
            max_rounds=max_rounds,
        )

        # -- Save round checkpoint (eval_watcher polls for this) --
        trainer.save_checkpoint(
            f"full-stage/full_round{round_num:03d}",
            trainer.phase_step,
            trainer.current_loss,
        )

        # -- Round training summary (training metrics only) --
        metrics = RoundMetrics(
            round_num=round_num,
            train_loss=ema.get("total", 0),
            q_af=ema.get("q_af", 100.0),
            rg_af=ema.get("drmsd_af", 100.0),
            ema=dict(ema),
        )
        round_history.append(metrics)

        logger.info("  Round %d complete:", round_num)
        logger.info(
            "    loss=%.4f  Q_af=%.1f%%  dRMSD_af=%.1f%%  E=%.3f",
            metrics.train_loss,
            metrics.q_af,
            metrics.rg_af,
            ema.get("e_nat", 0.0),
        )
        logger.info("    Checkpoint saved — eval_watcher will pick up full-stage/full_round%03d", round_num)

        # -- Round history table (training losses only) --
        logger.info("  Round history:")
        logger.info("  %5s  %8s  %6s  %7s  %7s", "Round", "Loss", "Q_af%", "dR_af%", "E/res")
        for m in round_history:
            logger.info(
                "  %5d  %8.4f  %6.1f  %7.1f  %7.3f", m.round_num, m.train_loss, m.q_af, m.rg_af, m.ema.get("e_nat", 0.0)
            )

    # -- Final checkpoint --
    trainer.save_checkpoint("full-stage/full_final", trainer.phase_step, trainer.current_loss)
    logger.info("Stage 1 complete: %d rounds, %d total steps", len(round_history), trainer.global_step)

    return TrainingState(
        global_step=trainer.global_step,
        phase_step=trainer.phase_step,
        phase="stage1",
        losses={"loss": trainer.current_loss},
        gates=trainer.model.get_gates() if hasattr(trainer.model, "get_gates") else {},
        best_composite_score=None,
        best_composite_score_initialized=False,
        best_val_step=0,
        early_stopping_counter=0,
        validation_history=[],
        converged=False,
        convergence_step=0,
        convergence_info={"rounds": len(round_history)},
    )
