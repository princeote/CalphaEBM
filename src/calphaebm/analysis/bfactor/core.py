"""Core B-factor calibration analysis.

Langevin dynamics → per-residue RMSF → simulated B-factors → correlation
with experimental B-factors from PDB X-ray structures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from calphaebm.data.pdb_parse import download_cif, parse_cif_ca_chains, split_chain_on_gaps
from calphaebm.models.total_energy import TotalEnergy
from calphaebm.utils.logging import get_logger

logger = get_logger()


# ── B-factor extraction from mmCIF ────────────────────────────────────────


def extract_ca_bfactors(
    pdb_id: str,
    cache_dir: str = "./pdb_cache",
) -> Optional[Dict]:
    """Extract Cα B-factors from a PDB/mmCIF file.

    Uses the CalphaEBM data pipeline (download_cif + parse_cif_ca_chains)
    for coordinate extraction, then reads B-factors from the same file.

    Returns:
        Dict with keys: pdb_id, chain, n_residues, bfactors (list),
        coords (Lx3 ndarray), residue_ids, mean_bfactor, std_bfactor.
        None if extraction fails.
    """
    try:
        from Bio.PDB import MMCIFParser
    except ImportError:
        logger.error("BioPython required: pip install biopython --break-system-packages")
        return None

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Download structure
    cif_path = download_cif(pdb_id, cache_dir=str(cache_path))
    if cif_path is None or not Path(cif_path).exists():
        logger.warning("Failed to download %s", pdb_id)
        return None

    # Parse with BioPython for B-factors
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, cif_path)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", pdb_id, e)
        return None

    model = structure[0]
    chains = list(model.get_chains())
    if not chains:
        logger.warning("No chains in %s", pdb_id)
        return None

    # Take first chain (matching CalphaEBM convention)
    chain = chains[0]

    bfactors = []
    coords = []
    residue_ids = []

    for residue in chain:
        if residue.id[0] != " ":
            continue
        if "CA" not in residue:
            continue
        ca = residue["CA"]
        bfactors.append(ca.get_bfactor())
        coords.append(ca.get_vector().get_array())
        residue_ids.append(residue.id[1])

    if len(bfactors) < 20:
        logger.warning("Too few Cα atoms in %s chain %s: %d", pdb_id, chain.id, len(bfactors))
        return None

    bfactors_arr = np.array(bfactors)
    return {
        "pdb_id": pdb_id,
        "chain": chain.id,
        "n_residues": len(bfactors),
        "bfactors": bfactors,
        "coords": np.array(coords),
        "residue_ids": residue_ids,
        "mean_bfactor": float(bfactors_arr.mean()),
        "std_bfactor": float(bfactors_arr.std()),
    }


# ── Langevin dynamics ─────────────────────────────────────────────────────


def run_langevin(
    model: TotalEnergy,
    R_init: torch.Tensor,
    seq: torch.Tensor,
    lengths: torch.Tensor,
    beta: float = 10.0,
    n_steps: int = 2000,
    step_size: float = 1e-4,
    force_cap: float = 50.0,
    save_every: int = 10,
) -> torch.Tensor:
    """Run overdamped Langevin dynamics, return trajectory snapshots.

    R_{t+1} = R_t - η ∇E(R_t) + √(2η/β) ξ

    Returns:
        trajectory: (n_snapshots, L_real, 3) coordinate snapshots.
    """
    L = int(lengths[0].item())
    R = R_init[:, :L].clone().detach()
    seq_real = seq[:, :L]
    lengths_real = lengths.clone()
    noise_scale = math.sqrt(2.0 * step_size / beta)

    snapshots = []

    for step in range(n_steps):
        R_grad = R.detach().clone().requires_grad_(True)
        E = model(R_grad, seq_real, lengths=lengths_real)
        E.backward()

        forces = -R_grad.grad.detach()

        # Clip forces
        force_norms = forces.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        clip_mask = force_norms > force_cap
        if clip_mask.any():
            forces = torch.where(clip_mask, forces * force_cap / force_norms, forces)

        noise = noise_scale * torch.randn_like(R)
        R = R + step_size * forces + noise

        if (step + 1) % save_every == 0:
            snapshots.append(R[0].detach().clone())

    if not snapshots:
        snapshots.append(R[0].detach().clone())

    return torch.stack(snapshots, dim=0)  # (n_snap, L, 3)


# ── RMSF and B-factor computation ─────────────────────────────────────────


def compute_rmsf(trajectory: torch.Tensor) -> np.ndarray:
    """Per-residue RMSF from trajectory. Returns (L,) in Å."""
    mean_coords = trajectory.mean(dim=0)
    displacements = trajectory - mean_coords.unsqueeze(0)
    msd = (displacements**2).sum(dim=-1).mean(dim=0)
    return msd.sqrt().numpy()


def rmsf_to_bfactor(rmsf: np.ndarray) -> np.ndarray:
    """Convert RMSF (Å) to B-factor (Å²): B = 8π²⟨u²⟩."""
    return 8.0 * np.pi**2 * rmsf**2


# ── Correlation ───────────────────────────────────────────────────────────


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / den) if den > 1e-10 else 0.0


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        if len(x) < 3:
            return 0.0
        r, _ = spearmanr(x, y)
        return float(r) if np.isfinite(r) else 0.0
    except ImportError:
        return float("nan")


# ── Per-structure analysis ────────────────────────────────────────────────


def load_structure_for_model(
    pdb_id: str,
    cache_dir: str = "./pdb_cache",
    device: str = "cpu",
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Load a PDB structure as model-ready tensors.

    Returns (R, seq, lengths, L) or None.
    """
    cache_path = Path(cache_dir)
    cif_path = download_cif(pdb_id, cache_dir=str(cache_path))
    if cif_path is None:
        return None

    chains = parse_cif_ca_chains(cif_path)
    if not chains:
        return None

    # Take first chain, split on gaps
    chain_id, (coords_np, seq_np) = list(chains.items())[0]
    segments = split_chain_on_gaps(coords_np, seq_np, max_gap=4.5)
    if not segments:
        return None

    # Take longest segment
    best = max(segments, key=lambda s: len(s[0]))
    coords_np, seq_np = best

    L = len(coords_np)
    if L < 20:
        return None

    R = torch.tensor(coords_np, dtype=torch.float32, device=device).unsqueeze(0)
    seq = torch.tensor(seq_np, dtype=torch.long, device=device).unsqueeze(0)
    lengths = torch.tensor([L], dtype=torch.long, device=device)

    return R, seq, lengths, L


def analyze_one_structure(
    model: TotalEnergy,
    pdb_id: str,
    pdb_data: Dict,
    betas: List[float],
    n_steps: int = 2000,
    step_size: float = 1e-4,
    save_every: int = 10,
    cache_dir: str = "./pdb_cache",
    device: str = "cpu",
) -> Dict:
    """Run Langevin at multiple β, compute RMSF, compare to B-factors."""

    L = pdb_data["n_residues"]
    exp_bfactors = np.array(pdb_data["bfactors"])

    logger.info(
        "  %s: L=%d, B=[%.1f, %.1f], mean=%.1f", pdb_id, L, exp_bfactors.min(), exp_bfactors.max(), exp_bfactors.mean()
    )

    loaded = load_structure_for_model(pdb_id, cache_dir=cache_dir, device=device)
    if loaded is None:
        return {"pdb_id": pdb_id, "error": "load_failed"}

    R, seq, lengths, L_model = loaded

    # Align lengths (B-factor source and model may differ slightly)
    min_L = min(L_model, L)
    R = R[:, :min_L]
    seq = seq[:, :min_L]
    lengths = torch.tensor([min_L], dtype=torch.long, device=device)
    exp_bfactors = exp_bfactors[:min_L]
    L = min_L

    results_per_beta = {}

    for beta in betas:
        logger.info("    β=%.0f: %d Langevin steps...", beta, n_steps)

        try:
            traj = run_langevin(
                model,
                R,
                seq,
                lengths,
                beta=beta,
                n_steps=n_steps,
                step_size=step_size,
                save_every=save_every,
            )

            rmsf = compute_rmsf(traj)
            sim_B = rmsf_to_bfactor(rmsf)

            # RMSD from native
            native = R[0, :L].detach().numpy()
            final = traj[-1].numpy()
            rmsd = float(np.sqrt(((final - native) ** 2).sum(-1).mean()))

            # Correlations
            r_pearson = pearson_corr(sim_B, exp_bfactors)
            r_spearman = spearman_corr(sim_B, exp_bfactors)
            r_log = pearson_corr(np.log(sim_B + 1.0), np.log(exp_bfactors + 1.0))

            mean_sim = float(sim_B.mean())
            mean_exp = float(exp_bfactors.mean())
            scale = mean_exp / max(mean_sim, 1e-6)

            results_per_beta[beta] = {
                "rmsd": round(rmsd, 3),
                "pearson_r": round(r_pearson, 4),
                "pearson_log_r": round(r_log, 4),
                "spearman_r": round(r_spearman, 4) if np.isfinite(r_spearman) else None,
                "mean_sim_B": round(mean_sim, 2),
                "mean_exp_B": round(mean_exp, 2),
                "scale_factor": round(scale, 2),
                "rmsf_mean": round(float(rmsf.mean()), 4),
                "rmsf_max": round(float(rmsf.max()), 4),
                "n_snapshots": int(traj.shape[0]),
                "sim_bfactors": [round(float(b), 2) for b in sim_B],
            }

            logger.info(
                "      RMSD=%.2fÅ  r=%.3f  rho=%s  RMSF=%.3fÅ  " "B_sim=%.1f  B_exp=%.1f  scale=%.1f",
                rmsd,
                r_pearson,
                f"{r_spearman:.3f}" if np.isfinite(r_spearman) else "N/A",
                rmsf.mean(),
                mean_sim,
                mean_exp,
                scale,
            )

        except Exception as e:
            logger.warning("    β=%.0f failed: %s", beta, e)
            results_per_beta[beta] = {"error": str(e)}

    # Best β
    best_beta, best_r = None, -1.0
    for beta, res in results_per_beta.items():
        r = res.get("pearson_r")
        if r is not None and r > best_r:
            best_r = r
            best_beta = beta

    return {
        "pdb_id": pdb_id,
        "chain": pdb_data["chain"],
        "n_residues": L,
        "exp_bfactors": [round(float(b), 2) for b in exp_bfactors],
        "mean_exp_B": round(float(exp_bfactors.mean()), 2),
        "results": {str(int(b)): v for b, v in results_per_beta.items()},
        "best_beta": best_beta,
        "best_pearson_r": round(best_r, 4),
    }


# ── Full analysis run ─────────────────────────────────────────────────────


def run_bfactor_analysis(
    model: TotalEnergy,
    pdb_ids: List[str],
    betas: List[float],
    n_steps: int = 2000,
    step_size: float = 1e-4,
    save_every: int = 10,
    cache_dir: str = "./pdb_cache",
    output_path: str = "bfactor_analysis.json",
    device: str = "cpu",
) -> Dict:
    """Run full B-factor calibration analysis.

    Returns results dict and saves to JSON.
    """

    # Extract experimental B-factors
    logger.info("Extracting B-factors from %d structures...", len(pdb_ids))
    pdb_data_list = []
    for pdb_id in pdb_ids:
        data = extract_ca_bfactors(pdb_id, cache_dir=cache_dir)
        if data is not None:
            pdb_data_list.append(data)
            logger.info("  %s: %d residues, mean B=%.1f Å²", pdb_id, data["n_residues"], data["mean_bfactor"])
        else:
            logger.warning("  %s: skipped", pdb_id)

    if not pdb_data_list:
        logger.error("No structures loaded.")
        return {}

    # Run per-structure analysis
    logger.info("Running Langevin B-factor analysis (β=%s, %d steps)...", betas, n_steps)

    all_results = []
    for pdb_data in pdb_data_list:
        result = analyze_one_structure(
            model,
            pdb_data["pdb_id"],
            pdb_data,
            betas=betas,
            n_steps=n_steps,
            step_size=step_size,
            save_every=save_every,
            cache_dir=cache_dir,
            device=device,
        )
        all_results.append(result)

    # Aggregate per-β correlations
    beta_summary = {}
    for beta in betas:
        rs = []
        for result in all_results:
            if "results" in result:
                bk = str(int(beta))
                if bk in result["results"]:
                    r = result["results"][bk].get("pearson_r")
                    if r is not None:
                        rs.append(r)
        if rs:
            beta_summary[beta] = {
                "mean_r": round(float(np.mean(rs)), 4),
                "min_r": round(float(np.min(rs)), 4),
                "max_r": round(float(np.max(rs)), 4),
                "n": len(rs),
            }

    # Find best β
    best_beta, best_r = None, -1.0
    for beta, bs in beta_summary.items():
        if bs["mean_r"] > best_r:
            best_r = bs["mean_r"]
            best_beta = beta

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("B-FACTOR CALIBRATION SUMMARY")
    logger.info("=" * 70)
    logger.info("  %-8s  %-10s  %-10s  %-10s", "β", "mean_r", "min_r", "max_r")
    logger.info("  " + "-" * 42)
    for beta in sorted(beta_summary.keys()):
        bs = beta_summary[beta]
        logger.info("  β=%-5.0f  r=%-8.4f  [%-6.4f, %-6.4f]", beta, bs["mean_r"], bs["min_r"], bs["max_r"])

    if best_beta is not None:
        kT_300 = 0.592  # kcal/mol
        logger.info("")
        logger.info("  Best β = %.0f (mean Pearson r = %.4f)", best_beta, best_r)
        logger.info("  → 1 model energy unit ≈ %.1f × kT(300K) ≈ %.2f kcal/mol", best_beta, best_beta * kT_300)

    logger.info("")
    logger.info("  Per-structure best β:")
    for result in all_results:
        if "best_beta" in result and result["best_beta"] is not None:
            logger.info("    %s: β*=%.0f (r=%.4f)", result["pdb_id"], result["best_beta"], result["best_pearson_r"])
    logger.info("=" * 70)

    # Save
    output = {
        "betas": betas,
        "langevin_steps": n_steps,
        "step_size": step_size,
        "beta_summary": {str(int(k)): v for k, v in beta_summary.items()},
        "best_beta": best_beta,
        "best_mean_pearson_r": best_r,
        "structures": all_results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", output_path)

    return output
