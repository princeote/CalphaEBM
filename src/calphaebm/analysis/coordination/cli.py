# src/calphaebm/analysis/coordination/cli.py
"""
CLI for coordination analysis: ``calphaebm analyze coordination``

Computes:
  1. Per-residue scalar coordination n_i         (per-AA statistics: mean, std, p5, p95)
  2. Per-residue group-conditional coordination  n_i^(k)  (20 AA × 5 groups)
  3. Per-chain scalar contact density ρ          (global stats + ρ*(L) curve fit)
  4. Per-chain group-conditional contact density ρ^(k)    (5 × ρ*(k)(L) curve fits)

All quantities use the same Best et al. (2013) Cα sigmoid (r_half=8.0, tau=0.2).

The 5-group physicochemical scheme:
  0  core_hydrophobic       PHE, ILE, LEU, MET, VAL
  1  amphipathic_hydrophobic ALA, PRO, TRP, TYR
  2  positive               HIS, LYS, ARG
  3  negative               ASP, GLU
  4  polar                  CYS, GLY, ASN, GLN, SER, THR

Relationships:
  n_i   = Σ_k  n_i^(k)          (scalar = sum of group components)
  ρ     = (1/L) Σ_i n_i         (chain-level mean)
  ρ^(k) = (1/L) Σ_i n_i^(k)    (chain-level group-k mean)
  ρ     = Σ_k  ρ^(k)            (scalar = sum of group densities)

Output JSON keys consumed by PackingEnergy v6:
  Scalar (backward-compatible):
    n_mean_list, n_std_list, n_lo_list, n_hi_list
    rho_fit
  Group-conditional (new):
    n_group_mean_list   [20][5]   n*(k) center for each (AA, group)
    n_group_std_list    [20][5]   sigma(k) width
    n_group_lo_list     [20][5]   p5 band lower bound
    n_group_hi_list     [20][5]   p5 band upper bound
    rho_group_fits      [5]       per-group ρ*(k)(L) curve fits
    rho_group_sigma     [5]       global σ per group density
    rho_group_lo        [5][bins] p5 per group (length-resolved)
    rho_group_hi        [5][bins] p95 per group (length-resolved)

Example::

    calphaebm analyze coordination \\
        --pdb-list train_hq.txt \\
        --cache-dir pdb_cache \\
        --output-dir analysis/coordination_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from calphaebm.data.pdb_parse import download_cif, parse_cif_ca_chains, split_chain_on_gaps
from calphaebm.utils.logging import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# AA order (matches CalphaEBM integer encoding in ChainCA.seq)
# ---------------------------------------------------------------------------

AA_ORDER = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]

AA3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

NUM_AA = 20

# ---------------------------------------------------------------------------
# 5-group physicochemical scheme for grouped coordination
# ---------------------------------------------------------------------------
# Grouping is based on burial propensity (rASA) from PDB statistics and
# physicochemical character, used to compute the group-conditional
# coordination vector n_i^(k) for the extended E_hp packing term.
#
# Group 0 — Core hydrophobic:      strongly buried, purely aliphatic/aromatic
# Group 1 — Amphipathic hydrophobic: smaller, interface-tolerant, or constrained
# Group 2 — Positive:               basic residues
# Group 3 — Negative:               acidic residues
# Group 4 — Polar/small:            surface-preferring, H-bond donors/acceptors
#
# AA index order matches AA_ORDER above.

GROUP_NAMES = [
    "core_hydrophobic",  # 0
    "amphipathic_hydrophobic",  # 1
    "positive",  # 2
    "negative",  # 3
    "polar",  # 4
]
NUM_GROUPS = 5

# Maps AA index (0-19, AA_ORDER) -> group index (0-4)
# ALA(0)->1  CYS(1)->4  ASP(2)->3  GLU(3)->3  PHE(4)->0
# GLY(5)->4  HIS(6)->2  ILE(7)->0  LYS(8)->2  LEU(9)->0
# MET(10)->0 ASN(11)->4 PRO(12)->1 GLN(13)->4 ARG(14)->2
# SER(15)->4 THR(16)->4 VAL(17)->0 TRP(18)->1 TYR(19)->1
AA_GROUP_ASSIGNMENT: list[int] = [
    1,  # ALA  — amphipathic (small, surface-tolerant)
    4,  # CYS  — polar       (often buried but via SH chemistry)
    3,  # ASP  — negative
    3,  # GLU  — negative
    0,  # PHE  — core hydrophobic (aromatic, no polar atoms)
    4,  # GLY  — polar/small
    2,  # HIS  — positive    (charged at physiological pH)
    0,  # ILE  — core hydrophobic
    2,  # LYS  — positive
    0,  # LEU  — core hydrophobic
    0,  # MET  — core hydrophobic
    4,  # ASN  — polar
    1,  # PRO  — amphipathic (conformationally constrained, surface)
    4,  # GLN  — polar
    2,  # ARG  — positive
    4,  # SER  — polar
    4,  # THR  — polar
    0,  # VAL  — core hydrophobic
    1,  # TRP  — amphipathic (aromatic + NH, interface-tolerant)
    1,  # TYR  — amphipathic (aromatic + OH, interface-tolerant)
]

# Membership lookup: group index -> list of AA 3-letter codes (for display)
GROUP_MEMBERS: dict[int, list[str]] = {
    0: ["PHE", "ILE", "LEU", "MET", "VAL"],
    1: ["ALA", "PRO", "TRP", "TYR"],
    2: ["HIS", "LYS", "ARG"],
    3: ["ASP", "GLU"],
    4: ["CYS", "GLY", "ASN", "GLN", "SER", "THR"],
}

# ---------------------------------------------------------------------------
# Best et al. (2013) Cα-adapted sigmoid parameters
# ---------------------------------------------------------------------------
# Original all-atom: cutoff=4.5Å, β=5.0 Å⁻¹, λ=1.8
# Cα adaptation: cutoff=8.0Å, β=5.0 Å⁻¹, λ=1.8
# Sigmoid form: σ(d) = 1 / (1 + exp(β * (d - r_cut)))
#   equivalent to: 1 / (1 + exp((d - r_cut) / τ))  with τ = 1/β = 0.2
BEST_R_HALF = 8.0  # Å — Cα contact midpoint
BEST_TAU = 0.2  # Å — steepness (= 1/β, β=5.0 Å⁻¹)
BEST_EXCLUDE = 3  # |i-j| > 3
BEST_MAX_DIST = 12.0  # Å — hard cutoff for efficiency


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_soft_coordination(
    coords: np.ndarray,
    r_half: float = 7.0,
    tau: float = 1.0,
    exclude: int = 3,
    max_dist: float = 10.0,
) -> np.ndarray:
    """Compute soft coordination number for each residue.

    Uses ``sigmoid((r_half - r_ij) / tau)`` summed over nonbonded neighbors.

    Args:
        coords: (L, 3) Cα coordinates.
        r_half: sigmoid midpoint distance (Å).
        tau: sigmoid steepness parameter.
        exclude: minimum sequence separation ``|i-j| > exclude``.
        max_dist: distance cutoff (Å).

    Returns:
        n_soft: (L,) soft coordination number per residue.
    """
    L = len(coords)

    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))

    # Sequence separation mask: |i-j| > exclude, i != j
    idx = np.arange(L)
    seq_sep = np.abs(idx[:, None] - idx[None, :])
    allowed = seq_sep > exclude
    np.fill_diagonal(allowed, False)

    # Distance cutoff
    allowed = allowed & (dist <= max_dist)

    # Soft sigmoid counting: sigmoid((r_half - r) / tau)
    r_clipped = np.clip(dist, 0, max_dist)
    with np.errstate(over="ignore"):
        g = 1.0 / (1.0 + np.exp((r_clipped - r_half) / tau))
    g = g * allowed.astype(np.float64)

    return g.sum(axis=1)


def compute_contact_density(n_soft: np.ndarray) -> float:
    """Compute per-residue contact density ρ = mean(n_i).

    Since n_i and ρ use the same sigmoid, ρ = (1/L) * Σ n_i.
    """
    return float(n_soft.mean())


def compute_grouped_coordination(
    coords: np.ndarray,
    seq: np.ndarray,
    group_assignment: list[int],
    r_half: float = BEST_R_HALF,
    tau: float = BEST_TAU,
    exclude: int = BEST_EXCLUDE,
    max_dist: float = BEST_MAX_DIST,
    n_groups: int = NUM_GROUPS,
) -> np.ndarray:
    """Compute group-conditional coordination vectors n_i^(k).

    Reuses the same sigmoid and distance matrix as compute_soft_coordination.
    For each residue i and group k:

        n_i^(k) = Σ_{j: |i-j|>exclude, group(s_j)=k} sigmoid((r_half - r_ij) / tau)

    This allows the packing energy to distinguish neighbor *identity*
    (which physicochemical group is doing the burying) rather than only
    neighbor *count*.

    Args:
        coords:           (L, 3) Cα coordinates.
        seq:              (L,) integer AA indices (0-19, matching AA_ORDER).
        group_assignment: list of length 20 mapping AA index -> group index.
        r_half:           sigmoid midpoint distance (Å).
        tau:              sigmoid steepness parameter (Å).
        exclude:          minimum sequence separation |i-j| > exclude.
        max_dist:         distance cutoff (Å).
        n_groups:         number of groups (5).

    Returns:
        n_grouped: (L, n_groups) float64 array.
                   n_grouped[i, k] = group-k coordination of residue i.
                   Note: sum over k gives n_soft[i] (the scalar coordination).
    """
    L = len(seq)

    # Full pairwise distances — same computation as compute_soft_coordination
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))  # (L, L)

    idx = np.arange(L)
    seq_sep = np.abs(idx[:, None] - idx[None, :])
    allowed = seq_sep > exclude
    np.fill_diagonal(allowed, False)
    allowed = allowed & (dist <= max_dist)

    r_clipped = np.clip(dist, 0.0, max_dist)
    with np.errstate(over="ignore"):
        g = 1.0 / (1.0 + np.exp((r_clipped - r_half) / tau))  # (L, L)
    g = g * allowed.astype(np.float64)

    # Group index for each residue j: (L,) integer array
    group_ids = np.array(
        [
            group_assignment[int(seq[j])] if 0 <= int(seq[j]) < len(group_assignment) else (n_groups - 1)
            for j in range(L)
        ],
        dtype=np.int32,
    )

    # n_i^(k) = sum_j [ g(r_ij) * (group_j == k) ]
    # Vectorised: build a (L, n_groups) boolean membership matrix, then
    # contract with g over the j axis.
    membership = group_ids[None, :] == np.arange(n_groups)[:, None, None]
    # membership: (n_groups, 1, L)  — broadcasts against g (L, L)
    # Reshape for efficient einsum: group_mask (L, n_groups)
    group_mask = group_ids[:, None] == np.arange(n_groups)[None, :]  # (L, n_groups)

    # n_grouped[i, k] = Σ_j g[i,j] * group_mask[j, k]
    n_grouped = g @ group_mask.astype(np.float64)  # (L, L) @ (L, n_groups) -> (L, n_groups)

    return n_grouped


# ---------------------------------------------------------------------------
# ρ*(L) curve fitting
# ---------------------------------------------------------------------------


def fit_rho_curve(
    lengths: np.ndarray,
    rhos: np.ndarray,
    bins: list[tuple[int, int]] | None = None,
) -> dict:
    """Fit ρ*(L) = a - b * exp(-L/c) to binned data.

    Args:
        lengths: (N,) chain lengths.
        rhos: (N,) contact densities.
        bins: length bin edges [(lo, hi), ...].

    Returns:
        dict with fit parameters and bin statistics.
    """
    from scipy import stats as sp_stats
    from scipy.optimize import curve_fit

    if bins is None:
        bins = [
            (5, 20),
            (20, 40),
            (40, 60),
            (60, 80),
            (80, 100),
            (100, 150),
            (150, 200),
            (200, 300),
            (300, 500),
            (500, 1000),
        ]

    # Bin statistics
    bin_stats = []
    bin_L = []
    bin_rho = []
    bin_weight = []

    for lo, hi in bins:
        mask = (lengths >= lo) & (lengths < hi)
        n = mask.sum()
        if n < 3:
            continue
        r = rhos[mask]
        mid = (lo + hi) / 2.0
        bin_stats.append(
            {
                "range": f"{lo}-{hi}",
                "n": int(n),
                "mean": round(float(r.mean()), 4),
                "std": round(float(r.std()), 4),
                "p5": round(float(np.percentile(r, 5)), 4),
                "p25": round(float(np.percentile(r, 25)), 4),
                "p75": round(float(np.percentile(r, 75)), 4),
                "p95": round(float(np.percentile(r, 95)), 4),
            }
        )
        bin_L.append(mid)
        bin_rho.append(float(r.mean()))
        bin_weight.append(n)

    bin_L = np.array(bin_L)
    bin_rho = np.array(bin_rho)
    bin_weight = np.array(bin_weight, dtype=float)

    # Fit ρ*(L) = a - b * exp(-L/c)
    def model(L, a, b, c):
        return a - b * np.exp(-L / c)

    try:
        popt, pcov = curve_fit(
            model,
            bin_L,
            bin_rho,
            p0=[2.7, 1.5, 80],
            sigma=1.0 / np.sqrt(bin_weight),
        )
        a, b, c = popt
        fit_ok = True
    except Exception as e:
        logger.warning("ρ*(L) curve fit failed: %s", e)
        a, b, c = 2.5, 1.0, 100.0
        fit_ok = False

    # Correlations
    r_pearson, p_pearson = sp_stats.pearsonr(lengths, rhos)
    r_spearman, p_spearman = sp_stats.spearmanr(lengths, rhos)

    return {
        "fit_a": round(float(a), 4),
        "fit_b": round(float(b), 4),
        "fit_c": round(float(c), 1),
        "fit_ok": fit_ok,
        "fit_formula": f"rho*(L) = {a:.3f} - {b:.3f} * exp(-L / {c:.1f})",
        "global_mean": round(float(rhos.mean()), 4),
        "global_std": round(float(rhos.std()), 4),
        "global_p5": round(float(np.percentile(rhos, 5)), 4),
        "global_p25": round(float(np.percentile(rhos, 25)), 4),
        "global_p75": round(float(np.percentile(rhos, 75)), 4),
        "global_p95": round(float(np.percentile(rhos, 95)), 4),
        "pearson_rho_vs_L": round(float(r_pearson), 4),
        "spearman_rho_vs_L": round(float(r_spearman), 4),
        "bins": bin_stats,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_distributions(coord_by_aa, n_mean_list, n_std_list, out_dir):
    """Plot per-AA coordination distributions with Gaussian overlay + KS test."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats as sp_stats
    except ImportError:
        logger.warning("matplotlib or scipy not available — skipping plots")
        return

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle(
        "Per-AA Coordination Number Distributions\n"
        "Histogram (PDB data) vs Gaussian fit (used in _Hydrophobic energy)",
        fontsize=14,
        fontweight="bold",
    )

    for aa_idx, aa in enumerate(AA_ORDER):
        ax = axes[aa_idx // 5, aa_idx % 5]
        vals = coord_by_aa.get(aa_idx, [])
        if len(vals) < 10:
            ax.set_title(f"{aa} ({AA3_TO_1[aa]}) — no data", fontsize=10)
            continue

        arr = np.array(vals)
        mu = n_mean_list[aa_idx]
        sigma = n_std_list[aa_idx]

        bins = np.linspace(0, max(arr.max(), 10), 50)
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.6,
            color="steelblue",
            edgecolor="white",
            linewidth=0.5,
            label=f"PDB (n={len(vals):,})",
        )

        x = np.linspace(0, max(arr.max(), 10), 200)
        gaussian = np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x, gaussian, "r-", linewidth=2, label=f"Gaussian(μ={mu:.1f}, σ={sigma:.2f})")

        ks_stat, ks_pval = sp_stats.kstest(arr, "norm", args=(mu, sigma))

        ax.axvline(mu, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(mu - sigma, color="red", linestyle=":", alpha=0.4, linewidth=1)
        ax.axvline(mu + sigma, color="red", linestyle=":", alpha=0.4, linewidth=1)

        aa1 = AA3_TO_1[aa]
        ax.set_title(f"{aa} ({aa1})  KS={ks_stat:.3f}, p={ks_pval:.2e}", fontsize=9)
        ax.set_xlabel("n (soft coordination)", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = Path(out_dir) / "coordination_distributions.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved distribution plot to %s", plot_path)

    # ── Summary figure: energy wells ──────────────────────────────────────
    fig2, axes2 = plt.subplots(4, 5, figsize=(20, 16))
    fig2.suptitle(
        "Coordination-Based Packing Energy Wells\n"
        "E_hp(n_i) = -w · exp(-(n_i - n*)² / 2σ²)  — energy minimum at native coordination",
        fontsize=14,
        fontweight="bold",
    )

    kd_raw = [
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
    kd_min, kd_max = min(kd_raw), max(kd_raw)
    kd_norm = [0.1 + 0.9 * (v - kd_min) / (kd_max - kd_min) for v in kd_raw]
    w_init = kd_norm

    for aa_idx, aa in enumerate(AA_ORDER):
        ax = axes2[aa_idx // 5, aa_idx % 5]
        mu = n_mean_list[aa_idx]
        sigma = n_std_list[aa_idx]
        w = w_init[aa_idx]

        x = np.linspace(0, 10, 200)
        gaussian = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        E_well = -w * gaussian

        ax.plot(x, E_well, "b-", linewidth=2)
        ax.fill_between(x, E_well, 0, alpha=0.15, color="blue")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(mu, color="red", linestyle="--", alpha=0.7, linewidth=1, label=f"n*={mu:.1f}")

        aa1 = AA3_TO_1[aa]
        ax.set_title(f"{aa} ({aa1})  w={w:.2f}, σ={sigma:.2f}", fontsize=9)
        ax.set_xlabel("n (coordination)", fontsize=8)
        ax.set_ylabel("E_hp (energy)", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.1 * w, 0.15 * w)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path2 = Path(out_dir) / "coordination_energy_wells.png"
    fig2.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Saved energy well plot to %s", plot_path2)


def _plot_rho(chain_data, rho_fit, out_dir):
    """Plot ρ vs L scatter with fitted curve and histograms."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping ρ plots")
        return

    lengths = np.array([d["L"] for d in chain_data])
    rhos = np.array([d["rho"] for d in chain_data])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Scatter: ρ vs L with fitted curve
    ax = axes[0]
    ax.scatter(lengths, rhos, s=3, alpha=0.3, color="steelblue", rasterized=True)

    a = rho_fit["fit_a"]
    b = rho_fit["fit_b"]
    c = rho_fit["fit_c"]
    L_fit = np.linspace(5, max(lengths) * 1.05, 200)
    rho_curve = a - b * np.exp(-L_fit / c)
    ax.plot(L_fit, rho_curve, "r-", linewidth=2, label=f"ρ*(L) = {a:.2f} - {b:.2f}·exp(-L/{c:.0f})")

    # p5/p95 by bin
    for bs in rho_fit["bins"]:
        rng = bs["range"]
        lo, hi = rng.split("-")
        mid = (int(lo) + int(hi)) / 2
        ax.errorbar(
            mid,
            bs["mean"],
            yerr=[[bs["mean"] - bs["p5"]], [bs["p95"] - bs["mean"]]],
            fmt="ko",
            markersize=4,
            capsize=3,
            linewidth=1,
        )

    ax.set_xlabel("Chain length L", fontsize=11)
    ax.set_ylabel("Contact density ρ (per residue)", fontsize=11)
    ax.set_title(f"ρ vs L  (Pearson r={rho_fit['pearson_rho_vs_L']:.3f})", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(lengths) * 1.05)

    # 2. Global ρ histogram
    ax = axes[1]
    ax.hist(rhos, bins=60, density=True, alpha=0.7, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(
        rho_fit["global_mean"], color="red", linestyle="--", linewidth=1.5, label=f"mean={rho_fit['global_mean']:.3f}"
    )
    ax.axvline(rho_fit["global_p5"], color="orange", linestyle=":", linewidth=1, label=f"p5={rho_fit['global_p5']:.3f}")
    ax.axvline(
        rho_fit["global_p95"], color="orange", linestyle=":", linewidth=1, label=f"p95={rho_fit['global_p95']:.3f}"
    )
    ax.set_xlabel("Contact density ρ", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Global ρ distribution", fontsize=12)
    ax.legend(fontsize=9)

    # 3. ρ by length bin (box plot style)
    ax = axes[2]
    bin_labels = []
    bin_means = []
    bin_p5 = []
    bin_p95 = []
    for bs in rho_fit["bins"]:
        bin_labels.append(bs["range"])
        bin_means.append(bs["mean"])
        bin_p5.append(bs["p5"])
        bin_p95.append(bs["p95"])

    x_pos = np.arange(len(bin_labels))
    ax.bar(x_pos, bin_means, color="steelblue", alpha=0.7, edgecolor="white")
    ax.errorbar(
        x_pos,
        bin_means,
        yerr=[np.array(bin_means) - np.array(bin_p5), np.array(bin_p95) - np.array(bin_means)],
        fmt="none",
        capsize=4,
        color="black",
        linewidth=1,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Length bin", fontsize=11)
    ax.set_ylabel("ρ (mean ± p5/p95)", fontsize=11)
    ax.set_title("ρ by chain length", fontsize=12)

    plt.tight_layout()
    plot_path = Path(out_dir) / "contact_density_rho.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ρ plot to %s", plot_path)


def _plot_group_distributions(
    grouped_by_aa: dict,
    n_group_mean_list: list,
    n_group_std_list: list,
    out_dir,
) -> None:
    """Plot per-group n_i^(k) distributions for all 20 AAs.

    Produces 5 figures — one per physicochemical group — each with a 4×5
    grid of AA subplots.  Each subplot shows:
      - Histogram of n_i^(k) values from PDB data
      - Gaussian overlay N(n*^(k), σ^(k))
      - KS test statistic
      - p5/p95 vertical lines (actual band)

    Output files:
      coordination_group_0_core_hp.png
      coordination_group_1_amphip_hp.png
      coordination_group_2_positive.png
      coordination_group_3_negative.png
      coordination_group_4_polar.png
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats as sp_stats
    except ImportError:
        logger.warning("matplotlib/scipy not available — skipping group plots")
        return

    group_labels = [
        "core_hp",
        "amphip_hp",
        "positive",
        "negative",
        "polar",
    ]
    group_titles = [
        "Group 0 — Core hydrophobic  [PHE, ILE, LEU, MET, VAL]",
        "Group 1 — Amphipathic hydrophobic  [ALA, PRO, TRP, TYR]",
        "Group 2 — Positive  [HIS, LYS, ARG]",
        "Group 3 — Negative  [ASP, GLU]",
        "Group 4 — Polar  [CYS, GLY, ASN, GLN, SER, THR]",
    ]
    group_colors = ["#2166ac", "#d6604d", "#4dac26", "#7b3294", "#e08214"]

    out_dir = Path(out_dir)

    for k in range(NUM_GROUPS):
        fig, axes = plt.subplots(4, 5, figsize=(22, 18))
        fig.suptitle(
            f"n_i^({k}) Distributions — {group_titles[k]}\n"
            f"Histogram (PDB) vs Gaussian fit (used in E_hp product Gaussian)",
            fontsize=13,
            fontweight="bold",
        )

        for aa_idx, aa in enumerate(AA_ORDER):
            ax = axes[aa_idx // 5, aa_idx % 5]
            vals = grouped_by_aa.get(aa_idx, [[] for _ in range(NUM_GROUPS)])[k]
            aa1 = AA3_TO_1.get(aa, "?")

            if len(vals) < 10:
                ax.set_title(f"{aa} ({aa1}) — no data", fontsize=9)
                ax.set_visible(False)
                continue

            arr = np.array(vals)
            mu = n_group_mean_list[aa_idx][k]
            sigma = n_group_std_list[aa_idx][k]
            sigma_plot = max(sigma, 0.1)  # for display only

            # histogram
            x_max = max(arr.max() * 1.05, mu + 3 * sigma_plot, 3.0)
            bins = np.linspace(0, x_max, 50)
            ax.hist(
                arr,
                bins=bins,
                density=True,
                alpha=0.55,
                color=group_colors[k],
                edgecolor="white",
                linewidth=0.4,
                label=f"PDB (n={len(vals):,})",
            )

            # Gaussian overlay
            x = np.linspace(0, x_max, 300)
            gauss = np.exp(-((x - mu) ** 2) / (2 * sigma_plot**2))
            gauss /= sigma_plot * np.sqrt(2 * np.pi)
            ax.plot(x, gauss, color="black", linewidth=1.8, label=f"N(μ={mu:.2f}, σ={sigma:.2f})")

            # mean and ±1σ
            ax.axvline(mu, color="black", linestyle="--", alpha=0.8, linewidth=1.0)
            ax.axvline(mu - sigma_plot, color="black", linestyle=":", alpha=0.4, linewidth=0.8)
            ax.axvline(mu + sigma_plot, color="black", linestyle=":", alpha=0.4, linewidth=0.8)

            # p5/p95 from data
            p5 = np.percentile(arr, 5)
            p95 = np.percentile(arr, 95)
            ax.axvline(p5, color="orange", linestyle="-.", alpha=0.7, linewidth=0.8)
            ax.axvline(p95, color="orange", linestyle="-.", alpha=0.7, linewidth=0.8)

            # KS test
            ks_stat, ks_p = sp_stats.kstest(arr, "norm", args=(mu, sigma_plot))

            # clamp warning
            clamp_warn = " ⚠ σ<0.8" if sigma < 0.8 else ""
            ax.set_title(
                f"{aa} ({aa1})  KS={ks_stat:.3f} p={ks_p:.1e}{clamp_warn}",
                fontsize=8,
                color="red" if sigma < 0.8 else "black",
            )
            ax.set_xlabel(f"n_i^({k})", fontsize=7)
            ax.set_ylabel("density", fontsize=7)
            ax.set_xlim(0, x_max)
            ax.tick_params(labelsize=7)
            if aa_idx == 0:
                ax.legend(fontsize=6, loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fname = f"coordination_group_{k}_{group_labels[k]}.png"
        plot_path = out_dir / fname
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved group-%d distribution plot to %s", k, plot_path)


def _plot_rho_group(chain_data: list, rho_group_fits: list, out_dir) -> None:
    """Plot per-group ρ^(k) vs L scatter with fitted curves and histograms.

    Produces one figure with 5 rows (one per group) × 3 columns:
      Col 0: scatter ρ^(k) vs L + fitted curve ρ*(k)(L) with p5/p95 by bin
      Col 1: histogram of global ρ^(k) distribution
      Col 2: ρ^(k) mean ± p5/p95 by length bin (bar chart)

    Output: coordination_rho_groups.png
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping ρ group plots")
        return

    group_labels = [
        "core_hp  [PHE,ILE,LEU,MET,VAL]",
        "amphip_hp  [ALA,PRO,TRP,TYR]",
        "positive  [HIS,LYS,ARG]",
        "negative  [ASP,GLU]",
        "polar  [CYS,GLY,ASN,GLN,SER,THR]",
    ]
    group_colors = ["#2166ac", "#d6604d", "#4dac26", "#7b3294", "#e08214"]

    lengths = np.array([d["L"] for d in chain_data])
    rhos_g = np.array([d["rho_grouped"] for d in chain_data])  # (N, 5)

    out_dir = Path(out_dir)
    fig, axes = plt.subplots(NUM_GROUPS, 3, figsize=(18, 5 * NUM_GROUPS))
    fig.suptitle(
        "Per-group contact density  ρ^(k) = (1/L) Σ_i n_i^(k)\n"
        "5-group scheme: scatter vs L, global histogram, by-length bar chart",
        fontsize=13,
        fontweight="bold",
    )

    for k in range(NUM_GROUPS):
        rhos_k = rhos_g[:, k]
        fit = rho_group_fits[k]
        a, b, c = fit["fit_a"], fit["fit_b"], fit["fit_c"]
        color = group_colors[k]
        label = group_labels[k]

        # ── Col 0: scatter + fitted curve ──────────────────────────────
        ax0 = axes[k, 0]
        ax0.scatter(lengths, rhos_k, s=3, alpha=0.25, color=color, rasterized=True)
        L_fit = np.linspace(5, max(lengths) * 1.05, 300)
        rho_fit = a - b * np.exp(-L_fit / c)
        ax0.plot(L_fit, rho_fit, "k-", linewidth=2, label=f"ρ*(L)={a:.3f}-{b:.3f}·e^(-L/{c:.0f})")

        # p5/p95 by length bin
        bins_def = [(40, 60), (60, 80), (80, 100), (100, 150), (150, 200), (200, 300), (300, 500)]
        for lo, hi in bins_def:
            mask = (lengths >= lo) & (lengths < hi)
            if mask.sum() < 3:
                continue
            mid = (lo + hi) / 2
            r = rhos_k[mask]
            ax0.errorbar(
                mid,
                r.mean(),
                yerr=[[r.mean() - np.percentile(r, 5)], [np.percentile(r, 95) - r.mean()]],
                fmt="o",
                color="black",
                markersize=3,
                capsize=3,
                linewidth=0.8,
                alpha=0.8,
            )

        ax0.set_xlabel("Chain length L", fontsize=9)
        ax0.set_ylabel(f"ρ^({k})", fontsize=9)
        ax0.set_title(f"k={k}: {label}  (r={fit['pearson_rho_vs_L']:.3f})", fontsize=9)
        ax0.legend(fontsize=7)
        ax0.set_xlim(0, max(lengths) * 1.05)

        # ── Col 1: global histogram ──────────────────────────────────
        ax1 = axes[k, 1]
        ax1.hist(rhos_k, bins=50, density=True, alpha=0.65, color=color, edgecolor="white", linewidth=0.3)
        m_k = rhos_k.mean()
        std_k = rhos_k.std()
        p5_k = np.percentile(rhos_k, 5)
        p95_k = np.percentile(rhos_k, 95)

        # Gaussian overlay on ρ distribution
        x = np.linspace(max(0, m_k - 4 * std_k), m_k + 4 * std_k, 200)
        gauss = np.exp(-((x - m_k) ** 2) / (2 * std_k**2)) / (std_k * np.sqrt(2 * np.pi))
        ax1.plot(x, gauss, "k-", linewidth=1.5, label=f"N(μ={m_k:.3f}, σ={std_k:.3f})")
        ax1.axvline(m_k, color="black", linestyle="--", linewidth=1.0, label=f"mean={m_k:.3f}")
        ax1.axvline(p5_k, color="orange", linestyle=":", linewidth=1.0, label=f"p5={p5_k:.3f}")
        ax1.axvline(p95_k, color="orange", linestyle=":", linewidth=1.0, label=f"p95={p95_k:.3f}")
        ax1.set_xlabel(f"ρ^({k})", fontsize=9)
        ax1.set_ylabel("density", fontsize=9)
        ax1.set_title(f"k={k}: global ρ^({k}) distribution", fontsize=9)
        ax1.legend(fontsize=7)

        # ── Col 2: by-length-bin bar chart ───────────────────────────
        ax2 = axes[k, 2]
        bin_labels_plot, bin_means_plot, bin_p5_plot, bin_p95_plot = [], [], [], []
        for bs in fit.get("bins", []):
            bin_labels_plot.append(bs["range"])
            bin_means_plot.append(bs["mean"])
            bin_p5_plot.append(bs["p5"])
            bin_p95_plot.append(bs["p95"])

        if bin_means_plot:
            x_pos = np.arange(len(bin_labels_plot))
            ax2.bar(x_pos, bin_means_plot, color=color, alpha=0.7, edgecolor="white")
            ax2.errorbar(
                x_pos,
                bin_means_plot,
                yerr=[
                    np.array(bin_means_plot) - np.array(bin_p5_plot),
                    np.array(bin_p95_plot) - np.array(bin_means_plot),
                ],
                fmt="none",
                capsize=3,
                color="black",
                linewidth=0.8,
            )
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(bin_labels_plot, rotation=45, ha="right", fontsize=7)

        ax2.set_xlabel("Length bin", fontsize=9)
        ax2.set_ylabel(f"ρ^({k}) mean ± p5/p95", fontsize=9)
        ax2.set_title(f"k={k}: ρ^({k}) by chain length", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = out_dir / "coordination_rho_groups.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ρ group plot to %s", plot_path)


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``coordination`` subcommand under ``calphaebm analyze``."""
    p = subparsers.add_parser(
        "coordination",
        help="Compute per-AA coordination stats (n_i) and contact density (ρ)",
        description=(
            "Compute soft coordination number n_i for every residue in the dataset "
            "and per-chain contact density ρ = mean(n_i), then output per-AA statistics "
            "(mean, std, p5, p95) and ρ*(L) curve fit. Uses Best et al. (2013) Cα-adapted "
            "sigmoid parameters by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pdb-list",
        type=str,
        required=True,
        help="File containing PDB IDs, one per line (e.g. train_hq.txt)",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="pdb_cache",
        help="Directory for mmCIF files (default: pdb_cache)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="analysis/coordination_analysis",
        help="Output directory for coord_n_star.json (default: analysis/coordination_analysis)",
    )
    # Sigmoid parameters — Best-style defaults
    p.add_argument(
        "--best-style",
        action="store_true",
        default=True,
        help="Use Best et al. (2013) Cα parameters: r_half=8.0, tau=0.2 (default: True)",
    )
    p.add_argument(
        "--no-best-style",
        action="store_false",
        dest="best_style",
        help="Use legacy parameters (r_half=7.0, tau=1.0) instead of Best-style",
    )
    p.add_argument(
        "--r-half",
        type=float,
        default=None,
        help="Sigmoid midpoint distance in Å (default: 8.0 best-style, 7.0 legacy)",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Sigmoid steepness τ in Å (default: 0.2 best-style, 1.0 legacy)",
    )
    p.add_argument(
        "--exclude",
        type=int,
        default=3,
        help="Minimum sequence separation |i-j| > exclude (default: 3)",
    )
    p.add_argument(
        "--max-dist",
        type=float,
        default=None,
        help="Distance cutoff in Å (default: 12.0 best-style, 10.0 legacy)",
    )
    p.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        default=[5, 95],
        metavar=("LO", "HI"),
        help="Lower and upper percentiles for constraint bounds (default: 5 95)",
    )
    p.add_argument(
        "--min-len",
        type=int,
        default=40,
        help="Minimum chain length (default: 40)",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Maximum chain length (default: 512)",
    )
    p.add_argument(
        "--max-pdbs",
        type=int,
        default=None,
        help="Maximum number of PDB IDs to process (default: all)",
    )
    p.add_argument(
        "--max-chains",
        type=int,
        default=None,
        help="Maximum number of chains to collect (default: no limit)",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating distribution plots",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity",
    )
    p.set_defaults(func=run)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Run coordination analysis."""

    # ── Resolve sigmoid parameters ────────────────────────────────────────
    if args.best_style:
        r_half = args.r_half if args.r_half is not None else BEST_R_HALF
        tau = args.tau if args.tau is not None else BEST_TAU
        max_dist = args.max_dist if args.max_dist is not None else BEST_MAX_DIST
        style_label = "Best et al. (2013) Cα-adapted"
    else:
        r_half = args.r_half if args.r_half is not None else 7.0
        tau = args.tau if args.tau is not None else 1.0
        max_dist = args.max_dist if args.max_dist is not None else 10.0
        style_label = "Legacy"

    logger.info("Sigmoid style: %s", style_label)
    logger.info(
        "  r_half=%.1f Å, tau=%.2f Å (β=%.1f Å⁻¹), exclude=%d, max_dist=%.1f Å",
        r_half,
        tau,
        1.0 / tau,
        args.exclude,
        max_dist,
    )

    pdb_list_path = Path(args.pdb_list)
    if not pdb_list_path.exists():
        logger.error("PDB list not found: %s", pdb_list_path)
        sys.exit(1)

    with open(pdb_list_path) as f:
        pdb_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if args.max_pdbs is not None:
        pdb_ids = pdb_ids[: args.max_pdbs]
    logger.info("Loaded %d PDB IDs from %s", len(pdb_ids), pdb_list_path)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect per-residue n_i and per-chain ρ ───────────────────────────
    coord_by_aa = defaultdict(list)  # aa_idx -> [n_soft, ...]
    grouped_by_aa = defaultdict(lambda: [[] for _ in range(NUM_GROUPS)])
    # grouped_by_aa[aa_idx][k] -> [n_i^(k), ...]  for each residue of type aa_idx
    chain_data = []  # list of {pdb_id, chain_id, L, rho, rg}
    n_structures = 0
    n_chains = 0
    n_residues = 0
    n_skipped = 0

    import requests

    session = requests.Session()

    for i, pdb_id in enumerate(pdb_ids):
        try:
            cif_path = download_cif(pdb_id, cache_dir=str(cache_dir), session=session)
        except Exception:
            n_skipped += 1
            if not args.quiet and n_skipped <= 5:
                logger.warning("Download failed for %s", pdb_id)
            continue

        if cif_path is None:
            n_skipped += 1
            continue

        try:
            raw_chains = parse_cif_ca_chains(str(cif_path), pdb_id)
        except Exception:
            n_skipped += 1
            if not args.quiet and n_skipped <= 5:
                logger.warning("Parse failed for %s", pdb_id)
            continue

        if not raw_chains:
            n_skipped += 1
            continue

        n_structures += 1

        for chain in raw_chains:
            fragments = split_chain_on_gaps(chain.coords, chain.seq, max_ca_jump=4.5)

            for frag_coords, frag_seq in fragments:
                L = len(frag_seq)
                if L < args.min_len or L > args.max_len:
                    continue

                coords = frag_coords.astype(np.float64)
                seq = frag_seq

                # Compute per-residue coordination (scalar)
                n_soft = compute_soft_coordination(
                    coords,
                    r_half=r_half,
                    tau=tau,
                    exclude=args.exclude,
                    max_dist=max_dist,
                )

                # Compute group-conditional coordination vectors (L, 5)
                n_grouped = compute_grouped_coordination(
                    coords,
                    seq,
                    group_assignment=AA_GROUP_ASSIGNMENT,
                    r_half=r_half,
                    tau=tau,
                    exclude=args.exclude,
                    max_dist=max_dist,
                )

                # Per-AA accumulation — scalar and grouped
                for j in range(L):
                    aa_idx = int(seq[j])
                    if 0 <= aa_idx < NUM_AA:
                        coord_by_aa[aa_idx].append(float(n_soft[j]))
                        for k in range(NUM_GROUPS):
                            grouped_by_aa[aa_idx][k].append(float(n_grouped[j, k]))

                # Per-chain ρ (scalar) and ρ^(k) (group vector)
                rho = compute_contact_density(n_soft)
                rho_grouped = n_grouped.sum(axis=0) / max(L, 1)  # (5,) group densities
                com = coords.mean(axis=0)
                rg = float(np.sqrt(((coords - com) ** 2).sum(axis=-1).mean()))

                chain_data.append(
                    {
                        "pdb_id": pdb_id,
                        "chain_id": getattr(chain, "chain_id", "A"),
                        "L": L,
                        "rho": rho,
                        "rho_grouped": rho_grouped.tolist(),  # [ρ^0 .. ρ^4]
                        "rg": rg,
                    }
                )

                n_chains += 1
                n_residues += L

                if args.max_chains is not None and n_chains >= args.max_chains:
                    break

            if args.max_chains is not None and n_chains >= args.max_chains:
                break

        if not args.quiet and (i + 1) % 500 == 0:
            logger.info(
                "  Processed %d/%d PDBs (%d chains, %d residues, %d skipped)",
                i + 1,
                len(pdb_ids),
                n_chains,
                n_residues,
                n_skipped,
            )

        if args.max_chains is not None and n_chains >= args.max_chains:
            logger.info("  Reached max_chains=%d, stopping", args.max_chains)
            break

    session.close()

    logger.info(
        "Done: %d PDBs, %d chains, %d residues, %d skipped",
        n_structures,
        n_chains,
        n_residues,
        n_skipped,
    )
    if n_chains == 0:
        logger.error("No valid chains found. Check --cache-dir and --pdb-list.")
        sys.exit(1)

    # ── Per-AA coordination statistics ────────────────────────────────────
    p_lo, p_hi = args.percentiles

    n_mean_list = []
    n_std_list = []
    n_lo_list = []
    n_hi_list = []
    n_mean_per_aa = {}
    n_std_per_aa = {}
    n_lo_per_aa = {}
    n_hi_per_aa = {}

    logger.info("")
    logger.info(
        "Per-AA coordination (r_half=%.1f, tau=%.2f, exclude=%d, max_dist=%.1f):",
        r_half,
        tau,
        args.exclude,
        max_dist,
    )
    logger.info(
        "  %4s %3s %8s %7s %7s %7s %7s",
        "AA",
        "1L",
        "count",
        "mean",
        "std",
        f"p{p_lo:.0f}",
        f"p{p_hi:.0f}",
    )
    logger.info("  " + "-" * 50)

    for aa_idx, aa in enumerate(AA_ORDER):
        vals = coord_by_aa.get(aa_idx, [])
        if len(vals) == 0:
            logger.warning("  No data for %s — using defaults", aa)
            n_mean_list.append(4.0)
            n_std_list.append(1.5)
            n_lo_list.append(1.0)
            n_hi_list.append(7.0)
            n_mean_per_aa[aa] = 4.0
            n_std_per_aa[aa] = 1.5
            n_lo_per_aa[aa] = 1.0
            n_hi_per_aa[aa] = 7.0
            continue

        arr = np.array(vals)
        mean_val = float(arr.mean())
        std_val = float(arr.std())
        lo_val = float(np.percentile(arr, p_lo))
        hi_val = float(np.percentile(arr, p_hi))

        n_mean_per_aa[aa] = round(mean_val, 3)
        n_std_per_aa[aa] = round(std_val, 3)
        n_lo_per_aa[aa] = round(lo_val, 3)
        n_hi_per_aa[aa] = round(hi_val, 3)
        n_mean_list.append(round(mean_val, 3))
        n_std_list.append(round(std_val, 3))
        n_lo_list.append(round(lo_val, 3))
        n_hi_list.append(round(hi_val, 3))

        aa1 = AA3_TO_1.get(aa, "?")
        logger.info(
            "  %4s (%s)  %7d  %6.2f  %6.2f  %6.2f  %6.2f",
            aa,
            aa1,
            len(vals),
            mean_val,
            std_val,
            lo_val,
            hi_val,
        )

    # ── Contact density ρ statistics + ρ*(L) fit ─────────────────────────
    lengths = np.array([d["L"] for d in chain_data])
    rhos = np.array([d["rho"] for d in chain_data])

    logger.info("")
    logger.info("=" * 60)
    logger.info("CONTACT DENSITY ρ = mean(n_i) STATISTICS")
    logger.info("=" * 60)
    logger.info("  N chains:   %d", len(rhos))
    logger.info("  L range:    %d - %d", lengths.min(), lengths.max())
    logger.info("  ρ mean:     %.3f", rhos.mean())
    logger.info("  ρ std:      %.3f", rhos.std())
    logger.info("  ρ p5:       %.3f", np.percentile(rhos, 5))
    logger.info("  ρ p95:      %.3f", np.percentile(rhos, 95))

    rho_fit = fit_rho_curve(lengths, rhos)

    logger.info("")
    logger.info("ρ*(L) fit: %s", rho_fit["fit_formula"])
    logger.info("  Pearson  ρ vs L: r=%.3f", rho_fit["pearson_rho_vs_L"])
    logger.info("  Spearman ρ vs L: r=%.3f", rho_fit["spearman_rho_vs_L"])

    logger.info("")
    logger.info("ρ BY LENGTH BIN:")
    logger.info("  %12s  %5s  %6s  %6s  %6s  %6s", "L range", "N", "mean", "std", "p5", "p95")
    logger.info("  " + "-" * 55)
    for bs in rho_fit["bins"]:
        logger.info(
            "  %12s  %5d  %6.3f  %6.3f  %6.3f  %6.3f", bs["range"], bs["n"], bs["mean"], bs["std"], bs["p5"], bs["p95"]
        )

    # ── Per-group contact density ρ^(k) statistics + fits ─────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PER-GROUP CONTACT DENSITY  ρ^(k) = (1/L) Σ_i n_i^(k)")
    logger.info("=" * 70)

    rho_group_fits: list[dict] = []
    rho_group_sigma: list[float] = []
    rho_group_global_mean: list[float] = []

    for k in range(NUM_GROUPS):
        rhos_k = np.array([d["rho_grouped"][k] for d in chain_data])
        sigma_k = float(rhos_k.std())
        mean_k = float(rhos_k.mean())
        rho_group_sigma.append(round(sigma_k, 4))
        rho_group_global_mean.append(round(mean_k, 4))

        fit_k = fit_rho_curve(lengths, rhos_k)
        rho_group_fits.append(fit_k)

        logger.info("")
        logger.info("  Group %d (%s):", k, GROUP_NAMES[k])
        logger.info("    members: %s", ", ".join(GROUP_MEMBERS[k]))
        logger.info(
            "    ρ^(%d) mean=%.3f  std=%.3f  p5=%.3f  p95=%.3f",
            k,
            mean_k,
            sigma_k,
            np.percentile(rhos_k, 5),
            np.percentile(rhos_k, 95),
        )
        logger.info("    ρ^(%d)*(L) fit: %s", k, fit_k["fit_formula"])
        logger.info("    Pearson  ρ^(%d) vs L: r=%.3f", k, fit_k["pearson_rho_vs_L"])

    # Verify partition: Σ_k ρ^(k) ≈ ρ
    rho_recon = np.array([sum(d["rho_grouped"]) for d in chain_data])
    max_err = float(np.abs(rho_recon - rhos).max())
    logger.info("")
    logger.info("Partition check: max |Σ_k ρ^(k) - ρ| = %.2e  (should be ~0)", max_err)

    # ── Per-AA grouped coordination statistics ────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PER-AA GROUP-CONDITIONAL COORDINATION  n_i^(k)  (5-group scheme)")
    logger.info("=" * 70)
    logger.info(
        "Groups: 0=core_hp [PHE,ILE,LEU,MET,VAL]  "
        "1=amphip_hp [ALA,PRO,TRP,TYR]  "
        "2=pos [HIS,LYS,ARG]  "
        "3=neg [ASP,GLU]  "
        "4=polar [CYS,GLY,ASN,GLN,SER,THR]"
    )
    logger.info("")
    logger.info(
        "  %4s (%s)  grp  %8s  %7s  %7s",
        "AA",
        "1L",
        "count",
        "mean",
        "std",
    )
    logger.info("  " + "-" * 50)

    # Storage: list of 20 lists of 5 floats
    n_group_mean_list: list[list[float]] = []
    n_group_std_list: list[list[float]] = []
    n_group_mean_per_aa: dict[str, list[float]] = {}
    n_group_std_per_aa: dict[str, list[float]] = {}

    for aa_idx, aa in enumerate(AA_ORDER):
        aa1 = AA3_TO_1.get(aa, "?")
        means = []
        stds = []
        for k in range(NUM_GROUPS):
            vals_k = grouped_by_aa[aa_idx][k]
            if len(vals_k) < 2:
                m, s = 0.0, 0.1
            else:
                arr_k = np.array(vals_k)
                m = float(arr_k.mean())
                s = float(arr_k.std())
            means.append(round(m, 4))
            stds.append(round(s, 4))
            logger.info(
                "  %4s (%s)  g=%d  %8d  %6.3f  %6.3f   [%s]",
                aa,
                aa1,
                k,
                len(grouped_by_aa[aa_idx][k]),
                m,
                s,
                GROUP_NAMES[k],
            )

        n_group_mean_list.append(means)
        n_group_std_list.append(stds)
        n_group_mean_per_aa[aa] = means
        n_group_std_per_aa[aa] = stds
        logger.info("  " + "·" * 42)

    # ── Per-AA grouped lo/hi bands ────────────────────────────────────────
    n_group_lo_list: list[list[float]] = []
    n_group_hi_list: list[list[float]] = []
    n_group_lo_per_aa: dict[str, list[float]] = {}
    n_group_hi_per_aa: dict[str, list[float]] = {}

    for aa_idx, aa in enumerate(AA_ORDER):
        los, his = [], []
        for k in range(NUM_GROUPS):
            vals_k = grouped_by_aa[aa_idx][k]
            if len(vals_k) < 2:
                lo_k, hi_k = 0.0, 1.0
            else:
                arr_k = np.array(vals_k)
                lo_k = float(np.percentile(arr_k, p_lo))
                hi_k = float(np.percentile(arr_k, p_hi))
            los.append(round(lo_k, 4))
            his.append(round(hi_k, 4))
        n_group_lo_list.append(los)
        n_group_hi_list.append(his)
        n_group_lo_per_aa[aa] = los
        n_group_hi_per_aa[aa] = his

    # ── Per-group rho^(k) lo/hi bands (from chain-level distributions) ────
    rho_group_lo: list[float] = []
    rho_group_hi: list[float] = []
    for k in range(NUM_GROUPS):
        rhos_k = np.array([d["rho_grouped"][k] for d in chain_data])
        rho_group_lo.append(round(float(np.percentile(rhos_k, p_lo)), 4))
        rho_group_hi.append(round(float(np.percentile(rhos_k, p_hi)), 4))

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        # ── Scalar per-AA coordination (backward-compatible) ──────────────
        "n_lo_per_aa": n_lo_per_aa,
        "n_hi_per_aa": n_hi_per_aa,
        "n_mean_per_aa": n_mean_per_aa,
        "n_std_per_aa": n_std_per_aa,
        "n_lo_list": n_lo_list,  # [20] p5  per AA
        "n_hi_list": n_hi_list,  # [20] p95 per AA
        "n_mean_list": n_mean_list,  # [20] mean per AA
        "n_std_list": n_std_list,  # [20] std  per AA
        # ── Group-conditional per-AA coordination (NEW) ───────────────────
        "n_group_mean_list": n_group_mean_list,  # [20][5] n*(k) centers
        "n_group_std_list": n_group_std_list,  # [20][5] sigma(k) widths
        "n_group_lo_list": n_group_lo_list,  # [20][5] p5  bands
        "n_group_hi_list": n_group_hi_list,  # [20][5] p95 bands
        "n_group_mean_per_aa": n_group_mean_per_aa,  # {aa: [m0..m4]}
        "n_group_std_per_aa": n_group_std_per_aa,  # {aa: [s0..s4]}
        "n_group_lo_per_aa": n_group_lo_per_aa,  # {aa: [lo0..lo4]}
        "n_group_hi_per_aa": n_group_hi_per_aa,  # {aa: [hi0..hi4]}
        # ── Scalar contact density ρ (backward-compatible) ────────────────
        "rho_fit": rho_fit,
        # ── Group-conditional contact density ρ^(k) (NEW) ────────────────
        "rho_group_fits": rho_group_fits,  # [5] fit dicts
        "rho_group_sigma": rho_group_sigma,  # [5] global std
        "rho_group_global_mean": rho_group_global_mean,  # [5] global mean
        "rho_group_lo": rho_group_lo,  # [5] p5  (global)
        "rho_group_hi": rho_group_hi,  # [5] p95 (global)
        # ── Group scheme ──────────────────────────────────────────────────
        "group_names": GROUP_NAMES,
        "group_assignment": AA_GROUP_ASSIGNMENT,
        "group_members": {str(k): v for k, v in GROUP_MEMBERS.items()},
        # ── Sigmoid parameters ────────────────────────────────────────────
        "sigmoid_style": "best" if args.best_style else "legacy",
        "sigmoid_r_half": r_half,
        "sigmoid_tau": tau,
        "sigmoid_beta": round(1.0 / tau, 2),
        "sigmoid_exclude": args.exclude,
        "sigmoid_max_dist": max_dist,
        # ── Metadata ──────────────────────────────────────────────────────
        "percentiles": [p_lo, p_hi],
        "n_structures": n_structures,
        "n_chains": n_chains,
        "n_residues": n_residues,
        "aa_order": AA_ORDER,
        "n_groups": NUM_GROUPS,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coord_n_star.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save raw chain data for external analysis
    raw_path = out_dir / "rho_per_chain.tsv"
    with open(raw_path, "w") as f:
        f.write("pdb_id\tchain_id\tL\trho\trg\n")
        for d in chain_data:
            f.write(f"{d['pdb_id']}\t{d['chain_id']}\t{d['L']}\t{d['rho']:.4f}\t{d['rg']:.2f}\n")

    logger.info("")
    logger.info("Saved to %s", out_path)
    logger.info("Saved raw chain data to %s", raw_path)
    logger.info("  n_structures: %d  n_chains: %d  n_residues: %d", n_structures, n_chains, n_residues)
    logger.info("")
    logger.info("Model builder keys:")
    logger.info("  coord_n_mean       = data['n_mean_list']        # _Hydrophobic Gaussian center n*")
    logger.info("  coord_n_std        = data['n_std_list']         # _Hydrophobic Gaussian width sigma")
    logger.info("  coord_n_lo         = data['n_lo_list']          # E_coord constraint lower bound")
    logger.info("  coord_n_hi         = data['n_hi_list']          # E_coord constraint upper bound")
    logger.info("  rho_fit            = data['rho_fit']            # ρ*(L) curve fit parameters")
    logger.info("  n_group_mean_list  = data['n_group_mean_list']  # grouped n_i^(k) centers [20x5]")
    logger.info("  n_group_std_list   = data['n_group_std_list']   # grouped n_i^(k) widths  [20x5]")
    logger.info("  group_assignment   = data['group_assignment']   # AA index -> group index [20]")

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        _plot_distributions(coord_by_aa, n_mean_list, n_std_list, out_dir)
        _plot_rho(chain_data, rho_fit, out_dir)
        _plot_group_distributions(
            grouped_by_aa,
            n_group_mean_list,
            n_group_std_list,
            out_dir,
        )
        _plot_rho_group(chain_data, rho_group_fits, out_dir)
