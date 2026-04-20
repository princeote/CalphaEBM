# src/calphaebm/analysis/hbonds/cli.py
"""
CLI for H-bond distance analysis: ``calphaebm analyze hbonds``

Computes Cα-Cα distance distributions for:
  1. Helical i→i+4 contacts (α H-bonds)
  2. Sheet nonlocal contacts (β H-bonds)
  3. Coil i→i+4 contacts (control)

Classification uses Ramachandran angle regions (not distance thresholds)
to avoid circular logic — we measure d(i,i+4) conditioned on secondary
structure, not the other way around.

Outputs are used to initialize the Gaussian distance kernels g_α and g_β
in the secondary structure energy term.

Example::

    calphaebm analyze hbonds \\
        --pdb-list train_hq.txt \\
        --cache-dir pdb_cache \\
        --output-dir analysis/hbond_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from calphaebm.data.pdb_parse import download_cif, parse_cif_ca_chains
from calphaebm.utils.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Angle computation
# ---------------------------------------------------------------------------


def _compute_angles(R: np.ndarray):
    """Compute bond angles (θ) and torsions (φ) from Cα coords.

    Uses the codebase functions to ensure convention consistency
    with the basin energy surfaces.

    Args:
        R: (L, 3) Cα coordinates

    Returns:
        theta_deg: (L-2,) bond angles in degrees
        phi_deg: (L-3,) torsion angles in degrees
    """
    L = R.shape[0]
    if L < 4:
        return np.array([]), np.array([])

    from calphaebm.geometry.internal import bond_angles, torsions

    R_t = torch.tensor(R, dtype=torch.float32).unsqueeze(0)
    theta_rad = bond_angles(R_t)
    phi_rad = torsions(R_t)
    theta_deg = np.degrees(theta_rad.squeeze(0).numpy())
    phi_deg = np.degrees(phi_rad.squeeze(0).numpy())
    return theta_deg, phi_deg


# ---------------------------------------------------------------------------
# Basin assignment from energy surfaces
# ---------------------------------------------------------------------------


def _load_basin_surfaces(basin_dir: Path):
    """Load the 4 Ramachandran basin energy surfaces.

    Returns (4, n_theta, n_phi) array of energies, plus bin edges.
    """
    surfaces = []
    for i in range(4):
        path = basin_dir / f"basin_{i}_energy.npy"
        if not path.exists():
            raise FileNotFoundError(f"Basin surface not found: {path}")
        surfaces.append(np.load(path))
    surfaces = np.stack(surfaces, axis=0)
    theta_edges = np.load(basin_dir / "theta_edges_deg.npy")
    phi_edges = np.load(basin_dir / "phi_edges_deg.npy")
    return surfaces, theta_edges, phi_edges


def _assign_basins(theta_deg, phi_deg, surfaces, theta_edges, phi_edges):
    """Assign each (θ, φ) to its lowest-energy basin (0-3).

    Basin 1 = helix, Basin 3 = sheet, Basins 0,2 = coil.
    """
    n_theta_bins = surfaces.shape[1]
    n_phi_bins = surfaces.shape[2]

    theta_idx = np.clip(np.digitize(theta_deg, theta_edges) - 1, 0, n_theta_bins - 1)
    phi_idx = np.clip(np.digitize(phi_deg, phi_edges) - 1, 0, n_phi_bins - 1)

    energies = np.stack([surfaces[b][theta_idx, phi_idx] for b in range(4)], axis=-1)

    return np.argmin(energies, axis=-1)


# ---------------------------------------------------------------------------
# Per-chain processing
# ---------------------------------------------------------------------------

# Basin identity: 1 = helix, 0 = sheet, 2 = PPII, 3 = turn
HELIX_BASIN = 1
SHEET_BASIN = 0


def _process_chain(R: np.ndarray, surfaces, theta_edges, phi_edges, max_sheet_dist: float = 12.0):
    """Process one chain. Returns helix, sheet, coil distance arrays.

    Classification by basin assignment from Ramachandran energy surfaces
    (data-driven from basins analysis on HQ data).

    Index convention:
        theta[k] is the bond angle at residue k+1 (k=0..L-3)
        phi[k] is the torsion at residue k+1 (k=0..L-4)
        Basin assignment valid for residues 1..L-3 (using both θ and φ)

    For helical i→i+4 H-bond:
        Need residues i, i+1, i+2, i+3 ALL in helix basin
        Need d(Cα_i, Cα_{i+4}) to exist → i+4 ≤ L-1
        So i ranges from 1 to min(n_valid-3, L-5)
    """
    L = R.shape[0]
    if L < 8:
        return [], [], []

    theta_deg, phi_deg = _compute_angles(R)
    if len(theta_deg) < 2 or len(phi_deg) < 1:
        return [], [], []

    n_valid = min(len(theta_deg) - 1, len(phi_deg))
    basins = _assign_basins(
        theta_deg[:n_valid],
        phi_deg[:n_valid],
        surfaces,
        theta_edges,
        phi_edges,
    )

    is_helix = basins == HELIX_BASIN
    is_sheet = basins == SHEET_BASIN
    is_coil = (~is_helix) & (~is_sheet)

    # ── Helical i→i+4 distances ──────────────────────────────────────
    helix_dists = []
    coil_dists = []

    max_i = min(n_valid - 3, L - 5)
    for i in range(1, max_i + 1):
        # Check all 4 consecutive residues: i, i+1, i+2, i+3
        # In the is_helix array (0-indexed from residue 1): indices i-1, i, i+1, i+2
        if is_helix[i - 1] and is_helix[i] and is_helix[i + 1] and is_helix[i + 2]:
            d = float(np.linalg.norm(R[i] - R[i + 4]))
            helix_dists.append(d)
        elif is_coil[i - 1] and is_coil[i] and is_coil[i + 1] and is_coil[i + 2]:
            d = float(np.linalg.norm(R[i] - R[i + 4]))
            coil_dists.append(d)

    # ── Sheet nonlocal distances ─────────────────────────────────────
    sheet_dists = []
    sheet_residues = np.where(is_sheet)[0] + 1  # convert to residue indices

    for a_idx in range(len(sheet_residues)):
        i = sheet_residues[a_idx]
        for b_idx in range(a_idx + 1, len(sheet_residues)):
            j = sheet_residues[b_idx]
            if abs(i - j) > 4:
                d = float(np.linalg.norm(R[i] - R[j]))
                if d < max_sheet_dist:
                    sheet_dists.append(d)

    return helix_dists, sheet_dists, coil_dists


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``hbonds`` subcommand under ``calphaebm analyze``."""
    p = subparsers.add_parser(
        "hbonds",
        help="Compute H-bond distance distributions for secondary structure energy",
        description=(
            "Compute Cα-Cα distance distributions for helical (i→i+4) and "
            "sheet (nonlocal) contacts using Ramachandran angle classification. "
            "Outputs are used to initialize g_α and g_β Gaussian kernels."
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
        "--basin-dir",
        type=str,
        default="analysis/secondary_analysis/data",
        help="Directory with basin_*_energy.npy files (default: analysis/secondary_analysis/data)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="analysis/hbond_analysis",
        help="Output directory (default: analysis/hbond_analysis)",
    )
    p.add_argument(
        "--max-sheet-dist",
        type=float,
        default=12.0,
        help="Max distance for sheet nonlocal contacts in Angstrom (default: 12.0)",
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
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
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
    """Run H-bond distance analysis."""

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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load basin energy surfaces for secondary structure classification
    basin_dir = Path(args.basin_dir)
    surfaces, theta_edges, phi_edges = _load_basin_surfaces(basin_dir)
    logger.info("Loaded %d basin surfaces (%s bins) from %s", surfaces.shape[0], surfaces.shape[1:], basin_dir)

    # Verify basin identity by checking peaks
    for b in range(4):
        S = surfaces[b]
        min_idx = np.unravel_index(S.argmin(), S.shape)
        peak_theta = (theta_edges[min_idx[0]] + theta_edges[min_idx[0] + 1]) / 2
        peak_phi = (phi_edges[min_idx[1]] + phi_edges[min_idx[1] + 1]) / 2
        labels = {HELIX_BASIN: "HELIX", SHEET_BASIN: "SHEET"}
        label = labels.get(b, "coil")
        logger.info("  Basin %d: peak at theta=%.1f, phi=%.1f -> %s", b, peak_theta, peak_phi, label)

    # ── Collect distances ─────────────────────────────────────────────
    all_helix = []
    all_sheet = []
    all_coil = []
    n_structures = 0
    n_chains = 0
    n_skipped = 0

    import requests

    session = requests.Session()

    for i, pdb_id in enumerate(pdb_ids):
        try:
            cif_path = download_cif(pdb_id, cache_dir=str(cache_dir), session=session)
        except Exception:
            n_skipped += 1
            continue

        if cif_path is None:
            n_skipped += 1
            continue

        try:
            raw_chains = parse_cif_ca_chains(str(cif_path), pdb_id)
        except Exception:
            n_skipped += 1
            continue

        if not raw_chains:
            n_skipped += 1
            continue

        n_structures += 1

        for chain in raw_chains:
            L = len(chain.seq)
            if L < args.min_len or L > args.max_len:
                continue

            R = chain.coords.astype(np.float64)
            h, s, c = _process_chain(
                R,
                surfaces,
                theta_edges,
                phi_edges,
                max_sheet_dist=args.max_sheet_dist,
            )
            all_helix.extend(h)
            all_sheet.extend(s)
            all_coil.extend(c)
            n_chains += 1

        if not args.quiet and (i + 1) % 500 == 0:
            logger.info(
                "  Processed %d/%d PDBs (%d chains, %d helix, %d sheet, %d coil)",
                i + 1,
                len(pdb_ids),
                n_chains,
                len(all_helix),
                len(all_sheet),
                len(all_coil),
            )

    session.close()

    all_helix = np.array(all_helix, dtype=np.float32)
    all_sheet = np.array(all_sheet, dtype=np.float32)
    all_coil = np.array(all_coil, dtype=np.float32)

    logger.info(
        "Done: %d PDBs, %d chains, %d skipped",
        n_structures,
        n_chains,
        n_skipped,
    )
    logger.info(
        "  Helical i→i+4:     %d distances",
        len(all_helix),
    )
    logger.info(
        "  Sheet nonlocal:    %d distances",
        len(all_sheet),
    )
    logger.info(
        "  Coil i→i+4:        %d distances",
        len(all_coil),
    )

    if n_chains == 0:
        logger.error("No valid chains found.")
        sys.exit(1)

    # ── Save raw data ─────────────────────────────────────────────────
    np.save(data_dir / "helix_d_i4.npy", all_helix)
    np.save(data_dir / "sheet_d_nonlocal.npy", all_sheet)
    np.save(data_dir / "coil_d_i4.npy", all_coil)
    logger.info("Saved .npy files to %s", data_dir)

    # ── Statistics ────────────────────────────────────────────────────
    stats = {
        "n_structures": n_structures,
        "n_chains": n_chains,
        "classification": "basin_argmin",
        "basin_dir": str(args.basin_dir),
        "helix_basin": HELIX_BASIN,
        "sheet_basin": SHEET_BASIN,
    }

    for name, arr in [("helix_i4", all_helix), ("sheet_nonlocal", all_sheet), ("coil_i4", all_coil)]:
        if len(arr) > 0:
            stats[name] = {
                "n": int(len(arr)),
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
                "median": round(float(np.median(arr)), 4),
                "p5": round(float(np.percentile(arr, 5)), 4),
                "p95": round(float(np.percentile(arr, 95)), 4),
            }
            logger.info(
                "  %s: n=%d  mean=%.3fÅ  std=%.3fÅ  [p5=%.2f, p95=%.2f]",
                name,
                len(arr),
                arr.mean(),
                arr.std(),
                np.percentile(arr, 5),
                np.percentile(arr, 95),
            )
        else:
            stats[name] = {"n": 0}
            logger.warning("  %s: no data", name)

    json_path = out_dir / "hbond_distance_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved stats to %s", json_path)

    # ── Print initialization values ───────────────────────────────────
    if stats["helix_i4"]["n"] > 0:
        h = stats["helix_i4"]
        logger.info("")
        logger.info("  g_alpha init: mu=%.2f sigma=%.2f", h["mean"], h["std"])

    # ── Plots ─────────────────────────────────────────────────────────
    if not args.no_plots:
        _plot_distributions(all_helix, all_sheet, all_coil, stats, out_dir)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_distributions(all_helix, all_sheet, all_coil, stats, out_dir):
    """Plot distance distributions with Gaussian fits."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
    except ImportError:
        logger.warning("matplotlib or scipy not available — skipping plots")
        return

    def gauss1(x, mu, sig, amp):
        return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    def gauss2(x, mu1, sig1, amp1, mu2, sig2, amp2):
        return amp1 * np.exp(-0.5 * ((x - mu1) / sig1) ** 2) + amp2 * np.exp(-0.5 * ((x - mu2) / sig2) ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "H-bond Cα-Cα Distance Distributions\n" "Classification by Ramachandran angle region (not distance)",
        fontsize=13,
        fontweight="bold",
    )

    # ── Helix i→i+4 ──────────────────────────────────────────────
    if len(all_helix) > 0:
        ax = axes[0]
        ax.hist(all_helix, bins=100, range=(3, 10), density=True, color="steelblue", alpha=0.7, edgecolor="none")
        mu = float(all_helix.mean())
        sig = float(all_helix.std())
        x = np.linspace(3, 10, 200)
        gauss = np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        ax.plot(x, gauss, "r-", linewidth=2, label=f"Gauss μ={mu:.2f} σ={sig:.2f}")
        ax.axvline(mu, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("d(Cα_i, Cα_{i+4}) [Å]")
        ax.set_ylabel("Density")
        ax.set_title(f"Helical i→i+4 (n={len(all_helix):,})\n→ g_α init")
        ax.legend(fontsize=9)

    # ── Sheet nonlocal — two Gaussians ────────────────────────────
    if len(all_sheet) > 100:
        ax = axes[1]
        counts, edges, _ = ax.hist(
            all_sheet,
            bins=120,
            range=(3, 12),
            density=True,
            color="coral",
            alpha=0.7,
            edgecolor="none",
        )
        bin_centers = (edges[:-1] + edges[1:]) / 2

        try:
            p0 = [5.0, 0.7, 0.3, 9.0, 1.2, 0.10]
            bounds = ([3, 0.1, 0.01, 6, 0.3, 0.01], [7, 2.0, 2.0, 12, 3.0, 2.0])
            popt, _ = curve_fit(gauss2, bin_centers, counts, p0=p0, bounds=bounds, maxfev=5000)
            mu1, sig1 = popt[0], popt[1]
            mu2, sig2 = popt[3], popt[4]
            if mu1 > mu2:
                mu1, sig1, mu2, sig2 = mu2, sig2, mu1, sig1
                popt = [mu1, sig1, popt[5], mu2, sig2, popt[2]]

            x = np.linspace(3, 12, 200)
            ax.plot(x, gauss2(x, *popt), "r-", linewidth=2, label="Two-Gauss fit")
            ax.plot(
                x, gauss1(x, *popt[:3]), "b--", linewidth=1.5, label=f"Peak 1 (anti-∥): μ={popt[0]:.2f} σ={popt[1]:.2f}"
            )
            ax.plot(x, gauss1(x, *popt[3:]), "g--", linewidth=1.5, label=f"Peak 2 (∥): μ={popt[3]:.2f} σ={popt[4]:.2f}")

            stats["sheet_peak1"] = {"mu": round(float(mu1), 4), "sigma": round(float(sig1), 4)}
            stats["sheet_peak2"] = {"mu": round(float(mu2), 4), "sigma": round(float(sig2), 4)}
            logger.info("  Sheet two-Gaussian fit:")
            logger.info("    Peak 1 (anti-parallel): mu=%.3f sigma=%.3f", mu1, sig1)
            logger.info("    Peak 2 (parallel):      mu=%.3f sigma=%.3f", mu2, sig2)

            # Re-save stats with peaks
            json_path = Path(out_dir) / "hbond_distance_stats.json"
            with open(json_path, "w") as f:
                json.dump(stats, f, indent=2)

        except Exception as e:
            logger.warning("Two-Gaussian fit failed: %s — using single Gaussian", e)
            mu = float(all_sheet.mean())
            sig = float(all_sheet.std())
            x = np.linspace(3, 12, 200)
            gauss = np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
            ax.plot(x, gauss, "r-", linewidth=2, label=f"Gauss μ={mu:.2f} σ={sig:.2f}")

        ax.set_xlabel("d(Cα_i, Cα_j) [Å]")
        ax.set_ylabel("Density")
        ax.set_title(f"Sheet nonlocal |i-j|>4 (n={len(all_sheet):,})\n→ g_β init")
        ax.legend(fontsize=8)

    # ── Coil control ──────────────────────────────────────────────
    if len(all_coil) > 0:
        ax = axes[2]
        ax.hist(
            all_coil, bins=100, range=(3, 16), density=True, color="gray", alpha=0.7, edgecolor="none", label="Coil"
        )
        if len(all_helix) > 0:
            ax.hist(
                all_helix,
                bins=100,
                range=(3, 16),
                density=True,
                color="steelblue",
                alpha=0.4,
                edgecolor="none",
                label="Helix",
            )
        mu_c = float(all_coil.mean())
        ax.axvline(mu_c, color="black", linestyle="--", alpha=0.5, label=f"Coil μ={mu_c:.2f}Å")
        ax.set_xlabel("d(Cα_i, Cα_{i+4}) [Å]")
        ax.set_ylabel("Density")
        ax.set_title(f"Coil i→i+4 control (n={len(all_coil):,})")
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = Path(out_dir) / "hbond_distances.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to %s", plot_path)
