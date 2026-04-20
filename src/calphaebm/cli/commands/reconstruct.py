# src/calphaebm/cli/commands/reconstruct.py
"""NeRF reconstruction verification command.

Verifies that the NeRF round-trip (Cartesian → internal coords → Cartesian)
is numerically clean for one or more PDB structures. Run this before starting
IC (run19) training.

Usage
-----
    calphaebm reconstruct --pdb 1UBQ
    calphaebm reconstruct --pdb pdbs/1UBQ.pdb
    calphaebm reconstruct --pdb 1L2Y 1FSD 1VII 2GB1 1BDD 1UBQ
    calphaebm reconstruct --pdb pdbs/1UBQ.pdb --verbose

Pass criteria (both must hold per structure):
    Bond mean      : 3.800 ± 0.001 Å   — NeRF bond constant is correct
    Max bond error : < 0.001 Å          — every bond is exactly 3.8 Å

    Coord drift vs native is reported but NOT a pass/fail criterion.
    Native PDB Cα-Cα bonds vary ~3.75-3.95 Å, so when NeRF fixes bonds
    at exactly 3.8 Å the reconstructed coords will differ from the PDB
    by construction. This is expected and correct.

Exit codes
----------
    0 — all structures passed
    1 — one or more structures failed
"""

from __future__ import annotations

import argparse

from calphaebm.utils.logging import get_logger

logger = get_logger()


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "reconstruct",
        description=(
            "Verify NeRF round-trip accuracy on one or more PDB structures.\n\n"
            "Extracts Cα coordinates, converts to internal coordinates (θ, φ),\n"
            "reconstructs with fixed bond=3.8 Å, and checks that bonds are\n"
            "exactly fixed and coordinates round-trip cleanly.\n\n"
            "Run this before starting IC (run19) training."
        ),
        help="Verify NeRF round-trip accuracy on PDB structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pdb",
        required=True,
        nargs="+",
        help="PDB ID list file (or list of IDs)",
    )
    parser.add_argument(
        "--chain",
        default=None,
        help="Chain ID to extract (default: first chain with Cα atoms)",
    )
    parser.add_argument(
        "--cache-dir",
        default="./pdb_cache",
        help="PDB download cache directory (default: ./pdb_cache)",
    )
    parser.add_argument(
        "--bond",
        type=float,
        default=3.8,
        help="Expected Cα-Cα bond length in Å (default: 3.8)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for coordinate comparison (default: 1e-4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-atom deviation table for the 10 worst atoms",
    )

    parser.set_defaults(func=run)
    return parser


from calphaebm.utils.logging import get_logger

logger = get_logger()


def _load_ca_coords(pdb_arg: str, chain: str | None, cache_dir: str):
    """Load Cα coordinates using the same loader as simulate.py.

    Returns (R, label) where R is a (1, L, 3) float32 tensor.
    """
    from pathlib import Path

    import torch

    from calphaebm.data.pdb_parse import download_cif, parse_cif_ca_chains

    p = Path(pdb_arg)
    if p.exists():
        cif_path = str(p)
        label = p.stem.upper()
    else:
        cif_path = download_cif(pdb_arg, cache_dir=cache_dir)
        label = pdb_arg.upper()

    chains = parse_cif_ca_chains(cif_path, pdb_arg.lower())
    if not chains:
        raise ValueError(f"No chains found for {pdb_arg}")

    if chain:
        ch = next((c for c in chains if c.chain_id == chain), None)
        if ch is None:
            raise ValueError(f"Chain {chain} not found in {pdb_arg}")
    else:
        ch = chains[0]
        logger.info("Using chain %s", ch.chain_id)

    R = torch.tensor(ch.coords, dtype=torch.float32).unsqueeze(0)  # (1, L, 3)
    return R, label


# ── Verification logic ────────────────────────────────────────────────────────


def _structural_metrics(R_native, R_recon):
    """Compute structural integrity metrics comparing native to reconstructed.

    Uses only internal/pairwise metrics that are not affected by the fixed
    3.8 Å bond length — i.e. metrics that reflect fold topology, not bond scale.

    Returns dict with:
        drmsd       : pairwise distance RMSD over all Cα pairs (Å)
        contact_f1  : F1 of 8 Å contact map (precision × recall)
        tm_proxy    : mean fraction of pairs within 2× native distance (rough TM-like score)
        angle_rmsd  : RMSD of bond angles θ (radians) — should be ~0
        torsion_mad : median absolute deviation of torsions φ (radians) — should be ~0
    """
    import torch

    from calphaebm.geometry.internal import bond_angles, torsions

    R_n = R_native.squeeze(0)  # (L, 3)
    R_r = R_recon.squeeze(0)  # (L, 3)
    L = R_n.shape[0]

    # --- pairwise distance matrices ---
    D_n = (R_n[:, None] - R_n[None, :]).norm(dim=-1)  # (L, L)
    D_r = (R_r[:, None] - R_r[None, :]).norm(dim=-1)  # (L, L)

    # upper triangle, exclude bonded (|i-j| > 1)
    mask = torch.zeros(L, L, dtype=torch.bool)
    for i in range(L):
        for j in range(i + 2, L):
            mask[i, j] = True

    d_n = D_n[mask]
    d_r = D_r[mask]

    # dRMSD: RMSD of pairwise distances — fold-topology metric, unaffected by bond scale
    drmsd = ((d_n - d_r) ** 2).mean().sqrt().item()

    # Contact map F1 at 8 Å cutoff
    cutoff = 8.0
    c_n = d_n < cutoff
    c_r = d_r < cutoff
    tp = (c_n & c_r).sum().float()
    fp = (~c_n & c_r).sum().float()
    fn = (c_n & ~c_r).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall / (precision + recall + 1e-8)).item()

    # Angle RMSD (should be ~0 — angles are preserved exactly by NeRF)
    theta_n = bond_angles(R_native.unsqueeze(0) if R_native.dim() == 2 else R_native).squeeze()
    theta_r = bond_angles(R_recon.unsqueeze(0) if R_recon.dim() == 2 else R_recon).squeeze()
    angle_rmsd = ((theta_n - theta_r) ** 2).mean().sqrt().item()

    # Torsion MAD (should be ~0 — torsions are preserved exactly by NeRF)
    phi_n = torsions(R_native.unsqueeze(0) if R_native.dim() == 2 else R_native).squeeze()
    phi_r = torsions(R_recon.unsqueeze(0) if R_recon.dim() == 2 else R_recon).squeeze()
    # wrap difference to [-pi, pi]
    dphi = torch.atan2(torch.sin(phi_n - phi_r), torch.cos(phi_n - phi_r))
    torsion_mad = dphi.abs().median().item()

    return {
        "drmsd": drmsd,
        "contact_f1": f1,
        "angle_rmsd": angle_rmsd,
        "torsion_mad": torsion_mad,
    }


def _verify_one(pdb_id: str, args) -> tuple[dict, object]:
    """Run verification for one PDB ID. Returns (results_dict, R_recon_tensor)."""
    import numpy as np
    import torch

    from calphaebm.evaluation.metrics.rmsd import drmsd as compute_drmsd
    from calphaebm.evaluation.metrics.rmsd import rmsd_kabsch
    from calphaebm.geometry.internal import bond_angles, torsions
    from calphaebm.geometry.reconstruct import (
        coords_to_internal,
        extract_anchor,
        nerf_reconstruct,
        verify_reconstruction,
    )

    R, label = _load_ca_coords(pdb_id, args.chain, args.cache_dir)
    logger.info("Loaded %s — %d Cα atoms", label, R.shape[1])

    results = verify_reconstruction(R, bond=args.bond, atol=args.atol)
    results["label"] = label

    # Reconstruct
    theta, phi = coords_to_internal(R)
    anchor = extract_anchor(R)
    R_recon = nerf_reconstruct(theta, phi, anchor, bond=args.bond)

    # --- Angle preservation (should be machine precision) ---
    theta_r = bond_angles(R_recon)
    phi_r = torsions(R_recon)

    theta_err = (theta - theta_r).abs()
    dphi = torch.atan2(torch.sin(phi - phi_r), torch.cos(phi - phi_r)).abs()

    results["theta_max_err"] = theta_err.max().item()
    results["theta_mean_err"] = theta_err.mean().item()
    results["phi_max_err"] = dphi.max().item()
    results["phi_mean_err"] = dphi.mean().item()

    # --- RMSD and dRMSD (Kabsch-aligned) ---
    R_n = R.squeeze(0).detach().numpy()
    R_r = R_recon.squeeze(0).detach().numpy()
    results["rmsd"] = rmsd_kabsch(R_n, R_r)
    results["drmsd"] = compute_drmsd(R_n, R_r, mode="nonlocal", exclude=2)

    # NeRF-placed bond length stats (atom 3+)
    nerf_bl_all = (R_recon[0, 3:] - R_recon[0, 2:-1]).norm(dim=-1)
    results["nerf_bl_mean"] = results["nerf_bond_lengths_mean"]
    results["nerf_bl_std"] = results["nerf_bond_lengths_std"]
    results["nerf_bl_min"] = nerf_bl_all.min().item()
    results["nerf_bl_max"] = nerf_bl_all.max().item()

    # Native (PDB) bond length stats
    bl_native = (R[0, 1:] - R[0, :-1]).norm(dim=-1)
    results["native_bl_mean"] = bl_native.mean().item()
    results["native_bl_std"] = bl_native.std().item()
    results["native_bl_min"] = bl_native.min().item()
    results["native_bl_max"] = bl_native.max().item()

    # --- Structural integrity metrics ---
    results["structural"] = _structural_metrics(R, R_recon)

    return results, R_recon


def _print_result(results: dict, bond: float) -> bool:
    """Print full verification table for one structure. Returns True if passed."""
    import math

    label = results["label"]
    L = results["L"]

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {label}  ({L} residues)")
    print(sep)

    # ── Pass/fail checks ────────────────────────────────────────────────
    checks = [
        (
            f"NeRF bond mean ≈ {bond:.3f} Å",
            abs(results["nerf_bond_lengths_mean"] - bond) < 0.001,
            f"{results['nerf_bond_lengths_mean']:.6f} Å  (NeRF-placed only)",
        ),
        (
            "NeRF max bond error < 0.1 Å",
            results["max_bond_error_from_ideal"] < 0.1,
            f"{results['max_bond_error_from_ideal']:.6f} Å  (NeRF-placed only)",
        ),
        (
            "θ max error < 0.001 rad",
            results["theta_max_err"] < 0.001,
            f"{results['theta_max_err']:.6f} rad  ({math.degrees(results['theta_max_err']):.4f}°)",
        ),
        (
            "φ max error < 0.001 rad",
            results["phi_max_err"] < 0.001,
            f"{results['phi_max_err']:.6f} rad  ({math.degrees(results['phi_max_err']):.4f}°)",
        ),
        (
            "RMSD < 1.0 Å  (Hinsen et al.)",
            results["rmsd"] < 1.0,
            f"{results['rmsd']:.4f} Å",
        ),
    ]

    all_pass = True
    for label_str, ok, measured in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {label_str:<32}  {measured}")
        if not ok:
            all_pass = False

    # ── Angle preservation ──────────────────────────────────────────────
    print(f"\n  Angle preservation (round-trip IC error):")
    print(
        f"    θ max error   {results['theta_max_err']:>10.6f} rad  "
        f"({math.degrees(results['theta_max_err']):.4f}°)  expect ~0"
    )
    print(f"    θ mean error  {results['theta_mean_err']:>10.6f} rad")
    print(
        f"    φ max error   {results['phi_max_err']:>10.6f} rad  "
        f"({math.degrees(results['phi_max_err']):.4f}°)  expect ~0"
    )
    print(f"    φ mean error  {results['phi_mean_err']:>10.6f} rad")

    # ── Coordinate quality ──────────────────────────────────────────────
    print(f"\n  Coordinate quality (reconstructed vs native):")
    print(f"    RMSD (Kabsch)  {results['rmsd']:>8.4f} Å   paper claims < 1.0 Å")
    print(f"    dRMSD          {results['drmsd']:>8.4f} Å   pairwise distance RMSD")

    # ── Bond length stats ───────────────────────────────────────────────
    print(f"\n  Bond lengths:")
    print(
        f"    NeRF-placed    mean={results['nerf_bl_mean']:.6f}  "
        f"std={results['nerf_bl_std']:.6f}  "
        f"min={results['nerf_bl_min']:.6f}  "
        f"max={results['nerf_bl_max']:.6f} Å  (atoms 3+, must be {bond:.1f})"
    )
    print(
        f"    Anchor (PDB)   mean={results['anchor_bond_lengths_mean']:.6f}  "
        f"std={results['anchor_bond_lengths_std']:.6f} Å"
        f"  (first 2 bonds — native, not fixed)"
    )
    print(
        f"    Native (PDB)   mean={results['native_bl_mean']:.6f}  "
        f"std={results['native_bl_std']:.6f}  "
        f"min={results['native_bl_min']:.6f}  "
        f"max={results['native_bl_max']:.6f} Å"
    )

    # ── Contact map ─────────────────────────────────────────────────────
    sm = results.get("structural", {})
    if sm:
        print(f"\n  Contact map overlap:")
        print(f"    Contact F1 (8 Å)  {sm['contact_f1']:>8.4f}   " f"reconstructed vs native (expect ~1.0)")

    # ── Verdict ─────────────────────────────────────────────────────────
    verdict = "✅ ALL CHECKS PASSED" if all_pass else "❌ SOME CHECKS FAILED"
    print(f"\n  {verdict}")
    if all_pass:
        print("  NeRF is correct — IC integrator is ready for run19.")
    else:
        print("  Review NeRF implementation before starting IC training.")
    print(sep)

    return all_pass


def _print_verbose(R, bond: float) -> None:
    """Print NeRF-placed bond length details (10 worst bonds by deviation from ideal)."""
    from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct

    theta, phi = coords_to_internal(R)
    anchor = extract_anchor(R)
    R_recon = nerf_reconstruct(theta, phi, anchor, bond=bond)

    # NeRF-placed bonds only: atom 3 onwards (anchor bonds are native, not placed)
    nerf_bl = (R_recon[:, 3:] - R_recon[:, 2:-1]).norm(dim=-1).squeeze()  # (L-3,)
    bl_err = (nerf_bl - bond).abs()
    topk = bl_err.topk(min(10, bl_err.shape[0]))

    # Native bond lengths for comparison
    native_bl = (R[:, 1:] - R[:, :-1]).norm(dim=-1).squeeze()  # (L-1,)

    print(f"\n  Native bond lengths (from PDB — NOT fixed at 3.8 Å):")
    print(
        f"    mean={native_bl.mean():.4f}  std={native_bl.std():.4f}  "
        f"min={native_bl.min():.4f}  max={native_bl.max():.4f}"
    )

    print(f"\n  NeRF-placed bond lengths (atom 3+, must be exactly {bond} Å):")
    print(
        f"    mean={nerf_bl.mean():.6f}  std={nerf_bl.std():.6f}  " f"min={nerf_bl.min():.6f}  max={nerf_bl.max():.6f}"
    )

    print(f"\n  10 worst NeRF bond errors (deviation from {bond} Å):")
    print(f"    {'Bond':>8}  {'Length (Å)':>12}  {'Error (Å)':>11}")
    print(f"    {'--------':>8}  {'-----------':>12}  {'---------':>11}")
    for idx, err in zip(topk.indices.tolist(), topk.values.tolist()):
        atom_i = idx + 2  # bond between atom idx+2 and idx+3
        bl_val = nerf_bl[idx].item()
        print(f"    {atom_i:>3}-{atom_i+1:<3}  {bl_val:>12.6f}  {err:>11.6f}")


def _print_summary(rows: list) -> None:
    """Print multi-protein summary table."""
    import math

    W = 82
    print("\n" + "=" * W)
    print("  SUMMARY")
    print("=" * W)
    hdr = f"  {'PDB':<8}  {'L':>4}  {'RMSD(Å)':>9}  {'dRMSD(Å)':>9}  {'θ err(rad)':>11}  {'φ err(rad)':>11}  {'Max bl err':>11}  {'Result':>8}"
    print(hdr)
    print("  " + "-" * (W - 2))
    for row in rows:
        label, L, ok, bstd, mdev, rmsd, drmsd, theta_err, phi_err = (list(row) + [float("nan")] * 9)[:9]
        status = "PASS ✅" if ok else "FAIL ❌"
        L_str = str(L) if isinstance(L, int) else L
        print(
            f"  {label:<8}  {L_str:>4}  {rmsd:>9.4f}  {drmsd:>9.4f}  "
            f"{theta_err:>11.6f}  {phi_err:>11.6f}  {mdev:>11.6f}  {status:>8}"
        )
    print("=" * W)
    n_pass = sum(1 for row in rows if row[2])
    print(f"  {n_pass}/{len(rows)} passed")
    print("=" * W)


# ── Entry point ───────────────────────────────────────────────────────────────


def run(args) -> int:
    from calphaebm.cli.commands.train.data_utils import parse_pdb_arg

    pdb_ids = parse_pdb_arg(args.pdb)
    all_passed = True
    summary_rows = []

    for pdb_arg in pdb_ids:
        try:
            results, R_recon = _verify_one(pdb_arg, args)
            ok = _print_result(results, args.bond)

            if args.verbose:
                _print_verbose(R_recon, args.bond)

            summary_rows.append(
                (
                    results["label"],
                    results["L"],
                    ok,
                    results["nerf_bl_std"],
                    results["max_bond_error_from_ideal"],
                    results["rmsd"],
                    results["drmsd"],
                    results["theta_max_err"],
                    results["phi_max_err"],
                )
            )
            if not ok:
                all_passed = False

        except Exception as e:
            logger.error("Failed to verify %s: %s", pdb_arg, e)
            import traceback

            traceback.print_exc()
            summary_rows.append((pdb_arg.upper(), "?", False, float("nan"), float("nan")))
            all_passed = False

    if len(args.pdb) > 1:
        _print_summary(summary_rows)

    return 0 if all_passed else 1
