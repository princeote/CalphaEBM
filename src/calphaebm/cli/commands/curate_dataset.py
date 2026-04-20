# src/calphaebm/cli/commands/curate_dataset.py

"""Curate a training dataset by applying coordinate-level quality filters.

Takes a PDB ID list (typically from `calphaebm build-dataset`) and applies
chain-level filters that require downloading and parsing coordinates:

  1. Length filter: L=40-512 (already in PDBChainDataset)
  2. Geometry validation: bond lengths, planarity (already in PDBChainDataset)
  3. Rg ratio filter: reject chains with Rg/Rg_Flory > cutoff (NEW)

The Rg filter removes elongated structures (coiled-coils, fibrils, extended
helices) that violate the Flory scaling assumption in E_Rg and corrupt
per-residue coordination statistics used by E_coord.

How this differs from build-dataset:
  build-dataset:   Queries RCSB, clusters at 30% identity, selects
                   representatives. Produces ~10K entity IDs. No coordinate-
                   level filtering — operates on metadata only.
  curate-dataset:  Takes entity/entry IDs, downloads coordinates, applies
                   shape/geometry filters. Produces a clean subset ready for
                   training. Requires network access for initial download,
                   but caches processed chains for subsequent runs.

Usage:
    # Step 1: Build initial candidate set (metadata only, fast)
    calphaebm build-dataset --target 15000 --resolution 3.0 \\
        --out-entries candidate_entries.txt

    # Step 2: Check what coordinate filters will remove (dry run)
    calphaebm curate-dataset --stats \\
        --pdb candidate_entries.txt \\
        --max-rg-ratio 1.3

    # Step 3: Apply filters and save clean list
    calphaebm curate-dataset \\
        --pdb candidate_entries.txt \\
        --exclude test_entries.txt \\
        --max-rg-ratio 1.3 \\
        --out train_entities_v2.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from calphaebm.utils.logging import get_logger

logger = get_logger()


def add_parser(subparsers) -> argparse.ArgumentParser:
    """Add curate-dataset command parser."""
    parser = subparsers.add_parser(
        "curate-dataset",
        description="Apply quality filters to a PDB dataset and report statistics",
        help="Filter dataset by Rg ratio, geometry, and chain length",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pdb",
        required=True,
        nargs="+",
        help="PDB ID list file(s) or individual IDs",
    )
    parser.add_argument(
        "--cache-dir",
        default="./pdb_cache",
        help="PDB cache directory (default: ./pdb_cache)",
    )
    parser.add_argument(
        "--processed-cache-dir",
        default="./processed_cache",
        help="Processed chain cache directory (default: ./processed_cache)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=40,
        help="Minimum chain length (default: 40)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Maximum chain length (default: 512)",
    )
    parser.add_argument(
        "--max-rg-ratio",
        type=float,
        default=1.3,
        help="Maximum Rg/Rg_Flory ratio (default: 1.3). Set to 0 to disable.",
    )

    # Output
    parser.add_argument(
        "--out",
        default="train_entities_curated.txt",
        help="Output file for curated PDB IDs (default: train_entities_curated.txt)",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="Output JSON metadata file (default: <out>.meta.json)",
    )
    parser.add_argument(
        "--train-percent",
        type=float,
        default=95.0,
        help="Percentage of data for training (default: 95, rest goes to val)",
    )
    parser.add_argument(
        "--out-val",
        default=None,
        help="Output file for val IDs (default: <out> with _val suffix)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )

    # Modes
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Stats-only mode: report filter statistics without saving",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        nargs="+",
        help="PDB ID list file(s) to exclude (e.g., test set)",
    )
    parser.add_argument(
        "--require-monomer-assembly",
        action="store_true",
        default=True,
        help="Filter out biological multimers via RCSB assembly API (default: True). "
        "This catches homodimers deposited as single chains with crystallographic "
        "symmetry, which the chain-count filter in PDBChainDataset misses.",
    )
    parser.add_argument(
        "--no-monomer-assembly",
        action="store_false",
        dest="require_monomer_assembly",
        help="Disable biological assembly filter (keep multimers).",
    )
    parser.add_argument(
        "--assembly-delay",
        type=float,
        default=0.05,
        help="Delay between RCSB assembly API calls in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing (ignore processed cache)",
    )

    parser.set_defaults(func=run)
    return parser


def _parse_pdb_ids(pdb_args: list[str]) -> list[str]:
    """Parse PDB IDs from file(s) or direct arguments.

    Handles comment lines (#) and inline comments (1ABC  # description).
    """
    ids = []
    for arg in pdb_args:
        p = Path(arg)
        if p.exists() and p.is_file():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Take first whitespace-delimited token (strips inline comments)
                    pdb_id = line.split()[0]
                    ids.append(pdb_id)
        else:
            ids.append(arg)
    return ids


def _parse_exclude_ids(exclude_args: list[str] | None) -> set[str]:
    """Parse exclusion PDB IDs.

    Handles comment lines (#), inline comments, and entity format (1ABC_1 → 1abc).
    """
    if not exclude_args:
        return set()
    ids = set()
    for arg in exclude_args:
        p = Path(arg)
        if p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # First token, then strip entity suffix
                    pdb_id = line.split()[0].split("_")[0].lower()
                    ids.add(pdb_id)
        else:
            ids.add(arg.split()[0].split("_")[0].lower())
    return ids


def run(args) -> int:
    """Run curate-dataset command."""
    import torch

    from calphaebm.data.pdb_chain_dataset import PDBChainDataset
    from calphaebm.evaluation.metrics.rg import radius_of_gyration

    # ── Parse input IDs ──────────────────────────────────────────────────
    pdb_ids = _parse_pdb_ids(args.pdb)
    logger.info("Input PDB IDs: %d", len(pdb_ids))

    # ── Exclude test set ─────────────────────────────────────────────────
    exclude_ids = _parse_exclude_ids(args.exclude)
    if exclude_ids:
        before = len(pdb_ids)
        pdb_ids = [pid for pid in pdb_ids if pid.split("_")[0].lower() not in exclude_ids]
        logger.info("Excluded %d test set entries: %d → %d", before - len(pdb_ids), before, len(pdb_ids))

    # ── Filter biological multimers via RCSB assembly API ────────────────
    assembly_info = {}
    multimer_ids_removed = []
    if args.require_monomer_assembly:
        import requests as _requests

        from calphaebm.data.rcsb_query import filter_multimers

        sess = _requests.Session()

        mono_ids, multi_ids, assembly_info = filter_multimers(
            entry_ids=pdb_ids,
            session=sess,
            delay=args.assembly_delay,
            verbose=True,
        )
        multimer_ids_removed = multi_ids
        before = len(pdb_ids)
        pdb_ids = mono_ids
        logger.info("Assembly filter: %d → %d (removed %d multimers)", before, len(pdb_ids), len(multi_ids))

    # ── Parse Rg ratio ───────────────────────────────────────────────────
    max_rg_ratio = args.max_rg_ratio
    if max_rg_ratio and max_rg_ratio <= 0:
        max_rg_ratio = None

    # ── Load chains (without Rg filter — we want full stats) ─────────────
    logger.info("Loading chains (L=%d-%d)...", args.min_len, args.max_len)
    ds = PDBChainDataset(
        pdb_ids=pdb_ids,
        cache_dir=args.cache_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        max_rg_ratio=None,  # no Rg filter yet — compute stats first
        cache_processed=True,
        processed_cache_dir=args.processed_cache_dir,
        force_reprocess=args.force_reprocess,
    )
    n_total = len(ds)
    logger.info("Chains after length + geometry: %d", n_total)

    if n_total == 0:
        logger.error("No chains passed length/geometry filters!")
        return 1

    # ── Compute Rg statistics ────────────────────────────────────────────
    logger.info("Computing Rg statistics...")
    rg_data = []
    for i in range(n_total):
        R, seq, pid, cid = ds[i]
        L = R.shape[0]
        rg = radius_of_gyration(R.numpy())
        rg_flory = 2.0 * L**0.38
        ratio = rg / rg_flory
        rg_data.append(
            {
                "pdb_id": pid,
                "chain_id": cid,
                "L": L,
                "rg": float(rg),
                "rg_flory": float(rg_flory),
                "ratio": float(ratio),
            }
        )

        if (i + 1) % 1000 == 0:
            logger.info("  Computed Rg for %d/%d chains", i + 1, n_total)

    ratios = np.array([d["ratio"] for d in rg_data])
    lengths = np.array([d["L"] for d in rg_data])

    # ── Report statistics ────────────────────────────────────────────────
    logger.info("")
    logger.info("═══════════════════════════════════════════════════════")
    logger.info("  DATASET CURATION REPORT")
    logger.info("═══════════════════════════════════════════════════════")
    logger.info("  Input PDB IDs:           %d", len(pdb_ids) + len(multimer_ids_removed) + len(exclude_ids))
    logger.info("  After test exclusion:    %d", len(pdb_ids) + len(multimer_ids_removed))
    if args.require_monomer_assembly:
        logger.info("  After assembly filter:   %d  (removed %d multimers)", len(pdb_ids), len(multimer_ids_removed))
    logger.info("  After length+geometry:   %d", n_total)
    logger.info("")

    # Rg ratio distribution
    logger.info("  Rg/Rg_Flory distribution:")
    for cutoff in [1.1, 1.2, 1.3, 1.4, 1.5, 2.0]:
        n = int((ratios <= cutoff).sum())
        pct = 100 * n / len(ratios)
        marker = " ← selected" if max_rg_ratio and abs(cutoff - max_rg_ratio) < 0.01 else ""
        logger.info("    ≤ %.1f: %5d / %d  (%5.1f%%)%s", cutoff, n, len(ratios), pct, marker)

    # Apply Rg filter
    if max_rg_ratio:
        mask = ratios <= max_rg_ratio
    else:
        mask = np.ones(len(ratios), dtype=bool)
    filtered_lengths = lengths[mask]
    n_pass = int(mask.sum())

    logger.info("")
    logger.info(
        "  With --max-rg-ratio %.1f:  %d chains pass (%d rejected)", max_rg_ratio or 999, n_pass, n_total - n_pass
    )

    # Length distribution
    logger.info("")
    logger.info("  Length distribution (after all filters):")
    bins = [(40, 80), (80, 150), (150, 250), (250, 400), (400, 512)]
    for lo, hi in bins:
        n = int(((filtered_lengths >= lo) & (filtered_lengths <= hi)).sum())
        pct = 100 * n / max(len(filtered_lengths), 1)
        bar = "█" * int(pct / 2)
        logger.info("    L=%3d-%3d: %5d  (%5.1f%%)  %s", lo, hi, n, pct, bar)

    # Rg stats for passing chains
    passing_ratios = ratios[mask]
    logger.info("")
    logger.info(
        "  Rg/Rg_Flory (passing chains): mean=%.3f  std=%.3f  " "p05=%.3f  p50=%.3f  p95=%.3f",
        np.mean(passing_ratios),
        np.std(passing_ratios),
        np.percentile(passing_ratios, 5),
        np.percentile(passing_ratios, 50),
        np.percentile(passing_ratios, 95),
    )

    logger.info("")
    train_pct = args.train_percent if hasattr(args, "train_percent") else 95.0
    logger.info(
        "  Train/val split (%.0f/%.0f): ~%d train + ~%d val",
        train_pct,
        100 - train_pct,
        int(n_pass * train_pct / 100),
        int(n_pass * (100 - train_pct) / 100),
    )

    # Top outliers
    sorted_by_ratio = sorted(rg_data, key=lambda d: -d["ratio"])
    logger.info("")
    logger.info("  Top 15 Rg outliers (rejected):")
    for d in sorted_by_ratio[:15]:
        logger.info(
            "    %s_%s  L=%3d  Rg=%6.1f  expected=%5.1f  ratio=%.2f",
            d["pdb_id"],
            d["chain_id"],
            d["L"],
            d["rg"],
            d["rg_flory"],
            d["ratio"],
        )

    logger.info("═══════════════════════════════════════════════════════")

    # ── Stats-only mode: done ────────────────────────────────────────────
    if args.stats:
        logger.info("Stats-only mode (--stats) — no output saved.")
        return 0

    # ── Save filtered PDB IDs ────────────────────────────────────────────
    passing_entries = sorted({rg_data[i]["pdb_id"] for i in range(len(rg_data)) if mask[i]})

    # Train/val split
    import random as _rng

    _rng.seed(args.seed)
    entries_shuffled = list(passing_entries)
    _rng.shuffle(entries_shuffled)
    n_train = int(len(entries_shuffled) * args.train_percent / 100.0)
    train_entries = sorted(entries_shuffled[:n_train])
    val_entries = sorted(entries_shuffled[n_train:])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for eid in train_entries:
            f.write(eid + "\n")
    logger.info("Saved %d train IDs to %s", len(train_entries), out_path)

    val_path = (
        Path(args.out_val)
        if args.out_val
        else out_path.with_name(
            out_path.stem.replace("train", "val") + out_path.suffix
            if "train" in out_path.stem
            else out_path.stem + "_val" + out_path.suffix
        )
    )
    with open(val_path, "w") as f:
        for eid in val_entries:
            f.write(eid + "\n")
    logger.info(
        "Saved %d val IDs to %s (%.0f/%.0f split, seed=%d)",
        len(val_entries),
        val_path,
        args.train_percent,
        100 - args.train_percent,
        args.seed,
    )

    # ── Save metadata ────────────────────────────────────────────────────
    meta_path = Path(args.meta) if args.meta else out_path.with_suffix(".meta.json")
    meta = {
        "input_pdb_ids": len(pdb_ids) + len(multimer_ids_removed) + len(exclude_ids),
        "excluded_test_ids": len(exclude_ids),
        "excluded_multimers": len(multimer_ids_removed),
        "require_monomer_assembly": args.require_monomer_assembly,
        "chains_after_length_geometry": n_total,
        "max_rg_ratio": max_rg_ratio,
        "chains_after_rg_filter": n_pass,
        "entries_saved": len(passing_entries),
        "min_len": args.min_len,
        "max_len": args.max_len,
        "rg_ratio_stats": {
            "mean": float(np.mean(passing_ratios)),
            "std": float(np.std(passing_ratios)),
            "p05": float(np.percentile(passing_ratios, 5)),
            "p50": float(np.percentile(passing_ratios, 50)),
            "p95": float(np.percentile(passing_ratios, 95)),
        },
        "length_distribution": {
            f"L{lo}-{hi}": int(((filtered_lengths >= lo) & (filtered_lengths <= hi)).sum()) for lo, hi in bins
        },
        "filters_applied": [
            f"length: {args.min_len}-{args.max_len}",
            "geometry: validate_ca_geometry",
            "gap_splitting: max_ca_jump=4.5",
            "monomer_assembly: RCSB biological assembly API"
            if args.require_monomer_assembly
            else "monomer_assembly: disabled",
            f"rg_ratio: ≤ {max_rg_ratio}" if max_rg_ratio else "rg_ratio: disabled",
        ],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    return 0
