# src/calphaebm/cli/commands/build_dataset.py

"""Build PDB70-like dataset command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calphaebm.data.build_pdb70_like import build_pdb70_like
from calphaebm.utils.logging import get_logger

logger = get_logger()


def add_parser(subparsers):
    """Add build-dataset command parser."""
    parser = subparsers.add_parser(
        "build-dataset",
        description="Build PDB70-like nonredundant dataset",
        help="Build dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--target",
        type=int,
        default=10000,
        help="Target number of entities (default: 10000)",
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=2.0,
        help="Max resolution in Å (default: 2.0)",
    )

    parser.add_argument(
        "--out-entities",
        type=Path,
        default=Path("pdb70_like_entities.txt"),
        help="Output file for entity IDs (default: pdb70_like_entities.txt)",
    )

    parser.add_argument(
        "--out-entries",
        type=Path,
        default=Path("pdb70_like_entries.txt"),
        help="Output file for entry IDs (default: pdb70_like_entries.txt)",
    )

    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("pdb70_like_meta.json"),
        help="Output file for metadata (default: pdb70_like_meta.json)",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=5000,
        help="Search API page size (default: 5000)",
    )

    parser.add_argument(
        "--graphql-batch",
        type=int,
        default=200,
        help="GraphQL batch size (default: 200)",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
        help="Experimental methods to include (default: X-ray + cryo-EM). "
        "Use 'X-RAY DIFFRACTION' and/or 'ELECTRON MICROSCOPY'.",
    )

    parser.set_defaults(func=run)


def run(args):
    """Run build-dataset command."""
    if args.target <= 0:
        raise ValueError(f"--target must be > 0 (got {args.target})")
    if args.resolution <= 0:
        raise ValueError(f"--resolution must be > 0 (got {args.resolution})")

    logger.info(f"Building PDB70-like dataset with target {args.target}")
    logger.info(f"Max resolution: {args.resolution} Å")
    logger.info(f"Methods: {', '.join(args.methods)}")
    logger.info(f"Search page size: {args.page_size}")
    logger.info(f"GraphQL batch size: {args.graphql_batch}")

    # Build polymer_entity IDs (what you want for training)
    res = build_pdb70_like(
        target_n=args.target,
        max_resolution=args.resolution,
        methods=args.methods,
        output_type="polymer_entity",
        page_size=args.page_size,
        graphql_batch_size=args.graphql_batch,
        verbose=True,
    )

    entity_ids = list(res.output_ids)  # polymer_entity IDs like "1ABC_1"
    entry_ids = sorted({eid.split("_")[0].upper() for eid in entity_ids if eid})

    # Ensure output dirs exist
    args.out_entities.parent.mkdir(parents=True, exist_ok=True)
    args.out_entries.parent.mkdir(parents=True, exist_ok=True)
    args.meta.parent.mkdir(parents=True, exist_ok=True)

    # Save entity IDs
    with open(args.out_entities, "w") as f:
        for pid in entity_ids:
            f.write(pid + "\n")

    # Save entry IDs
    with open(args.out_entries, "w") as f:
        for eid in entry_ids:
            f.write(eid + "\n")

    # Metadata
    meta = {
        "target": int(args.target),
        "resolution": float(args.resolution),
        "methods": args.methods,
        "output_type": "polymer_entity",
        "n_selected_entities": int(len(entity_ids)),
        "n_unique_entries": int(len(entry_ids)),
        "n_candidate_entries_scanned": int(getattr(res, "n_candidate_entries_scanned", -1)),
        "n_unique_clusters_approx": int(len(entity_ids)),
    }

    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved {len(entity_ids)} entities to {args.out_entities}")
    logger.info(f"Saved {len(entry_ids)} entries to {args.out_entries}")
    logger.info(f"Metadata: {args.meta}")

    return 0
