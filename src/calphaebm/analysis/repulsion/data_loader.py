"""Data loading utilities for repulsion analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm

from calphaebm.data.id_utils import normalize_to_entry_ids
from calphaebm.data.pdb_parse import download_cif, parse_cif_ca_chains, split_chain_on_gaps
from calphaebm.utils.logging import get_logger

logger = get_logger()


@dataclass
class LoadStats:
    n_pdbs_attempted: int = 0
    n_pdbs_failed: int = 0
    n_chains_processed: int = 0
    n_segments_processed: int = 0


def load_entry_ids(pdb_list_path: Path) -> list[str]:
    """Load IDs from file; accept entry IDs or entity IDs; normalize to entry IDs."""
    with open(pdb_list_path, "r") as f:
        raw = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return normalize_to_entry_ids(raw)


def iter_segments_from_entries(
    entry_ids: Iterable[str],
    cache_dir: Path,
    max_entries: Optional[int] = None,
    max_chains: Optional[int] = None,
    min_len: int = 5,
    verbose: bool = True,
):
    """Yield (coords, seq, entry_id) for contiguous segments from entry IDs."""
    entry_ids = list(entry_ids)
    if max_entries is not None:
        entry_ids = entry_ids[: int(max_entries)]

    stats = LoadStats()
    failures: list[tuple[str, str]] = []

    pbar = tqdm(entry_ids, desc="Processing PDBs", disable=not verbose)
    for eid in pbar:
        stats.n_pdbs_attempted += 1
        try:
            cif_path = download_cif(eid, str(cache_dir))
            chains = parse_cif_ca_chains(cif_path, eid.lower())
        except Exception as e:
            stats.n_pdbs_failed += 1
            failures.append((eid, f"download/parse_failed: {e}"))
            continue

        # Optionally limit chains per entry
        if max_chains is not None:
            chains = chains[: int(max_chains)]

        for ch in chains:
            stats.n_chains_processed += 1
            for coords, seq in split_chain_on_gaps(ch.coords, ch.seq):
                stats.n_segments_processed += 1
                if len(coords) < min_len:
                    continue
                coords_np = np.asarray(coords, dtype=np.float32)
                seq_np = np.asarray(seq, dtype=np.int64)
                yield coords_np, seq_np, eid, stats, failures
