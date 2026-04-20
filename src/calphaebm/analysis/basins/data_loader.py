# src/calphaebm/analysis/basins/data_loader.py

"""Data loading functions for basin analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from calphaebm.data.pdb_parse import download_cif, parse_cif_ca_chains, split_chain_on_gaps
from calphaebm.geometry.internal import bond_angles, torsions
from calphaebm.utils.logging import get_logger

logger = get_logger()


@dataclass
class LoadStats:
    n_pdbs_attempted: int = 0
    n_pdbs_failed: int = 0
    n_chains_processed: int = 0
    n_segments_processed: int = 0
    n_pairs_collected: int = 0


def load_angle_data(
    pdb_ids: List[str],
    cache_dir: Path,
    max_pdbs: Optional[int] = None,
    max_chains: Optional[int] = None,
    sample_every: int = 10,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, LoadStats, list[tuple[str, str]]]:
    """
    Load (θ,φ) angle data from PDB structures.

    Args:
        pdb_ids: list of PDB IDs to process
        cache_dir: directory for PDB cache
        max_pdbs: maximum number of PDB IDs to process
        max_chains: maximum number of chains to process across all PDBs
        sample_every: take every N-th (θ,φ) pair to reduce correlation
        verbose: show progress bar

    Returns:
        theta_deg: (N,) bond angles in degrees
        phi_deg:   (N,) torsion angles in degrees (flipped to match convention)
        stats:     LoadStats summary
        failures:  list of (pdb_id, reason)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if max_pdbs is not None:
        pdb_ids = pdb_ids[: int(max_pdbs)]

    theta_list: list[float] = []
    phi_list: list[float] = []
    failures: list[tuple[str, str]] = []
    stats = LoadStats(n_pdbs_attempted=len(pdb_ids))

    chains_seen = 0

    pbar = tqdm(pdb_ids, desc="Processing PDBs", disable=not verbose)
    for pdb_id in pbar:
        try:
            cif_path = download_cif(pdb_id, str(cache_dir))
            chains = parse_cif_ca_chains(cif_path, pdb_id.lower())
        except Exception as e:
            stats.n_pdbs_failed += 1
            failures.append((pdb_id, f"parse/download error: {e}"))
            continue

        for chain in chains:
            if max_chains is not None and chains_seen >= int(max_chains):
                break

            chains_seen += 1
            stats.n_chains_processed += 1

            # Split at gaps to get contiguous segments
            try:
                segments = split_chain_on_gaps(chain.coords, chain.seq)
            except Exception as e:
                failures.append((pdb_id, f"split_chain_on_gaps error: {e}"))
                continue

            for coords, _seq in segments:
                stats.n_segments_processed += 1
                if len(coords) < 5:  # need >=5 residues for (theta,phi) pairs
                    continue

                # Convert to tensor
                R = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)  # (1, L, 3)

                # Compute angles in radians
                theta = bond_angles(R).squeeze(0).detach().cpu().numpy()  # (L-2,)
                phi = torsions(R).squeeze(0).detach().cpu().numpy()  # (L-3,)

                theta_deg = np.degrees(theta).astype(np.float32)
                phi_deg = np.degrees(phi).astype(np.float32)

                # Align: φ[i] pairs with θ[i]
                n_pairs = min(theta_deg.shape[0], phi_deg.shape[0])
                if n_pairs <= 0:
                    continue

                # Subsample to reduce correlation
                for i in range(0, n_pairs, int(max(sample_every, 1))):
                    theta_list.append(float(theta_deg[i]))
                    phi_list.append(float(phi_deg[i]))

        if max_chains is not None and chains_seen >= int(max_chains):
            break

    theta_arr = np.array(theta_list, dtype=np.float32)
    phi_arr = np.array(phi_list, dtype=np.float32)
    stats.n_pairs_collected = int(theta_arr.size)

    logger.info(
        f"Collected {stats.n_pairs_collected} (θ,φ) pairs from "
        f"{stats.n_chains_processed} chains ({stats.n_pdbs_failed}/{stats.n_pdbs_attempted} PDBs failed)"
    )

    return theta_arr, phi_arr, stats, failures
