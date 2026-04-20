# src/calphaebm/analysis/backbone/data_loader.py

"""Data loading functions for backbone geometry analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from calphaebm.data.pdb_parse import download_cif, parse_cif_ca_chains, split_chain_on_gaps
from calphaebm.geometry.internal import bond_angles, bond_lengths, torsions
from calphaebm.utils.logging import get_logger

logger = get_logger()


def load_pdb_list(file_path: Path) -> List[str]:
    """Load PDB IDs from a text file (one per line)."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def extract_geometry_from_chains(
    pdb_ids: List[str],
    cache_dir: Path,
    max_chains: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract bond lengths, θ_i, θ_{i+1}, and φ from PDB chains.

    Conventions (Oldfield & Hubbard, Proteins 1994):
      - θ in degrees from bond_angles(R)
      - φ in degrees from torsions(R), standard convention:
            α-helix ≈ +50°, β-sheet ≈ -170°

    Returns:
        bond_lengths_all: (Nbonds,) Å
        theta_i:          (Nphi,) degrees
        theta_ip1:        (Nphi,) degrees
        phi:              (Nphi,) degrees
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    bond_lengths_list: list[float] = []
    theta_i_list: list[float] = []
    theta_ip1_list: list[float] = []
    phi_list: list[float] = []

    chains_processed = 0
    n_pdb_failed = 0

    pbar = tqdm(pdb_ids, desc="Processing PDBs")
    for pdb_id in pbar:
        try:
            cif_path = download_cif(pdb_id, str(cache_dir))
            chains = parse_cif_ca_chains(cif_path, pdb_id.lower())
        except Exception as e:
            n_pdb_failed += 1
            pbar.write(f"Error processing {pdb_id}: {e}")
            continue

        for chain in chains:
            if max_chains is not None and chains_processed >= int(max_chains):
                break

            chains_processed += 1

            # Split at gaps to get contiguous segments
            for coords, _seq in split_chain_on_gaps(chain.coords, chain.seq):
                if len(coords) < 5:
                    # Need >= 5 residues to have at least one φ and adjacent θ pair
                    continue

                R = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)

                # Avoid building autograd graphs during analysis
                with torch.no_grad():
                    lengths = bond_lengths(R).squeeze(0).cpu().numpy()  # (L-1,)
                    theta = bond_angles(R).squeeze(0).cpu().numpy()  # (L-2,)
                    phi = torsions(R).squeeze(0).cpu().numpy()  # (L-3,)

                theta_deg = np.degrees(theta).astype(np.float32)
                phi_deg = np.degrees(phi).astype(np.float32)

                # For each φ[i], pair with θ[i] and θ[i+1]
                n = min(phi_deg.shape[0], theta_deg.shape[0] - 1)
                if n <= 0:
                    continue

                bond_lengths_list.extend(lengths.astype(np.float32).tolist())
                theta_i_list.extend(theta_deg[:n].astype(np.float32).tolist())
                theta_ip1_list.extend(theta_deg[1 : n + 1].astype(np.float32).tolist())
                phi_list.extend(phi_deg[:n].astype(np.float32).tolist())

        if max_chains is not None and chains_processed >= int(max_chains):
            break

    logger.info(f"Chains processed: {chains_processed} | PDB failures: {n_pdb_failed}/{len(pdb_ids)}")

    return (
        np.array(bond_lengths_list, dtype=np.float32),
        np.array(theta_i_list, dtype=np.float32),
        np.array(theta_ip1_list, dtype=np.float32),
        np.array(phi_list, dtype=np.float32),
    )
