"""PyTorch Dataset for full-length PDB Cα chains — high-quality filtered.

Quality filters (all strict):
  1. Monomeric: single protein chain per PDB entry (no complex subunits)
  2. Complete: no missing residues (strict residue number continuity)
  3. B-factor: mean B < 40 Å², no more than 10% residues with B > 80 Å²
  4. Rg ratio: Rg/Rg_Flory ≤ 1.3 (no elongated assembly strands)
  5. Geometry: bond lengths, angles, chirality validated
  6. Length: 40 ≤ L ≤ 512

Usage:
    dataset = PDBChainDataset(
        pdb_ids=train_ids, cache_dir="./pdb_cache",
        min_len=40, max_len=512,
        max_rg_ratio=1.3,
        require_monomeric=True,
        require_complete=True,
        max_mean_bfactor=40.0,
        max_high_b_frac=0.10,
    )
    loader = DataLoader(dataset, batch_size=8, collate_fn=PDBChainDataset.collate)
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from calphaebm.data.pdb_parse import (
    DEFAULT_MAX_HIGH_B_FRAC,
    DEFAULT_MAX_MEAN_BFACTOR,
    DEFAULT_MAX_RESIDUE_BFACTOR,
    count_protein_chains,
    download_cif,
    is_complete_chain,
    parse_cif_ca_chains,
    passes_bfactor_filter,
    split_chain_on_gaps,
    validate_ca_geometry,
)
from calphaebm.utils.logging import get_logger

logger = get_logger()


class PDBChainDataset(Dataset):
    """Dataset of full-length Cα chains with strict quality filters.

    Every chain in this dataset is:
      - From a monomeric PDB entry (single protein chain)
      - Complete (no missing residues in the backbone)
      - Well-resolved (low B-factors)
      - Globular (Rg within Flory scaling bounds)
      - Geometrically valid (correct bond lengths/angles)

    Args:
        pdb_ids: List of PDB entry IDs to load.
        cache_dir: Directory for cached mmCIF files.
        min_len: Minimum chain length (residues). Default 40.
        max_len: Maximum chain length (residues). Default 512.
        max_ca_jump: Max Cα-Cα distance for gap detection (Å). Default 4.5.
        max_chains: Maximum number of chains to keep. Default None (all).
        max_rg_ratio: Maximum Rg/Rg_Flory ratio. Default 1.3.
        require_monomeric: If True, only keep chains from single-chain PDB
            entries. Default True.
        require_complete: If True, reject any chain with missing residues
            (gaps in residue numbering). Default True.
        max_mean_bfactor: Maximum mean Cα B-factor (Å²). Default 40.0.
        max_high_b_frac: Maximum fraction of residues with B > 80 Å².
            Default 0.10.
        center_coords: If True, center coordinates at origin.
        cache_processed: If True, cache to disk for faster reloading.
        processed_cache_dir: Cache directory for processed chains.
        force_reprocess: If True, ignore cache.
    """

    def __init__(
        self,
        pdb_ids: List[str],
        cache_dir: str = "./pdb_cache",
        min_len: int = 40,
        max_len: int = 512,
        max_ca_jump: float = 4.5,
        max_chains: Optional[int] = None,
        max_rg_ratio: Optional[float] = 1.3,
        require_monomeric: bool = True,
        require_complete: bool = True,
        max_mean_bfactor: float = DEFAULT_MAX_MEAN_BFACTOR,
        max_high_b_frac: float = DEFAULT_MAX_HIGH_B_FRAC,
        center_coords: bool = True,
        cache_processed: bool = True,
        processed_cache_dir: str = "./processed_cache",
        force_reprocess: bool = False,
    ):
        self.center_coords = center_coords
        self.min_len = min_len
        self.max_len = max_len

        pdb_ids_sorted = sorted(set(pdb_ids))

        # Cache key — includes all filter params
        ids_hash = hashlib.sha256("\n".join(pdb_ids_sorted).encode()).hexdigest()[:12]
        rg_str = f"_Rg{max_rg_ratio:.2f}" if max_rg_ratio else ""
        mono_str = "_mono" if require_monomeric else ""
        complete_str = "_complete" if require_complete else ""
        bfac_str = f"_B{max_mean_bfactor:.0f}" if max_mean_bfactor else ""
        param_str = (
            f"{ids_hash}_chain_L{min_len}-{max_len}_J{max_ca_jump:.1f}"
            f"_max{max_chains}{rg_str}{mono_str}{complete_str}{bfac_str}"
        )
        cache_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        cache_name = f"chains_{cache_hash}.pt"
        cache_path = Path(processed_cache_dir) / cache_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Try cache
        loaded = False
        if cache_processed and not force_reprocess and cache_path.exists():
            logger.debug("Loading cached chains from %s", cache_path)
            try:
                self.chains = torch.load(cache_path, weights_only=False)
                loaded = True
                logger.debug("Loaded %d chains from cache", len(self.chains))
            except Exception as e:
                logger.warning("Cache load failed: %s", e)

        if not loaded:
            self.chains = self._load_chains(
                pdb_ids_sorted,
                cache_dir,
                min_len,
                max_len,
                max_ca_jump,
                max_chains,
                max_rg_ratio,
                require_monomeric,
                require_complete,
                max_mean_bfactor,
                max_high_b_frac,
            )
            if cache_processed and self.chains:
                try:
                    torch.save(self.chains, cache_path)
                    logger.debug("Cached %d chains to %s", len(self.chains), cache_path)
                except Exception as e:
                    logger.warning("Cache save failed: %s", e)

        logger.info(
            "PDBChainDataset: %d clean chains (L=%d-%d) from %d PDB IDs",
            len(self.chains),
            min_len,
            max_len,
            len(pdb_ids_sorted),
        )

    def _load_chains(
        self,
        pdb_ids,
        cache_dir,
        min_len,
        max_len,
        max_ca_jump,
        max_chains,
        max_rg_ratio=None,
        require_monomeric=True,
        require_complete=True,
        max_mean_bfactor=DEFAULT_MAX_MEAN_BFACTOR,
        max_high_b_frac=DEFAULT_MAX_HIGH_B_FRAC,
    ) -> List[Dict[str, Any]]:
        """Load full chains with strict quality filtering."""
        import requests

        # Lazy import
        _rg_func = None
        if max_rg_ratio is not None:
            from calphaebm.evaluation.metrics.rg import radius_of_gyration

            _rg_func = radius_of_gyration

        chains = []
        stats = Counter()
        sess = requests.Session()

        logger.info(
            "Loading full chains from %d PDB IDs (L=%d-%d) with strict filters:",
            len(pdb_ids),
            min_len,
            max_len,
        )
        logger.info(
            "  monomeric=%s  complete=%s  max_mean_B=%.0f  max_high_B_frac=%.0f%%  max_rg=%.2f",
            require_monomeric,
            require_complete,
            max_mean_bfactor,
            max_high_b_frac * 100,
            max_rg_ratio if max_rg_ratio else 999,
        )

        for i, pid in enumerate(pdb_ids):
            if max_chains and len(chains) >= max_chains:
                break

            try:
                cif_path = download_cif(pid, cache_dir=cache_dir, session=sess)
            except Exception:
                stats["download_fail"] += 1
                continue

            try:
                raw_chains = parse_cif_ca_chains(str(cif_path), pid)
            except Exception:
                stats["parse_fail"] += 1
                continue

            # Monomeric filter: count protein chains (L≥20) in this entry
            if require_monomeric:
                n_protein_chains = sum(1 for c in raw_chains if len(c) >= 10)
                if n_protein_chains > 1:
                    stats["multi_chain"] += 1
                    continue
                if n_protein_chains == 0:
                    stats["no_protein_chain"] += 1
                    continue

            # Process each chain
            candidates_for_pid = []
            for chain_ca in raw_chains:
                L = len(chain_ca)

                # Length filter
                if L < min_len:
                    stats["too_short"] += 1
                    continue
                if L > max_len:
                    stats["too_long"] += 1
                    continue

                # Missing residues filter (strict — reject entire chain)
                if require_complete and chain_ca.has_missing_residues:
                    stats["missing_residues"] += 1
                    continue

                # B-factor filter
                b_ok, b_reason = passes_bfactor_filter(
                    chain_ca,
                    max_mean_b=max_mean_bfactor,
                    max_high_b_frac=max_high_b_frac,
                )
                if not b_ok:
                    stats["high_bfactor"] += 1
                    continue

                # Geometry validation (no gap splitting — chain must be clean as-is)
                valid, reason = validate_ca_geometry(chain_ca.coords, chain_ca.seq)
                if not valid:
                    stats["bad_geometry"] += 1
                    continue

                # Rg ratio filter
                if _rg_func is not None:
                    rg = _rg_func(chain_ca.coords)
                    rg_flory = 2.0 * L**0.38
                    if rg / rg_flory > max_rg_ratio:
                        stats["rg_outlier"] += 1
                        continue

                candidates_for_pid.append(
                    {
                        "pdb_id": chain_ca.pdb_id,
                        "chain_id": chain_ca.chain_id,
                        "coords": chain_ca.coords.astype(np.float32),
                        "seq": chain_ca.seq.astype(np.int64),
                    }
                )

            # Keep only the longest chain per PDB
            if candidates_for_pid:
                best = max(candidates_for_pid, key=lambda c: len(c["coords"]))
                chains.append(best)
                stats["accepted"] += 1
                stats["skipped_dup_chains"] += len(candidates_for_pid) - 1

            if (i + 1) % 500 == 0:
                logger.info("  Processed %d/%d IDs, %d chains accepted", i + 1, len(pdb_ids), len(chains))

        logger.info(
            "Chain loading complete:\n"
            "  accepted=%d (1 per PDB)\n"
            "  multi_chain=%d  missing_residues=%d  high_bfactor=%d\n"
            "  rg_outlier=%d  bad_geometry=%d  too_short=%d  too_long=%d\n"
            "  download_fail=%d  parse_fail=%d  skipped_dup=%d",
            stats["accepted"],
            stats["multi_chain"],
            stats["missing_residues"],
            stats["high_bfactor"],
            stats["rg_outlier"],
            stats["bad_geometry"],
            stats["too_short"],
            stats["too_long"],
            stats["download_fail"],
            stats["parse_fail"],
            stats["skipped_dup_chains"],
        )
        return chains

    def __len__(self) -> int:
        return len(self.chains)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        item = self.chains[idx]
        coords = torch.tensor(item["coords"], dtype=torch.float32)
        seq = torch.tensor(item["seq"], dtype=torch.long)

        if self.center_coords:
            coords = coords - coords.mean(dim=0, keepdim=True)

        return coords, seq, item["pdb_id"], item["chain_id"]

    def get_lengths(self) -> List[int]:
        """Return list of chain lengths."""
        return [len(c["coords"]) for c in self.chains]

    @staticmethod
    def collate(batch):
        """Collate variable-length chains into a padded batch.

        Returns 5-tuple:
            coords_padded: (B, L_max, 3) zero-padded coordinates.
            seq_padded:    (B, L_max) zero-padded sequence indices.
            pdb_ids:       list of B PDB ID strings.
            chain_ids:     list of B chain ID strings.
            lengths:       (B,) int64 tensor of real chain lengths.
        """
        if len(batch) == 1:
            coords, seq, pdb_id, chain_id = batch[0]
            lengths = torch.tensor([coords.shape[0]], dtype=torch.long)
            return (coords.unsqueeze(0), seq.unsqueeze(0), [pdb_id], [chain_id], lengths)

        max_len = max(coords.shape[0] for coords, _, _, _ in batch)
        B = len(batch)

        coords_padded = torch.zeros(B, max_len, 3)
        seq_padded = torch.zeros(B, max_len, dtype=torch.long)
        lengths = []
        pdb_ids = []
        chain_ids = []

        for i, (coords, seq, pid, cid) in enumerate(batch):
            L = coords.shape[0]
            coords_padded[i, :L] = coords
            seq_padded[i, :L] = seq
            lengths.append(L)
            pdb_ids.append(pid)
            chain_ids.append(cid)

        return (coords_padded, seq_padded, pdb_ids, chain_ids, torch.tensor(lengths, dtype=torch.long))
