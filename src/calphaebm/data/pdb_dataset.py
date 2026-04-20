"""PyTorch Dataset for PDB Cα segments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from calphaebm.data.pdb_parse import load_pdb_segments
from calphaebm.utils.logging import get_logger

logger = get_logger()


class PDBSegmentDataset(Dataset):
    """Dataset of fixed-length Cα segments from PDB structures.

    Args:
        pdb_ids: List of PDB IDs to load.
        cache_dir: Directory for cached mmCIF files.
        seg_len: Segment length (number of residues).
        stride: Stride for sliding window.
        max_ca_jump: Max allowed Cα-Cα distance for gap detection.
        limit_segments: Maximum number of segments to load.
        validate_geometry: If True, filter out segments with broken geometry.
        center_coords: If True, center coordinates at origin (subtract mean).
        transform: Optional transform to apply to each segment.
        device: Device to place tensors on (None = CPU).
        return_dict: If True, return a dict instead of a tuple.
        cache_processed: If True, cache processed segments to disk for faster loading.
        processed_cache_dir: Directory to store cached segment files.
        cache_name: Optional name for cache file (default: auto-generated from parameters).
        force_reprocess: If True, ignore cache and reprocess all PDBs.
    """

    def __init__(
        self,
        pdb_ids: List[str],
        cache_dir: str = "./pdb_cache",
        seg_len: int = 128,
        stride: int = 64,
        max_ca_jump: float = 4.5,
        limit_segments: Optional[int] = None,
        validate_geometry: bool = True,
        center_coords: bool = True,
        transform: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        return_dict: bool = False,
        cache_processed: bool = True,
        processed_cache_dir: str = "./processed_cache",
        cache_name: Optional[str] = None,
        force_reprocess: bool = False,
    ):
        self.device = device
        self.transform = transform
        self.center_coords = bool(center_coords)
        self.return_dict = bool(return_dict)

        if seg_len < 2:
            raise ValueError(f"seg_len must be >=2 (got {seg_len})")
        if stride < 1:
            raise ValueError(f"stride must be >=1 (got {stride})")

        # Sort and deduplicate IDs for deterministic cache keys
        pdb_ids_sorted = sorted(set(pdb_ids))

        # Generate deterministic cache key
        if cache_name is None:
            # Stable hash of IDs list
            ids_blob = "\n".join(pdb_ids_sorted).encode("utf-8")
            ids_hash = hashlib.sha256(ids_blob).hexdigest()[:12]

            # Parameter string for cache invalidation
            param_str = (
                f"{ids_hash}_L{seg_len}_S{stride}_J{max_ca_jump:.1f}" f"_lim{limit_segments}_vg{int(validate_geometry)}"
            )
            cache_hash = hashlib.sha256(param_str.encode("utf-8")).hexdigest()[:16]
            cache_name = f"segments_{cache_hash}.pt"

        cache_path = Path(processed_cache_dir) / cache_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Try to load from cache
        cache_loaded = False
        if cache_processed and not force_reprocess and cache_path.exists():
            logger.info(f"Loading cached segments from {cache_path}")
            try:
                # Handle different PyTorch versions
                try:
                    self.segments = torch.load(cache_path, weights_only=False)
                except TypeError:
                    self.segments = torch.load(cache_path)
                cache_loaded = True
                logger.info(f"Loaded {len(self.segments)} segments from cache")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reprocessing...")

        # Load from PDBs if cache not available or failed
        if not cache_loaded:
            if force_reprocess:
                logger.info("Force reprocess enabled: ignoring cache")
            elif cache_processed and not cache_path.exists():
                logger.info(f"No cache found at {cache_path}")

            self.segments = self._load_segments(
                pdb_ids=pdb_ids_sorted,
                cache_dir=cache_dir,
                seg_len=seg_len,
                stride=stride,
                max_ca_jump=max_ca_jump,
                limit_segments=limit_segments,
                validate_geometry=validate_geometry,
            )

            # Save to cache if enabled and we have segments
            if cache_processed and len(self.segments) > 0:
                logger.info(f"Saving {len(self.segments)} segments to cache: {cache_path}")
                try:
                    torch.save(self.segments, cache_path)
                    logger.info("Cache saved successfully")
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")

        logger.info(f"Loaded {len(self.segments)} clean segments")
        if len(self.segments) == 0:
            logger.warning("No segments loaded! Check PDB IDs and parameters.")

    def _load_segments(self, pdb_ids, cache_dir, seg_len, stride, max_ca_jump, limit_segments, validate_geometry):
        """Internal method to load segments from PDBs."""
        logger.info(f"Loading PDB segments from {len(pdb_ids)} IDs...")
        segments = load_pdb_segments(
            pdb_ids=pdb_ids,
            cache_dir=cache_dir,
            seg_len=seg_len,
            stride=stride,
            max_ca_jump=max_ca_jump,
            limit_segments=limit_segments,
            validate_geometry=validate_geometry,
        )
        return segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, str, str], Dict[str, Any]]:
        item = self.segments[idx]

        coords = torch.tensor(item["coords"], dtype=torch.float32)
        seq = torch.tensor(item["seq"], dtype=torch.long)

        if self.center_coords:
            coords = coords - coords.mean(dim=0, keepdim=True)

        if self.transform:
            coords, seq = self.transform(coords, seq)

        if self.device is not None:
            coords = coords.to(self.device)
            seq = seq.to(self.device)

        if self.return_dict:
            return {
                "coords": coords,
                "seq": seq,
                "pdb_id": item["pdb_id"],
                "chain_id": item["chain_id"],
            }

        return coords, seq, item["pdb_id"], item["chain_id"]

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        return {
            "pdb_id": self.segments[idx]["pdb_id"],
            "chain_id": self.segments[idx]["chain_id"],
            "length": len(self.segments[idx]["coords"]),
        }

    def filter_by_length(self, min_len: int, max_len: int) -> "PDBSegmentDataset":
        filtered = [s for s in self.segments if min_len <= len(s["coords"]) <= max_len]

        new_ds = PDBSegmentDataset.__new__(PDBSegmentDataset)
        new_ds.segments = filtered
        new_ds.device = self.device
        new_ds.transform = self.transform
        new_ds.center_coords = self.center_coords
        new_ds.return_dict = self.return_dict
        return new_ds

    @classmethod
    def from_cache(cls, cache_path: str, **kwargs):
        """Load dataset directly from a cache file."""
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        # Create instance without loading
        instance = cls.__new__(cls)
        instance.device = kwargs.get("device", None)
        instance.transform = kwargs.get("transform", None)
        instance.center_coords = kwargs.get("center_coords", True)
        instance.return_dict = kwargs.get("return_dict", False)

        # Load segments from cache
        logger.info(f"Loading dataset from cache: {cache_path}")
        try:
            try:
                instance.segments = torch.load(cache_path, weights_only=False)
            except TypeError:
                instance.segments = torch.load(cache_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load cache: {e}")

        logger.info(f"Loaded {len(instance.segments)} segments")
        return instance
