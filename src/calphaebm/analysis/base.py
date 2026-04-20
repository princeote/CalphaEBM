"""Base classes and utilities for analysis."""

from pathlib import Path
from typing import List, Optional

import numpy as np

from calphaebm.data.pdb_parse import download_cif, parse_cif_ca_chains, split_chain_on_gaps
from calphaebm.utils.logging import get_logger

logger = get_logger()


class AnalysisBase:
    """Base class for analysis tools."""

    def __init__(self, cache_dir: Path, output_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / "data"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

    def load_pdb_list(self, file_path: Path) -> List[str]:
        """Load PDB IDs from file."""
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    def process_chains(self, pdb_ids: List[str], max_chains: Optional[int] = None):
        """Process PDB chains - to be implemented by subclasses."""
        raise NotImplementedError


class AminoAcidMixin:
    """Mixin for amino acid related utilities."""

    AA_NAMES = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

    AA_INDICES = list(range(20))

    @staticmethod
    def idx_to_name(idx: int) -> str:
        """Convert index to amino acid name."""
        return AminoAcidMixin.AA_NAMES[idx]

    @staticmethod
    def name_to_idx(name: str) -> int:
        """Convert amino acid name to index."""
        return AminoAcidMixin.AA_NAMES.index(name)
