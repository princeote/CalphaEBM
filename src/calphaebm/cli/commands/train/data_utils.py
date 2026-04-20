"""Data utilities for reading PDB IDs and saving splits."""

import os
from pathlib import Path
from typing import List, Tuple

from calphaebm.utils.logging import get_logger

logger = get_logger()


def _read_id_lines(path: str) -> List[str]:
    """Read IDs from file, skipping comments and empty lines."""
    ids: List[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def _normalize_ids(raw: List[str]) -> List[str]:
    """Normalize PDB IDs (take first part, uppercase, deduplicate)."""
    ids = [x.split("_")[0].upper() for x in raw]
    seen = set()
    out: List[str] = []
    for x in ids:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_pdb_arg(values: List[str]) -> List[str]:
    """Parse PDB argument (list of IDs or file path)."""
    if len(values) == 1 and os.path.exists(values[0]) and os.path.isfile(values[0]):
        return _normalize_ids(_read_id_lines(values[0]))
    return _normalize_ids(values)


def save_split_ids(train_ids: List[str], val_ids: List[str], save_dir: str, prefix: str) -> Tuple[Path, Path]:
    """Save train/val ID splits for reproducibility."""
    save_dir_path = Path(save_dir) / prefix
    save_dir_path.mkdir(parents=True, exist_ok=True)

    train_path = save_dir_path / "train_ids.txt"
    val_path = save_dir_path / "val_ids.txt"

    with open(train_path, "w") as f:
        for pid in train_ids:
            f.write(f"{pid}\n")
    with open(val_path, "w") as f:
        for pid in val_ids:
            f.write(f"{pid}\n")

    logger.info("Saved train/val splits to %s", save_dir_path)
    return train_path, val_path
