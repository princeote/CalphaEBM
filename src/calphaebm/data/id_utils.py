# src/calphaebm/data/id_utils.py
from __future__ import annotations

from typing import Iterable, List, Tuple


def split_entity_id(x: str) -> Tuple[str, str | None]:
    """
    Split an ID that may be an entry ID (1ABC) or polymer entity ID (1ABC_1).

    Returns:
        (entry_id, entity_suffix) where entity_suffix is like "1" or None.
    """
    x = x.strip()
    if "_" in x:
        entry, suffix = x.split("_", 1)
        return entry, suffix
    return x, None


def normalize_to_entry_ids(ids: Iterable[str]) -> List[str]:
    """Convert a list of entry/entity IDs to entry IDs only (upper)."""
    out: List[str] = []
    for s in ids:
        s = s.strip()
        if not s or s.startswith("#"):
            continue
        entry, _ = split_entity_id(s)
        out.append(entry.upper())
    return out
