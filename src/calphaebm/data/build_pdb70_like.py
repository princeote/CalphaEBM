# src/calphaebm/data/build_pdb70_like.py

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Literal, Optional, Set, Tuple

import requests
from tqdm import tqdm

from calphaebm.data.rcsb_query import (
    PolymerEntityInfo,
    graphql_polymer_entities_for_entries,
    is_protein_only_entry,
    search_entries_xray_resolution,
)
from calphaebm.utils.logging import get_logger

logger = get_logger()

OutputType = Literal["polymer_entity", "entry"]


@dataclass
class BuildResult:
    output_ids: List[str]
    output_type: OutputType
    polymer_entity_ids_selected: List[str]
    entry_ids_selected: List[str]
    n_candidate_entries_scanned: int


def build_pdb70_like(
    target_n: int = 1000,
    max_resolution: float = 2.0,
    methods: Optional[List[str]] = None,
    output_type: OutputType = "entry",
    max_pages: Optional[int] = None,
    page_size: int = 1000,
    graphql_batch_size: int = 200,
    retries: int = 3,
    session: Optional[requests.Session] = None,
    verbose: bool = True,
) -> BuildResult:
    """
    Build a PDB70-like nonredundant set using RCSB cluster_id_70.

    Args:
        target_n: number of items to select (entries or polymer entities depending on output_type)
        max_resolution: resolution cutoff (Å)
        methods: experimental methods to include. Default: X-ray + cryo-EM.
        output_type: "entry" or "polymer_entity"
        max_pages: optional cap on paging through search results
        page_size: search page size (RCSB supports up to 10k, but 1k is safe)
        graphql_batch_size: number of entry IDs per GraphQL call
        retries: network retries within rcsb_query
        session: optional requests session
        verbose: show tqdm progress

    Returns:
        BuildResult with output_ids and traceability mappings.
    """
    if methods is None:
        methods = ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
    if target_n <= 0:
        raise ValueError(f"target_n must be positive, got {target_n}")
    if max_resolution <= 0:
        raise ValueError(f"max_resolution must be positive, got {max_resolution}")
    if output_type not in ("entry", "polymer_entity"):
        raise ValueError(f"output_type must be 'entry' or 'polymer_entity', got {output_type}")

    sess = session or requests.Session()

    selected_clusters: Set[str] = set()
    polymer_entity_ids: List[str] = []
    entry_ids: List[str] = []

    polymer_entity_ids_set: Set[str] = set()
    entry_ids_set: Set[str] = set()

    start = 0
    pages = 0
    n_scanned = 0

    pbar = tqdm(total=target_n, desc=f"Selecting {output_type} IDs", disable=not verbose)

    while True:
        if max_pages is not None and pages >= max_pages:
            break

        batch_entries, next_start = search_entries_xray_resolution(
            max_resolution=max_resolution,
            start=start,
            page_size=page_size,
            methods=methods,
            session=sess,
        )

        if not batch_entries:
            break

        pages += 1
        n_scanned += len(batch_entries)

        # GraphQL polymer entities for the batch
        infos: List[PolymerEntityInfo] = graphql_polymer_entities_for_entries(
            batch_entries,
            session=sess,
            batch_size=graphql_batch_size,
        )

        # Group by entry for protein-only check
        entry_to_entities: dict[str, list[PolymerEntityInfo]] = defaultdict(list)
        for pe in infos:
            if pe.entry_id:
                entry_to_entities[pe.entry_id].append(pe)

        # Process each polymer entity; enforce protein-only + one-per-cluster70
        for entry_id, pe_list in entry_to_entities.items():
            if not is_protein_only_entry(pe_list):
                continue

            for pe in pe_list:
                if not pe.cluster_id_70:
                    continue
                if pe.cluster_id_70 in selected_clusters:
                    continue

                # select this cluster
                selected_clusters.add(pe.cluster_id_70)

                if pe.polymer_entity_id:
                    if pe.polymer_entity_id not in polymer_entity_ids_set:
                        polymer_entity_ids.append(pe.polymer_entity_id)
                        polymer_entity_ids_set.add(pe.polymer_entity_id)

                if pe.entry_id:
                    if pe.entry_id not in entry_ids_set:
                        entry_ids.append(pe.entry_id)
                        entry_ids_set.add(pe.entry_id)

                # update progress based on output type
                if output_type == "polymer_entity":
                    if polymer_entity_ids and polymer_entity_ids[-1] == pe.polymer_entity_id:
                        pbar.update(1)
                else:
                    if entry_ids and entry_ids[-1] == pe.entry_id:
                        pbar.update(1)

                # stop condition
                if output_type == "polymer_entity" and len(polymer_entity_ids) >= target_n:
                    break
                if output_type == "entry" and len(entry_ids) >= target_n:
                    break

            # break out of entry loop too
            if output_type == "polymer_entity" and len(polymer_entity_ids) >= target_n:
                break
            if output_type == "entry" and len(entry_ids) >= target_n:
                break

        if output_type == "polymer_entity" and len(polymer_entity_ids) >= target_n:
            break
        if output_type == "entry" and len(entry_ids) >= target_n:
            break

        # Advance to next page
        if next_start is None:
            break
        start = next_start

    pbar.close()

    output_ids = polymer_entity_ids[:target_n] if output_type == "polymer_entity" else entry_ids[:target_n]

    logger.info(f"Selected {len(output_ids)} {output_type} IDs from {len(selected_clusters)} clusters")
    logger.info(f"Scanned {n_scanned} candidate entries")

    return BuildResult(
        output_ids=output_ids,
        output_type=output_type,
        polymer_entity_ids_selected=polymer_entity_ids[:target_n]
        if output_type == "polymer_entity"
        else polymer_entity_ids,
        entry_ids_selected=entry_ids[:target_n] if output_type == "entry" else entry_ids,
        n_candidate_entries_scanned=n_scanned,
    )


# --------------------------------------------------------------------------------------
# Backwards-compatible wrappers (Option B)
# --------------------------------------------------------------------------------------


def build_pdb70_like_polymer_entities(
    target_n: int = 1000,
    max_resolution: float = 2.0,
    **kwargs,
) -> List[str]:
    """
    Backwards-compatible wrapper returning polymer_entity IDs.

    This exists because older code imported `build_pdb70_like_polymer_entities`.
    """
    res = build_pdb70_like(
        target_n=target_n,
        max_resolution=max_resolution,
        output_type="polymer_entity",
        **kwargs,
    )
    return res.output_ids


def build_pdb70_like_entries(
    target_n: int = 1000,
    max_resolution: float = 2.0,
    **kwargs,
) -> List[str]:
    """Convenience wrapper returning entry IDs."""
    res = build_pdb70_like(
        target_n=target_n,
        max_resolution=max_resolution,
        output_type="entry",
        **kwargs,
    )
    return res.output_ids
