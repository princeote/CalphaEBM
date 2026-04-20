# src/calphaebm/data/rcsb_query.py

"""RCSB Search/Data API helpers to build nonredundant datasets.

Correctness fixes:
- Use valid Search API terminal 'service' values (RCSB Search v2 expects 'text' for metadata terminals).
- Handle GraphQL resolution_combined being a list (take min).
- Handle cluster membership identity possibly being float/int/str.
- More robust protein-type checks (Protein vs Polypeptide(L), etc.).
- Add lightweight retry/backoff for network resilience.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import requests

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"

_DEFAULT_TIMEOUT_S = 60
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF_S = 1.5


@dataclass(frozen=True)
class PolymerEntityInfo:
    """Information about a polymer entity from RCSB."""

    polymer_entity_id: str  # e.g., "1ABC_1"
    entry_id: str  # e.g., "1ABC"
    polymer_type: Optional[str]  # "Protein", "Polypeptide(L)", etc.
    cluster_id_70: Optional[str]  # 70% sequence identity cluster ID
    resolution: Optional[float]  # Best resolution in Å (lower is better)


def _post_json(
    url: str,
    payload: dict,
    session: requests.Session,
    timeout: int = _DEFAULT_TIMEOUT_S,
    retries: int = _DEFAULT_RETRIES,
) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.post(url, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                # Include response text to make debugging malformed queries easier
                raise requests.HTTPError(
                    f"{resp.status_code} {resp.reason} for url {resp.url}\nResponse: {resp.text[:2000]}",
                    response=resp,
                )
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(_DEFAULT_BACKOFF_S * attempt)
            else:
                raise
    assert last_err is not None
    raise last_err


def _min_resolution(res_val: Any) -> Optional[float]:
    """RCSB sometimes returns resolution_combined as list; normalize to min float."""
    if res_val is None:
        return None
    if isinstance(res_val, (int, float)):
        return float(res_val)
    if isinstance(res_val, list) and res_val:
        vals = [float(x) for x in res_val if x is not None]
        return min(vals) if vals else None
    return None


def _identity_is_70(x: Any) -> bool:
    """Membership identity may be int/float/str."""
    if x is None:
        return False
    try:
        return abs(float(x) - 70.0) < 1e-6
    except Exception:
        return False


def search_entries_xray_resolution(
    max_resolution: float = 2.0,
    page_size: int = 1000,
    start: int = 0,
    methods: Optional[List[str]] = None,
    session: Optional[requests.Session] = None,
) -> Tuple[List[str], Optional[int]]:
    """Search for entries with resolution <= max_resolution.

    Args:
        max_resolution: Resolution cutoff in Å.
        page_size: Number of results per page.
        start: Pagination offset.
        methods: Experimental methods to include. Default: X-ray only.
            Pass ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"] for both.
        session: Optional requests session.

    Returns:
        (entry_ids, next_start) where next_start is None if no more results.
    """
    sess = session or requests.Session()
    if methods is None:
        methods = ["X-RAY DIFFRACTION"]

    # Build method filter — single method uses exact_match, multiple uses "in"
    if len(methods) == 1:
        method_node = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": methods[0],
            },
        }
    else:
        method_node = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "in",
                "value": methods,
            },
        }

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                method_node,
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": float(max_resolution),
                    },
                },
            ],
        },
        "request_options": {
            "paginate": {"start": int(start), "rows": int(page_size)},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
        "return_type": "entry",
    }

    data = _post_json(SEARCH_URL, query, sess)

    ids = [r["identifier"] for r in data.get("result_set", []) if "identifier" in r]
    total = data.get("total_count")

    next_start = start + page_size
    if total is None or next_start >= total:
        next_start = None

    return ids, next_start


def graphql_polymer_entities_for_entries(
    entry_ids: List[str],
    session: Optional[requests.Session] = None,
    batch_size: int = 200,
    retries: int = _DEFAULT_RETRIES,
) -> List[PolymerEntityInfo]:
    """Fetch polymer entity information for entries (GraphQL).

    Args:
        entry_ids: List of PDB entry IDs.
        session: Optional requests session.
        batch_size: Number of entries per GraphQL request.
        retries: Number of retry attempts for network errors.
    """
    if not entry_ids:
        return []

    sess = session or requests.Session()

    query = """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        rcsb_entry_info {
          resolution_combined
        }
        polymer_entities {
          rcsb_id
          entity_poly {
            rcsb_entity_polymer_type
          }
          rcsb_cluster_membership {
            cluster_id
            identity
          }
        }
      }
    }
    """

    # Process in batches to avoid GraphQL payload limits
    out: List[PolymerEntityInfo] = []
    for i in range(0, len(entry_ids), batch_size):
        batch = entry_ids[i : i + batch_size]
        variables = {"ids": [eid.upper() for eid in batch]}

        payload = _post_json(
            GRAPHQL_URL,
            {"query": query, "variables": variables},
            sess,
            retries=retries,
        )

        if "errors" in payload:
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")

        entries = payload.get("data", {}).get("entries", []) or []

        for ent in entries:
            entry_id = ent.get("rcsb_id")
            res = _min_resolution(ent.get("rcsb_entry_info", {}).get("resolution_combined"))

            for pe in ent.get("polymer_entities") or []:
                pe_id = pe.get("rcsb_id")
                poly_type = pe.get("entity_poly", {}).get("rcsb_entity_polymer_type")

                cluster70 = None
                for cm in pe.get("rcsb_cluster_membership") or []:
                    if _identity_is_70(cm.get("identity")):
                        cluster70 = cm.get("cluster_id")
                        break

                if pe_id and entry_id:
                    out.append(
                        PolymerEntityInfo(
                            polymer_entity_id=pe_id,
                            entry_id=entry_id,
                            polymer_type=poly_type,
                            cluster_id_70=cluster70,
                            resolution=res,
                        )
                    )

    return out


def _is_protein_polymer_type(polymer_type: Optional[str]) -> bool:
    """RCSB polymer type varies; accept common protein-like labels."""
    if not polymer_type:
        return False
    t = polymer_type.strip().lower()
    return ("protein" in t) or ("polypeptide" in t)


def is_protein_only_entry(pe_list: List[PolymerEntityInfo]) -> bool:
    """Check if all polymer entities in an entry are protein-like."""
    if not pe_list:
        return False
    return all(_is_protein_polymer_type(pe.polymer_type) for pe in pe_list)


# ---------------------------------------------------------------------------
# Biological assembly (multimer) queries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssemblyInfo:
    """Biological assembly information for a PDB entry."""

    entry_id: str
    oligomeric_state: str  # "Monomer", "Homo 2-mer", "Hetero 4-mer", etc.
    oligomeric_count: int  # 1 for monomer, 2 for dimer, etc.
    stoichiometry: List[str]  # e.g. ["A2"] for homodimer
    symmetry: str  # e.g. "C2", "C1", "D2"
    kind: str  # "Global Symmetry", etc.
    is_monomer: bool  # convenience flag


def get_assembly_info_batch(
    entry_ids: List[str],
    session: Optional[requests.Session] = None,
    batch_size: int = 50,
    retries: int = _DEFAULT_RETRIES,
    delay: float = 0.05,
) -> dict[str, AssemblyInfo]:
    """Query RCSB REST API for biological assembly info.

    Uses the /rest/v1/core/assembly/{pdb_id}/1 endpoint for each entry.
    Returns a dict mapping entry_id -> AssemblyInfo.

    Note: This makes one HTTP request per entry (no batch GraphQL endpoint
    for assembly data). Use delay to avoid rate limiting.
    """
    sess = session or requests.Session()
    results: dict[str, AssemblyInfo] = {}

    for entry_id in entry_ids:
        try:
            url = f"https://data.rcsb.org/rest/v1/core/assembly/{entry_id.upper()}/1"
            resp = sess.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()

            # Extract assembly details
            info = data.get("rcsb_assembly_info", {})
            polymer_count = info.get("polymer_entity_instance_count", 1)

            # Symmetry info
            symmetry_list = data.get("rcsb_struct_symmetry", [])
            if symmetry_list:
                sym = symmetry_list[0]
                oligo_state = sym.get("oligomeric_state", "unknown")
                stoichiometry = sym.get("stoichiometry", [])
                symbol = sym.get("symbol", "C1")
                kind = sym.get("kind", "unknown")
            else:
                oligo_state = "Monomer" if polymer_count <= 1 else f"Homo {polymer_count}-mer"
                stoichiometry = ["A1"] if polymer_count <= 1 else [f"A{polymer_count}"]
                symbol = "C1"
                kind = "unknown"

            is_mono = oligo_state == "Monomer"

            results[entry_id.lower()] = AssemblyInfo(
                entry_id=entry_id.lower(),
                oligomeric_state=oligo_state,
                oligomeric_count=polymer_count,
                stoichiometry=stoichiometry if isinstance(stoichiometry, list) else [stoichiometry],
                symmetry=symbol,
                kind=kind,
                is_monomer=is_mono,
            )

        except Exception:
            pass  # skip failures silently

        if delay > 0:
            import time as _time

            _time.sleep(delay)

    return results


def filter_multimers(
    entry_ids: List[str],
    session: Optional[requests.Session] = None,
    delay: float = 0.05,
    verbose: bool = True,
) -> tuple[List[str], List[str], dict[str, AssemblyInfo]]:
    """Filter out biological multimers from a list of PDB entry IDs.

    Args:
        entry_ids: List of PDB entry IDs.
        session: Optional requests session.
        delay: Delay between API calls (seconds).
        verbose: Log progress.

    Returns:
        (monomer_ids, multimer_ids, assembly_info_dict)
    """
    from calphaebm.utils.logging import get_logger

    _logger = get_logger()

    sess = session or requests.Session()
    all_info: dict[str, AssemblyInfo] = {}

    if verbose:
        _logger.info("Checking biological assembly for %d entries...", len(entry_ids))

    # Process entries
    for i, eid in enumerate(entry_ids):
        info = get_assembly_info_batch([eid], session=sess, delay=delay)
        all_info.update(info)

        if verbose and (i + 1) % 500 == 0:
            n_multi = sum(1 for v in all_info.values() if not v.is_monomer)
            _logger.info("  %d/%d checked (%d multimers found)", i + 1, len(entry_ids), n_multi)

    monomers = []
    multimers = []
    for eid in entry_ids:
        info = all_info.get(eid.lower())
        if info is None:
            monomers.append(eid)  # keep if API failed
        elif info.is_monomer:
            monomers.append(eid)
        else:
            multimers.append(eid)

    if verbose:
        from collections import Counter

        states = Counter(v.oligomeric_state for v in all_info.values() if not v.is_monomer)
        _logger.info(
            "Assembly filter: %d monomers, %d multimers, %d unknown",
            len(monomers),
            len(multimers),
            len(entry_ids) - len(monomers) - len(multimers),
        )
        if states:
            _logger.info("  Multimer breakdown: %s", ", ".join(f"{k}: {v}" for k, v in states.most_common()))

    return monomers, multimers, all_info
