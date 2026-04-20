# src/calphaebm/data/filter_test_set.py
"""
Filter test structures out of a training ID list.

Supports:
1) Entry-level blacklist (fast): exclude any entity whose ENTRY (e.g. 1ABC) is in blacklist.
2) Cluster70 blacklist (stricter): exclude any entity whose cluster_id_70 matches a test cluster.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

from calphaebm.data.rcsb_query import graphql_polymer_entities_for_entries
from calphaebm.utils.logging import get_logger

logger = get_logger()


DEFAULT_TEST_ENTRIES: List[Tuple[str, str]] = [
    ("1L2Y", "Trp-cage (TC5b)"),
    ("1YRF", "Villin headpiece (HP35)"),
    ("1T8J", "BBA5 (ββ-α)"),
    ("2GB1", "Protein G B1 domain (GB1)"),
    ("1UBQ", "Ubiquitin"),
    ("5PTI", "BPTI"),
    ("2CI2", "Chymotrypsin inhibitor 2 (CI2)"),
    ("1SRL", "Src SH3 domain"),
    ("1I6C", "Pin1 WW domain"),
    ("1ENH", "Engrailed homeodomain (EnHD)"),
]


def _read_nonempty_lines(path: Path) -> List[str]:
    out: List[str] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def _read_entry_ids(path: Path) -> List[str]:
    out: List[str] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = line.split()[0].upper()
            if len(entry) != 4:
                logger.warning(f"Skipping non-entry token '{entry}' in {path}")
                continue
            out.append(entry)
    return out


def _read_entity_ids(path: Path) -> List[str]:
    out: List[str] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for x in lines:
            f.write(str(x) + "\n")


def filter_entities_by_entry_blacklist(
    entity_ids: List[str],
    entry_blacklist: Set[str],
) -> Tuple[List[str], int]:
    kept: List[str] = []
    dropped = 0
    for ent in entity_ids:
        entry = ent.split("_")[0].upper()
        if entry in entry_blacklist:
            dropped += 1
            continue
        kept.append(ent)
    return kept, dropped


def build_cluster70_blacklist_from_entries(
    test_entries: List[str],
    session: Optional[requests.Session] = None,
    batch_size: int = 200,
    sleep_s: float = 0.0,
) -> Set[str]:
    sess = session or requests.Session()
    clusters: Set[str] = set()

    for i in range(0, len(test_entries), batch_size):
        batch = test_entries[i : i + batch_size]
        infos = graphql_polymer_entities_for_entries(batch, session=sess)
        for pe in infos:
            if pe.cluster_id_70:
                clusters.add(pe.cluster_id_70)
        if sleep_s > 0:
            time.sleep(sleep_s)

    return clusters


def map_entities_to_cluster70(
    entity_ids: List[str],
    session: Optional[requests.Session] = None,
    entry_batch_size: int = 200,
    sleep_s: float = 0.0,
) -> Dict[str, Optional[str]]:
    sess = session or requests.Session()

    entries = sorted({ent.split("_")[0].upper() for ent in entity_ids})
    mapping: Dict[str, Optional[str]] = {}

    for i in range(0, len(entries), entry_batch_size):
        batch_entries = entries[i : i + entry_batch_size]
        infos = graphql_polymer_entities_for_entries(batch_entries, session=sess)
        for pe in infos:
            if pe.polymer_entity_id:
                mapping[pe.polymer_entity_id.upper()] = pe.cluster_id_70
        if sleep_s > 0:
            time.sleep(sleep_s)

    for ent in entity_ids:
        mapping.setdefault(ent.upper(), None)

    return mapping


def filter_entities_by_cluster70_blacklist(
    entity_ids: List[str],
    bad_clusters: Set[str],
    entity_to_cluster70: Dict[str, Optional[str]],
    keep_if_missing: bool = True,
) -> Tuple[List[str], int, int]:
    kept: List[str] = []
    dropped = 0
    missing = 0

    for ent in entity_ids:
        cid = entity_to_cluster70.get(ent.upper())
        if cid is None:
            missing += 1
            if keep_if_missing:
                kept.append(ent)
            else:
                dropped += 1
            continue
        if cid in bad_clusters:
            dropped += 1
            continue
        kept.append(ent)

    return kept, dropped, missing


def write_default_test_blacklist_entries(out_path: Path) -> None:
    lines = ["# 10-test-set entry IDs (exclude from training)"]
    for eid, name in DEFAULT_TEST_ENTRIES:
        lines.append(f"{eid}  # {name}")
    _write_lines(out_path, lines)
    logger.info(f"Wrote default entry blacklist to {out_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calphaebm.data.filter_test_set",
        description="Filter test structures out of training ID lists.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s0 = sub.add_parser("write-default-blacklist", help="Write the default 10-test-set entry blacklist.")
    s0.add_argument("--out", type=Path, default=Path("test_blacklist_entries.txt"))

    s1 = sub.add_parser("filter-by-entry", help="Filter entity IDs by entry blacklist.")
    s1.add_argument("--entities", type=Path, required=True)
    s1.add_argument("--blacklist", type=Path, required=True)
    s1.add_argument("--out", type=Path, required=True)

    s2 = sub.add_parser("build-cluster70-blacklist", help="Build cluster70 blacklist from test entry IDs.")
    s2.add_argument("--entries", type=Path, required=True, help="File of test entry IDs (one per line).")
    s2.add_argument("--out", type=Path, required=True)
    s2.add_argument("--batch-size", type=int, default=200)
    s2.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between GraphQL batches (default: 0).")

    s3 = sub.add_parser("filter-by-cluster70", help="Filter entity IDs by cluster70 blacklist.")
    s3.add_argument("--entities", type=Path, required=True)
    s3.add_argument("--clusters", type=Path, required=True)
    s3.add_argument("--out", type=Path, required=True)
    s3.add_argument("--entry-batch-size", type=int, default=200)
    s3.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between GraphQL batches (default: 0).")
    s3.add_argument("--drop-if-missing", action="store_true", help="Drop entities if cluster70 cannot be resolved.")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "write-default-blacklist":
        write_default_test_blacklist_entries(args.out)
        return 0

    if args.cmd == "filter-by-entry":
        entities = _read_entity_ids(args.entities)
        entry_blacklist = set(_read_entry_ids(args.blacklist))
        kept, dropped = filter_entities_by_entry_blacklist(entities, entry_blacklist)
        _write_lines(args.out, kept)
        logger.info(f"Kept {len(kept)} / {len(entities)} entities; dropped {dropped} by entry blacklist.")
        return 0

    if args.cmd == "build-cluster70-blacklist":
        test_entries = _read_entry_ids(args.entries)
        sess = requests.Session()
        clusters = build_cluster70_blacklist_from_entries(
            test_entries=test_entries,
            session=sess,
            batch_size=args.batch_size,
            sleep_s=args.sleep,
        )
        _write_lines(args.out, sorted(clusters))
        logger.info(f"Wrote {len(clusters)} cluster70 IDs to {args.out}")
        return 0

    if args.cmd == "filter-by-cluster70":
        entities = _read_entity_ids(args.entities)
        clusters = set(_read_nonempty_lines(args.clusters))
        sess = requests.Session()
        entity_to_cluster = map_entities_to_cluster70(
            entity_ids=entities,
            session=sess,
            entry_batch_size=args.entry_batch_size,
            sleep_s=args.sleep,
        )
        kept, dropped, missing = filter_entities_by_cluster70_blacklist(
            entity_ids=entities,
            bad_clusters=clusters,
            entity_to_cluster70=entity_to_cluster,
            keep_if_missing=not args.drop_if_missing,
        )
        _write_lines(args.out, kept)
        logger.info(
            f"Kept {len(kept)} / {len(entities)} entities; dropped {dropped} by cluster70 blacklist; "
            f"missing cluster70 for {missing} entities."
        )
        return 0

    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
