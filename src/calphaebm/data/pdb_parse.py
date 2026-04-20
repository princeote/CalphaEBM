# src/calphaebm/data/pdb_parse.py
"""Download and parse PDB/mmCIF files to extract Cα chains and fixed-length training windows.

Terminology (this file is explicit about it):
- chain: a Cα trace for a single PDB chain (variable length)
- fragment: a contiguous chunk of a chain after splitting on large Cα gaps (variable length)
- window: a fixed-length training example produced by sliding a (seg_len, stride) window over a fragment
- dataset examples: the list returned by `load_pdb_segments` (each item is one window)

Quality filters added:
- B-factor extraction: per-residue Cα B-factors stored in ChainCA
- Missing residue detection: strict check via residue number continuity
- Monomeric filter helper: count protein chains per entry
"""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests
from Bio.PDB import MMCIFParser

from calphaebm.data.aa_map import aa3_to_idx
from calphaebm.utils.logging import get_logger

logger = get_logger()

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

# Geometry validation constants (Å)
MIN_BOND_LENGTH = 3.5  # absolute minimum for Cα-Cα
MAX_BOND_LENGTH = 4.1  # absolute maximum for Cα-Cα
IDEAL_BOND_LENGTH = 3.8
MAX_BOND_DEVIATION = 0.3  # warning-only mean deviation threshold
MIN_FRAGMENT_LENGTH = 8  # minimum residues for a valid fragment/window
MAX_CONSECUTIVE_GAP = 0.5  # max allowed change between consecutive bonds (Å)

# Parser-side cleanup
DROP_NEAR_DUPLICATE_CA = True
NEAR_DUPLICATE_CA_EPS = 0.50  # Å; if consecutive CA points are closer than this, drop the later point

# Download robustness
DEFAULT_DOWNLOAD_TIMEOUT_S = 30
DEFAULT_DOWNLOAD_RETRIES = 3
DEFAULT_RETRY_BACKOFF_S = 1.5

# B-factor quality thresholds
DEFAULT_MAX_MEAN_BFACTOR = 40.0  # Å²; chains with higher mean B are poorly determined
DEFAULT_MAX_RESIDUE_BFACTOR = 80.0  # Å²; individual residues above this are disordered
DEFAULT_MAX_HIGH_B_FRAC = 0.10  # reject if >10% of residues have B > MAX_RESIDUE_BFACTOR


@dataclass
class ChainCA:
    """Cα trace of a single chain with quality metadata."""

    pdb_id: str
    chain_id: str
    coords: np.ndarray  # (L, 3) float32
    seq: np.ndarray  # (L,) int64 (amino acid indices)
    bfactors: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))  # (L,) float32
    resseqs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # (L,) residue numbers

    def __len__(self) -> int:
        return len(self.coords)

    @property
    def mean_bfactor(self) -> float:
        """Mean B-factor across all Cα atoms."""
        if len(self.bfactors) == 0:
            return 0.0
        return float(np.mean(self.bfactors))

    @property
    def max_bfactor(self) -> float:
        """Maximum B-factor."""
        if len(self.bfactors) == 0:
            return 0.0
        return float(np.max(self.bfactors))

    @property
    def high_b_fraction(self) -> float:
        """Fraction of residues with B-factor > DEFAULT_MAX_RESIDUE_BFACTOR."""
        if len(self.bfactors) == 0:
            return 0.0
        return float(np.mean(self.bfactors > DEFAULT_MAX_RESIDUE_BFACTOR))

    @property
    def has_missing_residues(self) -> bool:
        """Check for gaps in residue numbering (missing residues)."""
        if len(self.resseqs) < 2:
            return False
        diffs = np.diff(self.resseqs)
        # All consecutive residue number differences should be 1
        # Allow 0 (insertion codes) but not >1 (missing residues)
        return bool(np.any(diffs > 1))

    @property
    def n_missing_residues(self) -> int:
        """Count total missing residues from sequence gaps."""
        if len(self.resseqs) < 2:
            return 0
        diffs = np.diff(self.resseqs)
        gaps = diffs[diffs > 1] - 1  # each gap of N means N-1 missing
        return int(gaps.sum()) if len(gaps) > 0 else 0


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Write bytes to path atomically."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)


def download_cif(
    pdb_id: str,
    cache_dir: str,
    force: bool = False,
    session: Optional[requests.Session] = None,
    timeout_s: int = DEFAULT_DOWNLOAD_TIMEOUT_S,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
) -> str:
    """Download mmCIF file to cache directory."""
    pid = pdb_id.lower()
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{pid}.cif")

    if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    url = RCSB_CIF_URL.format(pdb_id=pid.upper())
    sess = session or requests.Session()

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading {pid} from RCSB (attempt {attempt}/{retries})...")
            resp = sess.get(url, timeout=timeout_s)
            resp.raise_for_status()
            _atomic_write_bytes(out_path, resp.content)
            logger.debug(f"Downloaded {pid} to {out_path}")
            return out_path
        except (requests.RequestException, OSError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(DEFAULT_RETRY_BACKOFF_S * attempt)
            continue

    assert last_err is not None
    logger.error(f"Failed to download {pid} after {retries} attempts: {last_err}")
    raise last_err


def _select_ca_coord_and_bfactor(res) -> Tuple[Optional[np.ndarray], float]:
    """Select a CA coordinate and B-factor from a Biopython residue robustly.

    Returns (coord, bfactor) or (None, 0.0) if CA not found.
    """
    if "CA" not in res:
        return None, 0.0

    ca = res["CA"]

    # Disordered atom handling
    if getattr(ca, "is_disordered", lambda: 0)():
        try:
            if hasattr(ca, "child_dict") and "A" in ca.child_dict:
                ca = ca.child_dict["A"]
            else:
                best = None
                best_occ = -1.0
                for alt in getattr(ca, "child_dict", {}).values():
                    occ = alt.get_occupancy()
                    if occ is None:
                        occ = 0.0
                    if occ > best_occ:
                        best_occ = occ
                        best = alt
                if best is not None:
                    ca = best
        except Exception:
            pass

    coord = ca.get_coord()
    if coord is None:
        return None, 0.0
    coord = np.asarray(coord, dtype=np.float32)

    if not np.isfinite(coord).all():
        return None, 0.0

    # Extract B-factor
    try:
        bfactor = float(ca.get_bfactor())
    except Exception:
        bfactor = 0.0

    return coord, bfactor


# Keep backward-compatible wrapper
def _select_ca_coord(res) -> Optional[np.ndarray]:
    """Select a CA coordinate from a Biopython residue robustly."""
    coord, _ = _select_ca_coord_and_bfactor(res)
    return coord


def parse_cif_ca_chains(
    cif_path: str,
    pdb_id: str,
    drop_near_duplicates: bool = DROP_NEAR_DUPLICATE_CA,
    near_duplicate_eps: float = NEAR_DUPLICATE_CA_EPS,
) -> List[ChainCA]:
    """Parse mmCIF and extract Cα traces with B-factors for each chain."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, cif_path)

    chains: List[ChainCA] = []
    model = next(structure.get_models())  # first model only

    for chain in model.get_chains():
        coords: List[np.ndarray] = []
        seq: List[int] = []
        bfactors: List[float] = []
        resseqs: List[int] = []

        seen_res_ids = set()

        for res in chain.get_residues():
            resname = res.get_resname()
            aa_idx = aa3_to_idx(resname)
            if aa_idx is None:
                continue

            # Deduplicate by (resseq, icode)
            try:
                rid = res.get_id()
                key = (rid[1], rid[2])
                if key in seen_res_ids:
                    continue
                seen_res_ids.add(key)
                resseq = rid[1]  # residue sequence number
            except Exception:
                resseq = len(coords) + 1  # fallback

            ca_coord, bfactor = _select_ca_coord_and_bfactor(res)
            if ca_coord is None:
                continue

            # Near-duplicate suppression
            if drop_near_duplicates and coords:
                d = float(np.linalg.norm(ca_coord - coords[-1]))
                if d < near_duplicate_eps:
                    continue

            coords.append(ca_coord.astype(np.float32))
            seq.append(int(aa_idx))
            bfactors.append(bfactor)
            resseqs.append(resseq)

        if len(coords) < 4:
            continue

        chains.append(
            ChainCA(
                pdb_id=pdb_id.lower(),
                chain_id=str(chain.id),
                coords=np.stack(coords, axis=0),
                seq=np.array(seq, dtype=np.int64),
                bfactors=np.array(bfactors, dtype=np.float32),
                resseqs=np.array(resseqs, dtype=np.int64),
            )
        )
        logger.debug(f"Found chain {chain.id} with {len(coords)} residues, " f"mean_B={np.mean(bfactors):.1f}")

    return chains


def count_protein_chains(cif_path: str, pdb_id: str) -> int:
    """Count the number of protein chains in an mmCIF file.

    Used for monomeric filter: entries with >1 protein chain are complexes.
    """
    chains = parse_cif_ca_chains(cif_path, pdb_id)
    # Count chains with at least 20 residues (ignore short peptide ligands)
    return sum(1 for c in chains if len(c) >= 10)


def is_complete_chain(chain: ChainCA) -> bool:
    """Check if chain has no missing residues (strict continuity)."""
    return not chain.has_missing_residues


def passes_bfactor_filter(
    chain: ChainCA,
    max_mean_b: float = DEFAULT_MAX_MEAN_BFACTOR,
    max_residue_b: float = DEFAULT_MAX_RESIDUE_BFACTOR,
    max_high_b_frac: float = DEFAULT_MAX_HIGH_B_FRAC,
) -> Tuple[bool, str]:
    """Check B-factor quality criteria.

    Returns (passed, reason) where reason explains rejection.
    """
    if len(chain.bfactors) == 0:
        return True, "no B-factors available"

    mean_b = chain.mean_bfactor
    if mean_b > max_mean_b:
        return False, f"mean B-factor {mean_b:.1f} > {max_mean_b}"

    high_frac = chain.high_b_fraction
    if high_frac > max_high_b_frac:
        return False, f"high-B fraction {high_frac:.1%} > {max_high_b_frac:.0%}"

    return True, "ok"


def validate_ca_geometry(
    coords: np.ndarray,
    seq: np.ndarray,
    stats: Optional[Counter] = None,
) -> Tuple[bool, str]:
    """Validate a Cα trace (fragment or window) for physical plausibility."""
    if len(coords) < MIN_FRAGMENT_LENGTH:
        if stats is not None:
            stats["too_short"] += 1
        return False, f"Too short: {len(coords)} < {MIN_FRAGMENT_LENGTH}"

    diffs = coords[1:] - coords[:-1]
    bond_lengths = np.sqrt((diffs * diffs).sum(axis=1))

    # Chain break detection
    if np.any(bond_lengths > MAX_BOND_LENGTH * 1.5):
        if stats is not None:
            stats["chain_break"] += 1
        max_gap = float(np.max(bond_lengths))
        return False, f"Chain break: max bond = {max_gap:.3f} Å"

    # Per-bond bounds
    for i, bl in enumerate(bond_lengths):
        bl = float(bl)
        if bl < MIN_BOND_LENGTH:
            if stats is not None:
                stats["bond_too_short"] += 1
            return False, f"Bond too short at {i}: {bl:.3f} Å < {MIN_BOND_LENGTH}"
        if bl > MAX_BOND_LENGTH:
            if stats is not None:
                stats["bond_too_long"] += 1
            return False, f"Bond too long at {i}: {bl:.3f} Å > {MAX_BOND_LENGTH}"

    # Sudden change detection
    if len(bond_lengths) > 2:
        bond_diffs = np.abs(np.diff(bond_lengths))
        if np.any(bond_diffs > MAX_CONSECUTIVE_GAP):
            if stats is not None:
                stats["sudden_change"] += 1
            max_diff = float(np.max(bond_diffs))
            return False, f"Sudden bond change: {max_diff:.3f} Å > {MAX_CONSECUTIVE_GAP}"

    # Warning-only deviation
    mean_dev = float(np.mean(np.abs(bond_lengths - IDEAL_BOND_LENGTH)))
    if mean_dev > MAX_BOND_DEVIATION:
        logger.debug(f"High mean bond deviation: {mean_dev:.3f} Å")

    if stats is not None:
        stats["valid"] += 1
    return True, "Valid"


def _print_validation_stats(title: str, stats: Counter) -> None:
    """Pretty-print validation/rejection statistics for one stage."""
    total = sum(stats.values())
    if total == 0:
        return

    print(f"\n=== {title} ===")
    print(f"Total validated: {total}")
    v = stats.get("valid", 0)
    print(f"Valid: {v} ({(v/total*100):.1f}%)")

    print("\nRejection reasons:")
    for reason in ["too_short", "chain_break", "bond_too_short", "bond_too_long", "sudden_change"]:
        count = stats.get(reason, 0)
        if count > 0:
            print(f"  {reason:20s}: {count:5d} ({count/total*100:.1f}%)")


def split_chain_on_gaps(
    coords: np.ndarray,
    seq: np.ndarray,
    max_ca_jump: float = 4.5,
    validate_fragments: bool = True,
    fragment_stats: Optional[Counter] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a chain at large Cα-Cα jumps (missing residues), producing contiguous fragments."""
    assert coords.ndim == 2 and coords.shape[1] == 3
    if coords.shape[0] <= 1:
        return []

    diffs = coords[1:] - coords[:-1]
    d = np.sqrt((diffs * diffs).sum(axis=1))

    breaks = np.where(d > max_ca_jump)[0] + 1

    fragments: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for end in breaks:
        if end - start >= MIN_FRAGMENT_LENGTH:
            frag_coords = coords[start:end]
            frag_seq = seq[start:end]
            if validate_fragments:
                ok, reason = validate_ca_geometry(frag_coords, frag_seq, stats=fragment_stats)
                if ok:
                    fragments.append((frag_coords, frag_seq))
                else:
                    logger.debug(f"Rejected fragment: {reason}")
            else:
                fragments.append((frag_coords, frag_seq))
        start = end

    # Last fragment
    if coords.shape[0] - start >= MIN_FRAGMENT_LENGTH:
        frag_coords = coords[start:]
        frag_seq = seq[start:]
        if validate_fragments:
            ok, reason = validate_ca_geometry(frag_coords, frag_seq, stats=fragment_stats)
            if ok:
                fragments.append((frag_coords, frag_seq))
            else:
                logger.debug(f"Rejected fragment: {reason}")
        else:
            fragments.append((frag_coords, frag_seq))

    return fragments


def iter_fixed_windows(
    coords: np.ndarray,
    seq: np.ndarray,
    window_len: int = 128,
    stride: int = 64,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield fixed-length windows from a contiguous fragment."""
    L = coords.shape[0]
    if L < window_len:
        return
    for start in range(0, L - window_len + 1, stride):
        end = start + window_len
        yield coords[start:end].copy(), seq[start:end].copy()


def load_pdb_segments(
    pdb_ids: List[str],
    cache_dir: str,
    seg_len: int = 128,
    stride: int = 64,
    max_ca_jump: float = 4.5,
    limit_segments: Optional[int] = None,
    validate_geometry: bool = True,
    reset_stats: bool = True,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """Download and parse PDBs, returning fixed-length windows (dataset examples)."""
    fragment_stats = Counter()
    window_stats = Counter()

    chains_seen = 0
    fragments_produced = 0
    windows_enumerated = 0
    windows_kept = 0

    windows: List[Dict[str, Any]] = []
    sess = session or requests.Session()

    for pid in pdb_ids:
        try:
            cif_path = download_cif(pid, cache_dir=cache_dir, session=sess)
            chains = parse_cif_ca_chains(cif_path, pdb_id=pid.lower())

            for chain in chains:
                chains_seen += 1

                fragments = split_chain_on_gaps(
                    chain.coords,
                    chain.seq,
                    max_ca_jump=max_ca_jump,
                    validate_fragments=validate_geometry,
                    fragment_stats=fragment_stats if validate_geometry else None,
                )
                fragments_produced += len(fragments)

                for frag_coords, frag_seq in fragments:
                    for cw, sw in iter_fixed_windows(frag_coords, frag_seq, window_len=seg_len, stride=stride):
                        windows_enumerated += 1

                        if validate_geometry:
                            ok, _ = validate_ca_geometry(cw, sw, stats=window_stats)
                            if not ok:
                                continue

                        windows.append(
                            {
                                "pdb_id": chain.pdb_id,
                                "chain_id": chain.chain_id,
                                "coords": cw.astype(np.float32),
                                "seq": sw.astype(np.int64),
                            }
                        )
                        windows_kept += 1

                        if limit_segments and windows_kept >= limit_segments:
                            logger.info(f"Reached dataset window limit: {limit_segments}")
                            logger.info(
                                "Summary so far: chains=%d, fragments=%d, windows_enumerated=%d, windows_kept=%d",
                                chains_seen,
                                fragments_produced,
                                windows_enumerated,
                                windows_kept,
                            )
                            if validate_geometry:
                                _print_validation_stats("Fragment geometry validation", fragment_stats)
                                _print_validation_stats("Window geometry validation", window_stats)
                            return windows

        except requests.RequestException as e:
            logger.warning(f"Network error processing {pid}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Failed to process {pid}: {e}")
            continue

    logger.info(
        "Dataset build complete: chains=%d, fragments=%d, windows_enumerated=%d, windows_kept=%d",
        chains_seen,
        fragments_produced,
        windows_enumerated,
        windows_kept,
    )
    if validate_geometry:
        _print_validation_stats("Fragment geometry validation", fragment_stats)
        _print_validation_stats("Window geometry validation", window_stats)

    return windows


def get_residue_sequence(pdb_id: str, cache_dir: str, chain_id: Optional[str] = None) -> List[str]:
    """Get 1-letter amino acid sequence for a PDB chain."""
    from calphaebm.data.aa_map import idx_to_aa1

    cif_path = download_cif(pdb_id, cache_dir=cache_dir)
    chains = parse_cif_ca_chains(cif_path, pdb_id)

    if not chains:
        raise ValueError(f"No chains found for {pdb_id}")

    if chain_id:
        chain = next((c for c in chains if c.chain_id == chain_id), None)
        if chain is None:
            raise ValueError(f"Chain {chain_id} not found in {pdb_id}")
    else:
        chain = chains[0]
        logger.info(f"Using chain {chain.chain_id} for {pdb_id}")

    return [idx_to_aa1(idx) for idx in chain.seq]
