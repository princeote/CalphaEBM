from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from calphaebm.data.aa_map import aa3_to_idx, idx_to_aa1  # FIXED: changed idx_to_aa3 to idx_to_aa1
from calphaebm.data.id_utils import normalize_to_entry_ids
from calphaebm.data.pdb_parse import download_cif, parse_cif_ca_chains, split_chain_on_gaps
from calphaebm.utils.logging import get_logger

from .config import (
    CONTACT_BINS,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PDB_LIST,
    EMPIRICAL_COUNT_THRESHOLD,
    EMPIRICAL_P_FOR_LOW_COUNTS,
    ENRICH_SHUFFLES,
    FDR_Q,
    MAX_DIST_A,
    MAX_PAIRS_PER_SEGMENT,
    MIN_SEG_LEN,
    MIN_SEQ_SEP,
    PAIR_SAMPLE_SEED,
    PLOT_MAX_POINTS,
    RDF_N_BINS,
    RDF_R_MAX_A,
    RDF_R_MIN_A,
    RDF_TAIL_END_A,
    RDF_TAIL_START_A,
)
from .enrichment import compute_contact_enrichment
from .plots import plot_enrichment_matrices, plot_rdf_analysis, plot_repulsive_wall
from .rdf import compute_rdf_from_counts, densify_wall, extract_repulsive_wall, save_repulsive_wall

logger = get_logger()


@dataclass
class LoadStats:
    n_pdbs_attempted: int = 0
    n_pdbs_failed: int = 0
    n_chains_processed: int = 0
    n_segments_processed: int = 0
    n_pairs_total: int = 0
    n_pairs_used_for_rdf: int = 0
    n_pairs_used_for_contacts: int = 0


def _read_ids(path: Path) -> List[str]:
    with open(path, "r") as f:
        raw = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return normalize_to_entry_ids(raw)


def _convert_seq_to_indices(seq_data: Union[List, np.ndarray, tuple]) -> np.ndarray:
    """
    Convert sequence data (strings or indices) to integer indices.

    Handles:
    - List of 3-letter AA codes (e.g., ["ALA", "GLY", ...])
    - List of 1-letter codes (e.g., ["A", "G", ...])
    - List or array of integer indices (0-19)
    - Mixed types

    Returns:
        np.ndarray of shape (L,) with integer indices (0-19)
    """
    seq_idx = []

    for item in seq_data:
        # Handle different input types
        if isinstance(item, (np.integer, int, np.int32, np.int64)):
            # Already an integer index
            idx = int(item)
            if idx < 0 or idx >= 20:
                logger.warning(f"Integer index {idx} out of range [0,19], mapping to 0")
                idx = 0
            seq_idx.append(idx)

        elif isinstance(item, str):
            # String - could be 3-letter or 1-letter code
            item_clean = item.strip().upper()

            # Try as 3-letter code first
            idx = aa3_to_idx(item_clean)
            if idx is not None:
                seq_idx.append(idx)
            else:
                # Try as 1-letter code
                from calphaebm.data.aa_map import aa1_to_idx

                idx = aa1_to_idx(item_clean)
                if idx is not None:
                    seq_idx.append(idx)
                else:
                    logger.warning(f"Unknown amino acid code '{item_clean}', using 0")
                    seq_idx.append(0)

        elif isinstance(item, (bytes, bytearray)):
            # Handle bytes
            try:
                item_str = item.decode("utf-8").strip().upper()
                idx = aa3_to_idx(item_str)
                if idx is not None:
                    seq_idx.append(idx)
                else:
                    idx = aa1_to_idx(item_str)
                    if idx is not None:
                        seq_idx.append(idx)
                    else:
                        seq_idx.append(0)
            except:
                seq_idx.append(0)

        else:
            # Unknown type - try to convert to string
            try:
                item_str = str(item).strip().upper()
                idx = aa3_to_idx(item_str)
                if idx is not None:
                    seq_idx.append(idx)
                else:
                    idx = aa1_to_idx(item_str)
                    if idx is not None:
                        seq_idx.append(idx)
                    else:
                        seq_idx.append(0)
            except:
                logger.warning(f"Unknown sequence item type {type(item)}, using 0")
                seq_idx.append(0)

    return np.array(seq_idx, dtype=np.int64)


def _accumulate_pairs_for_segment(
    coords: np.ndarray,  # (L,3) float
    seq_data: Any,  # sequence data in various formats
    rng: np.random.Generator,
    rdf_counts: np.ndarray,
    rdf_edges: np.ndarray,
    contact_counts: Dict[str, np.ndarray],
    max_dist: float,
    min_seq_sep: int,
    max_pairs_per_segment: int,
    stats: LoadStats,
) -> None:
    L = int(coords.shape[0])
    if L < int(MIN_SEG_LEN):
        return

    # Convert sequence to indices robustly
    seq_idx = _convert_seq_to_indices(seq_data)

    # Verify length matches
    if len(seq_idx) != L:
        logger.warning(f"Sequence length mismatch: {len(seq_idx)} vs coordinates {L}. Truncating.")
        min_len = min(len(seq_idx), L)
        seq_idx = seq_idx[:min_len]
        coords = coords[:min_len]
        L = min_len

    def bin_rdf(d: float) -> None:
        if d < float(rdf_edges[0]) or d >= float(rdf_edges[-1]):
            return
        k = int(np.searchsorted(rdf_edges, d, side="right") - 1)
        if 0 <= k < rdf_counts.shape[0]:
            rdf_counts[k] += 1.0

    def bin_contact(d: float, ai: int, aj: int) -> None:
        if d >= float(max_dist):
            return
        for name, b in CONTACT_BINS.items():
            if float(b["min"]) <= d < float(b["max"]):
                contact_counts[name][ai, aj] += 1.0
                contact_counts[name][aj, ai] += 1.0
                return

    # enumerate if small, rejection sample if large
    if L <= 500:
        for i in range(L):
            for j in range(i + 1, L):
                if (j - i) <= int(min_seq_sep):
                    continue
                d = float(np.linalg.norm(coords[i] - coords[j]))
                if d >= float(max_dist):
                    continue
                ai = int(seq_idx[i])
                aj = int(seq_idx[j])
                bin_rdf(d)
                bin_contact(d, ai, aj)
                stats.n_pairs_total += 1
                stats.n_pairs_used_for_rdf += 1
                stats.n_pairs_used_for_contacts += 1
        return

    target = int(max_pairs_per_segment)
    attempts = 0
    max_attempts = target * 50
    accepted = 0

    while accepted < target and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, L))
        j = int(rng.integers(0, L))
        if i >= j:
            continue
        if (j - i) <= int(min_seq_sep):
            continue
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if d >= float(max_dist):
            continue
        ai = int(seq_idx[i])
        aj = int(seq_idx[j])
        bin_rdf(d)
        bin_contact(d, ai, aj)
        accepted += 1
        stats.n_pairs_total += 1
        stats.n_pairs_used_for_rdf += 1
        stats.n_pairs_used_for_contacts += 1

    if accepted < max(1000, target // 10):
        logger.warning(
            f"Low acceptance in rejection sampling (accepted {accepted}/{target}) "
            f"for segment length L={L}. Consider increasing MAX_DIST_A if needed."
        )


class RepulsionAnalyzer:
    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.stats = LoadStats()

    def run(
        self,
        pdb_list: Path = DEFAULT_PDB_LIST,
        max_pdbs: Optional[int] = None,
        quiet: bool = False,
        plot_max_points: int = PLOT_MAX_POINTS,
    ) -> int:
        entry_ids = _read_ids(Path(pdb_list))
        if max_pdbs is not None:
            entry_ids = entry_ids[: int(max_pdbs)]

        logger.info(f"Loaded {len(entry_ids)} entry IDs (normalized) from {pdb_list}")

        rdf_edges = np.linspace(RDF_R_MIN_A, RDF_R_MAX_A, RDF_N_BINS + 1, dtype=np.float64)
        rdf_counts = np.zeros(RDF_N_BINS, dtype=np.float64)

        contact_counts: Dict[str, np.ndarray] = {
            name: np.zeros((20, 20), dtype=np.float64) for name in CONTACT_BINS.keys()
        }

        pbar = tqdm(entry_ids, desc="Processing PDBs", disable=quiet)
        for eid in pbar:
            self.stats.n_pdbs_attempted += 1
            try:
                cif = download_cif(eid.lower(), str(self.cache_dir))
                chains = parse_cif_ca_chains(cif, eid.lower())
            except Exception as e:
                self.stats.n_pdbs_failed += 1
                pbar.write(f"Failed {eid}: {e}")
                continue

            seed = (PAIR_SAMPLE_SEED * 1_000_003 + (hash(eid) % (2**32))) % (2**32)
            rng = np.random.default_rng(int(seed))

            for ch in chains:
                self.stats.n_chains_processed += 1
                for coords, seq in split_chain_on_gaps(ch.coords, ch.seq):
                    if len(coords) < int(MIN_SEG_LEN):
                        continue
                    self.stats.n_segments_processed += 1
                    _accumulate_pairs_for_segment(
                        coords=np.asarray(coords, dtype=np.float64),
                        seq_data=seq,  # Pass raw seq data, let converter handle it
                        rng=rng,
                        rdf_counts=rdf_counts,
                        rdf_edges=rdf_edges,
                        contact_counts=contact_counts,
                        max_dist=float(MAX_DIST_A),
                        min_seq_sep=int(MIN_SEQ_SEP),
                        max_pairs_per_segment=int(MAX_PAIRS_PER_SEGMENT),
                        stats=self.stats,
                    )

        logger.info(
            f"Chains processed: {self.stats.n_chains_processed} | "
            f"PDB failures: {self.stats.n_pdbs_failed}/{self.stats.n_pdbs_attempted}"
        )
        logger.info(f"Total accepted pairs: {self.stats.n_pairs_total}")

        # -------------------------
        # RDF -> PMF -> repulsive wall
        # -------------------------
        rdf = compute_rdf_from_counts(
            counts=rdf_counts,
            r_edges=rdf_edges,
            tail_start=float(RDF_TAIL_START_A),
            tail_end=float(RDF_TAIL_END_A),
        )

        r_centers, W_raw, r_star, W_rep = extract_repulsive_wall(
            rdf=rdf, r_star=None, enforce_monotone=True, ensure_nonnegative=True
        )
        r_dense, W_dense = densify_wall(r_centers=r_centers, W_rep=W_rep)

        save_repulsive_wall(
            out_dir=self.data_dir,
            r_dense=r_dense,
            W_dense=W_dense,
            r_star=r_star,
            meta_extra={
                "min_seq_sep_excluded": int(MIN_SEQ_SEP),
                "max_dist_A": float(MAX_DIST_A),
                "tail_window_A": [float(RDF_TAIL_START_A), float(RDF_TAIL_END_A)],
            },
        )

        # Save RDF products
        np.save(self.data_dir / "rdf_edges_A.npy", rdf.r_edges.astype(np.float32))
        np.save(self.data_dir / "rdf_centers_A.npy", rdf.r_centers.astype(np.float32))
        np.save(self.data_dir / "rdf_counts.npy", rdf.counts.astype(np.float32))
        np.save(self.data_dir / "rdf_g_r.npy", rdf.g_r.astype(np.float32))
        np.save(self.data_dir / "rdf_pmf.npy", rdf.pmf.astype(np.float32))

        # -------------------------
        # Enrichment (per bin)
        # -------------------------
        enr = compute_contact_enrichment(
            contact_counts=contact_counts,
            n_shuffles=int(ENRICH_SHUFFLES),
            empirical_p_for_low_counts=bool(EMPIRICAL_P_FOR_LOW_COUNTS),
            empirical_count_threshold=int(EMPIRICAL_COUNT_THRESHOLD),
            fdr_q=float(FDR_Q),
        )

        # save enrichment arrays
        for b in CONTACT_BINS.keys():
            np.save(self.data_dir / f"contact_counts_{b}.npy", contact_counts[b].astype(np.float32))
            np.save(self.data_dir / f"oe_{b}.npy", enr.oe[b].astype(np.float32))
            np.save(self.data_dir / f"log_oe_{b}.npy", enr.log_oe[b].astype(np.float32))
            # PMI == log_OE for pair frequencies; save under pmi_*.npy for packing init
            np.save(self.data_dir / f"pmi_{b}.npy", enr.log_oe[b].astype(np.float32))
            np.save(self.data_dir / f"z_{b}.npy", enr.z[b].astype(np.float32))
            np.save(self.data_dir / f"p_{b}.npy", enr.p[b].astype(np.float32))
            np.save(self.data_dir / f"q_{b}.npy", enr.q[b].astype(np.float32))

        # Summary JSON
        summary = {
            "n_pdbs_attempted": self.stats.n_pdbs_attempted,
            "n_pdbs_failed": self.stats.n_pdbs_failed,
            "n_chains_processed": self.stats.n_chains_processed,
            "n_segments_processed": self.stats.n_segments_processed,
            "n_pairs_total": self.stats.n_pairs_total,
            "min_seq_sep": int(MIN_SEQ_SEP),
            "contact_bins_A": CONTACT_BINS,
            "r_star_A": float(r_star),
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Plots
        plot_rdf_analysis(
            rdf_edges_A=rdf.r_edges,
            rdf_centers_A=rdf.r_centers,
            rdf_counts=rdf.counts,
            g_r=rdf.g_r,
            pmf=rdf.pmf,
            out_dir=self.output_dir,
        )
        plot_repulsive_wall(
            r_dense_A=r_dense,
            W_dense=W_dense,
            r_star_A=r_star,
            out_dir=self.output_dir,
        )
        plot_enrichment_matrices(
            oe=enr.oe,
            log_oe=enr.log_oe,
            z=enr.z,
            q=enr.q,
            contact_bins=CONTACT_BINS,
            out_dir=self.output_dir,
        )

        logger.info(f"✅ Repulsion+packing analysis complete. Outputs in {self.output_dir}")
        return 0


def run_repulsion_analysis(args) -> int:
    """Run repulsion analysis from CLI arguments."""
    analyzer = RepulsionAnalyzer(cache_dir=args.cache_dir, output_dir=args.output_dir)

    # Handle optional arguments with defaults
    max_pdbs = getattr(args, "max_pdbs", None)
    quiet = getattr(args, "quiet", False)
    plot_max_points = getattr(args, "plot_max_points", PLOT_MAX_POINTS)

    return analyzer.run(
        pdb_list=args.pdb_list,
        max_pdbs=max_pdbs,
        quiet=quiet,
        plot_max_points=plot_max_points,
    )
