"""
Configuration for repulsion + packing analysis.

Notes:
- MIN_SEQ_SEP controls nonbonded exclusion: only pairs with |i-j| > MIN_SEQ_SEP are counted.
- MIN_SEG_LEN controls minimum contiguous segment length to consider for sampling.
  (This is independent of MIN_SEQ_SEP.)

CRITICAL CONSISTENCY NOTES:
- This analysis uses MIN_SEQ_SEP = 3 (skip i,i+1,i+2,i+3)
- Models MUST use the SAME exclusion:
  * repulsion.py should use exclude=3 (NOT exclude=2)
  * packing.py should use exclude=3 (already correct)
- Tight bin starts at 5.0Å to match packing.py short-range gate (ramps on 4.5-5.0Å)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# -----------------------------
# Default paths
# -----------------------------
DEFAULT_CACHE_DIR = Path("./pdb_cache")
DEFAULT_PDB_LIST = Path("pdb1000_entries.txt")
DEFAULT_OUTPUT_DIR = Path("analysis/repulsion_packing")

# -----------------------------
# Geometry / sampling controls
# -----------------------------

# Exclude local pairs: only consider |i-j| > MIN_SEQ_SEP
# CRITICAL: This MUST match model defaults
# repulsion.py should use exclude=3 (currently uses exclude=2 - NEEDS FIX)
# packing.py uses exclude=3 (correct)
MIN_SEQ_SEP: int = 3

# Minimum segment length to include in analysis (must be >= MIN_SEQ_SEP+2 for any valid pair)
MIN_SEG_LEN: int = 8

# Backward-compatible alias (in case any older code expects this name)
MIN_SEGMENT_LEN: int = MIN_SEG_LEN

# Maximum distance considered for nonbonded analysis (Å)
MAX_DIST_A: float = 15.0

# For efficiency: cap sampled pairs per segment (large segments are subsampled)
MAX_PAIRS_PER_SEGMENT: int = 200_000

# Seed used to make sampling reproducible
PAIR_SAMPLE_SEED: int = 1337

# -----------------------------
# RDF settings
# -----------------------------
RDF_R_MIN_A: float = 0.15
RDF_R_MAX_A: float = 15.0
RDF_N_BINS: int = 300

# Tail window used to normalize g(r) -> 1
RDF_TAIL_START_A: float = 12.0
RDF_TAIL_END_A: float = 15.0

# PMF computation safeguards
PMF_MIN_G: float = 1e-12  # lower floor for g(r) before log
PMF_SMOOTH_SIGMA: float = 0.0  # (bins) optional smoothing of PMF; 0 disables

# Repulsive wall extraction
WALL_SPARSE_N: int = 48  # sparse grid points for monotone wall
WALL_DENSE_N: int = 200  # dense interpolation grid

# Smoothing in bins for the wall (0 disables)
WALL_SMOOTH_SIGMA_BINS: float = 1.0

# -----------------------------
# Backward-compatible aliases (legacy names without _A suffix)
# -----------------------------
RDF_R_MIN = RDF_R_MIN_A
RDF_R_MAX = RDF_R_MAX_A
MAX_DIST = MAX_DIST_A

RDF_TAIL_START = RDF_TAIL_START_A
RDF_TAIL_END = RDF_TAIL_END_A

WALL_SPARSE_GRID_N = WALL_SPARSE_N
WALL_DENSE_GRID_N = WALL_DENSE_N
WALL_SMOOTH_SIGMA = WALL_SMOOTH_SIGMA_BINS

# -----------------------------
# Packing contact bins (match your short-range gate)
# short-range packing gate ramps to 1 by ~5.0Å, so “tight” starts at 5.0Å
# -----------------------------
CONTACT_BINS: Dict[str, Dict[str, float]] = {
    "tight": {"min": 5.0, "max": 6.0},
    "medium": {"min": 6.0, "max": 8.0},
    "loose": {"min": 8.0, "max": 10.0},
}

# -----------------------------
# Enrichment statistics
# -----------------------------
ENRICH_SHUFFLES: int = 50

# Backward-compatible alias expected by some modules
ENRICHMENT_N_SHUFFLES: int = ENRICH_SHUFFLES

EMPIRICAL_P_FOR_LOW_COUNTS: bool = True
EMPIRICAL_COUNT_THRESHOLD: int = 50  # obs < threshold => empirical p-value from shuffles

# Multiple testing correction
FDR_Q: float = 0.05

# Plotting
PLOT_DPI: int = 300
PLOT_MAX_POINTS: int = 100_000

# Amino acid labels for plotting (20 canonical)
AA_NAMES = list("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class RepulsionPackingConfig:
    """Optional structured config, useful for passing around."""

    cache_dir: Path = DEFAULT_CACHE_DIR
    pdb_list: Path = DEFAULT_PDB_LIST
    output_dir: Path = DEFAULT_OUTPUT_DIR
