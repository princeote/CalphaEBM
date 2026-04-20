# src/calphaebm/analysis/basins/config.py

"""Configuration constants for basin analysis."""

from pathlib import Path

# Default paths
DEFAULT_CACHE_DIR = Path("./pdb_cache")
DEFAULT_PDB_LIST = Path("train_entities.no_test_entries.txt")
DEFAULT_OUTPUT_DIR = Path("analysis/secondary_analysis")

# Clustering parameters
DEFAULT_N_BASINS = 4
DEFAULT_CLUSTER_METHOD = "gmm"  # 'gmm' or 'kmeans'
DEFAULT_RANDOM_STATE = 42

# Data loading / sampling
DEFAULT_MAX_PDBS = 10000  # limit number of PDB IDs processed
DEFAULT_MAX_CHAINS = None  # optional: limit number of chains processed (None = no limit)
DEFAULT_SAMPLE_EVERY = 10  # sample every N residues to reduce local correlation

# Clustering feature options
DEFAULT_CIRCULAR_PHI = True  # use [sin(phi), cos(phi)] instead of phi directly
DEFAULT_STANDARDIZE = True  # standardize features before clustering

# Histogram / binning parameters (must match backbone analysis convention)
THETA_MIN = 50.0
THETA_MAX = 170.0
THETA_BINS = 24  # 24 bins -> 25 edges, step = 5°

PHI_MIN = -180.0
PHI_MAX = 180.0
PHI_BINS = 36  # 36 bins -> 37 edges, step = 10°

DEFAULT_SMOOTH_SIGMA = 2.0  # Gaussian smoothing sigma in bins
DEFAULT_PSEUDOCOUNT = 1e-6  # pseudocount for histogram stability

# Plotting
PLOT_DPI = 300
DEFAULT_PLOT_MAX_POINTS = 10_000
