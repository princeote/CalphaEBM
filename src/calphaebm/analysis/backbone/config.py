# src/calphaebm/analysis/backbone/config.py

"""Configuration constants for backbone geometry analysis.

Important:
- THETA_BINS and PHI_BINS refer to the NUMBER OF BINS (not edges).
- Therefore, edge arrays must have length (BINS + 1).
"""

from pathlib import Path

# Default paths
DEFAULT_CACHE_DIR = Path("./pdb_cache")
DEFAULT_PDB_LIST = Path("train_hq.txt")

# Align with basins expectation: backbone writes to analysis/backbone_geometry/data/*.npy
DEFAULT_OUTPUT_DIR = Path("analysis/backbone_geometry")

# Performance
MAX_POINTS_FOR_KDE = 200_000

# Bond length parameters
BOND_LENGTH_MIN = 2.0
BOND_LENGTH_MAX = 5.0
BOND_LENGTH_BINS = 60  # if you ever save a binned histogram; plotting uses fixed 0.05 Å steps
BOND_LENGTH_IDEAL = 3.8

# Angle binning (degrees)
THETA_MIN = 50.0
THETA_MAX = 170.0
THETA_BINS = 24  # 24 bins -> 25 edges; step = (170-50)/24 = 5°

PHI_MIN = -180.0
PHI_MAX = 180.0
PHI_BINS = 36  # 36 bins -> 37 edges; step = 10°

# Δφ parameters (degrees)
DELTA_PHI_BINS = 36  # 10° bins over [-180,180]
DELTA_PHI_EPS = 1e-6  # avoid log(0) without creating huge spikes

# Plot settings
PLOT_DPI = 300
