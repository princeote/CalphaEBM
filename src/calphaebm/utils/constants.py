# src/calphaebm/utils/constants.py

# src/calphaebm/utils/constants.py

"""Central constants used across CalphaEBM.

All values are defaults that can be overridden by configuration.
"""

# Chemical constants
NUM_AA = 20  # Standard amino acids
CA_CA_BOND_LENGTH = 3.8  # Approximate Cα-Cα distance in Å

# Embedding dimensions
EMB_DIM = 16  # Amino acid embedding dimension

# Training defaults
LEARNING_RATE = 3e-4
TRAIN_STEPS = 50000
BATCH_SIZE = 32

# IC training noise (run19+) — noise is in radians, not Å
# DSM_SIGMA (Å, Cartesian) removed: bond length is now a geometric identity,
# not a variable. Cartesian noise sigma is meaningless in IC simulation.
DSM_SIGMA_THETA = 0.05  # Noise sigma for bond angle θ (radians)
DSM_SIGMA_PHI = 0.10  # Noise sigma for torsion angle φ (radians)

# Local energy term constants (from statistical analysis)
# BOND_SPRING removed: Cα-Cα bond length is exactly 3.8 Å by NeRF construction.
# A harmonic spring on a fixed constraint is undefined and was the root cause
# of the run18 local energy runaway. See geometry/reconstruct.py.
THETA_PERSISTENCE_SPRING = 20.0  # Spring constant for θ-θ coupling
THETA_PHI_SCALE = 0.01  # Scale factor for tabulated (θ,φ) potential

# Data paths for tabulated potentials
THETA_PHI_GRID_PATH = "analysis/backbone_geometry/data/figure_3a_histogram.npy"
THETA_PHI_XEDGES_PATH = "analysis/backbone_geometry/data/figure_3a_xedges.npy"
THETA_PHI_YEDGES_PATH = "analysis/backbone_geometry/data/figure_3a_yedges.npy"
DELTA_PHI_GRID_PATH = "analysis/backbone_geometry/data/delta_phi_energy.npy"
DELTA_PHI_CENTERS_PATH = "analysis/backbone_geometry/data/delta_phi_centers.npy"

# Simulation defaults
BETA = 1.0  # Inverse temperature (kBT = 1 units)
STEP_SIZE = 2e-4  # Langevin step size
FORCE_CAP = 50.0  # Max force norm per atom
N_STEPS = 5000  # Default simulation steps
LOG_EVERY = 50  # Log frequency

# Nonbonded defaults
EXCLUDE = 2  # Sequence separation for nonbonded pairs
K_NEIGHBORS = 64  # Top-K neighbors for nonbonded
R_ON = 8.0  # Switching onset distance (Å)
R_CUT = 12.0  # Cutoff distance (Å)

# Repulsion defaults
RHO_MIN = 1.6  # Minimum effective radius (Å)
RHO_MAX = 2.8  # Maximum effective radius (Å)
RHO_BASE = 2.0  # Initial radius guess (Å)
DELTA = 0.3  # Softness parameter for repulsion
WALL_SCALE = 10.0  # Repulsion strength

# Contact definitions
CONTACT_CUTOFF = 8.0  # Distance for defining native contacts (Å)
CLASH_THRESHOLD = 3.8  # Minimum distance for clash detection (Å)

# Radial distribution function
RDF_RMAX = 20.0  # Max distance for RDF (Å)
RDF_DR = 0.25  # Bin width for RDF (Å)

# File paths (relative to project root)
CACHE_DIR = "./pdb_cache"
CHECKPOINT_DIR = "./checkpoints"
RUNS_DIR = "./runs"
