# src/calphaebm/__init__.py

"""CalphaEBM - Energy-based model for protein backbone sampling."""

__version__ = "0.1.0"

# Don't import training modules at top level - they have heavy dependencies
# and cause import errors when running analysis commands

# Instead, make them available as submodules that can be imported when needed
from calphaebm import analysis, cli, data, evaluation, geometry, models, simulation, training, utils

# Only import commonly used items that have no heavy dependencies
from calphaebm.utils.logging import get_logger, setup_logger

__all__ = [
    "analysis",
    "cli",
    "data",
    "evaluation",
    "geometry",
    "models",
    "simulation",
    "training",
    "utils",
    "get_logger",
    "setup_logger",
]
