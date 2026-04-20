"""Calibration modules for phased training."""

import importlib.util

from calphaebm.utils.logging import get_logger

logger = get_logger()

from calphaebm.training.calibrators.contrastive_calibrators import ContrastiveLossComputer

# Core calibrators
from calphaebm.training.calibrators.repulsion_calibrators import RepulsionCalibrator
from calphaebm.training.calibrators.subterm_calibrators import CalibrationResults, SubtermScaleCalibrator

__all__ = [
    "RepulsionCalibrator",
    "ContrastiveLossComputer",
    "SubtermScaleCalibrator",
    "CalibrationResults",
]

# Optional packing diagnostics
packing_spec = importlib.util.find_spec("calphaebm.training.calibrators.packing_calibrators")
if packing_spec is not None:
    try:
        from calphaebm.training.calibrators.packing_calibrators import PackingDiagnostics

        __all__.append("PackingDiagnostics")
        logger.debug("Packing diagnostics module loaded successfully")
    except ImportError as e:
        logger.warning(f"Packing diagnostics module found but failed to import: {e}")
else:
    logger.debug("Packing diagnostics module not found - skipping export")
