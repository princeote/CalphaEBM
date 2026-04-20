"""Utility functions for Langevin dynamics availability."""

from calphaebm.utils.logging import get_logger

logger = get_logger()

_HAS_LANGEVIN = None
_LANGEVIN_SAMPLE = None
_LANGEVIN_IMPORT_ERROR = None


def check_langevin_available():
    """Check if langevin_sample is available."""
    global _HAS_LANGEVIN, _LANGEVIN_SAMPLE, _LANGEVIN_IMPORT_ERROR

    if _HAS_LANGEVIN is None:
        try:
            from calphaebm.simulation.backends.pytorch import langevin_sample

            _LANGEVIN_SAMPLE = langevin_sample
            _HAS_LANGEVIN = True
            _LANGEVIN_IMPORT_ERROR = None
        except ImportError as e:
            _HAS_LANGEVIN = False
            _LANGEVIN_IMPORT_ERROR = str(e)
            logger.warning(f"langevin_sample not available. Error: {e}")

    return _HAS_LANGEVIN


def get_langevin_sample():
    """Get the langevin_sample function if available."""
    if check_langevin_available():
        return _LANGEVIN_SAMPLE
    return None


def get_langevin_error():
    """Get the import error message if langevin_sample is not available."""
    return _LANGEVIN_IMPORT_ERROR
