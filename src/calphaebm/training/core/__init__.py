"""Core training utilities for phased training."""

from calphaebm.training.core.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from calphaebm.training.core.config import PhaseConfig
from calphaebm.training.core.convergence import ConvergenceCriteria, ConvergenceMonitor
from calphaebm.training.core.freeze import freeze_module, set_requires_grad, unfreeze_module
from calphaebm.training.core.schedules import apply_gate_schedule, get_lr
from calphaebm.training.core.state import TrainingState, ValidationMetrics
from calphaebm.training.core.trainer import BaseTrainer

__all__ = [
    "TrainingState",
    "ValidationMetrics",
    "BaseTrainer",
    "ConvergenceMonitor",
    "ConvergenceCriteria",
    "PhaseConfig",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "freeze_module",
    "unfreeze_module",
    "set_requires_grad",
    "apply_gate_schedule",
    "get_lr",
]
