"""Training state dataclasses."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ValidationMetrics:
    """Physics-based validation metrics for a single validation run."""

    step: int
    composite_score: float
    bond_length_mean: float
    bond_length_std: float
    bond_length_rmsd: float
    train_loss: float
    native_vs_distorted_gap: float
    helix_vs_random_gap: float
    mean_energy: float
    energy_std: float
    ramachandran_corr: float = 0.0
    delta_phi_corr: float = 0.0
    sheet_vs_random_gap: Optional[float] = None
    energy_consistency: Optional[float] = None
    failure_reason: Optional[str] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingState:
    """Current training state."""

    global_step: int  # Total steps across all phases
    phase_step: int  # Steps within current phase
    phase: str
    losses: Dict[str, float] = field(default_factory=dict)
    gates: Dict[str, float] = field(default_factory=dict)
    best_composite_score: Optional[float] = None
    best_composite_score_initialized: bool = False
    best_val_step: int = 0
    early_stopping_counter: int = 0
    validation_history: List[ValidationMetrics] = field(default_factory=list)
    converged: bool = False
    convergence_step: Optional[int] = None
    convergence_info: Optional[Dict[str, Any]] = None
