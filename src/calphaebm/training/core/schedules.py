"""Learning rate and gate scheduling utilities."""

import math
from typing import Dict, List, Optional


def get_lr(initial_lr: float, step: int, total_steps: int, schedule: Optional[str], lr_final: Optional[float]) -> float:
    """Compute learning rate based on schedule."""
    if schedule is None:
        return initial_lr

    progress = step / total_steps
    lr_final = lr_final or initial_lr * 0.01

    if schedule == "linear":
        return initial_lr + (lr_final - initial_lr) * progress

    elif schedule == "cosine":
        return lr_final + 0.5 * (initial_lr - lr_final) * (1 + math.cos(math.pi * progress))

    elif schedule == "exponential":
        decay_rate = math.exp(math.log(lr_final / initial_lr))
        return initial_lr * (decay_rate**progress)

    return initial_lr


def apply_gate_schedule(
    model, phase_step: int, total_steps: int, gate_schedule: Dict[str, List[float]]
) -> Dict[str, float]:
    """Apply gate scheduling."""
    if not gate_schedule:
        return {}

    gates = {}
    for term, schedule in gate_schedule.items():
        if schedule and len(schedule) == 2:
            start, end = schedule
            progress = phase_step / total_steps
            gate_value = start + (end - start) * progress
            gates[term] = gate_value

    if gates and hasattr(model, "set_gates"):
        model.set_gates(**gates)

    return gates
