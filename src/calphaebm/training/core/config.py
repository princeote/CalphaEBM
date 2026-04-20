# src/calphaebm/training/core/config.py

"""Phase configuration class.

Design
------
PhaseConfig is a plain class (not a @dataclass) that accepts **kwargs in
its constructor.  Any unknown keyword argument is silently absorbed and
stored as an attribute with its provided value.  Known fields have defaults
in _DEFAULTS.

This means adding a new loss only requires changes in TWO places:
  1. CLI args   (src/calphaebm/cli/commands/train/config.py)
  2. Forwarding (src/calphaebm/cli/commands/train/train_main.py)

The PhaseConfig dataclass field list no longer needs to be updated for
every new hyperparameter — it will just work via **kwargs.

Required fields (no default, must be passed explicitly):
  name, terms, freeze, loss_fn, n_steps
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Defaults for all optional fields
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    # Core training
    "lr": 3e-4,
    "lr_schedule": None,
    "lr_final": None,
    "save_every": 500,
    "validate_every": 500,
    "early_stopping_patience": None,
    "early_stopping_min_delta": 0.001,
    "weight_decay": 0.01,
    # Gate schedules
    "gate_schedule": None,
    # DSM sigma
    "sigma": 0.25,
    "sigma_min": None,
    "sigma_max": None,
    # Full-phase gate ramping
    "ramp_gates": False,
    "ramp_start": None,
    "ramp_end": None,
    "ramp_steps": 5000,
    "freeze_gates_steps": 0,
    # Force balance loss
    "lambda_fb": 0.0,
    "fb_sigma_thermal": 0.3,
    "fb_clash_frac": 0.05,
    "fb_clash_sigma": 0.5,
    "fb_target_ss_ratio": 2.0,
    "fb_target_pack_ratio": 2.0,
    "fb_target_rep_ratio": 2.0,
    "fb_diag_every": 200,
    # Local geometry gap loss
    "lambda_geogap": 0.0,
    "geogap_margin": 2.0,
    "geogap_bond_sigma": 0.3,
    "geogap_angle_sigma": 0.2,
    "geogap_dihedral_sigma": 0.4,
    "geogap_frac_perturbed": 0.3,
    "geogap_diag_every": 200,
    # Packing-phase ramping
    "ramp_pack_start": None,
    "ramp_pack_end": None,
    # Packing logOE pre-training
    "packing_pretrain": False,
    "packing_logoe_data_dir": None,
    "packing_logoe_scale": 5.0,
    # Optimizer LR multipliers
    "scalar_lr_mult": 20.0,
    # Packing scalar freeze
    "freeze_packing_scalar": False,
    # Packing contrastive loss
    "lambda_pack_contrastive": 0.0,
    "pack_contrastive_margin": 0.5,
    # Energy balance loss
    "lambda_balance": 0.0,
    "balance_r": 3.0,
    # Secondary structure basin loss
    "lambda_basin": 0.0,
    "basin_margin": 0.5,
    # Native gap loss
    "lambda_native": 0.0,
    "native_margin": 0.5,
    "native_sigma_min": 0.05,
    "native_sigma_max": 0.50,
    # Validation speed controls
    "val_max_samples": 256,
    "val_langevin_steps": 500,
    # Langevin inverse temperature
    "langevin_beta": 1.0,
}

_REQUIRED = ("name", "terms", "freeze", "loss_fn", "n_steps")


class PhaseConfig:
    """Configuration for a training phase.

    Accepts any keyword arguments — unknown keys are stored as attributes
    with their provided values.  This allows new hyperparameters to be added
    without modifying this file.

    Required kwargs: name, terms, freeze, loss_fn, n_steps.
    All other kwargs fall back to _DEFAULTS if not provided.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Check required fields
        for key in _REQUIRED:
            if key not in kwargs:
                raise TypeError(f"PhaseConfig missing required argument: '{key}'")

        # Apply defaults first, then override with provided kwargs
        merged = {**_DEFAULTS, **kwargs}

        # Set all as attributes
        for key, val in merged.items():
            setattr(self, key, val)

        # Post-init logic (mirrors the old __post_init__)
        if self.gate_schedule is None:
            self.gate_schedule = {}
        if self.lr_schedule is not None and self.lr_final is None:
            self.lr_final = self.lr * 0.01

    def __repr__(self) -> str:
        required = {k: getattr(self, k) for k in _REQUIRED}
        return f"PhaseConfig({', '.join(f'{k}={v!r}' for k, v in required.items())}, ...)"
