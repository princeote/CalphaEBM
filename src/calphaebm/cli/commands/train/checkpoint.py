"""Checkpoint loading and weight application."""

from typing import Any, Dict, Optional, Tuple

import torch

from calphaebm.utils.logging import get_logger

logger = get_logger()


def load_checkpoint(
    resume_path: str,
    device: torch.device,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Load checkpoint and extract model state and trainer state."""
    checkpoint_state: Optional[Dict[str, Any]] = None
    trainer_state: Dict[str, Any] = {}

    logger.debug("Loading checkpoint: %s", resume_path)
    checkpoint = torch.load(resume_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            checkpoint_state = checkpoint["model_state"]
            logger.debug("Loaded model state from 'model_state' key")
        elif "model_state_dict" in checkpoint:
            checkpoint_state = checkpoint["model_state_dict"]
            logger.debug("Loaded model state from 'model_state_dict' key")
        elif "state_dict" in checkpoint:
            checkpoint_state = checkpoint["state_dict"]
            logger.debug("Loaded model state from 'state_dict' key")
        else:
            checkpoint_state = checkpoint
            logger.debug("Loaded checkpoint as direct state dict")

        trainer_state = {
            "global_step": checkpoint.get("global_step", 0),
            "phase_step": checkpoint.get("phase_step", 0),
            "best_composite_score": checkpoint.get("best_composite_score"),
            "best_composite_score_initialized": checkpoint.get("best_composite_score_initialized", False),
            "best_val_step": checkpoint.get("best_val_step", 0),
            "early_stopping_counter": checkpoint.get("early_stopping_counter", 0),
            "validation_history": checkpoint.get("validation_history", []),
        }
    else:
        checkpoint_state = checkpoint
        logger.debug("Loaded checkpoint as direct state dict")

    return checkpoint_state, trainer_state


def load_weights_into_model(
    model: torch.nn.Module,
    checkpoint_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Load checkpoint weights into model with key mapping.

    Handles:
    - module./orig_mod. prefixes from DDP/torch.compile
    - Legacy checkpoints that still contain _bond_spring_raw/_bond_spring_val
      (silently skipped via strict=False — the IC model has no bond_spring)
    """
    current_state = model.state_dict()

    # Strip wrapper prefixes
    state = checkpoint_state
    if any(k.startswith("module.") for k in state.keys()):
        logger.debug("Removing 'module.' prefix from checkpoint keys")
        state = {k.replace("module.", ""): v for k, v in state.items()}
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        logger.debug("Removing '_orig_mod.' prefix from checkpoint keys")
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # Map checkpoint keys to model keys
    mapped_state: Dict[str, Any] = {}

    # Legacy key mappings for local term (no bond_spring — IC model)
    # bond_spring keys from old checkpoints are intentionally absent here;
    # they will be silently skipped by strict=False below.
    local_param_map = {
        "theta_theta_weight": "local.theta_theta_weight",
        "delta_phi_weight": "local.delta_phi_weight",
    }

    for key, value in state.items():
        # Direct match
        if key in current_state:
            if value.shape == current_state[key].shape:
                mapped_state[key] = value
                logger.debug("Direct match: %s", key)
            else:
                logger.warning("Shape mismatch for %s: ckpt %s vs model %s", key, value.shape, current_state[key].shape)
            continue

        # Legacy local parameter mapping
        if key in local_param_map:
            target_key = local_param_map[key]
            if target_key in current_state and value.shape == current_state[target_key].shape:
                mapped_state[target_key] = value
                logger.debug("Mapped local param: %s -> %s", key, target_key)
                continue

        # Try adding 'local.' prefix
        if not key.startswith("local.") and f"local.{key}" in current_state:
            target_key = f"local.{key}"
            if value.shape == current_state[target_key].shape:
                mapped_state[target_key] = value
                logger.debug("Added local prefix: %s -> %s", key, target_key)
                continue

        # Gate parameters
        if key in ("gate_local", "gate_repulsion", "gate_secondary", "gate_packing"):
            if key in current_state and value.shape == current_state[key].shape:
                mapped_state[key] = value
                logger.debug("Gate param: %s", key)
                continue

        # Silently skip keys not in current model (e.g. legacy bond_spring keys)
        logger.debug("Skipping checkpoint key not in model: %s", key)

    missing, unexpected = model.load_state_dict(mapped_state, strict=False)

    stats = {
        "loaded": len(mapped_state),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }

    logger.debug(
        "Model weight loading complete: loaded=%d  missing=%d  unexpected=%d",
        stats["loaded"],
        len(stats["missing"]),
        len(stats["unexpected"]),
    )

    # Store stats on model for consolidated logging
    model._load_stats = stats

    return stats
