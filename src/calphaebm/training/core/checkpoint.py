"""Checkpoint saving and loading utilities."""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch

from calphaebm.utils.logging import get_logger

logger = get_logger()

# Critical hyperparameters that should match exactly
CRITICAL_HPARAMS = ["betas", "eps", "amsgrad", "maximize"]
# Semi-critical hyperparameters that may vary across torch versions
SEMI_CRITICAL_HPARAMS = ["foreach", "capturable", "differentiable"]
# Tunable hyperparameters that can differ
TUNABLE_HPARAMS = ["lr", "weight_decay"]


def _normalize_value(value: Any) -> Any:
    """Normalize a value for stable comparison and hashing.

    Returns JSON-serializable representation:
        - tuples/lists → lists
        - None → None
        - numbers → float
        - booleans → bool
        - strings → string
        - everything else → string
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        # Convert sequences to lists for JSON serialization
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    # Fallback for any other type
    return str(value)


def _extract_critical_hparams(group: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize critical hyperparameters from a parameter group.

    Only includes parameters that are explicitly set (non-None).
    """
    critical = {}
    for hparam in CRITICAL_HPARAMS + SEMI_CRITICAL_HPARAMS:
        if hparam in group and group[hparam] is not None:
            critical[hparam] = _normalize_value(group[hparam])
    return critical


def _compare_critical_hparams(cur_critical: Dict[str, Any], sav_critical: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Compare critical hyperparameters, handling torch version differences.

    Returns:
        (is_compatible, list_of_mismatches)
    """
    mismatches = []

    # Check all keys that appear in either dict
    all_keys = set(cur_critical.keys()) | set(sav_critical.keys())

    for key in all_keys:
        cur_val = cur_critical.get(key)
        sav_val = sav_critical.get(key)

        # Handle semi-critical params with version differences
        if key in SEMI_CRITICAL_HPARAMS:
            # Treat missing as False (PyTorch default)
            cur_bool = bool(cur_val) if cur_val is not None else False
            sav_bool = bool(sav_val) if sav_val is not None else False
            if cur_bool != sav_bool:
                mismatches.append(f"{key}: {cur_bool} vs {sav_bool} (semi-critical)")

        # Critical params must match exactly
        elif key in CRITICAL_HPARAMS:
            if cur_val != sav_val:
                return False, [f"{key}: {cur_val} vs {sav_val}"]

    return len(mismatches) == 0, mismatches


def _compute_group_key(param_info: List[Tuple[str, Tuple[int, ...]]], critical_hparams: Dict[str, Any]) -> str:
    """Compute a stable key for a parameter group based on its content.

    The key includes:
        - Sorted parameter names AND shapes (converted to JSON-safe lists)
        - Critical hyperparameter values (normalized)

    Returns a 32-character hex hash (128 bits).
    """
    # Sort by parameter name for stable ordering
    sorted_info = sorted(param_info, key=lambda x: x[0])

    # Convert to JSON-safe representation (lists instead of tuples)
    json_safe_info = [[name, list(shape)] for name, shape in sorted_info]  # shape as list, not tuple

    # Normalize critical hparams to JSON-safe form
    json_safe_hparams = _normalize_value(critical_hparams)

    # Create a dict with all components
    key_dict = {"param_info": json_safe_info, "critical_hparams": json_safe_hparams}

    # Convert to stable string representation
    key_str = json.dumps(key_dict, sort_keys=True)

    # Use full 32-character hex hash (128 bits) for collision resistance
    return hashlib.sha256(key_str.encode()).hexdigest()[:32]


def _tunable_key(sig: Dict[str, Any]) -> tuple:
    """Create a sort key for tunable hyperparameters."""
    lr = sig.get("lr")
    wd = sig.get("weight_decay")
    # Normalize None to 0.0 for sorting
    return (_normalize_value(lr) if lr is not None else 0.0, _normalize_value(wd) if wd is not None else 0.0)


def _get_group_signature(group: Dict[str, Any], name_map: Dict[torch.nn.Parameter, str]) -> Dict[str, Any]:
    """Generate a stable signature for a single parameter group."""
    # Get parameter names and shapes
    param_info = []
    for param in group["params"]:
        if param in name_map:
            param_info.append((name_map[param], tuple(param.shape)))
        else:
            raise ValueError(f"Parameter not found in model: {param}")

    # Sort by parameter name for stable ordering
    param_info.sort(key=lambda x: x[0])

    # Extract critical hyperparameters
    critical_hparams = _extract_critical_hparams(group)

    # Compute content-based key
    group_key = _compute_group_key(param_info, critical_hparams)

    # Build signature
    signature = {
        "key": group_key,
        "param_info": [(name, list(shape)) for name, shape in param_info],  # shapes as lists
        # Store all hyperparameters
        "lr": group.get("lr"),
        "weight_decay": group.get("weight_decay"),
        "betas": _normalize_value(group.get("betas")),
        "eps": group.get("eps"),
        "amsgrad": group.get("amsgrad"),
        "maximize": group.get("maximize"),
        "foreach": group.get("foreach"),
        "capturable": group.get("capturable"),
        "differentiable": group.get("differentiable"),
        # Store normalized critical hparams for comparison
        "critical_hparams": critical_hparams,
        "tunable_hparams": {h: _normalize_value(group.get(h)) for h in TUNABLE_HPARAMS if h in group},
    }

    return signature


def _get_optimizer_signature(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate a signature for each parameter group in the optimizer."""
    name_map = {param: name for name, param in model.named_parameters()}

    signatures_by_key = defaultdict(list)
    for group in optimizer.param_groups:
        sig = _get_group_signature(group, name_map)
        signatures_by_key[sig["key"]].append(sig)

    # Sort signatures within each key bucket by tunable hyperparameters
    for key in signatures_by_key:
        signatures_by_key[key].sort(key=_tunable_key)

    return dict(signatures_by_key)


def _verify_optimizer_compatibility(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module, saved_signatures: Dict[str, List[Dict[str, Any]]]
) -> bool:
    """Verify that current optimizer groups match saved signatures."""
    current_signatures = _get_optimizer_signature(optimizer, model)

    # Check group count
    current_total = sum(len(groups) for groups in current_signatures.values())
    saved_total = sum(len(groups) for groups in saved_signatures.values())

    if current_total != saved_total:
        logger.warning(f"Optimizer group count mismatch: current={current_total}, saved={saved_total}")
        return False

    # Check that all current groups have matching saved groups
    for key, cur_groups in current_signatures.items():
        if key not in saved_signatures:
            logger.warning(f"Group with key {key} not found in saved checkpoint")
            return False

        sav_groups = saved_signatures[key]

        if len(cur_groups) != len(sav_groups):
            logger.warning(f"Group count mismatch for key {key}: current={len(cur_groups)}, saved={len(sav_groups)}")
            return False

        # Sort groups by tunable hyperparameters for stable pairing
        cur_groups_sorted = sorted(cur_groups, key=_tunable_key)
        sav_groups_sorted = sorted(sav_groups, key=_tunable_key)

        for i, (cur_sig, sav_sig) in enumerate(zip(cur_groups_sorted, sav_groups_sorted)):
            # Verify parameter info matches
            if cur_sig["param_info"] != sav_sig["param_info"]:
                logger.warning(f"Parameter info mismatch for group {key}[{i}]")
                return False

            # Compare critical hyperparameters
            compatible, mismatches = _compare_critical_hparams(cur_sig["critical_hparams"], sav_sig["critical_hparams"])
            if not compatible:
                logger.error(f"Critical hyperparameter mismatch for group {key}[{i}]: {mismatches}")
                return False
            for mismatch in mismatches:
                logger.warning(f"Semi-critical hyperparameter mismatch for group {key}[{i}]: {mismatch}")

            # Check tunable hyperparameters (warn only)
            for hparam in TUNABLE_HPARAMS:
                cur_val = cur_sig.get(hparam)
                sav_val = sav_sig.get(hparam)
                if cur_val != sav_val:
                    logger.warning(
                        f"Tunable hyperparameter '{hparam}' mismatch in group {key}[{i}]: " f"{cur_val} vs {sav_val}"
                    )

    # Check saved doesn't have extra keys
    for key in saved_signatures:
        if key not in current_signatures:
            logger.warning(f"Saved group with key {key} not found in current optimizer")
            return False

    logger.info(f"Optimizer signature verified: {current_total} groups in {len(current_signatures)} keys")
    return True


# ========== PUBLIC API ==========


def save_checkpoint(
    path: str,
    global_step: int,
    phase_step: int,
    phase: str,
    loss: float,
    model_state: dict,
    gates: dict,
    optimizer_state=None,
    scheduler_state=None,
    best_composite_score=None,
    best_composite_score_initialized=False,
    best_val_step=0,
    early_stopping_counter=0,
    validation_history=None,
    convergence_info: Optional[Dict[str, Any]] = None,
    is_best=False,
    model=None,
    optimizer=None,
    signature_version: str = "2.5",
    config: Optional[Dict[str, Any]] = None,
):
    """Save training checkpoint with separate global and phase steps.

    Args:
        path: Path to save checkpoint
        global_step: Global training step
        phase_step: Step within current phase
        phase: Phase name
        loss: Current loss value
        model_state: Model state dict
        gates: Gate values dict
        optimizer_state: Optimizer state dict (optional)
        scheduler_state: Scheduler state dict (optional)
        best_composite_score: Best validation score (optional)
        best_composite_score_initialized: Whether best score is initialized
        best_val_step: Step where best score was achieved
        early_stopping_counter: Early stopping counter
        validation_history: List of validation metrics
        convergence_info: Convergence information dict
        is_best: Whether this is the best model so far
        model: Model object (for signature generation)
        optimizer: Optimizer object (for signature generation)
        signature_version: Version of signature format

    Returns:
        Path to the saved checkpoint (best path if is_best=True)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Generate optimizer signature if model and optimizer are provided
    optimizer_signature = None
    signature_error = None
    if model is not None and optimizer is not None:
        try:
            optimizer_signature = _get_optimizer_signature(optimizer, model)
            logger.debug(
                f"Generated optimizer signature with {sum(len(g) for g in optimizer_signature.values())} groups"
            )
        except Exception as e:
            signature_error = str(e)
            logger.error(f"Failed to generate optimizer signature: {e}")

    # Build model config if model is provided but config is not (#60)
    if config is None and model is not None:
        try:
            from calphaebm.utils.checkpoint import build_model_config

            config = build_model_config(model)
        except Exception as e:
            logger.debug(f"Could not auto-build model config: {e}")
            config = {}

    payload = {
        "global_step": global_step,
        "phase_step": phase_step,
        "phase": phase,
        "train_loss": loss,
        "model_state": model_state,
        "gates": gates,
        "config": config or {},
        "optimizer_state": optimizer_state,
        "optimizer_signature": optimizer_signature,
        "optimizer_signature_version": signature_version if optimizer_signature else None,
        "scheduler_state": scheduler_state,
        "best_composite_score": best_composite_score,
        "best_composite_score_initialized": best_composite_score_initialized,
        "best_val_step": best_val_step,
        "early_stopping_counter": early_stopping_counter,
        "validation_history": validation_history,
        "convergence_info": convergence_info,
        "signature_error": signature_error,
        "format_version": "2.5",
    }

    torch.save(payload, path)
    logger.info(f"Saved checkpoint: {path} (global={global_step}, phase={phase_step})")

    if is_best:
        if path.endswith("_best.pt"):
            best_path = path
        else:
            best_path = path.replace(".pt", "_best.pt")
        torch.save(payload, best_path)
        logger.info(f"Saved best model: {best_path}")
        return best_path

    return path


def _clean_filename_for_parsing(filename: str) -> str:
    """Remove _best suffix for step parsing."""
    return filename.replace("_best.pt", ".pt")


def _handle_old_signature_format(ckpt: Dict) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Convert old signature formats to current format."""
    sig_version = ckpt.get("optimizer_signature_version", "1.0")

    if sig_version == "2.5":
        # Already current format
        return ckpt.get("optimizer_signature")

    if sig_version in ["2.3", "2.4"]:
        # Previous formats - warn and skip
        logger.warning(f"Old optimizer signature format {sig_version} detected. " "Optimizer state will not be loaded.")
        return None

    # Unknown format
    logger.warning(f"Unknown optimizer signature format {sig_version}")
    return None


def load_checkpoint(
    path: str, model, optimizer=None, scheduler=None, device=torch.device("cpu"), load_optimizer=True, strict=False
):
    """Load checkpoint, with optional optimizer state and param group validation."""
    from .state import TrainingState

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Check format version for backward compatibility
    format_version = ckpt.get("format_version", "1.0")

    # Always load model state
    try:
        model.load_state_dict(ckpt["model_state"], strict=strict)
    except Exception as e:
        logger.error(f"Failed to load model state: {e}")
        raise

    # Only load optimizer state if requested and compatible
    optimizer_loaded = False
    if load_optimizer and optimizer is not None and "optimizer_state" in ckpt:
        # Handle signature format conversion
        optimizer_signature = _handle_old_signature_format(ckpt)

        if optimizer_signature is not None:
            # New format with signature
            try:
                if _verify_optimizer_compatibility(optimizer, model, optimizer_signature):
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                    logger.info("Optimizer state loaded successfully (signature verified)")
                    optimizer_loaded = True
                else:
                    logger.warning("Optimizer signature mismatch - skipping optimizer state load")
                    logger.info("Training will resume with freshly initialized optimizer")
            except Exception as e:
                logger.error(f"Error during optimizer verification: {e}")
                logger.info("Training will resume with freshly initialized optimizer")
        else:
            # Old format - try safe load with warning
            logger.warning(
                "Old checkpoint format (no compatible optimizer signature). Attempting to load optimizer state..."
            )
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                logger.info("Optimizer state loaded (no signature validation)")
                optimizer_loaded = True
            except Exception as e:
                logger.warning(f"Could not load optimizer state from old checkpoint: {e}")
                logger.info("Training will resume with freshly initialized optimizer")

    # Only load scheduler state if optimizer was loaded
    if optimizer_loaded and scheduler is not None and "scheduler_state" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
            logger.info("Scheduler state loaded")
        except Exception as e:
            logger.warning(f"Could not load scheduler state: {e}")

    if "gates" in ckpt and hasattr(model, "set_gates"):
        model.set_gates(**ckpt["gates"])

    # Handle different format versions
    if format_version == "1.0":
        # Old format: step was phase_step
        phase_step = ckpt.get("step", 0)
        global_step = ckpt.get("global_step", phase_step)
    else:
        # New format: explicit separation
        global_step = ckpt.get("global_step", 0)
        phase_step = ckpt.get("phase_step", 0)

    # Extract convergence info
    convergence_info = ckpt.get("convergence_info", {}) or {}

    # Log signature error if present
    if "signature_error" in ckpt and ckpt["signature_error"]:
        logger.warning(f"Checkpoint was saved with signature error: {ckpt['signature_error']}")

    # Try to verify filename matches payload (for non-best checkpoints)
    try:
        clean_name = _clean_filename_for_parsing(os.path.basename(path))
        if clean_name.startswith("step") and clean_name.endswith(".pt") and "best" not in path:
            filename_step = int(clean_name.replace("step", "").replace(".pt", ""))
            if filename_step != phase_step:
                logger.warning(f"Checkpoint filename step {filename_step} != payload phase_step {phase_step}")
    except (ValueError, IndexError):
        pass

    return TrainingState(
        global_step=global_step,
        phase_step=phase_step,
        phase=ckpt.get("phase", "unknown"),
        losses={"loss": ckpt.get("train_loss", 0.0)},
        gates=ckpt.get("gates", {}),
        best_composite_score=ckpt.get("best_composite_score"),
        best_composite_score_initialized=ckpt.get("best_composite_score_initialized", False),
        best_val_step=ckpt.get("best_val_step", 0),
        early_stopping_counter=ckpt.get("early_stopping_counter", 0),
        validation_history=ckpt.get("validation_history", []),
        convergence_info=convergence_info,
        converged=convergence_info.get("converged", False),
        convergence_step=convergence_info.get("convergence_step"),
    )


def find_latest_checkpoint(phase_dir: str) -> Optional[str]:
    """Find latest checkpoint in a phase directory.

    Args:
        phase_dir: Full path to phase directory (e.g., checkpoints/run1/local/)

    Returns:
        Path to latest checkpoint or None
    """
    if not os.path.exists(phase_dir):
        return None

    checkpoints = [f for f in os.listdir(phase_dir) if f.startswith("step") and f.endswith(".pt") and "best" not in f]
    if not checkpoints:
        return None

    # Parse step numbers from filenames
    steps = []
    valid_checkpoints = []
    for f in checkpoints:
        try:
            step = int(f.replace("step", "").replace(".pt", ""))
            steps.append(step)
            valid_checkpoints.append(f)
        except ValueError:
            logger.warning(f"Skipping invalid checkpoint filename: {f}")
            continue

    if not steps:
        return None

    # Find checkpoint with highest step number
    latest_idx = max(range(len(steps)), key=lambda i: steps[i])
    return os.path.join(phase_dir, valid_checkpoints[latest_idx])
