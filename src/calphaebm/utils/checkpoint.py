# src/calphaebm/utils/checkpoint.py
"""Checkpoint save/load utilities with config preservation (#60).

Solves the recurring problem of loading checkpoints without knowing
which constructor args were used. Every checkpoint now contains:
  - state_dict: model weights and buffers
  - config: dict of all constructor args and hyperparameters
  - metadata: step, round, loss, timestamp, version

Backward compatible: loading old checkpoints (raw state_dict)
still works — config will be empty dict.

Usage:
    from calphaebm.checkpoint_utils import save_checkpoint, load_checkpoint

    # Save
    save_checkpoint(
        path="round003.pt",
        model=model,
        config={"rg_lambda": 1.0, "coord_lambda": 0.1, ...},
        step=5000,
        round_num=3,
        loss=0.42,
    )

    # Load
    state_dict, config, meta = load_checkpoint("round003.pt", device)
    # config has all constructor args
    # meta has step, round, loss, timestamp
    model.load_state_dict(state_dict, strict=False)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from calphaebm.utils.logging import get_logger

logger = get_logger()

CHECKPOINT_VERSION = "v5"


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: Optional[Dict[str, Any]] = None,
    step: int = 0,
    round_num: int = 0,
    loss: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model checkpoint with config and metadata.

    Args:
        path:      Output file path (.pt).
        model:     The model to save.
        config:    Dict of constructor args / hyperparameters.
        step:      Current training step.
        round_num: Current round number.
        loss:      Current loss value.
        extra:     Any additional metadata to store.

    Returns:
        Path to saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "config": config or {},
        "meta": {
            "step": step,
            "round": round_num,
            "loss": loss,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": CHECKPOINT_VERSION,
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
    }
    if extra:
        checkpoint["meta"].update(extra)

    torch.save(checkpoint, path)
    logger.info(
        "Saved checkpoint: %s (step=%d, round=%d, loss=%.4f, config=%d keys)",
        path,
        step,
        round_num,
        loss,
        len(checkpoint["config"]),
    )
    return path


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
    """Load checkpoint, handling both v5 (dict) and legacy (raw state_dict) formats.

    Args:
        path:   Checkpoint file path.
        device: Device to map tensors to.

    Returns:
        state_dict: Model state dict.
        config:     Constructor args (empty dict for legacy checkpoints).
        meta:       Metadata dict (step, round, loss, timestamp, version).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = torch.load(path, map_location=device, weights_only=False)

    if isinstance(data, dict) and "state_dict" in data:
        # v5 format: structured checkpoint
        state_dict = data["state_dict"]
        config = data.get("config", {})
        meta = data.get("meta", {})
        version = meta.get("version", "unknown")
        logger.info(
            "Loaded checkpoint: %s (version=%s, step=%d, config=%d keys)",
            path,
            version,
            meta.get("step", 0),
            len(config),
        )
    elif isinstance(data, dict) and any(k.startswith("local.") or k.startswith("packing.") for k in data.keys()):
        # Legacy format: raw state_dict (keys look like module paths)
        state_dict = data
        config = {}
        meta = {"version": "legacy", "step": 0}
        logger.info("Loaded legacy checkpoint: %s (%d keys, no config)", path, len(state_dict))
    else:
        # Unknown format — try as state_dict
        state_dict = data
        config = {}
        meta = {"version": "unknown", "step": 0}
        logger.warning("Unknown checkpoint format: %s — treating as raw state_dict", path)

    return state_dict, config, meta


def build_model_config(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract current config from a live model for checkpoint saving.

    Captures all buffer values and architecture params that would be
    needed to reconstruct the model from scratch.
    """
    config: Dict[str, Any] = {}

    # Model-level gates
    for gate_name in ("gate_local", "gate_repulsion", "gate_secondary", "gate_packing"):
        if hasattr(model, gate_name):
            config[gate_name] = float(getattr(model, gate_name).item())

    # Packing term
    pack = getattr(model, "packing", None)
    if pack is not None:
        config["packing"] = {}
        # Scalar and vector buffers
        for buf_name in (
            "rg_lambda",
            "rg_r0",
            "rg_nu",
            "rg_dead_zone",
            "rg_m",
            "rg_alpha",
            "coord_lambda",
            "coord_m",
            "coord_alpha",
            "rho_sigma",
            "rho_m",
            "rho_alpha",
            "rho_penalty_lambda",
            "rho_fit_a",
            "rho_fit_b",
            "rho_fit_c",
        ):
            buf = getattr(pack, buf_name, None)
            if buf is not None:
                config["packing"][buf_name] = buf.cpu().tolist() if buf.numel() > 1 else float(buf.item())

        # Architecture
        config["packing"]["topk"] = pack.topk
        config["packing"]["exclude"] = pack.exclude
        config["packing"]["max_dist"] = pack.max_dist
        config["packing"]["normalize_by_length"] = pack.normalize_by_length
        config["packing"]["num_aa"] = pack.num_aa

        # Per-AA coordination stats (lists)
        burial = getattr(pack, "burial", None)
        if burial is not None:
            if hasattr(burial, "n_star"):
                config["packing"]["n_mean"] = burial.n_star.cpu().tolist()
            if hasattr(burial, "sigma"):
                config["packing"]["n_std"] = burial.sigma.cpu().tolist()
            config["packing"]["lambda_hp"] = float(burial.lambda_hp.item())

            # Sigmoid params
            config["packing"]["sigmoid_r_half"] = burial.COORD_R_HALF
            config["packing"]["sigmoid_tau"] = burial.COORD_TAU

        # Coord bounds
        if hasattr(pack, "coord_n_lo"):
            config["packing"]["n_lo"] = pack.coord_n_lo.cpu().tolist()
        if hasattr(pack, "coord_n_hi"):
            config["packing"]["n_hi"] = pack.coord_n_hi.cpu().tolist()

        # Lambda rho (trainable)
        if hasattr(pack, "lambda_rho"):
            config["packing"]["lambda_rho"] = float(pack.lambda_rho.item())

    # Local term
    local = getattr(model, "local", None)
    if local is not None:
        config["local"] = {}
        if hasattr(local, "weight"):
            config["local"]["weight"] = float(local.weight.item())
        if hasattr(local, "window_size"):
            config["local"]["window_size"] = local.window_size

    # Secondary term
    sec = getattr(model, "secondary", None)
    if sec is not None:
        config["secondary"] = {}
        if hasattr(sec, "ram_weight"):
            config["secondary"]["ram_weight"] = float(sec.ram_weight.item())

    # Repulsion term
    rep = getattr(model, "repulsion", None)
    if rep is not None:
        config["repulsion"] = {}
        if hasattr(rep, "lambda_rep"):
            config["repulsion"]["lambda_rep"] = float(rep.lambda_rep.item())

    return config


def apply_config_overrides(
    model: torch.nn.Module,
    config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """Apply config values to model buffers after load.

    Used to override checkpoint values with current intended values.
    Replaces the manual buffer re-override pattern.

    Args:
        model:     The model after load_state_dict.
        config:    Config from checkpoint (for logging old values).
        overrides: Dict of {buffer_path: value} to apply.
                   e.g. {"packing.rg_lambda": 1.0, "packing.coord_lambda": 0.1}
    """
    if not overrides:
        return

    for path, new_val in overrides.items():
        parts = path.split(".")
        obj = model
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
            buf_name = parts[-1]
            buf = getattr(obj, buf_name, None)
            if buf is not None and isinstance(buf, torch.Tensor):
                old_val = buf.item()
                buf.fill_(float(new_val))
                if abs(old_val - new_val) > 1e-6:
                    logger.info("  Buffer override: %s  %.4f → %.4f", path, old_val, new_val)
        except (AttributeError, RuntimeError) as e:
            logger.warning("  Failed to override %s: %s", path, e)
