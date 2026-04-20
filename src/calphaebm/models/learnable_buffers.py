# src/calphaebm/models/learnable_buffers.py
"""Utility for optionally-learnable buffer parameters with drift tracking.

Usage:
    from calphaebm.models.learnable_buffers import reg, buffer_drift_report

    class MyModule(nn.Module):
        def __init__(self, learn_buffers=False):
            super().__init__()
            reg(self, 'sigma', torch.tensor(0.8), learnable=learn_buffers)

    # Later, in diagnostics:
    report = buffer_drift_report(model)
    for line in report:
        logger.info(line)

State-dict keys are identical regardless of learnable flag,
so checkpoints load across configurations.  Initial values are
stored as non-persistent buffers (not saved in checkpoints).
"""

import torch
import torch.nn as nn


def reg(
    module: nn.Module,
    name: str,
    tensor: torch.Tensor,
    learnable: bool = False,
) -> None:
    """Register a tensor as either a learnable nn.Parameter or a fixed buffer.

    When learnable=True, also stores the initial value as a non-persistent
    buffer ``_init_{name}`` for drift tracking.

    Parameters
    ----------
    module : nn.Module
        The module to register on.
    name : str
        Attribute name.
    tensor : torch.Tensor
        Initial value (will be cloned).
    learnable : bool
        If True -> nn.Parameter (included in optimizer, gets gradients).
        If False -> buffer (not in optimizer, no gradients).
    """
    t = tensor.clone().detach()
    if learnable:
        setattr(module, name, nn.Parameter(t.clone()))
        # Store initial value for drift tracking (not saved in checkpoint)
        module.register_buffer(f"_init_{name}", t.clone(), persistent=False)
    else:
        module.register_buffer(name, t)


def buffer_drift_report(model: nn.Module, top_n: int = 20) -> list[str]:
    """Walk model tree and report drift of all learnable buffers from init.

    Returns a list of formatted log lines showing:
      - Parameter path
      - Shape
      - Init mean -> current mean
      - Max |drift|
      - Drift as % of init magnitude

    Only reports parameters that have a matching ``_init_*`` buffer.
    Sorted by max |drift| descending.
    """
    drifts = []

    for mod_name, mod in model.named_modules():
        for attr_name in list(mod._parameters.keys()):
            init_name = f"_init_{attr_name}"
            if init_name not in mod._buffers:
                continue

            param = mod._parameters[attr_name]
            init_val = mod._buffers[init_name]

            if param is None or init_val is None:
                continue

            delta = param.data - init_val
            path = f"{mod_name}.{attr_name}" if mod_name else attr_name

            init_mean = init_val.mean().item()
            curr_mean = param.data.mean().item()
            delta_mean = delta.mean().item()
            max_abs_drift = delta.abs().max().item()
            rms_drift = delta.pow(2).mean().sqrt().item()
            init_mag = init_val.abs().mean().item()
            pct = (max_abs_drift / max(init_mag, 1e-8)) * 100

            drifts.append(
                {
                    "path": path,
                    "shape": list(param.shape),
                    "init_mean": init_mean,
                    "curr_mean": curr_mean,
                    "delta_mean": delta_mean,
                    "max_drift": max_abs_drift,
                    "rms_drift": rms_drift,
                    "pct": pct,
                    "numel": param.numel(),
                }
            )

    # Sort by max drift descending
    drifts.sort(key=lambda d: d["max_drift"], reverse=True)

    lines = []
    if not drifts:
        lines.append("  Learnable buffers: none active")
        return lines

    total_params = sum(d["numel"] for d in drifts)
    lines.append(f"  Learnable buffers: {len(drifts)} groups, {total_params} params")
    lines.append(f'  {"path":<45s} {"shape":<14s} {"init->curr":<24s} {"max|d|":>8s} {"rms":>8s} {"d%":>7s}')
    lines.append(f'  {"."*45} {"."*14} {"."*24} {"."*8} {"."*8} {"."*7}')

    for d in drifts[:top_n]:
        shape_str = str(d["shape"])
        arrow = f'{d["init_mean"]:+.4f}->{d["curr_mean"]:+.4f}'
        lines.append(
            f'  {d["path"]:<45s} {shape_str:<14s} {arrow:<24s} '
            f'{d["max_drift"]:8.4f} {d["rms_drift"]:8.4f} {d["pct"]:6.1f}%'
        )

    if len(drifts) > top_n:
        lines.append(f"  ... and {len(drifts) - top_n} more (sorted by max|d|)")

    return lines
