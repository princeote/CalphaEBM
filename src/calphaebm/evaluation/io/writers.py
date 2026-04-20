# src/calphaebm/evaluation/io/writers.py

"""Save evaluation results to various formats."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union


def save_metrics_json(
    metrics: Dict[str, Any],
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """Save metrics to JSON file."""
    path = Path(path)

    # Convert numpy types to Python native
    clean = {}
    for k, v in metrics.items():
        if hasattr(v, "item"):
            clean[k] = v.item()
        elif isinstance(v, (list, tuple)) and len(v) > 0 and hasattr(v[0], "item"):
            # BUG FIX: guard len(v) > 0 before accessing v[0]
            clean[k] = [x.item() for x in v]
        else:
            clean[k] = v

    with open(path, "w") as f:
        json.dump(clean, f, indent=indent)


def save_metrics_txt(
    metrics: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """Save metrics as human-readable text.

    If metrics contains a single key 'summary' whose value is a multi-line
    string, write it directly rather than routing through the k/v formatter
    (which would collapse newlines onto a single line).
    """
    path = Path(path)

    # BUG FIX: if this is just a summary string wrapper, write it directly
    if set(metrics.keys()) == {"summary"} and isinstance(metrics["summary"], str):
        with open(path, "w") as f:
            f.write(metrics["summary"])
        return

    lines = []
    lines.append("=" * 60)
    lines.append("CalphaEBM Evaluation Results")
    lines.append("=" * 60)
    lines.append("")

    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"{k:30s}: {v:.6f}")
        elif isinstance(v, int):
            lines.append(f"{k:30s}: {v:d}")
        elif isinstance(v, str):
            lines.append(v)  # write multi-line strings as-is, no prefix
        elif isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], float):
                mean_val = sum(v) / len(v)
                lines.append(f"{k:30s}: [{len(v)} values] mean = {mean_val:.6f}")
            else:
                lines.append(f"{k:30s}: {v}")
        else:
            lines.append(f"{k:30s}: {v}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def save_metrics_csv(
    metrics: Dict[str, List[float]],
    path: Union[str, Path],
) -> None:
    """Save time-series metrics to CSV."""
    path = Path(path)

    # Find all keys with lists of same length
    list_metrics = {k: v for k, v in metrics.items() if isinstance(v, (list, tuple))}

    if not list_metrics:
        raise ValueError("No list metrics to save")

    # Check lengths match
    lengths = {len(v) for v in list_metrics.values()}
    if len(lengths) > 1:
        raise ValueError(f"Inconsistent lengths: {lengths}")

    n = lengths.pop()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list_metrics.keys())

        for i in range(n):
            row = [list_metrics[k][i] for k in list_metrics.keys()]
            writer.writerow(row)
