"""Phase-specific helpers for term selection and config choices."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from calphaebm.utils.logging import get_logger

logger = get_logger()

ALL_TERMS: Set[str] = {"local", "repulsion", "secondary", "packing"}


def _normalize_terms(energy_terms: Sequence[str]) -> Set[str]:
    """Lowercase + strip, and expand 'all' if present."""
    terms = {str(t).strip().lower() for t in energy_terms if str(t).strip()}
    if "all" in terms:
        return set(ALL_TERMS)
    return terms


def determine_terms_for_phase(phase: str, energy_terms: Sequence[str]) -> Set[str]:
    """Determine which terms should be included based on phase + user selection.

    Rules:
    - local is always included
    - 'all' expands to all terms
    - each phase may enforce additional terms for safety/consistency
    """
    phase = str(phase).strip().lower()
    terms = _normalize_terms(energy_terms)

    # Local is always present
    terms.add("local")

    if phase == "local":
        # local-only training
        terms.discard("repulsion")
        terms.discard("secondary")
        terms.discard("packing")
        logger.info("Phase local: using local only")
        return terms

    if phase == "secondary":
        # Secondary training typically uses repulsion for safety
        terms.update({"secondary", "repulsion"})
        logger.info("Phase secondary: using local + secondary + repulsion for safety")
        return terms

    if phase == "repulsion":
        terms.add("repulsion")
        # If the user requested secondary (or 'all'), keep it for realistic sampling / calibration.
        # Otherwise do a strict safety pass with local+repulsion only.
        if "secondary" in terms:
            logger.info("Phase repulsion: including secondary term (calibration / realistic sampling)")
        else:
            terms.discard("secondary")
            terms.discard("packing")
            logger.info("Phase repulsion: using local + repulsion only (safety pass)")
        return terms

    if phase == "packing":
        # Packing training expects a realistic background landscape; keep repulsion+secondary on.
        terms.update({"repulsion", "secondary", "packing"})
        logger.info("Phase packing: using local + repulsion + secondary + packing")
        return terms

    if phase == "full":
        terms.update({"repulsion", "secondary", "packing"})
        logger.info("Phase full: using all terms")
        return terms

    raise ValueError(f"Unknown phase: {phase}")


def get_active_terms_for_phase(phase: str, model_has_secondary: bool = False) -> List[str]:
    """Get the active term list used for reporting / phase bookkeeping.

    Notes:
    - Repulsion phase may optionally include secondary if the model has it.
    """
    phase = str(phase).strip().lower()

    if phase == "repulsion":
        base = ["local", "repulsion"]
        if model_has_secondary:
            logger.info("Repulsion phase: including secondary in active terms")
            return base + ["secondary"]
        logger.info("Repulsion phase: local + repulsion only")
        return base

    active_terms_map: Dict[str, List[str]] = {
        "local": ["local"],
        "secondary": ["local", "secondary", "repulsion"],
        "packing": ["local", "secondary", "repulsion", "packing"],
        "full": ["local", "secondary", "repulsion", "packing"],
    }
    if phase not in active_terms_map:
        raise ValueError(f"Unknown phase: {phase}")
    return active_terms_map[phase]


def get_loss_fn_for_phase(phase: str) -> str:
    """Get the loss function name for a given phase."""
    phase = str(phase).strip().lower()
    loss_fns = {
        "local": "dsm",
        "secondary": "contrastive_secondary",
        "repulsion": "repulsion_calibrate",
        "packing": "contrastive_packing",
        "full": "dsm_low_lr",
    }
    if phase not in loss_fns:
        raise ValueError(f"Unknown phase: {phase}")
    return loss_fns[phase]


def get_validate_every_for_phase(phase: str, validate_every: int) -> int:
    """Determine validation frequency based on phase."""
    phase = str(phase).strip().lower()
    if phase in ("local", "secondary", "packing", "full"):
        return int(validate_every)
    if phase == "repulsion":
        # calibration-only
        return 0
    raise ValueError(f"Unknown phase: {phase}")


def get_ramp_config(args) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    """Return (ramp_start, ramp_end) dictionaries for gate ramping.

    This is used by the CLI to populate PhaseConfig fields.

    Supported:
    - Full phase: requires --ramp-gates and uses per-term start/end flags.
    - Packing phase: uses --ramp-pack-start/--ramp-pack-end with --ramp-steps (>0).
      This is *packing-only* and does NOT require --ramp-gates.
    """
    phase = str(getattr(args, "phase", "")).strip().lower()

    # Full-phase multi-gate ramp (explicit opt-in)
    if phase == "full" and bool(getattr(args, "ramp_gates", False)):
        ramp_start = {
            "local": float(getattr(args, "ramp_local_start")),
            "repulsion": float(getattr(args, "ramp_rep_start")),
            "secondary": float(getattr(args, "ramp_ss_start")),
            "packing": float(getattr(args, "ramp_pack_start")),
        }
        ramp_end = {
            "local": float(getattr(args, "ramp_local_end")),
            "repulsion": float(getattr(args, "ramp_rep_end")),
            "secondary": float(getattr(args, "ramp_ss_end")),
            "packing": float(getattr(args, "ramp_pack_end")),
        }
        return ramp_start, ramp_end

    # Packing-only ramp (outer packing gate), independent of --ramp-gates
    if phase == "packing":
        start = getattr(args, "ramp_pack_start", None)
        end = getattr(args, "ramp_pack_end", None)
        steps = int(getattr(args, "ramp_steps", 0) or 0)
        if start is not None and end is not None and steps > 0:
            ramp_start = {"packing": float(start)}
            ramp_end = {"packing": float(end)}
            return ramp_start, ramp_end

    return None, None
