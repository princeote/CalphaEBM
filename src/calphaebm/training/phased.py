"""Main phased trainer - orchestrates different training phases."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from calphaebm.training.core.config import PhaseConfig
from calphaebm.training.core.state import TrainingState
from calphaebm.training.core.trainer import BaseTrainer
from calphaebm.training.phases.full_phase import run_full_phase
from calphaebm.training.phases.local_phase import run_local_phase
from calphaebm.training.phases.packing_phase import run_packing_phase
from calphaebm.training.phases.repulsion_phase import run_repulsion_phase
from calphaebm.training.phases.secondary_phase import run_secondary_phase
from calphaebm.utils.logging import get_logger

logger = get_logger()


def _get_packing_ramp_from_config(config: PhaseConfig) -> Tuple[Optional[float], Optional[float], int]:
    """Extract a packing ramp (start, end, steps) from config if present.

    Supports both styles:
      1) packing-specific fields: ramp_pack_start / ramp_pack_end / ramp_steps
      2) full-phase dict fields: ramp_start["packing"] / ramp_end["packing"] / ramp_steps
    """
    ramp_steps = int(getattr(config, "ramp_steps", 0) or 0)

    ramp_pack_start = getattr(config, "ramp_pack_start", None)
    ramp_pack_end = getattr(config, "ramp_pack_end", None)

    # Fall back to dict-style ramps (used for full phase)
    ramp_start_dict = getattr(config, "ramp_start", None) or {}
    ramp_end_dict = getattr(config, "ramp_end", None) or {}

    if ramp_pack_start is None and isinstance(ramp_start_dict, dict):
        ramp_pack_start = ramp_start_dict.get("packing", None)
    if ramp_pack_end is None and isinstance(ramp_end_dict, dict):
        ramp_pack_end = ramp_end_dict.get("packing", None)

    start = float(ramp_pack_start) if ramp_pack_start is not None else None
    end = float(ramp_pack_end) if ramp_pack_end is not None else None

    return start, end, ramp_steps


class PhasedTrainer(BaseTrainer):
    """Manager for phased training of energy models."""

    PHASE_RUNNERS = {
        "local": run_local_phase,
        "secondary": run_secondary_phase,
        "repulsion": run_repulsion_phase,
        "packing": run_packing_phase,
        "full": run_full_phase,
    }

    def run_phase(
        self,
        config: PhaseConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        native_structures: Optional[Dict[str, torch.Tensor]] = None,
        resume: Optional[str] = None,
    ) -> TrainingState:
        self.config = config
        self.converged = False
        self.convergence_step = None

        self._setup_convergence_monitor(config)

        logger.info("Starting phase: %s", config.name)
        logger.info("Active terms: %s", config.terms)
        logger.debug("Frozen terms: %s", config.freeze)
        if getattr(config, "lr_schedule", None):
            logger.debug("LR schedule: %s (%s -> %s)", config.lr_schedule, config.lr, config.lr_final)

        logger.debug("Convergence criteria:")
        logger.debug("  Loss window: %d steps", int(self.convergence_config.loss_window))
        logger.debug("  Loss tolerance: %.1f%%", float(self.convergence_config.loss_tolerance) * 100.0)

        if hasattr(self.model, "get_gates"):
            logger.debug("Current gates before phase: %s", self.model.get_gates())
        else:
            logger.debug("Model does not have get_gates method")

        # Detect whether packing ramp is configured for this phase.
        pack_ramp_start, pack_ramp_end, ramp_steps = _get_packing_ramp_from_config(config)
        packing_ramp_configured = (
            config.name == "packing" and pack_ramp_start is not None and pack_ramp_end is not None and ramp_steps > 0
        )

        # Configure gates:
        # - keep existing values for active terms
        # - set inactive terms to 0.0
        #
        # IMPORTANT:
        # - If we are in the packing phase AND a packing ramp is configured, we must NOT
        #   leave packing at 0.0 (checkpoint could have 0.0) and must NOT “safeguard” it to 1.0.
        #   Instead, set packing to the ramp start here (packing_phase will take over thereafter).
        if hasattr(self.model, "set_gates"):
            all_terms = ["local", "repulsion", "secondary", "packing"]
            current_gates = self.model.get_gates() if hasattr(self.model, "get_gates") else {}

            gates: Dict[str, float] = {}
            for term in all_terms:
                if term in config.terms:
                    if term == "packing" and packing_ramp_configured:
                        gates[term] = float(pack_ramp_start)
                    else:
                        gates[term] = float(current_gates.get(term, 1.0))
                else:
                    gates[term] = 0.0

            self.model.set_gates(**gates)
            logger.debug("Gates set to: %s", gates)

            # Safeguard: ensure active terms are not accidentally zeroed.
            # Do NOT overwrite the packing ramp start (packing phase).
            if hasattr(self.model, "get_gates"):
                final_gates = self.model.get_gates()

                for term in config.terms:
                    if term not in final_gates:
                        continue
                    if float(final_gates[term]) != 0.0:
                        continue

                    if term == "packing" and packing_ramp_configured:
                        safe_start = float(pack_ramp_start)
                        self.model.set_gates(packing=safe_start)
                        logger.warning(
                            "Active term packing has gate 0.0 with ramp configured; setting packing gate to ramp start %.4f",
                            safe_start,
                        )
                        continue

                    logger.warning("Active term %s has gate 0.0; setting to 1.0", term)
                    self.model.set_gates(**{term: 1.0})

                logger.debug("Final gates after verification: %s", self.model.get_gates())
        else:
            logger.warning("Model does not have set_gates method - cannot configure gates")

        # Apply freezing - affects gradients only
        from calphaebm.training.core.freeze import freeze_module

        for term in config.freeze:
            if hasattr(self.model, term) and getattr(self.model, term) is not None:
                freeze_module(getattr(self.model, term))

        phase_runner = self.PHASE_RUNNERS.get(config.name)
        if phase_runner is None:
            raise ValueError(f"Unknown phase: {config.name}")

        state = phase_runner(
            trainer=self,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            native_structures=native_structures,
            resume=resume,
        )

        from calphaebm.training.core.freeze import unfreeze_module

        for term in config.freeze:
            if hasattr(self.model, term) and getattr(self.model, term) is not None:
                unfreeze_module(getattr(self.model, term))

        return state
