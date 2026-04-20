# src/calphaebm/cli/commands/train/model_builder.py
"""Model construction and sanity checks."""

from typing import Any, Dict, Set

import torch

from calphaebm.defaults import MODEL as _M
from calphaebm.models.energy import create_total_energy
from calphaebm.utils.constants import EMB_DIM
from calphaebm.utils.logging import get_logger

logger = get_logger()


def _learnable_buffer_flags(args) -> dict:
    """Extract learnable buffer flags from CLI args.

    --learn-all-buffers sets all individual flags to True,
    EXCEPT learn_gate_geometry (forward is detached, no gradients flow).
    """
    all_on = getattr(args, "learn_all_buffers", False)
    return {
        "learn_packing_coords": all_on or getattr(args, "learn_packing_coords", False),
        "learn_packing_density": all_on or getattr(args, "learn_packing_density", False),
        "learn_penalty_shapes": all_on or getattr(args, "learn_penalty_shapes", False),
        "learn_packing_bounds": all_on or getattr(args, "learn_packing_bounds", False),
        "learn_penalty_strengths": all_on or getattr(args, "learn_penalty_strengths", False),
        "learn_gate_geometry": getattr(args, "learn_gate_geometry", False),  # explicit only
        "learn_hbond_geometry": all_on or getattr(args, "learn_hbond_geometry", False),
    }


def build_model(terms_set: Set[str], device: torch.device, args) -> torch.nn.Module:
    """Build energy model using the factory function.

    Each module that needs a Rama validity gate (packing, local, secondary)
    loads its basin peaks directly from the basin surface files via
    RamaValidityGate.from_data_dir() or .from_basin_surfaces() at
    construction time.  No post-construction override needed.
    """
    include_repulsion = "repulsion" in terms_set
    include_secondary = "secondary" in terms_set
    include_packing = "packing" in terms_set

    logger.debug(
        "Building model: repulsion=%s secondary=%s packing=%s",
        include_repulsion,
        include_secondary,
        include_packing,
    )

    model_kwargs: Dict[str, Any] = {
        # Paths
        "backbone_data_dir": args.backbone_data_dir,
        "secondary_data_dir": args.secondary_data_dir,
        "repulsion_data_dir": args.repulsion_data_dir,
        "packing_data_dir": args.packing_data_dir,
        "device": device,
        # Architecture
        "emb_dim": EMB_DIM,
        "hidden_dims": (128, 128),
        # Nonbonded parameters
        "K_neighbors": args.repulsion_K,
        "exclude_nonbonded": args.repulsion_exclude,
        "repulsion_r_on": args.repulsion_r_on,
        "repulsion_r_cut": args.repulsion_r_cut,
        "packing_r_on": args.packing_r_on,
        "packing_r_cut": args.packing_r_cut,
        # Packing-specific
        "packing_short_gate_on": args.packing_short_gate_on,
        "packing_short_gate_off": args.packing_short_gate_off,
        "packing_rbf_centers": tuple(args.packing_rbf_centers),
        "packing_rbf_width": args.packing_rbf_width,
        "packing_max_dist": args.packing_max_dist,
        "packing_init_from": args.packing_init_from,
        "packing_normalize_by_length": args.packing_normalize_by_length,
        "packing_debug_scale": getattr(args, "packing_debug_scale", False),
        "packing_debug_every": getattr(args, "packing_debug_every", 200),
        # Rg Flory size restraint (part of packing energy)
        "packing_rg_lambda": getattr(args, "packing_rg_lambda", _M["rg_lambda"]),
        "packing_rg_r0": getattr(args, "packing_rg_r0", _M["rg_r0"]),
        "packing_rg_nu": getattr(args, "packing_rg_nu", _M["rg_nu"]),
        # Coordination number penalty (part of packing energy)
        "coord_lambda": getattr(args, "hp_penalty_lambda", getattr(args, "coord_lambda", _M["coord_lambda"])),
        # Local term init weights
        "init_theta_theta_weight": args.init_theta_theta_weight,
        "init_delta_phi_weight": args.init_delta_phi_weight,
        "local_window_size": getattr(args, "local_window_size", 8),
        # Term inclusion
        "include_repulsion": include_repulsion,
        "include_secondary": include_secondary,
        "include_packing": include_packing,
        # Debug
        "debug_mode": args.debug_mode,
        # Learnable buffer flags
        **_learnable_buffer_flags(args),
    }

    # Load coordination stats + ρ params from JSON (v6: group-conditional)
    coord_n_star_file = getattr(args, "coord_n_star_file", None)
    if coord_n_star_file is not None:
        import json
        from pathlib import Path as _Path

        _nstar_path = _Path(coord_n_star_file)
        if _nstar_path.exists():
            with open(_nstar_path) as _f:
                _nstar = json.load(_f)

            _style = _nstar.get("sigmoid_style", "unknown")
            _n_grps = _nstar.get("n_groups", 5)
            _grp_names = _nstar.get("group_names", [])
            logger.info("Loaded coord stats from %s  (style=%s, n_groups=%d)", coord_n_star_file, _style, _n_grps)

            # ── Build packing_extra (v6) ──────────────────────────────────
            _packing_extra = {}

            # -- Group assignment [20] AA index -> group index
            _ga = _nstar.get("group_assignment")
            if _ga is not None:
                _packing_extra["group_assignment"] = _ga
                logger.info("  group_assignment: %s", _ga)
            else:
                logger.warning("  group_assignment not in JSON — using built-in default")

            # -- Group-conditional n_i^(k) centers [20 × 5]
            _ng_mean = _nstar.get("n_group_mean_list")
            _ng_std = _nstar.get("n_group_std_list")
            _ng_lo = _nstar.get("n_group_lo_list")
            _ng_hi = _nstar.get("n_group_hi_list")

            if _ng_mean is not None:
                _packing_extra["n_group_mean"] = _ng_mean
                logger.info(
                    "  n_group_mean [20×5]: loaded (range %.2f – %.2f)",
                    min(v for row in _ng_mean for v in row),
                    max(v for row in _ng_mean for v in row),
                )
            else:
                logger.warning("  n_group_mean_list not found — E_hp will use defaults")

            if _ng_std is not None:
                _packing_extra["n_group_std"] = _ng_std
                _flat_std = [v for row in _ng_std for v in row]
                n_clamped = sum(1 for v in _flat_std if v < 0.8)
                logger.info(
                    "  n_group_std  [20×5]: loaded (range %.2f – %.2f, " "%d/%d entries will be clamped to 0.8)",
                    min(_flat_std),
                    max(_flat_std),
                    n_clamped,
                    len(_flat_std),
                )
            else:
                logger.warning("  n_group_std_list not found — E_hp will use sigma=0.8")

            if _ng_lo is not None and _ng_hi is not None:
                _packing_extra["n_group_lo"] = _ng_lo
                _packing_extra["n_group_hi"] = _ng_hi
                logger.info("  n_group_lo/hi [20×5]: loaded — E_hp_pen per-group bands")
            else:
                logger.warning(
                    "  n_group_lo/hi not found — E_hp_pen will have no bands " "(setting coord_lambda=0 to suppress)"
                )
                model_kwargs["coord_lambda"] = 0.0

            # -- Per-group ρ^(k)*(L) curves [5 dicts]
            _rho_grp_fits = _nstar.get("rho_group_fits")
            if _rho_grp_fits is not None:
                _packing_extra["rho_group_fits"] = _rho_grp_fits
                logger.info("  rho_group_fits [5]:")
                for _k, _fit in enumerate(_rho_grp_fits):
                    _gname = _grp_names[_k] if _k < len(_grp_names) else str(_k)
                    logger.info(
                        "    k=%d (%s): ρ*(L)=%.3f - %.3f·exp(-L/%.1f)",
                        _k,
                        _gname,
                        _fit.get("fit_a", 0),
                        _fit.get("fit_b", 0),
                        _fit.get("fit_c", 1),
                    )
            else:
                logger.warning("  rho_group_fits not found — E_rho will use scalar defaults")

            # -- Per-group ρ^(k) sigma [5]
            _rho_grp_sigma = _nstar.get("rho_group_sigma")
            if _rho_grp_sigma is not None:
                _packing_extra["rho_group_sigma"] = _rho_grp_sigma
                logger.info("  rho_group_sigma [5]: %s", ", ".join(f"{v:.3f}" for v in _rho_grp_sigma))
            else:
                logger.warning("  rho_group_sigma not found — E_rho will use σ=0.3 per group")

            # -- Per-group ρ^(k) penalty bounds [5]
            _rho_grp_lo = _nstar.get("rho_group_lo")
            _rho_grp_hi = _nstar.get("rho_group_hi")
            if _rho_grp_lo is not None and _rho_grp_hi is not None:
                _packing_extra["rho_group_lo"] = _rho_grp_lo
                _packing_extra["rho_group_hi"] = _rho_grp_hi
                logger.info("  rho_group_lo [5]: %s", ", ".join(f"{v:.3f}" for v in _rho_grp_lo))
                logger.info("  rho_group_hi [5]: %s", ", ".join(f"{v:.3f}" for v in _rho_grp_hi))
            else:
                logger.warning(
                    "  rho_group_lo/hi not found — E_rho_pen will derive " "bounds from rho_star ± 1.35·sigma"
                )

            # -- Constraint and penalty hyperparams from CLI/config
            _packing_extra["rg_dead_zone"] = getattr(args, "packing_rg_dead_zone", _M["rg_dead_zone"])
            _packing_extra["rg_m"] = getattr(args, "packing_rg_m", _M["rg_m"])
            _packing_extra["rg_alpha"] = getattr(args, "packing_rg_alpha", _M["rg_alpha"])
            _packing_extra["coord_m"] = getattr(args, "packing_coord_m", _M["coord_m"])
            _packing_extra["coord_alpha"] = getattr(args, "packing_coord_alpha", _M["coord_alpha"])
            _packing_extra["rho_lambda"] = getattr(args, "packing_rho_lambda", _M["rho_lambda"])
            _packing_extra["rho_m"] = getattr(args, "packing_rho_m", _M["rho_m"])
            _packing_extra["rho_alpha"] = getattr(args, "packing_rho_alpha", _M["rho_alpha"])
            _packing_extra["rho_penalty_lambda"] = getattr(
                args,
                "rho_penalty_lambda",
                getattr(args, "packing_rho_penalty_lambda", _M["rho_penalty_lambda"]),
            )

            model_kwargs["packing_extra"] = _packing_extra

        else:
            logger.warning("coord_n_star_file not found: %s — packing will use defaults", coord_n_star_file)

    model = create_total_energy(**model_kwargs)

    return model


def verify_model_terms(model: torch.nn.Module, phase: str) -> bool:
    """Verify that the model has the required terms for the phase."""
    has_local = hasattr(model, "local") and model.local is not None
    has_repulsion = hasattr(model, "repulsion") and model.repulsion is not None
    has_secondary = hasattr(model, "secondary") and model.secondary is not None
    has_packing = hasattr(model, "packing") and model.packing is not None

    logger.debug(
        "Model terms: local=%s repulsion=%s secondary=%s packing=%s",
        has_local,
        has_repulsion,
        has_secondary,
        has_packing,
    )

    if phase == "repulsion" and not has_repulsion:
        logger.error("Repulsion phase requires repulsion term, but it is missing")
        return False

    return True
