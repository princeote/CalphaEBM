# src/calphaebm/training/phases/full_phase.py
"""Phase 6: Full fine-tuning phase with all terms trainable (IC version).

Uses dsm_ic_loss, force_balance_ic_loss, local_geogap_ic_loss.
All sigma values are in radians. bond_spring is gone.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from calphaebm.training.core.state import TrainingState
from calphaebm.training.losses.balance_loss import energy_balance_loss
from calphaebm.training.losses.contrastive_losses import packing_contrastive_loss
from calphaebm.training.losses.discrim_loss import subterm_discrimination_loss
from calphaebm.training.losses.dsm import dsm_ic_loss
from calphaebm.training.losses.elt_losses import elt_frustration_loss, elt_funnel_loss
from calphaebm.training.losses.force_balance import force_balance_ic_diagnostics, force_balance_ic_loss
from calphaebm.training.losses.local_geogap_loss import local_geogap_ic_diagnostics, local_geogap_ic_loss
from calphaebm.training.losses.native_gap_loss import native_gap_loss
from calphaebm.training.losses.rg_contrastive_loss import rg_contrastive_loss
from calphaebm.training.losses.secondary_basin_loss import SecondaryBasinLoss, secondary_basin_diagnostics
from calphaebm.utils.logging import ProgressBar, get_logger
from calphaebm.utils.neighbors import pairwise_distances

logger = get_logger()


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _compute_ramp_gates(
    phase_step: int,
    ramp_steps: int,
    ramp_start: Dict[str, float],
    ramp_end: Dict[str, float],
    use_quadratic: bool = False,
) -> Dict[str, float]:
    if phase_step >= ramp_steps:
        return dict(ramp_end)
    progress_frac = _clamp01(phase_step / float(max(ramp_steps, 1)))
    rep_progress = _clamp01(progress_frac * progress_frac) if use_quadratic else progress_frac
    gates: Dict[str, float] = {}
    for gate_name, start_val in ramp_start.items():
        end_val = ramp_end.get(gate_name, start_val)
        if gate_name == "repulsion" and use_quadratic:
            gates[gate_name] = start_val + (end_val - start_val) * rep_progress
        else:
            gates[gate_name] = start_val + (end_val - start_val) * progress_frac
    return gates


def _is_gate_param(name: str, p: torch.nn.Parameter) -> bool:
    if "gate_" in name:
        return True
    if p.numel() == 1:
        tail = name.split(".")[-1]
        if tail in {"gate_local", "gate_repulsion", "gate_secondary", "gate_packing"}:
            return True
    return False


def _setup_optimizer_with_frozen_gates(
    trainer,
    config,
    ramp_active: bool,
    ramp_steps: int,
    gate_lr_mult: float = 1.0,
) -> Tuple[Optional[torch.optim.Optimizer], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    gate_params: List[torch.nn.Parameter] = []
    main_params: List[torch.nn.Parameter] = []
    scalar_params: List[torch.nn.Parameter] = []
    frozen_scalar_params: List[torch.nn.Parameter] = []

    freeze_packing_scalar = bool(getattr(config, "freeze_packing_scalar", False))
    if freeze_packing_scalar:
        logger.debug("freeze_packing_scalar=True: λ_pack will be frozen at lr=0")

    for name, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_gate_param(name, p):
            gate_params.append(p)
        elif p.numel() == 1:
            if freeze_packing_scalar and "_lambda_pack_raw" in name:
                frozen_scalar_params.append(p)
            else:
                scalar_params.append(p)
        else:
            main_params.append(p)

    logger.debug(
        "Optimizer param groups: %d trainable scalar-λ | %d frozen scalar-λ (lr=0) | %d MLP | %d gate",
        len(scalar_params),
        len(frozen_scalar_params),
        len(main_params),
        len(gate_params),
    )

    if not main_params and not gate_params and not scalar_params and not frozen_scalar_params:
        return None, [], []

    lr = float(getattr(config, "lr", 0.0))
    weight_decay = float(getattr(config, "weight_decay", 0.0) or 0.0)
    scalar_lr_mult = float(getattr(config, "scalar_lr_mult", 1.0))
    scalar_lr = lr * scalar_lr_mult
    gate_lr = 0.0 if (ramp_active and ramp_steps > 0) else lr * float(gate_lr_mult)

    if ramp_active and ramp_steps > 0:
        logger.debug("Gates frozen during ramp (gate LR = 0.0)")

    logger.debug("Optimizer LRs: MLP=%.2e  scalar-λ=%.2e (x%.0f)  gates=%.2e", lr, scalar_lr, scalar_lr_mult, gate_lr)

    param_groups = [
        {"params": main_params, "lr": lr, "weight_decay": weight_decay, "name": "mlp"},
        {"params": scalar_params, "lr": scalar_lr, "weight_decay": 0.0, "name": "scalar_lambda"},
        {"params": gate_params, "lr": gate_lr, "weight_decay": weight_decay, "name": "gates"},
        {"params": frozen_scalar_params, "lr": 0.0, "weight_decay": 0.0, "name": "scalar_lambda_frozen"},
    ]
    param_groups = [g for g in param_groups if g["params"]]

    optimizer = torch.optim.AdamW(param_groups)
    optimizer.gate_lr_mult = float(gate_lr_mult)
    optimizer.scalar_lr_mult = scalar_lr_mult

    all_trainable = main_params + gate_params
    return optimizer, all_trainable, gate_params


def _enforce_gate_lr_multiplier(optimizer: torch.optim.Optimizer) -> None:
    if optimizer is None or len(optimizer.param_groups) < 1:
        return
    gate_lr_mult = float(getattr(optimizer, "gate_lr_mult", 1.0))
    scalar_lr_mult = float(getattr(optimizer, "scalar_lr_mult", 1.0))
    main_lr = None
    for g in optimizer.param_groups:
        if g.get("name") == "mlp":
            main_lr = float(g["lr"])
            break
    if main_lr is None:
        main_lr = float(optimizer.param_groups[0]["lr"])
    for g in optimizer.param_groups:
        if g.get("name") == "scalar_lambda":
            g["lr"] = main_lr * scalar_lr_mult
        elif g.get("name") == "gates":
            g["lr"] = main_lr * gate_lr_mult


def _set_gates_trainable(gate_params: List[torch.nn.Parameter], trainable: bool) -> None:
    for p in gate_params:
        p.requires_grad_(trainable)


def _find_gate_param_names(model: torch.nn.Module) -> List[str]:
    return [name for name, p in model.named_parameters() if _is_gate_param(name, p)]


def run_full_phase(trainer, config, train_loader, val_loader=None, native_structures=None, resume=None):
    n_steps = int(getattr(config, "n_steps", 0) or 0)
    lr = float(getattr(config, "lr", 0.0))
    lr_final_raw = getattr(config, "lr_final", None)
    lr_final = float(lr_final_raw) if lr_final_raw is not None else lr
    lr_schedule = str(getattr(config, "lr_schedule", "")) or None
    save_every = int(getattr(config, "save_every", 0) or 0)
    validate_every = int(getattr(config, "validate_every", 0) or 0)

    # IC DSM sigma (radians)
    sigma_rad = float(getattr(config, "sigma_rad", getattr(config, "sigma", 0.08)))
    sigma_min_raw = getattr(config, "sigma_min_rad", getattr(config, "sigma_min", None))
    sigma_max_raw = getattr(config, "sigma_max_rad", getattr(config, "sigma_max", None))
    sigma_min = float(sigma_min_raw) if sigma_min_raw is not None else None
    sigma_max = float(sigma_max_raw) if sigma_max_raw is not None else None

    # Differential sigma (per-coordinate noise ranges)
    sigma_min_theta_raw = getattr(config, "sigma_min_theta", None)
    sigma_max_theta_raw = getattr(config, "sigma_max_theta", None)
    sigma_min_phi_raw = getattr(config, "sigma_min_phi", None)
    sigma_max_phi_raw = getattr(config, "sigma_max_phi", None)

    sigma_min_theta = float(sigma_min_theta_raw) if sigma_min_theta_raw is not None else None
    sigma_max_theta = float(sigma_max_theta_raw) if sigma_max_theta_raw is not None else None
    sigma_min_phi = float(sigma_min_phi_raw) if sigma_min_phi_raw is not None else None
    sigma_max_phi = float(sigma_max_phi_raw) if sigma_max_phi_raw is not None else None

    use_diff_sigma = (
        sigma_min_theta is not None
        and sigma_max_theta is not None
        and sigma_min_phi is not None
        and sigma_max_phi is not None
    )

    if use_diff_sigma:
        logger.debug(
            "Differential IC DSM: σ_θ ~ LogUniform(%.3f, %.3f), σ_φ ~ LogUniform(%.3f, %.3f) rad",
            sigma_min_theta,
            sigma_max_theta,
            sigma_min_phi,
            sigma_max_phi,
        )
    elif sigma_min is not None and sigma_max is not None:
        if sigma_min >= sigma_max:
            raise ValueError(f"--sigma-min-rad ({sigma_min}) must be < --sigma-max-rad ({sigma_max})")
        logger.debug("Multi-scale IC DSM: sigma ~ LogUniform(%.3f, %.3f) rad", sigma_min, sigma_max)
    else:
        logger.debug("Fixed-sigma IC DSM: sigma=%.3f rad", sigma_rad)

    # Alpha augmentation — bidirectional Rg perturbation for DSM (run53+)
    alpha_min_raw = getattr(config, "alpha_min", 0.65)
    alpha_max_raw = getattr(config, "alpha_max", 1.25)
    alpha_min = float(alpha_min_raw)
    alpha_max = float(alpha_max_raw)
    use_alpha_aug = alpha_min < alpha_max and not (alpha_min == 1.0 and alpha_max == 1.0)
    if use_alpha_aug:
        logger.info(
            "Alpha-augmented DSM: α ~ U(%.2f, %.2f), 3× forward passes per step",
            alpha_min,
            alpha_max,
        )

    # Force balance (IC)
    lambda_fb = float(getattr(config, "lambda_fb", 0.0) or 0.0)
    fb_sigma_theta = float(getattr(config, "fb_sigma_theta", getattr(config, "fb_sigma_thermal", 0.10)) or 0.10)
    fb_sigma_phi = float(getattr(config, "fb_sigma_phi", getattr(config, "fb_sigma_thermal", 0.15)) or 0.15)
    fb_clash_phi_frac = float(getattr(config, "fb_clash_phi_frac", getattr(config, "fb_clash_frac", 0.05)) or 0.05)
    fb_clash_phi_sigma = float(getattr(config, "fb_clash_phi_sigma", getattr(config, "fb_clash_sigma", 0.30)) or 0.30)
    fb_target_ss_ratio = float(getattr(config, "fb_target_ss_ratio", 2.0) or 2.0)
    fb_target_pack_ratio = float(getattr(config, "fb_target_pack_ratio", 2.0) or 2.0)
    fb_target_rep_ratio = float(getattr(config, "fb_target_rep_ratio", 2.0) or 2.0)
    fb_diag_every = int(getattr(config, "fb_diag_every", 200) or 200)
    use_force_balance = lambda_fb > 0.0

    if use_force_balance:
        logger.debug(
            "IC Force balance loss: lambda=%.3f, σ_θ=%.3f, σ_φ=%.3f, clash_phi_frac=%.2f",
            lambda_fb,
            fb_sigma_theta,
            fb_sigma_phi,
            fb_clash_phi_frac,
        )

    # Geogap (IC)
    lambda_geogap = float(getattr(config, "lambda_geogap", 0.0) or 0.0)
    geogap_margin = float(getattr(config, "geogap_margin", 2.0) or 2.0)
    geogap_theta_sigma = float(
        getattr(config, "geogap_theta_sigma", getattr(config, "geogap_angle_sigma", 0.10)) or 0.10
    )
    geogap_phi_sigma = float(
        getattr(config, "geogap_phi_sigma", getattr(config, "geogap_dihedral_sigma", 0.20)) or 0.20
    )
    geogap_frac_perturbed = float(getattr(config, "geogap_frac_perturbed", 0.3) or 0.3)
    geogap_diag_every = int(getattr(config, "geogap_diag_every", 200) or 200)
    use_geogap = lambda_geogap > 0.0

    if use_geogap:
        logger.debug(
            "IC Geogap loss: lambda=%.3f, margin=%.2f, θ_sigma=%.3f, φ_sigma=%.3f, frac=%.2f",
            lambda_geogap,
            geogap_margin,
            geogap_theta_sigma,
            geogap_phi_sigma,
            geogap_frac_perturbed,
        )

    lambda_pack_contrastive = float(getattr(config, "lambda_pack_contrastive", 0.0) or 0.0)
    pack_contrastive_margin = float(getattr(config, "pack_contrastive_margin", 0.5) or 0.5)
    pack_contrastive_mode = str(getattr(config, "pack_contrastive_mode", "continuous") or "continuous")
    pack_contrastive_T_base = float(getattr(config, "pack_contrastive_T_base", 1.0) or 1.0)
    use_pack_contrastive = lambda_pack_contrastive > 0.0

    if use_pack_contrastive:
        if pack_contrastive_mode == "continuous":
            logger.debug(
                "Packing contrastive loss: lambda=%.3f, mode=continuous, T_base=%.2f",
                lambda_pack_contrastive,
                pack_contrastive_T_base,
            )
        else:
            logger.debug(
                "Packing contrastive loss: lambda=%.3f, margin=%.2f",
                lambda_pack_contrastive,
                pack_contrastive_margin,
            )

    lambda_balance = float(getattr(config, "lambda_balance", 0.0) or 0.0)
    balance_r = float(getattr(config, "balance_r", 8.0) or 8.0)
    balance_r_term = float(getattr(config, "balance_r_term", 4.0) or 4.0)
    use_balance = lambda_balance > 0.0

    if use_balance:
        logger.debug(
            "Energy balance loss: lambda=%.3f, subterm r=%.1f [1/%.0f,%.0f], term r=%.1f [1/%.0f,%.0f]",
            lambda_balance,
            balance_r,
            balance_r,
            balance_r,
            balance_r_term,
            balance_r_term,
            balance_r_term,
        )

    # Per-subterm discrimination maintenance — prevents subterm collapse
    # L_discrim = (1/K) * Σ_k softplus(E_k(native) - E_k(perturbed))
    lambda_discrim = float(getattr(config, "lambda_discrim", 0.0) or 0.0)
    discrim_every = int(getattr(config, "discrim_every", 4) or 4)
    discrim_sigma_min = float(getattr(config, "discrim_sigma_min", 0.05) or 0.05)
    discrim_sigma_max = float(getattr(config, "discrim_sigma_max", 2.0) or 2.0)
    discrim_mode = str(getattr(config, "discrim_mode", "mean") or "mean")
    use_discrim = lambda_discrim > 0.0

    if use_discrim:
        logger.info(
            "Per-subterm discrimination loss: lambda=%.3f, every %d steps, sigma=[%.3f, %.3f], mode=%s",
            lambda_discrim,
            discrim_every,
            discrim_sigma_min,
            discrim_sigma_max,
            discrim_mode,
        )

    # Secondary structure basin loss — enforces E_ss(helix) < E_ss(extended)
    # on real batch geometry classified by basin-derived Ramachandran windows.
    # Replaces geogap for run20+: geogap caused θθ dominance; basin loss
    # directly targets the secondary term inversion without crowding local.
    lambda_basin = float(getattr(config, "lambda_basin", 0.0) or 0.0)
    basin_margin = float(getattr(config, "basin_margin", 0.5) or 0.5)
    basin_mode = str(getattr(config, "basin_mode", "continuous") or "continuous")
    basin_T_base = float(getattr(config, "basin_T_base", 2.0) or 2.0)
    use_basin = lambda_basin > 0.0

    if use_basin:
        if basin_mode == "continuous":
            logger.debug(
                "Secondary basin loss: lambda=%.3f, mode=continuous, T_base=%.2f",
                lambda_basin,
                basin_T_base,
            )
        else:
            logger.debug(
                "Secondary basin loss: lambda=%.3f, margin=%.2f",
                lambda_basin,
                basin_margin,
            )

    # Native gap loss — deepens full-model energy well around native structures.
    lambda_native = float(getattr(config, "lambda_native", 0.0) or 0.0)
    native_margin = float(getattr(config, "native_margin", 0.5) or 0.5)
    native_sigma_min = float(getattr(config, "native_sigma_min", 0.05) or 0.05)
    native_sigma_max = float(getattr(config, "native_sigma_max", 0.50) or 0.50)
    native_mode = str(getattr(config, "native_mode", "continuous") or "continuous")
    native_T_base = float(getattr(config, "native_T_base", 5.0) or 5.0)
    use_native_gap = lambda_native > 0.0

    if use_native_gap:
        if native_mode == "continuous":
            logger.debug(
                "Native gap loss: lambda=%.3f, mode=continuous, T_base=%.2f, sigma=[%.2f, %.2f] rad",
                lambda_native,
                native_T_base,
                native_sigma_min,
                native_sigma_max,
            )
        else:
            logger.debug(
                "Native gap loss: lambda=%.3f, margin=%.2f, sigma=[%.2f, %.2f] rad",
                lambda_native,
                native_margin,
                native_sigma_min,
                native_sigma_max,
            )

    # ELT losses — Energy Landscape Theory (Q-funnel + Z-score + Frustration)
    lambda_funnel = float(getattr(config, "lambda_funnel", 0.0) or 0.0)
    funnel_T = float(getattr(config, "funnel_T", 2.0) or 2.0)
    funnel_n_decoys = int(getattr(config, "funnel_n_decoys", 10) or 10)
    funnel_slope_clamp = float(getattr(config, "funnel_slope_clamp", 10.0) or 10.0)
    funnel_sigma_min = float(getattr(config, "funnel_sigma_min", 0.05) or 0.05)
    funnel_sigma_max = float(getattr(config, "funnel_sigma_max", 2.0) or 2.0)
    funnel_contact_cutoff = float(getattr(config, "funnel_contact_cutoff", 9.5) or 9.5)
    lambda_zscore = float(getattr(config, "lambda_zscore", 0.0) or 0.0)
    target_zscore = float(getattr(config, "target_zscore", 3.0) or 3.0)
    lambda_gap_elt = float(getattr(config, "lambda_gap", 0.0) or 0.0)
    gap_margin = float(getattr(config, "gap_margin", 0.5) or 0.5)
    lambda_frustration = float(getattr(config, "lambda_frustration", 0.0) or 0.0)
    frustration_T = float(getattr(config, "frustration_T", 2.0) or 2.0)
    frustration_n_perms = int(getattr(config, "frustration_n_perms", 4) or 4)
    elt_every = int(getattr(config, "elt_every", 5) or 5)

    use_elt = lambda_funnel > 0 or lambda_zscore > 0 or lambda_gap_elt > 0 or lambda_frustration > 0
    if use_elt:
        logger.debug(
            "ELT losses: funnel=%.3f(T=%.1f) zscore=%.3f(target=%.1f) gap=%.3f(margin=%.2f) frust=%.3f(T=%.1f) "
            "every=%d decoys=%d sigma=[%.2f,%.2f]",
            lambda_funnel,
            funnel_T,
            lambda_zscore,
            target_zscore,
            lambda_gap_elt,
            gap_margin,
            lambda_frustration,
            frustration_T,
            elt_every,
            funnel_n_decoys,
            funnel_sigma_min,
            funnel_sigma_max,
        )

    # Native depth loss — push E(native) below a target to deepen basin
    _raw_depth = getattr(config, "lambda_native_depth", None)
    lambda_native_depth = float(_raw_depth) if _raw_depth is not None else 1.0
    _raw_target = getattr(config, "target_native_depth", None)
    target_native_depth = float(_raw_target) if _raw_target is not None else -1.0
    use_native_depth = lambda_native_depth > 0.0
    if use_native_depth:
        logger.info(
            "Native depth loss: lambda=%.3f, target=%.2f E/res, " "loss=exp(clamp(E_native - target, max=5))",
            lambda_native_depth,
            target_native_depth,
        )

    # Rg contrastive loss — penalise E(native) > E(scaled) for α~U(α_min, α_max)
    # Prevents compaction/swelling artifacts in long Langevin dynamics.
    # Unlike the old Rg gate (which modified the energy function during training
    # but not at inference), this is a pure training loss — E_θ is identical at
    # training and inference time.
    lambda_rg = float(getattr(config, "lambda_rg", 0.0) or 0.0)
    rg_alpha_min = float(getattr(config, "rg_alpha_min", 0.75))
    rg_alpha_max = float(getattr(config, "rg_alpha_max", 1.25))
    use_rg_loss = lambda_rg > 0.0
    if use_rg_loss:
        logger.info(
            "Rg contrastive loss: lambda=%.3f, α~U(%.2f,%.2f), " "loss=softplus(E_native - E_scaled)",
            lambda_rg,
            rg_alpha_min,
            rg_alpha_max,
        )

    # Lambda floor for hb_beta (prevents calibration collapse)
    lambda_hb_beta_floor = float(getattr(config, "lambda_hb_beta_floor", 0.0) or 0.0)
    if lambda_hb_beta_floor > 0:
        logger.info(
            "Lambda hb_beta floor: %.3f (softplus raw floor=%.3f)",
            lambda_hb_beta_floor,
            math.log(math.exp(lambda_hb_beta_floor) - 1),
        )

    # Subterm disable: clamp specified lambdas to zero after each step
    disable_subterms = list(getattr(config, "disable_subterms", []) or [])
    _disabled_params = []  # list of (name, param, zero_val) for post-step clamping
    if disable_subterms:
        import torch.nn.functional as F

        _SUBTERM_MAP = {
            # name → (module_path, attr, is_softplus)
            # 4-mer architecture
            "theta_phi": ("local", "_lambda_raw", True),
            # Old 3-subterm architecture
            "theta_theta": ("local", "_theta_theta_mlp_w", True),
            "delta_phi": ("local", "_delta_phi_weight_raw", True),
            "phi_phi": ("local", "_phi_phi_mlp_w", True),
            "ram": ("secondary", "ram_weight", False),
            "hb_alpha": ("secondary.hb_helix", "lambda_hb", False),
            "hb_beta": ("secondary.hb_sheet", "lambda_hb", False),
            "repulsion": ("repulsion", "_lambda_rep_raw", True),
            "geom": ("packing", "_lambda_pack_raw", True),
            "contact": ("packing.burial", "_lambda_hp_raw", True),
        }
        # MLP subterms: also zero+freeze the MLP weights (not just lambda)
        _MLP_SUBTERMS = {
            "theta_phi": ("local", "f_theta_phi"),
            "theta_theta": ("local", "f_theta_theta"),
            "phi_phi": ("local", "f_phi_phi"),
        }
        for name in disable_subterms:
            if name not in _SUBTERM_MAP:
                logger.warning("Unknown subterm '%s' — valid: %s", name, list(_SUBTERM_MAP.keys()))
                continue
            mod_path, attr, is_softplus = _SUBTERM_MAP[name]
            # Walk module path (e.g. "secondary.hb_helix" → model.secondary.hb_helix)
            mod = trainer.model
            for part in mod_path.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    break
            if mod is None:
                logger.warning("Disable subterm '%s': module '%s' not found", name, mod_path)
                continue
            param = getattr(mod, attr, None)
            if param is None or not isinstance(param, torch.Tensor):
                logger.warning("Disable subterm '%s': param '%s' not found or not a tensor", name, attr)
                continue
            # Set initial value to zero and freeze
            zero_val = -100.0 if is_softplus else 0.0

            # Check if param is a leaf (nn.Parameter) or computed (property)
            if param.is_leaf:
                with torch.no_grad():
                    param.fill_(zero_val)
                param.requires_grad_(False)
                _disabled_params.append((name, param, zero_val))
                effective = float(F.softplus(param).item()) if is_softplus else float(param.item())
                logger.info(
                    "Disabled subterm '%s': %s.%s = %.4f (λ=%.6f, frozen)", name, mod_path, attr, zero_val, effective
                )
            else:
                # Non-leaf: search module's parameters for the underlying raw param
                found_raw = False
                # Try common naming conventions
                candidates = [
                    f"_{attr}_raw",  # _theta_theta_weight_raw
                    attr.replace("weight", "raw"),  # theta_theta_raw
                    f"_{attr.replace('weight', 'raw')}",  # _theta_theta_raw
                ]
                for cand in candidates:
                    raw_param = getattr(mod, cand, None)
                    if raw_param is not None and isinstance(raw_param, torch.Tensor) and raw_param.is_leaf:
                        with torch.no_grad():
                            raw_param.fill_(-100.0)
                        raw_param.requires_grad_(False)
                        _disabled_params.append((name, raw_param, -100.0))
                        effective = float(F.softplus(raw_param).item())
                        logger.info(
                            "Disabled subterm '%s': %s.%s (via raw param %s) = -100.0 (λ=%.6f, frozen)",
                            name,
                            mod_path,
                            attr,
                            cand,
                            effective,
                        )
                        found_raw = True
                        break
                if not found_raw:
                    # Brute force: find ALL parameters matching this subterm and zero+freeze them
                    # This handles cases like θθ MLP where the "lambda" is computed from MLP weights
                    keyword = name.replace("_", "")  # "theta_theta" → "thetatheta"
                    matched = []
                    for pname, pparam in mod.named_parameters():
                        if keyword in pname.replace("_", "") and pparam.is_leaf:
                            with torch.no_grad():
                                pparam.zero_()  # Zero, not -100! Makes MLP output 0.
                            pparam.requires_grad_(False)
                            matched.append(pname)
                    if matched:
                        # Store the first param for post-step safety clamp
                        first_param = dict(mod.named_parameters())[matched[0]]
                        _disabled_params.append((name, first_param, 0.0))
                        logger.info(
                            "Disabled subterm '%s': zeroed+froze %d params in %s: %s",
                            name,
                            len(matched),
                            mod_path,
                            matched,
                        )
                        found_raw = True
                if not found_raw:
                    logger.error(
                        "Disable subterm '%s': FAILED — no leaf parameter found in %s. " "Available params: %s",
                        name,
                        mod_path,
                        [n for n, _ in mod.named_parameters()],
                    )
            # Also zero+freeze MLP weights for MLP-based subterms
            if name in _MLP_SUBTERMS:
                mlp_mod_path, mlp_attr = _MLP_SUBTERMS[name]
                mlp_parent = trainer.model
                for part in mlp_mod_path.split("."):
                    mlp_parent = getattr(mlp_parent, part, None)
                    if mlp_parent is None:
                        break
                if mlp_parent is not None:
                    mlp = getattr(mlp_parent, mlp_attr, None)
                    if mlp is not None:
                        zeroed = []
                        for pname, pparam in mlp.named_parameters():
                            if pparam.is_leaf:
                                with torch.no_grad():
                                    pparam.zero_()
                                pparam.requires_grad_(False)
                                zeroed.append(pname)
                        if zeroed:
                            logger.info(
                                "Disabled subterm '%s': zeroed+froze %d MLP params in %s.%s: %s",
                                name,
                                len(zeroed),
                                mlp_mod_path,
                                mlp_attr,
                                zeroed,
                            )
        if _disabled_params:
            logger.info("Disabled %d subterms: %s", len(_disabled_params), [n for n, _, _ in _disabled_params])

    freeze_gates_steps = int(getattr(config, "freeze_gates_steps", 0) or 0)
    ramp_gates = bool(getattr(config, "ramp_gates", False))
    ramp_start_raw = getattr(config, "ramp_start", None)
    ramp_end_raw = getattr(config, "ramp_end", None)
    ramp_steps = int(getattr(config, "ramp_steps", 0) or 0)
    use_quadratic = bool(getattr(config, "ramp_rep_quadratic", False))

    spike_threshold_start = float(getattr(config, "spike_threshold_start", 500.0))
    spike_threshold_end = float(getattr(config, "spike_threshold_end", 1000.0))
    gate_lr_mult = float(getattr(config, "gate_lr_mult", 1.0))

    ramp_active = bool(ramp_gates and ramp_start_raw is not None and ramp_end_raw is not None and ramp_steps > 0)
    ramp_start = dict(ramp_start_raw) if ramp_active else {}
    ramp_end = dict(ramp_end_raw) if ramp_active else {}

    if ramp_active:
        logger.debug("GATE RAMPING ACTIVE: %d steps  %s -> %s", ramp_steps, ramp_start, ramp_end)
        if hasattr(trainer.model, "set_gates"):
            trainer.model.set_gates(**ramp_start)

    optimizer, all_trainable_params, gate_params = _setup_optimizer_with_frozen_gates(
        trainer, config, ramp_active, ramp_steps, gate_lr_mult
    )
    trainer.optimizer = optimizer

    trainer.scheduler = None
    if optimizer and lr_schedule and lr > 0 and n_steps > 0:
        if lr_schedule == "cosine":
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr_final)
        elif lr_schedule == "linear":

            def lambda_fn(step: int) -> float:
                t = min(max(step, 0), n_steps) / max(1, n_steps)
                return 1.0 - t * (1.0 - lr_final / max(lr, 1e-12))

            trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
        elif lr_schedule == "exponential":
            gamma = math.exp(math.log(max(lr_final, 1e-12) / max(lr, 1e-12)) / max(1, n_steps))
            trainer.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    start_step = 0
    if resume:
        if resume == "auto":
            resume = trainer.find_latest_checkpoint(config.name)
        if resume:
            state = trainer.load_checkpoint(resume, load_optimizer=True)
            start_step = int(state.phase_step)
            logger.debug("Resumed from phase step %d (global=%d)", start_step, int(state.global_step))
            # Defensively log restored gate values so we can verify they came back correctly
            if hasattr(trainer.model, "get_gates"):
                restored_gates = trainer.model.get_gates()
                logger.debug("Gates after checkpoint restore: %s", restored_gates)

    gates_unfrozen = False
    if (
        ramp_active
        and start_step >= ramp_steps
        and trainer.optimizer is not None
        and len(trainer.optimizer.param_groups) > 1
    ):
        gates_unfrozen = True
        _enforce_gate_lr_multiplier(trainer.optimizer)

    gates_frozen_by_cli = False
    if freeze_gates_steps > 0 and gate_params:
        _set_gates_trainable(gate_params, trainable=False)
        gates_frozen_by_cli = True
        if trainer.optimizer is not None and len(trainer.optimizer.param_groups) > 1:
            trainer.optimizer.param_groups[1]["lr"] = 0.0
        logger.debug("Gates requires_grad=False for first %d steps (freeze_gates_steps)", freeze_gates_steps)

    trainer.model.train()
    trainer._init_validators()

    data_iter = iter(train_loader)
    progress = ProgressBar(n_steps, prefix=f"Phase {config.name}")

    last_fb_diag: dict = {}
    spike_counter = 0
    consecutive_spikes = 0
    ramp_paused = False
    pause_counter = 0
    PAUSE_DURATION = 200
    ramp_complete_logged = False

    bad_grad_streak = 0
    bad_grad_streak_max = int(getattr(config, "bad_grad_streak_max", 200) or 200)

    # Construct basin loss once — reads basin peak positions from model.secondary.
    # Cheap: just reads theta_centers/phi_centers from already-loaded basin grids.
    _basin_loss_fn = None
    if use_basin:
        try:
            _basin_loss_fn = SecondaryBasinLoss(
                trainer.model.secondary,
                margin=basin_margin,
                mode=basin_mode,
                T_base=basin_T_base,
            )
        except Exception as e:
            logger.warning("SecondaryBasinLoss construction failed: %s — basin loss disabled", e)
            use_basin = False

    for phase_step in range(start_step + 1, n_steps + 1):
        trainer.global_step += 1
        trainer.phase_step = phase_step

        loss: Optional[torch.Tensor] = None
        forces: Optional[torch.Tensor] = None

        if gates_frozen_by_cli and phase_step == freeze_gates_steps + 1:
            _set_gates_trainable(gate_params, trainable=True)
            gates_frozen_by_cli = False
            if trainer.optimizer is not None and len(trainer.optimizer.param_groups) > 1:
                if ramp_active and phase_step <= ramp_steps:
                    trainer.optimizer.param_groups[1]["lr"] = 0.0
                else:
                    gates_unfrozen = True
                    _enforce_gate_lr_multiplier(trainer.optimizer)
            logger.info("Gates unfrozen after freeze_gates_steps at phase_step=%d", phase_step)

        if not ramp_active:
            trainer._apply_gate_schedule(config)
        else:
            trainer._apply_non_gate_schedule(config)

        apply_ramp_this_step = True
        if ramp_active and ramp_paused and phase_step <= ramp_steps:
            pause_counter += 1
            if pause_counter >= PAUSE_DURATION:
                logger.info("Resuming ramp after %d step pause", PAUSE_DURATION)
                ramp_paused = False
                pause_counter = 0
            else:
                apply_ramp_this_step = False

        if ramp_active and apply_ramp_this_step and phase_step <= ramp_steps:
            scheduled_gates = _compute_ramp_gates(phase_step, ramp_steps, ramp_start, ramp_end, use_quadratic)
            if scheduled_gates and hasattr(trainer.model, "set_gates"):
                trainer.model.set_gates(**scheduled_gates)

        if ramp_active and (phase_step == ramp_steps + 1) and not gates_unfrozen and not gates_frozen_by_cli:
            if trainer.optimizer is not None and len(trainer.optimizer.param_groups) > 1:
                gates_unfrozen = True
                _enforce_gate_lr_multiplier(trainer.optimizer)
                logger.info("GATES UNFROZEN after ramp")

        if ramp_active and (phase_step == ramp_steps) and not ramp_complete_logged:
            ramp_complete_logged = True
            g = trainer.model.get_gates() if hasattr(trainer.model, "get_gates") else {}
            logger.info("RAMP COMPLETE - Final gates: %s", g)

        try:
            R, seq, _, _, lengths = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            R, seq, _, _, lengths = next(data_iter)

        R = R.to(trainer.device)
        seq = seq.to(trainer.device)
        lengths = lengths.to(trainer.device)
        current_lr = trainer.scheduler.get_last_lr()[0] if trainer.scheduler else lr

        skip_step = False
        skip_reason = ""
        _loss_dsm_val = None
        _dsm_diag = {}
        _loss_fb_val = None
        _loss_gap_val = None
        _loss_pack_c_val = None
        _loss_balance_val = None
        _balance_absmeans = None  # per-subterm mean(|gate×E|) from energy_balance_loss
        _balance_term_absmeans = None  # per-term mean(|gate×E|) aggregated
        _loss_native_val = None
        _native_diag = None
        _loss_basin_val = None
        _loss_funnel_val = None
        _loss_zscore_val = None
        _loss_elt_gap_val = None
        _loss_frust_val = None
        _loss_native_depth_val = None
        _loss_discrim_val = None
        _discrim_diag = None
        _loss_rg_val = None
        _e_native_val = None
        _elt_diag = None
        _basin_diag = None
        _pack_c_diag = None
        _geogap_diag = None

        try:
            loss = dsm_ic_loss(
                trainer.model,
                R,
                seq,
                sigma=sigma_rad,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sigma_min_theta=sigma_min_theta,
                sigma_max_theta=sigma_max_theta,
                sigma_min_phi=sigma_min_phi,
                sigma_max_phi=sigma_max_phi,
                lengths=lengths,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                diag=_dsm_diag,
            )

            if loss is None:
                skip_step, skip_reason = True, "loss_none"
            elif (loss.ndim == 0) and (float(loss.detach().item()) == 0.0):
                skip_step, skip_reason = True, "dsm_skip_sentinel"
            elif not torch.isfinite(loss).all():
                skip_step, skip_reason = True, "nonfinite_loss"
            else:
                _loss_dsm_val = float(loss.detach().item())

            # IC force balance
            if use_force_balance and not skip_step and loss is not None:
                try:
                    loss_fb = force_balance_ic_loss(
                        trainer.model,
                        R,
                        seq,
                        sigma_theta=fb_sigma_theta,
                        sigma_phi=fb_sigma_phi,
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        clash_phi_frac=fb_clash_phi_frac,
                        clash_phi_sigma=fb_clash_phi_sigma,
                        target_ss_ratio=fb_target_ss_ratio,
                        target_pack_ratio=fb_target_pack_ratio,
                        target_rep_ratio=fb_target_rep_ratio,
                        lengths=lengths,
                    )
                    if torch.isfinite(loss_fb):
                        _loss_fb_val = float(loss_fb.detach().item())
                        loss = loss + lambda_fb * loss_fb
                    else:
                        logger.warning("Non-finite IC force balance loss at step %d", phase_step)
                except Exception as e:
                    logger.warning("IC Force balance loss error at step %d: %s", phase_step, e)

            # IC force balance diagnostics
            if use_force_balance and not skip_step and phase_step % fb_diag_every == 0:
                try:
                    diag = force_balance_ic_diagnostics(
                        trainer.model,
                        R.detach(),
                        seq.detach(),
                        sigma_theta=fb_sigma_theta,
                        sigma_phi=fb_sigma_phi,
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        clash_phi_frac=fb_clash_phi_frac,
                        clash_phi_sigma=fb_clash_phi_sigma,
                        target_ss_ratio=fb_target_ss_ratio,
                        target_pack_ratio=fb_target_pack_ratio,
                        target_rep_ratio=fb_target_rep_ratio,
                        lengths=lengths,
                    )
                    last_fb_diag = diag
                    sc = diag["scales"]
                    ra = diag["ratios"]
                    tg = diag.get("targets", {})
                    mt = diag.get("met", {})

                    def _fmt(term):
                        r = ra.get(term, 0.0)
                        t = tg.get(term, 2.0)
                        ok = "OK" if mt.get(term, False) else "FAIL"
                        return f"F_{term[:3]}={sc.get(term,0):.4f}(r={r:.3f}/tgt={t:.1f} {ok})"

                    logger.info(
                        "[force_balance_ic] step %d | F_local=%.4f | %s | %s | %s",
                        phase_step,
                        sc.get("local", 0),
                        _fmt("repulsion"),
                        _fmt("secondary"),
                        _fmt("packing"),
                    )
                except Exception as e:
                    logger.warning("IC Force balance diagnostics error: %s", e)

            # IC geogap
            if use_geogap and not skip_step and loss is not None:
                try:
                    loss_gap = local_geogap_ic_loss(
                        trainer.model,
                        R,
                        seq,
                        margin=geogap_margin,
                        theta_perturb_sigma=geogap_theta_sigma,
                        phi_perturb_sigma=geogap_phi_sigma,
                        frac_perturbed=geogap_frac_perturbed,
                        lengths=lengths,
                    )
                    if torch.isfinite(loss_gap):
                        _loss_gap_val = float(loss_gap.detach().item())
                        loss = loss + lambda_geogap * loss_gap
                    else:
                        logger.warning("Non-finite IC geogap loss at step %d", phase_step)
                except Exception as e:
                    logger.warning("IC Geogap loss error at step %d: %s", phase_step, e)

            # IC geogap diagnostics
            if use_geogap and not skip_step and phase_step % geogap_diag_every == 0:
                try:
                    _geogap_diag = local_geogap_ic_diagnostics(
                        trainer.model,
                        R.detach(),
                        seq.detach(),
                        margin=geogap_margin,
                        theta_perturb_sigma=geogap_theta_sigma,
                        phi_perturb_sigma=geogap_phi_sigma,
                        frac_perturbed=geogap_frac_perturbed,
                        lengths=lengths,
                    )
                except Exception as e:
                    logger.warning("IC Geogap diagnostics error: %s", e)

            # Packing contrastive loss — keeps MLP signal alive during full phase
            if use_pack_contrastive and not skip_step and loss is not None:
                try:
                    _want_diag = phase_step % 200 == 0
                    _pack_c_result = packing_contrastive_loss(
                        trainer.model,
                        R,
                        seq,
                        margin=pack_contrastive_margin,
                        sigma_min=sigma_min if sigma_min is not None else 0.02,
                        sigma_max=sigma_max if sigma_max is not None else 0.30,
                        mode=pack_contrastive_mode,
                        T_base=pack_contrastive_T_base,
                        return_diag=_want_diag,
                        lengths=lengths,
                    )
                    if _want_diag:
                        loss_pack_c, _pack_c_diag = _pack_c_result
                    else:
                        loss_pack_c = _pack_c_result
                    if torch.isfinite(loss_pack_c):
                        _loss_pack_c_val = float(loss_pack_c.detach().item())
                        loss = loss + lambda_pack_contrastive * loss_pack_c
                    else:
                        logger.warning("Non-finite packing contrastive loss at step %d", phase_step)
                except Exception as e:
                    logger.warning("Packing contrastive loss error at step %d: %s", phase_step, e)

            # Secondary basin loss — enforces E_ss(helix) < E_ss(extended)
            # using real batch geometry classified by basin-derived Ramachandran windows.
            if use_basin and not skip_step and loss is not None and _basin_loss_fn is not None:
                try:
                    loss_basin, _basin_diag = _basin_loss_fn(trainer.model, R, seq, lengths=lengths)
                    if torch.isfinite(loss_basin):
                        _loss_basin_val = float(loss_basin.detach().item())
                        loss = loss + lambda_basin * loss_basin
                    else:
                        logger.warning("Non-finite secondary basin loss at step %d", phase_step)
                except Exception as e:
                    logger.warning("Secondary basin loss error at step %d: %s", phase_step, e)

            # Native gap loss — penalises when full-model energy well around native
            # is too shallow to resist thermal fluctuations at beta=1.
            # Sigma drawn log-uniformly: covers both local and large-scale deformations.
            if use_native_gap and not skip_step and loss is not None:
                try:
                    loss_native, _native_diag = native_gap_loss(
                        trainer.model,
                        R,
                        seq,
                        margin=native_margin,
                        sigma_min=native_sigma_min,
                        sigma_max=native_sigma_max,
                        mode=native_mode,
                        T_base=native_T_base,
                        return_diag=True,
                        lengths=lengths,
                    )
                    if torch.isfinite(loss_native):
                        _loss_native_val = float(loss_native.detach().item())
                        loss = loss + lambda_native * loss_native
                    else:
                        logger.warning("Non-finite native gap loss at step %d", phase_step)
                except Exception as e:
                    logger.warning("Native gap loss error at step %d: %s", phase_step, e)

            # ELT losses — Q-funnel + Z-score + Frustration
            # Amortized: run every elt_every steps (default 5) to spread cost.
            if use_elt and not skip_step and loss is not None and phase_step % elt_every == 0:
                _want_elt_diag = phase_step % 200 == 0
                try:
                    # Q-funnel + Z-score (shared decoys)
                    if lambda_funnel > 0 or lambda_zscore > 0 or lambda_gap_elt > 0:
                        loss_funnel, loss_zscore, loss_elt_gap, elt_d = elt_funnel_loss(
                            trainer.model,
                            R,
                            seq,
                            lengths,
                            n_decoys=funnel_n_decoys,
                            T_funnel=funnel_T,
                            target_zscore=target_zscore,
                            gap_margin=gap_margin,
                            slope_clamp=funnel_slope_clamp,
                            sigma_min=funnel_sigma_min,
                            sigma_max=funnel_sigma_max,
                            contact_cutoff=funnel_contact_cutoff,
                            return_diag=_want_elt_diag,
                        )
                        if torch.isfinite(loss_funnel) and lambda_funnel > 0:
                            _loss_funnel_val = float(loss_funnel.detach().item())
                            loss = loss + lambda_funnel * loss_funnel
                        if torch.isfinite(loss_zscore) and lambda_zscore > 0:
                            _loss_zscore_val = float(loss_zscore.detach().item())
                            loss = loss + lambda_zscore * loss_zscore
                        if torch.isfinite(loss_elt_gap) and lambda_gap_elt > 0:
                            _loss_elt_gap_val = float(loss_elt_gap.detach().item())
                            loss = loss + lambda_gap_elt * loss_elt_gap
                        if _want_elt_diag and elt_d:
                            _elt_diag = elt_d

                    # Frustration
                    if lambda_frustration > 0:
                        loss_frust, frust_d = elt_frustration_loss(
                            trainer.model,
                            R,
                            seq,
                            lengths,
                            n_perms=frustration_n_perms,
                            T_frust=frustration_T,
                            return_diag=_want_elt_diag,
                        )
                        if torch.isfinite(loss_frust):
                            _loss_frust_val = float(loss_frust.detach().item())
                            loss = loss + lambda_frustration * loss_frust
                        if _want_elt_diag and frust_d:
                            _elt_diag = {**(_elt_diag or {}), **frust_d}

                except Exception as e:
                    logger.warning("ELT loss error at step %d: %s", phase_step, e)

            # Native depth loss — exponential push toward target E/res
            # loss = exp(clamp(E_native - target, max=5))
            # Deepens the basin by making native energy more negative.
            if use_native_depth and not skip_step and loss is not None:
                try:
                    E_nat = trainer.model(R, seq, lengths=lengths).mean()
                    _e_native_val = float(E_nat.detach().item())
                    exponent = (E_nat - target_native_depth).clamp(max=5.0)
                    loss_depth = torch.exp(exponent)
                    if torch.isfinite(loss_depth):
                        _loss_native_depth_val = float(loss_depth.detach().item())
                        loss = loss + lambda_native_depth * loss_depth
                except Exception as e:
                    logger.warning("Native depth loss error at step %d: %s", phase_step, e)

            # Rg contrastive loss — penalise E(native) > E(Rg-scaled)
            # Scales coordinates by α~U(α_min, α_max) toward/away from COM.
            # softplus(E_native - E_scaled) → 0 when native is lower.
            # Prevents compaction/swelling artifacts without modifying E_θ.
            if use_rg_loss and not skip_step and loss is not None:
                try:
                    loss_rg = rg_contrastive_loss(
                        trainer.model,
                        R,
                        seq,
                        lengths=lengths,
                        alpha_min=rg_alpha_min,
                        alpha_max=rg_alpha_max,
                    )
                    if torch.isfinite(loss_rg):
                        _loss_rg_val = float(loss_rg.detach().item())
                        loss = loss + lambda_rg * loss_rg
                except Exception as e:
                    logger.warning("Rg contrastive loss error at step %d: %s", phase_step, e)

            # Energy balance loss — penalises pairwise term ratios outside [1/r, r]
            # Same R/seq batch as DSM; uses gate×E contributions.
            if use_balance and not skip_step and loss is not None:
                try:
                    loss_bal, _balance_absmeans, _balance_term_absmeans = energy_balance_loss(
                        trainer.model,
                        R,
                        seq,
                        r=balance_r,
                        r_term=balance_r_term,
                        lengths=lengths,
                        exclude_subterms=set(disable_subterms),
                    )
                    if torch.isfinite(loss_bal):
                        _loss_balance_val = float(loss_bal.detach().item())
                        loss = loss + lambda_balance * loss_bal
                    else:
                        logger.warning("Non-finite energy balance loss at step %d", phase_step)
                except Exception as e:
                    logger.warning("Energy balance loss error at step %d: %s", phase_step, e)

            # Per-subterm discrimination maintenance — prevents collapse
            # mode='mean': L = (1/K) Σ_k softplus(-gap_k)
            # mode='max':  L = max_k softplus(-gap_k)  [focuses on worst subterm]
            if use_discrim and not skip_step and loss is not None and phase_step % discrim_every == 0:
                try:
                    loss_disc, _discrim_diag = subterm_discrimination_loss(
                        trainer.model,
                        R,
                        seq,
                        lengths=lengths,
                        sigma_min=discrim_sigma_min,
                        sigma_max=discrim_sigma_max,
                        exclude_subterms=set(disable_subterms),
                        mode=discrim_mode,
                    )
                    if torch.isfinite(loss_disc):
                        _loss_discrim_val = float(loss_disc.detach().item())
                        loss = loss + lambda_discrim * loss_disc
                    else:
                        logger.warning("Non-finite discrim loss at step %d", phase_step)
                except Exception as e:
                    logger.warning("Discrim loss error at step %d: %s", phase_step, e)

            trainer.current_loss = float(loss.detach().item()) if loss is not None else 0.0

            # backward / step
            if trainer.optimizer and (loss is not None) and loss.requires_grad and not skip_step:
                trainer.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)

                nan_grads = False
                bad_name = bad_shape = None
                for name, p in trainer.model.named_parameters():
                    if p.grad is None:
                        continue
                    if gates_frozen_by_cli and _is_gate_param(name, p):
                        continue
                    if not torch.isfinite(p.grad).all():
                        nan_grads = True
                        bad_name = name
                        bad_shape = tuple(p.shape)
                        g = p.grad
                        n_nan = int(torch.isnan(g).sum().item())
                        n_inf = int(torch.isinf(g).sum().item())
                        finite_vals = g[torch.isfinite(g)]
                        g_max = float(finite_vals.abs().max().item()) if finite_vals.numel() > 0 else float("nan")
                        logger.error(
                            "NaN/Inf grad: %s %s nan=%d inf=%d finite_max=%.3e",
                            name,
                            tuple(p.shape),
                            n_nan,
                            n_inf,
                            g_max,
                        )

                if nan_grads:
                    bad_grad_streak += 1
                    logger.error(
                        "First bad param: %s %s (streak=%d/%d)",
                        bad_name,
                        bad_shape,
                        bad_grad_streak,
                        bad_grad_streak_max,
                    )
                    torch.save(
                        {
                            "name": bad_name,
                            "shape": bad_shape,
                            "phase_step": phase_step,
                            "loss": float(trainer.current_loss),
                            "R": R.detach().cpu(),
                            "seq": seq.detach().cpu(),
                        },
                        f"debug_nan_grad_step_{phase_step}.pt",
                    )
                    trainer.optimizer.zero_grad(set_to_none=True)
                    if bad_grad_streak >= bad_grad_streak_max:
                        logger.error("Aborting phase: bad grad streak reached threshold.")
                        break
                    progress.update(1)
                    continue

                bad_grad_streak = 0
                trainer.optimizer.step()

                # Lambda floor: clamp hb_beta raw param so softplus >= floor
                if lambda_hb_beta_floor > 0:
                    ss = getattr(trainer.model, "secondary", None)
                    if ss is not None:
                        raw_param = getattr(ss, "_lambda_hb_beta_raw", None)
                        if raw_param is not None:
                            raw_floor = math.log(math.exp(lambda_hb_beta_floor) - 1)
                            with torch.no_grad():
                                raw_param.clamp_(min=raw_floor)

                # Safety: re-zero disabled subterms (belt + suspenders with freeze)
                for _dname, _dparam, _dzero in _disabled_params:
                    with torch.no_grad():
                        _dparam.fill_(_dzero)

                if trainer.scheduler:
                    trainer.scheduler.step()
                    if gates_unfrozen and trainer.optimizer is not None and len(trainer.optimizer.param_groups) > 1:
                        _enforce_gate_lr_multiplier(trainer.optimizer)

            # forces for diagnostics
            forces = None
            if not skip_step:
                R_forces = R.detach().clone().requires_grad_(True)
                E_clean = trainer.model(R_forces, seq, lengths=lengths).sum()
                forces = -torch.autograd.grad(E_clean, R_forces, create_graph=False, retain_graph=False)[0]
                force_norms = torch.norm(forces, dim=-1)
                max_force = float(force_norms.max().item())

                spike_threshold = spike_threshold_end
                if ramp_active and ramp_steps > 0:
                    ramp_progress = _clamp01(phase_step / float(max(ramp_steps, 1)))
                    spike_threshold = (
                        spike_threshold_start + (spike_threshold_end - spike_threshold_start) * ramp_progress
                    )

                if max_force > spike_threshold:
                    spike_counter += 1
                    consecutive_spikes += 1
                    logger.warning("Force spike at step %d: max|F|=%.2f > %.0f", phase_step, max_force, spike_threshold)
                    if ramp_active and phase_step <= ramp_steps and consecutive_spikes >= 3 and not ramp_paused:
                        logger.warning(
                            "%d consecutive spikes - pausing ramp for %d steps", consecutive_spikes, PAUSE_DURATION
                        )
                        ramp_paused = True
                        pause_counter = 0
                else:
                    consecutive_spikes = 0

            if skip_step and (phase_step % 50 == 0):
                logger.warning("Skipped optimizer step at step %d (%s)", phase_step, skip_reason)

        except Exception as e:
            logger.error("Error at step %d: %s", phase_step, str(e))
            import traceback

            traceback.print_exc()
            continue

        R_detached = R.detach()
        seq_detached = seq.detach()

        if phase_step % 200 == 0 and not skip_step:
            try:
                # Pre-compute values for diagnostics (avoids double computation)
                _precomputed = {}
                with torch.no_grad():
                    # Term energies — one call per subterm
                    _term_e = {}
                    for _tname in ("local", "repulsion", "secondary", "packing"):
                        _tmod = getattr(trainer.model, _tname, None)
                        if _tmod is not None:
                            _term_e[_tname] = float(_tmod(R_detached, seq_detached, lengths=lengths).mean().item())
                            if _tname == "local" and hasattr(_tmod, "forward_learned"):
                                _term_e["local_learned"] = float(
                                    _tmod.forward_learned(R_detached, seq_detached, lengths=lengths).mean().item()
                                )
                    _precomputed["term_energies"] = _term_e

                    # Safety metrics — padding-aware
                    B_diag, L_diag = R_detached.shape[:2]
                    _exclude = 3
                    _idx = torch.arange(L_diag, device=R_detached.device)
                    _sep = (_idx[:, None] - _idx[None, :]).abs()
                    _triu = torch.triu(
                        torch.ones(L_diag, L_diag, device=R_detached.device, dtype=torch.bool), diagonal=1
                    )
                    _nb_mask = (_sep > _exclude) & _triu
                    _D = torch.sqrt(((R_detached.unsqueeze(2) - R_detached.unsqueeze(1)) ** 2).sum(-1) + 1e-8)
                    if lengths is not None:
                        _valid = _idx.unsqueeze(0) < lengths.unsqueeze(1)
                        _pair_valid = _valid.unsqueeze(2) & _valid.unsqueeze(1)
                        _D = _D.masked_fill(~_pair_valid, float("inf"))
                    _D_flat = _D.reshape(B_diag, -1)
                    _nb_dists = _D_flat[:, _nb_mask.reshape(-1)]
                    if _nb_dists.numel() > 0:
                        _precomputed["safety"] = {
                            "exclude": _exclude,
                            "min_dist": float(_nb_dists.amin(dim=1).median().item()),
                            "frac_below_40": float((_nb_dists < 4.0).float().mean(dim=1).median().item()),
                            "frac_below_45": float((_nb_dists < 4.5).float().mean(dim=1).median().item()),
                        }
                    else:
                        _precomputed["safety"] = {
                            "exclude": _exclude,
                            "min_dist": float("inf"),
                            "frac_below_40": 0.0,
                            "frac_below_45": 0.0,
                        }

                    # Force metrics from the force spike block
                    if forces is not None:
                        _fnorms = torch.norm(forces, dim=-1)
                        _precomputed["max_force"] = float(_fnorms.max().item())
                        _precomputed["clip_frac"] = float((_fnorms > 50.0).float().mean().item())

                trainer.diagnostic_logger.log_step_block(
                    phase_step=phase_step,
                    n_steps=n_steps,
                    loss=trainer.current_loss,
                    lr=current_lr,
                    R=R_detached,
                    seq=seq_detached,
                    forces=forces,
                    force_cap=50.0,
                    geogap_diag=_geogap_diag,
                    geogap_margin=geogap_margin,
                    loss_pack_c=_loss_pack_c_val,
                    lambda_pack_contrastive=lambda_pack_contrastive,
                    pack_contrastive_margin=pack_contrastive_margin,
                    loss_balance=_loss_balance_val,
                    lambda_balance=lambda_balance,
                    term_absmeans=_balance_absmeans,
                    term_absmeans_agg=_balance_term_absmeans,
                    balance_r_term=balance_r_term,
                    loss_basin=_loss_basin_val,
                    lambda_basin=lambda_basin,
                    basin_margin=basin_margin,
                    loss_native=_loss_native_val,
                    lambda_native=lambda_native,
                    native_margin=native_margin,
                    native_diag=_native_diag,
                    lengths=lengths,
                    elt_diag=_elt_diag,
                    loss_funnel=_loss_funnel_val,
                    lambda_funnel=lambda_funnel,
                    loss_zscore=_loss_zscore_val,
                    lambda_zscore=lambda_zscore,
                    target_zscore=target_zscore,
                    loss_elt_gap=_loss_elt_gap_val,
                    lambda_gap_elt=lambda_gap_elt,
                    loss_frust=_loss_frust_val,
                    lambda_frustration=lambda_frustration,
                    loss_native_depth=_loss_native_depth_val,
                    lambda_native_depth=lambda_native_depth,
                    target_native_depth=target_native_depth,
                    e_native_depth=_e_native_val,
                    precomputed=_precomputed,
                    loss_dsm=_loss_dsm_val,
                    dsm_diag=_dsm_diag,
                )
            except Exception as e:
                logger.warning("Diagnostic block error at step %d: %s", phase_step, e)

            # Per-subterm discrimination display (separate from main diagnostic block)
            if use_discrim and _loss_discrim_val is not None and _discrim_diag:
                gaps_str = "  ".join(f"{k}={v:+.3f}" for k, v in sorted(_discrim_diag.items()))
                logger.info(
                    "  Discrim: loss=%.4f(x%.1f=%.4f)  gaps: %s",
                    _loss_discrim_val,
                    lambda_discrim,
                    lambda_discrim * _loss_discrim_val,
                    gaps_str,
                )

            if use_basin and _basin_loss_fn is not None:
                try:
                    secondary_basin_diagnostics(
                        trainer.model,
                        R_detached,
                        seq_detached,
                        _basin_loss_fn,
                        lengths=lengths,
                    )
                except Exception as e:
                    logger.warning("Basin diagnostics error at step %d: %s", phase_step, e)

        if phase_step % 100 == 0:
            if trainer._check_convergence():
                logger.info("Training converged at step %d", phase_step)
                trainer.save_checkpoint(config.name, phase_step, trainer.current_loss, is_best=True)
                break

        if phase_step % 50 == 0:
            parts = []
            if _loss_dsm_val is not None:
                if _dsm_diag.get("n_samples", 1) > 1:
                    _ds = _dsm_diag.get("dsm_std", 0)
                    _da = _dsm_diag.get("dsm_alpha")
                    _dm = _dsm_diag.get("dsm_mixed")
                    _da_s = f"{_da:.4f}" if _da is not None else "skip"
                    _dm_s = f"{_dm:.4f}" if _dm is not None else "skip"
                    parts.append(f"dsm={_loss_dsm_val:.4f}(std={_ds:.4f},α={_da_s},mix={_dm_s})")
                else:
                    parts.append(f"dsm={_loss_dsm_val:.4f}")
            if _loss_fb_val is not None:
                parts.append(f"fb={_loss_fb_val:.4f}(x{lambda_fb}={lambda_fb*_loss_fb_val:.4f})")
            if _loss_gap_val is not None:
                parts.append(f"gap={_loss_gap_val:.4f}(x{lambda_geogap}={lambda_geogap*_loss_gap_val:.4f})")
            if _loss_pack_c_val is not None:
                parts.append(
                    f"pack_c={_loss_pack_c_val:.4f}(x{lambda_pack_contrastive}={lambda_pack_contrastive*_loss_pack_c_val:.4f})"
                )
            if _loss_balance_val is not None:
                parts.append(
                    f"balance={_loss_balance_val:.4f}(x{lambda_balance}={lambda_balance*_loss_balance_val:.4f})"
                )
            if _loss_native_val is not None:
                parts.append(f"native={_loss_native_val:.4f}(x{lambda_native}={lambda_native*_loss_native_val:.4f})")
            if _loss_basin_val is not None:
                parts.append(f"basin={_loss_basin_val:.4f}(x{lambda_basin}={lambda_basin*_loss_basin_val:.4f})")
            if _loss_funnel_val is not None:
                parts.append(f"funnel={_loss_funnel_val:.4f}(x{lambda_funnel}={lambda_funnel*_loss_funnel_val:.4f})")
            if _loss_zscore_val is not None:
                parts.append(f"zscore={_loss_zscore_val:.4f}(x{lambda_zscore}={lambda_zscore*_loss_zscore_val:.4f})")
            if _loss_elt_gap_val is not None:
                parts.append(
                    f"elt_gap={_loss_elt_gap_val:.4f}(x{lambda_gap_elt}={lambda_gap_elt*_loss_elt_gap_val:.4f})"
                )
            if _loss_frust_val is not None:
                parts.append(
                    f"frust={_loss_frust_val:.4f}(x{lambda_frustration}={lambda_frustration*_loss_frust_val:.4f})"
                )
            if _loss_native_depth_val is not None:
                parts.append(
                    f"depth={_loss_native_depth_val:.4f}(x{lambda_native_depth}={lambda_native_depth*_loss_native_depth_val:.4f})"
                )
            if _loss_discrim_val is not None:
                parts.append(
                    f"discrim={_loss_discrim_val:.4f}(x{lambda_discrim}={lambda_discrim*_loss_discrim_val:.4f})"
                )
            if _loss_rg_val is not None:
                parts.append(f"rg={_loss_rg_val:.4f}(x{lambda_rg}={lambda_rg*_loss_rg_val:.4f})")
            loss_str = " ".join(parts)
            flags = []
            if freeze_gates_steps > 0 and phase_step <= freeze_gates_steps:
                flags.append("gates=frozen")
            if spike_counter > 0:
                flags.append(f"spikes={spike_counter}")
            if bad_grad_streak > 0:
                flags.append(f"badgrad={bad_grad_streak}")
            flag_str = "  " + "  ".join(flags) if flags else ""
            logger.info(
                "[%s] step %6d/%d (global=%d) | loss=%.4f | lr=%.2e%s%s",
                config.name,
                phase_step,
                n_steps,
                trainer.global_step,
                trainer.current_loss,
                current_lr,
                (f"  {loss_str}" if loss_str else ""),
                flag_str,
            )

        # ── Update EMA tracker every step ────────────────────────────────
        if hasattr(trainer, "diagnostic_logger") and trainer.diagnostic_logger is not None:
            _ema_kw = {}
            # Total loss and DSM
            if trainer.current_loss:
                _ema_kw["total"] = trainer.current_loss
            if _loss_dsm_val is not None:
                _ema_kw["dsm"] = _loss_dsm_val
                # Track individual DSM components for α-augmented mode
                if _dsm_diag.get("n_samples", 1) > 1:
                    _ema_kw["dsm_std"] = _dsm_diag.get("dsm_std", _loss_dsm_val)
                    if _dsm_diag.get("dsm_alpha") is not None:
                        _ema_kw["dsm_alpha"] = _dsm_diag["dsm_alpha"]
                    if _dsm_diag.get("dsm_mixed") is not None:
                        _ema_kw["dsm_mixed"] = _dsm_diag["dsm_mixed"]
            if _loss_balance_val is not None:
                _ema_kw["balance"] = _loss_balance_val
            if _loss_basin_val is not None:
                _ema_kw["basin_loss"] = _loss_basin_val
            if _basin_diag and "gap" in _basin_diag:
                _ema_kw["basin_gap"] = _basin_diag["gap"]
            # ELT metrics (computed every elt_every steps)
            if _elt_diag:
                if "n_anti_funnel" in _elt_diag and "n_pairs" in _elt_diag:
                    n_total = _elt_diag["n_pairs"]
                    if n_total > 0:
                        _ema_kw["anti_funnel"] = 100.0 * _elt_diag["n_anti_funnel"] / n_total
                if "Z_mean" in _elt_diag:
                    _ema_kw["z_score"] = _elt_diag["Z_mean"]
                if "E_decoy_pr" in _elt_diag:
                    _ema_kw["e_decoy"] = _elt_diag["E_decoy_pr"]
                if "mean_slope" in _elt_diag:
                    _ema_kw["slope"] = _elt_diag["mean_slope"]
                if "gap_mean" in _elt_diag:
                    _ema_kw["elt_gap"] = _elt_diag["gap_mean"]
                if "f_mean" in _elt_diag and lambda_frustration > 0:
                    _ema_kw["frust"] = _elt_diag["f_mean"]
                if "frac_frustrated" in _elt_diag and lambda_frustration > 0:
                    _ema_kw["frac_frust"] = 100.0 * _elt_diag["frac_frustrated"]
            # Note: Z-score EMA only updates on diagnostic steps (every 200).
            # Do NOT fall back to _loss_zscore_val — that's the loss, not Z.
            # Subterm |E| from balance (available every step)
            if _balance_absmeans:
                if "secondary_hb_alpha" in _balance_absmeans:
                    _ema_kw["hb_alpha"] = _balance_absmeans["secondary_hb_alpha"]
                if "secondary_hb_beta" in _balance_absmeans:
                    _ema_kw["hb_beta"] = _balance_absmeans["secondary_hb_beta"]
            # Term-level E% from balance term aggregates
            if _balance_term_absmeans:
                _abs_total = sum(_balance_term_absmeans.values()) or 1.0
                for _tk in ["local", "secondary", "repulsion", "packing"]:
                    if _tk in _balance_term_absmeans:
                        _ema_kw[f"E%{_tk}"] = 100.0 * _balance_term_absmeans[_tk] / _abs_total
            if forces is not None:
                _ema_kw["max_force"] = float(torch.norm(forces, dim=-1).max().item())
            if _e_native_val is not None:
                _ema_kw["e_native"] = _e_native_val
            if _loss_discrim_val is not None:
                _ema_kw["discrim"] = _loss_discrim_val
            trainer.diagnostic_logger.update_ema(**_ema_kw)

        if save_every > 0 and phase_step % save_every == 0:
            trainer.save_checkpoint(config.name, phase_step, trainer.current_loss)

        if val_loader and validate_every > 0 and phase_step % validate_every == 0:
            # Always save before validation in case the process is interrupted during the (potentially long) val run
            if not (save_every > 0 and phase_step % save_every == 0):
                trainer.save_checkpoint(config.name, phase_step, trainer.current_loss)
            val_metrics = trainer._validate(val_loader, phase_step)
            if math.isfinite(val_metrics.composite_score):
                if trainer._check_early_stopping(val_metrics.composite_score, config):
                    logger.info("Early stopping phase %s", config.name)
                    break
                if (trainer.best_composite_score is None) or (
                    val_metrics.composite_score < trainer.best_composite_score
                ):
                    trainer.best_composite_score = val_metrics.composite_score
                    trainer.best_composite_score_initialized = True
                    trainer.best_val_step = phase_step
                    trainer.save_checkpoint(config.name, phase_step, trainer.current_loss, is_best=True)
            trainer.model.train()

        progress.update(1)

    if not getattr(trainer, "converged", False):
        final_loss = float(getattr(trainer, "current_loss", 0.0))
        if final_loss == 0.0 and loss is not None and hasattr(loss, "item"):
            final_loss = float(loss.item())
        trainer.save_checkpoint(config.name, trainer.phase_step, final_loss)

    logger.info(
        "Phase %s complete%s",
        config.name,
        f" (converged at step {trainer.convergence_step})" if getattr(trainer, "converged", False) else "",
    )

    return TrainingState(
        global_step=trainer.global_step,
        phase_step=trainer.phase_step,
        phase=config.name,
        losses={"loss": float(getattr(trainer, "current_loss", 0.0))},
        gates=trainer.model.get_gates() if hasattr(trainer.model, "get_gates") else {},
        best_composite_score=trainer.best_composite_score,
        best_composite_score_initialized=trainer.best_composite_score_initialized,
        best_val_step=trainer.best_val_step,
        early_stopping_counter=trainer.early_stopping_counter,
        validation_history=trainer.validation_history,
        converged=trainer.converged,
        convergence_step=trainer.convergence_step,
        convergence_info=trainer.convergence_info,
    )
