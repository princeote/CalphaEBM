# src/calphaebm/defaults.py
"""Single source of truth for ALL hyperparameter defaults.

Every hardcoded number in the codebase should reference this file.
Organized by category. Imported by config.py, model_builder.py,
train_main.py, sc_defaults.py, full_stage.py, etc.

Usage:
    from calphaebm.defaults import MODEL, TRAIN, SC, EVAL

    rg_lambda = MODEL["rg_lambda"]           # 1.0
    funnel_m  = TRAIN["funnel_m"]            # 5.0
    collect_steps = SC["collect_steps"]      # 100_000

History:
    Run4: scattered across 8+ files, 25+ hardcoded values
    Run5: consolidated here (#7)
"""

# =====================================================================
# MODEL — architecture and energy function parameters
# =====================================================================

MODEL = {
    # Neighbour graph
    "topk": 64,
    "exclude": 3,
    "max_dist": 12.0,  # Å — Best-style (was 10.0 in v4)
    # Best et al. (2013) Cα-adapted sigmoid
    "sigmoid_r_half": 8.0,  # Å
    "sigmoid_tau": 0.2,  # Å (β = 5.0 Å⁻¹)
    # Rg Flory restraint (exponential with dead zone)
    "rg_lambda": 1.0,
    "rg_r0": 2.0,  # Å
    "rg_nu": 0.38,
    "rg_dead_zone": 0.30,  # ±30% of Rg*
    "rg_m": 1.0,  # penalty saturation
    "rg_alpha": 3.0,  # penalty steepness
    # Coordination penalty (exponential)
    "coord_lambda": 1.0,
    "coord_m": 1.0,  # penalty saturation
    "coord_alpha": 2.0,  # penalty steepness
    # Contact density ρ
    "rho_lambda": 1.0,  # reward scale (trainable)
    "rho_sigma": 0.7,  # Gaussian width
    "rho_m": 1.0,  # penalty saturation
    "rho_alpha": 2.0,  # penalty steepness
    "rho_penalty_lambda": 1.0,
    "rho_fit_a": 5.226,  # ρ*(L) = a - b·exp(-L/c)
    "rho_fit_b": 2.116,
    "rho_fit_c": 112.3,
    # Legacy packing params (accepted, mostly ignored in v5)
    "packing_r_on": 8.0,
    "packing_r_cut": 10.0,
    "packing_short_gate_on": 4.5,
    "packing_short_gate_off": 5.0,
    "packing_rbf_centers": (5.5, 7.0, 9.0),
    "packing_rbf_width": 1.0,
    "packing_init_from": "log_oe",
    # Local term
    "local_window_size": 8,
    "init_theta_theta_weight": 1.0,
    "init_delta_phi_weight": 1.0,
    # Repulsion
    "repulsion_K": 64,
    "repulsion_exclude": 3,
    "repulsion_r_on": 8.0,
    "repulsion_r_cut": 10.0,
}


# =====================================================================
# TRAIN — full-stage PDB training (stage=full)
# =====================================================================

TRAIN = {
    # Rounds
    "max_rounds": 10,
    "steps_per_round": 3000,
    "lr": 5e-4,
    "lr_final": 5e-5,
    "log_every": 100,
    # PDB batch losses
    "lambda_depth": 1.0,
    "target_depth": -3.0,
    "lambda_balance": 0.001,
    "balance_r": 7.0,  # subterm-level ratio bound (7 learned subterms)
    "balance_r_term": 4.0,  # term-level ratio bound (4 terms)
    "lambda_dsm": 1.0,
    # IC-noised losses
    "lambda_discrim": 2.0,
    "disc_T": 2.0,
    "lambda_qf": 1.0,
    "lambda_drmsd": 2.0,  # dRMSD-funnel (replaces lambda_rg)
    "lambda_gap": 3.0,
    "gap_margin": 1.0,  # DEPRECATED — kept for backward compat
    # Saturating exponential margins (Run5)
    "funnel_m": 5.0,
    "funnel_alpha": 5.0,
    "gap_m": 5.0,
    "gap_alpha": 5.0,
    # IC noise
    "sigma_min": 0.0524,  # π/60 rad (3°)
    "sigma_max": 1.0472,  # π/3 rad (60°)
    "n_decoys": 8,
    "T_funnel": 2.0,
    "decoy_every": 1,
    "discrim_every": 2,
    # DSM
    "dsm_sigma_min": 0.0524,
    "dsm_sigma_max": 1.0472,
    # Basin eval
    "eval_proteins": 64,
    "eval_steps": 5000,
    "eval_beta": 100.0,
    "eval_workers": 64,
    "eval_timeout": 86400,  # 24 hours — no artificial limit on eval
    "sampler": "mala",
    # Convergence
    "converge_q": 0.98,
    "converge_rmsd": 2.0,
    "converge_rg_lo": 95.0,
    "converge_rg_hi": 105.0,
}


# =====================================================================
# SC — self-consistent training (stage=sc)
# =====================================================================

SC = {
    # Collection
    "collect_proteins": 64,
    "collect_steps": 100_000,
    "collect_beta": 100.0,
    "collect_beta_min": 0.01,  # β = L × s, s ~ LogU[min, max]
    "collect_beta_max": 1.0,  # Run8: wide range for unfolded negatives
    "collect_betas": None,
    "collect_save_every": 100,
    "collect_step_size": 1e-4,
    "collect_force_cap": 100.0,
    "collect_n_workers": 64,
    "collect_max_len": 512,
    "sampler": "mala",
    # Retraining
    "retrain_steps": 2000,
    "retrain_lr": 1e-5,
    "log_every": 100,
    "save_every": 500,
    # PDB batch losses
    "lambda_elt": 1.0,
    "lambda_gap": 2.0,
    "lambda_discrim": 2.0,
    "lambda_depth": 2.0,
    "target_depth": -3.0,
    "lambda_balance": 0.01,
    "balance_r": 7.0,
    "balance_r_term": 4.0,
    # Sampled negative losses
    "lambda_sampled_hsm": 1.0,
    "lambda_sampled_qf": 1.0,
    "lambda_sampled_drmsd_funnel": 2.0,  # dRMSD-funnel (replaces lambda_sampled_rg_funnel)
    "lambda_sampled_gap": 2.0,
    "sc_margin": 1.0,
    # Saturating exponential margins (Run5)
    "funnel_m": 5.0,
    "funnel_alpha": 5.0,
    "gap_m": 5.0,
    "gap_alpha": 5.0,
    # Evaluation
    "eval_steps": 10000,
    "eval_beta": 100.0,
    "eval_proteins": 64,
    "eval_timeout": 86400,
    # Collector thresholds
    "rg_compact": 0.90,
    "rg_swollen": 1.10,
    "q_false_basin": 0.90,
    "rmsd_drift": 5.0,
    "rmsf_frozen": 0.3,
    "ss_change_thr": 0.3,
    "max_negatives_per_protein": 8,
    # Run control
    "n_rounds": 10,
    "convergence_threshold": 0.05,
    "min_negatives": 10,
    # Convergence
    "converge_q": 0.98,
    "converge_rmsd": 2.0,
    "converge_rg_lo": 95.0,
    "converge_rg_hi": 105.0,
}


# =====================================================================
# Convenience
# =====================================================================


def get(section: str, key: str):
    """Get a default value by section and key.

    Args:
        section: "model", "train", or "sc"
        key: parameter name

    Example:
        get("model", "rg_lambda")  # 1.0
    """
    d = {"model": MODEL, "train": TRAIN, "sc": SC}
    return d[section][key]
