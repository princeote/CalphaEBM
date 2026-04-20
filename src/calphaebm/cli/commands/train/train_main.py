# src/calphaebm/cli/commands/train/train_main.py

"""Train command main driver."""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from calphaebm.data.pdb_chain_dataset import PDBChainDataset
from calphaebm.data.pdb_dataset import PDBSegmentDataset
from calphaebm.defaults import MODEL as _M
from calphaebm.defaults import TRAIN as _T
from calphaebm.training.core.config import PhaseConfig
from calphaebm.training.phased import PhasedTrainer
from calphaebm.utils.logging import get_logger
from calphaebm.utils.seed import seed_all

from .checkpoint import load_checkpoint, load_weights_into_model
from .data_utils import parse_pdb_arg, save_split_ids
from .model_builder import build_model, verify_model_terms
from .phase_utils import (
    determine_terms_for_phase,
    get_active_terms_for_phase,
    get_loss_fn_for_phase,
    get_ramp_config,
    get_validate_every_for_phase,
)

logger = get_logger()


def load_native_structures(pdb_ids, cache_dir, device):
    """Load native structures for validation RMSD calculation."""
    return None


def _read_ids_file(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Take first token only (strips inline comments like "1ABC  # description")
        ids.append(s.split()[0])
    return ids


def _try_load_saved_split(split_dir: Path) -> tuple[list[str], list[str]] | None:
    """Try to load existing train/val split IDs from disk."""
    candidates = [
        ("train_ids.txt", "val_ids.txt"),
        ("train_ids.list", "val_ids.list"),
        ("train_ids", "val_ids"),
        ("train.txt", "val.txt"),
    ]
    for tr_name, va_name in candidates:
        tr_path = split_dir / tr_name
        va_path = split_dir / va_name
        if tr_path.exists() and va_path.exists():
            train_ids = _read_ids_file(tr_path)
            val_ids = _read_ids_file(va_path)
            if train_ids and val_ids:
                logger.info("Loaded existing train/val split from %s (%s, %s)", str(split_dir), tr_name, va_name)
                return train_ids, val_ids
            logger.warning("Found split files but they were empty: %s and %s", str(tr_path), str(va_path))
    return None


def run(args):
    """Run train command."""
    # Convert "none" string to None for lr_schedule
    if getattr(args, "lr_schedule", None) == "none":
        args.lr_schedule = None

    seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Stage-based dispatch: map --stage to --phase for model builder ───────
    if getattr(args, "stage", None) is not None:
        _stage_to_phase = {"full": "full", "sc": "self-consistent"}
        args.phase = _stage_to_phase[args.stage]
        logger.info("Stage mode: --stage %s (mapped to phase=%s)", args.stage, args.phase)
    elif getattr(args, "phase", None) is None:
        logger.error("Must specify --stage (full|sc) or --phase")
        return 1

    # ── Determine train/val split ────────────────────────────────────────────
    split_dir = Path(args.ckpt_dir) / args.ckpt_prefix
    split_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "train_pdb", None) and getattr(args, "val_pdb", None):
        # Explicit train/val files
        train_ids = _read_ids_file(Path(args.train_pdb))
        val_ids = _read_ids_file(Path(args.val_pdb))
        if not train_ids:
            logger.error("No valid training PDB IDs found in %s", args.train_pdb)
            return 1
        if not val_ids:
            logger.error("No valid validation PDB IDs found in %s", args.val_pdb)
            return 1
        logger.info(
            "Explicit split: %d train from %s, %d val from %s",
            len(train_ids),
            args.train_pdb,
            len(val_ids),
            args.val_pdb,
        )
        save_split_ids(train_ids, val_ids, args.ckpt_dir, args.ckpt_prefix)
    else:
        # Legacy: --pdb with 80/20 split
        pdb_ids = parse_pdb_arg(args.pdb)
        if not pdb_ids:
            logger.error("No valid PDB IDs found")
            return 1
        logger.info("Using %d PDB entries", len(pdb_ids))
        loaded = _try_load_saved_split(split_dir)
        if loaded is not None:
            train_ids, val_ids = loaded
        else:
            rng = random.Random(42)
            rng.shuffle(pdb_ids)
            split = int(0.8 * len(pdb_ids))
            train_ids = pdb_ids[:split]
            val_ids = pdb_ids[split:]
            save_split_ids(train_ids, val_ids, args.ckpt_dir, args.ckpt_prefix)
            logger.info("Saved train/val splits to %s", str(split_dir))
    logger.info("Training on %d IDs, validating on %d IDs", len(train_ids), len(val_ids))

    # ── Datasets / loaders ───────────────────────────────────────────────────
    # Full protein chains for all phases. Eliminates boundary artifacts from
    # segmentation (~15-20% of residues corrupted at segment boundaries).
    # Cap at --elt-max-len (default 512) for memory safety.
    # Falls back to segments only if PDBChainDataset fails or is empty.
    cache_processed = not args.no_cache

    elt_max_len = int(getattr(args, "elt_max_len", 512) or 512)
    elt_batch_size = int(getattr(args, "elt_batch_size", 8) or 8)
    max_rg_ratio = getattr(args, "max_rg_ratio", 1.3)
    if max_rg_ratio and max_rg_ratio <= 0:
        max_rg_ratio = None  # disable

    use_full_chains = True  # always try full chains first
    train_loader = None

    try:
        train_chain_dataset = PDBChainDataset(
            pdb_ids=train_ids,
            cache_dir=args.cache_dir,
            min_len=40,
            max_len=elt_max_len,
            max_rg_ratio=max_rg_ratio,
            cache_processed=cache_processed,
            processed_cache_dir=args.processed_cache_dir,
            force_reprocess=args.force_reprocess,
        )
        if len(train_chain_dataset) > 0:
            train_loader = DataLoader(
                train_chain_dataset,
                batch_size=elt_batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
                collate_fn=PDBChainDataset.collate,
            )
            logger.info(
                "FULL-CHAIN MODE: %d chains (L=40-%d), %d batches of %d — " "no segmentation, no boundary artifacts",
                len(train_chain_dataset),
                elt_max_len,
                len(train_loader),
                elt_batch_size,
            )
        else:
            logger.error(
                "PDBChainDataset is empty (no chains with L=40-%d). " "FALLING BACK TO SEGMENTS.",
                elt_max_len,
            )
            use_full_chains = False
    except Exception as e:
        logger.error("PDBChainDataset failed: %s — FALLING BACK TO SEGMENTS.", e)
        use_full_chains = False

    if not use_full_chains:
        # ── Segment fallback ──────────────────────────────────────────
        logger.warning(
            "Using SEGMENT batches (seg_len=%d, batch=%d). "
            "Boundary residues (~15-20%%) will have corrupted packing context. "
            "Fix PDBChainDataset or increase --elt-max-len to avoid this.",
            args.seg_len,
            args.batch_size,
        )
        train_seg_dataset = PDBSegmentDataset(
            pdb_ids=train_ids,
            cache_dir=args.cache_dir,
            seg_len=args.seg_len,
            stride=args.stride,
            limit_segments=args.limit,
            validate_geometry=True,
            cache_processed=cache_processed,
            processed_cache_dir=args.processed_cache_dir,
            force_reprocess=args.force_reprocess,
        )
        train_loader = DataLoader(
            train_seg_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
        )
        logger.info("Train dataset: %d segments (seg_len=%d)", len(train_seg_dataset), args.seg_len)

    # ── Val loader (also full chains) ─────────────────────────────────────
    val_loader = None
    try:
        val_chain_dataset = PDBChainDataset(
            pdb_ids=val_ids,
            cache_dir=args.cache_dir,
            min_len=40,
            max_len=elt_max_len,
            max_rg_ratio=max_rg_ratio,
            cache_processed=cache_processed,
            processed_cache_dir=args.processed_cache_dir,
            force_reprocess=args.force_reprocess,
        )
        if len(val_chain_dataset) > 0:
            val_loader = DataLoader(
                val_chain_dataset,
                batch_size=elt_batch_size,
                shuffle=False,
                num_workers=4,
                drop_last=False,
                collate_fn=PDBChainDataset.collate,
            )
            logger.info(
                "Val dataset: %d full chains (L=40-%d), %d batches",
                len(val_chain_dataset),
                elt_max_len,
                len(val_loader),
            )
        else:
            logger.warning("Val PDBChainDataset is empty — falling back to segments for validation")
    except Exception as e:
        logger.warning("Val PDBChainDataset failed: %s — falling back to segments", e)

    if val_loader is None:
        val_seg_dataset = PDBSegmentDataset(
            pdb_ids=val_ids,
            cache_dir=args.cache_dir,
            seg_len=args.seg_len,
            stride=args.stride,
            limit_segments=max(args.limit // 5, 1),
            validate_geometry=True,
            cache_processed=cache_processed,
            processed_cache_dir=args.processed_cache_dir,
            force_reprocess=args.force_reprocess,
        )
        val_loader = DataLoader(
            val_seg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False
        )
        logger.info("Val dataset: %d segments (fallback)", len(val_seg_dataset))

    # ── Determine terms for model building ───────────────────────────────────
    # Self-consistent uses same terms as full phase
    _phase_for_terms = "full" if args.phase == "self-consistent" else args.phase
    build_terms_set = determine_terms_for_phase(_phase_for_terms, args.energy_terms)
    logger.info("Model build terms: %s", sorted(build_terms_set))

    # ── Load checkpoint (optional) ───────────────────────────────────────────
    checkpoint_state = None
    trainer_state: dict = {}

    if args.resume:
        resume_path = args.resume
        if resume_path == "auto":
            phase_dir = Path(args.ckpt_dir) / args.ckpt_prefix / args.phase
            from .checkpoint import find_latest_checkpoint as _find_latest_checkpoint

            resume_path = _find_latest_checkpoint(phase_dir)
            if resume_path is None:
                logger.error("No checkpoint found for auto-resume in %s", str(phase_dir))
                return 1
        checkpoint_state, trainer_state = load_checkpoint(resume_path, device)

    # ── Build model + load weights ────────────────────────────────────────────
    model = build_model(build_terms_set, device, args)

    if checkpoint_state is not None:
        load_weights_into_model(model, checkpoint_state)

    # ── Override registered buffers from CLI (checkpoint overwrites these) ────
    # Buffers are saved in state_dict and restored by load_weights_into_model,
    # but hyperparameter buffers (rg_lambda, coord_lambda, etc.) should be
    # controlled by CLI flags, not frozen at checkpoint values.
    # EXCEPTION: when buffers are learnable Parameters (--learn-*-buffers),
    # their trained values must be preserved — CLI defaults would overwrite them.
    if checkpoint_state is not None and hasattr(model, "packing") and model.packing is not None:
        _packing = model.packing
        _buf_overrides = {
            "rg_lambda": getattr(args, "packing_rg_lambda", None),
            "rg_r0": getattr(args, "packing_rg_r0", None),
            "rg_nu": getattr(args, "packing_rg_nu", None),
            "coord_lambda": getattr(args, "hp_penalty_lambda", getattr(args, "coord_lambda", None)),
        }
        for _buf_name, _cli_val in _buf_overrides.items():
            if _cli_val is not None and hasattr(_packing, _buf_name):
                _attr = getattr(_packing, _buf_name)
                # Skip override if this is a learnable Parameter — preserve trained value
                if isinstance(_attr, torch.nn.Parameter) and _attr.requires_grad:
                    logger.info(
                        "Buffer override SKIPPED: packing.%s = %.4f (learnable, preserving trained value)",
                        _buf_name,
                        _attr.item(),
                    )
                    continue
                _old_val = _attr.item()
                _attr.data.fill_(float(_cli_val))
                _new_val = _attr.item()
                if abs(_old_val - _new_val) > 1e-6:
                    logger.info("Buffer override: packing.%s  %.4f → %.4f (CLI)", _buf_name, _old_val, _new_val)

    # ── CLI init overrides (applied AFTER checkpoint load) ────────────────────
    # bond_spring is GONE — only angle weights remain.
    import math as _math

    import torch.nn.functional as _F

    _INIT_DEFAULTS = {
        "init_theta_theta_weight": 1.0,
        "init_delta_phi_weight": 1.0,
    }

    def _set_softplus_raw(param, val):
        raw = val if val > 20.0 else _math.log(_math.exp(val) - 1.0)
        param.data.fill_(raw)

    if args.resume_model_only and checkpoint_state is not None:
        local = getattr(model, "local", None)
        if local is not None:
            # Old 3-subterm architecture overrides
            if hasattr(local, "_theta_theta_mlp_w") or hasattr(local, "_theta_theta_weight_raw"):
                if args.init_theta_theta_weight != _INIT_DEFAULTS["init_theta_theta_weight"]:
                    _raw = getattr(local, "_theta_theta_mlp_w", getattr(local, "_theta_theta_weight_raw", None))
                    if _raw is not None:
                        _set_softplus_raw(_raw, args.init_theta_theta_weight)
                        logger.info("CLI override: theta_theta_weight → %.4f", args.init_theta_theta_weight)
            if hasattr(local, "_delta_phi_weight_raw"):
                if args.init_delta_phi_weight != _INIT_DEFAULTS["init_delta_phi_weight"]:
                    _set_softplus_raw(local._delta_phi_weight_raw, args.init_delta_phi_weight)
                    logger.info("CLI override: delta_phi_weight → %.4f", args.init_delta_phi_weight)

    # ── Gate overrides (applied AFTER checkpoint load) ────────────────────────
    import torch as _torch

    _gate_overrides = {
        "gate_local": getattr(args, "set_gate_local", None),
        "gate_secondary": getattr(args, "set_gate_secondary", None),
        "gate_repulsion": getattr(args, "set_gate_repulsion", None),
        "gate_packing": getattr(args, "set_gate_packing", None),
    }
    for _gate_name, _gate_val in _gate_overrides.items():
        if _gate_val is not None:
            if hasattr(model, _gate_name):
                getattr(model, _gate_name).fill_(_gate_val)
                logger.info("Gate override: %s → %.4f", _gate_name, _gate_val)
            else:
                logger.warning("--set-%s requested but model has no %s", _gate_name.replace("_", "-"), _gate_name)

    # ── Log actual values post-load ───────────────────────────────────────────
    if getattr(model, "local", None) is not None:
        local = model.local
        g = getattr(model, "gate_local", _torch.tensor(1.0)).item()
        if hasattr(local, "theta_phi_weight"):
            # 4-mer architecture
            w = local.theta_phi_weight.item()
            logger.info("Local lambdas (pre-gate):  theta_phi=%.3f", w)
            logger.info("Local lambdas (×gate=%.4f): theta_phi=%.3f", g, w * g)
        else:
            # Old 3-subterm architecture
            tw = local.theta_theta_weight.item() if hasattr(local, "theta_theta_weight") else 0.0
            dw = local.delta_phi_weight.item() if hasattr(local, "delta_phi_weight") else 0.0
            logger.info("Local lambdas (pre-gate):  theta_theta=%.3f  delta_phi=%.3f", tw, dw)
            logger.info("Local lambdas (×gate=%.4f): theta_theta=%.3f  delta_phi=%.3f", g, tw * g, dw * g)
    for _gate_name in ["gate_local", "gate_secondary", "gate_repulsion", "gate_packing"]:
        if hasattr(model, _gate_name):
            logger.info("Gate %s = %.4f", _gate_name, getattr(model, _gate_name).item())

    if not verify_model_terms(model, _phase_for_terms):
        return 1

    # ── Stage-based dispatch ─────────────────────────────────────────────────
    if getattr(args, "stage", None) == "full":
        from calphaebm.training.full_stage import run_full_stage

        # Build a config namespace from args
        class _StageConfig:
            pass

        _cfg = _StageConfig()
        # Round structure
        _cfg.max_rounds = int(getattr(args, "max_rounds", _T["max_rounds"]))
        _cfg.steps_per_round = int(getattr(args, "steps_per_round", _T["steps_per_round"]))
        _lr = getattr(args, "lr", None)
        _cfg.lr = float(_lr if _lr is not None else _T["lr"])
        _lr_f = getattr(args, "lr_final", None)
        _cfg.lr_final = float(_lr_f if _lr_f is not None else _cfg.lr / 10)
        _cfg.log_every = int(getattr(args, "log_every", _T["log_every"]))
        # PDB batch losses
        _ld = getattr(args, "lambda_native_depth", None)
        _cfg.lambda_depth = float(_ld if _ld is not None else _T["lambda_depth"])
        _td = getattr(args, "target_native_depth", None)
        _cfg.target_depth = float(_td if _td is not None else _T["target_depth"])
        _lb = getattr(args, "lambda_balance", None)
        _cfg.lambda_balance = float(_lb if _lb is not None else _T["lambda_balance"])
        _cfg.balance_r = float(getattr(args, "balance_r", _T["balance_r"]))
        _cfg.balance_r_term = float(getattr(args, "balance_r_term", _T["balance_r_term"]))
        _ldsm = getattr(args, "lambda_dsm", None)
        _cfg.lambda_dsm = float(_ldsm if _ldsm is not None else _T["lambda_dsm"])
        # IC-noised losses
        _cfg.lambda_discrim = float(getattr(args, "lambda_discrim", _T["lambda_discrim"]))
        _cfg.disc_T = float(getattr(args, "disc_T", _T["disc_T"]))
        _cfg.lambda_qf = float(getattr(args, "lambda_qf", _T["lambda_qf"]))
        # dRMSD-funnel — honour deprecated --lambda-rg if explicitly set
        _lambda_rg_alias = getattr(args, "lambda_rg", None)
        _cfg.lambda_drmsd = float(
            _lambda_rg_alias if _lambda_rg_alias is not None else getattr(args, "lambda_drmsd", _T["lambda_drmsd"])
        )
        _cfg.lambda_gap = float(getattr(args, "lambda_gap", _T["lambda_gap"]))
        _cfg.gap_margin = float(getattr(args, "gap_margin", _T["gap_margin"]))
        # Run5: saturating exponential margins
        _cfg.funnel_m = float(getattr(args, "funnel_m", _T["funnel_m"]))
        _cfg.funnel_alpha = float(getattr(args, "funnel_alpha", _T["funnel_alpha"]))
        _cfg.gap_m = float(getattr(args, "gap_m", _T["gap_m"]))
        _cfg.gap_alpha = float(getattr(args, "gap_alpha", _T["gap_alpha"]))
        # IC noise params
        _cfg.sigma_min = float(getattr(args, "sigma_min_ic", _T["sigma_min"]))
        _cfg.sigma_max = float(getattr(args, "sigma_max_ic", _T["sigma_max"]))
        _cfg.n_decoys = int(getattr(args, "n_decoys", _T["n_decoys"]))
        _cfg.T_funnel = float(getattr(args, "T_funnel", _T["T_funnel"]))
        _cfg.dsm_sigma_min = _cfg.sigma_min
        _cfg.dsm_sigma_max = _cfg.sigma_max
        # Amortise
        _cfg.decoy_every = int(getattr(args, "decoy_every", 1))
        _cfg.discrim_every = int(getattr(args, "discrim_every", 2))
        # Disabled subterms
        _cfg.disable_subterms = list(getattr(args, "disable_subterms", None) or [])
        # Basin eval
        _cfg.eval_proteins = int(getattr(args, "val_proteins", 16))
        _cfg.eval_steps = int(getattr(args, "val_steps", 5000))
        _cfg.eval_beta = float(getattr(args, "val_beta", 100.0))
        _cfg.eval_workers = int(getattr(args, "n_workers", 16))
        _cfg.collect_proteins = int(getattr(args, "collect_proteins", 1024))
        _cfg.collect_max_len = int(getattr(args, "collect_max_len", getattr(args, "elt_max_len", 512)))
        # Convergence
        _cfg.converge_q = float(getattr(args, "converge_q", 0.95))
        _cfg.converge_rmsd = float(getattr(args, "converge_rmsd", 5.0))
        _cfg.converge_rg_lo = float(getattr(args, "converge_rg_lo", 95.0))
        _cfg.converge_rg_hi = float(getattr(args, "converge_rg_hi", 105.0))

        # Create trainer wrapper
        from calphaebm.training.phased import PhasedTrainer

        trainer = PhasedTrainer(model=model, device=device, ckpt_dir=args.ckpt_dir, experiment_prefix=args.ckpt_prefix)

        # Restore trainer state from checkpoint (global_step, phase_step, etc.)
        # --resume: full restore (training continues where it left off)
        # --resume-model-only: skip (fresh start with trained weights)
        _is_model_only = getattr(args, "resume_model_only", False)
        if trainer_state and not _is_model_only:
            trainer.global_step = trainer_state.get("global_step", 0)
            trainer.phase_step = trainer_state.get("phase_step", 0)
            trainer.best_composite_score = trainer_state.get("best_composite_score")
            trainer.best_composite_score_initialized = trainer_state.get("best_composite_score_initialized", False)
            trainer.best_val_step = trainer_state.get("best_val_step", 0)
            trainer.early_stopping_counter = trainer_state.get("early_stopping_counter", 0)
            trainer.validation_history = trainer_state.get("validation_history", [])
            logger.info(
                "Full resume: restored trainer state (global=%d, phase=%d)", trainer.global_step, trainer.phase_step
            )
        elif _is_model_only and checkpoint_state is not None:
            logger.info("Model-only resume: training state reset (global_step=0, fresh start)")

        result = run_full_stage(trainer, _cfg, train_loader, val_loader)
        logger.info("Stage full complete")
        return 0

    if getattr(args, "stage", None) == "sc":
        args.phase = "self-consistent"
        # Fall through to existing SC dispatch below

    # ── Self-consistent phase: separate loop ─────────────────────────────────
    # Dispatches early — all the PhaseConfig/gate/active_terms logic below is
    # only needed for standard training phases, not for the SC meta-loop.
    if args.phase == "self-consistent":
        from calphaebm.training.sc_defaults import SC_DEFAULTS as _D
        from calphaebm.training.self_consistent import SelfConsistentTrainer

        sc_out_dir = str(Path(args.ckpt_dir) / args.ckpt_prefix / "self-consistent")
        sc_trainer = SelfConsistentTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            # Collection
            collect_proteins=int(getattr(args, "collect_proteins", _D["collect_proteins"])),
            collect_steps=int(getattr(args, "collect_steps", _D["collect_steps"])),
            collect_beta=float(getattr(args, "collect_beta", _D["collect_beta"])),
            collect_step_size=float(getattr(args, "collect_step_size", _D["collect_step_size"])),
            collect_save_every=int(getattr(args, "collect_save_every", _D["collect_save_every"])),
            collect_n_workers=int(getattr(args, "n_workers", _D["collect_n_workers"])),
            collect_max_len=int(getattr(args, "collect_max_len", _D["collect_max_len"])),
            # Synthetic decoy losses
            retrain_steps=int(getattr(args, "retrain_steps", _D["retrain_steps"])),
            retrain_lr=float(getattr(args, "retrain_lr", _D["retrain_lr"])),
            lambda_elt=float(getattr(args, "lambda_funnel", _D["lambda_elt"])),
            lambda_gap=float(getattr(args, "lambda_gap", _D["lambda_gap"])),
            lambda_depth=float(getattr(args, "lambda_native_depth", _D["lambda_depth"])),
            target_depth=float(getattr(args, "target_native_depth", _D["target_depth"])),
            # Model-level losses
            lambda_discrim=float(getattr(args, "lambda_discrim", _D["lambda_discrim"])),
            lambda_balance=float(getattr(args, "lambda_balance", _D["lambda_balance"])),
            balance_r=float(getattr(args, "balance_r", _D["balance_r"])),
            balance_r_term=float(getattr(args, "balance_r_term", _D["balance_r_term"])),
            # Model-sampled negative losses (3× Synthetic counterparts)
            lambda_sampled_hsm=float(getattr(args, "lambda_sampled_hsm", _D["lambda_sampled_hsm"])),
            lambda_sampled_qf=float(getattr(args, "lambda_sampled_qf", _D["lambda_sampled_qf"])),
            lambda_sampled_drmsd_funnel=float(
                getattr(args, "lambda_sampled_drmsd_funnel", _D["lambda_sampled_drmsd_funnel"])
            ),
            lambda_sampled_gap=float(getattr(args, "lambda_sampled_gap", _D["lambda_sampled_gap"])),
            sc_margin=float(getattr(args, "sc_margin", _D["sc_margin"])),
            # Run5: saturating exponential margins
            funnel_m=float(getattr(args, "funnel_m", _D.get("funnel_m", 5.0))),
            funnel_alpha=float(getattr(args, "funnel_alpha", _D.get("funnel_alpha", 5.0))),
            gap_m=float(getattr(args, "gap_m", _D.get("gap_m", 5.0))),
            gap_alpha=float(getattr(args, "gap_alpha", _D.get("gap_alpha", 5.0))),
            # Evaluation
            eval_steps=int(getattr(args, "sc_eval_steps", _D["eval_steps"])),
            eval_beta=float(getattr(args, "sc_eval_beta", _D["eval_beta"])),
            eval_proteins=int(getattr(args, "sc_eval_proteins", _D["eval_proteins"])),
            # Thresholds
            rg_compact=float(getattr(args, "rg_compact", _D["rg_compact"])),
            rg_swollen=float(getattr(args, "rg_swollen", _D["rg_swollen"])),
            q_false_basin=float(getattr(args, "q_false_basin", _D["q_false_basin"])),
            rmsd_drift=float(getattr(args, "rmsd_drift", _D["rmsd_drift"])),
            rmsf_frozen=float(getattr(args, "rmsf_frozen", _D["rmsf_frozen"])),
            ss_change_thr=float(getattr(args, "ss_change_thr", _D["ss_change_thr"])),
            max_negatives_per_protein=int(getattr(args, "max_negatives_per_protein", _D["max_negatives_per_protein"])),
            # Disabled subterms
            disable_subterms=list(getattr(args, "disable_subterms", None) or []),
            # Output
            out_dir=sc_out_dir,
        )

        results = sc_trainer.run(
            n_rounds=int(getattr(args, "n_rounds", _D["n_rounds"])),
            convergence_threshold=float(getattr(args, "convergence_threshold", _D["convergence_threshold"])),
            min_negatives=int(getattr(args, "min_negatives", _D["min_negatives"])),
            resume_round=int(getattr(args, "sc_resume_round", 0)),
        )

        logger.info("Phase self-consistent complete")
        return 0

    if args.debug_mode and getattr(model, "secondary", None) is not None and hasattr(model.secondary, "debug_mode"):
        model.secondary.debug_mode = True
        logger.info("Debug mode enabled for secondary term")

    # ── Active terms ──────────────────────────────────────────────────────────
    energy_terms_lower = [t.lower() for t in (args.energy_terms or [])]
    requested_secondary = ("secondary" in energy_terms_lower) or ("all" in energy_terms_lower)
    model_has_secondary = getattr(model, "secondary", None) is not None

    if args.phase == "repulsion":
        active_terms = ["local", "repulsion"]
        if requested_secondary:
            if not model_has_secondary:
                logger.warning("Secondary requested for repulsion phase but model.secondary is missing")
            else:
                active_terms.append("secondary")
            logger.info("Phase repulsion: including secondary term (calibration mode)")
        else:
            logger.info("Phase repulsion: local + repulsion only (safety pass)")
    else:
        active_terms = get_active_terms_for_phase(_phase_for_terms)

    if "secondary" in active_terms and not model_has_secondary:
        logger.error("Active terms include 'secondary' but model.secondary is None")
        return 1
    if "repulsion" in active_terms and getattr(model, "repulsion", None) is None:
        logger.error("Active terms include 'repulsion' but model.repulsion is None")
        return 1
    if "packing" in active_terms and getattr(model, "packing", None) is None:
        logger.error("Active terms include 'packing' but model.packing is None")
        return 1

    logger.info("Active terms for phase %s: %s", args.phase, active_terms)

    # ── PhaseConfig construction ──────────────────────────────────────────────
    loss_fn = get_loss_fn_for_phase(args.phase)
    validate_every = get_validate_every_for_phase(args.phase, args.validate_every)
    ramp_start, ramp_end = get_ramp_config(args)

    lr_final = args.lr_final
    if lr_final is None and args.lr_schedule is not None:
        lr_final = args.lr * 0.01

    gate_schedule = {
        "secondary": [0.0, 1.0] if args.phase == "secondary" else None,
        "packing": None,
        "repulsion": None,
        "local": None,
    }

    config_kwargs: dict = {}

    if args.phase == "packing":
        config_kwargs["ramp_pack_start"] = float(args.ramp_pack_start)
        config_kwargs["ramp_pack_end"] = float(args.ramp_pack_end)
        config_kwargs["ramp_steps"] = int(args.ramp_steps)
        config_kwargs["packing_pretrain"] = bool(getattr(args, "packing_pretrain", False))
        logoe_dir = getattr(args, "packing_logoe_data_dir", None)
        if logoe_dir:
            config_kwargs["packing_logoe_data_dir"] = logoe_dir
        config_kwargs["packing_logoe_scale"] = float(getattr(args, "packing_logoe_scale", 5.0))

    ramp_gates = bool(getattr(args, "ramp_gates", False)) if args.phase == "full" else False
    if args.phase == "full":
        config_kwargs["ramp_gates"] = ramp_gates
        config_kwargs["ramp_steps"] = int(args.ramp_steps)
        config_kwargs["ramp_start"] = ramp_start
        config_kwargs["ramp_end"] = ramp_end

    config_kwargs["freeze_gates_steps"] = int(getattr(args, "freeze_gates_steps", 0) or 0)
    config_kwargs["freeze_packing_scalar"] = bool(getattr(args, "freeze_packing_scalar", False))
    config_kwargs["scalar_lr_mult"] = float(getattr(args, "scalar_lr_mult", 20.0) or 20.0)
    config_kwargs["lambda_pack_contrastive"] = float(getattr(args, "lambda_pack_contrastive", 0.0) or 0.0)
    config_kwargs["pack_contrastive_margin"] = float(getattr(args, "pack_contrastive_margin", 0.5) or 0.5)
    config_kwargs["pack_contrastive_mode"] = str(getattr(args, "pack_contrastive_mode", "continuous") or "continuous")
    config_kwargs["pack_contrastive_T_base"] = float(getattr(args, "pack_contrastive_T_base", 2.0) or 2.0)
    config_kwargs["lambda_balance"] = float(getattr(args, "lambda_balance", 0.0) or 0.0)
    config_kwargs["balance_r"] = float(getattr(args, "balance_r", 3.0) or 3.0)
    config_kwargs["balance_r_term"] = float(getattr(args, "balance_r_term", 4.0) or 4.0)
    config_kwargs["lambda_discrim"] = float(getattr(args, "lambda_discrim", 0.0) or 0.0)
    config_kwargs["discrim_every"] = int(getattr(args, "discrim_every", 4) or 4)
    config_kwargs["discrim_sigma_min"] = float(getattr(args, "discrim_sigma_min", 0.05) or 0.05)
    config_kwargs["discrim_sigma_max"] = float(getattr(args, "discrim_sigma_max", 2.0) or 2.0)
    config_kwargs["discrim_mode"] = str(getattr(args, "discrim_mode", "mean") or "mean")
    # dRMSD-funnel — honour deprecated --lambda-rg if set in legacy scripts
    _lr_alias = getattr(args, "lambda_rg", None)
    config_kwargs["lambda_drmsd"] = float(
        _lr_alias if _lr_alias is not None else getattr(args, "lambda_drmsd", 0.0) or 0.0
    )
    config_kwargs["rg_alpha_min"] = float(getattr(args, "rg_alpha_min", 0.75))
    config_kwargs["rg_alpha_max"] = float(getattr(args, "rg_alpha_max", 1.25))

    # ── IC DSM sigma forwarding (RADIANS, not Å) ──────────────────────────────
    if hasattr(args, "sigma_rad") and args.sigma_rad is not None:
        config_kwargs["sigma"] = float(args.sigma_rad)

    # Multi-scale sigma — available to ALL phases (local, secondary, packing, full)
    sigma_min_rad = getattr(args, "sigma_min_rad", None)
    sigma_max_rad = getattr(args, "sigma_max_rad", None)
    if sigma_min_rad is not None:
        config_kwargs["sigma_min"] = float(sigma_min_rad)
    if sigma_max_rad is not None:
        config_kwargs["sigma_max"] = float(sigma_max_rad)

    # DSM alpha augmentation — bidirectional Rg perturbation (run53+)
    config_kwargs["alpha_min"] = float(getattr(args, "dsm_alpha_min", 0.65))
    config_kwargs["alpha_max"] = float(getattr(args, "dsm_alpha_max", 1.25))

    if args.phase == "full":
        # ── IC Force balance loss ─────────────────────────────────────────────
        config_kwargs["lambda_fb"] = float(getattr(args, "lambda_fb", 0.0) or 0.0)
        config_kwargs["fb_clash_frac"] = float(getattr(args, "fb_clash_phi_frac", 0.1) or 0.1)
        config_kwargs["fb_clash_sigma"] = float(getattr(args, "fb_clash_phi_sigma", 1.0) or 1.0)
        config_kwargs["fb_target_ss_ratio"] = float(getattr(args, "fb_target_ss_ratio", 2.0) or 2.0)
        config_kwargs["fb_target_pack_ratio"] = float(getattr(args, "fb_target_pack_ratio", 2.0) or 2.0)
        config_kwargs["fb_target_rep_ratio"] = float(getattr(args, "fb_target_rep_ratio", 2.0) or 2.0)
        config_kwargs["fb_diag_every"] = int(getattr(args, "fb_diag_every", 200) or 200)

        # ── IC Local geometry gap loss ────────────────────────────────────────
        config_kwargs["lambda_geogap"] = float(getattr(args, "lambda_geogap", 0.0) or 0.0)
        config_kwargs["geogap_margin"] = float(getattr(args, "geogap_margin", 2.0) or 2.0)
        config_kwargs["geogap_angle_sigma"] = float(getattr(args, "geogap_theta_sigma", 0.25) or 0.25)
        config_kwargs["geogap_dihedral_sigma"] = float(getattr(args, "geogap_phi_sigma", 0.5) or 0.5)
        config_kwargs["geogap_frac_perturbed"] = float(getattr(args, "geogap_frac_perturbed", 0.3) or 0.3)
        config_kwargs["geogap_diag_every"] = int(getattr(args, "geogap_diag_every", 200) or 200)

        # ── Secondary structure basin loss ────────────────────────────────────
        config_kwargs["lambda_basin"] = float(getattr(args, "lambda_basin", 0.0) or 0.0)
        config_kwargs["basin_margin"] = float(getattr(args, "basin_margin", 0.5) or 0.5)
        config_kwargs["basin_mode"] = str(getattr(args, "basin_mode", "continuous") or "continuous")
        config_kwargs["basin_T_base"] = float(getattr(args, "basin_T_base", 2.0) or 2.0)

        # ── Native gap loss ───────────────────────────────────────────────────
        config_kwargs["lambda_native"] = float(getattr(args, "lambda_native", 0.0) or 0.0)
        config_kwargs["native_margin"] = float(getattr(args, "native_margin", 0.5) or 0.5)
        config_kwargs["native_sigma_min"] = float(getattr(args, "native_sigma_min", 0.05) or 0.05)
        config_kwargs["native_sigma_max"] = float(getattr(args, "native_sigma_max", 0.50) or 0.50)
        config_kwargs["native_mode"] = str(getattr(args, "native_mode", "continuous") or "continuous")
        config_kwargs["native_T_base"] = float(getattr(args, "native_T_base", 5.0) or 5.0)

        # ── ELT losses (Q-funnel + Z-score + Gap + Frustration) ────────
        config_kwargs["lambda_funnel"] = float(getattr(args, "lambda_funnel", 0.0) or 0.0)
        config_kwargs["funnel_T"] = float(getattr(args, "funnel_T", 2.0) or 2.0)
        config_kwargs["funnel_n_decoys"] = int(getattr(args, "funnel_n_decoys", 10) or 10)
        config_kwargs["funnel_slope_clamp"] = float(getattr(args, "funnel_slope_clamp", 10.0) or 10.0)
        config_kwargs["funnel_sigma_min"] = float(getattr(args, "funnel_sigma_min", 0.05) or 0.05)
        config_kwargs["funnel_sigma_max"] = float(getattr(args, "funnel_sigma_max", 2.0) or 2.0)
        config_kwargs["funnel_contact_cutoff"] = float(getattr(args, "funnel_contact_cutoff", 9.5) or 9.5)
        config_kwargs["lambda_zscore"] = float(getattr(args, "lambda_zscore", 0.0) or 0.0)
        config_kwargs["target_zscore"] = float(getattr(args, "target_zscore", 3.0) or 3.0)
        config_kwargs["lambda_gap"] = float(getattr(args, "lambda_gap", 0.0) or 0.0)
        config_kwargs["gap_margin"] = float(getattr(args, "gap_margin", 0.5) or 0.5)
        # Run5: saturating exponential margins
        config_kwargs["funnel_m"] = float(getattr(args, "funnel_m", 5.0))
        config_kwargs["funnel_alpha"] = float(getattr(args, "funnel_alpha", 5.0))
        config_kwargs["gap_m"] = float(getattr(args, "gap_m", 5.0))
        config_kwargs["gap_alpha"] = float(getattr(args, "gap_alpha", 5.0))
        config_kwargs["lambda_frustration"] = float(getattr(args, "lambda_frustration", 0.0) or 0.0)
        config_kwargs["frustration_T"] = float(getattr(args, "frustration_T", 2.0) or 2.0)
        config_kwargs["frustration_n_perms"] = int(getattr(args, "frustration_n_perms", 4) or 4)
        config_kwargs["elt_every"] = int(getattr(args, "elt_every", 5) or 5)

        # ── Native depth loss — deepen basin ──────────────────────────────
        _nd = getattr(args, "lambda_native_depth", None)
        if _nd is not None:
            config_kwargs["lambda_native_depth"] = float(_nd)
        _tnd = getattr(args, "target_native_depth", None)
        if _tnd is not None:
            config_kwargs["target_native_depth"] = float(_tnd)

        # ── Lambda floor for hb_beta ──────────────────────────────────────
        config_kwargs["lambda_hb_beta_floor"] = float(getattr(args, "lambda_hb_beta_floor", 0.0) or 0.0)

        # ── Subterm disable list ──────────────────────────────────────
        config_kwargs["disable_subterms"] = list(getattr(args, "disable_subterms", []) or [])

        # ── Langevin inverse temperature (validation) ─────────────────────────
        config_kwargs["langevin_beta"] = float(getattr(args, "langevin_beta", 1.0) or 1.0)

    config = PhaseConfig(
        name=args.phase,
        terms=active_terms,
        freeze=args.freeze,
        loss_fn=loss_fn,
        n_steps=int(args.steps),
        lr=float(args.lr),
        lr_schedule=args.lr_schedule,
        lr_final=float(lr_final) if lr_final is not None else None,
        save_every=int(args.ckpt_every),
        validate_every=int(validate_every),
        early_stopping_patience=args.early_stopping,
        gate_schedule=gate_schedule,
        **config_kwargs,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = PhasedTrainer(
        model=model,
        device=device,
        ckpt_dir=args.ckpt_dir,
        experiment_prefix=args.ckpt_prefix,
    )

    # ── Restore trainer state ────────────────────────────────────────────────
    # --resume: full resume — restore global_step, phase_step, best scores, etc.
    #   Training continues exactly where it left off.
    # --resume-model-only: warm start — model weights loaded, but training state
    #   resets (global_step=0, fresh optimizer, gates→1.0). Used when starting a
    #   new experiment from a pre-trained checkpoint.
    _is_model_only = getattr(args, "resume_model_only", False)
    if trainer_state and not _is_model_only:
        trainer.global_step = trainer_state.get("global_step", 0)
        trainer.phase_step = trainer_state.get("phase_step", 0)
        trainer.best_composite_score = trainer_state.get("best_composite_score")
        trainer.best_composite_score_initialized = trainer_state.get("best_composite_score_initialized", False)
        trainer.best_val_step = trainer_state.get("best_val_step", 0)
        trainer.early_stopping_counter = trainer_state.get("early_stopping_counter", 0)
        trainer.validation_history = trainer_state.get("validation_history", [])
        logger.info(
            "Full resume: restored trainer state (global=%d, phase=%d)", trainer.global_step, trainer.phase_step
        )
    elif _is_model_only and checkpoint_state is not None:
        logger.info("Model-only resume: training state reset (global_step=0, fresh start)")

    trainer.val_max_samples = int(getattr(args, "val_max_samples", 256))
    trainer.val_langevin_steps = int(getattr(args, "val_langevin_steps", 500))
    _val_step_size = getattr(args, "val_step_size", None)
    trainer.val_step_size = float(_val_step_size) if _val_step_size is not None else None
    trainer.val_langevin_beta = float(getattr(args, "langevin_beta", 1.0))
    logger.info(
        "Validation config: max_samples=%d  langevin_steps=%d  step_size=%s  beta=%.1f",
        trainer.val_max_samples,
        trainer.val_langevin_steps,
        f"{trainer.val_step_size:.1e}" if trainer.val_step_size else "default(1e-4)",
        trainer.val_langevin_beta,
    )

    # ── Full phase gate handling ──────────────────────────────────────────────
    _is_full_resume = args.phase == "full" and args.resume and not getattr(args, "resume_model_only", False)
    if args.phase == "full" and hasattr(model, "set_gates"):
        if ramp_gates and ramp_start is not None:
            logger.info("Seeding gates to ramp_start values: %s", ramp_start)
            model.set_gates(**ramp_start)
        elif _is_full_resume:
            # Checkpoint gates are authoritative — do not overwrite them
            current = model.get_gates() if hasattr(model, "get_gates") else {}
            logger.info("Full phase resume: preserving checkpoint gates: %s", current)
        else:
            model.set_gates(local=1.0, repulsion=1.0, secondary=1.0, packing=1.0)
            logger.info("Gates reset to 1.0 (internal lambdas handle scaling)")

    # ── Non-full phase gate correction ────────────────────────────────────────
    # Outer gates should always be 1.0 except where explicitly ramped.
    # secondary/packing/local phases must not inherit a stale gate_secondary=0.15
    # that was written into the checkpoint by a prior gate_schedule application.
    elif args.phase in ("secondary", "local", "repulsion") and hasattr(model, "set_gates"):
        model.set_gates(local=1.0, repulsion=1.0, secondary=1.0, packing=1.0)
        logger.info("Gates reset to 1.0 for phase '%s' (outer gates; internal lambdas handle scaling)", args.phase)
    elif args.phase == "packing" and hasattr(model, "set_gates"):
        # Packing phase sets its own packing gate via ramp — only fix the others
        current_pack_gate = float(model.gate_packing.item())
        model.set_gates(local=1.0, repulsion=1.0, secondary=1.0, packing=current_pack_gate)
        logger.info("Gates corrected for phase 'packing': local/rep/ss reset to 1.0, packing=%.4f", current_pack_gate)
        if int(getattr(args, "freeze_gates_steps", 0) or 0) > 0:
            logger.info("Gates will be frozen for first %d steps", int(args.freeze_gates_steps))

    # Re-apply gate overrides AFTER phase reset
    _gate_map = {
        "gate_local": getattr(args, "set_gate_local", None),
        "gate_secondary": getattr(args, "set_gate_secondary", None),
        "gate_repulsion": getattr(args, "set_gate_repulsion", None),
        "gate_packing": getattr(args, "set_gate_packing", None),
    }
    if any(v is not None for v in _gate_map.values()) and hasattr(model, "set_gates"):
        _kwargs = {name.replace("gate_", ""): val for name, val in _gate_map.items() if val is not None}
        model.set_gates(**_kwargs)
        for name, val in _kwargs.items():
            logger.info("Gate override (post-reset): gate_%s → %.4f", name, val)

    if args.phase == "full" and hasattr(model, "get_gates"):
        try:
            logger.info("Initial gates for phase start: %s", model.get_gates())
        except Exception:
            logger.warning("Could not read model gates via get_gates()")

    native_structures = load_native_structures(val_ids, args.cache_dir, device)

    state = trainer.run_phase(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        native_structures=native_structures,
        resume=None,
    )

    # ── Final logging ─────────────────────────────────────────────────────────
    validation_history = getattr(state, "validation_history", [])
    if state.best_composite_score is None:
        logger.info(
            "Best composite score: N/A (%s) at step %s",
            "no validation ran" if not validation_history else "not initialized",
            str(state.best_val_step),
        )
    elif not math.isfinite(state.best_composite_score):
        logger.warning(
            "Best composite score is non-finite: %s at step %s",
            str(state.best_composite_score),
            str(state.best_val_step),
        )
    else:
        logger.info("Best composite score: %.6f at step %s", state.best_composite_score, str(state.best_val_step))

    logger.info("Phase %s complete", args.phase)
    return 0
