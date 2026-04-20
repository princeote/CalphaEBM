# src/calphaebm/evaluation/core_evaluation.py
"""Core evaluation utilities — shared by basin_evaluation and training_evaluation.

No evaluation logic here. This module owns:
  - Canonical v6 model architecture defaults
  - Checkpoint-agnostic model loading (three strategies)
  - Structure loading from PDB dataset or pre-parsed ID list
  - DataLoader wrapper for pre-loaded structure lists

Import from here in basin_evaluation.py and training_evaluation.py.
Never import the reverse direction.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from calphaebm.utils.logging import get_logger

logger = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# v6 model architecture defaults
# ─────────────────────────────────────────────────────────────────────────────


class _v6_model_args:  # noqa: N801
    """Canonical v6 model defaults — pass to build_model as the args object.

    Matches the training configuration used in run6_full_stage.sh exactly.
    Used when a checkpoint contains only a state_dict and the model must
    be rebuilt before weights are loaded.
    """

    backbone_data_dir = "analysis/backbone_geometry/data"
    secondary_data_dir = "analysis/secondary_analysis/data"
    repulsion_data_dir = "analysis/repulsion_analysis/data"
    packing_data_dir = "analysis/repulsion_analysis/data"
    coord_n_star_file = "analysis/coordination_analysis/coord_n_star.json"
    # Nonbonded
    repulsion_K = 64
    repulsion_exclude = 3
    repulsion_r_on = 8.0
    repulsion_r_cut = 10.0
    packing_r_on = 8.0
    packing_r_cut = 10.0
    # Packing
    packing_short_gate_on = 4.5
    packing_short_gate_off = 5.0
    packing_rbf_centers = [5.5, 7.0, 9.0]
    packing_rbf_width = 1.0
    packing_max_dist = 12.0
    packing_init_from = "log_oe"
    packing_normalize_by_length = True
    packing_debug_scale = False
    packing_debug_every = 200
    packing_geom_calibration = None
    # Rg / coord (v6)
    coord_lambda = 1.0
    packing_rg_lambda = 1.0
    packing_rg_r0 = 2.0
    packing_rg_nu = 0.38
    packing_rg_dead_zone = 0.30
    packing_rg_m = 1.0
    packing_rg_alpha = 3.0
    packing_coord_m = 1.0
    packing_coord_alpha = 2.0
    packing_rho_lambda = 1.0
    packing_rho_m = 1.0
    packing_rho_alpha = 2.0
    hp_penalty_lambda = 1.0
    rho_penalty_lambda = 1.0
    # Local
    init_theta_theta_weight = 1.0
    init_delta_phi_weight = 1.0
    local_window_size = 8
    debug_mode = False


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_model(ckpt_path: Path, device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    """Load a TotalEnergy model from checkpoint onto device.

    Three strategies tried in order:
      1. Direct torch.load — checkpoint is a complete serialised nn.Module
      2. State dict   — checkpoint dict has 'model_state_dict'; rebuild + load
      3. load_checkpoint — use training infrastructure to extract state dict
    """
    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Strategy 1: already a complete module
    if isinstance(raw, torch.nn.Module):
        raw.eval()
        logger.debug("Loaded checkpoint as complete model object: %s", ckpt_path.name)
        return raw

    from calphaebm.cli.commands.train.model_builder import build_model

    model = build_model({"local", "repulsion", "secondary", "packing"}, device, _v6_model_args())

    # Strategy 2: dict with model_state_dict key
    if isinstance(raw, dict) and "model_state_dict" in raw:
        from calphaebm.cli.commands.train.checkpoint import load_weights_into_model

        load_weights_into_model(model, raw["model_state_dict"])
        logger.debug("Loaded checkpoint via state_dict: %s", ckpt_path.name)

    # Strategy 3: use training checkpoint loader
    else:
        from calphaebm.cli.commands.train.checkpoint import load_checkpoint as _lc
        from calphaebm.cli.commands.train.checkpoint import load_weights_into_model

        state, _ = _lc(str(ckpt_path), device)
        load_weights_into_model(model, state)
        logger.debug("Loaded checkpoint via load_checkpoint: %s", ckpt_path.name)

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Structure loading
# ─────────────────────────────────────────────────────────────────────────────


def load_structures(
    pdb_source,  # file path (str/Path) OR list of PDB IDs
    cache_dir: str,
    n_samples: int,
    max_len: int = 512,
    min_len: int = 10,
    processed_cache_dir: Optional[str] = None,
) -> List[Tuple]:
    """Load up to n_samples structures as (R, seq, pdb_id, chain_id, L) tuples.

    Accepts either:
      - A file path (str/Path) pointing to a PDB list file → uses pdb_list= ctor
      - A list of PDB ID strings                           → uses pdb_ids=  ctor
    """
    from calphaebm.data.pdb_chain_dataset import PDBChainDataset

    # Resolve to a list of PDB IDs regardless of input type
    if isinstance(pdb_source, (str, Path)) and Path(pdb_source).is_file():
        raw = Path(pdb_source).read_text().splitlines()
        pdb_ids = [l.strip() for l in raw if l.strip() and not l.strip().startswith("#")]
    else:
        pdb_ids = list(pdb_source)

    common = dict(
        cache_dir=cache_dir,
        min_len=min_len,
        max_len=max_len,
        max_chains=n_samples * 2,
        cache_processed=processed_cache_dir is not None,
        processed_cache_dir=processed_cache_dir or "./processed_cache",
    )

    ds = PDBChainDataset(pdb_ids=pdb_ids, **common)

    collate = getattr(ds, "collate_fn", None) or getattr(ds.__class__, "collate", None)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)

    structures: List[Tuple] = []
    for batch in loader:
        R, seq = batch[0], batch[1]
        pdb_ids = batch[2] if len(batch) > 2 else [f"unk_{i}" for i in range(R.shape[0])]
        chain_ids = batch[3] if len(batch) > 3 else ["A"] * R.shape[0]
        lengths = batch[4] if len(batch) >= 5 else torch.full((R.shape[0],), R.shape[1])
        for b in range(R.shape[0]):
            Lb = int(lengths[b].item())
            if Lb < min_len or Lb > max_len:
                continue
            structures.append(
                (
                    R[b, :Lb].cpu().clone(),
                    seq[b, :Lb].cpu().clone(),
                    pdb_ids[b] if b < len(pdb_ids) else "unk",
                    chain_ids[b] if b < len(chain_ids) else "A",
                    Lb,
                )
            )
            if len(structures) >= n_samples:
                break
        if len(structures) >= n_samples:
            break

    return structures


def structures_to_loader(structures: List[Tuple]) -> DataLoader:
    """Wrap a pre-loaded structure list as a minimal padded DataLoader."""

    class _DS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            R, seq, pdb_id, chain_id, L = self.items[i]
            return R, seq, pdb_id, chain_id, torch.tensor(L)

    def _collate(batch):
        Rs, seqs, pids, cids, lens = zip(*batch)
        Lmax = max(r.shape[0] for r in Rs)
        R_pad = torch.zeros(len(Rs), Lmax, 3)
        s_pad = torch.zeros(len(Rs), Lmax, dtype=torch.long)
        for i, (r, s) in enumerate(zip(Rs, seqs)):
            Lb = r.shape[0]
            R_pad[i, :Lb] = r
            s_pad[i, :Lb] = s
        return R_pad, s_pad, list(pids), list(cids), torch.stack(lens)

    return DataLoader(_DS(structures), batch_size=16, shuffle=False, collate_fn=_collate)
