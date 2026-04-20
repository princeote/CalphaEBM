"""Energy balance loss — subterm-level.

Penalises any pair of subterms whose ratio of absolute mean energies falls
outside [1/r, r].

Subterms (7 active in balance, local_bond excluded — bonds are guaranteed by NeRF):
  1. local_thetaphi     — 4-mer sliding-window MLP (sequence-conditioned)
  2. secondary_ram      — mixture-of-basins Ramachandran
  3. secondary_hb_alpha — E_hb(α) helical H-bond
  4. secondary_hb_beta  — E_hb(β) sheet H-bond
  5. repulsion          — pairwise repulsion
  6. packing_hp_reward  — hydrophobic coordination reward (v5)
  7. packing_rho_reward — contact density reward (v5)

Excluded from balance (analytical safety nets, ~0 at native):
  - packing_hp_penalty  — per-AA coordination band penalty (v5)
  - packing_rho_penalty — contact density deviation penalty (v5)
  - packing_rg          — Flory Rg restraint

Note: H-bond subterms are physics-constrained and inherently sparse — they only
fire when basin probabilities AND distances align. Use a wide r (≥8) and low
lambda_balance (≤0.1) to avoid penalizing their natural smallness.

Design
------
- Uses the SAME R/seq batch as DSM — no extra forward passes beyond what
  balance_loss itself needs.
- Each subterm is evaluated with its gate (if any) applied.
- Returns (loss, term_absmeans) so diagnostics can use the same values
  without a second forward pass.

Why subterm-level?
------------------
The 4-term balance loss (local, secondary, repulsion, packing) was too coarse.
Within secondary, the θφ/φφ MLPs could grow to dominate the basin potentials
(which encode real PDB statistics) while the 4-term loss saw a healthy
"secondary" total. At the subterm level this is caught directly: if secondary_ram
shrinks below 1/r of secondary_thetaphi, a violation fires.

Similarly, local_thetatheta growing to 2.7× while local_deltaphi stays at 0.05
is caught — the 4-term loss never saw this since it only checked "local" total.

Formula
-------
For each active subterm t:
    a_t = gate_t * mean(|E_t(R, seq)|)   over batch

For each ordered pair (i, j):
    ratio_ij = a_i / (a_j + eps)
    violation = relu(ratio_ij - r)² + relu(1/r - ratio_ij)²

    L_balance = sum_{i != j} violation

Usage
-----
    from calphaebm.training.losses.balance_loss import energy_balance_loss

    loss_bal, term_absmeans = energy_balance_loss(model, R, seq, r=3.0)
    loss = loss + lambda_balance * loss_bal
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gate(model: torch.nn.Module, name: str) -> float:
    """Return scalar gate value for a top-level term (default 1.0)."""
    attr = f"gate_{name}"
    g = getattr(model, attr, None)
    if g is None:
        return 1.0
    try:
        return float(g.item())
    except Exception:
        return 1.0


def _collect_subterms(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    lengths: torch.Tensor | None = None,
    exclude_subterms: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate all active subterms, returning (B,) tensors with gates applied.

    Subterms that are unavailable (module missing) are silently skipped.
    Subterms in exclude_subterms are also skipped (e.g. disabled via --disable-subterms).
    local_bond is always excluded (guaranteed by NeRF — zero by construction).
    """
    out: dict[str, torch.Tensor] = {}
    _skip = exclude_subterms or set()

    # ── local ──────────────────────────────────────────────────────────
    local = getattr(model, "local", None)
    if local is not None:
        g = _gate(model, "local")
        # 4-mer architecture: single subterm
        if hasattr(local, "theta_phi_energy") and "theta_phi" not in _skip:
            out["local_thetaphi"] = g * local.theta_phi_energy(R, seq, lengths=lengths)
        # Old 3-subterm architecture
        if hasattr(local, "theta_theta_energy") and "theta_theta" not in _skip:
            try:
                out["local_thetatheta"] = g * local.theta_theta_energy(R, seq, lengths=lengths)
            except TypeError:
                out["local_thetatheta"] = g * local.theta_theta_energy(R)
        if hasattr(local, "delta_phi_energy") and "delta_phi" not in _skip:
            out["local_deltaphi"] = g * local.delta_phi_energy(R, lengths=lengths)
        if hasattr(local, "phi_phi_energy") and "phi_phi" not in _skip:
            out["local_phiphi"] = g * local.phi_phi_energy(R, seq, lengths=lengths)

    # ── secondary ──────────────────────────────────────────────────────
    sec = getattr(model, "secondary", None)
    if sec is not None:
        g = _gate(model, "secondary")
        try:
            E_ram, E_hb_a, E_hb_b = sec.subterm_energies(R, seq, lengths=lengths)
            if "ram" not in _skip:
                out["secondary_ram"] = g * E_ram
            if "hb_alpha" not in _skip:
                out["secondary_hb_alpha"] = g * E_hb_a
            if "hb_beta" not in _skip:
                out["secondary_hb_beta"] = g * E_hb_b
        except Exception:
            out["secondary"] = g * sec(R, seq)

    # ── repulsion ──────────────────────────────────────────────────────
    rep = getattr(model, "repulsion", None)
    if rep is not None and "repulsion" not in _skip:
        g = _gate(model, "repulsion")
        out["repulsion"] = g * rep(R, seq, lengths=lengths)

    # ── packing ────────────────────────────────────────────────────────
    pack = getattr(model, "packing", None)
    if pack is not None:
        g = _gate(model, "packing")
        try:
            subterms_pack = pack.subterm_energies(R, seq, lengths=lengths)
            if len(subterms_pack) == 5:
                # v5: (E_hp_reward, E_hp_penalty, E_rho_reward, E_rho_penalty, E_rg_penalty)
                E_hp_rew, E_hp_pen, E_rho_rew, E_rho_pen, E_rg_pen = subterms_pack
                # Balance includes reward terms only (learned, trainable)
                if "packing_hp_reward" not in _skip:
                    out["packing_hp_reward"] = g * E_hp_rew
                if "packing_rho_reward" not in _skip:
                    out["packing_rho_reward"] = g * E_rho_rew
                # Constraints excluded: hp_penalty, rho_penalty, rg_penalty
                # (analytical safety nets, ~0 at native)
            else:
                # v4 fallback: (E_hp, E_coord, E_rg)
                E_hp, E_coord, E_rg = subterms_pack
                if "packing_contact" not in _skip:
                    out["packing_contact"] = g * E_hp
                # E_coord and E_rg excluded from balance
        except Exception:
            out["packing"] = g * pack(R, seq, lengths=lengths)

    return out


def _pairwise_ratio_loss(
    vals: list[torch.Tensor],
    r: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pairwise ratio violation loss over a list of |E| tensors."""
    device = vals[0].device
    dtype = vals[0].dtype
    if len(vals) < 2:
        return torch.zeros((), device=device, dtype=dtype)
    r_t = torch.tensor(r, device=device, dtype=dtype)
    inv_r_t = torch.tensor(1.0 / r, device=device, dtype=dtype)
    loss = torch.zeros((), device=device, dtype=dtype)
    for i in range(len(vals)):
        for j in range(len(vals)):
            if i == j:
                continue
            ratio = vals[i] / (vals[j] + eps)
            loss = loss + F.relu(ratio - r_t) ** 2
            loss = loss + F.relu(inv_r_t - ratio) ** 2
    return loss


def energy_balance_loss(
    model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    r: float = 7.0,
    r_term: float = 4.0,
    eps: float = 1e-6,
    lengths: torch.Tensor | None = None,
    exclude_subterms: set[str] | None = None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    """Dual-level pairwise ratio balance loss.

    Level 1 (subterm, r=`r`):   6 learned subterms, ratios in [1/r, r].
    Level 2 (term,    r=`r_term`): up to 4 terms, ratios in [1/r_term, r_term].

    Analytical safety nets (E_coord, E_Rg) are excluded — they are ~0 at native
    and should not be equalized against ~0.3/res learned terms.

    Subterms listed in exclude_subterms are skipped entirely (not evaluated,
    not included in ratios). Use for disabled subterms that output zero.

    Parameters
    ----------
    model  : TotalEnergy model.
    R      : (B, L, 3) coordinates.
    seq    : (B, L) sequence indices.
    r      : subterm-level ratio bound (default 7.0 = number of balanced subterms).
    r_term : term-level ratio bound (default 4.0).
    eps    : denominator stabiliser.
    lengths : (B,) actual chain lengths.
    exclude_subterms : set of subterm names to skip (e.g. {"theta_theta", "delta_phi"}).

    Returns
    -------
    (loss, subterm_absmeans, term_absmeans)
      loss              : scalar loss = subterm_loss + term_loss (differentiable).
      subterm_absmeans  : dict[str, float] — active subterm mean(|gate×E|).
      term_absmeans     : dict[str, float] — active term mean(|gate×E|), aggregated.
    """
    device = R.device
    dtype = R.dtype

    subterms = _collect_subterms(model, R, seq, lengths=lengths, exclude_subterms=exclude_subterms)

    # mean(|E|) per subterm
    absmeans: dict[str, torch.Tensor] = {name: E.abs().mean() for name, E in subterms.items()}

    subterm_absmeans: dict[str, float] = {name: float(v.detach().item()) for name, v in absmeans.items()}

    # ── subterm-level loss (r) ─────────────────────────────────────────
    sub_vals = list(absmeans.values())
    loss_sub = _pairwise_ratio_loss(sub_vals, r, eps)

    # ── term-level aggregation ─────────────────────────────────────────
    term_agg: dict[str, torch.Tensor] = {}
    for name, val in absmeans.items():
        if name.startswith("local_"):
            term_agg["local"] = term_agg.get("local", torch.zeros((), device=device, dtype=dtype)) + val
        elif name.startswith("secondary_"):
            term_agg["secondary"] = term_agg.get("secondary", torch.zeros((), device=device, dtype=dtype)) + val
        elif name.startswith("packing_"):
            term_agg["packing"] = term_agg.get("packing", torch.zeros((), device=device, dtype=dtype)) + val
        else:
            term_agg[name] = val

    term_absmeans: dict[str, float] = {name: float(v.detach().item()) for name, v in term_agg.items()}

    # ── term-level loss (r_term) ───────────────────────────────────────
    term_vals = list(term_agg.values())
    loss_term = _pairwise_ratio_loss(term_vals, r_term, eps)

    loss = loss_sub + loss_term

    return loss, subterm_absmeans, term_absmeans
