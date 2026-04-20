from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import erfc  # FIXED: import erfc from scipy.special

from calphaebm.utils.logging import get_logger

from .config import EMPIRICAL_COUNT_THRESHOLD, EMPIRICAL_P_FOR_LOW_COUNTS, ENRICH_SHUFFLES, FDR_Q

logger = get_logger()


@dataclass(frozen=True)
class EnrichmentResult:
    observed: Dict[str, np.ndarray]  # (20,20) counts per bin
    expected: Dict[str, np.ndarray]  # (20,20) expected counts per bin
    oe: Dict[str, np.ndarray]  # (20,20) observed/expected
    log_oe: Dict[str, np.ndarray]  # (20,20) log(observed/expected)
    z: Dict[str, np.ndarray]  # (20,20) z-score from shuffle dist
    p: Dict[str, np.ndarray]  # (20,20) p-values (possibly empirical for low counts)
    q: Dict[str, np.ndarray]  # (20,20) BH-FDR q-values
    shuffle_mean: Dict[str, np.ndarray]  # (20,20)
    shuffle_std: Dict[str, np.ndarray]  # (20,20)


def _bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; returns q-values array with same shape."""
    p = np.asarray(pvals, dtype=np.float64).ravel()
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    qvals = ranked * m / (np.arange(1, m + 1))
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals = np.clip(qvals, 0.0, 1.0)
    out = np.empty_like(p)
    out[order] = qvals
    return out.reshape(pvals.shape)


def compute_contact_enrichment(
    contact_counts: Dict[str, np.ndarray],
    n_shuffles: int = ENRICH_SHUFFLES,
    empirical_p_for_low_counts: bool = EMPIRICAL_P_FOR_LOW_COUNTS,
    empirical_count_threshold: int = EMPIRICAL_COUNT_THRESHOLD,
    fdr_q: float = FDR_Q,
    rng: Optional[np.random.Generator] = None,
) -> EnrichmentResult:
    """
    Compute enrichment matrices for each contact bin.

    We treat contact_counts[bin] as *observed* pair counts aggregated over the dataset,
    then create a null by shuffling residue identities over the same contact graph.
    (This preserves the |i-j|>MIN_SEQ_SEP constraint because the *pair list* is fixed
    during shuffling; only labels are permuted.)

    Returns:
        EnrichmentResult with OE/logOE plus z/p/q per bin.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    bins = list(contact_counts.keys())
    obs = {b: np.asarray(contact_counts[b], dtype=np.float64) for b in bins}

    # Total counts per bin
    totals = {b: float(np.sum(obs[b])) for b in bins}

    # Background frequencies from observed marginals (per bin)
    expected = {}
    oe = {}
    log_oe = {}
    z = {}
    p = {}
    q = {}
    sh_mean = {}
    sh_std = {}

    for b in bins:
        O = obs[b]
        if O.shape != (20, 20):
            raise ValueError(f"contact_counts[{b}] must be (20,20); got {O.shape}")

        # Expected under independence: outer product of marginals
        row = O.sum(axis=1)
        col = O.sum(axis=0)
        denom = float(row.sum())
        if denom <= 0:
            logger.warning(f"No contacts in bin '{b}'. Returning zeros.")
            expected[b] = np.zeros_like(O)
            oe[b] = np.ones_like(O)
            log_oe[b] = np.zeros_like(O)
            z[b] = np.zeros_like(O)
            p[b] = np.ones_like(O)
            q[b] = np.ones_like(O)
            sh_mean[b] = np.zeros_like(O)
            sh_std[b] = np.ones_like(O)
            continue

        P_i = row / denom
        P_j = col / denom
        E = np.outer(P_i, P_j) * denom
        expected[b] = E

        # Shuffle null distribution by permuting labels in a multinomial approximation
        # We approximate: shuffles ~ multinomial(denom, outer(P_i, P_j)) but implement by sampling counts.
        # This is fast and captures low-count discreteness reasonably well.
        probs = (np.outer(P_i, P_j)).ravel()
        probs = probs / max(float(np.sum(probs)), 1e-12)

        shuffle_all = np.zeros((int(n_shuffles), 20, 20), dtype=np.float64)
        for s in range(int(n_shuffles)):
            draw = rng.multinomial(int(round(denom)), probs)
            shuffle_all[s] = draw.reshape(20, 20)

        mu = shuffle_all.mean(axis=0)
        sd = shuffle_all.std(axis=0, ddof=1)
        sh_mean[b] = mu
        sh_std[b] = np.maximum(sd, 1e-9)

        # OE and logOE
        OE = O / np.maximum(E, 1e-12)
        oe[b] = OE
        log_oe[b] = np.log(np.maximum(OE, 1e-12))

        # Z-score
        Z = (O - mu) / sh_std[b]
        z[b] = Z

        # P-values: empirical for low counts (optional), otherwise normal approx via Z
        P = np.ones_like(O, dtype=np.float64)

        if empirical_p_for_low_counts:
            # empirical two-sided p from shuffle distribution, for entries with obs < threshold
            low_mask = O < float(empirical_count_threshold)
            if np.any(low_mask):
                # for each (i,j) in low_mask, compute empirical tail areas
                # (vectorized over shuffles but indexed per cell)
                for i, j in zip(*np.where(low_mask)):
                    sh = shuffle_all[:, i, j]
                    obs_ij = O[i, j]
                    p_ge = float(np.mean(sh >= obs_ij))
                    p_le = float(np.mean(sh <= obs_ij))
                    p_two = 2.0 * min(p_ge, p_le)
                    P[i, j] = min(max(p_two, 0.0), 1.0)  # clip

        # normal approx for remaining entries (or all if empirical disabled)
        # two-sided p ~= 2*(1 - Phi(|z|))
        # implement with erfc for numeric stability
        rem_mask = P == 1.0  # still default => not set by empirical
        if np.any(rem_mask):
            zz = np.abs(Z[rem_mask]) / np.sqrt(2.0)
            # FIXED: use erfc from scipy.special
            P[rem_mask] = np.clip(erfc(zz), 0.0, 1.0)

        p[b] = P
        q[b] = _bh_fdr(P, q=float(fdr_q))

    return EnrichmentResult(
        observed=obs,
        expected=expected,
        oe=oe,
        log_oe=log_oe,
        z=z,
        p=p,
        q=q,
        shuffle_mean=sh_mean,
        shuffle_std=sh_std,
    )
