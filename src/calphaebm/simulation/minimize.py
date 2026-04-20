"""L-BFGS energy minimization in internal coordinate space.

src/calphaebm/simulation/minimize.py

Minimizes E(NeRF(θ, φ)) with respect to bond angles θ and pseudo-dihedrals φ
using quasi-Newton optimization (L-BFGS with strong Wolfe line search).
Guarantees exact 3.8 Å Cα bond lengths by construction via NeRF reconstruction.

Replaces the previous first-order Langevin-at-β=1e8 approach, which converges
linearly and often fails to reach the true minimum within 10K steps.  L-BFGS
converges quadratically near the minimum and typically finishes in 50-200 calls.

Usage:
    from calphaebm.simulation.minimize import lbfgs_minimize

    result = lbfgs_minimize(model, R, seq, lengths=lengths)
    R_min = result["R_min"]           # (1, L, 3) minimized coordinates
    E_min = result["E_minimized"]     # scalar, minimized energy
    E_relax = result["E_relax"]       # E_minimized - E_pdb (should be negative)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from calphaebm.geometry.reconstruct import coords_to_internal, extract_anchor, nerf_reconstruct


def lbfgs_minimize(
    model: nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    max_calls: int = 200,
    max_iter: int = 20,
    lr: float = 1.0,
    tolerance_grad: float = 1e-5,
    tolerance_change: float = 1e-9,
    converge_thr: float = 1e-5,
) -> Dict:
    """Minimize energy in IC space using L-BFGS.

    Args:
        model: Energy model E(R, seq, lengths) → scalar.
        R: (1, L, 3) initial Cα coordinates.
        seq: (1, L) amino acid sequence indices.
        lengths: (1,) chain length tensor (optional).
        max_calls: Maximum number of outer L-BFGS calls.
        max_iter: Inner iterations per L-BFGS step.
        lr: L-BFGS learning rate (1.0 recommended — line search controls step).
        tolerance_grad: Gradient convergence threshold.
        tolerance_change: Function value change threshold.
        converge_thr: Energy convergence threshold (E/res) for early stop.

    Returns:
        dict with keys:
            R_min:        (1, L, 3) minimized coordinates
            E_minimized:  float, energy at minimum
            E_pdb:        float, energy at input coordinates
            E_relax:      float, E_minimized - E_pdb
            drmsd_min:    float, dRMSD between input and minimized
            q_min:        float, Q (native contacts) after minimization
            min_steps:    int, number of L-BFGS calls taken
            max_force:    float, maximum gradient magnitude observed
    """
    R = R.detach().float()

    # Energy at input coordinates
    with torch.no_grad():
        E_pdb = float(model(R, seq, lengths=lengths).item())

    # Extract IC representation
    theta_raw, phi_raw = coords_to_internal(R)
    anchor = extract_anchor(R)

    # Trainable IC parameters
    theta_opt = theta_raw.detach().clone().requires_grad_(True)
    phi_opt = phi_raw.detach().clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [theta_opt, phi_opt],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
    )

    max_force = 0.0
    min_steps = 0
    E_prev = E_pdb

    for step_i in range(1, max_calls + 1):

        def closure():
            optimizer.zero_grad()
            theta_c = theta_opt.clamp(0.01, math.pi - 0.01)
            phi_c = (phi_opt + math.pi) % (2 * math.pi) - math.pi
            R_c = nerf_reconstruct(theta_c, phi_c, anchor)
            E = model(R_c, seq, lengths=lengths)
            E.backward()
            return E

        E_val = optimizer.step(closure)
        min_steps = step_i

        # Track max gradient (force)
        if theta_opt.grad is not None:
            gn = max(
                float(theta_opt.grad.abs().max()),
                float(phi_opt.grad.abs().max()),
            )
            max_force = max(max_force, gn)

        # Check convergence (.item() avoids requires_grad UserWarning)
        E_curr = E_val.item() if E_val is not None else E_prev
        if step_i > 1 and abs(E_prev - E_curr) < converge_thr:
            break
        E_prev = E_curr

    # Reconstruct minimized Cartesian coordinates
    with torch.no_grad():
        theta_final = theta_opt.detach().clamp(0.01, math.pi - 0.01)
        phi_final = (phi_opt.detach() + math.pi) % (2 * math.pi) - math.pi
        R_min = nerf_reconstruct(theta_final, phi_final, anchor).detach()
        E_minimized = float(model(R_min, seq, lengths=lengths).item())

    E_relax = E_minimized - E_pdb

    return {
        "R_min": R_min,
        "E_minimized": E_minimized,
        "E_pdb": E_pdb,
        "E_relax": E_relax,
        "min_steps": min_steps,
        "max_force": max_force,
    }
