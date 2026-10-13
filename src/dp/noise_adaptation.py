"""Uncertainty-driven privacy-noise adaptation (Component 1).

CONCEPT (as proposed)
---------------------
Give each sample its own noise multiplier

    sigma_i = sigma_base * (1 - lambda_adapt * u_i),     u_i in [0, 1]

so a borderline, high-uncertainty transaction (u_i -> 1) is perturbed less
than a confident one (u_i -> 0), preserving the discriminative signal that
uniform DP-SGD destroys first.

WHY IT IS NOT IMPLEMENTED LITERALLY
-----------------------------------
A noise variance that depends on the sample's own data breaks the standard
Gaussian-mechanism analysis: on adjacent datasets the *noise distributions*
differ, and "charge the accountant at the smallest sigma" is not a valid
bound (it was also tried: it produced epsilon in the thousands).

EQUIVALENT REALISATION THAT IS EXACTLY DP
-----------------------------------------
What matters to a sample is its signal-to-noise ratio in the released update.
Scaling the sample's *clipped gradient* by w_i and adding *fixed* noise
sigma_base*C gives it the same SNR as keeping its gradient and adding noise
sigma_base*C/w_i. So we use a per-sample clipping bound

    C_i = C * w_i,   w_i = (1 - lambda_adapt) + lambda_adapt * u_i  in [1-lambda, 1]

and fixed Gaussian noise N(0, (sigma_base*C)^2). Every sample's contribution
is bounded by C, the sum's sensitivity is C, and the standard RDP accountant
applies unchanged. A borderline sample keeps its full clipped gradient; a
confident one is attenuated to (1-lambda) of it. lambda_adapt = 0 reproduces
DP-SGD exactly. The effective per-sample noise multiplier sigma_base / w_i is
returned by ``conceptual_sigma`` for reporting and matches the proposal's
formula with sigma_min = (1-lambda) * sigma_base.

u_i is the LAC conformal nonconformity 1 - p_true, which is already in [0, 1]
so no data-dependent normalisation is required (see src/conformal/scores.py).
"""

import numpy as np
import torch


def uncertainty_to_clip_scale(u, lambda_adapt: float, min_scale: float = 0.1):
    """w_i = (1 - lambda) + lambda * u_i, clamped to [min_scale, 1]."""
    w = (1.0 - lambda_adapt) + lambda_adapt * u
    if isinstance(w, torch.Tensor):
        return torch.clamp(w, min=min_scale, max=1.0)
    return np.clip(w, min_scale, 1.0)


def conceptual_sigma(u, sigma_base: float, lambda_adapt: float, min_scale: float = 0.1):
    """Effective per-sample noise multiplier sigma_base / w_i (for reporting)."""
    return sigma_base / uncertainty_to_clip_scale(u, lambda_adapt, min_scale)


def compute_adapted_sigma(normalised_scores, sigma_base, lambda_adapt, sigma_min_ratio=0.1):
    """Backwards-compatible name for conceptual_sigma (kept for older scripts)."""
    return conceptual_sigma(normalised_scores, sigma_base, lambda_adapt, sigma_min_ratio)
