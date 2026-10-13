"""Privacy accounting helpers.

BUDGET-FIRST CALIBRATION
------------------------
Rather than training for a fixed number of epochs and reporting whatever
epsilon falls out (which is private selection on private data), the target
epsilon is fixed *a priori* and the noise multiplier sigma is solved for the
planned number of optimiser steps. In the federation each client has its own
sampling rate q_k = B / n_k and therefore its own sigma_k for the same target.

COMPOSITION ACROSS ROUNDS
-------------------------
One ``RDPAccountant`` per client lives for the whole federation, so epsilon
composes over R rounds x E local epochs. ``epsilon_after_steps`` rebuilds the
same number from (sigma, q, steps) deterministically - used by the Flower
adapter, whose client objects are re-created every round.

CLIENT-UNCERTAINTY STATISTIC (Component 2)
-----------------------------------------
The mean LAC score on the calibration fold is a statistic of private data.
Its sensitivity is 1/n_cal (scores are in [0, 1]), so Laplace noise of scale
1/(n_cal * eps_u) makes it eps_u-DP per round; over R rounds this is
R * eps_u under basic composition, reported as a separate line of the ledger.
"""

import math
import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


def steps_per_epoch(n: int, batch_size: int) -> int:
    """Poisson batches Opacus draws per epoch: DPDataLoader sets q = 1/len(loader) = 1/ceil(n/B)."""
    return max(1, int(math.ceil(n / max(1, batch_size))))


def opacus_sample_rate(n: int, batch_size: int) -> float:
    return 1.0 / steps_per_epoch(n, batch_size)


def calibrate_sigma(target_epsilon: float, delta: float, n: int, batch_size: int,
                    total_epochs: int) -> float:
    """Noise multiplier that spends target_epsilon over total_epochs (RDP), using Opacus's q."""
    q = opacus_sample_rate(n, batch_size)
    steps = steps_per_epoch(n, batch_size) * total_epochs
    return float(get_noise_multiplier(target_epsilon=target_epsilon, target_delta=delta,
                                      sample_rate=q, steps=steps, accountant='rdp'))


def epsilon_after_steps(sigma: float, sample_rate: float, steps: int, delta: float) -> float:
    if steps <= 0 or sigma <= 0:
        return 0.0 if sigma > 0 else float('inf')
    acc = RDPAccountant()
    acc.history = [(float(sigma), float(sample_rate), int(steps))]
    return float(acc.get_epsilon(delta=delta))


class PrivacyBudgetTracker:
    """Thin wrapper around RDPAccountant with the project's reporting conventions."""

    def __init__(self, target_delta=1e-5):
        self.delta = target_delta
        self.accountant = RDPAccountant()

    def step(self, noise_multiplier, sample_rate, num_steps=1):
        self.accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        if num_steps > 1:
            # collapse into one history entry (RDP composition is additive in steps)
            sig, q, k = self.accountant.history[-1]
            self.accountant.history[-1] = (sig, q, k + num_steps - 1)

    def get_epsilon(self):
        if not self.accountant.history:
            return 0.0
        return float(self.accountant.get_epsilon(delta=self.delta))

    def total_steps(self):
        return int(sum(k for _, _, k in self.accountant.history))

    def is_budget_exhausted(self, target_epsilon):
        return target_epsilon is not None and target_epsilon > 0 and self.get_epsilon() >= target_epsilon


def noisy_mean_bounded(values: np.ndarray, eps: float, rng: np.random.Generator,
                       lo=0.0, hi=1.0) -> float:
    """eps-DP (Laplace) release of the mean of values bounded in [lo, hi]."""
    n = max(1, len(values))
    mean = float(np.clip(values, lo, hi).mean()) if n > 0 else 0.5 * (lo + hi)
    if eps is None or eps <= 0 or math.isinf(eps):
        return mean
    scale = (hi - lo) / (n * eps)
    return float(np.clip(mean + rng.laplace(0.0, scale), lo, hi))
