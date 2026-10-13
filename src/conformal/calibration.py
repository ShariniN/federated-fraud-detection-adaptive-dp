"""Split conformal calibration, marginal or class-conditional (Mondrian).

WHY MONDRIAN
------------
At 3.5 % prevalence a *marginal* 90 % coverage guarantee is dominated by the
benign class: a predictor could cover 93 % of benign transactions and almost
no fraud and still satisfy it. Class-conditional (Mondrian) calibration fits
one quantile per class, so the guarantee holds for the fraud class on its
own - which is the only coverage figure a bank cares about.

EXCHANGEABILITY
---------------
Conformal validity assumes calibration and test points are exchangeable. The
project's hold-out is *later in time* than the calibration folds, so the
guarantee is approximate under drift; the empirical coverage gap is reported
rather than assumed away.

PRIVACY
-------
The calibrated quantile is an order statistic of private calibration scores
and is not covered by the DP-SGD accountant. It is listed as an unaccounted
operation in the privacy ledger (see PROJECT_SUMMARY.md).
"""

import numpy as np
from .scores import lac_scores, raps_scores, score_all_labels


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile: the ceil((n+1)(1-alpha))-th order statistic."""
    n = len(scores)
    if n == 0:
        return float('inf')
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


class ConformalCalibrator:
    """Fits conformal thresholds on calibration probabilities and builds prediction sets."""

    def __init__(self, alpha=0.1, score='lac', mondrian=True, lambda_reg=0.01, n_classes=2):
        assert score in ('lac', 'raps')
        self.alpha = alpha
        self.score = score
        self.mondrian = mondrian
        self.lambda_reg = lambda_reg
        self.n_classes = n_classes
        self.q = None            # dict class -> threshold (mondrian) or {'all': q}
        self.n_cal = {}
        self.is_calibrated = False

    # ----------------------------------------------------------------- scoring
    def _true_scores(self, probs, y):
        if self.score == 'lac':
            return lac_scores(probs, y)
        return raps_scores(probs, y, self.lambda_reg)

    # ----------------------------------------------------------------- fit
    def fit(self, probs_cal: np.ndarray, y_cal: np.ndarray):
        s = self._true_scores(probs_cal, y_cal)
        self.q = {}
        if self.mondrian:
            for c in range(self.n_classes):
                m = y_cal == c
                self.n_cal[c] = int(m.sum())
                self.q[c] = conformal_quantile(s[m], self.alpha)
        else:
            self.n_cal['all'] = int(len(s))
            self.q['all'] = conformal_quantile(s, self.alpha)
        self.is_calibrated = True
        return self

    def threshold_for(self, c):
        return self.q[c] if self.mondrian else self.q['all']

    # ----------------------------------------------------------------- sets
    def prediction_sets(self, probs: np.ndarray) -> np.ndarray:
        """Boolean [n, C]: label c is in the set iff s(x, c) <= q_c."""
        assert self.is_calibrated
        all_scores = score_all_labels(probs, self.score, self.lambda_reg)
        sets = np.zeros_like(all_scores, dtype=bool)
        for c in range(self.n_classes):
            sets[:, c] = all_scores[:, c] <= self.threshold_for(c)
        return sets

    def evaluate(self, probs: np.ndarray, y: np.ndarray) -> dict:
        """Coverage (marginal and per class), set-size statistics and review-tier rates."""
        sets = self.prediction_sets(probs)
        covered = sets[np.arange(len(y)), y]
        size = sets.sum(axis=1)
        out = {
            'alpha': self.alpha, 'score': self.score, 'mondrian': self.mondrian,
            'thresholds': {str(k): float(v) for k, v in self.q.items()},
            'n_cal': {str(k): v for k, v in self.n_cal.items()},
            'coverage_marginal': float(covered.mean()),
            'coverage_benign': float(covered[y == 0].mean()) if (y == 0).any() else None,
            'coverage_fraud': float(covered[y == 1].mean()) if (y == 1).any() else None,
            'mean_set_size': float(size.mean()),
            'rate_empty': float((size == 0).mean()),
            'rate_singleton': float((size == 1).mean()),
            'rate_both': float((size == 2).mean()),
        }
        out.update(review_tier(sets, y))
        return out


def review_tier(sets: np.ndarray, y: np.ndarray = None) -> dict:
    """Turn prediction sets into the three-way operational decision.

    {0}      -> auto-approve
    {1}      -> auto-block
    {0,1}/{} -> human review (the model cannot commit)
    """
    approve = sets[:, 0] & ~sets[:, 1]
    block = sets[:, 1] & ~sets[:, 0]
    review = ~(approve | block)
    out = {
        'rate_auto_approve': float(approve.mean()),
        'rate_auto_block': float(block.mean()),
        'rate_human_review': float(review.mean()),
    }
    if y is not None:
        n_fraud = max(1, int((y == 1).sum()))
        n_benign = max(1, int((y == 0).sum()))
        out['fraud_auto_approved'] = int((approve & (y == 1)).sum())          # missed fraud (cost)
        out['fraud_auto_approved_rate'] = out['fraud_auto_approved'] / n_fraud
        out['benign_auto_blocked'] = int((block & (y == 0)).sum())            # wrongly declined
        out['benign_auto_blocked_rate'] = out['benign_auto_blocked'] / n_benign
        out['fraud_caught_or_reviewed_rate'] = float(((block | review) & (y == 1)).sum() / n_fraud)
        out['review_fraud_precision'] = float((review & (y == 1)).sum() / max(1, review.sum()))
    return out
