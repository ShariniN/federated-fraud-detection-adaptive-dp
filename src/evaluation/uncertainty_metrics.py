"""Conformal / uncertainty validation wrappers."""

import numpy as np
from scipy.stats import spearmanr

from ..conformal.scores import predict_proba, lac_scores
from ..conformal.calibration import ConformalCalibrator


def compute_coverage(model, X_cal, y_cal, X_test, y_test, alpha=0.1, score='lac', mondrian=True, device='cpu'):
    cal = ConformalCalibrator(alpha=alpha, score=score, mondrian=mondrian).fit(predict_proba(model, X_cal, device), y_cal)
    return cal.evaluate(predict_proba(model, X_test, device), y_test)


def uncertainty_allocation(model, X, y, lambda_adapt=0.5, min_scale=0.1, device='cpu'):
    """Component-1 diagnostic: how the per-sample sensitivity budget is allocated by class.

    Returns mean uncertainty u and mean clip scale w = (1-lambda) + lambda*u for fraud and benign
    samples, the fraud/benign ratio of w, and the share of frauds that are 'borderline' (u > 0.5,
    i.e. the model would misclassify them). Under uniform DP-SGD every sample has w = 1.
    """
    from ..dp.noise_adaptation import uncertainty_to_clip_scale
    probs = predict_proba(model, X, device)
    u = lac_scores(probs, y)
    w = uncertainty_to_clip_scale(u, lambda_adapt, min_scale)
    f, b = y == 1, y == 0
    out = {'lambda_adapt': lambda_adapt,
           'mean_u_fraud': float(u[f].mean()) if f.any() else None, 'mean_u_benign': float(u[b].mean()) if b.any() else None,
           'mean_w_fraud': float(w[f].mean()) if f.any() else None, 'mean_w_benign': float(w[b].mean()) if b.any() else None,
           'borderline_fraud_share': float((u[f] > 0.5).mean()) if f.any() else None}
    out['w_ratio_fraud_over_benign'] = (out['mean_w_fraud'] / out['mean_w_benign']) if (out['mean_w_fraud'] and out['mean_w_benign']) else None
    return out


def compute_uncertainty_fraud_correlation(model, X_test, y_test, device='cpu'):
    """Kept for compatibility. NOTE: with the LAC score this is identically -1 for true frauds
    (u = 1 - p_fraud), so it carries no information; use uncertainty_allocation instead."""
    probs = predict_proba(model, X_test, device)
    u = lac_scores(probs, y_test)
    fraud = y_test == 1
    if fraud.sum() < 3:
        return 0.0, 1.0
    corr, p = spearmanr(u[fraud], probs[fraud, 1])
    return (0.0, 1.0) if np.isnan(corr) else (float(corr), float(p))


def uncertainty_vs_size(weights, n_list):
    """Spearman rho between aggregation weights and client sizes: the 'is UW-FedAvg just FedAvg' test."""
    rho, p = spearmanr(weights, n_list)
    return (0.0, 1.0) if np.isnan(rho) else (float(rho), float(p))
