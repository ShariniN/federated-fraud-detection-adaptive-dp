"""Utility evaluation metrics.

AUC-PR is the primary metric: at ~3.5 % fraud prevalence AUC-ROC is inflated
by the volume of easy negatives. Threshold-dependent figures (F1, recall,
precision, confusion matrix) are always reported at an operating point chosen
on a *calibration* split, never on the test split.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score,
                             f1_score, precision_score, average_precision_score, confusion_matrix)


def compute_utility_metrics(y_true, y_pred, y_prob):
    """Legacy signature kept for older scripts."""
    if len(np.unique(y_true)) < 2:
        return {"auc_pr": 0.5, "auc_roc": 0.5, "recall_fraud": 0.0, "f1_fraud": 0.0,
                "average_precision": 0.0, "accuracy": float(np.mean(y_pred == y_true))}
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {"auc_pr": float(auc(recall, precision)), "auc_roc": float(roc_auc_score(y_true, y_prob)),
            "recall_fraud": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_fraud": float(f1_score(y_true, y_pred, zero_division=0)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "accuracy": float(np.mean(y_pred == y_true))}


def predict_fraud_proba(model, X, device='cpu', batch_size=8192):
    raw = getattr(model, '_module', model)
    raw.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[lo:lo + batch_size], dtype=torch.float32, device=device)
            out[lo:lo + batch_size] = F.softmax(raw(xb), dim=1)[:, 1].cpu().numpy()
    return out


def best_f1_threshold(y_true, proba, grid=None):
    """Threshold maximising F1 on the given (calibration) split."""
    grid = np.arange(0.02, 0.99, 0.01) if grid is None else grid
    best_tau, best_f1 = 0.5, -1.0
    for tau in grid:
        f1 = f1_score(y_true, (proba >= tau).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return round(best_tau, 2), float(best_f1)


def tpr_at_fpr(y_true, score, fpr_targets=(0.001, 0.01)):
    """True-positive rate at fixed false-positive rates (the MIA reporting standard)."""
    fpr, tpr, _ = roc_curve(y_true, score)
    out = {}
    for t in fpr_targets:
        out[f'tpr_at_fpr_{t:g}'] = float(np.max(tpr[fpr <= t])) if np.any(fpr <= t) else 0.0
    return out


def evaluate_model(model, X_test, y_test, X_cal=None, y_cal=None, device='cpu', tau=None):
    """All reported utility metrics for a trained model.

    The operating threshold is fitted on (X_cal, y_cal) unless ``tau`` is given.
    """
    proba = predict_fraud_proba(model, X_test, device)
    out = {}
    if len(np.unique(y_test)) < 2:
        out.update({'auc_pr': None, 'auc_roc': None})
    else:
        out['auc_pr'] = float(average_precision_score(y_test, proba))
        out['auc_roc'] = float(roc_auc_score(y_test, proba))
    if tau is None:
        if X_cal is not None and y_cal is not None and int(np.sum(y_cal)) > 0:
            tau, _ = best_f1_threshold(y_cal, predict_fraud_proba(model, X_cal, device))
            out['tau_source'] = 'calibration'
        else:
            tau, out['tau_source'] = 0.5, 'default'
    out['tau'] = float(tau)
    pred = (proba >= tau).astype(int)
    out['f1'] = float(f1_score(y_test, pred, zero_division=0))
    out['recall'] = float(recall_score(y_test, pred, zero_division=0))
    out['precision'] = float(precision_score(y_test, pred, zero_division=0))
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    out['confusion'] = {'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])}
    out['prevalence'] = float(np.mean(y_test))
    return out
