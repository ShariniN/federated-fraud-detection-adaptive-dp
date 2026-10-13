# -*- coding: utf-8 -*-
"""Shared training and evaluation harness for all experiments.

THRESHOLD SELECTION
-------------------
The decision threshold is searched on the pooled client CALIBRATION folds and
then applied, unchanged, to the hold-out test set. The calibration folds are
held out of training and are never resampled, which is precisely what they were
created for. Searching the threshold on the test set inflates every
threshold-dependent figure (F1, precision, recall, confusion matrix); the
threshold-independent metrics AUC-PR and AUC-ROC are unaffected either way.

`compute_all_metrics` reports both operating points so the size of that
optimism is visible rather than assumed.
"""

import os, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print('WARNING: Opacus not installed.')

N_CLIENTS = 5


def load_real_data(processed_dir='data/processed_clean', n_clients=N_CLIENTS):
    """
    Load all real IEEE-CIS client partitions.

    Returns the pooled client training folds (raw, un-resampled), the pooled
    client calibration folds (used for threshold selection), and the public
    hold-out (the unified test set, drawn from the most recent transactions).

    Returns:
        (X_train, y_train, X_cal, y_cal, X_test, y_test, input_dim)
    """
    holdout_path = os.path.join(processed_dir, 'public_holdout.pkl')
    with open(holdout_path, 'rb') as f:
        pub = pickle.load(f)
    X_test = pub['X'].astype(np.float32)
    y_test = pub['y'].astype(np.int64)

    X_trains, y_trains, X_cals, y_cals = [], [], [], []
    n_loaded = 0
    for cid in range(n_clients):
        path = os.path.join(processed_dir, 'partitions', 'client_' + str(cid) + '.pkl')
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            d = pickle.load(f)
        Xtr, ytr = d['train']
        X_trains.append(Xtr.astype(np.float32))
        y_trains.append(ytr.astype(np.int64))
        Xc, yc = d['calibration']
        X_cals.append(Xc.astype(np.float32))
        y_cals.append(yc.astype(np.int64))
        n_loaded += 1

    if n_loaded == 0:
        raise FileNotFoundError(
            'No client partitions found under ' + processed_dir +
            '. Run: python src/preprocessing/pipeline.py'
        )

    X_train = np.concatenate(X_trains, axis=0)
    y_train = np.concatenate(y_trains, axis=0)
    X_cal = np.concatenate(X_cals, axis=0)
    y_cal = np.concatenate(y_cals, axis=0)

    print('  Real data loaded (' + str(n_loaded) + ' clients pooled, raw pre-SMOTE):')
    print('    Train : ' + str(X_train.shape[0]) + ' samples  |  fraud=' +
          str(int(y_train.sum())) + ' (' + str(round(y_train.mean() * 100, 2)) + '%)')
    print('    Calib : ' + str(X_cal.shape[0]) + ' samples  |  fraud=' +
          str(int(y_cal.sum())) + ' (' + str(round(y_cal.mean() * 100, 2)) + '%)')
    print('    Test  : ' + str(X_test.shape[0]) + ' samples  |  fraud=' +
          str(int(y_test.sum())) + ' (' + str(round(y_test.mean() * 100, 2)) + '%)')
    print('    Dim   : ' + str(X_train.shape[1]))
    return X_train, y_train, X_cal, y_cal, X_test, y_test, int(X_train.shape[1])


def apply_borderline_smote(X, y, sampling_strategy=0.3, random_state=42):
    """Apply Borderline-SMOTE once. Used by every experiment script."""
    from imblearn.over_sampling import BorderlineSMOTE
    n_fraud = int(y.sum())
    n_benign = len(y) - n_fraud
    if n_fraud < 2 or n_benign == 0:
        print('  SMOTE skipped: fewer than 2 fraud samples')
        return X, y
    if n_fraud / n_benign >= sampling_strategy:
        print('  SMOTE skipped: fraud ratio ' + str(round(n_fraud / n_benign, 3)) +
              ' already meets target ' + str(sampling_strategy))
        return X, y
    k = min(5, n_fraud - 1)
    try:
        sm = BorderlineSMOTE(sampling_strategy=sampling_strategy,
                             k_neighbors=k, random_state=random_state, kind='borderline-1')
        Xr, yr = sm.fit_resample(X, y)
        print('  Borderline-SMOTE: fraud ' + str(n_fraud) + '->' + str(int(yr.sum())) +
              ' | total ' + str(len(y)) + '->' + str(len(yr)))
        return Xr, yr
    except Exception as e:
        print('  Borderline-SMOTE failed (' + str(e) + ') - using raw data')
        return X, y


def find_best_threshold(y_true, fraud_proba):
    """Scan thresholds and return the one maximising F1 on the given split.

    This must be called on a CALIBRATION split, never on the test split.
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_tau, best_f1, best_recall, best_prec = 0.50, 0.0, 0.0, 0.0
    for tau in thresholds:
        y_pred = (fraud_proba >= tau).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            best_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            best_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    return round(float(best_tau), 2), best_f1, best_recall, best_prec


def _predict_proba(model, X, device='cpu', batch_size=8192):
    raw_model = getattr(model, '_module', model)
    raw_model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, len(X), batch_size):
            hi = min(lo + batch_size, len(X))
            xb = torch.tensor(X[lo:hi], dtype=torch.float32).to(device)
            out[lo:hi] = torch.softmax(raw_model(xb), dim=1)[:, 1].cpu().numpy()
    return out


def compute_all_metrics(model, X_test, y_test, X_cal=None, y_cal=None,
                        device='cpu', label=''):
    """Compute all reported metrics.

    The operating threshold is chosen on (X_cal, y_cal) when supplied. The
    test-optimal threshold is also reported, purely to quantify how much
    optimism threshold-on-test selection would have introduced.
    """
    fraud_proba = _predict_proba(model, X_test, device)

    auc_pr = average_precision_score(y_test, fraud_proba)
    auc_roc = roc_auc_score(y_test, fraud_proba)

    y_pred_default = (fraud_proba >= 0.50).astype(int)
    f1_default = f1_score(y_test, y_pred_default, pos_label=1, zero_division=0)
    rec_default = recall_score(y_test, y_pred_default, pos_label=1, zero_division=0)

    # Honest operating point: threshold fitted on the calibration fold
    if X_cal is not None and y_cal is not None and int(np.sum(y_cal)) > 0:
        cal_proba = _predict_proba(model, X_cal, device)
        best_tau, _, _, _ = find_best_threshold(y_cal, cal_proba)
        tau_source = 'calibration split'
    else:
        best_tau = 0.50
        tau_source = 'default 0.50 (no calibration data available)'

    y_pred_best = (fraud_proba >= best_tau).astype(int)
    best_f1 = f1_score(y_test, y_pred_best, pos_label=1, zero_division=0)
    best_recall = recall_score(y_test, y_pred_best, pos_label=1, zero_division=0)
    best_prec = precision_score(y_test, y_pred_best, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_best)

    # Reference only: what threshold-on-test selection would have reported
    oracle_tau, oracle_f1, _, _ = find_best_threshold(y_test, fraud_proba)

    if label:
        print('\n  ' + '-' * 50)
        print('  Metrics: ' + label)
        print('  ' + '-' * 50)
    print('  [Threshold-independent]')
    print('    AUC-PR  (primary) : ' + str(round(auc_pr, 4)))
    print('    AUC-ROC           : ' + str(round(auc_roc, 4)))
    print('  [Default tau=0.50]')
    print('    F1 (fraud)        : ' + str(round(f1_default, 4)))
    print('    Recall (fraud)    : ' + str(round(rec_default, 4)))
    print('  [Calibrated tau=' + str(best_tau) + ' from ' + tau_source + ']')
    print('    F1 (fraud)        : ' + str(round(best_f1, 4)))
    print('    Recall (fraud)    : ' + str(round(best_recall, 4)))
    print('    Precision (fraud) : ' + str(round(best_prec, 4)))
    print('  [Reference] threshold-on-test would give tau=' + str(oracle_tau) +
          ', F1=' + str(round(oracle_f1, 4)) +
          '  (optimism +' + str(round(oracle_f1 - best_f1, 4)) + ')')
    print('  Confusion Matrix (tau=' + str(best_tau) + '):')
    print('    TN=' + str(cm[0, 0]) + '  FP=' + str(cm[0, 1]))
    print('    FN=' + str(cm[1, 0]) + '  TP=' + str(cm[1, 1]))

    return {
        'auc_pr': auc_pr, 'auc_roc': auc_roc,
        'f1_default': f1_default, 'recall_default': rec_default,
        'best_tau': best_tau, 'tau_source': tau_source,
        'f1_calibrated': best_f1,
        'recall_calibrated': best_recall, 'precision_calibrated': best_prec,
        'oracle_tau': oracle_tau, 'oracle_f1': oracle_f1,
        'threshold_optimism': float(oracle_f1 - best_f1),
        'confusion_matrix': cm, 'fraud_proba': fraud_proba,
    }


def plot_confusion_matrix(cm, model_name, tau, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ['Legitimate', 'Fraud']
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=13,
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title('Confusion Matrix - ' + model_name +
                 '\n(tau=' + str(round(tau, 2)) + ' from calibration split)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Confusion matrix saved: ' + save_path)


def train_model_dp(model, X_train, y_train, device='cpu',
                   epochs=15, lr=1e-3, batch_size=512,
                   max_grad_norm=1.0, noise_multiplier=1.0,
                   target_delta=1e-5, verbose=True):
    """Train a model under DP-SGD via Opacus, returning (model, epsilon)."""
    import warnings
    model = model.to(device)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    n_fraud = int(y_train.sum())
    n_benign = len(y_train) - n_fraud
    if n_fraud > 0 and n_benign > 0:
        weights = torch.tensor([1.0, n_benign / n_fraud], dtype=torch.float32).to(device)
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_t, y_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    epsilon = None
    pe = None

    if OPACUS_AVAILABLE:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pe = PrivacyEngine()
            model, optimizer, dataloader = pe.make_private(
                module=model, optimizer=optimizer, data_loader=dataloader,
                noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
            )

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for Xb, yb in dataloader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if verbose and (epoch % 5 == 0 or epoch == 1):
            avg = total_loss / max(n_batches, 1)
            if pe is not None:
                epsilon = pe.get_epsilon(target_delta)
                print('    Epoch ' + str(epoch) + '/' + str(epochs) +
                      '  loss=' + str(round(avg, 4)) + '  eps=' + str(round(epsilon, 3)))
            else:
                print('    Epoch ' + str(epoch) + '/' + str(epochs) +
                      '  loss=' + str(round(avg, 4)) + '  (no DP)')

    if pe is not None:
        epsilon = pe.get_epsilon(target_delta)

    final_model = getattr(model, '_module', model)
    return final_model, epsilon
