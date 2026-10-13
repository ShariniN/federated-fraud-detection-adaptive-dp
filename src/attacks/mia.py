"""Membership inference attacks (empirical privacy validation).

Two attacks, both reported with AUC **and** TPR at 0.1 % / 1 % FPR (Carlini et
al. 2022): accuracy or AUC alone hide an attack that identifies a small number
of members with certainty.

* Yeom loss attack: score = -loss(x, y). Members tend to have lower loss.
* Offline LiRA: K shadow models trained on random halves of a reference pool.
  For each target record, the OUT-shadow logit-scaled confidences give a
  Gaussian N(mu_out, sd_out); the membership score is the one-sided
  probability that an OUT model would produce a confidence at least as high
  as the target model's. (Offline variant = no per-record IN models needed.)

POSITIVE CONTROL. Every attack must be run against the non-private model
first. A failing attack on a DP model proves nothing unless the same attack
succeeds when the defence is removed.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

from ..models.mlp import FraudMLP
from ..training import train_epochs
from ..evaluation.metrics import tpr_at_fpr


def _true_class_conf(model, X, y, device='cpu', batch=8192):
    raw = getattr(model, '_module', model)
    raw.eval()
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for lo in range(0, len(X), batch):
            xb = torch.as_tensor(X[lo:lo + batch], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[lo:lo + batch], dtype=torch.long, device=device)
            p = F.softmax(raw(xb), dim=1).gather(1, yb.view(-1, 1)).squeeze(1)
            out[lo:lo + batch] = p.cpu().numpy()
    return out


def _logit_scale(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def attack_report(is_member, score):
    fpr, tpr, _ = roc_curve(is_member, score)
    rep = {'auc': float(roc_auc_score(is_member, score)), 'n_members': int(is_member.sum()),
           'n_non_members': int((~is_member.astype(bool)).sum())}
    rep.update(tpr_at_fpr(is_member, score))
    # subsampled ROC for plotting (log-log)
    idx = np.unique(np.linspace(0, len(fpr) - 1, 400).astype(int))
    rep['roc'] = {'fpr': fpr[idx].tolist(), 'tpr': tpr[idx].tolist()}
    return rep


def yeom_loss_attack(model, X_in, y_in, X_out, y_out, device='cpu'):
    conf_in = _true_class_conf(model, X_in, y_in, device)
    conf_out = _true_class_conf(model, X_out, y_out, device)
    score = np.concatenate([-(-np.log(np.clip(conf_in, 1e-7, 1))), -(-np.log(np.clip(conf_out, 1e-7, 1)))])
    is_member = np.concatenate([np.ones(len(conf_in)), np.zeros(len(conf_out))])
    return attack_report(is_member, score)


class OfflineLiRA:
    def __init__(self, input_dim=72, n_shadow=8, epochs=5, batch_size=1024, lr=1e-3, device='cpu', seed=0):
        self.input_dim, self.n_shadow, self.epochs = input_dim, n_shadow, epochs
        self.batch_size, self.lr, self.device, self.seed = batch_size, lr, device, seed
        self.shadows = []

    def fit_shadows(self, X_pool, y_pool, log=print):
        """Train K non-private shadow models on random halves of the reference pool."""
        rng = np.random.default_rng(self.seed)
        n = len(y_pool)
        self.shadow_masks = []
        for k in range(self.n_shadow):
            mask = rng.random(n) < 0.5
            m = FraudMLP(input_dim=self.input_dim)
            gen = torch.Generator().manual_seed(self.seed * 7919 + k)
            train_epochs(m, X_pool[mask], y_pool[mask], epochs=self.epochs, batch_size=self.batch_size,
                         lr=self.lr, sigma=None, generator=gen, device=self.device)
            self.shadows.append(m)
            self.shadow_masks.append(mask)
            log(f'    shadow {k + 1}/{self.n_shadow} trained on {int(mask.sum())} rows')
        return self

    def score(self, target_model, X, y, pool_index=None):
        """Membership score for records (X, y). pool_index: their indices in the shadow pool
        (so their own OUT shadows are used); None -> every shadow is treated as OUT."""
        t = _logit_scale(_true_class_conf(target_model, X, y, self.device))
        shadow_conf = np.stack([_logit_scale(_true_class_conf(s, X, y, self.device)) for s in self.shadows], 0)
        if pool_index is not None:
            in_mask = np.stack([m[pool_index] for m in self.shadow_masks], 0)      # [K, n]
            shadow_conf = np.where(in_mask, np.nan, shadow_conf)
        mu = np.nanmean(shadow_conf, axis=0)
        sd = np.nanstd(shadow_conf, axis=0) + 1e-3
        # global-variance fallback where a record has <2 OUT shadows
        bad = np.isnan(mu) | (np.sum(~np.isnan(shadow_conf), axis=0) < 2)
        mu[bad] = np.nanmean(shadow_conf)
        sd[bad] = np.nanstd(shadow_conf) + 1e-3
        return norm.logcdf(t, loc=mu, scale=sd)          # log P_out(conf <= observed): high = member-like

    def attack(self, target_model, X_in, y_in, X_out, y_out, idx_in=None, idx_out=None):
        s_in = self.score(target_model, X_in, y_in, idx_in)
        s_out = self.score(target_model, X_out, y_out, idx_out)
        is_member = np.concatenate([np.ones(len(s_in)), np.zeros(len(s_out))])
        return attack_report(is_member, np.concatenate([s_in, s_out]))


class LiRAAttack:
    """Backward-compatible facade used by src.evaluation.privacy_metrics."""

    def __init__(self, target_model, input_dim=72, device='cpu'):
        self.target_model, self.input_dim, self.device = target_model, input_dim, device

    def run_loss_attack(self, train_x, train_y, test_x, test_y):
        return yeom_loss_attack(self.target_model, train_x, train_y, test_x, test_y, self.device)['auc']
