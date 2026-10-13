"""In-process federated simulation engine.

Every client is a ``BankClient`` living in this process; rounds are executed
sequentially and deterministically (seeded), which makes runs reproducible and
debuggable. The same client and aggregation code is exposed through Flower in
``fl_client.py`` / ``fl_strategy.py`` for the framework demonstration.

A run writes to ``<out_dir>/<name>/seed_<s>/``:
    history.json    per-round global metrics, per-client epsilon/uncertainty/weights
    privacy.json    per-client (n_k, q, sigma, steps, epsilon) table + ledger
    local_eval.json global model evaluated on each bank's own test fold
    global_model.npz final global weights
"""

import os
import time

import numpy as np

from ..models.mlp import FraudMLP
from ..training import state_to_numpy, numpy_to_state
from ..evaluation.metrics import evaluate_model
from ..utils import set_seed, save_json, load_partitions, load_holdout, pooled
from .client import BankClient
from .server import aggregate


DEFAULT_CFG = {
    'name': 'run',
    'fl': {'n_clients': 5, 'n_rounds': 20, 'local_epochs': 1, 'batch_size': 1024, 'lr': 1e-3,
           'weighting': 'size', 'mu': 0.0, 'u_floor': 0.05, 'clip_lo': 0.5, 'clip_hi': 2.0,
           'trim_ratio': 0.2},
    'dp': {'private': True, 'target_epsilon': 3.0, 'delta': 1e-5, 'max_grad_norm': 1.0,
           'lambda_adapt': 0.0, 'min_scale': 0.1, 'eps_uncertainty': 0.05, 'sigma': None},
    'data': {'processed_dir': 'data/processed_clean'},
    'attack': {'malicious_clients': [], 'type': None, 'replacement_boost': 5.0},
}


def merge_cfg(cfg):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CFG.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def client_cfg(cfg):
    fl, dp = cfg['fl'], cfg['dp']
    return {'batch_size': fl['batch_size'], 'local_epochs': fl['local_epochs'], 'lr': fl['lr'],
            'n_rounds': fl['n_rounds'], 'mu': fl.get('mu', 0.0), 'private': dp['private'],
            'target_epsilon': dp['target_epsilon'], 'delta': dp['delta'], 'sigma': dp.get('sigma'),
            'max_grad_norm': dp['max_grad_norm'], 'lambda_adapt': dp['lambda_adapt'],
            'min_scale': dp['min_scale'],
            'eps_uncertainty': dp['eps_uncertainty'] if fl['weighting'].startswith('uncertainty') else 0.0,
            'replacement_boost': cfg['attack'].get('replacement_boost', 5.0),
            'use_class_weights': True}


def run_federated(cfg, seed, out_dir, parts=None, holdout=None, device='cpu', log=print):
    cfg = merge_cfg(cfg)
    fl, dp = cfg['fl'], cfg['dp']
    run_dir = os.path.join(out_dir, cfg['name'], f'seed_{seed}')
    os.makedirs(run_dir, exist_ok=True)
    set_seed(seed)

    parts = parts if parts is not None else load_partitions(cfg['data']['processed_dir'], fl['n_clients'])
    X_ho, y_ho = holdout if holdout is not None else load_holdout(cfg['data']['processed_dir'])
    X_cal, y_cal = pooled(parts, 'calibration')
    input_dim = X_ho.shape[1]

    ccfg = client_cfg(cfg)
    bad = set(cfg['attack'].get('malicious_clients') or [])
    clients = [BankClient(p['client_id'] if 'client_id' in p else i, p, ccfg, seed=seed, device=device,
                          malicious=(cfg['attack']['type'] if i in bad else None))
               for i, p in enumerate(parts)]

    global_arrays = state_to_numpy(FraudMLP(input_dim=input_dim))
    eval_model = FraudMLP(input_dim=input_dim).to(device)

    def evaluate(arrays):
        numpy_to_state(eval_model, arrays)
        return evaluate_model(eval_model, X_ho, y_ho, X_cal, y_cal, device)

    history = []
    m0 = evaluate(global_arrays)
    history.append({'round': 0, 'global': m0, 'epsilon_max': 0.0, 'epsilon_mean': 0.0})
    log(f"[{cfg['name']} seed {seed}] clients: " +
        ', '.join(f"c{c.cid} n={c.n} sigma={c.sigma:.3f}" if c.sigma else f"c{c.cid} n={c.n} (non-private)"
                  for c in clients))

    t_start = time.time()
    for rnd in range(1, fl['n_rounds'] + 1):
        t0 = time.time()
        arrays_list, n_list, metrics_list = [], [], []
        for c in clients:
            arrs, n, m = c.fit(global_arrays, rnd)
            arrays_list.append(arrs); n_list.append(n); metrics_list.append(m)
        u_list = [m['uncertainty'] for m in metrics_list]
        global_arrays, weights = aggregate(fl['weighting'], arrays_list, n_list, u_list,
                                           u_floor=fl['u_floor'], clip_lo=fl['clip_lo'],
                                           clip_hi=fl['clip_hi'], trim_ratio=fl['trim_ratio'])
        g = evaluate(global_arrays)
        eps = [m['epsilon'] for m in metrics_list]
        rec = {'round': rnd, 'global': g,
               'clients': [{k: m[k] for k in ('client', 'epsilon', 'uncertainty', 'uncertainty_raw',
                                              'train_loss', 'mean_clip_scale', 'update_norm', 'n_train')}
                           for m in metrics_list],
               'weights': None if weights is None else [float(w) for w in weights],
               'epsilon_max': float(max(eps)), 'epsilon_mean': float(np.mean(eps)),
               'epsilon_uncertainty_total': (rnd * ccfg['eps_uncertainty']) if ccfg['eps_uncertainty'] else 0.0,
               'seconds': time.time() - t0}
        history.append(rec)
        save_json({'config': cfg, 'seed': seed, 'history': history}, os.path.join(run_dir, 'history.json'))
        wtxt = '' if weights is None else ' w=' + '/'.join(f'{w:.2f}' for w in weights)
        log(f"  r{rnd:02d} AUC-PR={g['auc_pr']:.4f} AUC-ROC={g['auc_roc']:.4f} F1={g['f1']:.3f} "
            f"eps_max={rec['epsilon_max']:.3f}{wtxt} ({rec['seconds']:.0f}s)")

    # ---- final artefacts
    privacy = {'per_client': [c.privacy_row() for c in clients],
               'ledger': {
                   'dp_sgd_epsilon_max': history[-1]['epsilon_max'],
                   'dp_sgd_delta': dp['delta'],
                   'uncertainty_release_epsilon_total': history[-1].get('epsilon_uncertainty_total', 0.0),
                   'unaccounted': ['imputer / scaler / PCA fitted on the training pool',
                                   'decision threshold selected on calibration folds',
                                   'conformal quantile (order statistic of calibration scores)']}}
    save_json(privacy, os.path.join(run_dir, 'privacy.json'))
    local_eval = [dict(client=c.cid, n_test=len(c.y_test), fraud_rate_test=float(c.y_test.mean()),
                       **c.evaluate_local(global_arrays)) for c in clients]
    save_json(local_eval, os.path.join(run_dir, 'local_eval.json'))
    np.savez(os.path.join(run_dir, 'global_model.npz'), *global_arrays)

    final = history[-1]['global']
    summary = {'name': cfg['name'], 'seed': seed, 'auc_pr': final['auc_pr'], 'auc_roc': final['auc_roc'],
               'f1': final['f1'], 'recall': final['recall'], 'precision': final['precision'],
               'epsilon_max': history[-1]['epsilon_max'], 'rounds': fl['n_rounds'],
               'best_auc_pr': max(h['global']['auc_pr'] for h in history),
               'seconds': time.time() - t_start, 'run_dir': run_dir}
    save_json(summary, os.path.join(run_dir, 'summary.json'))
    log(f"[{cfg['name']} seed {seed}] done: AUC-PR={final['auc_pr']:.4f} eps_max={summary['epsilon_max']:.3f} "
        f"in {summary['seconds']/60:.1f} min")
    return summary


def load_global_model(run_dir, input_dim=72):
    data = np.load(os.path.join(run_dir, 'global_model.npz'))
    arrays = [data[f'arr_{i}'] for i in range(len(data.files))]
    return numpy_to_state(FraudMLP(input_dim=input_dim), arrays)
