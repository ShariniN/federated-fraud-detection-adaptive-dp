# -*- coding: utf-8 -*-
"""Baselines that make the federated results readable.

  i   local-only, per bank (non-private and DP at the target epsilon) - what federation must beat
  ii  centralised non-private on the pooled training folds               - utility upper bound
  iii centralised DP-SGD at epsilon in {1, 3, 8} (budget-first sigma)    - isolates the privacy tax

No SMOTE anywhere (class-weighted loss only). Threshold on the pooled calibration folds.
Outputs: results/baselines/{baselines.json, table.md, models/*.pt}.

    python experiments/run_baselines.py --seeds 0 1 2 --epochs 15
"""
import argparse
import os
import time

import numpy as np
import torch
from opacus.accountants import RDPAccountant

from _common import Logger, save_json, load_json, mean_sd, write_markdown_table, RESULTS_ROOT
from src.utils import set_seed, load_partitions, load_holdout, pooled, torch_generator
from src.models.mlp import FraudMLP
from src.training import train_epochs
from src.dp.accountant import calibrate_sigma
from src.evaluation.metrics import evaluate_model
from src.evaluation.plots import plot_per_bank, plot_bars

OUT = os.path.join(RESULTS_ROOT, 'baselines')


def train_one(X, y, *, epochs, batch, lr, epsilon=None, delta=1e-5, seed=0, device='cpu'):
    set_seed(seed)
    model = FraudMLP(input_dim=X.shape[1])
    acc, sigma = None, None
    if epsilon is not None:
        acc = RDPAccountant()
        sigma = calibrate_sigma(epsilon, delta, len(y), batch, epochs)
    stats = train_epochs(model, X, y, epochs=epochs, batch_size=batch, lr=lr, sigma=sigma,
                         accountant=acc, generator=torch_generator(seed), device=device)
    eps = float(acc.get_epsilon(delta=delta)) if acc is not None else float('inf')
    return model, sigma, eps, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epsilons', type=float, nargs='+', default=[1.0, 3.0, 8.0])
    ap.add_argument('--target-epsilon', type=float, default=3.0, help='epsilon for the local-only DP arm')
    ap.add_argument('--delta', type=float, default=1e-5)
    ap.add_argument('--skip-local', action='store_true')
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'models'), exist_ok=True)
    log = Logger(os.path.join(OUT, 'baselines.log'))

    parts = load_partitions()
    X_ho, y_ho = load_holdout()
    X_tr, y_tr = pooled(parts, 'train')
    X_cal, y_cal = pooled(parts, 'calibration')
    log(f'pooled train {len(y_tr):,} rows ({y_tr.mean():.2%} fraud), hold-out {len(y_ho):,}')

    results_path = os.path.join(OUT, 'baselines.json')
    results = load_json(results_path) if os.path.exists(results_path) else {'centralised': [], 'local_only': []}
    done = {(r['arm'], r['seed']) for r in results['centralised']}

    # ---------------------------------------------------------------- centralised
    arms = [('central_nonprivate', None)] + [(f'central_dp_eps{e:g}', e) for e in args.epsilons]
    for arm, eps in arms:
        for s in args.seeds:
            if (arm, s) in done:
                log(f'{arm} seed {s} exists - skipped'); continue
            t0 = time.time()
            model, sigma, eps_spent, stats = train_one(X_tr, y_tr, epochs=args.epochs, batch=args.batch, lr=args.lr,
                                                       epsilon=eps, delta=args.delta, seed=s)
            m = evaluate_model(model, X_ho, y_ho, X_cal, y_cal)
            per_bank = [evaluate_model(model, p['test'][0], p['test'][1], p['calibration'][0], p['calibration'][1])['auc_pr']
                        for p in parts]
            rec = {'arm': arm, 'seed': s, 'target_epsilon': eps, 'epsilon': eps_spent, 'sigma': sigma,
                   'epochs': args.epochs, 'batch': args.batch, 'n_train': int(len(y_tr)), **m,
                   'per_bank_auc_pr': per_bank, 'seconds': time.time() - t0}
            results['centralised'].append(rec)
            torch.save(model.state_dict(), os.path.join(OUT, 'models', f'{arm}_seed{s}.pt'))
            save_json(results, results_path)
            log(f"{arm} seed {s}: AUC-PR {m['auc_pr']:.4f} AUC-ROC {m['auc_roc']:.4f} F1 {m['f1']:.3f} "
                f"eps {eps_spent:.3g} sigma {sigma} ({rec['seconds']:.0f}s)")

    # ---------------------------------------------------------------- local-only per bank
    if not args.skip_local:
        done_local = {(r['arm'], r['client'], r['seed']) for r in results['local_only']}
        for private in (False, True):
            arm = f'local_dp_eps{args.target_epsilon:g}' if private else 'local_nonprivate'
            for p in parts:
                cid = p['client_id']
                Xc, yc = p['train']
                for s in args.seeds:
                    if (arm, cid, s) in done_local:
                        continue
                    t0 = time.time()
                    model, sigma, eps_spent, stats = train_one(
                        Xc, yc, epochs=args.epochs, batch=args.batch, lr=args.lr,
                        epsilon=args.target_epsilon if private else None, delta=args.delta, seed=s)
                    m_local = evaluate_model(model, p['test'][0], p['test'][1], p['calibration'][0], p['calibration'][1])
                    m_glob = evaluate_model(model, X_ho, y_ho, p['calibration'][0], p['calibration'][1])
                    rec = {'arm': arm, 'client': cid, 'seed': s, 'n_train': int(len(yc)), 'fraud_train': int(yc.sum()),
                           'epsilon': eps_spent, 'sigma': sigma,
                           'local_test': m_local, 'holdout': m_glob, 'seconds': time.time() - t0}
                    results['local_only'].append(rec)
                    save_json(results, results_path)
                    log(f"{arm} bank {cid} seed {s}: own-test AUC-PR {m_local['auc_pr']}, hold-out AUC-PR "
                        f"{m_glob['auc_pr']:.4f} ({rec['seconds']:.0f}s)")

    # ---------------------------------------------------------------- tables + figures
    rows = []
    for arm, _ in arms:
        rs = [r for r in results['centralised'] if r['arm'] == arm]
        if not rs:
            continue
        m, sd = mean_sd([r['auc_pr'] for r in rs])
        rows.append([arm, len(rs), m, sd, mean_sd([r['auc_roc'] for r in rs])[0], mean_sd([r['f1'] for r in rs])[0],
                     mean_sd([r['epsilon'] for r in rs])[0] if rs[0]['target_epsilon'] else 'inf'])
    table = write_markdown_table(rows, ['arm', 'seeds', 'AUC-PR', 'sd', 'AUC-ROC', 'F1', 'epsilon'],
                                 os.path.join(OUT, 'table_centralised.md'))
    log('\n' + table)
    if rows:
        plot_bars([r[0] for r in rows], [r[2] for r in rows], [r[3] or 0 for r in rows],
                  os.path.join(OUT, 'centralised_auc_pr.png'), title='Centralised baselines (hold-out AUC-PR)',
                  baseline=float(y_ho.mean()))

    if results['local_only']:
        bank_ids = [p['client_id'] for p in parts]
        series = {}
        lrows = []
        for arm in sorted({r['arm'] for r in results['local_only']}):
            own = []; glob = []
            for cid in bank_ids:
                rs = [r for r in results['local_only'] if r['arm'] == arm and r['client'] == cid]
                own.append(mean_sd([r['local_test']['auc_pr'] for r in rs])[0] if rs else None)
                glob.append(mean_sd([r['holdout']['auc_pr'] for r in rs])[0] if rs else None)
                if rs:
                    lrows.append([arm, cid, rs[0]['n_train'], rs[0]['fraud_train'], len(rs), own[-1], glob[-1],
                                  mean_sd([r['epsilon'] for r in rs])[0] if 'dp' in arm else 'inf'])
            series[f'{arm} (own test)'] = own
            series[f'{arm} (hold-out)'] = glob
        table = write_markdown_table(lrows, ['arm', 'bank', 'n_train', 'fraud', 'seeds', 'AUC-PR own test',
                                             'AUC-PR hold-out', 'epsilon'], os.path.join(OUT, 'table_local_only.md'))
        log('\n' + table)
        plot_per_bank(bank_ids, series, os.path.join(OUT, 'local_only_per_bank.png'),
                      title='Local-only training per bank (what federation must beat)')
    log('done')


if __name__ == '__main__':
    main()
