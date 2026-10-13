# -*- coding: utf-8 -*-
"""Conformal prediction review tier on trained models.

For each model (federated global models from results/federated/<name>/seed_0 and
the centralised baselines in results/baselines/models) this script:
  1. calibrates split-conformal thresholds on the pooled client calibration folds,
     marginal AND class-conditional (Mondrian), at alpha in {0.05, 0.10, 0.20};
  2. evaluates coverage (marginal / benign / fraud), set sizes and the three-way
     operational decision (auto-approve / auto-block / human review) on the
     time-shifted public hold-out - the coverage gap is the drift penalty;
  3. repeats the calibration per bank (each bank's own calibration -> own test fold)
     to show per-bank operating points;
  4. attaches an illustrative cost model to the review-tier rates.

    python experiments/run_conformal.py --models full_system baseline_dp_fedavg central_dp_eps3 central_nonprivate
"""
import argparse
import glob
import os

import numpy as np
import torch

from _common import Logger, save_json, write_markdown_table, RESULTS_ROOT
from src.utils import load_partitions, load_holdout, pooled
from src.models.mlp import FraudMLP
from src.federated.simulation import load_global_model
from src.conformal import ConformalCalibrator, predict_proba, cost_model, lac_scores
from src.evaluation.uncertainty_metrics import uncertainty_allocation
from src.evaluation.plots import plot_review_tier

OUT = os.path.join(RESULTS_ROOT, 'conformal')


def find_model(name, input_dim=72):
    fed = sorted(glob.glob(os.path.join(RESULTS_ROOT, 'federated', name, 'seed_*')))
    if fed:
        return load_global_model(fed[0], input_dim), fed[0]
    pt = sorted(glob.glob(os.path.join(RESULTS_ROOT, 'baselines', 'models', f'{name}_seed*.pt')))
    if pt:
        m = FraudMLP(input_dim=input_dim); m.load_state_dict(torch.load(pt[0])); return m, pt[0]
    raise FileNotFoundError(f'no trained model found for {name!r}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['full_system', 'baseline_dp_fedavg', 'central_dp_eps3', 'central_nonprivate'])
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.05, 0.10, 0.20])
    ap.add_argument('--score', default='lac', choices=['lac', 'raps'])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log = Logger(os.path.join(OUT, 'conformal.log'))
    parts = load_partitions(); X_ho, y_ho = load_holdout()
    X_cal, y_cal = pooled(parts, 'calibration')
    X_te, y_te = pooled(parts, 'test')

    all_results = {}
    tier_rows = []
    for name in args.models:
        try:
            model, src = find_model(name)
        except FileNotFoundError as e:
            log(f'skip {name}: {e}'); continue
        log(f'== {name}  ({src})')
        p_cal = predict_proba(model, X_cal); p_ho = predict_proba(model, X_ho); p_te = predict_proba(model, X_te)
        res = {'source': src, 'settings': []}
        for alpha in args.alphas:
            for mondrian in (False, True):
                cal = ConformalCalibrator(alpha=alpha, score=args.score, mondrian=mondrian).fit(p_cal, y_cal)
                ev_ho = cal.evaluate(p_ho, y_ho)                  # time-shifted hold-out (deployment view)
                ev_te = cal.evaluate(p_te, y_te)                  # exchangeable pooled client tests (validity check)
                rec = {'alpha': alpha, 'mondrian': mondrian, 'holdout': ev_ho, 'client_tests': ev_te,
                       'coverage_gap_fraud_drift': (ev_te['coverage_fraud'] or 0) - (ev_ho['coverage_fraud'] or 0),
                       'cost_holdout': cost_model(ev_ho, len(y_ho))}
                res['settings'].append(rec)
                log(f"  alpha={alpha:.2f} {'mondrian' if mondrian else 'marginal':8s} | hold-out cov fraud={ev_ho['coverage_fraud']:.3f} "
                    f"benign={ev_ho['coverage_benign']:.3f} | approve={ev_ho['rate_auto_approve']:.3f} block={ev_ho['rate_auto_block']:.3f} "
                    f"review={ev_ho['rate_human_review']:.3f} | missed fraud={ev_ho['fraud_auto_approved_rate']:.3f} "
                    f"| exch. cov fraud={ev_te['coverage_fraud']:.3f}")
                if mondrian and abs(alpha - 0.10) < 1e-9:
                    tier_rows.append({'name': name, **{k: ev_ho[k] for k in ('rate_auto_approve', 'rate_auto_block', 'rate_human_review')}})
        # per-bank calibration (own cal -> own test), Mondrian alpha=0.1
        per_bank = []
        for p in parts:
            cal = ConformalCalibrator(alpha=0.10, score=args.score, mondrian=True).fit(predict_proba(model, p['calibration'][0]), p['calibration'][1])
            ev = cal.evaluate(predict_proba(model, p['test'][0]), p['test'][1])
            per_bank.append({'client': p['client_id'], 'n_cal_fraud': int(p['calibration'][1].sum()),
                             **{k: ev[k] for k in ('coverage_fraud', 'coverage_benign', 'rate_auto_approve',
                                                    'rate_auto_block', 'rate_human_review', 'thresholds')}})
        res['per_bank_mondrian_alpha0.1'] = per_bank
        res['uncertainty_allocation'] = uncertainty_allocation(model, X_ho, y_ho, lambda_adapt=0.5)
        res['mean_lac_uncertainty_holdout'] = {'fraud': float(lac_scores(p_ho, y_ho)[y_ho == 1].mean()),
                                               'benign': float(lac_scores(p_ho, y_ho)[y_ho == 0].mean())}
        all_results[name] = res
        save_json(all_results, os.path.join(OUT, 'conformal_results.json'))

    rows = []
    for name, res in all_results.items():
        for s in res['settings']:
            h = s['holdout']
            rows.append([name, s['alpha'], 'Mondrian' if s['mondrian'] else 'marginal', h['coverage_marginal'],
                         h['coverage_fraud'], h['coverage_benign'], h['rate_auto_approve'], h['rate_auto_block'],
                         h['rate_human_review'], h['fraud_auto_approved_rate'], s['cost_holdout']['cost_total']])
    table = write_markdown_table(rows, ['model', 'alpha', 'calibration', 'cov marginal', 'cov fraud', 'cov benign',
                                        'auto-approve', 'auto-block', 'human review', 'missed fraud rate', 'cost (illustrative)'],
                                 os.path.join(OUT, 'table_conformal.md'))
    log('\n' + table)
    if tier_rows:
        plot_review_tier(tier_rows, os.path.join(OUT, 'review_tier.png'),
                         title='Review tier on the hold-out (Mondrian, alpha = 0.10)')
    log('done')


if __name__ == '__main__':
    main()
