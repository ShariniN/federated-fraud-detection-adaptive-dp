# -*- coding: utf-8 -*-
"""Ablation grid, privacy-budget sweep and heterogeneity sweep.

  --group ablation   A baseline_dp_fedavg / B component1_only / C component2_only / D full_system
                     (+ fedavg_nonprivate and fedprox_dp references), all at epsilon = 3, alpha = 0.5
  --group epsilon    baseline vs full system at epsilon in {1, 3, 8} (+ non-private reference)
  --group alpha      baseline vs full system on re-partitioned pools with Dirichlet alpha in {0.1, 0.5, 5.0}
  --group all

Every run is resumable (existing seeds are skipped). Figures and markdown
tables are (re)built from whatever has finished, so the script can be re-run
after a partial grid.

    python experiments/run_ablation.py --group ablation --seeds 0 1 2 --rounds 20
"""
import argparse
import os

import numpy as np

from _common import Logger, load_yaml, collect_summaries, collect_histories, mean_sd, save_json, write_markdown_table, RESULTS_ROOT
from run_federated import run_config, RESULTS
from src.utils import load_partitions, load_holdout, load_pool
from src.preprocessing.partition import dirichlet_partition
from src.evaluation.plots import plot_convergence, plot_bars, plot_privacy_utility, plot_weights_vs_size
from src.evaluation.uncertainty_metrics import uncertainty_vs_size

CFG_DIR = os.path.join('experiments', 'configs')
ABLATION = ['baseline_dp_fedavg', 'component1_only', 'component2_only', 'full_system', 'fedavg_nonprivate', 'fedprox_dp']
LABELS = {'baseline_dp_fedavg': 'A: DP-FedAvg', 'component1_only': 'B: +Component 1', 'component2_only': 'C: +Component 2',
          'full_system': 'D: full system', 'fedavg_nonprivate': 'FedAvg (no DP)', 'fedprox_dp': 'FedProx DP',
          'full_system_clipped': 'D: full (clipped w)'}


def cfg_for(name, rounds=None, epsilon=None, run_name=None):
    cfg = load_yaml(os.path.join(CFG_DIR, f'{name}.yaml'))
    if rounds: cfg['fl']['n_rounds'] = rounds
    if epsilon is not None: cfg['dp']['target_epsilon'] = epsilon
    if run_name: cfg['name'] = run_name
    return cfg


def summarise(names, results_dir, log, tag):
    rows, curves = [], {}
    for n in names:
        sums = collect_summaries(results_dir, n)
        if not sums:
            continue
        m, sd = mean_sd([s['auc_pr'] for s in sums])
        rows.append([LABELS.get(n, n), len(sums), m, sd, mean_sd([s['auc_roc'] for s in sums])[0],
                     mean_sd([s['f1'] for s in sums])[0], mean_sd([s['recall'] for s in sums])[0],
                     mean_sd([s['epsilon_max'] for s in sums])[0]])
        hists = collect_histories(results_dir, n)
        curves[LABELS.get(n, n)] = [[(h['round'], h['global']['auc_pr']) for h in H['history']] for H in hists]
    if rows:
        table = write_markdown_table(rows, ['system', 'seeds', 'AUC-PR', 'sd', 'AUC-ROC', 'F1', 'recall', 'eps_max'],
                                     os.path.join(results_dir, f'table_{tag}.md'))
        log('\n' + table)
        plot_bars([r[0] for r in rows], [r[2] for r in rows], [r[3] or 0 for r in rows],
                  os.path.join(results_dir, f'bars_{tag}.png'), title=f'{tag}: hold-out AUC-PR (mean +/- sd over seeds)')
        plot_convergence(curves, os.path.join(results_dir, f'convergence_{tag}.png'))
    return rows


def weights_diagnostic(results_dir, name, log):
    """Weights-vs-n_k plot and Spearman rho for an uncertainty-weighted run (audit item S4)."""
    hists = collect_histories(results_dir, name)
    if not hists:
        return
    H = hists[0]['history']
    W = [h['weights'] for h in H if h.get('weights')]
    if not W:
        return
    n_list = [c['n_train'] for c in H[1]['clients']]
    rho, p = uncertainty_vs_size(np.mean(W, axis=0), n_list)
    plot_weights_vs_size(W, n_list, os.path.join(results_dir, name, 'weights_vs_size.png'), rho=rho)
    save_json({'spearman_rho_weight_vs_n': rho, 'p_value': p, 'mean_weights': np.mean(W, axis=0).tolist(),
               'n_list': n_list}, os.path.join(results_dir, name, 'weights_diagnostic.json'))
    log(f'[{name}] Spearman rho(weight, n_k) = {rho:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=['ablation', 'epsilon', 'alpha', 'all'], default='ablation')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--epsilons', type=float, nargs='+', default=[1.0, 3.0, 8.0])
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.5, 5.0])
    ap.add_argument('--results', default=RESULTS)
    ap.add_argument('--summarise-only', action='store_true')
    args = ap.parse_args()
    log = Logger(os.path.join(args.results, 'logs', f'ablation_{args.group}.log'))
    parts = load_partitions(); holdout = load_holdout()

    if args.group in ('ablation', 'all'):
        if not args.summarise_only:
            for n in ABLATION:
                run_config(cfg_for(n, args.rounds), args.seeds, args.results, parts=parts, holdout=holdout, log=log)
        summarise(ABLATION, args.results, log, 'ablation')
        for n in ('component2_only', 'full_system'):
            weights_diagnostic(args.results, n, log)

    if args.group in ('epsilon', 'all'):
        series = {}
        for base in ('baseline_dp_fedavg', 'full_system'):
            pts = []
            for e in args.epsilons:
                rn = f'{base}_eps{e:g}'
                if not args.summarise_only:
                    run_config(cfg_for(base, args.rounds, epsilon=e, run_name=rn), args.seeds, args.results,
                               parts=parts, holdout=holdout, log=log)
                sums = collect_summaries(args.results, rn)
                if sums:
                    m, sd = mean_sd([s['auc_pr'] for s in sums]); pts.append((e, m, sd))
            series[LABELS[base]] = pts
        sums = collect_summaries(args.results, 'fedavg_nonprivate')
        if sums:
            m, sd = mean_sd([s['auc_pr'] for s in sums])
            series['FedAvg (no DP)'] = [(float('inf'), m, sd)]
        names = [f'{b}_eps{e:g}' for b in ('baseline_dp_fedavg', 'full_system') for e in args.epsilons]
        LABELS.update({n: n for n in names})
        summarise(names, args.results, log, 'epsilon_sweep')
        if any(series.values()):
            plot_privacy_utility(series, os.path.join(args.results, 'privacy_utility.png'))
            save_json(series, os.path.join(args.results, 'privacy_utility.json'))

    if args.group in ('alpha', 'all'):
        X_pool, y_pool = load_pool()
        names = []
        for a in args.alphas:
            pa = dirichlet_partition(X_pool, y_pool, 5, alpha=a, random_state=42, verbose=False)
            for i, p in enumerate(pa):
                p.setdefault('client_id', i)
            for base in ('baseline_dp_fedavg', 'full_system'):
                rn = f'{base}_alpha{a:g}'
                names.append(rn); LABELS[rn] = f'{LABELS[base]} alpha={a:g}'
                if not args.summarise_only:
                    run_config(cfg_for(base, args.rounds, run_name=rn), args.seeds, args.results,
                               parts=pa, holdout=holdout, log=log)
        summarise(names, args.results, log, 'alpha_sweep')
    log('done')


if __name__ == '__main__':
    main()
