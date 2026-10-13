# -*- coding: utf-8 -*-
"""Run one federated configuration for one or more seeds.

    python experiments/run_federated.py --config experiments/configs/full_system.yaml --seeds 0 1 2
    python experiments/run_federated.py --config experiments/configs/baseline_dp_fedavg.yaml --rounds 5 --seeds 0

Results: results/federated/<name>/seed_<s>/{history,privacy,local_eval,summary}.json + global_model.npz
Existing seeds are skipped unless --force is given, so long grids can be resumed.
"""
import argparse
from _common import Logger, load_yaml, already_done, collect_summaries, mean_sd, save_json, ROOT, RESULTS_ROOT
import os
import time

from src.federated.simulation import run_federated, merge_cfg
from src.utils import load_partitions, load_holdout

RESULTS = os.path.join(RESULTS_ROOT, 'federated')


def run_config(cfg, seeds, results_dir=RESULTS, force=False, parts=None, holdout=None, log=print):
    cfg = merge_cfg(cfg)
    parts = parts if parts is not None else load_partitions(cfg['data']['processed_dir'], cfg['fl']['n_clients'])
    holdout = holdout if holdout is not None else load_holdout(cfg['data']['processed_dir'])
    for s in seeds:
        if not force and already_done(results_dir, cfg['name'], s):
            log(f"[{cfg['name']} seed {s}] exists - skipped")
            continue
        run_federated(cfg, s, results_dir, parts=parts, holdout=holdout, log=log)
    sums = collect_summaries(results_dir, cfg['name'])
    m, sd = mean_sd([x['auc_pr'] for x in sums])
    agg = {'name': cfg['name'], 'n_seeds': len(sums), 'auc_pr_mean': m, 'auc_pr_sd': sd,
           'auc_roc_mean': mean_sd([x['auc_roc'] for x in sums])[0],
           'f1_mean': mean_sd([x['f1'] for x in sums])[0],
           'epsilon_max_mean': mean_sd([x['epsilon_max'] for x in sums])[0],
           'seeds': [x['seed'] for x in sums]}
    save_json(agg, os.path.join(results_dir, cfg['name'], 'aggregate.json'))
    if m is not None:
        log(f"[{cfg['name']}] {len(sums)} seed(s): AUC-PR {m:.4f} +/- {sd:.4f}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--local-epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--epsilon', type=float, default=None)
    ap.add_argument('--name', default=None, help='override run name')
    ap.add_argument('--results', default=RESULTS)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.rounds is not None: cfg.setdefault('fl', {})['n_rounds'] = args.rounds
    if args.local_epochs is not None: cfg.setdefault('fl', {})['local_epochs'] = args.local_epochs
    if args.batch is not None: cfg.setdefault('fl', {})['batch_size'] = args.batch
    if args.epsilon is not None: cfg.setdefault('dp', {})['target_epsilon'] = args.epsilon
    if args.name: cfg['name'] = args.name
    log = Logger(os.path.join(args.results, 'logs', f"{cfg.get('name', 'run')}.log"))
    log(f'config {args.config} seeds {args.seeds}')
    run_config(cfg, args.seeds, args.results, args.force, log=log)


if __name__ == '__main__':
    main()
