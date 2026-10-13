# -*- coding: utf-8 -*-
"""Robustness to a malicious participant: attack the framework, then harden it.

One bank (default: client 1, the one holding the largest fraud share) is
malicious. Two behaviours:
  label_flip         trains on inverted labels
  model_replacement  scales its update x5 AND reports zero uncertainty, which
                     is exactly how an adversary games uncertainty weighting

Each attack is run against five aggregation rules: size (FedAvg), uncertainty
(UW-FedAvg), uncertainty_clipped (hardened UW-FedAvg), trimmed_mean, median.
Component 1 stays on throughout. Clean runs of the same rules are the reference.

    python experiments/run_robustness.py --rounds 10 --seeds 0
"""
import argparse
import os

from _common import Logger, load_yaml, collect_summaries, mean_sd, write_markdown_table, RESULTS_ROOT
from run_federated import run_config, RESULTS
from src.utils import load_partitions, load_holdout
from src.evaluation.plots import plot_bars

RULES = ['size', 'uncertainty', 'uncertainty_clipped', 'trimmed_mean', 'median']
ATTACKS = ['clean', 'label_flip', 'model_replacement']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--malicious', type=int, nargs='+', default=[1])
    ap.add_argument('--boost', type=float, default=5.0)
    ap.add_argument('--results', default=os.path.join(RESULTS, 'robustness'))
    ap.add_argument('--summarise-only', action='store_true')
    args = ap.parse_args()
    log = Logger(os.path.join(args.results, 'logs', 'robustness.log'))
    parts = load_partitions(); holdout = load_holdout()
    base = load_yaml(os.path.join('experiments', 'configs', 'full_system.yaml'))

    grid = {}
    for attack in ATTACKS:
        for rule in RULES:
            cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
            cfg['name'] = f'{attack}__{rule}'
            cfg['fl'] = dict(cfg['fl'], n_rounds=args.rounds, weighting=rule)
            cfg['dp'] = dict(cfg['dp'], eps_uncertainty=(0.05 if rule.startswith('uncertainty') else 0.0))
            cfg['attack'] = {'malicious_clients': [] if attack == 'clean' else args.malicious,
                             'type': None if attack == 'clean' else attack, 'replacement_boost': args.boost}
            if not args.summarise_only:
                run_config(cfg, args.seeds, args.results, parts=parts, holdout=holdout, log=log)
            sums = collect_summaries(args.results, cfg['name'])
            grid[(attack, rule)] = mean_sd([s['auc_pr'] for s in sums]) if sums else (None, None)

    rows = []
    for rule in RULES:
        clean = grid[('clean', rule)][0]
        r = [rule, clean]
        for attack in ATTACKS[1:]:
            m = grid[(attack, rule)][0]
            r += [m, (None if (m is None or clean is None) else m - clean)]
        rows.append(r)
    table = write_markdown_table(rows, ['aggregation', 'clean AUC-PR', 'label-flip AUC-PR', 'delta', 'model-replacement AUC-PR', 'delta'],
                                 os.path.join(args.results, 'table_robustness.md'))
    log('\n' + table)
    for attack in ATTACKS:
        vals = [grid[(attack, r)] for r in RULES]
        if all(v[0] is not None for v in vals):
            plot_bars(RULES, [v[0] for v in vals], [v[1] or 0 for v in vals],
                      os.path.join(args.results, f'bars_{attack}.png'),
                      title=f'{attack}: hold-out AUC-PR by aggregation rule (malicious client {args.malicious})')
    log('done')


if __name__ == '__main__':
    main()
