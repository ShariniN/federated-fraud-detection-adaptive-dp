# -*- coding: utf-8 -*-
"""Compile every finished experiment into results/RESULTS.md (tables + figure links).

Safe to run at any time; sections whose inputs are missing are skipped.
    python experiments/make_results_report.py
"""
import glob
import os
import datetime

import numpy as np

from _common import load_json, mean_sd, write_markdown_table, collect_summaries, collect_histories, RESULTS_ROOT

FED = os.path.join(RESULTS_ROOT, 'federated')
LABELS = {'baseline_dp_fedavg': 'A: DP-FedAvg (uniform noise, size weights)', 'component1_only': 'B: + Component 1 (adaptive sensitivity)',
          'component2_only': 'C: + Component 2 (uncertainty weights)', 'full_system': 'D: full hierarchical system',
          'full_system_clipped': 'D-hardened: full system, clipped weights', 'fedavg_nonprivate': 'FedAvg, no DP (reference)',
          'fedprox_dp': 'FedProx + uniform DP (reference)'}


def rel(p):
    return os.path.relpath(p, RESULTS_ROOT).replace('\\', '/')


def md_file(path):
    return open(path, encoding='utf-8').read().strip() if os.path.exists(path) else None


def main():
    out = [f'# Results — compiled {datetime.datetime.now():%Y-%m-%d %H:%M}', '',
           'All numbers are hold-out (most recent 10 % of transactions, 59,054 rows, 3.75 % fraud) unless stated. '
           'AUC-PR is the primary metric (random = 0.0375). Thresholds for F1/recall are fitted on the pooled client '
           'calibration folds. epsilon values are per client (max over clients) at delta = 1e-5, RDP accountant, '
           'composed over all rounds; budget-first calibration targets epsilon = 3 unless stated.', '']

    # ------------------------------------------------------------------ baselines
    t = md_file(os.path.join(RESULTS_ROOT, 'baselines', 'table_centralised.md'))
    if t:
        out += ['## 1. Centralised baselines (pooled client training folds, no SMOTE, class weights)', '', t, '',
                f'![]({rel(os.path.join(RESULTS_ROOT, "baselines", "centralised_auc_pr.png"))})', '']
    t = md_file(os.path.join(RESULTS_ROOT, 'baselines', 'table_local_only.md'))
    if t:
        out += ['## 2. Local-only training per bank (what federation must beat)', '', t, '',
                f'![]({rel(os.path.join(RESULTS_ROOT, "baselines", "local_only_per_bank.png"))})', '']

    # ------------------------------------------------------------------ ablation
    t = md_file(os.path.join(FED, 'table_ablation.md'))
    if t:
        out += ['## 3. Ablation: which component does the work? (alpha = 0.5, epsilon = 3, 20 rounds)', '', t, '',
                f'![]({rel(os.path.join(FED, "bars_ablation.png"))})', '',
                f'![]({rel(os.path.join(FED, "convergence_ablation.png"))})', '']
        for n in ('component2_only', 'full_system', 'full_system_clipped'):
            d = os.path.join(FED, n, 'weights_diagnostic.json')
            if os.path.exists(d):
                j = load_json(d)
                out += [f"*{LABELS.get(n, n)}*: Spearman rho(aggregation weight, n_k) = **{j['spearman_rho_weight_vs_n']:.2f}** "
                        f"(5 clients, rank correlation only); mean weights {['%.2f' % w for w in j['mean_weights']]} vs n_k/N "
                        f"{['%.2f' % (x / sum(j['n_list'])) for x in j['n_list']]}. ![]({rel(os.path.join(FED, n, 'weights_vs_size.png'))})", '']

    # ------------------------------------------------------------------ federation dividend / privacy tax
    base = load_json(os.path.join(RESULTS_ROOT, 'baselines', 'baselines.json')) if os.path.exists(os.path.join(RESULTS_ROOT, 'baselines', 'baselines.json')) else None
    fed_local = {}
    for n in ('baseline_dp_fedavg', 'full_system', 'fedavg_nonprivate'):
        runs = sorted(glob.glob(os.path.join(FED, n, 'seed_*', 'local_eval.json')))
        if runs:
            per = [load_json(r) for r in runs]
            fed_local[n] = {c['client']: mean_sd([p[i]['auc_pr'] for p in per])[0] for i, c in enumerate(per[0])}
    if base and fed_local:
        banks = sorted({r['client'] for r in base['local_only']})
        rows = []
        for b in banks:
            def loc(arm):
                rs = [r for r in base['local_only'] if r['arm'] == arm and r['client'] == b]
                return mean_sd([r['local_test']['auc_pr'] for r in rs])[0] if rs else None
            l_np, l_dp = loc('local_nonprivate'), loc('local_dp_eps3')
            f_np = fed_local.get('fedavg_nonprivate', {}).get(b); f_dp = fed_local.get('baseline_dp_fedavg', {}).get(b)
            f_full = fed_local.get('full_system', {}).get(b)
            n_tr = next(r['n_train'] for r in base['local_only'] if r['client'] == b)
            rows.append([f'bank {b}', n_tr, l_np, l_dp, f_np, f_dp, f_full,
                         (None if (f_full is None or l_dp is None) else f_full - l_dp),
                         (None if (f_full is None or f_np is None) else f_full - f_np)])
        out += ['## 4. Federation dividend and privacy tax, per bank (AUC-PR on each bank\'s own test fold)', '',
                write_markdown_table(rows, ['bank', 'n_train', 'local no-DP', 'local DP eps3', 'federated no-DP', 'federated DP (A)',
                                            'federated DP full (D)', 'dividend = D - local DP', 'privacy tax = D - fed no-DP']),
                '', 'Positive dividend: the bank does better inside the private federation than training alone under the same '
                'privacy budget. Privacy tax: what formal DP costs relative to a non-private federation.', '']

    # ------------------------------------------------------------------ privacy ledger
    p = sorted(glob.glob(os.path.join(FED, 'full_system', 'seed_*', 'privacy.json')))
    if p:
        j = load_json(p[0])
        rows = [[r['client'], r['n_train'], f"{r['fraud_rate']:.3%}", f"{r['q']:.4f}", r['sigma'], r['steps'], r['epsilon'], r['delta']]
                for r in j['per_client']]
        out += ['## 5. Privacy ledger (full system, seed 0)', '',
                write_markdown_table(rows, ['client', 'n_train', 'fraud rate', 'q = 1/ceil(n/B)', 'sigma', 'steps', 'epsilon', 'delta']), '',
                f"* DP-SGD: max client epsilon = {j['ledger']['dp_sgd_epsilon_max']:.3f} at delta = {j['ledger']['dp_sgd_delta']:g} (RDP, composed over all rounds).",
                f"* Uncertainty release (Component 2): {j['ledger']['uncertainty_release_epsilon_total']:.2f} (pure DP, Laplace, basic composition over rounds).",
                '* Not accounted (fitted on private data, stated as scope): ' + '; '.join(j['ledger']['unaccounted']) + '.', '']

    # ------------------------------------------------------------------ epsilon / alpha sweeps
    t = md_file(os.path.join(FED, 'table_epsilon_sweep.md'))
    if t:
        out += ['## 6. Privacy-budget sweep (baseline vs full system)', '', t, '',
                f'![]({rel(os.path.join(FED, "privacy_utility.png"))})', '']
    t = md_file(os.path.join(FED, 'table_alpha_sweep.md'))
    if t:
        out += ['## 7. Heterogeneity sweep (Dirichlet alpha; re-partitioned pools, epsilon = 3)', '', t, '',
                f'![]({rel(os.path.join(FED, "bars_alpha_sweep.png"))})', '']

    # ------------------------------------------------------------------ conformal
    t = md_file(os.path.join(RESULTS_ROOT, 'conformal', 'table_conformal.md'))
    if t:
        out += ['## 8. Conformal review tier', '', t, '', f'![]({rel(os.path.join(RESULTS_ROOT, "conformal", "review_tier.png"))})', '']
        cj = os.path.join(RESULTS_ROOT, 'conformal', 'conformal_results.json')
        if os.path.exists(cj):
            j = load_json(cj)
            for name, r in j.items():
                a = r.get('uncertainty_allocation', {})
                if a:
                    out += [f"*{name}* (Component-1 allocation at lambda = 0.5 on the hold-out): mean uncertainty fraud "
                            f"{a['mean_u_fraud']:.3f} vs benign {a['mean_u_benign']:.3f}; mean clip scale w fraud {a['mean_w_fraud']:.3f} "
                            f"vs benign {a['mean_w_benign']:.3f} (ratio {a['w_ratio_fraud_over_benign']:.2f}); "
                            f"borderline frauds (u > 0.5): {a['borderline_fraud_share']:.1%}."]
            out += ['']

    # ------------------------------------------------------------------ attacks
    t = md_file(os.path.join(RESULTS_ROOT, 'attacks', 'table_mia.md'))
    if t:
        out += ['## 9. Membership inference (positive control = non-private target)', '', t, '',
                f'![]({rel(os.path.join(RESULTS_ROOT, "attacks", "mia_roc_loglog.png"))})', '']
        gi = os.path.join(RESULTS_ROOT, 'attacks', 'gradient_inversion.json')
        if os.path.exists(gi):
            j = load_json(gi)
            rows = [[s['index'], s['no_protection']['mse'], s['no_protection']['pearson_r'], s['clip_only']['mse'],
                     s['clip_plus_noise']['mse'], s['clip_plus_noise']['pearson_r'], s['no_protection']['baseline_mse_random']] for s in j['samples']]
            out += ['### Gradient inversion of a single transaction\'s gradient', '',
                    write_markdown_table(rows, ['sample', 'MSE no protection', 'r', 'MSE clip only', f"MSE clip+noise (sigma={j['sigma_used']:.2f})", 'r', 'MSE random guess']), '']

    # ------------------------------------------------------------------ robustness
    t = md_file(os.path.join(FED, 'robustness', 'table_robustness.md'))
    if t:
        out += ['## 10. Robustness to one malicious bank (client 1), Component 1 on', '', t, '',
                f'![]({rel(os.path.join(FED, "robustness", "bars_model_replacement.png"))})', '']

    # ------------------------------------------------------------------ flower
    f = sorted(glob.glob(os.path.join(RESULTS_ROOT, 'federated_flower', '*', 'seed_*', 'history.json')))
    if f:
        j = load_json(f[0])
        out += ['## 11. Flower framework run', '',
                f"`{j['config']['name']}` through Flower {__import__('flwr').__version__} simulation, {len(j['history']) - 1} rounds: "
                f"final AUC-PR {j['history'][-1]['global']['auc_pr']:.4f}; per-round server log in `{rel(f[0])}`.", '']

    path = os.path.join(RESULTS_ROOT, 'RESULTS.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out) + '\n')
    print('wrote', path)


if __name__ == '__main__':
    main()
