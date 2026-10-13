# -*- coding: utf-8 -*-
"""Empirical privacy validation: membership inference with a positive control, and gradient inversion.

MEMBERSHIP INFERENCE
  The pooled training folds are split at random into IN (used to train each
  target model) and OUT (never seen). Targets: non-private, DP at epsilon in
  {8, 3, 1}. Attacks: Yeom loss attack and offline LiRA (K shadow models on
  the same pool). Reported: AUC, TPR at 0.1 % and 1 % FPR, log-log ROC.
  The non-private target is the positive control - if the attack does not
  work there, a failure on the DP models proves nothing.

GRADIENT INVERSION
  A single fraud transaction's gradient at the non-private model is inverted
  (iDLG-style) with (a) no protection, (b) clipping only, (c) clipping + the
  project's Gaussian noise. Reconstruction MSE and correlation are reported.

    python experiments/run_attacks.py --epsilons 8 3 1 --shadows 8
"""
import argparse
import os
import time

import numpy as np
import torch
from opacus.accountants import RDPAccountant

from _common import Logger, save_json, write_markdown_table, RESULTS_ROOT
from src.utils import set_seed, load_partitions, load_holdout, pooled, torch_generator
from src.models.mlp import FraudMLP
from src.training import train_epochs
from src.dp.accountant import calibrate_sigma
from src.evaluation.metrics import evaluate_model
from src.attacks.mia import yeom_loss_attack, OfflineLiRA
from src.attacks.gradient_inversion import sample_gradient, invert, reconstruction_error
from src.evaluation.plots import plot_roc_loglog, plot_bars

OUT = os.path.join(RESULTS_ROOT, 'attacks')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epsilons', type=float, nargs='+', default=[8.0, 3.0, 1.0])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--shadows', type=int, default=8)
    ap.add_argument('--shadow-epochs', type=int, default=5)
    ap.add_argument('--n-eval', type=int, default=20000, help='members and non-members each')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--delta', type=float, default=1e-5)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log = Logger(os.path.join(OUT, 'attacks.log'))
    set_seed(args.seed)

    parts = load_partitions(); X_ho, y_ho = load_holdout()
    X_pool, y_pool = pooled(parts, 'train'); X_cal, y_cal = pooled(parts, 'calibration')
    rng = np.random.default_rng(args.seed)
    in_mask = rng.random(len(y_pool)) < 0.5
    X_in, y_in, X_out, y_out = X_pool[in_mask], y_pool[in_mask], X_pool[~in_mask], y_pool[~in_mask]
    idx_in_all = np.where(in_mask)[0]; idx_out_all = np.where(~in_mask)[0]
    log(f'pool {len(y_pool):,}: IN {len(y_in):,} / OUT {len(y_out):,}')

    # evaluation subsets (balanced by count; keep the natural fraud mix)
    sel_in = rng.choice(len(y_in), size=min(args.n_eval, len(y_in)), replace=False)
    sel_out = rng.choice(len(y_out), size=min(args.n_eval, len(y_out)), replace=False)
    Xe_in, ye_in, Xe_out, ye_out = X_in[sel_in], y_in[sel_in], X_out[sel_out], y_out[sel_out]

    # ---------------------------------------------------------------- targets
    targets = [('nonprivate', None)] + [(f'dp_eps{e:g}', e) for e in args.epsilons]
    models = {}
    for name, eps in targets:
        t0 = time.time()
        set_seed(args.seed)
        m = FraudMLP(input_dim=X_in.shape[1]); acc = None; sigma = None
        if eps is not None:
            acc = RDPAccountant(); sigma = calibrate_sigma(eps, args.delta, len(y_in), args.batch, args.epochs)
        train_epochs(m, X_in, y_in, epochs=args.epochs, batch_size=args.batch, lr=1e-3, sigma=sigma,
                     accountant=acc, generator=torch_generator(args.seed), device='cpu')
        util = evaluate_model(m, X_ho, y_ho, X_cal, y_cal)
        models[name] = {'model': m, 'epsilon': float(acc.get_epsilon(args.delta)) if acc else float('inf'),
                        'sigma': sigma, 'auc_pr': util['auc_pr']}
        log(f"target {name}: eps={models[name]['epsilon']:.3g} hold-out AUC-PR={util['auc_pr']:.4f} ({time.time()-t0:.0f}s)")

    # ---------------------------------------------------------------- shadows
    t0 = time.time()
    lira = OfflineLiRA(input_dim=X_pool.shape[1], n_shadow=args.shadows, epochs=args.shadow_epochs,
                       batch_size=args.batch, seed=args.seed).fit_shadows(X_pool, y_pool, log=log)
    log(f'{args.shadows} shadow models in {time.time()-t0:.0f}s')

    # ---------------------------------------------------------------- attacks
    results = {}
    roc_curves = {}
    for name, info in models.items():
        m = info['model']
        yeom = yeom_loss_attack(m, Xe_in, ye_in, Xe_out, ye_out)
        lr = lira.attack(m, Xe_in, ye_in, Xe_out, ye_out, idx_in=idx_in_all[sel_in], idx_out=idx_out_all[sel_out])
        # fraud-only subset: the minority records SMOTE would have amplified are the interesting ones
        fi, fo = ye_in == 1, ye_out == 1
        lr_fraud = lira.attack(m, Xe_in[fi], ye_in[fi], Xe_out[fo], ye_out[fo],
                               idx_in=idx_in_all[sel_in][fi], idx_out=idx_out_all[sel_out][fo])
        results[name] = {'epsilon': info['epsilon'], 'sigma': info['sigma'], 'auc_pr': info['auc_pr'],
                         'yeom': yeom, 'lira': lr, 'lira_fraud_only': lr_fraud}
        roc_curves[f"{name} LiRA (eps={info['epsilon']:.3g})"] = (lr['roc']['fpr'], lr['roc']['tpr'])
        log(f"{name}: Yeom AUC={yeom['auc']:.4f} TPR@1%={yeom['tpr_at_fpr_0.01']:.4f} | LiRA AUC={lr['auc']:.4f} "
            f"TPR@0.1%={lr['tpr_at_fpr_0.001']:.4f} TPR@1%={lr['tpr_at_fpr_0.01']:.4f} | fraud-only LiRA AUC={lr_fraud['auc']:.4f}")
        torch.save(m.state_dict(), os.path.join(OUT, f'target_{name}.pt'))
    save_json({'settings': vars(args), 'results': results}, os.path.join(OUT, 'mia_results.json'))
    plot_roc_loglog(roc_curves, os.path.join(OUT, 'mia_roc_loglog.png'))
    rows = [[n, r['epsilon'], r['auc_pr'], r['yeom']['auc'], r['yeom']['tpr_at_fpr_0.01'], r['lira']['auc'],
             r['lira']['tpr_at_fpr_0.001'], r['lira']['tpr_at_fpr_0.01'], r['lira_fraud_only']['auc']] for n, r in results.items()]
    table = write_markdown_table(rows, ['target', 'epsilon', 'AUC-PR', 'Yeom AUC', 'Yeom TPR@1%FPR', 'LiRA AUC',
                                        'LiRA TPR@0.1%FPR', 'LiRA TPR@1%FPR', 'LiRA AUC (fraud only)'],
                                 os.path.join(OUT, 'table_mia.md'))
    log('\n' + table)
    plot_bars([f"{n}\n(eps={r['epsilon']:.3g})" for n, r in results.items()], [r['lira']['auc'] for r in results.values()],
              [0] * len(results), os.path.join(OUT, 'mia_auc.png'), ylabel='LiRA attack AUC',
              title='Membership inference vs privacy budget (0.5 = no leakage)', baseline=0.5)

    # ---------------------------------------------------------------- gradient inversion
    m = models['nonprivate']['model']
    fraud_idx = np.where(y_in == 1)[0][:3]
    inv = []
    gen = torch.Generator().manual_seed(args.seed)
    sigma_ref = models.get(f'dp_eps{args.epsilons[1]:g}', models[list(models)[-1]])['sigma'] or 1.0
    for i in fraud_idx:
        x = torch.as_tensor(X_in[i:i + 1]); y = torch.as_tensor(y_in[i:i + 1])
        row = {'index': int(i)}
        for cond, clip, sig in (('no_protection', None, None), ('clip_only', 1.0, None), ('clip_plus_noise', 1.0, float(sigma_ref))):
            g = sample_gradient(m, x, y, clip=clip, sigma=sig, generator=gen)
            x_hat, hist = invert(m, g, y, input_dim=X_in.shape[1], n_iter=300, lr=0.3, seed=args.seed)
            row[cond] = reconstruction_error(x, x_hat) | {'final_match_loss': hist[-1]}
        inv.append(row)
        log(f"inversion sample {i}: MSE no-protection={row['no_protection']['mse']:.3f} (r={row['no_protection']['pearson_r']:.2f}) "
            f"clip={row['clip_only']['mse']:.3f} clip+noise(sigma={sigma_ref:.2f})={row['clip_plus_noise']['mse']:.3f} "
            f"(r={row['clip_plus_noise']['pearson_r']:.2f}) random-baseline={row['no_protection']['baseline_mse_random']:.3f}")
    save_json({'sigma_used': sigma_ref, 'samples': inv}, os.path.join(OUT, 'gradient_inversion.json'))
    log('done')


if __name__ == '__main__':
    main()
