# -*- coding: utf-8 -*-
"""
Pipeline validation harness.

Checks the invariants the preprocessing pipeline is supposed to guarantee, so
that a silent regression in any preprocessing step is caught before it reaches
an experiment. Run after every pipeline execution:

    python experiments/validate_pipeline.py

Exits non-zero if any check fails, so it can be wired into CI later.
"""
import os, sys, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

PROCESSED = os.path.join('data', 'processed_clean')
N_CLIENTS = 5
EXPECTED_DIM = 72
MIN_FRAUD_PER_CLIENT = 25

_results = []


def check(name, condition, detail=''):
    _results.append((name, bool(condition), detail))
    status = 'PASS' if condition else 'FAIL'
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    return bool(condition)


def main():
    print('=' * 72)
    print('  PIPELINE VALIDATION')
    print('=' * 72)

    holdout_path = os.path.join(PROCESSED, 'public_holdout.pkl')
    if not os.path.exists(holdout_path):
        print(f'ERROR: {holdout_path} not found. Run: python src/preprocessing/pipeline.py')
        return 1

    with open(holdout_path, 'rb') as f:
        pub = pickle.load(f)
    X_pub, y_pub = pub['X'], pub['y']

    clients = []
    for c in range(N_CLIENTS):
        p = os.path.join(PROCESSED, 'partitions', f'client_{c}.pkl')
        if not os.path.exists(p):
            print(f'ERROR: missing {p}')
            return 1
        with open(p, 'rb') as f:
            clients.append(pickle.load(f))

    print('\n-- Shape and dtype --')
    check('hold-out feature dimension is 72', X_pub.shape[1] == EXPECTED_DIM,
          f'got {X_pub.shape[1]}')
    check('hold-out features are float32', X_pub.dtype == np.float32, str(X_pub.dtype))
    dims = {d['train'][0].shape[1] for d in clients}
    check('all clients share the hold-out feature schema',
          dims == {EXPECTED_DIM}, f'client dims {sorted(dims)}')

    print('\n-- Numerical health --')
    check('no NaN in hold-out features', not np.isnan(X_pub).any())
    check('no Inf in hold-out features', not np.isinf(X_pub).any())
    bad = [c for c, d in enumerate(clients)
           if np.isnan(d['train'][0]).any() or np.isinf(d['train'][0]).any()]
    check('no NaN/Inf in any client training fold', not bad, f'offenders {bad}')

    print('\n-- Split disjointness (per client, by row identity) --')
    # Asserted on original pool indices rather than on feature-vector equality:
    # the projected pool contains a few hundred genuine duplicate transactions,
    # so comparing vectors would report false overlaps.
    have_idx = all('indices' in d for d in clients)
    check('partitions carry original row indices', have_idx)
    if have_idx:
        for c, d in enumerate(clients):
            tr = set(d['indices']['train'].tolist())
            ca = set(d['indices']['calibration'].tolist())
            te = set(d['indices']['test'].tolist())
            overlap = (tr & ca) | (tr & te) | (ca & te)
            check(f'client {c}: train/cal/test disjoint', not overlap,
                  f'{len(overlap)} shared rows')

        print('\n-- Disjointness across clients --')
        seen, collisions = set(), 0
        for d in clients:
            allidx = (set(d['indices']['train'].tolist())
                      | set(d['indices']['calibration'].tolist())
                      | set(d['indices']['test'].tolist()))
            collisions += len(seen & allidx)
            seen |= allidx
        check('no row assigned to more than one client', collisions == 0,
              f'{collisions} collisions')

    print('\n-- Class balance discipline --')
    for c, d in enumerate(clients):
        ytr, ycal, yte = d['train'][1], d['calibration'][1], d['test'][1]
        # Calibration and test must retain the natural distribution. If the
        # training fold were resampled in-pipeline its rate would diverge sharply.
        tr, cal = ytr.mean(), ycal.mean()
        ratio = tr / cal if cal > 0 else float('inf')
        check(f'client {c}: training fold not pre-resampled',
              ratio < 2.0,
              f'train {tr*100:.2f}% vs cal {cal*100:.2f}% (ratio {ratio:.2f}x)')

    print('\n-- Minority floor --')
    for c, d in enumerate(clients):
        nf = int(d['train'][1].sum())
        check(f'client {c}: >= {MIN_FRAUD_PER_CLIENT} fraud in training fold',
              nf >= MIN_FRAUD_PER_CLIENT, f'{nf} fraud')

    print('\n-- Calibration folds usable for threshold selection --')
    for c, d in enumerate(clients):
        nf = int(d['calibration'][1].sum())
        check(f'client {c}: calibration fold contains fraud', nf > 0, f'{nf} fraud')
    pooled_cal_fraud = sum(int(d['calibration'][1].sum()) for d in clients)
    check('pooled calibration fold has enough positives to fit a threshold',
          pooled_cal_fraud >= 50, f'{pooled_cal_fraud} fraud')

    print('\n-- Dataset scale --')
    total = sum(len(d['train'][1]) for d in clients)
    fraud = sum(int(d['train'][1].sum()) for d in clients)
    check('full dataset used (pooled train > 300k rows)', total > 300_000,
          f'{total:,} rows')
    check('pooled training fraud rate is realistic (2-6%)',
          0.02 <= fraud / total <= 0.06, f'{fraud/total*100:.2f}% ({fraud:,} fraud)')
    check('hold-out is roughly 10% of the dataset',
          0.08 <= len(y_pub) / (len(y_pub) + total + 0.0) <= 0.20,
          f'{len(y_pub):,} hold-out rows')

    print('\n' + '=' * 72)
    failed = [n for n, ok, _ in _results if not ok]
    print(f'  {len(_results) - len(failed)}/{len(_results)} checks passed')
    if failed:
        print('  FAILED:')
        for n in failed:
            print(f'    - {n}')
    print('=' * 72)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
