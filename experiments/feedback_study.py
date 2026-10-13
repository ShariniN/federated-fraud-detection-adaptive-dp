# -*- coding: utf-8 -*-
"""Midpoint-feedback study (2 Oct 2026): justify every pipeline and model choice by comparison.

Supervisor feedback asked for evidence that each preprocessing choice beats its
alternatives, that the split direction matters, that float32 loses nothing, that
the model ranking is not an artefact of the pipeline, and that FraudMLP is the
right model. Everything here reuses the project's own preprocessing functions,
so the `base` variant reproduces data/processed_clean exactly (checked by `prep`).

Subcommands (all resumable; finished work is skipped):
    python experiments/feedback_study.py prep                 # build every pipeline variant
    python experiments/feedback_study.py floatcheck           # float32 vs float64
    python experiments/feedback_study.py train --stream A     # MLP + LR jobs
    python experiments/feedback_study.py train --stream B     # Tabular ResNet jobs
    python experiments/feedback_study.py report               # tables -> results/feedback/

Training protocol is identical to the midpoint: pooled client training folds,
Borderline-SMOTE (ratio 0.3, k=5, seed 42) on training data only, DP-SGD via
Opacus (C=1, sigma=1, batch 512, 15 epochs, Adam 1e-3, delta 1e-5), threshold
chosen on the pooled calibration folds, scored on the hold-out. Unlike the
midpoint, every run is seeded (seeds 0, 1, 2) so variants can be compared.
"""

import argparse
import gc
import json
import os
import time

from _common import RESULTS_ROOT, Logger  # sets cwd to the project root and sys.path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing.pipeline import _classify_columns, _downcast
from src.preprocessing.imputation import (_existing, _impute_chunked, apply_imputation,
                                          impute_categorical)
from src.preprocessing.encoding import (apply_frequency_encoding, apply_ordinal_encoding,
                                        apply_target_encoding, frequency_encode,
                                        ordinal_encode, target_encode)
from src.preprocessing.transforms import (_transform_chunked, apply_transforms,
                                          power_transform, standard_scale)
from src.preprocessing.pca import apply_pca, fit_pca
from src.preprocessing.partition import dirichlet_partition
from src.evaluation.train_utils import (apply_borderline_smote, compute_all_metrics,
                                        train_model_dp)
from src.models.mlp import FraudMLP
from src.models.baselines import LogisticRegressionModel, TabularResNet
from src.utils import set_seed

OUT = os.path.join(RESULTS_ROOT, 'feedback')
CACHE = os.path.join(OUT, 'cache')
RUNS = os.path.join(OUT, 'runs')
LOG = Logger(os.path.join(OUT, 'feedback.log'))
RAW = os.path.join('data', 'raw')
SEEDS = (0, 1, 2)
DP = dict(epochs=15, lr=1e-3, batch_size=512, max_grad_norm=1.0,
          noise_multiplier=1.0, target_delta=1e-5)

BASE = dict(split='temporal', impute='median', flags=True, encode='frequency',
            scale='yj_std', pca=72)
# One factor changed at a time from BASE, plus two combinations.
VARIANTS = {
    'base':             {},
    'split_random':     dict(split='random'),
    'split_reverse':    dict(split='reverse'),
    'impute_mean':      dict(impute='mean'),
    'impute_noflags':   dict(flags=False),
    'encode_target':    dict(encode='target'),
    'encode_label':     dict(encode='label'),
    'scale_std':        dict(scale='std'),
    'scale_minmax':     dict(scale='minmax'),
    'pca_32':           dict(pca=32),
    'pca_128':          dict(pca=128),
    'pca_none':         dict(pca=None),
    'combo_naive':      dict(impute='mean', flags=False, encode='label', scale='minmax', pca=None),
    'combo_std_nopca':  dict(scale='std', pca=None),
}


# Variants on which Logistic Regression is also run: the steps that can change model ranking.
LR_VARIANTS = ('pca_none', 'pca_32', 'pca_128', 'scale_std', 'scale_minmax',
               'encode_label', 'encode_target', 'combo_naive')


def cfg_of(name):
    c = dict(BASE)
    c.update(VARIANTS[name])
    return c


# --------------------------------------------------------------------------- data

def raw_merged():
    """Merged transaction+identity frame (float32), cached as a pickle."""
    path = os.path.join(CACHE, 'raw_merged.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    LOG('loading raw CSVs...')
    df = _downcast(pd.read_csv(os.path.join(RAW, 'train_transaction.csv')))
    df_id = _downcast(pd.read_csv(os.path.join(RAW, 'train_identity.csv')))
    df = pd.merge(df, df_id, on='TransactionID', how='left')
    del df_id
    os.makedirs(CACHE, exist_ok=True)
    df.to_pickle(path)
    return df


def split_frame(df, how, holdout=0.10, seed=42):
    df = df.sort_values('TransactionDT')
    y = df['isFraud'].values.astype(np.int64)
    feats = df.drop(columns=['isFraud', 'TransactionID'])
    n = len(feats)
    n_fed = int((1.0 - holdout) * n)
    if how == 'temporal':        # oldest 90% train, newest 10% test (the project's choice)
        fed_idx, pub_idx = np.arange(n_fed), np.arange(n_fed, n)
    elif how == 'reverse':       # newest 90% train, oldest 10% test (the pre-audit bug)
        fed_idx, pub_idx = np.arange(n - n_fed, n), np.arange(0, n - n_fed)
    elif how == 'random':        # stratified random 10% test
        fed_idx, pub_idx = train_test_split(np.arange(n), test_size=n - n_fed,
                                            stratify=y, random_state=seed)
        fed_idx.sort(); pub_idx.sort()
    else:
        raise ValueError(how)
    return (feats.iloc[fed_idx].copy(), y[fed_idx], feats.iloc[pub_idx].copy(), y[pub_idx])


def impute_numerical_s(df, columns, strategy):
    cols = _existing(df, columns)
    imp = SimpleImputer(strategy=strategy).fit(df[cols])
    return _impute_chunked(df.copy(), cols, imp), imp


def label_encode(df_fed, df_pub, columns):
    """Integer codes for every category, fitted on the training pool; unseen -> -1."""
    for c in _existing(df_fed, columns):
        codes, uniques = pd.factorize(df_fed[c], sort=False)
        df_fed[c] = codes.astype(np.int32)
        lut = pd.Series(np.arange(len(uniques), dtype=np.int32), index=uniques)
        df_pub[c] = df_pub[c].map(lut).fillna(-1).astype(np.int32)
    return df_fed, df_pub


def minmax_scale(df_fed, df_pub, cols, chunk=50_000):
    sc = MinMaxScaler()
    for lo in range(0, len(df_fed), chunk):
        sc.partial_fit(df_fed[cols].iloc[lo:lo + chunk].to_numpy(dtype=np.float64))
    df_fed[cols] = _transform_chunked(df_fed, cols, sc)
    df_pub[cols] = _transform_chunked(df_pub, cols, sc)
    return df_fed, df_pub


def pre_pca_cache_key(c):
    return f"{c['split']}-{c['impute']}-{'flags' if c['flags'] else 'noflags'}-{c['encode']}-{c['scale']}"


def build_pre_pca(c):
    """Steps 1-5 of the pipeline under configuration c. Returns float32 matrices."""
    key = pre_pca_cache_key(c)
    path = os.path.join(CACHE, f'prepca_{key}.npz')
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d['X_fed'], d['y_fed'], d['X_pub'], d['y_pub'], list(d['cols'])
    LOG(f'  steps 1-5 for {key}')
    df_fed, y_fed, df_pub, y_pub = split_frame(raw_merged(), c['split'])
    gc.collect()
    categorical_cols, ordinal_cols, freq_cols, numerical_cols = _classify_columns(df_fed)

    df_fed, num_imp = impute_numerical_s(df_fed, numerical_cols, c['impute'])
    df_fed, cat_imp = impute_categorical(df_fed, categorical_cols, add_flags=c['flags'])
    df_pub = apply_imputation(df_pub, num_imp, cat_imp, numerical_cols, categorical_cols,
                              add_flags=c['flags'])
    missing_cols = [f'{x}_missing' for x in categorical_cols if f'{x}_missing' in df_fed.columns]

    if c['encode'] == 'label':
        df_fed, df_pub = label_encode(df_fed, df_pub, ordinal_cols + freq_cols)
    else:
        df_fed, maps = ordinal_encode(df_fed, ordinal_cols)
        df_pub = apply_ordinal_encoding(df_pub, ordinal_cols, maps)
        if c['encode'] == 'frequency':
            df_fed, maps = frequency_encode(df_fed, freq_cols, df_fed)
            df_pub = apply_frequency_encoding(df_pub, freq_cols, maps)
        elif c['encode'] == 'target':
            fold = df_fed[freq_cols].copy()
            fold['isFraud'] = y_fed
            df_fed, maps = target_encode(df_fed, freq_cols, 'isFraud', fold)
            df_pub = apply_target_encoding(df_pub, freq_cols, maps)
            del fold
        else:
            raise ValueError(c['encode'])
    gc.collect()

    cols = numerical_cols + missing_cols + freq_cols + ordinal_cols
    if c['scale'] == 'yj_std':
        df_fed, pt = power_transform(df_fed, cols, random_state=42)
        df_fed, sc = standard_scale(df_fed, cols)
        df_pub = apply_transforms(df_pub, cols, pt, sc)
    elif c['scale'] == 'std':
        df_fed, sc = standard_scale(df_fed, cols)
        df_pub = apply_transforms(df_pub, cols, None, sc)
    elif c['scale'] == 'minmax':
        df_fed, df_pub = minmax_scale(df_fed, df_pub, cols)
    else:
        raise ValueError(c['scale'])

    X_fed = df_fed[cols].to_numpy(dtype=np.float32)
    X_pub = df_pub[cols].to_numpy(dtype=np.float32)
    del df_fed, df_pub
    gc.collect()
    np.savez(path, X_fed=X_fed, y_fed=y_fed, X_pub=X_pub, y_pub=y_pub, cols=np.array(cols))
    return X_fed, y_fed, X_pub, y_pub, cols


def variant_path(name):
    return os.path.join(CACHE, f'variant_{name}.npz')


def pool_partitions(X_fed, y_fed):
    parts = dirichlet_partition(X_fed, y_fed, 5, alpha=0.5, random_state=42,
                                min_samples_per_client=25, verbose=False)
    cat = lambda split, i: np.concatenate([p[split][i] for p in parts])
    return cat('train', 0), cat('train', 1), cat('calibration', 0), cat('calibration', 1)


def build_variant(name):
    path = variant_path(name)
    if os.path.exists(path):
        return
    c = cfg_of(name)
    t0 = time.time()
    LOG(f'prep {name}: {c}')
    X_fed, y_fed, X_pub, y_pub, cols = build_pre_pca(c)
    var = None
    if c['pca'] is not None:
        frame = pd.DataFrame(X_fed, columns=cols)
        pca, feats = fit_pca(frame, n_components=c['pca'], columns=cols, random_state=42)
        var = float(np.sum(pca.explained_variance_ratio_))
        X_fed = apply_pca(frame, pca, feats).to_numpy(dtype=np.float32)
        del frame
        X_pub = apply_pca(pd.DataFrame(X_pub, columns=cols), pca, feats).to_numpy(dtype=np.float32)
    Xtr, ytr, Xc, yc = pool_partitions(X_fed, y_fed)
    tmp = path[:-4] + '.tmp.npz'
    np.savez(tmp, X_train=Xtr, y_train=ytr, X_cal=Xc, y_cal=yc, X_test=X_pub, y_test=y_pub,
             dim=X_pub.shape[1], pca_variance=-1.0 if var is None else var)
    os.replace(tmp, path)
    LOG(f'prep {name}: dim={X_pub.shape[1]} train={len(ytr)} cal={len(yc)} test={len(y_pub)} '
        f'test-fraud={y_pub.mean():.4f} pca_var={var} ({time.time() - t0:.0f}s)')


def check_base_matches_official():
    """The base variant must reproduce the official processed data exactly."""
    from src.evaluation.train_utils import load_real_data
    X, y, Xc, yc, Xt, yt, _ = load_real_data('data/processed_clean')
    d = np.load(variant_path('base'))
    same = (np.array_equal(d['y_train'], y) and np.array_equal(d['y_test'], yt)
            and np.array_equal(d['y_cal'], yc))
    diff = max(float(np.max(np.abs(d['X_train'] - X))), float(np.max(np.abs(d['X_test'] - Xt))))
    LOG(f'base vs official processed_clean: labels identical={same}, max |feature diff|={diff:.3g}')
    return same, diff


# --------------------------------------------------------------------------- training

def build_model(name, dim):
    return {'mlp': lambda: FraudMLP(input_dim=dim),
            'lr': lambda: LogisticRegressionModel(input_dim=dim),
            'resnet': lambda: TabularResNet(input_dim=dim)}[name]()


def resampled_train(name, how):
    path = os.path.join(CACHE, f'train_{name}_{how}.npz')
    if os.path.exists(path):
        d = np.load(path)
        return d['X'], d['y']
    d = np.load(variant_path(name))
    X, y = d['X_train'], d['y_train']
    if how == 'borderline':
        X, y = apply_borderline_smote(X, y)                     # ratio 0.3, seed 42
    elif how in ('smote', 'adasyn'):
        from imblearn.over_sampling import ADASYN, SMOTE
        s = SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=42) if how == 'smote' \
            else ADASYN(sampling_strategy=0.3, n_neighbors=5, random_state=42)
        X, y = s.fit_resample(X, y)
    elif how != 'none':
        raise ValueError(how)
    X = X.astype(np.float32)
    np.savez(path, X=X, y=y)
    return X, y


def train_nonprivate(model, X, y, seed, epochs=15, lr=1e-3, batch_size=512):
    """Same loop as train_model_dp without Opacus (no clipping, no noise)."""
    n_fraud = int(y.sum())
    w = torch.tensor([1.0, (len(y) - n_fraud) / max(n_fraud, 1)], dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long)),
                    batch_size=batch_size, shuffle=True, generator=g)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    return model


def job_id(j):
    return f"{j['variant']}__{j['model']}__{j['resample']}__{'dp' if j['private'] else 'nodp'}__s{j['seed']}"


def run_job(j):
    out = os.path.join(RUNS, job_id(j) + '.json')
    if os.path.exists(out):
        return
    while not os.path.exists(variant_path(j['variant'])):   # stream B can overtake `prep`
        time.sleep(60)
    time.sleep(2)
    d = np.load(variant_path(j['variant']))
    X, y = resampled_train(j['variant'], j['resample'])
    set_seed(j['seed'])
    model = build_model(j['model'], int(d['dim']))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t0 = time.time()
    if j['private']:
        model, eps = train_model_dp(model, X, y, verbose=False, **DP)
    else:
        model, eps = train_nonprivate(model, X, y, j['seed']), float('inf')
    m = compute_all_metrics(model, d['X_test'], d['y_test'], X_cal=d['X_cal'], y_cal=d['y_cal'])
    rec = dict(j, params=params, dim=int(d['dim']), n_train=int(len(y)), epsilon=eps,
               time_s=round(time.time() - t0, 1),
               **{k: float(m[k]) for k in ('auc_pr', 'auc_roc', 'f1_default', 'f1_calibrated',
                                           'recall_calibrated', 'precision_calibrated',
                                           'best_tau', 'threshold_optimism')},
               confusion_matrix=np.asarray(m['confusion_matrix']).tolist())
    os.makedirs(RUNS, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(rec, f, indent=1)
    LOG(f"{job_id(j)}: AUC-PR {rec['auc_pr']:.4f} AUC-ROC {rec['auc_roc']:.4f} "
        f"F1 {rec['f1_calibrated']:.4f} eps {eps if eps == float('inf') else round(eps, 3)} "
        f"params {params} ({rec['time_s']}s)")


def jobs(stream):
    J = lambda v, m, r='borderline', p=True, s=0: dict(variant=v, model=m, resample=r, private=p, seed=s)
    if stream == 'A':
        # Most important first: FraudMLP on every variant (points 3-6), then resampling,
        # then Logistic Regression on the variants that test pipeline-model interaction (point 7).
        out = [J('base', m, s=s) for s in SEEDS for m in ('mlp', 'lr')]
        out += [J('base', m, p=False, s=s) for s in SEEDS for m in ('mlp', 'lr')]
        out += [J(v, 'mlp', s=s) for v in VARIANTS if v != 'base' for s in SEEDS]
        out += [J('base', 'mlp', r, s=s) for r in ('none', 'smote', 'adasyn') for s in SEEDS]
        out += [J(v, 'lr', s=s) for v in LR_VARIANTS for s in SEEDS]
        return out
    if stream == 'B':
        out = [J('base', 'resnet', s=s) for s in SEEDS]
        out += [J('base', 'resnet', p=False, s=s) for s in SEEDS]
        out += [J(v, 'resnet') for v in ('pca_none', 'pca_32', 'scale_minmax', 'encode_label',
                                          'pca_128', 'scale_std', 'split_random')]
        return out
    raise ValueError(stream)


# --------------------------------------------------------------------------- float32 check

def float_check():
    """(a) Does casting the raw data to float32 merge or distort any values? (full data)
    (b) Does running the whole pipeline in float32 instead of float64 change the
        features or the model's accuracy? (25% subsample: float64 does not fit in RAM at full size)"""
    out_path = os.path.join(OUT, 'float_check.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}

    if 'raw' not in res:
        LOG('float check (a): raw float64 -> float32 on all 590,540 rows')
        rows = []
        for fname in ('train_transaction.csv', 'train_identity.csv'):
            df = pd.read_csv(os.path.join(RAW, fname))
            for c in df.columns:
                if df[c].dtype != np.float64:
                    continue
                v = df[c].to_numpy()
                v = v[~np.isnan(v)]
                if len(v) == 0:
                    continue
                v32 = v.astype(np.float32).astype(np.float64)
                err = np.abs(v32 - v)
                nz = np.abs(v) > 0
                rows.append(dict(column=c, max_abs=float(np.abs(v).max()),
                                 max_abs_err=float(err.max()),
                                 max_rel_err=float((err[nz] / np.abs(v[nz])).max()) if nz.any() else 0.0,
                                 distinct64=int(len(np.unique(v))),
                                 distinct32=int(len(np.unique(v32)))))
            del df
            gc.collect()
        merged = [r['column'] for r in rows if r['distinct32'] < r['distinct64']]
        amt = next(r for r in rows if r['column'] == 'TransactionAmt')
        res['raw'] = dict(float_columns=len(rows), columns_with_merged_values=merged,
                          max_rel_err=max(r['max_rel_err'] for r in rows),
                          transaction_amt=amt, per_column=rows)
        json.dump(res, open(out_path, 'w'), indent=1)
        LOG(f"  {len(rows)} float columns; columns where float32 merges distinct values: {merged}; "
            f"max relative error {res['raw']['max_rel_err']:.2e}; TransactionAmt max abs error "
            f"{amt['max_abs_err']:.2e}")

    if 'pipeline' not in res:
        LOG('float check (b): identical pipeline in float32 vs float64 on a 25% subsample')
        tx = pd.read_csv(os.path.join(RAW, 'train_transaction.csv'))      # float64, as on disk
        tx = tx.sample(frac=0.25, random_state=42)
        df64 = pd.merge(tx, pd.read_csv(os.path.join(RAW, 'train_identity.csv')),
                        on='TransactionID', how='left')
        del tx; gc.collect()
        df32 = _downcast(df64.copy())                                     # the project's load step
        feats = {'float32': mini_pipeline(df32, np.float32), 'float64': mini_pipeline(df64, np.float64)}
        df = df64
        a, b = feats['float32'], feats['float64']
        Xa, Xb = a[0].astype(np.float64), b[0]
        diff = float(np.max(np.abs(Xa - Xb)))
        sign = np.sign(np.sum(Xa * Xb, axis=0))                 # PCA axes may flip sign
        Xa_aligned = Xa * sign
        diff_aligned = float(np.max(np.abs(Xa_aligned - Xb)))
        corr = float(np.min([np.corrcoef(Xa_aligned[:, i], Xb[:, i])[0, 1] for i in range(Xb.shape[1])]))
        res['pipeline'] = dict(rows=len(df), max_abs_feature_diff_after_sign_alignment=diff_aligned,
                               max_abs_feature_diff_raw=diff, min_component_correlation=corr, runs={})
        for dtname, (Xf, yf, Xp, yp) in feats.items():
            Xtr, ytr, Xc, yc = pool_partitions(Xf.astype(np.float32), yf)
            Xs, ys = apply_borderline_smote(Xtr, ytr)
            for s in SEEDS:
                set_seed(s)
                mdl, eps = train_model_dp(FraudMLP(input_dim=Xf.shape[1]), Xs, ys, verbose=False, **DP)
                m = compute_all_metrics(mdl, Xp.astype(np.float32), yp, X_cal=Xc, y_cal=yc)
                res['pipeline']['runs'].setdefault(dtname, []).append(
                    dict(seed=s, auc_pr=float(m['auc_pr']), auc_roc=float(m['auc_roc']),
                         f1=float(m['f1_calibrated']), epsilon=float(eps)))
        json.dump(res, open(out_path, 'w'), indent=1)
        r = res['pipeline']
        LOG(f"  max feature diff {diff_aligned:.2e}, min component correlation {corr:.6f}; "
            f"AUC-PR f32 {[round(x['auc_pr'], 4) for x in r['runs']['float32']]} "
            f"f64 {[round(x['auc_pr'], 4) for x in r['runs']['float64']]}")


def mini_pipeline(df, dtype):
    """The project's steps 2-6 with the storage precision as a parameter.

    Mirrors the real pipeline: each transform is computed in float64 and its output
    stored in `dtype` before the next step (the project stores float32), and PCA is
    fitted on the stored matrix. The float32 and float64 calls differ only in `dtype`."""
    df = df.sort_values('TransactionDT')
    y = df['isFraud'].values.astype(np.int64)
    feats = df.drop(columns=['isFraud', 'TransactionID'])
    n_fed = int(0.9 * len(feats))
    fed, pub = feats.iloc[:n_fed], feats.iloc[n_fed:]
    cat_cols, ord_cols, freq_cols, num_cols = _classify_columns(fed)

    num_imp = SimpleImputer(strategy='median').fit(fed[num_cols].to_numpy(dtype=np.float64))
    cat_imp = SimpleImputer(strategy='most_frequent').fit(fed[cat_cols].astype(object))
    fed_cat = pd.DataFrame(cat_imp.transform(fed[cat_cols].astype(object)), columns=cat_cols)
    ord_maps = {c: {v: i for i, v in enumerate(sorted(fed_cat[c].unique(), key=str))} for c in ord_cols}
    freq_maps = {c: fed_cat[c].value_counts(normalize=True) for c in freq_cols}

    def assemble(part):
        X_num = num_imp.transform(part[num_cols].to_numpy(dtype=np.float64))
        flags = part[cat_cols].isna().to_numpy().astype(np.float64)
        cat = pd.DataFrame(cat_imp.transform(part[cat_cols].astype(object)), columns=cat_cols)
        X_freq = np.column_stack([cat[c].map(freq_maps[c]).fillna(0.0).to_numpy(dtype=np.float64)
                                  for c in freq_cols])
        X_ord = np.column_stack([cat[c].map(ord_maps[c]).fillna(-1).to_numpy(dtype=np.float64)
                                 for c in ord_cols])
        return np.hstack([X_num, flags, X_freq, X_ord]).astype(dtype)

    Xf, Xp = assemble(fed), assemble(pub)
    rs = np.random.RandomState(42)
    fit_rows = rs.choice(len(Xf), size=min(100_000, len(Xf)), replace=False)
    pt = PowerTransformer(method='yeo-johnson', standardize=False).fit(Xf[fit_rows].astype(np.float64))
    def guard(X):   # same overflow guard as transforms._transform_chunked in the real pipeline
        X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
        return np.clip(X, -1e9, 1e9)
    Xf = guard(pt.transform(Xf.astype(np.float64))).astype(dtype)
    Xp = guard(pt.transform(Xp.astype(np.float64))).astype(dtype)
    sc = StandardScaler().fit(Xf.astype(np.float64))
    Xf = np.nan_to_num(sc.transform(Xf.astype(np.float64))).astype(dtype)
    Xp = np.nan_to_num(sc.transform(Xp.astype(np.float64))).astype(dtype)
    pca = PCA(n_components=72, svd_solver='randomized', random_state=42).fit(Xf)
    return (pca.transform(Xf).astype(dtype), y[:n_fed], pca.transform(Xp).astype(dtype), y[n_fed:])


# --------------------------------------------------------------------------- report

def report():
    """Compile finished runs into results/feedback/FEEDBACK_TABLES.md and feedback_summary.csv."""
    import glob
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RUNS, '*.json')))]
    if not recs:
        print('no runs yet'); return
    df = pd.DataFrame(recs)
    df['privacy'] = np.where(df['private'], 'DP', 'non-private')
    keys = ['variant', 'model', 'resample', 'privacy']
    agg = df.groupby(keys).agg(
        n=('seed', 'count'), auc_pr=('auc_pr', 'mean'), auc_pr_sd=('auc_pr', 'std'),
        auc_roc=('auc_roc', 'mean'), auc_roc_sd=('auc_roc', 'std'), f1=('f1_calibrated', 'mean'),
        recall=('recall_calibrated', 'mean'), precision=('precision_calibrated', 'mean'),
        params=('params', 'first'), dim=('dim', 'first'), epsilon=('epsilon', 'first'),
        time_s=('time_s', 'mean')).reset_index()
    agg.to_csv(os.path.join(OUT, 'feedback_summary.csv'), index=False)

    def row(v, m='mlp', r='borderline', priv='DP'):
        x = agg[(agg['variant'] == v) & (agg['model'] == m) & (agg['resample'] == r) & (agg['privacy'] == priv)]
        return None if x.empty else x.iloc[0]

    def fmt(mean, sd, n):
        if mean is None or pd.isna(mean):
            return 'pending'
        txt = f'{mean:.3f}'
        if n > 1 and not pd.isna(sd):
            txt += f' ± {sd:.3f}'
        if n < 3:
            txt += f' (n={n})'
        return txt

    names = {'mlp': 'FraudMLP', 'lr': 'Logistic Regression', 'resnet': 'Tabular ResNet'}

    def table(title, entries, models=('mlp',)):
        hdr = ['Option'] + [f'{names[M]} AUC-PR' for M in models] + \
              ['FraudMLP AUC-ROC', 'FraudMLP F1', 'Δ AUC-PR vs current (FraudMLP)']
        lines = [f'### {title}', '', '| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
        b = row('base')
        for label, v, r in entries:
            cells = [label]
            for M in models:
                x = row(v, M, r)
                cells.append('pending' if x is None else fmt(x.auc_pr, x.auc_pr_sd, x.n))
            x = row(v, 'mlp', r)
            cells.append('pending' if x is None else fmt(x.auc_roc, x.auc_roc_sd, x.n))
            cells.append('pending' if x is None else f'{x.f1:.3f}')
            if (v, r) == ('base', 'borderline'):
                cells.append('—')
            elif x is None or b is None:
                cells.append('pending')
            else:
                cells.append(f'{x.auc_pr - b.auc_pr:+.3f}')
            lines.append('| ' + ' | '.join(cells) + ' |')
        return '\n'.join(lines) + '\n'

    out = ['# Feedback study — result tables', '',
           'Mean ± standard deviation over seeds 0, 1, 2 (n is shown when fewer than 3 seeds have finished). '
           'DP-SGD with ε = 0.631 (PRV accountant) unless stated. "Current" = the midpoint pipeline. '
           'Threshold chosen on calibration data; scored on the hold-out.', '']

    lines = ['### Model comparison (current pipeline)', '',
             '| Model | Parameters | DP AUC-PR | DP AUC-ROC | DP F1 | Non-private AUC-PR | Cost of DP (AUC-PR) | Training time (DP) |',
             '|---|---|---|---|---|---|---|---|']
    for M in ('mlp', 'resnet', 'lr'):
        d, nd = row('base', M), row('base', M, priv='non-private')
        lines.append('| ' + ' | '.join([
            names[M], '' if d is None else f'{int(d.params):,}',
            'pending' if d is None else fmt(d.auc_pr, d.auc_pr_sd, d.n),
            'pending' if d is None else fmt(d.auc_roc, d.auc_roc_sd, d.n),
            'pending' if d is None else f'{d.f1:.3f}',
            'pending' if nd is None else fmt(nd.auc_pr, nd.auc_pr_sd, nd.n),
            'pending' if (d is None or nd is None) else f'{d.auc_pr - nd.auc_pr:+.3f}',
            'pending' if d is None else f'{d.time_s / 60:.1f} min']) + ' |')
    out.append('\n'.join(lines) + '\n')

    out.append(table('Split / sorting (point 4)', [
        ('Temporal: oldest 90% train, newest 10% test (current)', 'base', 'borderline'),
        ('Random stratified 10% test', 'split_random', 'borderline'),
        ('Reversed: newest 90% train, oldest 10% test', 'split_reverse', 'borderline')]))
    out.append(table('Imputation (point 5)', [
        ('Median + missing indicators (current)', 'base', 'borderline'),
        ('Mean + missing indicators', 'impute_mean', 'borderline'),
        ('Median, no missing indicators', 'impute_noflags', 'borderline')]))
    out.append(table('Encoding (points 5 and 7)', [
        ('Frequency (current)', 'base', 'borderline'),
        ('Target encoding (reads the labels)', 'encode_target', 'borderline'),
        ('Label / integer codes', 'encode_label', 'borderline')], models=('mlp', 'lr', 'resnet')))
    out.append(table('Scaling (points 5 and 7)', [
        ('Yeo-Johnson + standardisation (current)', 'base', 'borderline'),
        ('Standardisation only', 'scale_std', 'borderline'),
        ('Min-max', 'scale_minmax', 'borderline')], models=('mlp', 'lr', 'resnet')))
    out.append(table('Dimensionality (points 5 and 7)', [
        ('PCA 32', 'pca_32', 'borderline'), ('PCA 72 (current)', 'base', 'borderline'),
        ('PCA 128', 'pca_128', 'borderline'), ('No PCA (480 features)', 'pca_none', 'borderline')],
        models=('mlp', 'lr', 'resnet')))
    out.append(table('Class balancing (point 5)', [
        ('None (class weights only)', 'base', 'none'), ('SMOTE', 'base', 'smote'),
        ('Borderline-SMOTE (current)', 'base', 'borderline'), ('ADASYN', 'base', 'adasyn')]))
    out.append(table('Combinations (point 6)', [
        ('Current pipeline', 'base', 'borderline'),
        ('Naive: mean, no indicators, label codes, min-max, no PCA', 'combo_naive', 'borderline'),
        ('Standardisation only + no PCA', 'combo_std_nopca', 'borderline')], models=('mlp', 'lr')))

    lines = ['### Does the model ranking depend on the pipeline? (point 7)', '',
             '| Pipeline | FraudMLP | Logistic Regression | Tabular ResNet | Highest AUC-PR |', '|---|---|---|---|---|']
    for v in ('base', 'pca_32', 'pca_128', 'pca_none', 'scale_std', 'scale_minmax',
              'encode_label', 'encode_target', 'combo_naive'):
        cells, best = [v], {}
        for M in ('mlp', 'lr', 'resnet'):
            x = row(v, M)
            cells.append('—' if x is None else fmt(x.auc_pr, x.auc_pr_sd, x.n))
            if x is not None:
                best[M] = x.auc_pr
        cells.append(names[max(best, key=best.get)] if len(best) >= 2 else '—')
        lines.append('| ' + ' | '.join(cells) + ' |')
    out.append('\n'.join(lines) + '\n')

    fc = os.path.join(OUT, 'float_check.json')
    if os.path.exists(fc):
        f = json.load(open(fc))
        lines = ['### float32 vs float64 (point 3)', '']
        if 'raw' in f:
            r = f['raw']
            lines.append(f"- Raw data, all 590,540 rows: {r['float_columns']} decimal columns; columns where float32 "
                         f"merges two different values: {len(r['columns_with_merged_values'])}; largest relative error "
                         f"{r['max_rel_err']:.1e}; largest payment-amount error {r['transaction_amt']['max_abs_err']:.1e}.")
        if 'pipeline' in f:
            p = f['pipeline']
            m32 = [x['auc_pr'] for x in p['runs'].get('float32', [])]
            m64 = [x['auc_pr'] for x in p['runs'].get('float64', [])]
            lines.append(f"- Identical pipeline on a 25% subsample ({p['rows']:,} rows): largest feature difference "
                         f"{p['max_abs_feature_diff_after_sign_alignment']:.1e}; lowest component correlation "
                         f"{p['min_component_correlation']:.6f}; FraudMLP AUC-PR float32 "
                         f"{np.mean(m32):.4f} ± {np.std(m32, ddof=1):.4f} vs float64 {np.mean(m64):.4f} ± {np.std(m64, ddof=1):.4f}.")
        out.append('\n'.join(lines) + '\n')

    text = '\n'.join(out)
    with open(os.path.join(OUT, 'FEEDBACK_TABLES.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(text)
    print(f'{len(df)} runs compiled')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prep', 'floatcheck', 'train', 'report'])
    ap.add_argument('--stream', choices=['A', 'B'])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == 'prep':
        for name in (a.only or VARIANTS):
            build_variant(name)
            gc.collect()
        check_base_matches_official()
    elif a.cmd == 'floatcheck':
        float_check()
    elif a.cmd == 'train':
        todo = jobs(a.stream)
        LOG(f'stream {a.stream}: {len(todo)} jobs')
        for j in todo:
            run_job(j)
            gc.collect()
        LOG(f'stream {a.stream}: done')
    else:
        report()


if __name__ == '__main__':
    main()
