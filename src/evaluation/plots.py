"""Figures for the final-phase experiments (matplotlib only)."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PALETTE = ['#1f4e79', '#2e8b57', '#d2691e', '#c0392b', '#7d3c98', '#5d6d7e', '#117a65', '#b7950b']


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('  figure saved:', path)


def plot_convergence(curves, path, title='Hold-out AUC-PR per round', ylabel='AUC-PR'):
    """curves: {name: list of per-seed lists of (round, value)}. Draws mean and +/- sd band."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, seeds) in enumerate(curves.items()):
        rounds = [r for r, _ in seeds[0]]
        vals = np.array([[v for _, v in s] for s in seeds], dtype=float)
        mean, sd = vals.mean(0), vals.std(0)
        c = PALETTE[i % len(PALETTE)]
        ax.plot(rounds, mean, marker='o', ms=3, lw=2, color=c, label=f'{name} (n={len(seeds)})')
        if len(seeds) > 1:
            ax.fill_between(rounds, mean - sd, mean + sd, color=c, alpha=0.15)
    ax.set_xlabel('Communication round'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, path)


def plot_bars(labels, means, sds, path, ylabel='AUC-PR', title='', baseline=None, colors=None):
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 5))
    x = np.arange(len(labels))
    cols = colors or [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    ax.bar(x, means, yerr=sds, capsize=4, color=cols, alpha=0.9)
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.005, f'{m:.3f}', ha='center', va='bottom', fontsize=8)
    if baseline is not None:
        ax.axhline(baseline, color='grey', ls='--', lw=1, label='random (prevalence)')
        ax.legend(fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis='y', alpha=0.3)
    _save(fig, path)


def plot_privacy_utility(series, path, title='Privacy-utility trade-off'):
    """series: {system: [(epsilon, mean, sd), ...]}; epsilon may be inf (plotted at far right)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    finite = [e for s in series.values() for e, _, _ in s if np.isfinite(e)]
    x_inf = (max(finite) * 3) if finite else 100
    for i, (name, pts) in enumerate(series.items()):
        pts = sorted(pts, key=lambda p: p[0])
        xs = [x_inf if not np.isfinite(e) else e for e, _, _ in pts]
        ys = [m for _, m, _ in pts]; es = [s for _, _, s in pts]
        ax.errorbar(xs, ys, yerr=es, marker='o', lw=2, capsize=3, color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xscale('log')
    ticks = sorted(set(finite)) + ([x_inf] if any(not np.isfinite(e) for s in series.values() for e, _, _ in s) else [])
    ax.set_xticks(ticks); ax.set_xticklabels([('inf' if t == x_inf else f'{t:g}') for t in ticks])
    ax.set_xlabel('epsilon (delta = 1e-5)'); ax.set_ylabel('Hold-out AUC-PR'); ax.set_title(title)
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
    _save(fig, path)


def plot_weights_vs_size(weights_by_round, n_list, path, rho=None):
    """weights_by_round: [R, K] array of aggregation weights. Scatter of final-round weight vs n_k/N."""
    w = np.asarray(weights_by_round, dtype=float)
    n = np.asarray(n_list, dtype=float); base = n / n.sum()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for k in range(w.shape[1]):
        ax.plot(range(1, w.shape[0] + 1), w[:, k], marker='.', label=f'client {k} (n={int(n[k]):,})',
                color=PALETTE[k % len(PALETTE)])
        ax.axhline(base[k], color=PALETTE[k % len(PALETTE)], ls=':', lw=1)
    ax.set_xlabel('round'); ax.set_ylabel('aggregation weight'); ax.set_title('UW-FedAvg weights (dotted = FedAvg n_k/N)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.scatter(base, w.mean(0), s=80, color=PALETTE[:w.shape[1]])
    for k in range(w.shape[1]):
        ax.annotate(f'c{k}', (base[k], w.mean(0)[k]), textcoords='offset points', xytext=(5, 4))
    lim = max(base.max(), w.mean(0).max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='weight = n_k/N')
    ax.set_xlabel('n_k / N'); ax.set_ylabel('mean weight over rounds')
    ax.set_title('Is UW-FedAvg just FedAvg?' + (f'  Spearman rho = {rho:.2f}' if rho is not None else ''))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, path)


def plot_roc_loglog(curves, path, title='Membership inference (log-log ROC)'):
    """curves: {name: (fpr, tpr)}"""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, (name, (fpr, tpr)) in enumerate(curves.items()):
        fpr = np.clip(np.asarray(fpr), 1e-5, 1); tpr = np.clip(np.asarray(tpr), 1e-5, 1)
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i % len(PALETTE)], label=name)
    ax.plot([1e-5, 1], [1e-5, 1], 'k--', lw=1, label='chance')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-4, 1); ax.set_ylim(1e-4, 1)
    ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate'); ax.set_title(title)
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
    _save(fig, path)


def plot_review_tier(rows, path, title='Conformal review tier'):
    """rows: list of dicts with keys name, rate_auto_approve, rate_auto_block, rate_human_review."""
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(rows)), 5))
    names = [r['name'] for r in rows]
    a = np.array([r['rate_auto_approve'] for r in rows]); b = np.array([r['rate_auto_block'] for r in rows])
    h = np.array([r['rate_human_review'] for r in rows])
    x = np.arange(len(rows))
    ax.bar(x, a, color='#2e8b57', label='auto-approve')
    ax.bar(x, b, bottom=a, color='#c0392b', label='auto-block')
    ax.bar(x, h, bottom=a + b, color='#f0ad4e', label='human review')
    for xi, (aa, bb, hh) in enumerate(zip(a, b, h)):
        ax.text(xi, aa / 2, f'{aa:.1%}', ha='center', va='center', fontsize=8, color='white')
        ax.text(xi, aa + bb + hh / 2, f'{hh:.1%}', ha='center', va='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right'); ax.set_ylabel('share of transactions')
    ax.set_title(title); ax.legend(fontsize=8, loc='lower right')
    _save(fig, path)


def plot_per_bank(bank_ids, series, path, title='Per-bank AUC-PR on each bank\'s own test fold', ylabel='AUC-PR'):
    """series: {name: [value per bank]}"""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bank_ids)); w = 0.8 / max(1, len(series))
    for i, (name, vals) in enumerate(series.items()):
        vals = [np.nan if v is None else v for v in vals]
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x); ax.set_xticklabels([f'bank {b}' for b in bank_ids])
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=8)
    _save(fig, path)
