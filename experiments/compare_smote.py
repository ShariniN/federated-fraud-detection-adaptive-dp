# -*- coding: utf-8 -*-
"""
SMOTE Technique Comparison | FraudMLP | Real IEEE-CIS
Each strategy applied to RAW pre-SMOTE data (clean controlled comparison)
"""
import os, sys, time, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.mlp import FraudMLP
from src.evaluation.train_utils import load_real_data, train_model_dp, compute_all_metrics, plot_confusion_matrix

DP_CONFIG = dict(epochs=15, lr=1e-3, batch_size=512, max_grad_norm=1.0, noise_multiplier=1.0, target_delta=1e-5)
OUT_DIR = os.path.join('results', 'smote_comparison')

def _k(y, default=5):
    n = int(y.sum())
    return min(default, n-1) if n > 1 else 1

def apply_none(X, y):
    n_fraud = int(y.sum()); n_benign = len(y) - n_fraud
    print('    No SMOTE: fraud=' + str(n_fraud) + '  benign=' + str(n_benign) +
          '  rate=' + str(round(y.mean()*100,2)) + '%')
    return X, y

def apply_standard(X, y, ss=0.3, rs=42):
    from imblearn.over_sampling import SMOTE
    try:
        res = SMOTE(sampling_strategy=ss, k_neighbors=_k(y), random_state=rs)
        Xr, yr = res.fit_resample(X, y)
        print('    Standard SMOTE: fraud ' + str(int(y.sum())) + '->' + str(int(yr.sum())))
        return Xr, yr
    except Exception as e:
        print('    Standard SMOTE failed: ' + str(e)); return X, y

def apply_borderline(X, y, ss=0.3, rs=42):
    from imblearn.over_sampling import BorderlineSMOTE
    try:
        res = BorderlineSMOTE(sampling_strategy=ss, k_neighbors=_k(y), random_state=rs, kind='borderline-1')
        Xr, yr = res.fit_resample(X, y)
        print('    Borderline-SMOTE: fraud ' + str(int(y.sum())) + '->' + str(int(yr.sum())))
        return Xr, yr
    except Exception as e:
        print('    Borderline-SMOTE failed: ' + str(e)); return X, y

def apply_adasyn(X, y, ss=0.3, rs=42):
    from imblearn.over_sampling import ADASYN
    try:
        res = ADASYN(sampling_strategy=ss, n_neighbors=_k(y), random_state=rs)
        Xr, yr = res.fit_resample(X, y)
        print('    ADASYN: fraud ' + str(int(y.sum())) + '->' + str(int(yr.sum())))
        return Xr, yr
    except Exception as e:
        print('    ADASYN failed: ' + str(e)); return X, y

STRATEGIES = [
    {'name': 'No SMOTE',         'color': '#64748b', 'fn': apply_none},
    {'name': 'Standard SMOTE',   'color': '#dc2626', 'fn': apply_standard},
    {'name': 'Borderline-SMOTE', 'color': '#2563eb', 'fn': apply_borderline},
    {'name': 'ADASYN',           'color': '#7c3aed', 'fn': apply_adasyn},
]

def plot_smote_comparison(results, out_dir):
    names  = [r['name'] for r in results]
    colors = [r['color'] for r in results]
    aucpr  = [r['metrics']['auc_pr'] for r in results]
    aucroc = [r['metrics']['auc_roc'] for r in results]
    f1_def = [r['metrics']['f1_default'] for r in results]
    f1_cal = [r['metrics']['f1_calibrated'] for r in results]
    x = np.arange(len(names)); w = 0.20
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('SMOTE Technique Comparison  |  FraudMLP  |  Real IEEE-CIS\n'
                 'Each strategy applied to RAW data | DP-SGD: sigma=1.0, C=1.0, batch=512',
                 fontsize=12, fontweight='bold')
    ax = axes[0]
    b1 = ax.bar(x-1.5*w, aucpr,  w, label='AUC-PR (primary)', color=[c+'ee' for c in colors])
    b2 = ax.bar(x-0.5*w, aucroc, w, label='AUC-ROC',          color=colors, alpha=0.6)
    b3 = ax.bar(x+0.5*w, f1_def, w, label='F1 (tau=0.50)',    color=colors, alpha=0.35)
    b4 = ax.bar(x+1.5*w, f1_cal, w, label='F1 (calibrated)',  color=colors, alpha=0.9, hatch='//')
    for bg in [b1, b4]:
        for bar in bg:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.01, str(round(h,3)),
                        ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10, ha='right')
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
    ax.set_title('Metrics by Resampling Strategy (Applied to Raw Imbalanced Data)')
    ax.legend(fontsize=9)
    ax.axhline(0.5, color='grey', linestyle='--', alpha=0.4)
    best_idx = int(np.argmax(aucpr))
    ax.bar(x[best_idx]-1.5*w, aucpr[best_idx], w, color=colors[best_idx],
           edgecolor='gold', linewidth=2.5, zorder=5)
    ax.annotate('BEST', xy=(x[best_idx]-1.5*w, aucpr[best_idx]+0.04),
                ha='center', fontsize=9, color='goldenrod', fontweight='bold')
    ax2 = axes[1]
    fr_before = [r['fraud_rate_before'] for r in results]
    fr_after  = [r['fraud_rate_after'] for r in results]
    bw = 0.35; xa = np.arange(len(names))
    ax2.bar(xa-bw/2, fr_before, bw, label='Before resampling (raw)', color='salmon', alpha=0.8)
    ax2.bar(xa+bw/2, fr_after,  bw, label='After SMOTE',  color='steelblue', alpha=0.8)
    ax2.set_xticks(xa); ax2.set_xticklabels(names, rotation=10, ha='right')
    ax2.set_ylabel('Fraud Rate in Training Set')
    ax2.set_title('Class Balance Before vs After Each Strategy')
    ax2.legend(fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'smote_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print('  Plot saved: ' + out_path)

def main():
    device = 'cpu'; os.makedirs(OUT_DIR, exist_ok=True)
    print('='*65)
    print('  SMOTE Comparison  |  FraudMLP  |  Real IEEE-CIS  |  DP-SGD')
    print('  Each strategy applied to RAW pre-SMOTE data (clean comparison)')
    print('='*65)

    X_raw, y_raw, X_cal, y_cal, X_test, y_test, input_dim = load_real_data()
    fr_raw = float(y_raw.mean())
    print('\n  Raw training fraud rate: ' + str(round(fr_raw*100,2)) + '% (real imbalanced data)')

    results = []; rows = []
    for strat in STRATEGIES:
        print('\n' + '='*65); print('  Strategy: ' + strat['name']); print('='*65)
        # Apply each strategy independently to the SAME raw data
        X_tr, y_tr = strat['fn'](X_raw.copy(), y_raw.copy())
        fr_after = float(y_tr.mean())
        print('    Train: ' + str(len(y_tr)) + ' samples  fraud=' + str(round(fr_after*100,2)) + '%')
        t0 = time.time()
        model = FraudMLP(input_dim=input_dim)
        trained, eps = train_model_dp(model, X_tr, y_tr, device=device, **DP_CONFIG)
        elapsed = round(time.time()-t0, 1)
        metrics = compute_all_metrics(trained, X_test, y_test, X_cal=X_cal, y_cal=y_cal,
                                      device=device, label=strat['name'])
        cm_path = os.path.join(OUT_DIR, 'cm_' + strat['name'].replace(' ', '_') + '.png')
        plot_confusion_matrix(metrics['confusion_matrix'], strat['name'], metrics['best_tau'], cm_path)
        results.append({**strat, 'metrics': metrics,
                        'fraud_rate_before': fr_raw, 'fraud_rate_after': fr_after, 'elapsed_s': elapsed})
        rows.append({'Strategy': strat['name'], 'Train_samples': len(y_tr),
                     'Fraud_rate_raw': round(fr_raw,4), 'Fraud_rate_after': round(fr_after,4),
                     'AUC-PR': round(metrics['auc_pr'],4), 'AUC-ROC': round(metrics['auc_roc'],4),
                     'F1_default_0.50': round(metrics['f1_default'],4),
                     'F1_calibrated': round(metrics['f1_calibrated'],4),
                     'Best_Tau': metrics['best_tau'],
                     'Recall_cal': round(metrics['recall_calibrated'],4),
                     'Precision_cal': round(metrics['precision_calibrated'],4),
                     'Epsilon': round(float(eps),3) if eps else 'N/A', 'Time_s': elapsed})

    print('\n' + '='*75)
    print('  SMOTE COMPARISON SUMMARY  |  RAW DATA  |  FraudMLP  |  DP-SGD')
    print('='*75)
    print('Strategy'.ljust(20)+'Samples'.rjust(9)+'FraudRate'.rjust(10)+
          'AUC-PR'.rjust(8)+'F1(0.50)'.rjust(10)+'F1(cal)'.rjust(9)+'tau'.rjust(6)+'Recall'.rjust(8))
    print('-'*75)
    for r in rows:
        print(str(r['Strategy']).ljust(20)+str(r['Train_samples']).rjust(9)+
              (str(round(r['Fraud_rate_after']*100,1))+'%').rjust(10)+
              str(r['AUC-PR']).rjust(8)+str(r['F1_default_0.50']).rjust(10)+
              str(r['F1_calibrated']).rjust(9)+str(r['Best_Tau']).rjust(6)+
              str(r['Recall_cal']).rjust(8))
    best = max(results, key=lambda r: r['metrics']['auc_pr'])
    print('\n  Best SMOTE strategy by AUC-PR: ' + best['name'])
    print('  Note: This is a clean comparison - each strategy applied to the same raw')
    print('  imbalanced data (' + str(round(fr_raw*100,2)) + '% fraud). No prior resampling in pipeline.')
    plot_smote_comparison(results, OUT_DIR)
    csv_path = os.path.join(OUT_DIR, 'full_metrics_smote.csv')
    with open(csv_path, 'w', newline='') as f:
        w2 = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w2.writeheader(); w2.writerows(rows)
    print('  CSV saved: ' + csv_path)
    print('\nDone.')

if __name__ == '__main__':
    main()
