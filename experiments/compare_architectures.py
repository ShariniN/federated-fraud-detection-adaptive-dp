# -*- coding: utf-8 -*-
"""
Architecture Comparison Under DP-SGD
Real IEEE-CIS data | Borderline-SMOTE applied ONCE | 3 models compared
"""
import os, sys, time, json, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.mlp import FraudMLP
from src.models.baselines import LogisticRegressionModel, TabularResNet
from src.evaluation.train_utils import (
    load_real_data, apply_borderline_smote, train_model_dp,
    compute_all_metrics, plot_confusion_matrix
)

DP_CONFIG = dict(epochs=15, lr=1e-3, batch_size=512,
                 max_grad_norm=1.0, noise_multiplier=1.0, target_delta=1e-5)
ARCHITECTURES = [
    {'name': 'FraudMLP',            'color': '#2563eb', 'model_cls': FraudMLP},
    {'name': 'Logistic Regression', 'color': '#dc2626', 'model_cls': LogisticRegressionModel},
    {'name': 'Tabular ResNet',      'color': '#16a34a', 'model_cls': TabularResNet},
]
OUT_DIR = os.path.join('results', 'architecture_comparison')

def run_experiment(arch, X_train, y_train, X_cal, y_cal, X_test, y_test, input_dim, device):
    model = arch['model_cls'](input_dim=input_dim)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('  Parameters: ' + str(param_count))
    trained, eps = train_model_dp(model, X_train, y_train, device=device, **DP_CONFIG)
    metrics = compute_all_metrics(trained, X_test, y_test, X_cal=X_cal, y_cal=y_cal,
                                  device=device, label=arch['name'])
    cm_path = os.path.join(OUT_DIR, 'cm_' + arch['name'].replace(' ', '_') + '.png')
    plot_confusion_matrix(metrics['confusion_matrix'], arch['name'], metrics['best_tau'], cm_path)
    metrics['param_count'] = param_count
    metrics['epsilon'] = round(float(eps), 3) if eps else None
    return metrics

def plot_comparison(results, out_dir):
    names  = [r['name'] for r in results]
    colors = [r['color'] for r in results]
    aucpr  = [r['metrics']['auc_pr'] for r in results]
    aucroc = [r['metrics']['auc_roc'] for r in results]
    f1_def = [r['metrics']['f1_default'] for r in results]
    f1_cal = [r['metrics']['f1_calibrated'] for r in results]
    recall = [r['metrics']['recall_calibrated'] for r in results]
    params = [r['metrics']['param_count'] for r in results]
    x = np.arange(len(names)); w = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Architecture Comparison Under DP-SGD  |  Real IEEE-CIS Dataset\n'
                 'Borderline-SMOTE applied once | sigma=1.0, C=1.0, batch=512, epochs=15',
                 fontsize=12, fontweight='bold')
    ax = axes[0]
    b1 = ax.bar(x-2*w, aucpr,  w, label='AUC-PR (primary)', color=[c+'dd' for c in colors])
    b2 = ax.bar(x-w,   aucroc, w, label='AUC-ROC',          color=colors, alpha=0.75)
    b3 = ax.bar(x,     f1_def, w, label='F1 (tau=0.50)',    color=colors, alpha=0.45)
    b4 = ax.bar(x+w,   f1_cal, w, label='F1 (calibrated)',  color=colors, alpha=0.9, hatch='//')
    b5 = ax.bar(x+2*w, recall, w, label='Recall (cal.)',    color=colors, alpha=0.55, hatch='..')
    for bg in [b1, b4]:
        for bar in bg:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.01, str(round(h,3)),
                        ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10, ha='right')
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
    ax.set_title('All 6 Metrics by Architecture')
    ax.legend(fontsize=8, loc='upper right')
    ax.axhline(0.5, color='grey', linestyle='--', alpha=0.4)
    ax2 = axes[1]
    for nm, p, a, c in zip(names, params, aucpr, colors):
        ax2.scatter(p, a, s=250, color=c, zorder=5, edgecolors='black', linewidths=0.8)
        ax2.annotate(nm, (p, a), textcoords='offset points', xytext=(8,4), fontsize=9)
    ax2.set_xlabel('Parameter Count'); ax2.set_ylabel('AUC-PR')
    ax2.set_title('Model Complexity vs Privacy-Utility Trade-off')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'architecture_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print('  Plot saved: ' + out_path)

def main():
    device = 'cpu'; os.makedirs(OUT_DIR, exist_ok=True)
    print('='*65)
    print('  Architecture Comparison  |  Real IEEE-CIS  |  DP-SGD')
    print('  Data: processed_clean (raw pre-SMOTE partitions)')
    print('='*65)

    X_raw, y_raw, X_cal, y_cal, X_test, y_test, input_dim = load_real_data()

    # Apply Borderline-SMOTE ONCE - same resampled data used for ALL 3 models
    print('\n  Applying Borderline-SMOTE once (shared across all architectures)...')
    X_train, y_train = apply_borderline_smote(X_raw, y_raw)
    print('  Train after SMOTE: ' + str(len(y_train)) + ' samples  fraud=' +
          str(round(y_train.mean()*100,2)) + '%')

    results = []; rows = []
    for arch in ARCHITECTURES:
        print('\n' + '='*65); print('  Model: ' + arch['name']); print('='*65)
        t0 = time.time()
        metrics = run_experiment(arch, X_train, y_train, X_cal, y_cal, X_test, y_test, input_dim, device)
        elapsed = round(time.time()-t0, 1)
        results.append({**arch, 'metrics': metrics, 'elapsed_s': elapsed})
        rows.append({'Architecture': arch['name'], 'Params': metrics['param_count'],
                     'AUC-PR': round(metrics['auc_pr'],4), 'AUC-ROC': round(metrics['auc_roc'],4),
                     'F1_default_0.50': round(metrics['f1_default'],4),
                     'F1_calibrated': round(metrics['f1_calibrated'],4),
                     'Best_Tau': metrics['best_tau'],
                     'Recall_calibrated': round(metrics['recall_calibrated'],4),
                     'Precision_cal': round(metrics['precision_calibrated'],4),
                     'Threshold_optimism': round(metrics['threshold_optimism'],4),
                     'Epsilon': metrics['epsilon'] if metrics['epsilon'] else 'N/A',
                     'Time_s': elapsed})

    print('\n' + '='*75)
    print('  FULL METRICS SUMMARY  |  Real IEEE-CIS  |  DP-SGD  |  Borderline-SMOTE')
    print('='*75)
    print('Architecture'.ljust(22)+'Params'.rjust(9)+'AUC-PR'.rjust(8)+'AUC-ROC'.rjust(9)+
          'F1(0.50)'.rjust(10)+'F1(cal)'.rjust(9)+'tau'.rjust(6)+'Recall'.rjust(8)+'Prec'.rjust(8)+'Eps'.rjust(7))
    print('-'*75)
    for r in rows:
        print(str(r['Architecture']).ljust(22)+str(r['Params']).rjust(9)+
              str(r['AUC-PR']).rjust(8)+str(r['AUC-ROC']).rjust(9)+
              str(r['F1_default_0.50']).rjust(10)+str(r['F1_calibrated']).rjust(9)+
              str(r['Best_Tau']).rjust(6)+str(r['Recall_calibrated']).rjust(8)+
              str(r['Precision_cal']).rjust(8)+str(r['Epsilon']).rjust(7))
    best = max(results, key=lambda r: r['metrics']['auc_pr'])
    print('\n  Best by AUC-PR: ' + best['name'] + ' (' + str(round(best['metrics']['auc_pr'],4)) + ')')
    plot_comparison(results, OUT_DIR)
    csv_path = os.path.join(OUT_DIR, 'full_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w2 = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w2.writeheader(); w2.writerows(rows)
    print('  CSV saved: ' + csv_path)
    json_path = os.path.join(OUT_DIR, 'comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump({'summary': rows, 'config': DP_CONFIG, 'dataset': 'real_ieee_cis_clean'}, f, indent=4)
    print('  JSON saved: ' + json_path)
    print('\nDone.')

if __name__ == '__main__':
    main()
