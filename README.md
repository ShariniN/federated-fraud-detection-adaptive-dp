# Uncertainty-Aware Federated Fraud Detection with Adaptive Differential Privacy

Final Year Project · BEng (Hons) Software Engineering, Staffordshire University (APIIT Colombo) · *in progress*

Banks can't pool transaction data to fight fraud (GDPR, PCI-DSS), so each one fights cross-bank fraud rings alone. Federated learning (FL) with differential privacy (DP) lets them train a shared model without sharing data. Standard DP-FedAvg, though, adds the **same** noise to every transaction, and the rare, borderline fraud cases are the ones most damaged by that noise.

This project proposes a hierarchical framework that adapts privacy noise at two levels:

| # | Component | What it does |
|---|---|---|
| 1 | **Sample-level adaptive DP** | A conformal prediction (RAPS) nonconformity score per transaction sets its noise multiplier: ambiguous samples get less noise, confident ones get more, with a hard floor. Built on Opacus with exact privacy accounting. |
| 2 | **Uncertainty-weighted aggregation (UW-FedAvg)** | Each bank sends its mean calibration-set uncertainty (Laplace-noised) with its update; the server weights updates by data volume × confidence. A clipped, hardened variant resists malicious clients. |
| 3 | **Hierarchy** | Both run at the same time: 1 inside each client's training loop, 2 at the server. |

**Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection): 590,540 transactions, 3.5 % fraud. Not included here (Kaggle terms); download it and run the pipeline.
**Primary metric:** AUC-PR (AUC-ROC is misleading at 3.5 % prevalence).

## Selected results

Membership inference (Yeom and offline LiRA) against the central model at several privacy budgets:

| Model | ε | AUC-PR | LiRA AUC | LiRA TPR @ 1 % FPR |
|---|---|---|---|---|
| Non-private | ∞ | 0.396 | 0.510 | 0.025 |
| DP-SGD | 8 | 0.381 | 0.523 | 0.000 |
| DP-SGD | 3 | 0.371 | 0.523 | 0.000 |
| DP-SGD | 1 | 0.349 | 0.523 | 0.000 |

More in [`results/`](results/): architecture, noise-mechanism and resampling comparisons, federated runs, and attack results.

## Stack

Python · PyTorch · Opacus (DP-SGD) · Flower (federated learning) · scikit-learn · imbalanced-learn · pandas

## Layout

```
src/
  preprocessing/   pipeline, imputation, encoding, PCA, bank partitioning, SMOTE
  models/          FraudMLP (2,898 params), logistic regression, Tabular ResNet
  dp/              adaptive per-sample optimizer on Opacus, noise adaptation, accountant
  conformal/       LAC + RAPS scores, calibration, prediction sets
  federated/       bank client, server (size / uncertainty / clipped / trimmed-mean / median), Flower adapters
  attacks/         membership inference (Yeom, offline LiRA), gradient inversion
  evaluation/      metrics, privacy and uncertainty metrics, plots
experiments/       runnable scripts + YAML configs for every experiment
results/           JSON / CSV / PNG outputs
```

## Running it

```bash
pip install -r requirements.txt
# put the Kaggle CSVs in data/raw/
python src/preprocessing/pipeline.py          # -> data/processed_clean/
python experiments/validate_pipeline.py       # 32 data-validation checks

python experiments/run_baselines.py --seeds 0 1 2 --epochs 15
python experiments/run_ablation.py --group ablation --seeds 0 1 2 --rounds 20
python experiments/run_attacks.py --shadows 8
python experiments/run_robustness.py --rounds 10 --seeds 0
python experiments/run_flower.py --rounds 5
```

All final-phase runs are seeded and resumable.
