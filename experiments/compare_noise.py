# -*- coding: utf-8 -*-
"""Noise Mechanism Comparison - FraudMLP + Borderline-SMOTE - Real IEEE-CIS."""
import os, sys, time, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.models.mlp import FraudMLP
from src.evaluation.train_utils import load_real_data, compute_all_metrics, plot_confusion_matrix

try:
    from imblearn.over_sampling import BorderlineSMOTE
    IMBLEARN = True
except ImportError:
    IMBLEARN = False

try:
    from opacus import PrivacyEngine
    OPACUS = True
except ImportError:
    OPACUS = False

OUT_DIR = os.path.join("results", "noise_comparison")
EPOCHS = 15; LR = 1e-3; BATCH = 512; SIGMA = 1.0; C = 1.0; DELTA = 1e-5

def _smote(X, y, ss=0.3, rs=42):
    if not IMBLEARN: return X, y
    n = int(y.sum())
    if n < 2: return X, y
    k = min(5, n-1)
    try:
        res = BorderlineSMOTE(sampling_strategy=ss, k_neighbors=k, random_state=rs, kind="borderline-1")
        Xr, yr = res.fit_resample(X, y)
        print("  SMOTE: " + str(n) + " -> " + str(int(yr.sum())) + " fraud")
        return Xr, yr
    except Exception as e:
        print("  SMOTE failed: " + str(e)); return X, y

def _loader(X, y, bs=BATCH):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

def _weights(y, device):
    nf = int(y.sum()); nb = len(y) - nf
    return torch.tensor([1.0, nb/nf], dtype=torch.float32).to(device) if nf > 0 and nb > 0 else None

def _base_train(model, X, y, device, noise_fn=None, epochs=EPOCHS):
    model = model.to(device)
    w = _weights(y, device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = _loader(X, y)
    model.train()
    for ep in range(1, epochs+1):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            if noise_fn:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            norm = p.grad.norm(2)
                            if norm > C: p.grad.mul_(C / norm)
                            p.grad.add_(noise_fn(p.grad.shape, device) / BATCH)
            optimizer.step()
        if ep % 5 == 0 or ep == 1:
            print("    Epoch " + str(ep) + "/" + str(epochs))
    return model

def train_no_noise(model, X, y, device):
    print("  Non-private baseline (no noise)")
    m = _base_train(model, X, y, device, noise_fn=None)
    return m, float("inf")

def train_gaussian(model, X, y, device):
    import warnings
    if not OPACUS:
        print("  WARNING: Opacus unavailable - manual Gaussian noise")
        fn = lambda shape, dev: torch.tensor(np.random.normal(0, SIGMA*C, shape), dtype=torch.float32).to(dev)
        m = _base_train(model, X, y, device, noise_fn=fn)
        return m, None
    model = model.to(device)
    w = _weights(y, device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = _loader(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pe = PrivacyEngine()
        model, optimizer, loader = pe.make_private(module=model, optimizer=optimizer, data_loader=loader,
                                                   noise_multiplier=SIGMA, max_grad_norm=C)
    model.train()
    for ep in range(1, EPOCHS+1):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss(weight=w)(model(Xb), yb).backward()
            optimizer.step()
        if ep % 5 == 0 or ep == 1:
            eps = pe.get_epsilon(DELTA)
            print("    Epoch " + str(ep) + "/" + str(EPOCHS) + "  eps=" + str(round(eps,2)))
    # Unwrap Opacus GradSampleModule to restore predict_proba and custom methods
    final_model = getattr(model, "_module", model)
    return final_model, pe.get_epsilon(DELTA)


def train_laplace(model, X, y, device, eps_budget=2.83):
    # eps_budget parameterises the Laplace SCALE (b = C / eps_budget).
    # It is NOT an accounted privacy budget: this arm uses per-parameter
    # clipping on batch-averaged gradients, so no formal guarantee applies.
    scale = C / eps_budget
    fn = lambda shape, dev: torch.tensor(np.random.laplace(0, scale, shape), dtype=torch.float32).to(dev)
    m = _base_train(model, X, y, device, noise_fn=fn)
    return m, eps_budget

def train_uniform(model, X, y, device):
    a = C * (3**0.5) * SIGMA
    fn = lambda shape, dev: torch.FloatTensor(shape).uniform_(-a, a).to(dev)
    m = _base_train(model, X, y, device, noise_fn=fn)
    return m, None

def analyse_noise(name, dim=5000):
    """Analytic characterisation of each noise DISTRIBUTION.

    NOTE: this is computed on a synthetic gradient surrogate, not on the trained
    model's own gradients. It characterises the shape and scale of each noise
    distribution relative to a reference signal; it is not a measurement of the
    signal-to-noise ratio actually experienced during training. Reported as
    'SNR_ref_dB' for that reason.
    """
    np.random.seed(42)
    raw = np.random.normal(0.05, 0.2, (BATCH, dim))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    clipped = raw * np.minimum(1, C / (norms + 1e-12))
    signal = clipped.mean(axis=0); sig_pw = np.mean(signal**2)
    if name == "Gaussian": noise = np.random.normal(0, SIGMA*C, dim) / BATCH
    elif name == "Laplace": noise = np.random.laplace(0, C/2.83, dim) / BATCH
    elif name == "Uniform": noise = np.random.uniform(-C*(3**0.5)*SIGMA, C*(3**0.5)*SIGMA, dim) / BATCH
    else: noise = np.zeros(dim)
    noi_pw = np.mean(noise**2)
    snr_db = 10*np.log10(sig_pw/(noi_pw+1e-12)) if noi_pw > 0 else float("inf")
    if name == "Gaussian":
        th = (SIGMA*C)/BATCH
        ks, pv = scipy_stats.kstest((noise - 0)/th, "norm")
    else: pv = None
    return {"snr_db": snr_db, "ks_pval": pv}

def plot_noise_comparison(results, out_dir):
    names  = [r["name"] for r in results]
    colors = [r["color"] for r in results]
    aucpr  = [r["metrics"]["auc_pr"] for r in results]
    f1_cal = [r["metrics"]["f1_calibrated"] for r in results]
    aucroc = [r["metrics"]["auc_roc"] for r in results]
    snrs   = [r["noise"]["snr_db"] if r["noise"]["snr_db"] != float("inf") else 30 for r in results]
    x = np.arange(len(names)); w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Noise Mechanism Comparison  |  FraudMLP + Borderline-SMOTE  |  Real IEEE-CIS\nC=1.0, sigma=1.0, batch=512, epochs=15", fontsize=12, fontweight="bold")
    ax = axes[0]
    b1 = ax.bar(x-w, aucpr,  w, label="AUC-PR (primary)", color=[c+"ee" for c in colors])
    b2 = ax.bar(x,   aucroc, w, label="AUC-ROC",          color=colors, alpha=0.6)
    b3 = ax.bar(x+w, f1_cal, w, label="F1 (calibrated)",  color=colors, alpha=0.85, hatch="//")
    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.01, str(round(h,3)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Detection Metrics by Noise Mechanism")
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.4)
    for idx, nm in enumerate(names):
        if nm in ("Gaussian", "No Noise"):
            ax.bar(x[idx]-w, aucpr[idx], w, color=colors[idx], edgecolor="gold", linewidth=2.5, zorder=5)
    ax2 = axes[1]
    snr_colors = ["#2563eb" if n == "Gaussian" else "#94a3b8" for n in names]
    bars = ax2.bar(x, snrs, color=snr_colors, edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=10, ha="right")
    ax2.set_ylabel("Reference SNR (dB, analytic)")
    ax2.set_title("SNR per Noise Mechanism\n(more negative = stronger privacy noise)")
    ax2.axhline(0, color="red", linestyle="--", alpha=0.5, label="SNR=0 dB")
    ax2.legend()
    for bar, snr in zip(bars, snrs):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+(0.3 if bar.get_height()>=0 else -1.5),
                 str(round(snr,1))+" dB", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "noise_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print("  Plot saved: " + out_path)

MECHANISMS = [
    {"name": "No Noise", "color": "#94a3b8", "trainer": train_no_noise},
    {"name": "Gaussian", "color": "#2563eb", "trainer": train_gaussian},
    {"name": "Laplace",  "color": "#dc2626", "trainer": train_laplace},
    {"name": "Uniform",  "color": "#7c3aed", "trainer": train_uniform},
]

def main():
    device = "cpu"; os.makedirs(OUT_DIR, exist_ok=True)
    print("="*65); print("  Noise Comparison  |  FraudMLP  |  Real IEEE-CIS"); print("="*65)
    X_raw, y_raw, X_cal, y_cal, X_test, y_test, input_dim = load_real_data('data/processed_clean')
    print("\n  Applying Borderline-SMOTE...")
    X_tr, y_tr = _smote(X_raw, y_raw)
    results = []; rows = []
    for mech in MECHANISMS:
        print("\n" + "="*65); print("  Mechanism: " + mech["name"]); print("="*65)
        t0 = time.time()
        model = FraudMLP(input_dim=input_dim)
        trained, eps = mech["trainer"](model, X_tr, y_tr, device)
        elapsed = round(time.time()-t0, 1)
        metrics = compute_all_metrics(trained, X_test, y_test, X_cal=X_cal, y_cal=y_cal,
                                      device=device, label=mech["name"])
        ns = analyse_noise(mech["name"])
        cm_path = os.path.join(OUT_DIR, "cm_" + mech["name"].replace(" ", "_") + ".png")
        plot_confusion_matrix(metrics["confusion_matrix"], mech["name"], metrics["best_tau"], cm_path)
        results.append({**mech, "metrics": metrics, "noise": ns, "elapsed_s": elapsed})
        ks_str = str(round(ns["ks_pval"],4)) if ns["ks_pval"] is not None else "N/A"
        print("  SNR: " + str(round(ns["snr_db"],2)) + " dB  KS p-val: " + ks_str)
        eps_str = str(round(float(eps),2)) if eps and eps != float("inf") else str(eps)
        rows.append({"Mechanism": mech["name"],
                     "AUC-PR": round(metrics["auc_pr"],4), "AUC-ROC": round(metrics["auc_roc"],4),
                     "F1_default_0.50": round(metrics["f1_default"],4),
                     "F1_calibrated": round(metrics["f1_calibrated"],4),
                     "Best_Tau": metrics["best_tau"],
                     "Recall_cal": round(metrics["recall_calibrated"],4),
                     "Precision_cal": round(metrics["precision_calibrated"],4),
                     "SNR_ref_dB": round(ns["snr_db"],2), "KS_pval": ks_str,
                     "Epsilon": eps_str, "Formal_DP": ("yes" if mech["name"] == "Gaussian" else "no"),
                     "Time_s": elapsed})
    print("\n" + "="*80)
    print("  NOISE COMPARISON SUMMARY  |  FraudMLP + Borderline-SMOTE  |  Real IEEE-CIS")
    print("="*80)
    print("Mechanism".ljust(12)+"AUC-PR".rjust(8)+"AUC-ROC".rjust(9)+
          "F1(0.50)".rjust(10)+"F1(cal)".rjust(9)+"tau".rjust(6)+"SNRref".rjust(9)+"KS-pval".rjust(9)+"formalDP".rjust(10))
    print("-"*80)
    for r in rows:
        print(str(r["Mechanism"]).ljust(12)+str(r["AUC-PR"]).rjust(8)+str(r["AUC-ROC"]).rjust(9)+
              str(r["F1_default_0.50"]).rjust(10)+str(r["F1_calibrated"]).rjust(9)+
              str(r["Best_Tau"]).rjust(6)+str(r["SNR_ref_dB"]).rjust(9)+str(r["KS_pval"]).rjust(9)+str(r["Formal_DP"]).rjust(10))
    print("\n  Gaussian selected: formal (eps,delta)-DP via Renyi accountant.")
    print("  No Noise = non-private upper bound reference.")
    plot_noise_comparison(results, OUT_DIR)
    csv_path = os.path.join(OUT_DIR, "full_metrics_noise.csv")
    with open(csv_path, "w", newline="") as f:
        w2 = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w2.writeheader(); w2.writerows(rows)
    print("  CSV saved: " + csv_path)
    print("\nDone.")

if __name__ == "__main__":
    main()
