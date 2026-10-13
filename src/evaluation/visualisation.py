"""
Evaluation visualisation module.

Generates professional, publication-quality figures:
1. Privacy-utility curves (AUC-PR vs. epsilon).
2. Convergence curves (AUC-PR vs. training rounds).
3. Confusion matrices.
4. Conformal score distributions (density plots comparing fraud vs benign).
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Set style for professional visualisations
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "font.family": "serif"
})


def plot_privacy_utility_curve(results_df, save_path):
    """
    Plot AUC-PR vs. Epsilon curve for different system variants.

    Args:
        results_df: pandas DataFrame containing:
                    ['System', 'Epsilon', 'AUC-PR']
        save_path: Filepath to save the plot.
    """
    plt.figure(figsize=(8, 5))
    
    # Plot curves with error bands if multiple seeds exist, otherwise simple line plot
    # results_df should contain columns: System, Epsilon, AUC-PR
    # We will use sns.lineplot which automatically aggregates multiple values (seeds) per X
    ax = sns.lineplot(
        data=results_df,
        x="Epsilon",
        y="AUC-PR",
        hue="System",
        style="System",
        markers=True,
        dashes=False,
        linewidth=2,
        errorbar="sd"  # standard deviation shaded region
    )
    
    ax.set_xscale("log")  # standard for DP-SGD epsilon sweeps
    ax.set_xticks([1, 3, 5, 10])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.title("Privacy-Utility Tradeoff (AUC-PR vs. Epsilon)")
    plt.xlabel("Privacy Budget Epsilon (ε) at δ=1e-5")
    plt.ylabel("Utility Metric (AUC-PR)")
    plt.legend(title="System Configuration", frameon=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Privacy-Utility plot to {save_path}")


def plot_convergence(history_dicts, save_path):
    """
    Plot utility convergence curves over training rounds.

    Args:
        history_dicts: Dict of {system_name: list_of_round_metrics_dicts}
        save_path: Filepath to save the plot.
    """
    plt.figure(figsize=(9, 5.5))
    
    for system_name, rounds in history_dicts.items():
        rounds_idx = [r["round"] for r in rounds if r["round"] > 0]
        auc_pr = [r["auc_pr"] for r in rounds if r["round"] > 0]
        
        plt.plot(rounds_idx, auc_pr, label=system_name, linewidth=2)
        
    plt.title("Utility Convergence Curve")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Holdout AUC-PR")
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Convergence curve to {save_path}")


def plot_confusion_matrices(y_true, y_pred, save_path):
    """
    Plot confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predictions.
        save_path: Filepath to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalise confusion matrix to rates
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # 1. Absolute counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Benign", "Fraud"], yticklabels=["Benign", "Fraud"])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    
    # 2. Normalised rates
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=["Benign", "Fraud"], yticklabels=["Benign", "Fraud"])
    axes[1].set_title("Confusion Matrix (Rates)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    
    plt.suptitle("Fraud Classification Performance")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrices to {save_path}")


def plot_conformal_score_distribution(scores, labels, q_hat, save_path):
    """
    Plot probability density functions of conformal scores for fraud vs. benign.

    Args:
        scores: Conformal scores (numpy array).
        labels: Class labels.
        q_hat: Conformal threshold.
        save_path: Filepath to save the plot.
    """
    plt.figure(figsize=(8, 5))
    
    sns.kdeplot(scores[labels == 0], label="Benign", fill=True, alpha=0.4, color="skyblue")
    sns.kdeplot(scores[labels == 1], label="Fraud", fill=True, alpha=0.4, color="salmon")
    
    # Add vertical line at conformal threshold q_hat
    plt.axvline(q_hat, color="darkred", linestyle="--", linewidth=1.5,
                label=f"Conformal Threshold q_hat ({q_hat:.3f})")
    
    plt.title("Distribution of Conformal Nonconformity Scores")
    plt.xlabel("RAPS Nonconformity Score")
    plt.ylabel("Density")
    plt.legend(frameon=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Conformal Score Distribution to {save_path}")
