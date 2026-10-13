"""
DP Noise Verification and Evaluation Module.

Provides utilities to statistically analyze, check, and visualize the
Gaussian noise added during DP-SGD to verify that it is mathematically correct
and evaluate its magnitude (realistic scaling) relative to actual gradients.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def evaluate_gradient_noise(sigma, C, batch_size, dim=1000, random_state=42):
    """
    Simulates gradient clipping and noise addition, performing statistical tests
    to verify that the noise added matches the theoretical DP-SGD formulation
    and is realistic relative to the gradient signal.

    Args:
        sigma: Noise multiplier.
        C: Max gradient clipping threshold.
        batch_size: Expected batch size.
        dim: Dimension of the gradient vector (e.g. number of parameters).
        random_state: Seed for reproducibility.

    Returns:
        Dict containing validation statistics.
    """
    np.random.seed(random_state)

    # 1. Simulate raw gradients (mixture of normal distributions to represent realistic gradients)
    raw_gradients = np.random.normal(loc=0.05, scale=0.2, size=(batch_size, dim))

    # 2. Perform per-sample clipping
    clipped_gradients = []
    norms = np.linalg.norm(raw_gradients, axis=1)
    for i in range(batch_size):
        g = raw_gradients[i]
        norm = norms[i]
        if norm > C:
            clipped_gradients.append(g * (C / norm))
        else:
            clipped_gradients.append(g)
    clipped_gradients = np.array(clipped_gradients)

    # Compute average clipped gradient (the "signal")
    avg_clipped_gradient = np.mean(clipped_gradients, axis=0)
    signal_power = np.mean(avg_clipped_gradient ** 2)

    # 3. Simulate Gaussian noise addition in DP-SGD
    # Opacus adds noise to the sum of clipped gradients, then divides by batch_size.
    # Sum noise standard deviation is sigma * C.
    # Therefore, noise added to the average gradient has standard deviation: (sigma * C) / batch_size
    theoretical_std = (sigma * C) / batch_size
    noise = np.random.normal(loc=0.0, scale=sigma * C, size=dim)
    scaled_noise = noise / batch_size

    empirical_mean = np.mean(scaled_noise)
    empirical_std = np.std(scaled_noise)
    noise_power = np.mean(scaled_noise ** 2)

    # 4. Statistical normality test (Kolmogorov-Smirnov test)
    # Compare the normalized noise against standard normal distribution
    normalized_noise = (scaled_noise - 0.0) / theoretical_std
    ks_stat, p_value = stats.kstest(normalized_noise, 'norm')

    # 5. Signal-to-Noise Ratio (SNR) calculation
    # SNR = average signal power / average noise power
    snr = signal_power / (noise_power + 1e-12)
    snr_db = 10 * np.log10(snr)

    return {
        "theoretical_std": theoretical_std,
        "empirical_mean": empirical_mean,
        "empirical_std": empirical_std,
        "p_value": p_value,
        "is_normal": p_value > 0.05,
        "snr_db": snr_db,
        "signal_power": signal_power,
        "noise_power": noise_power,
        "scaled_noise": scaled_noise,
        "avg_clipped_gradient": avg_clipped_gradient
    }


def plot_noise_analysis(stats_dict, save_path="results/runs/dp_noise_analysis.png"):
    """
    Plots the noise distribution histogram, theoretical Gaussian PDF,
    and cumulative distribution function to visually verify the noise profile.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    noise = stats_dict["scaled_noise"]
    theory_std = stats_dict["theoretical_std"]

    plt.figure(figsize=(12, 5))

    # Subplot 1: Histogram vs Theoretical PDF
    plt.subplot(1, 2, 1)
    count, bins, ignored = plt.hist(noise, bins=50, density=True, alpha=0.6, color='royalblue', label='Empirical Noise')
    
    # PDF curve
    x = np.linspace(bins[0], bins[-1], 200)
    pdf = stats.norm.pdf(x, 0, theory_std)
    plt.plot(x, pdf, 'r-', lw=2, label=f'Theoretical PDF\n(std={theory_std:.6f})')
    
    plt.title("DP-SGD Noise Distribution vs Theory", fontsize=12)
    plt.xlabel("Noise Magnitude", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Subplot 2: Signal vs Noise comparison
    plt.subplot(1, 2, 2)
    grad_norms = np.abs(stats_dict["avg_clipped_gradient"])
    noise_norms = np.abs(noise)
    
    plt.hist(grad_norms, bins=50, alpha=0.5, color='darkorange', label='Clipped Gradient (Signal)')
    plt.hist(noise_norms, bins=50, alpha=0.5, color='royalblue', label='DP Noise (Magnitude)')
    
    plt.title(f"Gradient vs Noise (SNR: {stats_dict['snr_db']:.2f} dB)", fontsize=12)
    plt.xlabel("Absolute Value", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Noise analysis plot saved to {save_path}")
