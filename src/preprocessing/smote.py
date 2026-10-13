"""
Borderline-SMOTE oversampling for fraud minority class.

NOT standard SMOTE — Borderline-SMOTE specifically oversamples minority samples
near the decision boundary. This has algorithmic alignment: Borderline-SMOTE
targets the SAME samples that will have high conformal prediction uncertainty.


WARNING: MATHEMATICALLY UNSOUND UNDER DP-SGD
As per arXiv:2411.05733, applying Borderline-SMOTE before DP-SGD voids the epsilon guarantee. 
Removing one real record changes the k-nearest neighbours of many synthetic rows, 
blowing up sensitivity exponentially in the feature dimension.
Do NOT use this module in the federated DP pipeline. It is retained only for the 
centralized non-private baseline (Midpoint).

CRITICAL constraints:
    - Apply ONLY to training set (NEVER calibration or test)
    - Apply AFTER PCA (oversample in the compressed feature space)
    - Apply LOCALLY at each client (no synthetic data shared across clients)
"""

import numpy as np
from imblearn.over_sampling import BorderlineSMOTE


def apply_borderline_smote(X, y, sampling_strategy=0.2, k_neighbors=5,
                           random_state=42):
    """
    Apply Borderline-SMOTE to oversample the minority (fraud) class.

    Args:
        X: Feature matrix (numpy array or DataFrame), shape [n_samples, n_features].
        y: Target labels (numpy array), shape [n_samples].
        sampling_strategy: Target ratio of minority to majority class (default: 0.2 = 1:5).
        k_neighbors: Number of nearest neighbours for SMOTE (default: 5).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_resampled, y_resampled).
    """
    n_fraud_before = int(np.sum(y == 1))
    n_benign_before = int(np.sum(y == 0))

    # Guard: if no fraud samples or no benign samples, skip SMOTE
    if n_fraud_before == 0 or n_benign_before == 0:
        print(f"  SKIPPING SMOTE: class count zero (benign: {n_benign_before}, fraud: {n_fraud_before})")
        return X, y

    # Guard: if only 1 fraud sample, we cannot run SMOTE (need at least 2 samples for 1-NN)
    if n_fraud_before < 2:
        print(f"  WARNING: Only {n_fraud_before} fraud sample(s). Replicating to meet ratio.")
        # Duplicate the minority samples to reach target count
        target_count = int(n_benign_before * sampling_strategy)
        if target_count > n_fraud_before:
            fraud_idx = np.where(y == 1)[0]
            extra_idx = np.random.choice(fraud_idx, size=target_count - n_fraud_before, replace=True)
            X_extra = X[extra_idx]
            y_extra = y[extra_idx]
            X_res = np.concatenate([X, X_extra], axis=0)
            y_res = np.concatenate([y, y_extra], axis=0)
            return X_res, y_res
        return X, y

    # Guard: if current ratio is already close to or higher than target ratio, skip SMOTE
    current_ratio = n_fraud_before / n_benign_before
    if current_ratio >= (sampling_strategy - 0.02):
        print(f"  SKIPPING SMOTE: current ratio ({current_ratio:.3f}) is sufficiently close to or higher than target ({sampling_strategy:.3f})")
        return X, y

    # Edge case: if too few fraud samples for k_neighbors
    if n_fraud_before <= k_neighbors:
        k_neighbors = max(1, n_fraud_before - 1)
        print(f"  WARNING: Only {n_fraud_before} fraud samples — "
              f"reducing k_neighbors to {k_neighbors}")

    smote = BorderlineSMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
        kind='borderline-1'
    )

    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except Exception as e:
        print(f"  WARNING: Borderline-SMOTE failed ({e}), using standard SMOTE")
        from imblearn.over_sampling import SMOTE
        try:
            smote_fallback = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=random_state
            )
            X_resampled, y_resampled = smote_fallback.fit_resample(X, y)
        except Exception as e2:
            print(f"  WARNING: Standard SMOTE also failed ({e2}). Returning original data.")
            return X, y

    n_fraud_after = int(np.sum(y_resampled == 1))
    n_benign_after = int(np.sum(y_resampled == 0))

    print(f"  SMOTE: {n_fraud_before}/{n_benign_before} (fraud/benign) -> "
          f"{n_fraud_after}/{n_benign_after}")

    return X_resampled, y_resampled
