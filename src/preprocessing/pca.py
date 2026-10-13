"""
PCA dimensionality reduction for IEEE-CIS Fraud Detection.

Reduces 480 engineered features to 72 PCA components.

Why PCA is a PRIVACY-UTILITY strategy (not just preprocessing):
    DP noise is added independently across every gradient coordinate, so the
    noise burden grows with the dimension of the parameter space.
    Compressing 480 input features to 72 shrinks the first layer's parameter
    count by the same factor, which directly improves the privacy-utility
    tradeoff rather than merely accelerating training.

Memory: the randomised SVD solver and float32 input keep the full 531k-row
fit within a few hundred MB, instead of forming a dense float64 covariance.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

DEFAULT_CHUNK = 50_000


def fit_pca(df, n_components=72, columns=None, random_state=42):
    """
    Fit PCA on a DataFrame (should be the training fold).

    Args:
        df: DataFrame to fit PCA on.
        n_components: Number of components to retain.
        columns: Specific columns to include. If None, uses all numeric columns.
        random_state: Seed for the randomised SVD solver.

    Returns:
        Tuple of (fitted PCA object, list of columns used).
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    pca = PCA(n_components=n_components, svd_solver='randomized',
              random_state=random_state)
    pca.fit(df[columns].to_numpy(dtype=np.float32))

    explained_var = float(np.sum(pca.explained_variance_ratio_)) * 100
    print(f"  PCA: {n_components} components explain {explained_var:.1f}% of variance "
          f"(from {len(columns)} features)")

    return pca, columns


def apply_pca(df, pca, columns, keep_cols=None, chunk_size=DEFAULT_CHUNK):
    """
    Apply fitted PCA to a DataFrame, projecting in row blocks.

    Args:
        df: DataFrame to transform.
        pca: Fitted PCA object.
        columns: Columns that were used to fit PCA.
        keep_cols: Additional columns to preserve (e.g., 'isFraud').
        chunk_size: Rows per projection block.

    Returns:
        Transformed DataFrame with PCA components and any kept columns.
    """
    existing_cols = [c for c in columns if c in df.columns]

    n = len(df)
    out = np.empty((n, pca.n_components_), dtype=np.float32)
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        block = df[existing_cols].iloc[lo:hi].to_numpy(dtype=np.float32)
        out[lo:hi] = pca.transform(block).astype(np.float32)
        del block

    pca_columns = [f'PC_{i + 1:03d}' for i in range(pca.n_components_)]
    result = pd.DataFrame(out, columns=pca_columns, index=df.index)

    if keep_cols:
        for col in keep_cols:
            if col in df.columns:
                result[col] = df[col].values

    return result
