"""
Encoding module for IEEE-CIS Fraud Detection preprocessing.

- Target encoding for high-cardinality categoricals (card1-6, addr1/2, email domains)
- Ordinal encoding for low-cardinality categoricals (DeviceType)

CRITICAL: Target encoding must be computed on a VALIDATION fold ONLY —
computing on training data leaks label information.
"""

import numpy as np
import pandas as pd


def target_encode(df, columns, target_col, fold_df, smoothing=10.0):
    """
    Target encoding: replace each category with the mean of the target variable
    for that category, computed on a held-out fold.

    Uses additive smoothing to handle rare categories:
        encoded = (count * mean_cat + smoothing * global_mean) / (count + smoothing)

    Args:
        df: DataFrame to transform.
        columns: List of column names to target-encode.
        target_col: Name of the binary target column (e.g. 'isFraud').
        fold_df: The held-out fold DataFrame to compute encoding from
                 (NEVER the training set — that would leak labels).
        smoothing: Smoothing factor for rare categories (higher = more regularisation).

    Returns:
        Tuple of (transformed DataFrame, encoding_maps dict).
    """
    df = df.copy()
    encoding_maps = {}
    global_mean = fold_df[target_col].mean()

    for col in columns:
        if col not in df.columns:
            continue

        # Compute per-category statistics from the fold
        stats = fold_df.groupby(col)[target_col].agg(['mean', 'count'])
        stats.columns = ['mean_target', 'count']

        # Smoothed target encoding
        smoothed = (
            (stats['count'] * stats['mean_target'] + smoothing * global_mean)
            / (stats['count'] + smoothing)
        )

        encoding_maps[col] = {
            'map': smoothed.to_dict(),
            'global_mean': global_mean,
        }

        # Apply encoding — unseen categories get global mean
        df[col] = df[col].map(smoothed).fillna(global_mean).astype(np.float32)

    return df, encoding_maps


def apply_target_encoding(df, columns, encoding_maps):
    """
    Apply pre-computed target encoding maps to a new DataFrame.

    Args:
        df: DataFrame to transform.
        columns: List of column names to encode.
        encoding_maps: Dict of {col: {'map': dict, 'global_mean': float}}.

    Returns:
        Transformed DataFrame.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns or col not in encoding_maps:
            continue

        enc = encoding_maps[col]
        df[col] = (
            df[col].map(enc['map']).fillna(enc['global_mean']).astype(np.float32)
        )

    return df


def ordinal_encode(df, columns):
    """
    Ordinal encoding for low-cardinality categorical columns.
    Maps each unique value to an integer.

    Args:
        df: DataFrame to transform.
        columns: List of column names to ordinal-encode.

    Returns:
        Tuple of (transformed DataFrame, ordinal_maps dict).
    """
    df = df.copy()
    ordinal_maps = {}

    for col in columns:
        if col not in df.columns:
            continue

        # Create ordinal mapping
        unique_vals = df[col].dropna().unique()
        mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        ordinal_maps[col] = mapping

        # Apply mapping — NaN stays as -1
        df[col] = df[col].map(mapping).fillna(-1).astype(np.int32)

    return df, ordinal_maps


def apply_ordinal_encoding(df, columns, ordinal_maps):
    """
    Apply pre-computed ordinal encoding maps to a new DataFrame.

    Args:
        df: DataFrame to transform.
        columns: List of column names to encode.
        ordinal_maps: Dict of {col: {val: int}}.

    Returns:
        Transformed DataFrame.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns or col not in ordinal_maps:
            continue

        df[col] = df[col].map(ordinal_maps[col]).fillna(-1).astype(np.int32)

    return df

def frequency_encode(df, columns, fold_df):
    """
    Frequency encoding: replace each category with its relative frequency in the fold.
    Label-free, does not leak target information.
    """
    df = df.copy()
    encoding_maps = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        freq_map = fold_df[col].value_counts(normalize=True).to_dict()
        encoding_maps[col] = freq_map
        
        # Unseen categories get 0.0 frequency
        df[col] = df[col].map(freq_map).fillna(0.0).astype(np.float32)
        
    return df, encoding_maps

def apply_frequency_encoding(df, columns, encoding_maps):
    df = df.copy()
    for col in columns:
        if col not in df.columns or col not in encoding_maps:
            continue
        df[col] = df[col].map(encoding_maps[col]).fillna(0.0).astype(np.float32)
    return df
