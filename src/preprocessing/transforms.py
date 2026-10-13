"""
Feature transformation module for IEEE-CIS Fraud Detection preprocessing.

- Yeo-Johnson power transform (handles zero AND negative values — Box-Cox cannot)
- StandardScaler (MANDATORY for DP-SGD: normalises gradient magnitudes)

Both transformers fit on training fold only; applied to all splits.

MEMORY POLICY
-------------
The full IEEE-CIS dataset is ~531k training rows x ~480 columns. Materialising
that as float64 costs ~1.9 GB per array, and PowerTransformer needs several
such arrays at once, which exhausts RAM on a typical laptop.

Two measures keep the full dataset within a few GB:

1. `fit_sample_size` fits the Yeo-Johnson lambdas on a random row sample.
   Only 480 scalar parameters are being estimated, so 100k rows is far more
   than sufficient; the lambdas are stable well below that.
2. `chunk_size` applies the transform in row blocks, so only one block is held
   in float64 at a time. Output is written back as float32.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

DEFAULT_CHUNK = 50_000
DEFAULT_FIT_SAMPLE = 100_000


def _existing(df, columns):
    return [c for c in columns if c in df.columns]


def _transform_chunked(df, columns, transformer, chunk_size=DEFAULT_CHUNK):
    """Apply a fitted transformer in row blocks, returning float32.

    Only one `chunk_size` block exists as float64 at any moment, which is what
    keeps peak memory flat as the number of rows grows.
    """
    n = len(df)
    out = np.empty((n, len(columns)), dtype=np.float32)
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        block = transformer.transform(
            df[columns].iloc[lo:hi].to_numpy(dtype=np.float64)
        )
        # Clean and clip to prevent infinity overflow in float32
        np.nan_to_num(block, nan=0.0, posinf=1e9, neginf=-1e9, copy=False)
        np.clip(block, -1e9, 1e9, out=block)
        out[lo:hi] = block.astype(np.float32)
        del block
    return out


def power_transform(df, columns, fit_df=None, chunk_size=DEFAULT_CHUNK,
                    fit_sample_size=DEFAULT_FIT_SAMPLE, random_state=42):
    """
    Apply Yeo-Johnson power transform to numerical columns.

    Why Yeo-Johnson over Box-Cox:
    - Box-Cox requires strictly positive values
    - V-columns in IEEE-CIS contain zero and negative values
    - Yeo-Johnson handles the full real line

    Args:
        df: DataFrame to transform.
        columns: List of numerical column names.
        fit_df: DataFrame to fit on (training fold). If None, fits on df.
        chunk_size: Rows per transform block.
        fit_sample_size: Max rows used to estimate the lambdas. None uses all rows.
        random_state: Seed for the fitting subsample.

    Returns:
        Tuple of (transformed DataFrame, fitted PowerTransformer).
    """
    if fit_df is None:
        fit_df = df

    existing_cols = _existing(df, columns)
    if not existing_cols:
        return df, None

    fit_frame = fit_df[existing_cols]
    if fit_sample_size is not None and len(fit_frame) > fit_sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(fit_frame), size=fit_sample_size, replace=False)
        idx.sort()
        fit_frame = fit_frame.iloc[idx]

    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    transformer.fit(fit_frame.to_numpy(dtype=np.float64))
    del fit_frame

    df = df.copy()
    df[existing_cols] = _transform_chunked(df, existing_cols, transformer, chunk_size)

    return df, transformer


def standard_scale(df, columns, fit_df=None, chunk_size=DEFAULT_CHUNK):
    """
    Apply StandardScaler (zero mean, unit variance) to numerical columns.

    MANDATORY for DP-SGD: without scaling, gradient magnitudes vary wildly
    across features, making the global clipping bound C meaningless.
    Gradient clipping at C=1.0 only makes sense if all features are on
    comparable scales.

    StandardScaler estimates only a mean and variance per column, so it is
    fitted on the full fold (cheaply, via partial_fit over chunks) rather than
    on a subsample.

    Args:
        df: DataFrame to transform.
        columns: List of numerical column names.
        fit_df: DataFrame to fit on (training fold). If None, fits on df.
        chunk_size: Rows per block for both fitting and transforming.

    Returns:
        Tuple of (transformed DataFrame, fitted StandardScaler).
    """
    if fit_df is None:
        fit_df = df

    existing_cols = _existing(df, columns)
    if not existing_cols:
        return df, None

    scaler = StandardScaler()
    for lo in range(0, len(fit_df), chunk_size):
        hi = min(lo + chunk_size, len(fit_df))
        scaler.partial_fit(
            fit_df[existing_cols].iloc[lo:hi].to_numpy(dtype=np.float64)
        )

    df = df.copy()
    df[existing_cols] = _transform_chunked(df, existing_cols, scaler, chunk_size)

    return df, scaler


def apply_transforms(df, columns, power_transformer, standard_scaler,
                     chunk_size=DEFAULT_CHUNK):
    """
    Apply pre-fitted power transform and standard scaler to a new DataFrame.

    Args:
        df: DataFrame to transform.
        columns: List of numerical column names.
        power_transformer: Fitted PowerTransformer (or None).
        standard_scaler: Fitted StandardScaler (or None).
        chunk_size: Rows per transform block.

    Returns:
        Transformed DataFrame.
    """
    existing_cols = _existing(df, columns)
    if not existing_cols:
        return df

    df = df.copy()

    if power_transformer is not None:
        df[existing_cols] = _transform_chunked(
            df, existing_cols, power_transformer, chunk_size)

    if standard_scaler is not None:
        df[existing_cols] = _transform_chunked(
            df, existing_cols, standard_scaler, chunk_size)

    return df
