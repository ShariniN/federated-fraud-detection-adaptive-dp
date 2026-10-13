"""
Imputation module for IEEE-CIS Fraud Detection preprocessing.

Handles missing value imputation:
- Numerical: median imputation (robust to fraud outlier skew)
- Categorical: mode imputation + 'missing' category flag

CRITICAL: Fit ONLY on training fold; apply (transform) to val/test.

MEMORY POLICY
-------------
Two measures keep the full 590k-row dataset within a few GB:

1. Imputed values are written back in row chunks as float32, so a full float64
   copy of the numerical block (~1.9 GB on the full dataset) is never held.
2. The categorical missing-indicator flags are built as a single DataFrame and
   concatenated once, rather than inserted column by column. Inserting ~45
   columns one at a time fragments the frame and triggers a full copy per
   insertion, which is both slow and memory-hungry.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

DEFAULT_CHUNK = 50_000


def _existing(df, columns):
    return [c for c in columns if c in df.columns]


def _impute_chunked(df, columns, imputer, chunk_size=DEFAULT_CHUNK, dtype=np.float32):
    """Write imputed values back in row blocks to keep peak memory flat."""
    positions = [df.columns.get_loc(c) for c in columns]
    for lo in range(0, len(df), chunk_size):
        hi = min(lo + chunk_size, len(df))
        block = imputer.transform(df[columns].iloc[lo:hi])
        if dtype is not None:
            block = block.astype(dtype)
        df.iloc[lo:hi, positions] = block
        del block
    return df


def make_missing_flags(df, columns):
    """Build all '<col>_missing' indicator columns as one frame.

    Returns a DataFrame aligned to df's index. Concatenating this once is far
    cheaper than inserting the columns individually.
    """
    existing_cols = _existing(df, columns)
    if not existing_cols:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(
        {f'{c}_missing': df[c].isna().astype(np.int8) for c in existing_cols},
        index=df.index,
    )


def impute_numerical(df, columns, fit_df=None, chunk_size=DEFAULT_CHUNK):
    """
    Impute numerical columns using median strategy.

    Args:
        df: DataFrame to transform.
        columns: List of numerical column names.
        fit_df: DataFrame to fit the imputer on (training fold).
                If None, fits on df itself (only valid during initial pipeline).
        chunk_size: Rows per write-back block.

    Returns:
        Tuple of (transformed DataFrame, fitted imputer).
    """
    if fit_df is None:
        fit_df = df

    existing_cols = _existing(df, columns)
    if not existing_cols:
        return df, None

    imputer = SimpleImputer(strategy='median')
    imputer.fit(fit_df[existing_cols])

    df = df.copy()
    df = _impute_chunked(df, existing_cols, imputer, chunk_size)

    return df, imputer


def impute_categorical(df, columns, fit_df=None, add_flags=True):
    """
    Impute categorical columns using mode strategy.
    Also adds a binary '_missing' flag column for each imputed column.

    The flags are computed BEFORE imputation (they record where data was
    genuinely absent) and appended in a single concat.

    Args:
        df: DataFrame to transform.
        columns: List of categorical column names.
        fit_df: DataFrame to fit the imputer on (training fold).
                If None, fits on df itself.
        add_flags: Whether to append the '_missing' indicator columns.

    Returns:
        Tuple of (transformed DataFrame, fitted imputer).
    """
    if fit_df is None:
        fit_df = df

    existing_cols = _existing(df, columns)
    if not existing_cols:
        return df, None

    flags = make_missing_flags(df, existing_cols) if add_flags else None

    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(fit_df[existing_cols])

    df = df.copy()
    df[existing_cols] = imputer.transform(df[existing_cols])

    if flags is not None and not flags.empty:
        df = pd.concat([df, flags], axis=1)

    return df, imputer


def apply_imputation(df, numerical_imputer, categorical_imputer,
                     numerical_cols, categorical_cols,
                     chunk_size=DEFAULT_CHUNK, add_flags=True):
    """
    Apply pre-fitted imputers to a new DataFrame (val/test splits).

    Mirrors the ordering used when fitting: numerical first, then categorical,
    with the missing-indicator flags appended last.

    Args:
        df: DataFrame to transform.
        numerical_imputer: Fitted SimpleImputer for numerical columns.
        categorical_imputer: Fitted SimpleImputer for categorical columns.
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        chunk_size: Rows per write-back block.
        add_flags: Whether to append the '_missing' indicator columns.

    Returns:
        Transformed DataFrame.
    """
    df = df.copy()

    if numerical_imputer is not None:
        existing_num = _existing(df, numerical_cols)
        if existing_num:
            df = _impute_chunked(df, existing_num, numerical_imputer, chunk_size)

    if categorical_imputer is not None:
        existing_cat = _existing(df, categorical_cols)
        if existing_cat:
            flags = make_missing_flags(df, existing_cat) if add_flags else None
            df[existing_cat] = categorical_imputer.transform(df[existing_cat])
            if flags is not None and not flags.empty:
                df = pd.concat([df, flags], axis=1)

    return df
