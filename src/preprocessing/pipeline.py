"""
End-to-end preprocessing pipeline for IEEE-CIS Fraud Detection dataset.

Coordinates:
1. Loading and merging transaction and identity files (downcast to float32).
2. Temporal sort, then splitting the EARLIEST 90% as the federated training
   pool and the LATEST 10% as the public hold-out.
3. Fitting every preprocessing step strictly on the federated training pool:
   - Imputation (median for numericals, mode for categoricals + missing flags)
   - Categorical frequency/ordinal encoding
   - Yeo-Johnson power transform
   - Standard scaling
   - PCA to 72 components
4. Applying these fitted transforms to both the federated and hold-out sets.
5. Partitioning the federated dataset among clients using a Dirichlet
   distribution with a per-client minority floor.
6. Creating per-client stratified splits (70% train, 15% calibration, 15% test).
7. Saving RAW (un-resampled) partitions by default.

DESIGN NOTES
------------
Temporal direction: the hold-out is drawn from the most recent transactions
and the model trains on everything before it, so evaluation measures forward
generalisation, matching how a deployed fraud model is actually used.

Fitting fold: all transforms — including the frequency encoder — are fitted on
the training pool only. Fitting any transform on the hold-out would let
evaluation-fold statistics inform the training features.

Resampling: `apply_smote` defaults to False so the saved partitions preserve
the true class distribution. Each experiment applies resampling exactly once,
which keeps the resampling comparison a controlled experiment and prevents
synthetic samples being generated from synthetic parents.
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing.imputation import impute_numerical, impute_categorical, apply_imputation
from src.preprocessing.encoding import frequency_encode, ordinal_encode, apply_frequency_encoding, apply_ordinal_encoding
from src.preprocessing.transforms import power_transform, standard_scale, apply_transforms
from src.preprocessing.pca import fit_pca, apply_pca
from src.preprocessing.smote import apply_borderline_smote
from src.preprocessing.partition import dirichlet_partition, save_partitions

CATEGORICAL_BASE = [
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
    'DeviceType', 'DeviceInfo', 'ProductCD',
]
ORDINAL_BASE = ['DeviceType', 'card4', 'card6', 'ProductCD']


def _downcast(df):
    """float64 -> float32 halves every downstream array."""
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    return df


def _classify_columns(df_features):
    categorical_cols = list(CATEGORICAL_BASE)
    id_cat_cols = [f'id_{i:02d}' for i in range(12, 39)] + ['id_12']
    categorical_cols.extend([c for c in id_cat_cols if c in df_features.columns])
    categorical_cols.extend(
        df_features.select_dtypes(include=['object', 'category']).columns.tolist()
    )

    seen = set()
    categorical_cols = [
        c for c in categorical_cols
        if c in df_features.columns and not (c in seen or seen.add(c))
    ]

    ordinal_cols = [c for c in ORDINAL_BASE if c in df_features.columns]
    freq_encode_cols = [c for c in categorical_cols if c not in ordinal_cols]
    numerical_cols = [
        c for c in df_features.columns
        if c not in categorical_cols and c != 'TransactionDT'
    ]
    return categorical_cols, ordinal_cols, freq_encode_cols, numerical_cols


def run_full_pipeline(raw_dir, output_dir, n_clients=5, alpha=0.5,
                      subsample_fraction=None, random_state=42,
                      apply_smote=False, n_components=72,
                      min_samples_per_client=25, holdout_fraction=0.10):
    """
    Run the full preprocessing pipeline.

    Args:
        raw_dir: Directory containing train_transaction.csv / train_identity.csv.
        output_dir: Directory to write partitions and the public hold-out.
        n_clients: Number of simulated client banks.
        alpha: Dirichlet concentration (lower = more heterogeneous).
        subsample_fraction: Fraction of raw rows to use. None uses everything.
        random_state: Seed.
        apply_smote: Resample client training folds in-pipeline. Leave False so
            each experiment applies resampling exactly once.
        n_components: PCA target dimensionality.
        min_samples_per_client: Per-class floor guaranteed to every client.
        holdout_fraction: Fraction of the most recent transactions held out.
    """
    print("Step 1: Loading raw IEEE-CIS datasets...")
    tx_path = os.path.join(raw_dir, "train_transaction.csv")
    id_path = os.path.join(raw_dir, "train_identity.csv")

    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Missing transaction file: {tx_path}")

    df = _downcast(pd.read_csv(tx_path))
    print(f"  Loaded transactions: {df.shape}")

    if os.path.exists(id_path):
        df_id = _downcast(pd.read_csv(id_path))
        print(f"  Loaded identity: {df_id.shape}")
        df = pd.merge(df, df_id, on="TransactionID", how="left")
        print(f"  Merged dataset shape: {df.shape}")
        del df_id
        gc.collect()
    else:
        print("  WARNING: Identity file not found. Preprocessing with transaction data only.")

    if subsample_fraction is not None and subsample_fraction < 1.0:
        print(f"  Subsampling raw dataset to fraction: {subsample_fraction}")
        df = df.sample(frac=subsample_fraction, random_state=random_state).reset_index(drop=True)
        print(f"  Subsampled dataset shape: {df.shape}")

    print("Step 2: Temporal sort and train/hold-out split...")
    df = df.sort_values("TransactionDT")

    y = df['isFraud'].values.astype(np.int64)
    df_features = df.drop(columns=['isFraud', 'TransactionID'])
    del df
    gc.collect()

    categorical_cols, ordinal_cols, freq_encode_cols, numerical_cols = \
        _classify_columns(df_features)
    print(f"  Features: {len(numerical_cols)} numerical, "
          f"{len(freq_encode_cols)} frequency-encode, {len(ordinal_cols)} ordinal")

    # Earliest (1 - holdout_fraction) trains; the most recent holdout_fraction
    # is held out, so evaluation measures forward-in-time generalisation.
    n = len(df_features)
    n_fed = int((1.0 - holdout_fraction) * n)

    df_fed = df_features.iloc[:n_fed].copy()
    y_fed = y[:n_fed]
    df_pub = df_features.iloc[n_fed:].copy()
    y_pub = y[n_fed:]
    del df_features
    gc.collect()
    print(f"  Training pool: {df_fed.shape[0]} rows (earliest)  |  "
          f"Hold-out: {df_pub.shape[0]} rows (most recent)")

    print("Step 3: Imputing missing values (fitted on the training pool)...")
    df_fed, num_imputer = impute_numerical(df_fed, numerical_cols)
    df_fed, cat_imputer = impute_categorical(df_fed, categorical_cols)
    df_pub = apply_imputation(
        df_pub, num_imputer, cat_imputer, numerical_cols, categorical_cols
    )
    del num_imputer, cat_imputer
    gc.collect()

    missing_cols = [f'{c}_missing' for c in categorical_cols
                    if f'{c}_missing' in df_fed.columns]
    numerical_cols_extended = numerical_cols + missing_cols

    print("Step 4: Encoding categorical variables (fitted on the training pool)...")
    df_fed, ord_maps = ordinal_encode(df_fed, ordinal_cols)
    df_pub = apply_ordinal_encoding(df_pub, ordinal_cols, ord_maps)

    df_fed, freq_maps = frequency_encode(df_fed, freq_encode_cols, df_fed)
    df_pub = apply_frequency_encoding(df_pub, freq_encode_cols, freq_maps)
    del ord_maps, freq_maps
    gc.collect()

    print("Step 5: Applying power transform and scaling...")
    all_scale_cols = numerical_cols_extended + freq_encode_cols + ordinal_cols

    df_fed, pt = power_transform(df_fed, all_scale_cols, random_state=random_state)
    df_fed, scaler = standard_scale(df_fed, all_scale_cols)
    df_pub = apply_transforms(df_pub, all_scale_cols, pt, scaler)
    del pt, scaler
    gc.collect()

    print(f"Step 6: Performing PCA to {n_components} components...")
    pca, pca_features = fit_pca(df_fed, n_components=n_components,
                                columns=all_scale_cols, random_state=random_state)

    X_fed_pca = apply_pca(df_fed, pca, pca_features).to_numpy(dtype=np.float32)
    del df_fed
    gc.collect()
    X_pub_pca = apply_pca(df_pub, pca, pca_features).to_numpy(dtype=np.float32)
    del df_pub, pca
    gc.collect()

    # Persist the pre-partition pool so the federated phase can re-partition
    # under different heterogeneity settings without re-running preprocessing.
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "federated_pool.pkl"), 'wb') as f:
        pickle.dump({'X': X_fed_pca, 'y': y_fed}, f)
    print(f"  Saved federated pool: {X_fed_pca.shape[0]} rows")

    print(f"Step 7: Partitioning federated data across {n_clients} clients "
          f"(alpha={alpha}, floor={min_samples_per_client})...")
    partitions = dirichlet_partition(
        X_fed_pca, y_fed, n_clients, alpha=alpha, random_state=random_state,
        min_samples_per_client=min_samples_per_client,
    )

    if apply_smote:
        print("Step 8: Applying local Borderline-SMOTE on training splits...")
        for client in partitions:
            X_train, y_train = client['train']
            client['train'] = apply_borderline_smote(
                X_train, y_train, sampling_strategy=0.2, k_neighbors=5,
                random_state=random_state
            )
    else:
        print("Step 8: Skipping SMOTE - saving raw partitions "
              "(each experiment resamples exactly once).")

    print(f"Step 9: Saving partitions to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    save_partitions(partitions, os.path.join(output_dir, "partitions"))

    with open(os.path.join(output_dir, "public_holdout.pkl"), 'wb') as f:
        pickle.dump({'X': X_pub_pca, 'y': y_pub}, f)
    print(f"  Saved public hold-out set: {X_pub_pca.shape[0]} rows, "
          f"fraud rate {y_pub.mean() * 100:.2f}%")

    total = sum(len(p['train'][1]) for p in partitions)
    fraud = sum(int(p['train'][1].sum()) for p in partitions)
    print(f"  Pooled training set: {total} rows, {fraud} fraud "
          f"({fraud / max(1, total) * 100:.2f}%)")
    print("Preprocessing completed successfully!")

    return partitions


if __name__ == '__main__':
    raw_dir = os.path.join('data', 'raw')
    output_dir = os.path.join('data', 'processed_clean')
    os.makedirs(output_dir, exist_ok=True)
    run_full_pipeline(
        raw_dir=raw_dir,
        output_dir=output_dir,
        n_clients=5,
        alpha=0.5,
        subsample_fraction=None,   # full 590,540-transaction dataset
        random_state=42,
        apply_smote=False,
        min_samples_per_client=25,
    )
