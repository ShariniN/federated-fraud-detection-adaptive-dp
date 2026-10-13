"""
Synthetic IEEE-CIS Fraud Detection dataset generator.

Generates realistic mock datasets to test the preprocessing, model training,
and federated learning simulations without needing the real 1.5GB Kaggle dataset.
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_data(output_dir='data/raw', num_samples=10000, random_state=42):
    """
    Generate mock train_transaction.csv and train_identity.csv datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)
    
    print(f"Generating {num_samples} synthetic samples...")

    # 1. Target column: ~3.5% fraud rate
    is_fraud = rng.choice([0, 1], size=num_samples, p=[0.965, 0.035])
    
    # 2. TransactionIDs
    transaction_ids = np.arange(2987000, 2987000 + num_samples)

    # 3. Build Transaction DataFrame
    tx_data = {
        'TransactionID': transaction_ids,
        'isFraud': is_fraud,
        'TransactionDT': np.sort(rng.randint(86400, 86400 * 30, size=num_samples)), # 30 days
        'TransactionAmt': np.round(rng.exponential(scale=130, size=num_samples) + 1.0, 2),
        'ProductCD': rng.choice(['W', 'H', 'C', 'S', 'R'], size=num_samples),
        'card1': rng.randint(1000, 20000, size=num_samples),
        'card2': rng.choice([np.nan, 111, 170, 225, 321, 490, 555], size=num_samples),
        'card3': rng.choice([np.nan, 150, 185], size=num_samples),
        'card4': rng.choice([np.nan, 'visa', 'mastercard', 'american express', 'discover'], size=num_samples),
        'card5': rng.choice([np.nan, 102, 117, 166, 226], size=num_samples),
        'card6': rng.choice([np.nan, 'debit', 'credit'], size=num_samples),
        'addr1': rng.choice([np.nan, 204, 299, 325, 476], size=num_samples),
        'addr2': rng.choice([np.nan, 87], size=num_samples),
        'dist1': rng.choice([np.nan, 1, 10, 100], size=num_samples, p=[0.7, 0.1, 0.1, 0.1]),
        'dist2': rng.choice([np.nan, 2, 20, 200], size=num_samples, p=[0.9, 0.05, 0.03, 0.02]),
        'P_emaildomain': rng.choice([np.nan, 'gmail.com', 'yahoo.com', 'anonymous.com', 'hotmail.com'], size=num_samples),
        'R_emaildomain': rng.choice([np.nan, 'gmail.com', 'yahoo.com', 'hotmail.com'], size=num_samples),
    }

    # Add V-features V1 to V339 (mostly numerical, sparse, correlated)
    # Let's generate a subset of V features to keep memory small but structural
    for v in range(1, 340):
        # Correlate some V features with fraud rate
        if v % 10 == 0:
            val = rng.normal(loc=is_fraud * 2.0, scale=1.0)
        else:
            val = rng.normal(loc=0.0, scale=0.5, size=num_samples)
        
        # Add random missingness (30%)
        mask = rng.choice([True, False], size=num_samples, p=[0.3, 0.7])
        val[mask] = np.nan
        tx_data[f'V{v}'] = val

    df_tx = pd.DataFrame(tx_data)
    tx_filepath = os.path.join(output_dir, 'train_transaction.csv')
    df_tx.to_csv(tx_filepath, index=False)
    print(f"  Saved synthetic transactions: {df_tx.shape} to {tx_filepath}")

    # 4. Build Identity DataFrame
    # Only ~25% of transactions have identity information
    id_samples = num_samples // 4
    id_tx_ids = rng.choice(transaction_ids, size=id_samples, replace=False)

    id_data = {
        'TransactionID': id_tx_ids,
        'id_01': rng.uniform(-100, 0, size=id_samples),
        'id_02': rng.uniform(0, 500000, size=id_samples),
        'id_03': rng.choice([np.nan, 0, 1, 2], size=id_samples),
        'id_04': rng.choice([np.nan, 0, -1], size=id_samples),
        'DeviceType': rng.choice([np.nan, 'desktop', 'mobile'], size=id_samples),
        'DeviceInfo': rng.choice([np.nan, 'Windows', 'iOS Device', 'MacOS', 'Trident/7.0'], size=id_samples),
    }

    # Add a few more id categorical features
    for idx in range(12, 39):
        id_data[f'id_{idx}'] = rng.choice([np.nan, 'T', 'F', 'Found', 'NotFound'], size=id_samples)

    df_id = pd.DataFrame(id_data)
    id_filepath = os.path.join(output_dir, 'train_identity.csv')
    df_id.to_csv(id_filepath, index=False)
    print(f"  Saved synthetic identity: {df_id.shape} to {id_filepath}")

    return tx_filepath, id_filepath


if __name__ == "__main__":
    generate_synthetic_data()
