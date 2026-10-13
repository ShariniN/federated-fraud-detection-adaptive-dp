"""
Non-IID client data partitioning for simulated banking institutions.

Uses Dir(alpha) to control heterogeneity:
    alpha = 1.0  -> Near-IID (all banks similar)
    alpha = 0.5  -> Moderate non-IID (realistic banking)
    alpha = 0.1  -> Strongly non-IID (extreme heterogeneity, stress test)

Each client independently splits its local data:
    70% training | 15% calibration | 15% test

WHY NOT INDEPENDENT PER-CLASS DIRICHLET
---------------------------------------
The textbook construction draws one proportion vector per class independently.
Because the benign and fraud draws are then uncorrelated, a client can receive
a large share of the fraud class alongside almost none of the benign class.
On the full IEEE-CIS dataset that construction produced one client holding
50,545 transactions with 6 fraud cases, and another holding 7,560 transactions
of which 7,542 were fraud, a 99.8% fraud rate. Neither resembles a bank, and
neither is a meaningful federated participant.

SIZE-AND-RATE CONSTRUCTION
--------------------------
This module instead models the two properties that genuinely differ between
real institutions, and keeps both plausible:

  1. Institution SIZE. A Dir(alpha) draw sets each client's share of the
     benign pool, so banks differ substantially in transaction volume.
  2. Institution FRAUD RATE. A second Dir(alpha) draw, renormalised to mean 1,
     scales each client's fraud rate relative to the global rate, and is then
     clipped to [min_rate_multiplier, max_rate_multiplier] times that rate.

Fraud is allocated in proportion to (client size x rate multiplier), and
`min_samples_per_client` enforces an absolute floor so every client has enough
positives to train on and to run Borderline-SMOTE, which needs k+1 minority
neighbours. Heterogeneity is preserved; it is simply confined to the region
where every client is still recognisably a bank.

Set `strategy='per_class'` to reproduce the unconstrained textbook behaviour.

STRATIFIED LOCAL SPLITS
-----------------------
Each client's train/calibration/test split is stratified by label, so a client
with few fraud cases cannot end up with a calibration fold containing none.
The calibration fold selects the decision threshold, which is undefined
without positives.
"""

import numpy as np
import pandas as pd
import pickle
import os

DEFAULT_MIN_PER_CLIENT = 25
DEFAULT_MIN_RATE_MULT = 0.15
DEFAULT_MAX_RATE_MULT = 6.0
DEFAULT_MIN_CLIENT_SIZE = 5000


def _largest_remainder(weights, total):
    """Allocate `total` items across `weights`, preserving the exact sum."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    exact = weights * total
    counts = np.floor(exact).astype(np.int64)
    shortfall = int(total - counts.sum())
    if shortfall > 0:
        order = np.argsort(-(exact - counts))
        for i in range(shortfall):
            counts[order[i % len(counts)]] += 1
    return counts


def _enforce_floor(counts, floor):
    """Raise every entry to `floor`, taking the deficit from the largest."""
    counts = counts.copy()
    floor = int(max(0, floor))
    if floor == 0 or counts.sum() < floor * len(counts):
        return counts
    while True:
        deficit_idx = np.where(counts < floor)[0]
        if len(deficit_idx) == 0:
            break
        need = int(np.sum(floor - counts[deficit_idx]))
        donors = np.argsort(-counts)
        moved = 0
        for d in donors:
            if counts[d] <= floor:
                continue
            take = min(counts[d] - floor, need - moved)
            counts[d] -= take
            moved += take
            if moved >= need:
                break
        if moved == 0:
            break
        for i in deficit_idx:
            give = min(floor - counts[i], moved)
            counts[i] += give
            moved -= give
            if moved <= 0:
                break
    return counts


def _cap_fraud_rates(fraud_counts, benign_counts, max_rate, n_fraud):
    """Hold every client's fraud rate at or below `max_rate`.

    Allocating fraud in proportion to (size x multiplier) and then normalising
    to the available fraud total can push a small client above the intended
    cap, because normalisation rescales by a benign-weighted mean rather than
    by the cap itself. This redistributes the excess onto clients that still
    have headroom, repeating until every client is within the cap.
    """
    fraud_counts = fraud_counts.astype(np.int64).copy()
    for _ in range(50):
        limits = np.floor(max_rate / (1.0 - max_rate) * benign_counts).astype(np.int64)
        over = fraud_counts > limits
        if not over.any():
            break
        excess = int(np.sum(fraud_counts[over] - limits[over]))
        fraud_counts[over] = limits[over]
        headroom = limits - fraud_counts
        headroom[headroom < 0] = 0
        if headroom.sum() <= 0:
            break
        take = _largest_remainder(headroom, min(excess, int(headroom.sum())))
        fraud_counts = fraud_counts + take
    # Preserve the exact total where headroom allows
    drift = n_fraud - int(fraud_counts.sum())
    if drift > 0:
        limits = np.floor(max_rate / (1.0 - max_rate) * benign_counts).astype(np.int64)
        headroom = np.maximum(limits - fraud_counts, 0)
        if headroom.sum() > 0:
            fraud_counts += _largest_remainder(headroom, min(drift, int(headroom.sum())))
    return fraud_counts


def _allocate_size_and_rate(y, n_clients, alpha, rng, floor,
                            min_rate_mult, max_rate_mult, min_client_size):
    """Allocate indices per client by institution size and fraud rate."""
    benign_idx = np.where(y == 0)[0]
    fraud_idx = np.where(y == 1)[0]
    rng.shuffle(benign_idx)
    rng.shuffle(fraud_idx)

    global_rate = len(fraud_idx) / max(1, len(y))
    max_rate = float(np.clip(global_rate * max_rate_mult, 1e-6, 0.60))

    # 1. Institution size, from the benign pool. Every client needs a benign
    #    base large enough to be a plausible bank and to support a local split.
    size_props = rng.dirichlet([alpha] * n_clients)
    benign_counts = _largest_remainder(size_props, len(benign_idx))
    benign_counts = _enforce_floor(benign_counts, max(min_client_size, floor * 10))

    # 2. Institution fraud rate, as a multiplier on the global rate
    rate_props = rng.dirichlet([alpha] * n_clients)
    multipliers = np.clip(rate_props * n_clients, min_rate_mult, max_rate_mult)

    fraud_weights = benign_counts.astype(np.float64) * multipliers
    fraud_counts = _largest_remainder(fraud_weights, len(fraud_idx))
    fraud_counts = _cap_fraud_rates(fraud_counts, benign_counts, max_rate, len(fraud_idx))
    fraud_counts = _enforce_floor(fraud_counts, floor)

    client_indices = []
    b0 = f0 = 0
    for k in range(n_clients):
        b1 = b0 + int(benign_counts[k])
        f1 = f0 + int(fraud_counts[k])
        client_indices.append(
            np.concatenate([benign_idx[b0:b1], fraud_idx[f0:f1]])
        )
        b0, f0 = b1, f1
    return client_indices


def _allocate_per_class(y, n_clients, alpha, rng, floor):
    """Textbook independent per-class Dirichlet allocation."""
    client_indices = [[] for _ in range(n_clients)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        props = rng.dirichlet([alpha] * n_clients)
        counts = _largest_remainder(props, len(cls_idx))
        counts = _enforce_floor(counts, floor)
        cur = 0
        for k in range(n_clients):
            client_indices[k].extend(cls_idx[cur:cur + int(counts[k])].tolist())
            cur += int(counts[k])
    return [np.array(c, dtype=int) for c in client_indices]


def _stratified_split(indices, y, rng, train_frac=0.70, cal_frac=0.15):
    """Split one client's indices into train/cal/test, stratified by label."""
    train_idx, cal_idx, test_idx = [], [], []

    for cls in np.unique(y[indices]):
        cls_idx = indices[y[indices] == cls].copy()
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(round(train_frac * n))
        n_cal = int(round(cal_frac * n))
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_cal = max(1, min(n_cal, n - n_train - 1))
        train_idx.append(cls_idx[:n_train])
        cal_idx.append(cls_idx[n_train:n_train + n_cal])
        test_idx.append(cls_idx[n_train + n_cal:])

    def _cat(parts):
        return np.concatenate(parts) if parts else np.array([], dtype=int)

    train_idx, cal_idx, test_idx = _cat(train_idx), _cat(cal_idx), _cat(test_idx)
    for arr in (train_idx, cal_idx, test_idx):
        rng.shuffle(arr)
    return train_idx, cal_idx, test_idx


def dirichlet_partition(X, y, n_clients, alpha=0.5, random_state=42,
                        min_samples_per_client=DEFAULT_MIN_PER_CLIENT,
                        strategy='size_and_rate',
                        min_rate_multiplier=DEFAULT_MIN_RATE_MULT,
                        max_rate_multiplier=DEFAULT_MAX_RATE_MULT,
                        min_client_size=DEFAULT_MIN_CLIENT_SIZE,
                        verbose=True):
    """
    Partition data across simulated client banks.

    Args:
        X: Feature matrix, shape [n_samples, n_features].
        y: Binary labels, shape [n_samples].
        n_clients: Number of clients (simulated banks).
        alpha: Dirichlet concentration (lower = more heterogeneous).
        random_state: Random seed.
        min_samples_per_client: Absolute floor of minority samples per client.
        strategy: 'size_and_rate' (default) or 'per_class' (textbook).
        min_rate_multiplier: Lower clip on a client's fraud rate, as a
            multiple of the global fraud rate.
        max_rate_multiplier: Upper clip on the same.
        min_client_size: Minimum benign rows guaranteed to every client, so no
            client is too small to form a usable train/calibration/test split.
        verbose: Print a per-client summary.

    Returns:
        List of dicts, one per client:
        [{'client_id': k, 'train': (X, y), 'calibration': (X, y), 'test': (X, y)}, ...]
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)

    if strategy == 'size_and_rate' and set(np.unique(y).tolist()) <= {0, 1}:
        client_indices = _allocate_size_and_rate(
            y, n_clients, alpha, rng, min_samples_per_client,
            min_rate_multiplier, max_rate_multiplier, min_client_size)
    else:
        client_indices = _allocate_per_class(
            y, n_clients, alpha, rng, min_samples_per_client)

    global_rate = float(np.mean(y == 1)) * 100

    partitions = []
    for k in range(n_clients):
        indices = np.asarray(client_indices[k], dtype=int)
        rng.shuffle(indices)
        train_idx, cal_idx, test_idx = _stratified_split(indices, y, rng)

        partitions.append({
            'client_id': k,
            'train': (X[train_idx], y[train_idx]),
            'calibration': (X[cal_idx], y[cal_idx]),
            'test': (X[test_idx], y[test_idx]),
            # Original row indices into the federated pool. Retained so that
            # membership can be established exactly in the final phase (the
            # membership inference attack needs to know which real records were
            # trained on), and so that split disjointness can be asserted on row
            # identity rather than on feature-vector equality - the pool
            # contains genuine duplicate transactions.
            'indices': {
                'train': train_idx.copy(),
                'calibration': cal_idx.copy(),
                'test': test_idx.copy(),
            },
        })

        if verbose:
            n_fraud = int(np.sum(y[train_idx] == 1))
            n_total = len(train_idx)
            rate = n_fraud / max(1, n_total) * 100
            print(f"  Client {k}: {n_total:>7,} train | {len(cal_idx):>6,} cal | "
                  f"{len(test_idx):>6,} test | fraud {n_fraud:>5,} ({rate:4.1f}%, "
                  f"{rate / max(global_rate, 1e-9):4.1f}x global) | "
                  f"cal fraud {int(np.sum(y[cal_idx] == 1)):>5,}")

    if verbose:
        print(f"  Global fraud rate: {global_rate:.2f}%")

    return partitions


def save_partitions(partitions, output_dir):
    """Save per-client partitions to pickle files."""
    os.makedirs(output_dir, exist_ok=True)
    for partition in partitions:
        k = partition['client_id']
        filepath = os.path.join(output_dir, f'client_{k}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(partition, f)
        print(f"  Saved client {k} partition to {filepath}")


def load_partitions(partition_dir, n_clients):
    """Load per-client partitions from pickle files."""
    partitions = []
    for k in range(n_clients):
        filepath = os.path.join(partition_dir, f'client_{k}.pkl')
        with open(filepath, 'rb') as f:
            partitions.append(pickle.load(f))
    return partitions
