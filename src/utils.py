"""Small shared helpers: seeding, timing, JSON-safe conversion, data loading."""

import os
import json
import time
import random
import pickle

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run is repeatable on the same machine."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def json_safe(obj):
    """Recursively convert numpy / torch scalars and arrays so json.dump accepts them."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(json_safe(obj), f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0


# --------------------------------------------------------------------------- data

def load_partitions(processed_dir='data/processed_clean', n_clients=5):
    """Load the per-client partitions written by the preprocessing pipeline.

    Returns a list of dicts with keys train / calibration / test (each (X, y)),
    plus 'indices' (original pool row indices) and 'client_id'.
    """
    parts = []
    for cid in range(n_clients):
        path = os.path.join(processed_dir, 'partitions', f'client_{cid}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found - run src/preprocessing/pipeline.py first')
        with open(path, 'rb') as f:
            d = pickle.load(f)
        for split in ('train', 'calibration', 'test'):
            X, y = d[split]
            d[split] = (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64))
        parts.append(d)
    return parts


def load_holdout(processed_dir='data/processed_clean'):
    with open(os.path.join(processed_dir, 'public_holdout.pkl'), 'rb') as f:
        pub = pickle.load(f)
    return np.asarray(pub['X'], dtype=np.float32), np.asarray(pub['y'], dtype=np.int64)


def load_pool(processed_dir='data/processed_clean'):
    """The full pre-partition training pool (531,486 x 72) - used for re-partitioning sweeps."""
    with open(os.path.join(processed_dir, 'federated_pool.pkl'), 'rb') as f:
        pool = pickle.load(f)
    return np.asarray(pool['X'], dtype=np.float32), np.asarray(pool['y'], dtype=np.int64)


def pooled(parts, split):
    X = np.concatenate([p[split][0] for p in parts], axis=0)
    y = np.concatenate([p[split][1] for p in parts], axis=0)
    return X, y
