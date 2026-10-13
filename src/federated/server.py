"""Server-side aggregation rules.

* size                 FedAvg: alpha_k = n_k / N                        (McMahan et al. 2017)
* uncertainty          UW-FedAvg: alpha_k ∝ n_k / (u_k + u_floor)        (Component 2)
* uncertainty_clipped  UW-FedAvg with each weight clipped to
                       [clip_lo, clip_hi] x (n_k / N) - the hardened variant
                       that limits how much a client can gain by under-
                       reporting its uncertainty
* trimmed_mean         coordinate-wise trimmed mean (drops the top and bottom
                       trim_ratio fraction of clients per coordinate)
* median               coordinate-wise median

All rules operate on lists of numpy weight arrays (one list per client).
"""

import numpy as np


def compute_weights(rule, n_list, u_list=None, *, u_floor=0.05, clip_lo=0.5, clip_hi=2.0):
    n = np.asarray(n_list, dtype=np.float64)
    base = n / n.sum()
    if rule == 'size' or u_list is None:
        return base
    u = np.clip(np.asarray(u_list, dtype=np.float64), 0.0, 1.0)
    w = base / (u + u_floor)
    w = w / w.sum()
    if rule == 'uncertainty':
        return w
    if rule == 'uncertainty_clipped':
        w = np.clip(w, clip_lo * base, clip_hi * base)
        return w / w.sum()
    raise ValueError(f'unknown weighting rule {rule!r}')


def weighted_average(client_arrays, weights):
    weights = np.asarray(weights, dtype=np.float64)
    return [sum(w * arrs[i] for w, arrs in zip(weights, client_arrays)).astype(client_arrays[0][i].dtype)
            for i in range(len(client_arrays[0]))]


def trimmed_mean(client_arrays, trim_ratio=0.2):
    k = len(client_arrays)
    t = int(np.floor(trim_ratio * k))
    out = []
    for i in range(len(client_arrays[0])):
        stack = np.stack([arrs[i] for arrs in client_arrays], axis=0)
        s = np.sort(stack, axis=0)
        out.append(s[t:k - t].mean(axis=0).astype(stack.dtype) if k - 2 * t > 0 else s.mean(axis=0))
    return out


def coordinate_median(client_arrays):
    return [np.median(np.stack([arrs[i] for arrs in client_arrays], axis=0), axis=0).astype(client_arrays[0][i].dtype)
            for i in range(len(client_arrays[0]))]


def aggregate(rule, client_arrays, n_list, u_list=None, **kw):
    """Return (new_global_arrays, weights_used_or_None)."""
    if rule == 'trimmed_mean':
        return trimmed_mean(client_arrays, kw.get('trim_ratio', 0.2)), None
    if rule == 'median':
        return coordinate_median(client_arrays), None
    w = compute_weights(rule, n_list, u_list, u_floor=kw.get('u_floor', 0.05),
                        clip_lo=kw.get('clip_lo', 0.5), clip_hi=kw.get('clip_hi', 2.0))
    return weighted_average(client_arrays, w), w


# legacy names used by older scripts
def aggregate_fedavg(global_model, client_models, client_weights):
    import torch
    sd = global_model.state_dict()
    for key in sd:
        sd[key] = torch.stack([m.state_dict()[key] * w for m, w in zip(client_models, client_weights)]).sum(0)
    global_model.load_state_dict(sd)
    return global_model
