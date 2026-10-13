"""One training routine shared by the centralised baselines and every federated client.

Modes
-----
* non-private:   plain Adam on class-weighted cross-entropy (sigma is None).
* DP-SGD:        Opacus per-sample clipping + Gaussian noise, Poisson sampling,
                 through ``AdaptiveDPOptimizer`` so the same code path handles
                 uniform (lambda_adapt = 0) and uncertainty-adaptive clipping.
* FedProx:       optional proximal pull towards the global weights applied
                 AFTER the (noised) optimiser step. The proximal term is
                 data-independent, so adding it outside the clipped/noised
                 gradient costs no privacy and does not waste clipping budget.

Class imbalance is handled by class weighting only. Interpolative
oversampling (SMOTE) is deliberately absent from every private arm because it
inflates DP-SGD sensitivity (arXiv:2411.05733); the resampling experiment in
results/smote_comparison shows it buys <= +0.03 AUC-PR anyway.
"""

import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .dp.adaptive_optimizer import make_private_adaptive
from .dp.noise_adaptation import uncertainty_to_clip_scale
from .conformal.scores import lac_scores_torch

warnings.filterwarnings('ignore', message='.*Full backward hook.*')
warnings.filterwarnings('ignore', message='.*Secure RNG.*')
warnings.filterwarnings('ignore', message='.*drop_last.*')


def class_weights(y, device='cpu'):
    n_fraud = int(y.sum())
    n_benign = len(y) - n_fraud
    if n_fraud == 0 or n_benign == 0:
        return None
    return torch.tensor([1.0, n_benign / n_fraud], dtype=torch.float32, device=device)


def train_epochs(model, X, y, *, epochs, batch_size, lr, sigma=None, max_grad_norm=1.0,
                 lambda_adapt=0.0, min_scale=0.1, accountant=None, global_params=None, mu=0.0,
                 use_class_weights=True, generator=None, device='cpu', log=None):
    """Train ``model`` in place for ``epochs`` and return a dict of training statistics.

    sigma=None -> non-private. Otherwise DP-SGD with the given noise multiplier;
    ``accountant`` (opacus RDPAccountant) must be supplied and is stepped once
    per optimiser step so it can persist across federated rounds.
    """
    model = model.to(device)
    model.train()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    n = len(y_t)
    bs = int(min(batch_size, n))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=bs, shuffle=True, generator=generator)
    weights = class_weights(y, device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)

    private = sigma is not None and sigma > 0
    if private:
        assert accountant is not None, 'DP training needs an accountant'
        net, opt, loader, sample_rate = make_private_adaptive(
            model, loader, noise_multiplier=sigma, max_grad_norm=max_grad_norm,
            accountant=accountant, lr=lr, generator=generator)
    else:
        net = model
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sample_rate = bs / n

    if global_params is not None and mu > 0:
        g_params = [p.detach().clone().to(device) for p in global_params]
    else:
        g_params = None

    steps = 0
    losses = []
    scale_log = []
    for _ in range(int(epochs)):
        for xb, yb in loader:
            if len(yb) == 0:
                continue
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = net(xb)
            loss = criterion(logits, yb)
            if private and lambda_adapt > 0:
                with torch.no_grad():
                    u = lac_scores_torch(torch.softmax(logits, dim=1), yb)
                    opt.set_sample_scales(uncertainty_to_clip_scale(u, lambda_adapt, min_scale))
            loss.backward()
            opt.step()
            if private:
                scale_log.append(opt.last_mean_scale)
            if g_params is not None:                       # FedProx pull, post-noise
                with torch.no_grad():
                    for p, g in zip(model.parameters(), g_params):
                        p.sub_(lr * mu * (p - g))
            losses.append(float(loss.item()))
            steps += 1

    if private:
        net.remove_hooks() if hasattr(net, 'remove_hooks') else None
    stats = {
        'steps': steps, 'sample_rate': float(sample_rate),
        'train_loss': float(np.mean(losses)) if losses else None,
        'mean_clip_scale': float(np.mean(scale_log)) if scale_log else 1.0,
        'private': private, 'sigma': float(sigma) if private else None,
    }
    return stats


def state_to_numpy(model):
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def numpy_to_state(model, arrays):
    sd = model.state_dict()
    new = {k: torch.as_tensor(a, dtype=v.dtype) for (k, v), a in zip(sd.items(), arrays)}
    model.load_state_dict(new, strict=True)
    return model


def clone_model(model):
    return copy.deepcopy(model)
