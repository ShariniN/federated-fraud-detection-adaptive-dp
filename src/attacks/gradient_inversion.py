"""Gradient inversion (iDLG-style) reconstruction from a single-sample gradient.

Demonstrates what an honest-but-curious server could recover from an
*un-noised* per-sample gradient, and that clipping + Gaussian noise at the
project's DP-SGD settings destroys the reconstruction. The attacker knows the
model weights (it is the server) and the label (iDLG recovers binary labels
analytically; here it is simply given).
"""

import numpy as np
import torch
import torch.nn as nn


def sample_gradient(model, x, y, clip=None, sigma=None, generator=None):
    """Gradient of the loss at one sample, optionally clipped to norm `clip` and noised with
    N(0, (sigma*clip)^2) per coordinate (what DP-SGD would release for a batch of one)."""
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    grads = [g.detach().clone() for g in grads]
    if clip is not None:
        norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
        factor = min(1.0, clip / (float(norm) + 1e-6))
        grads = [g * factor for g in grads]
        if sigma is not None and sigma > 0:
            grads = [g + torch.randn(g.shape, generator=generator) * sigma * clip for g in grads]
    return grads


def invert(model, target_grads, y, input_dim=72, n_iter=300, lr=0.5, seed=0, log=None):
    """Optimise a dummy input so its gradient matches target_grads. Returns (x_hat, loss history)."""
    torch.manual_seed(seed)
    x_hat = torch.randn(1, input_dim, requires_grad=True)
    opt = torch.optim.Adam([x_hat], lr=lr)
    params = list(model.parameters())
    hist = []
    for it in range(n_iter):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x_hat), y)
        g = torch.autograd.grad(loss, params, create_graph=True)
        match = sum(((a - b) ** 2).sum() for a, b in zip(g, target_grads))
        match.backward()
        opt.step()
        hist.append(float(match.item()))
        if log and it % 100 == 0:
            log(f'    iter {it:03d} gradient-match loss {hist[-1]:.6f}')
    return x_hat.detach(), hist


def reconstruction_error(x_true, x_hat):
    x_true = x_true.reshape(-1).cpu().numpy(); x_hat = x_hat.reshape(-1).cpu().numpy()
    mse = float(np.mean((x_true - x_hat) ** 2))
    corr = float(np.corrcoef(x_true, x_hat)[0, 1]) if np.std(x_hat) > 0 else 0.0
    return {'mse': mse, 'pearson_r': corr, 'baseline_mse_random': float(np.mean((x_true - np.random.randn(*x_true.shape)) ** 2))}


class iDLGAttack:
    """Compatibility facade."""

    def __init__(self, model, input_dim=72, device='cpu'):
        self.model, self.input_dim, self.device = model, input_dim, device

    def reconstruct(self, target_gradients, x_real_shape, y_real_label, n_iterations=300, lr=0.5):
        grads = [target_gradients[n] for n, _ in self.model.named_parameters()]
        y = torch.tensor([int(y_real_label)], dtype=torch.long)
        x_hat, hist = invert(self.model, grads, y, input_dim=x_real_shape[-1], n_iter=n_iterations, lr=lr)
        return x_hat.numpy(), hist
