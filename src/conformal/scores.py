"""Conformal nonconformity scores.

Two scores are provided:

* **LAC** (least-ambiguous set-valued classifier, Sadinle et al. 2019):
  ``s(x, y) = 1 - p_y(x)``. Bounded in [0, 1] and monotone in the model's
  confidence in the true label, which is what makes it the right *uncertainty
  signal* for the adaptive-DP mechanism (Component 1) and for the client
  uncertainty statistic (Component 2). Because it is bounded by a fixed
  constant, no data-dependent normalisation is needed - this is what keeps the
  sensitivity argument in ``src/dp`` exact.

* **RAPS** (Angelopoulos et al. 2021): cumulative probability mass of the
  classes ranked at or above the true class, plus a rank penalty. This is the
  score named in the proposal and is used for the prediction sets of the review
  tier. For a binary problem it is *not* monotone in confidence when the true
  label is the top class (p_true = 0.99 scores 0.99, p_true = 0.55 scores 0.55),
  which is why it is not used as the adaptation signal.

All functions are vectorised and operate on a probability matrix ``probs``
of shape [n, n_classes] so they can be reused inside the DP training loop
(where the probabilities come from the same forward pass as the loss).
"""

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------- torch (in-loop)

def lac_scores_torch(probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - p_true, shape [n]. probs: [n, C] softmax outputs."""
    return 1.0 - probs.gather(1, y.view(-1, 1)).squeeze(1)


def raps_scores_torch(probs: torch.Tensor, y: torch.Tensor, lambda_reg=0.01, k_reg=1) -> torch.Tensor:
    """RAPS score of the true label: cumulative mass down to y plus lambda*(rank+1-k_reg)_+."""
    sorted_p, order = torch.sort(probs, dim=1, descending=True)
    cum = torch.cumsum(sorted_p, dim=1)
    rank = (order == y.view(-1, 1)).float().argmax(dim=1)          # 0-based rank of true label
    mass = cum.gather(1, rank.view(-1, 1)).squeeze(1)
    penalty = lambda_reg * torch.clamp(rank.float() + 1 - k_reg, min=0.0)
    return mass + penalty


# ----------------------------------------------------------------- numpy (offline)

def lac_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.0 - probs[np.arange(len(y)), y]


def raps_scores(probs: np.ndarray, y: np.ndarray, lambda_reg=0.01, k_reg=1) -> np.ndarray:
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rank = np.argmax(order == y.reshape(-1, 1), axis=1)
    mass = cum[np.arange(len(y)), rank]
    penalty = lambda_reg * np.maximum(rank + 1 - k_reg, 0)
    return mass + penalty


def score_all_labels(probs: np.ndarray, score='lac', lambda_reg=0.01) -> np.ndarray:
    """Score every candidate label: returns [n, C] with s(x, c) for each class c."""
    n, C = probs.shape
    out = np.empty((n, C), dtype=np.float64)
    for c in range(C):
        yc = np.full(n, c, dtype=np.int64)
        out[:, c] = lac_scores(probs, yc) if score == 'lac' else raps_scores(probs, yc, lambda_reg)
    return out


# ----------------------------------------------------------------- model helpers

def predict_proba(model, X, device='cpu', batch_size=8192) -> np.ndarray:
    """Softmax probabilities [n, 2] for a numpy feature matrix, in eval mode."""
    raw = getattr(model, '_module', model)
    raw.eval()
    out = np.empty((len(X), 2), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[lo:lo + batch_size], dtype=torch.float32, device=device)
            out[lo:lo + batch_size] = F.softmax(raw(xb), dim=1).cpu().numpy()
    return out


def compute_raps_scores(model, x, y_true, lambda_reg=0.01):
    """Backwards-compatible wrapper (torch tensors in, tensor of scores out)."""
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
        return raps_scores_torch(probs, y_true, lambda_reg=lambda_reg).float()
