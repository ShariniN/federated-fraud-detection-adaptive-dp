"""A simulated bank: local data, local DP training, local conformal statistics.

Each client owns
* its train / calibration / test folds (never leave the object),
* one ``RDPAccountant`` for the whole federation (epsilon composes over rounds),
* its own noise multiplier sigma_k, solved budget-first from its sampling rate,
* an optional adversarial behaviour for the robustness experiments.

``fit`` returns the locally trained weights, the number of training rows and
a metrics dict that includes the client's *noised* mean conformal uncertainty
(Component 2 input) and its cumulative epsilon.
"""

import numpy as np
import torch
from opacus.accountants import RDPAccountant

from ..models.mlp import FraudMLP
from ..training import train_epochs, state_to_numpy, numpy_to_state
from ..dp.accountant import calibrate_sigma, noisy_mean_bounded, opacus_sample_rate
from ..conformal.scores import predict_proba, lac_scores
from ..evaluation.metrics import evaluate_model


class BankClient:
    def __init__(self, cid, partition, cfg, *, seed=0, device='cpu', malicious=None):
        """
        cfg keys used: batch_size, local_epochs, lr, max_grad_norm, target_epsilon, delta,
                       private (bool), sigma (optional fixed), lambda_adapt, min_scale,
                       mu (FedProx), eps_uncertainty, n_rounds, use_class_weights
        malicious: None | 'label_flip' | 'model_replacement'
        """
        self.cid = cid
        self.X_train, self.y_train = partition['train']
        self.X_cal, self.y_cal = partition['calibration']
        self.X_test, self.y_test = partition['test']
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self.malicious = malicious
        self.rng = np.random.default_rng(seed * 1000 + cid)
        self.n = len(self.y_train)
        self.input_dim = self.X_train.shape[1]

        self.private = bool(cfg.get('private', True))
        self.accountant = RDPAccountant() if self.private else None
        self.sigma = None
        if self.private:
            if cfg.get('sigma') is not None:
                self.sigma = float(cfg['sigma'])
            else:
                self.sigma = calibrate_sigma(cfg['target_epsilon'], cfg['delta'], self.n,
                                             cfg['batch_size'], cfg['local_epochs'] * cfg['n_rounds'])
        self.prev_epsilon = 0.0
        self.round_log = []

        if self.malicious == 'label_flip':
            self.y_train = 1 - self.y_train

    # ------------------------------------------------------------------ privacy
    def epsilon(self):
        if not self.private or not self.accountant.history:
            return 0.0 if self.private else float('inf')
        return float(self.accountant.get_epsilon(delta=self.cfg['delta']))

    # ------------------------------------------------------------------ training
    def fit(self, global_arrays, rnd):
        cfg = self.cfg
        model = numpy_to_state(FraudMLP(input_dim=self.input_dim), global_arrays).to(self.device)
        global_model = numpy_to_state(FraudMLP(input_dim=self.input_dim), global_arrays).to(self.device)
        gen = torch.Generator().manual_seed(self.seed * 100003 + self.cid * 1009 + rnd)

        stats = train_epochs(
            model, self.X_train, self.y_train,
            epochs=cfg['local_epochs'], batch_size=cfg['batch_size'], lr=cfg['lr'],
            sigma=self.sigma if self.private else None, max_grad_norm=cfg.get('max_grad_norm', 1.0),
            lambda_adapt=cfg.get('lambda_adapt', 0.0), min_scale=cfg.get('min_scale', 0.1),
            accountant=self.accountant, global_params=list(global_model.parameters()),
            mu=cfg.get('mu', 0.0), use_class_weights=cfg.get('use_class_weights', True),
            generator=gen, device=self.device)

        eps = self.epsilon()
        if self.private:
            assert eps > self.prev_epsilon - 1e-9, f'accountant reset detected on client {self.cid}'
            self.prev_epsilon = eps

        # Component 2 input: mean conformal (LAC) nonconformity on the local calibration fold,
        # released with Laplace noise (sensitivity 1/n_cal).
        probs_cal = predict_proba(model, self.X_cal, self.device)
        u_raw = float(lac_scores(probs_cal, self.y_cal).mean())
        u_noisy = noisy_mean_bounded(lac_scores(probs_cal, self.y_cal), cfg.get('eps_uncertainty', 0.0),
                                     self.rng)

        arrays = state_to_numpy(model)
        if self.malicious == 'model_replacement':
            boost = float(cfg.get('replacement_boost', 5.0))
            arrays = [g + boost * (a - g) for a, g in zip(arrays, global_arrays)]
            u_noisy = 0.0                                  # claims perfect confidence to capture the aggregate
        update_norm = float(np.sqrt(sum(float(((a - g) ** 2).sum()) for a, g in zip(arrays, global_arrays))))

        metrics = {
            'client': self.cid, 'round': rnd, 'n_train': self.n, 'n_fraud': int(self.y_train.sum()),
            'sigma': self.sigma, 'sample_rate': stats['sample_rate'], 'steps_total': (
                int(sum(k for _, _, k in self.accountant.history)) if self.private else None),
            'epsilon': eps, 'train_loss': stats['train_loss'],
            'mean_clip_scale': stats['mean_clip_scale'],
            'uncertainty_raw': u_raw, 'uncertainty': u_noisy, 'update_norm': update_norm,
            'malicious': self.malicious,
        }
        self.round_log.append(metrics)
        return arrays, self.n, metrics

    # ------------------------------------------------------------------ evaluation
    def evaluate_local(self, global_arrays, per_bank_threshold=True):
        """Global model on this bank's own test fold, with the bank's own calibrated threshold."""
        model = numpy_to_state(FraudMLP(input_dim=self.input_dim), global_arrays).to(self.device)
        if per_bank_threshold:
            return evaluate_model(model, self.X_test, self.y_test, self.X_cal, self.y_cal, self.device)
        return evaluate_model(model, self.X_test, self.y_test, device=self.device, tau=0.5)

    def privacy_row(self):
        return {'client': self.cid, 'n_train': self.n, 'fraud_rate': float(self.y_train.mean()),
                'batch': self.cfg['batch_size'], 'q': opacus_sample_rate(self.n, self.cfg['batch_size']),
                'sigma': self.sigma, 'steps': int(sum(k for _, _, k in self.accountant.history)) if self.private else None,
                'epsilon': self.epsilon(), 'delta': self.cfg['delta']}
