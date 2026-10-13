"""Flower NumPyClient adapter around ``BankClient``.

Flower's simulation re-creates client objects every round, so the per-client
accountant cannot live in the object. It is rebuilt exactly from
(sigma_k, q_k, steps_per_round x rounds_completed) before each local fit -
RDP composition is additive in steps, so this reproduces the in-process
engine's epsilon to the last digit.
"""

import numpy as np
import flwr as fl

from .client import BankClient
from ..dp.accountant import steps_per_epoch, opacus_sample_rate


class FraudFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, partition, client_cfg, seed=0, device='cpu', malicious=None):
        self.bank = BankClient(cid, partition, client_cfg, seed=seed, device=device, malicious=malicious)
        self.cfg = client_cfg

    def get_parameters(self, config):
        from ..models.mlp import FraudMLP
        from ..training import state_to_numpy
        return state_to_numpy(FraudMLP(input_dim=self.bank.input_dim))

    def _restore_accountant(self, server_round):
        if not self.bank.private:
            return
        done = int(server_round) - 1
        if done > 0:
            steps = steps_per_epoch(self.bank.n, self.cfg['batch_size']) * self.cfg['local_epochs'] * done
            self.bank.accountant.history = [(float(self.bank.sigma),
                                             opacus_sample_rate(self.bank.n, self.cfg['batch_size']), steps)]
            self.bank.prev_epsilon = self.bank.epsilon()

    def fit(self, parameters, config):
        rnd = int(config.get('server_round', 1))
        self._restore_accountant(rnd)
        arrays, n, metrics = self.bank.fit([np.asarray(p) for p in parameters], rnd)
        scalar = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and v is not None else 0.0)
                  for k, v in metrics.items() if k in ('epsilon', 'uncertainty', 'uncertainty_raw',
                                                        'train_loss', 'mean_clip_scale', 'update_norm')}
        scalar['client'] = int(self.bank.cid)
        return arrays, int(n), scalar

    def evaluate(self, parameters, config):
        m = self.bank.evaluate_local([np.asarray(p) for p in parameters])
        return float(1.0 - (m['auc_pr'] or 0.0)), int(len(self.bank.y_test)), {'auc_pr': float(m['auc_pr'] or 0.0)}
