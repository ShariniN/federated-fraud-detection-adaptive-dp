"""Flower strategy that delegates aggregation to ``server.aggregate``.

Supports every rule in server.py (size / uncertainty / uncertainty_clipped /
trimmed_mean / median) and records the per-round weights and client epsilons
so a Flower run produces the same history structure as the in-process engine.
"""

import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from .server import aggregate


class UWFedAvgStrategy(FedAvg):
    def __init__(self, weighting='size', agg_kwargs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighting = weighting
        self.agg_kwargs = agg_kwargs or {}
        self.round_log = []

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        results = sorted(results, key=lambda r: int(r[1].metrics.get('client', 0)))
        arrays = [parameters_to_ndarrays(r.parameters) for _, r in results]
        n_list = [r.num_examples for _, r in results]
        u_list = [float(r.metrics.get('uncertainty', 0.5)) for _, r in results]
        eps_list = [float(r.metrics.get('epsilon', 0.0)) for _, r in results]
        new_arrays, weights = aggregate(self.weighting, arrays, n_list, u_list, **self.agg_kwargs)
        rec = {'round': server_round, 'weights': None if weights is None else [float(w) for w in weights],
               'epsilon': eps_list, 'uncertainty': u_list, 'n': n_list}
        self.round_log.append(rec)
        print(f"  [server r{server_round:02d}] eps={['%.3f' % e for e in eps_list]} "
              f"u={['%.3f' % u for u in u_list]} w={None if weights is None else ['%.2f' % w for w in weights]}")
        return ndarrays_to_parameters(new_arrays), {'epsilon_max': float(max(eps_list))}
