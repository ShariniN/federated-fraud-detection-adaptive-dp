# -*- coding: utf-8 -*-
"""Run a federated configuration through the Flower framework (Ray-backed simulation).

Uses exactly the same BankClient training code and aggregation rules as the
in-process engine; Flower provides the client/server orchestration. The
per-client accountant is rebuilt deterministically each round (see
src/federated/fl_client.py). Intended as the framework demonstration - for
grids use run_federated.py / run_ablation.py, which are faster and seeded.

    python experiments/run_flower.py --config experiments/configs/full_system.yaml --rounds 3
"""
import argparse
import logging
import os
import warnings

import numpy as np
import flwr as fl

from _common import Logger, load_yaml, save_json, RESULTS_ROOT
from src.utils import load_partitions, load_holdout, pooled, set_seed
from src.models.mlp import FraudMLP
from src.training import state_to_numpy, numpy_to_state
from src.federated.simulation import merge_cfg, client_cfg
from src.federated.fl_client import FraudFlowerClient
from src.federated.fl_strategy import UWFedAvgStrategy
from src.evaluation.metrics import evaluate_model

warnings.filterwarnings('ignore')
logging.getLogger('flwr').setLevel(logging.ERROR)
OUT = os.path.join(RESULTS_ROOT, 'federated_flower')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join('experiments', 'configs', 'full_system.yaml'))
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    cfg = merge_cfg(load_yaml(args.config))
    if args.rounds: cfg['fl']['n_rounds'] = args.rounds
    run_dir = os.path.join(OUT, cfg['name'], f'seed_{args.seed}')
    os.makedirs(run_dir, exist_ok=True)
    log = Logger(os.path.join(run_dir, 'flower.log'))
    set_seed(args.seed)

    parts = load_partitions(cfg['data']['processed_dir'], cfg['fl']['n_clients'])
    X_ho, y_ho = load_holdout(cfg['data']['processed_dir']); X_cal, y_cal = pooled(parts, 'calibration')
    ccfg = client_cfg(cfg)
    input_dim = X_ho.shape[1]
    history = []

    def client_fn(cid):
        i = int(cid)
        return FraudFlowerClient(i, parts[i], ccfg, seed=args.seed).to_client()

    def evaluate_fn(server_round, parameters, config):
        model = numpy_to_state(FraudMLP(input_dim=input_dim), [np.asarray(p) for p in parameters])
        m = evaluate_model(model, X_ho, y_ho, X_cal, y_cal)
        history.append({'round': server_round, 'global': m})
        log(f"  r{server_round:02d} AUC-PR={m['auc_pr']:.4f} AUC-ROC={m['auc_roc']:.4f} F1={m['f1']:.3f}")
        return 1.0 - m['auc_pr'], {'auc_pr': m['auc_pr']}

    strategy = UWFedAvgStrategy(
        weighting=cfg['fl']['weighting'],
        agg_kwargs={'u_floor': cfg['fl']['u_floor'], 'clip_lo': cfg['fl']['clip_lo'], 'clip_hi': cfg['fl']['clip_hi']},
        fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=cfg['fl']['n_clients'],
        min_available_clients=cfg['fl']['n_clients'], evaluate_fn=evaluate_fn,
        on_fit_config_fn=lambda rnd: {'server_round': rnd},
        initial_parameters=fl.common.ndarrays_to_parameters(state_to_numpy(FraudMLP(input_dim=input_dim))))
    log(f"Flower simulation: {cfg['name']} rounds={cfg['fl']['n_rounds']} weighting={cfg['fl']['weighting']}")
    fl.simulation.start_simulation(client_fn=client_fn, num_clients=cfg['fl']['n_clients'],
                                   config=fl.server.ServerConfig(num_rounds=cfg['fl']['n_rounds']), strategy=strategy,
                                   client_resources={'num_cpus': 1}, ray_init_args={'include_dashboard': False, 'log_to_driver': False})
    save_json({'config': cfg, 'history': history, 'server_log': strategy.round_log}, os.path.join(run_dir, 'history.json'))
    log(f"done: final AUC-PR {history[-1]['global']['auc_pr']:.4f}")


if __name__ == '__main__':
    main()
