"""Shared bits for the experiment CLIs: path setup, logging, result aggregation."""

import os
import sys
import glob
import json
import time
import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import yaml

# All experiment outputs live under this root ('results' by default). Override for smoke tests:
#   FYP_RESULTS_ROOT=<dir>
RESULTS_ROOT = os.environ.get('FYP_RESULTS_ROOT', 'results')

from src.utils import save_json, load_json  # noqa: E402


class Logger:
    """print() that also appends to a log file (experiment runs are long and often backgrounded)."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path

    def __call__(self, msg=''):
        stamp = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{stamp}] {msg}'
        print(line, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def mean_sd(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None, None
    return float(v.mean()), float(v.std(ddof=0))


def collect_summaries(results_dir, name):
    """All seed summaries for one federated run name."""
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, name, 'seed_*', 'summary.json'))):
        out.append(load_json(p))
    return out


def collect_histories(results_dir, name):
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, name, 'seed_*', 'history.json'))):
        out.append(load_json(p))
    return out


def already_done(results_dir, name, seed):
    return os.path.exists(os.path.join(results_dir, name, f'seed_{seed}', 'summary.json'))


def write_markdown_table(rows, headers, path=None):
    """rows: list of lists; returns the markdown text (and writes it if path given)."""
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join('' if v is None else (f'{v:.4f}' if isinstance(v, float) else str(v)) for v in r) + ' |')
    txt = '\n'.join(lines)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(txt + '\n')
    return txt
