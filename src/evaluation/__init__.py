"""Evaluation package.

Only dependency-light modules are imported eagerly. Plotting pulls in seaborn
and matplotlib, and the privacy/uncertainty wrappers depend on the attack and
conformal packages scheduled for the final phase, so those are resolved lazily
via ``__getattr__``. This keeps ``import src.evaluation`` working — and so the
experiment scripts runnable — even when an optional dependency is absent.
"""

from .metrics import compute_utility_metrics
from .train_utils import (
    load_real_data,
    train_model_dp,
    compute_all_metrics,
    plot_confusion_matrix,
    apply_borderline_smote,
)

__all__ = [
    'compute_utility_metrics',
    'load_real_data',
    'train_model_dp',
    'compute_all_metrics',
    'plot_confusion_matrix',
    'apply_borderline_smote',
    'plot_privacy_utility_curve',
    'plot_convergence',
    'plot_confusion_matrices',
    'plot_conformal_score_distribution',
]

_LAZY = {
    'plot_privacy_utility_curve': '.visualisation',
    'plot_convergence': '.visualisation',
    'plot_confusion_matrices': '.visualisation',
    'plot_conformal_score_distribution': '.visualisation',
}


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    module = import_module(module_name, __name__)
    return getattr(module, name)
