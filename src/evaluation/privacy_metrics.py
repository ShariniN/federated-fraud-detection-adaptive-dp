"""Empirical privacy validation wrappers (thin layer over src.attacks)."""

from ..attacks.mia import yeom_loss_attack, OfflineLiRA, attack_report
from ..attacks.gradient_inversion import sample_gradient, invert, reconstruction_error


def compute_mia_auc(model, train_x, train_y, test_x, test_y, method='loss', lira=None, device='cpu'):
    """Return the attack report dict (auc, tpr_at_fpr_0.001, tpr_at_fpr_0.01, roc)."""
    if method == 'loss':
        return yeom_loss_attack(model, train_x, train_y, test_x, test_y, device)
    if method == 'lira':
        assert lira is not None, 'pass a fitted OfflineLiRA'
        return lira.attack(model, train_x, train_y, test_x, test_y)
    raise ValueError(method)


__all__ = ['compute_mia_auc', 'yeom_loss_attack', 'OfflineLiRA', 'attack_report',
           'sample_gradient', 'invert', 'reconstruction_error']
