"""Adaptive DP-SGD optimizer: Opacus DPOptimizer with per-sample clipping bounds.

The only change to Opacus is ``clip_and_accumulate``: instead of one global
bound ``max_grad_norm`` for every sample, sample i is clipped to
``max_grad_norm * scale_i`` with ``scale_i in (0, 1]`` (set for the current
batch via ``set_sample_scales``). Noise is still ``noise_multiplier *
max_grad_norm``, so the mechanism's sensitivity - and therefore the privacy
accounting - is identical to standard DP-SGD. See noise_adaptation.py for why
this realises the proposal's per-sample noise adaptation.

``make_private_adaptive`` wires a model, optimizer and loader together the way
``PrivacyEngine.make_private`` does, but with this optimizer class and an
externally owned accountant so a federated client can carry one accountant
across all rounds.
"""

import torch
from opacus import GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed


class AdaptiveDPOptimizer(DPOptimizer):
    """DPOptimizer whose per-sample clipping bound can be scaled per sample."""

    def __init__(self, optimizer, *, noise_multiplier, max_grad_norm, expected_batch_size,
                 loss_reduction='mean', generator=None, secure_mode=False, **kwargs):
        super().__init__(optimizer, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
                         expected_batch_size=expected_batch_size, loss_reduction=loss_reduction,
                         generator=generator, secure_mode=secure_mode, **kwargs)
        self._sample_scales = None
        self.last_mean_scale = 1.0

    def set_sample_scales(self, scales):
        """scales: 1-D tensor (len = current batch) in (0, 1], or None for uniform clipping."""
        self._sample_scales = None if scales is None else scales.detach().flatten()

    def clip_and_accumulate(self):
        if len(self.grad_samples[0]) == 0:
            per_sample_clip_factor = torch.zeros((0,), device=self.grad_samples[0].device)
        else:
            per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples]
            dev = per_param_norms[0].device
            per_param_norms = [n.to(dev) for n in per_param_norms]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

            bound = torch.full_like(per_sample_norms, float(self.max_grad_norm))
            s = self._sample_scales
            if s is not None and len(s) == len(per_sample_norms):
                bound = bound * s.to(dev).to(per_sample_norms.dtype)
                self.last_mean_scale = float(s.mean())
            else:
                self.last_mean_scale = 1.0
            per_sample_clip_factor = (bound / (per_sample_norms + 1e-6)).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p).to(p.dtype)
            cf = per_sample_clip_factor.to(grad_sample.device).to(p.dtype)
            grad = torch.einsum('i,i...', cf, grad_sample)
            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad
            _mark_as_processed(p.grad_sample)
        self._sample_scales = None


def make_private_adaptive(model, data_loader, *, noise_multiplier, max_grad_norm, accountant,
                          lr=1e-3, generator=None, optimizer_cls=torch.optim.Adam):
    """Wrap (model, loader) for DP-SGD with the adaptive optimizer and a shared accountant.

    Returns (grad_sample_module, optimizer, dp_loader, sample_rate).
    """
    n = len(data_loader.dataset)
    sample_rate = 1.0 / len(data_loader)                 # = batch_size / n (Opacus convention)
    gsm = GradSampleModule(model, batch_first=True, loss_reduction='mean')
    dp_loader = DPDataLoader.from_data_loader(data_loader, generator=generator, distributed=False)
    base_opt = optimizer_cls(gsm.parameters(), lr=lr)
    opt = AdaptiveDPOptimizer(base_opt, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
                              expected_batch_size=int(sample_rate * n), loss_reduction='mean',
                              generator=generator)
    opt.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=sample_rate))
    return gsm, opt, dp_loader, sample_rate
