"""Action distributions for the RL core.

`DiagGaussian` is a diagonal Gaussian over unconstrained actions with an optional boolean
mask of action dims to ignore. Masked dims are for slots that cannot be acted on (e.g. coins
that are not tradable): they contribute exactly 0 to `log_prob` and `entropy`, so they get no
policy gradient and do not distort the PPO ratio, and `sample`/`mode` return the mean there.
"""

from __future__ import annotations

import math

import torch

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
# Safety net against a log_std that drifts to +-inf; far outside any useful exploration range.
LOG_STD_MIN, LOG_STD_MAX = -10.0, 3.0


class DiagGaussian:
    """Independent Normal per action dim; `log_prob` and `entropy` are summed over the last dim.

    mean:    [..., A]
    log_std: [A] (state-independent) or anything broadcastable to mean's shape, e.g. [B, A]
    mask:    optional bool [..., A], True = dim is active. Masked dims are ignored.
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, mask: torch.Tensor | None = None):
        self.mean = mean
        self.log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        self.std = self.log_std.exp()
        if mask is not None:
            if mask.shape != mean.shape:
                raise ValueError(f"mask shape {tuple(mask.shape)} must match mean shape {tuple(mean.shape)}")
            mask = mask.to(device=mean.device, dtype=torch.bool)
        self.mask = mask

    @property
    def mode(self) -> torch.Tensor:
        return self.mean

    def _keep(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out masked dims (using where, so NaN/inf in a masked dim cannot leak into grads)."""
        return x if self.mask is None else torch.where(self.mask, x, torch.zeros_like(x))

    def rsample(self) -> torch.Tensor:
        """Reparameterised sample (gradients flow to mean and log_std)."""
        a = self.mean + self.std * torch.randn_like(self.mean)
        return a if self.mask is None else torch.where(self.mask, a, self.mean)

    def sample(self) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        diff = self._keep(actions - self.mean)
        per_dim = -0.5 * (diff / self.std) ** 2 - self.log_std - _HALF_LOG_2PI
        return self._keep(per_dim).sum(-1)

    def entropy(self) -> torch.Tensor:
        return self._keep(self.log_std + 0.5 + _HALF_LOG_2PI).sum(-1)
