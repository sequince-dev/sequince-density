"""Shared base class for normalizing-flow adapters."""

from __future__ import annotations

import inspect
from typing import Any

from sequince.transforms import IdentityTransform, ParticleTransform


class Flow:
    """Base class for backend flow adapters.

    This mirrors the Aspire flow surface closely enough for future migration,
    while remaining compatible with ``sequince.transforms.FlowTransform``.
    """

    xp = None

    def __init__(
        self,
        dims: int,
        device: Any = None,
        particle_transform: ParticleTransform | None = None,
    ) -> None:
        self.dims = dims
        self.device = device
        self.particle_transform = (
            particle_transform
            if particle_transform is not None
            else IdentityTransform()
        )

    def fit(self, samples, context=None):
        raise NotImplementedError

    def forward(self, samples, context=None):
        raise NotImplementedError

    def inverse(self, samples, context=None):
        raise NotImplementedError

    def log_prob(self, samples, context=None):
        raise NotImplementedError

    def sample(self, n_samples: int, context=None, *, rng=None):
        raise NotImplementedError

    def sample_and_log_prob(self, n_samples: int, context=None, *, rng=None):
        raise NotImplementedError

    def fit_particle_transform(self, x, context=None):
        self.particle_transform = self.particle_transform.fit(
            x, context=context
        )
        transformed, _ = self.particle_transform.forward(x, context=context)
        return transformed

    def forward_particle_transform(self, x, context=None):
        return self.particle_transform.forward(x, context=context)

    def inverse_particle_transform(self, x, context=None):
        return self.particle_transform.inverse(x, context=context)

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        sig = inspect.signature(cls.__init__)
        bound = sig.bind_partial(obj, *args, **kwargs)
        bound.apply_defaults()
        obj._init_args = {
            key: value
            for key, value in bound.arguments.items()
            if key != "self"
        }
        return obj
