"""Shared base class for normalizing-flow adapters."""

from __future__ import annotations

import inspect
from types import ModuleType
from typing import Any

from sequince.transforms import (
    CompositeTransform,
    IdentityTransform,
    ParticleTransform,
    PeriodicBoundaryTransform,
)


def _validate_particle_transform(transform: ParticleTransform) -> None:
    if isinstance(transform, PeriodicBoundaryTransform):
        raise ValueError(
            "PeriodicBoundaryTransform is unsupported by flow adapters: "
            "periodic wrapping is many-to-one, so a Jacobian correction "
            "alone does not give the correct density."
        )
    if isinstance(transform, CompositeTransform):
        for component in transform.transforms:
            _validate_particle_transform(component)


class Flow:
    """Base class for backend flow adapters.

    This mirrors the Aspire flow surface closely enough for future migration,
    while remaining compatible with ``sequince.transforms.FlowTransform``.

    Periodic particle transforms, including components of composite transforms,
    are unsupported because these adapters do not sum over periodic images.
    """

    xp: ModuleType | None = None

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
        self.fit_history: dict[str, list[float]] = {
            "training_loss": [],
            "validation_loss": [],
        }

    @property
    def particle_transform(self) -> ParticleTransform:
        """Coordinate transform used for sampling and density evaluation."""
        return self._particle_transform

    @particle_transform.setter
    def particle_transform(self, transform: ParticleTransform) -> None:
        _validate_particle_transform(transform)
        self._particle_transform = transform

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

    def fit_particle_transform(self, x, context=None, *, rng=None):
        self.particle_transform = self.particle_transform.fit(
            x, context=context, rng=rng
        )
        transformed, _, _ = self.particle_transform.forward(
            x, context=context, rng=rng
        )
        return transformed

    def forward_particle_transform(self, x, context=None, *, rng=None):
        return self.particle_transform.forward(x, context=context, rng=rng)

    def inverse_particle_transform(
        self, x, context=None, *, auxiliary=None, rng=None
    ):
        return self.particle_transform.inverse(
            x, context=context, auxiliary=auxiliary, rng=rng
        )

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
