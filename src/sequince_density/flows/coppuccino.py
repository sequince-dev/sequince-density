"""Eager copula normalizing flows backed by Coppuccino."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import ModuleType
from typing import Any

import numpy as np
from orng import RandomGenerator
from sequince.array_api import to_numpy
from sequince.steps.context import StepContext
from sequince.transforms import (
    BoundedToUnboundedTransform,
    CompositeTransform,
    ParticleTransform,
)
from sequince.typing import Array, ArrayLike

from ..utils import draw_seed
from .base import Flow

_IMPORT_LOCK = Lock()


@contextmanager
def _backend() -> Generator[ModuleType, None, None]:
    """Import Coppuccino and scope float64 to its own operations."""
    try:
        import jax
        from jax._src.config import enable_x64 as x64_state

        with _IMPORT_LOCK:
            # The public flag can reflect an enclosing thread-local scope.
            # Coppuccino's import updates the process-wide value.
            previous = x64_state.get_global()
            try:
                backend = import_module("coppuccino")
            finally:
                if x64_state.get_global() != previous:
                    jax.config.update("jax_enable_x64", previous)
    except ModuleNotFoundError as exc:
        if exc.name in {"coppuccino", "jax"}:
            raise ImportError(
                "Install sequince-density[coppuccino] to use CoppuccinoFlow."
            ) from exc
        raise
    # JAX 0.7 keeps the context manager on the config state.
    with getattr(jax, "enable_x64", x64_state)(True):
        yield backend


def _has_bounded_transform(transform: ParticleTransform) -> bool:
    if isinstance(transform, CompositeTransform):
        return any(
            _has_bounded_transform(item) for item in transform.transforms
        )
    return isinstance(transform, BoundedToUnboundedTransform)


def _apply_bijection(
    flow: Any, values: Array, *, inverse: bool
) -> tuple[Array, Array]:
    import jax

    # Coppuccino nests the learned copula inside the marginal map.
    bijection = flow.merge_transforms().bijection
    if inverse:
        return jax.vmap(bijection.inverse_and_log_det)(values)
    return jax.vmap(bijection.transform_and_log_det)(values)


class CoppuccinoFlow(Flow):
    """Fit one copula flow with unbounded marginals.

    Parameters
    ----------
    dims : int
        Number of columns in the training and scoring arrays.
    seed : int, default=0
        Seed for calls without an explicit RNG.
    particle_transform : ParticleTransform, optional
        Outer coordinate transform. Bounded-to-unbounded and periodic
        transforms are unsupported.
    dtype : dtype, optional
        If supplied, must be float64. The eager API uses NumPy float64.
    device : optional
        Accepted for compatibility. JAX controls internal device placement.
    **kwargs
        Coppuccino training options, such as `flow_layers` or `max_epochs`.
    """

    xp = np

    def __init__(
        self,
        dims: int,
        *,
        seed: int = 0,
        particle_transform: ParticleTransform | None = None,
        dtype: Any = None,
        device: Any = None,
        **kwargs: Any,
    ) -> None:
        if dtype is not None and np.dtype(dtype) != np.dtype(np.float64):
            raise ValueError("CoppuccinoFlow requires float64.")
        self._validate_options(kwargs)
        super().__init__(
            dims, device=None, particle_transform=particle_transform
        )
        self._fit_options = {"show_progress": False, **kwargs}
        self._rng = np.random.default_rng(seed)
        self._flow: Any = None

    def _validate_transform(self, transform: ParticleTransform) -> None:
        super()._validate_transform(transform)
        if _has_bounded_transform(transform):
            raise ValueError(
                "CoppuccinoFlow does not support bounded-to-unbounded "
                "particle transforms. In Aspire, set "
                "bounded_to_unbounded=False and enforce support in log_prior."
            )

    @staticmethod
    def _validate_options(options: Mapping[str, Any]) -> None:
        if "prior_bounds" in options:
            raise ValueError(
                "CoppuccinoFlow only supports unbounded marginals."
            )
        if "rng_seed" in options:
            raise ValueError("Use seed or rng to seed CoppuccinoFlow.")

    def _seed(self, rng: RandomGenerator | int | None) -> int:
        seed = draw_seed(rng)
        if seed is None:
            seed = int(self._rng.integers(2**31 - 1))
        return seed

    def _values(
        self, samples: ArrayLike, *, allow_single: bool = False
    ) -> np.ndarray:
        values = to_numpy(samples, dtype=np.float64)
        allowed_ranks = (1, 2) if allow_single else (2,)
        if values.ndim not in allowed_ranks or values.shape[-1] != self.dims:
            shape = f"(n_samples, {self.dims})"
            if allow_single:
                shape += f" or ({self.dims},)"
            raise ValueError(f"samples must have shape {shape}.")
        return values

    def _require_fitted(self) -> Any:
        if self._flow is None:
            raise RuntimeError("CoppuccinoFlow has not been fitted.")
        return self._flow

    def fit(
        self,
        samples: ArrayLike,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        **kwargs: Any,
    ) -> CoppuccinoFlow:
        """Fit to a finite array of shape `(n_samples, dims)`."""
        self._validate_options(kwargs)
        values = self._values(samples)
        if not values.size or not np.all(np.isfinite(values)):
            raise ValueError("training samples must be nonempty and finite.")

        transform = self.particle_transform.fit(
            values, context=context, rng=rng
        )
        self._validate_transform(transform)
        transformed, _, _ = transform.forward(values, context=context, rng=rng)
        transformed = self._values(transformed)
        if not np.all(np.isfinite(transformed)):
            raise ValueError("transformed training samples must be finite.")

        options = self._fit_options | kwargs
        with _backend() as backend:
            flow = backend.normalizing_flows_fit(
                transformed, rng_seed=self._seed(rng), **options
            )
        self.particle_transform = transform
        self._flow = flow
        return self

    def forward(
        self, samples: ArrayLike, context: StepContext | None = None
    ) -> tuple[Array, Array]:
        """Map particles to the base space with their log-Jacobian."""
        flow = self._require_fitted()
        values = self._values(samples, allow_single=True)
        transformed, log_det, _ = self.forward_particle_transform(
            np.atleast_2d(values), context=context
        )
        with _backend():
            from equinox import filter_jit

            latent, flow_log_det = filter_jit(_apply_bijection)(
                flow, transformed, inverse=True
            )
        return to_numpy(latent).reshape(values.shape), (
            to_numpy(flow_log_det) + log_det
        ).reshape(values.shape[:-1])

    def inverse(
        self, samples: ArrayLike, context: StepContext | None = None
    ) -> tuple[Array, Array]:
        """Map base-space points to particle space with their log-Jacobian."""
        flow = self._require_fitted()
        latent = self._values(samples, allow_single=True)
        with _backend():
            from equinox import filter_jit

            transformed, flow_log_det = filter_jit(_apply_bijection)(
                flow, np.atleast_2d(latent), inverse=False
            )
        values, log_det = self.inverse_particle_transform(
            to_numpy(transformed), context=context
        )
        return values.reshape(latent.shape), (
            to_numpy(flow_log_det) + log_det
        ).reshape(latent.shape[:-1])

    def log_prob(
        self, samples: ArrayLike, context: StepContext | None = None
    ) -> np.ndarray:
        """Return the log density of each point in particle space."""
        flow = self._require_fitted()
        values = self._values(samples, allow_single=True)
        transformed, log_det, _ = self.forward_particle_transform(
            np.atleast_2d(values), context=context
        )
        with _backend() as backend:
            log_prob = backend.log_prob(flow, self._values(transformed))
        return (log_prob + log_det).reshape(values.shape[:-1])

    def sample(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
    ) -> Array:
        """Draw an array with shape `(n_samples, dims)`."""
        flow = self._require_fitted()
        with _backend() as backend:
            transformed = backend.sample(
                flow, n_samples, rng_seed=self._seed(rng)
            )
        values, _ = self.inverse_particle_transform(
            transformed, context=context, rng=rng
        )
        return values

    def sample_and_log_prob(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator | int | None = None,
    ) -> tuple[Array, Array]:
        """Draw particles and return their log densities in particle space."""
        flow = self._require_fitted()
        with _backend() as backend:
            transformed, log_prob = backend.sample_and_log_prob(
                flow, n_samples, rng_seed=self._seed(rng)
            )
        values, log_det = self.inverse_particle_transform(
            transformed, context=context, rng=rng
        )
        return values, log_prob - log_det

    def state_dict(self) -> dict[str, Any]:
        """Capture the fitted model, outer transform, and local RNG state."""
        payload = None
        if self._flow is not None:
            with TemporaryDirectory() as directory, _backend() as backend:
                path = Path(directory) / "flow.pkl"
                backend.save_flow(self._flow, path)
                payload = path.read_bytes()
        return {
            "version": 1,
            "dims": self.dims,
            "flow": payload,
            "particle_transform": self.particle_transform,
            "rng": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state from a trusted checkpoint."""
        if state.get("version") != 1:
            raise ValueError("Unsupported CoppuccinoFlow checkpoint version.")
        if state["dims"] != self.dims:
            raise ValueError("CoppuccinoFlow checkpoint dimensions differ.")
        transform = state["particle_transform"]
        self._validate_transform(transform)
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]

        flow = None
        if state["flow"] is not None:
            with TemporaryDirectory() as directory, _backend() as backend:
                path = Path(directory) / "flow.pkl"
                path.write_bytes(state["flow"])
                flow = backend.load_flow(path)
        self.particle_transform = transform
        self._rng = rng
        self._flow = flow
