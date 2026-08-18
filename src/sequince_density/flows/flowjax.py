"""FlowJAX-backed flow adapter."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jrandom
from flowjax.distributions import Normal
from flowjax.train import fit_to_data

from ..utils import draw_seed
from .base import Flow

logger = logging.getLogger(__name__)


def _resolve_flow_factory(flow_type: str | Callable) -> Callable:
    if callable(flow_type):
        return flow_type
    import flowjax.flows

    try:
        return getattr(flowjax.flows, flow_type)
    except AttributeError as exc:
        raise ValueError(f"Unknown FlowJAX flow type: {flow_type}") from exc


class FlowJax(Flow):
    """FlowJAX-backed normalizing flow.

    The adapter always exposes native JAX arrays. Conversion to the SMC
    particle namespace is handled by :class:`sequince.FlowProposal`.
    """

    xp = jnp

    def __init__(
        self,
        dims: int,
        flow_type: str | Callable = "masked_autoregressive_flow",
        *,
        key: Any | None = None,
        seed: int = 0,
        particle_transform=None,
        dtype=None,
        device: Any | None = None,
        **kwargs,
    ) -> None:
        if device is not None:
            logger.warning("FlowJax ignores the device argument.")
        super().__init__(
            dims=dims,
            device=None,
            particle_transform=particle_transform,
        )
        self.dtype = jnp.dtype(dtype or jnp.float32)
        self.key = jrandom.key(seed) if key is None else key
        self.key, flow_key = jrandom.split(self.key)
        factory = _resolve_flow_factory(flow_type)
        self._flow = factory(
            flow_key,
            base_dist=Normal(jnp.zeros(dims, dtype=self.dtype)),
            **kwargs,
        )

    @property
    def flow(self):
        return self._flow

    def fit(self, samples, context=None, *, rng=None, **kwargs) -> FlowJax:
        values = jnp.asarray(samples, dtype=self.dtype)
        transformed = jnp.asarray(
            self.fit_particle_transform(values, context=context),
            dtype=self.dtype,
        )
        fit_key = self._random_key(rng)
        self._flow, losses = fit_to_data(
            fit_key,
            self._flow,
            transformed,
            **kwargs,
        )
        self.fit_history = {
            "training_loss": [float(value) for value in losses["train"]],
            "validation_loss": [float(value) for value in losses["val"]],
        }
        return self

    def state_dict(self):
        import equinox as eqx

        buffer = BytesIO()
        eqx.tree_serialise_leaves(buffer, self._flow)
        return {
            "flow": buffer.getvalue(),
            "key": self.key,
            "particle_transform": self.particle_transform,
            "fit_history": self.fit_history,
        }

    def load_state_dict(self, state):
        import equinox as eqx

        self._flow = eqx.tree_deserialise_leaves(
            BytesIO(state["flow"]), self._flow
        )
        self.key = state["key"]
        if state.get("particle_transform") is not None:
            self.particle_transform = state["particle_transform"]
        self.fit_history = dict(state.get("fit_history", self.fit_history))

    def forward(self, samples, context=None):
        values = jnp.asarray(samples, dtype=self.dtype)
        transformed, transform_log_det = self.forward_particle_transform(
            values,
            context=context,
        )
        latent, flow_log_det = jax.vmap(
            self.flow.bijection.inverse_and_log_det
        )(jnp.asarray(transformed, dtype=self.dtype))
        return latent, flow_log_det + transform_log_det

    def inverse(self, samples, context=None):
        latent = jnp.asarray(samples, dtype=self.dtype)
        transformed, flow_log_det = jax.vmap(
            self.flow.bijection.transform_and_log_det
        )(latent)
        values, transform_log_det = self.inverse_particle_transform(
            transformed,
            context=context,
        )
        return values, flow_log_det + transform_log_det

    def log_prob(self, samples, context=None):
        values = jnp.asarray(samples, dtype=self.dtype)
        transformed, log_det = self.forward_particle_transform(
            values,
            context=context,
        )
        return self.flow.log_prob(
            jnp.asarray(transformed, dtype=self.dtype)
        ) + jnp.asarray(log_det, dtype=self.dtype)

    def _random_key(self, rng):
        seed = draw_seed(rng)
        if seed is not None:
            return jrandom.key(seed)
        self.key, sample_key = jrandom.split(self.key)
        return sample_key

    def sample(self, n_samples: int, context=None, *, rng=None):
        sample_key = self._random_key(rng)
        transformed = self.flow.sample(sample_key, (n_samples,))
        values, _ = self.inverse_particle_transform(
            transformed,
            context=context,
        )
        return values

    def sample_and_log_prob(self, n_samples: int, context=None, *, rng=None):
        sample_key = self._random_key(rng)
        transformed, log_prob = self.flow.sample_and_log_prob(
            sample_key,
            (n_samples,),
        )
        values, log_det = self.inverse_particle_transform(
            transformed,
            context=context,
        )
        return values, log_prob - log_det


__all__ = ["FlowJax"]
