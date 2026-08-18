"""Zuko-backed proposal and flow adapter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import array_api_compat.torch as torch_api
import torch
import zuko

from ..utils import torch_random_context
from .base import Flow

if TYPE_CHECKING:
    from sequince.transforms import ParticleTransform

logger = logging.getLogger(__name__)


class ZukoFlow(Flow):
    """Zuko-backed flow adapter for Sequince."""

    xp = torch_api

    def __init__(
        self,
        dims: int,
        flow_class: str | Callable = "MAF",
        particle_transform: ParticleTransform | None = None,
        device: str | torch.DeviceObjType = "cpu",
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> None:
        resolved_dtype = dtype or torch.get_default_dtype()
        super().__init__(
            dims=dims,
            device=torch.device(device),
            particle_transform=particle_transform,
        )
        self.dtype = resolved_dtype

        if isinstance(flow_class, str):
            flow_constructor = getattr(zuko.flows, flow_class)
        else:
            flow_constructor = flow_class

        hidden_features = kwargs.pop("hidden_features", None)
        if hidden_features is not None:
            kwargs["hidden_features"] = list(map(int, hidden_features))

        self._flow = flow_constructor(self.dims, 0, **kwargs)
        self._flow.to(device=self.device, dtype=self.dtype)
        if hasattr(self._flow, "compile"):
            self._flow.compile()
        logger.debug("Initialized Zuko flow:\n%s", self._flow)

    def _as_torch(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    @property
    def flow(self):
        return self._flow

    def loss_fn(self, x):
        return -self.flow().log_prob(x).mean()

    def fit(
        self,
        samples,
        context=None,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 500,
        validation_fraction: float = 0.2,
        *,
        rng=None,
    ) -> "ZukoFlow":
        with torch_random_context(rng, device=self.device):
            return self._fit(
                samples,
                context=context,
                n_epochs=n_epochs,
                lr=lr,
                batch_size=batch_size,
                validation_fraction=validation_fraction,
            )

    def _fit(
        self,
        samples,
        context=None,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 500,
        validation_fraction: float = 0.2,
    ) -> "ZukoFlow":
        x = self._as_torch(samples)
        x_prime = self.fit_particle_transform(x, context=context)
        indices = torch.randperm(x_prime.shape[0], device=x_prime.device)
        x_prime = x_prime[indices]

        n_val = int(validation_fraction * x_prime.shape[0])
        if n_val <= 0 or n_val >= x_prime.shape[0]:
            x_train = x_prime
            x_val = x_prime
        else:
            x_train = x_prime[:-n_val]
            x_val = x_prime[-n_val:]

        dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train),
            shuffle=True,
            batch_size=min(batch_size, x_train.shape[0]),
        )
        val_dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_val),
            shuffle=False,
            batch_size=min(batch_size, x_val.shape[0]),
        )

        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        best_state = None
        best_val_loss = float("inf")
        training_loss = []
        validation_loss = []

        for _ in range(n_epochs):
            self.flow.train()
            train_loss = 0.0
            n_train_batches = 0
            for (x_batch,) in dataset:
                loss = self.loss_fn(x_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_train_batches += 1

            self.flow.eval()
            val_loss = 0.0
            n_batches = 0
            for (x_batch,) in val_dataset:
                with torch.no_grad():
                    val_loss += self.loss_fn(x_batch).item()
                n_batches += 1

            avg_val_loss = val_loss / max(n_batches, 1)
            training_loss.append(train_loss / max(n_train_batches, 1))
            validation_loss.append(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.flow.state_dict().items()
                }

        if best_state is not None:
            self.flow.load_state_dict(best_state)

        self.fit_history = {
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        }
        self.flow.eval()
        return self

    def state_dict(self):
        return {
            "flow": {
                key: value.detach().cpu().clone()
                for key, value in self.flow.state_dict().items()
            },
            "particle_transform": self.particle_transform,
            "fit_history": self.fit_history,
        }

    def load_state_dict(self, state):
        self.flow.load_state_dict(state["flow"])
        if state.get("particle_transform") is not None:
            self.particle_transform = state["particle_transform"]
        self.fit_history = dict(state.get("fit_history", self.fit_history))
        self.flow.eval()

    def sample_and_log_prob(self, n_samples: int, context=None, *, rng=None):
        with torch_random_context(rng, device=self.device), torch.no_grad():
            x_prime, log_prob = self.flow().rsample_and_log_prob((n_samples,))
        x, log_abs_det_jacobian = self.inverse_particle_transform(
            x_prime, context=context
        )
        return x, log_prob - log_abs_det_jacobian

    def sample(self, n_samples: int, context=None, *, rng=None):
        with torch_random_context(rng, device=self.device), torch.no_grad():
            x_prime = self.flow().rsample((n_samples,))
        x, _ = self.inverse_particle_transform(x_prime, context=context)
        return x

    def log_prob(self, x, context=None):
        x = self._as_torch(x)
        x_prime, log_abs_det_jacobian = self.forward_particle_transform(
            x, context=context
        )
        return self.flow().log_prob(x_prime) + log_abs_det_jacobian

    def forward(self, x, context=None):
        x = self._as_torch(x)
        x_prime, log_j_rescale = self.forward_particle_transform(
            x, context=context
        )
        z, log_abs_det_jacobian = self.flow().transform.call_and_ladj(x_prime)
        return z, log_abs_det_jacobian + log_j_rescale

    def inverse(self, z, context=None):
        z = self._as_torch(z)
        with torch.no_grad():
            x_prime, log_abs_det_jacobian = (
                self.flow().transform.inv.call_and_ladj(z)
            )
        x, log_j_rescale = self.inverse_particle_transform(
            x_prime, context=context
        )
        return x, log_j_rescale + log_abs_det_jacobian


class ZukoFlowMatching(ZukoFlow):
    """Flow wrapper for training Zuko flows using flow matching.

    Note that this flow is only compatible with CNF flows, as flow matching is
    not implemented for discrete flows in Zuko.

    The `eta` parameter controls the amount of noise added to the target
    velocity field during training.
    """

    def __init__(
        self,
        dims: int,
        particle_transform: ParticleTransform | None = None,
        device: str | torch.DeviceObjType = "cpu",
        eta: float = 1e-3,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        kwargs.setdefault("hidden_features", 4 * [100])
        super().__init__(
            dims,
            device=device,
            particle_transform=particle_transform,
            flow_class="CNF",
            dtype=dtype,
            **kwargs,
        )
        self.eta = eta

    def loss_fn(self, theta: torch.Tensor) -> torch.Tensor:
        t = torch.rand(
            theta.shape[:-1], dtype=theta.dtype, device=theta.device
        )
        t_ = t[..., None]
        eps = torch.randn_like(theta)
        theta_prime = (1 - t_) * theta + (t_ + self.eta) * eps
        v = eps - theta
        return (self.flow.transform.f(t, theta_prime) - v).square().mean()
