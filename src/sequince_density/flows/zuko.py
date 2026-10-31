"""Zuko-backed proposal and flow adapter."""

from __future__ import annotations

import logging
from typing import Callable

import array_api_compat.torch as torch_api
import torch
import zuko
from tqdm.auto import tqdm

from ..utils import torch_random_context, validate_training_data
from .base import Flow

logger = logging.getLogger(__name__)


class ZukoFlow(Flow):
    """Zuko-backed flow adapter for Sequince."""

    xp = torch_api

    def __init__(
        self,
        dims: int,
        flow_class: str | Callable = "MAF",
        particle_transform=None,
        seed: int = 1234,
        device: str = "cpu",
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
        self.seed = seed
        torch.manual_seed(seed)

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
        clip_grad: float | None = None,
        lr_annealing: bool = False,
        patience: int | None = None,
        show_progress: bool = False,
        *,
        rng=None,
    ) -> "ZukoFlow":
        """Fit by maximum likelihood and restore the best validation epoch.

        Parameters
        ----------
        samples : array_like, shape (n_samples, dims)
            Nonempty, finite training particles. Transformed particles must
            also be finite and have the same shape.
        context : object, optional
            Context passed to the particle transform.
        n_epochs : int, default=100
            Maximum number of training epochs.
        lr : float, default=1e-3
            Initial Adam learning rate.
        batch_size : int, default=500
            Maximum number of particles per training or validation batch.
        validation_fraction : float, default=0.2
            Fraction held out for validation. If rounding gives no validation
            particles (including zero), or leaves no training particles, use
            all particles for both training and validation.
        clip_grad : float or None, default=None
            Positive maximum total gradient norm, applied before each optimizer
            step. None disables clipping.
        lr_annealing : bool, default=False
            Apply cosine learning-rate annealing from lr to zero over n_epochs,
            stepping the scheduler after each completed epoch.
        patience : int or None, default=None
            Positive number of consecutive epochs without a strictly lower
            validation loss before stopping. Improvement resets the counter;
            ties count as unsuccessful epochs. None disables early stopping.
        show_progress : bool, default=False
            Show epoch progress and training and validation losses.
        rng : RandomGenerator or int or None, optional
            RNG for fitting, isolated from global Torch state when supplied.

        Returns
        -------
        ZukoFlow
            This fitted adapter. fit_history contains losses for every
            completed epoch, including epochs after the best validation epoch.
        """
        with torch_random_context(rng, device=self.device):
            return self._fit(
                samples,
                context=context,
                n_epochs=n_epochs,
                lr=lr,
                batch_size=batch_size,
                validation_fraction=validation_fraction,
                clip_grad=clip_grad,
                lr_annealing=lr_annealing,
                patience=patience,
                show_progress=show_progress,
                rng=rng,
            )

    def _fit(
        self,
        samples,
        context=None,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 500,
        validation_fraction: float = 0.2,
        clip_grad: float | None = None,
        lr_annealing: bool = False,
        patience: int | None = None,
        show_progress: bool = False,
        rng=None,
    ) -> "ZukoFlow":
        if clip_grad is not None and (not 0 < clip_grad < float("inf")):
            raise ValueError("clip_grad must be finite and positive or None.")
        if patience is not None and (
            isinstance(patience, bool)
            or not isinstance(patience, int)
            or patience < 1
        ):
            raise ValueError("patience must be a positive integer or None.")
        x = self._as_torch(samples)
        validate_training_data(x, self.dims, "Raw")
        x_prime = self.fit_particle_transform(x, context=context, rng=rng)
        validate_training_data(x_prime, self.dims, "Transformed")
        if x_prime.shape != x.shape:
            raise ValueError("Transformed training data must preserve shape.")
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
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
            if lr_annealing
            else None
        )
        unsuccessful_epochs = 0
        best_state = None
        best_val_loss = float("inf")
        training_loss = []
        validation_loss = []

        with tqdm(
            range(n_epochs), desc="Epochs", disable=not show_progress
        ) as progress:
            for _ in progress:
                self.flow.train()
                train_loss = 0.0
                n_train_batches = 0
                for (x_batch,) in dataset:
                    loss = self.loss_fn(x_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.flow.parameters(), clip_grad
                        )
                    optimizer.step()
                    train_loss += loss.item()
                    n_train_batches += 1

                if scheduler is not None:
                    scheduler.step()
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
                    unsuccessful_epochs = 0
                    best_val_loss = avg_val_loss
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.flow.state_dict().items()
                    }
                else:
                    unsuccessful_epochs += 1
                if show_progress:
                    progress.set_postfix(
                        train_loss=f"{training_loss[-1]:.4f}",
                        val_loss=f"{avg_val_loss:.4f}",
                    )
                if patience is not None and unsuccessful_epochs >= patience:
                    break

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
        if state.get("particle_transform") is not None:
            self.particle_transform = state["particle_transform"]
        self.flow.load_state_dict(state["flow"])
        self.fit_history = dict(state.get("fit_history", self.fit_history))
        self.flow.eval()

    def sample_and_log_prob(self, n_samples: int, context=None, *, rng=None):
        with torch_random_context(rng, device=self.device), torch.no_grad():
            x_prime, log_prob = self.flow().rsample_and_log_prob((n_samples,))
        x, log_abs_det_jacobian = self.inverse_particle_transform(
            x_prime, context=context, rng=rng
        )
        return x, log_prob - log_abs_det_jacobian

    def sample(self, n_samples: int, context=None, *, rng=None):
        with torch_random_context(rng, device=self.device), torch.no_grad():
            x_prime = self.flow().rsample((n_samples,))
        x, _ = self.inverse_particle_transform(
            x_prime, context=context, rng=rng
        )
        return x

    def log_prob(self, x, context=None):
        x = self._as_torch(x)
        x_prime, log_abs_det_jacobian, _ = self.forward_particle_transform(
            x, context=context
        )
        return self.flow().log_prob(x_prime) + log_abs_det_jacobian

    def forward(self, x, context=None):
        x = self._as_torch(x)
        x_prime, log_j_rescale, _ = self.forward_particle_transform(
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
