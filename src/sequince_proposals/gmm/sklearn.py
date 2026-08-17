"""Scikit-learn Gaussian-mixture proposal adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from orng import RandomGenerator
from sequince.array_api import asarray, to_numpy

from ..utils import draw_seed

if TYPE_CHECKING:
    from sklearn.mixture import GaussianMixture


def _resolve_rng(rng: RandomGenerator | int | None) -> RandomGenerator:
    if isinstance(rng, RandomGenerator):
        return rng
    if isinstance(rng, int):
        return RandomGenerator(seed=rng)
    return RandomGenerator()


class GaussianMixtureProposal:
    """Full-covariance Gaussian mixture fitted with scikit-learn.

    Fitting runs eagerly on CPU NumPy arrays. Density values and samples are
    restored to the namespace and device of the relevant input particles.
    Sampling uses :class:`orng.RandomGenerator` rather than sklearn's internal
    random state.

    Weighted fitting is not supported by ``sklearn.mixture.GaussianMixture``.
    The optional ``weights`` argument is accepted only to validate uniform SMC
    weights explicitly.
    """

    def __init__(
        self,
        n_components: int,
        *,
        covariance_type: str = "full",
        regularization: float = 1e-6,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if covariance_type != "full":
            raise ValueError("Only covariance_type='full' is supported.")
        if regularization <= 0 or not np.isfinite(regularization):
            raise ValueError("regularization must be finite and positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if tol <= 0 or not np.isfinite(tol):
            raise ValueError("tol must be finite and positive.")
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.regularization = float(regularization)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._model = self._new_model()
        self._fit_like: Any | None = None

    def _new_model(self, *, random_state: int = 0) -> GaussianMixture:
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError as exc:
            raise ImportError(
                "GaussianMixtureProposal requires scikit-learn. Install the "
                "GMM extra with `pip install 'sequince-proposals[gmm]'`."
            ) from exc
        return GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.regularization,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=random_state,
        )

    def _require_fitted(self) -> GaussianMixture:
        if not hasattr(self._model, "weights_"):
            raise RuntimeError("GaussianMixtureProposal has not been fitted.")
        return self._model

    @property
    def mixture_weights(self) -> np.ndarray:
        return self._require_fitted().weights_.copy()

    @property
    def means(self) -> np.ndarray:
        return self._require_fitted().means_.copy()

    @property
    def covariances(self) -> np.ndarray:
        return self._require_fitted().covariances_.copy()

    @property
    def converged(self) -> bool:
        return bool(getattr(self._model, "converged_", False))

    @property
    def n_iter(self) -> int:
        return int(getattr(self._model, "n_iter_", 0))

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return fit diagnostics consumed by proposal rejuvenation."""
        return {
            "n_components": self.n_components,
            "converged": self.converged,
            "n_iter": self.n_iter,
        }

    def fit(
        self,
        particles: Any,
        context: Any | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        weights: Any | None = None,
    ) -> GaussianMixtureProposal:
        """Fit the mixture to a two-dimensional particle array."""
        del context
        values = to_numpy(particles, dtype=float)
        if values.ndim != 2:
            raise ValueError(
                "particles must have shape (n_particles, n_dims)."
            )
        n_particles, n_dims = values.shape
        if self.n_components > n_particles:
            raise ValueError("n_components must not exceed n_particles.")
        if n_dims < 1:
            raise ValueError("particles must contain at least one dimension.")
        if not np.all(np.isfinite(values)):
            raise ValueError("particles must contain only finite values.")
        if weights is not None:
            sample_weights = to_numpy(weights, dtype=float).reshape(-1)
            if sample_weights.shape != (n_particles,):
                raise ValueError(
                    "weights must contain one value per particle."
                )
            if not np.all(np.isfinite(sample_weights)):
                raise ValueError("weights must contain only finite values.")
            if np.any(sample_weights < 0) or sample_weights.sum() <= 0:
                raise ValueError(
                    "weights must be non-negative with positive sum."
                )
            normalized = sample_weights / sample_weights.sum()
            uniform = np.full(n_particles, 1.0 / n_particles)
            if not np.allclose(normalized, uniform):
                raise NotImplementedError(
                    "Weighted GMM fitting is not supported by scikit-learn; "
                    "fit a post-resampling population with uniform weights."
                )

        seed = draw_seed(rng)
        self._model = self._new_model(
            random_state=0 if seed is None else seed
        ).fit(values)
        if not np.all(np.isfinite(self._model.covariances_)):
            raise ValueError("Fitted GMM covariance matrices are not finite.")
        for covariance in self._model.covariances_:
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "Fitted GMM covariance matrix is not positive definite."
                ) from exc
        self._fit_like = particles
        return self

    def log_prob(self, particles: Any, context: Any | None = None):
        del context
        model = self._require_fitted()
        values = to_numpy(particles, dtype=float)
        if values.ndim != 2 or values.shape[1] != model.means_.shape[1]:
            raise ValueError("particles have an incompatible shape.")
        return asarray(model.score_samples(values), like=particles)

    def sample(
        self,
        n_samples: int,
        context: Any | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        xp: Any | None = None,
    ):
        del context
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        model = self._require_fitted()
        generator = _resolve_rng(rng)
        components = to_numpy(
            generator.choice(
                self.n_components,
                size=n_samples,
                p=model.weights_,
            ),
            dtype=int,
        )
        standard_normal = to_numpy(
            generator.normal(size=(n_samples, model.means_.shape[1])),
            dtype=float,
        )
        cholesky = np.linalg.cholesky(model.covariances_)
        samples = model.means_[components] + np.einsum(
            "nij,nj->ni", cholesky[components], standard_normal
        )
        if xp is not None:
            return xp.asarray(samples)
        if self._fit_like is None:  # pragma: no cover
            return samples
        return asarray(samples, like=self._fit_like)

    def sample_and_log_prob(
        self,
        n_samples: int,
        context: Any | None = None,
        *,
        rng: RandomGenerator | int | None = None,
        xp: Any | None = None,
    ):
        samples = self.sample(
            n_samples,
            context=context,
            rng=rng,
            xp=xp,
        )
        return samples, self.log_prob(samples, context=context)

    def state_dict(self) -> dict[str, Any]:
        if not hasattr(self._model, "weights_"):
            return {}
        return {
            "weights": self._model.weights_.copy(),
            "means": self._model.means_.copy(),
            "covariances": self._model.covariances_.copy(),
            "converged": self.converged,
            "n_iter": self.n_iter,
            "lower_bound": float(self._model.lower_bound_),
            "n_features_in": int(self._model.n_features_in_),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state:
            self._model = self._new_model()
            self._fit_like = None
            return
        required = {"weights", "means", "covariances"}
        if not required.issubset(state):
            raise ValueError("GMM checkpoint state is incomplete.")
        weights = np.asarray(state["weights"], dtype=float)
        means = np.asarray(state["means"], dtype=float)
        covariances = np.asarray(state["covariances"], dtype=float)
        if weights.shape != (self.n_components,):
            raise ValueError(
                "Checkpoint mixture weights have the wrong shape."
            )
        if means.ndim != 2 or means.shape[0] != self.n_components:
            raise ValueError("Checkpoint means have the wrong shape.")
        expected = (self.n_components, means.shape[1], means.shape[1])
        if covariances.shape != expected:
            raise ValueError("Checkpoint covariances have the wrong shape.")
        precisions_cholesky = np.empty_like(covariances)
        for component, covariance in enumerate(covariances):
            try:
                covariance_cholesky = np.linalg.cholesky(covariance)
                precisions_cholesky[component] = np.linalg.solve(
                    covariance_cholesky, np.eye(means.shape[1])
                ).T
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "Checkpoint covariance is not positive definite."
                ) from exc
        model = self._new_model()
        model.weights_ = weights / weights.sum()
        model.means_ = means.copy()
        model.covariances_ = covariances.copy()
        model.precisions_cholesky_ = precisions_cholesky
        model.converged_ = bool(state.get("converged", False))
        model.n_iter_ = int(state.get("n_iter", 0))
        model.lower_bound_ = float(state.get("lower_bound", -np.inf))
        model.n_features_in_ = int(state.get("n_features_in", means.shape[1]))
        self._model = model
        self._fit_like = np.empty((0, means.shape[1]))


__all__ = ["GaussianMixtureProposal"]
