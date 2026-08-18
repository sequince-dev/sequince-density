from __future__ import annotations

import sys

import numpy as np
import pytest
from array_api_compat import array_namespace
from orng import RandomGenerator
from sequince.array_api import to_numpy

from sequince_density.gmm import GaussianMixtureProposal


def _bimodal_particles(seed: int = 4, n: int = 600) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.normal((-3.0, 0.0), (0.5, 0.7), size=(n // 2, 2)),
            rng.normal((3.0, 1.0), (0.6, 0.4), size=(n // 2, 2)),
        ]
    )


def test_fits_one_component_gaussian() -> None:
    particles = np.random.default_rng(2).normal(
        (1.0, -2.0), (0.6, 1.2), size=(1000, 2)
    )
    proposal = GaussianMixtureProposal(1).fit(particles)

    assert proposal.converged
    assert proposal.n_iter >= 2
    assert proposal.mixture_weights == pytest.approx([1.0])
    assert proposal.means[0] == pytest.approx([1.0, -2.0], abs=0.1)
    assert np.all(np.linalg.eigvalsh(proposal.covariances[0]) > 0)


def test_fits_and_redraws_separated_mixture() -> None:
    particles = _bimodal_particles()
    proposal = GaussianMixtureProposal(2).fit(particles)
    samples, log_prob = proposal.sample_and_log_prob(
        1000, rng=RandomGenerator(seed=12)
    )

    assert samples.shape == (1000, 2)
    assert log_prob.shape == (1000,)
    assert np.all(np.isfinite(log_prob))
    assert proposal.mixture_weights.sum() == pytest.approx(1.0)
    assert 0.35 < np.mean(np.asarray(samples)[:, 0] < 0.0) < 0.65


def test_sampling_is_reproducible() -> None:
    proposal = GaussianMixtureProposal(2).fit(_bimodal_particles())

    first = proposal.sample(50, rng=RandomGenerator(seed=123))
    second = proposal.sample(50, rng=RandomGenerator(seed=123))

    np.testing.assert_allclose(first, second)


def test_fitting_is_reproducible() -> None:
    particles = _bimodal_particles()
    first = GaussianMixtureProposal(2).fit(
        particles, rng=RandomGenerator(seed=123)
    )
    second = GaussianMixtureProposal(2).fit(
        particles, rng=RandomGenerator(seed=123)
    )

    np.testing.assert_allclose(first.means, second.means)
    np.testing.assert_allclose(first.covariances, second.covariances)


def test_regularization_handles_singular_particles() -> None:
    values = np.linspace(-1.0, 1.0, 50)
    particles = np.column_stack((values, 2.0 * values))
    proposal = GaussianMixtureProposal(2, regularization=1e-5).fit(particles)

    assert np.all(np.linalg.eigvalsh(proposal.covariances) > 0)
    assert np.all(np.isfinite(proposal.log_prob(particles)))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_components": 0}, "n_components"),
        ({"n_components": 1, "regularization": 0.0}, "regularization"),
        ({"n_components": 1, "covariance_type": "diag"}, "full"),
    ],
)
def test_validates_configuration(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GaussianMixtureProposal(**kwargs)


def test_missing_sklearn_has_installation_hint(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sklearn.mixture", None)

    with pytest.raises(ImportError, match=r"sequince-density\[gmm\]"):
        GaussianMixtureProposal(1)


def test_validates_fit_inputs() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        GaussianMixtureProposal(3).fit(np.ones((2, 1)))
    with pytest.raises(ValueError, match="finite"):
        GaussianMixtureProposal(1).fit(np.array([[np.nan], [0.0]]))


def test_rejects_nonuniform_weighted_fit() -> None:
    particles = _bimodal_particles(n=100)
    weights = np.linspace(1.0, 2.0, len(particles))

    with pytest.raises(NotImplementedError, match="Weighted GMM"):
        GaussianMixtureProposal(2).fit(particles, weights=weights)


def test_accepts_uniform_weighted_fit() -> None:
    particles = _bimodal_particles(n=100)
    weights = np.full(len(particles), 1.0 / len(particles))

    proposal = GaussianMixtureProposal(2).fit(particles, weights=weights)

    assert proposal.converged


def test_state_round_trip() -> None:
    fitted = GaussianMixtureProposal(2).fit(_bimodal_particles())
    restored = GaussianMixtureProposal(2)
    restored.load_state_dict(fitted.state_dict())

    points = np.array([[-3.0, 0.0], [3.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(
        restored.log_prob(points), fitted.log_prob(points)
    )
    np.testing.assert_allclose(restored.means, fitted.means)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_preserves_array_api_backend(backend: str) -> None:
    if backend == "torch":
        torch = pytest.importorskip("torch")
        particles = torch.tensor(
            _bimodal_particles(n=100), dtype=torch.float32
        )
        rng = RandomGenerator(backend="torch", seed=31)
    else:
        jnp = pytest.importorskip("jax.numpy")
        particles = jnp.asarray(_bimodal_particles(n=100), dtype=jnp.float32)
        rng = RandomGenerator(backend="jax", seed=31)

    proposal = GaussianMixtureProposal(2).fit(particles)
    samples = proposal.sample(12, rng=rng)
    log_prob = proposal.log_prob(samples)

    assert array_namespace(samples) is array_namespace(particles)
    assert array_namespace(log_prob) is array_namespace(particles)
    assert samples.shape == (12, 2)
    assert np.all(np.isfinite(to_numpy(log_prob)))


def test_requested_namespace_overrides_fitted_backend() -> None:
    torch = pytest.importorskip("torch")
    particles = torch.tensor(_bimodal_particles(n=100), dtype=torch.float32)
    proposal = GaussianMixtureProposal(2).fit(particles)

    samples, log_prob = proposal.sample_and_log_prob(
        12,
        rng=RandomGenerator(seed=31),
        xp=np,
    )

    assert isinstance(samples, np.ndarray)
    assert isinstance(log_prob, np.ndarray)
