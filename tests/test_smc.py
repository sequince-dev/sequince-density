from __future__ import annotations

import numpy as np
import pytest
from orng import RandomGenerator
from sequince import ParticleState
from sequince.posterior import TemperedPosterior
from sequince.prebuilt_samplers import make_standard_bayesian_smc
from sequince.steps import GaussianMixtureRejuvenation, SystematicResampling
from sequince.steps.base import StepContext
from sequince.steps.mutation import get_mutation_step
from sequince.transforms import AffineTransform

from sequince_proposals import GaussianMixtureProposal


def _standard_normal(particles, context=None):
    del context
    values = np.asarray(particles)
    return -0.5 * np.sum(values**2, axis=-1)


def _zero_likelihood(particles, context=None):
    del context
    return np.zeros(np.asarray(particles).shape[0])


@pytest.fixture
def standard_normal_posterior() -> TemperedPosterior:
    return TemperedPosterior(_standard_normal, _zero_likelihood)


@pytest.fixture
def quadratic_posterior() -> TemperedPosterior:
    return TemperedPosterior(
        lambda particles: -np.sum(particles**2, axis=-1),
        lambda particles: np.sum(particles, axis=-1),
    )


def _population() -> np.ndarray:
    rng = np.random.default_rng(22)
    return np.concatenate(
        [rng.normal(-3.0, 0.5, (200, 1)), rng.normal(3.0, 0.5, (200, 1))]
    )


def test_rejuvenation_replaces_population_and_refreshes_target(
    quadratic_posterior,
) -> None:
    particles = _population()
    context = StepContext(t=3, metadata={"temperature": 0.7})
    old_evaluation = quadratic_posterior.prepare_evaluation(particles, context)
    state = ParticleState(
        particles,
        log_weights=np.linspace(-3.0, 0.0, len(particles)),
        log_evidence=4.25,
        evaluation=old_evaluation,
    )
    step = GaussianMixtureRejuvenation(
        quadratic_posterior,
        GaussianMixtureProposal(n_components=2),
        transform=AffineTransform(),
    )

    updated, returned_context, output = step.apply(
        state, context, RandomGenerator(seed=7)
    )

    assert returned_context is context
    assert updated.particles.shape == particles.shape
    assert not np.array_equal(updated.particles, particles)
    assert updated.log_weights == pytest.approx(-np.log(len(particles)))
    assert updated.log_evidence == pytest.approx(4.25)
    assert updated.evaluation is not None
    assert updated.evaluation is not old_evaluation
    expected = quadratic_posterior.evaluate(updated.particles, context)
    for component in expected.components:
        np.testing.assert_allclose(
            updated.evaluation[component], expected[component]
        )
    assert output["n_components"] == 2
    assert output["converged"]
    assert output["approximate"] is True


def test_rejuvenation_operates_after_resampling(
    standard_normal_posterior,
) -> None:
    particles = _population()
    state = ParticleState(
        particles,
        log_weights=np.concatenate([np.full(200, -20.0), np.zeros(200)]),
    )
    context = StepContext(t=0)
    rng = RandomGenerator(seed=19)
    resampled, context, output = SystematicResampling(
        always_resample=True
    ).apply(state, context, rng)
    step = GaussianMixtureRejuvenation(
        standard_normal_posterior, GaussianMixtureProposal(n_components=2)
    )

    updated, _, _ = step.apply(resampled, context, rng)

    assert output["resampled"] is True
    assert np.allclose(resampled.weights, 1.0 / len(particles))
    assert np.allclose(updated.weights, 1.0 / len(particles))


def test_factory_builds_gmm(standard_normal_posterior) -> None:
    mutation = get_mutation_step(
        "gmm",
        standard_normal_posterior,
        mutation_kwargs={"proposal": GaussianMixtureProposal(3)},
    )

    assert isinstance(mutation, GaussianMixtureRejuvenation)
    assert mutation.proposal.n_components == 3


def test_factory_rejects_multiple_steps(standard_normal_posterior) -> None:
    with pytest.raises(ValueError, match="mutation_n_steps=1"):
        get_mutation_step(
            "gmm",
            standard_normal_posterior,
            mutation_n_steps=2,
            mutation_kwargs={"proposal": GaussianMixtureProposal(2)},
        )


def test_prebuilt_sampler_discovers_gmm() -> None:
    smc, _, _ = make_standard_bayesian_smc(
        _standard_normal,
        _zero_likelihood,
        mutation="gmm",
        mutation_kwargs={"proposal": GaussianMixtureProposal(2)},
    )

    assert isinstance(smc._steps[2], GaussianMixtureRejuvenation)  # noqa: SLF001


def test_functional_api_is_explicitly_unsupported(
    standard_normal_posterior,
) -> None:
    step = GaussianMixtureRejuvenation(
        standard_normal_posterior, GaussianMixtureProposal(n_components=1)
    )
    with pytest.raises(NotImplementedError, match="functional/JIT"):
        step.apply_functional(
            ParticleState(_population()), StepContext(t=0), 0
        )


def test_rejuvenation_checkpoint_round_trip(
    standard_normal_posterior,
) -> None:
    step = GaussianMixtureRejuvenation(
        standard_normal_posterior, GaussianMixtureProposal(n_components=2)
    )
    step.apply(ParticleState(_population()), StepContext(t=0), 4)
    restored = GaussianMixtureRejuvenation(
        standard_normal_posterior, GaussianMixtureProposal(n_components=2)
    )

    restored.load_state_dict(step.state_dict())

    np.testing.assert_allclose(restored.proposal.means, step.proposal.means)
