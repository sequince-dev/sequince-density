from importlib.util import find_spec

import numpy as np
import pytest
from sequince.proposals import FlowProposal
from sequince.transforms import (
    AffineTransform,
    BoundedToUnboundedTransform,
    CompositeTransform,
)

from sequince_density import CoppuccinoFlow


def test_coppuccino_rejects_bounded_transforms() -> None:
    transform = CompositeTransform(
        [BoundedToUnboundedTransform({0: (0, 1)}), AffineTransform()]
    )
    with pytest.raises(ValueError, match="bounded-to-unbounded"):
        CoppuccinoFlow(dims=2, particle_transform=transform)
    with pytest.raises(ValueError, match="unbounded marginals"):
        CoppuccinoFlow(dims=2, prior_bounds=[[-1, 1], [-1, 1]])


@pytest.mark.skipif(find_spec("coppuccino") is None, reason="needs coppuccino")
def test_coppuccino_proposal_and_transform() -> None:
    import jax
    from jax._src.config import enable_x64 as x64_state

    initial_x64 = jax.config.jax_enable_x64
    particles = np.random.default_rng(21).normal(size=(128, 2))
    particles = particles * [2.0, 3.0] + [4.0, -2.0]
    flow = CoppuccinoFlow(
        dims=2, particle_transform=AffineTransform(), flow_layers=1
    )
    proposal = FlowProposal(
        flow, fit_kwargs={"max_epochs": 2, "batch_size": 32}
    )
    with getattr(jax, "enable_x64", x64_state)(not initial_x64):
        proposal.fit(particles, rng=22)
        assert jax.config.jax_enable_x64 != initial_x64
    assert jax.config.jax_enable_x64 == initial_x64

    samples, log_prob = proposal.sample_and_log_prob(16, rng=23, xp=np)
    np.testing.assert_allclose(
        proposal.sample(16, rng=23, xp=np), samples, atol=1e-10
    )
    np.testing.assert_allclose(proposal.log_prob(samples), log_prob, atol=1e-8)

    latent, forward_log_det = flow.forward(samples)
    restored, inverse_log_det = flow.inverse(latent)
    np.testing.assert_allclose(restored, samples, atol=1e-8)
    np.testing.assert_allclose(forward_log_det, -inverse_log_det, atol=1e-8)
    base_log_prob = -0.5 * np.sum(latent**2 + np.log(2 * np.pi), axis=1)
    np.testing.assert_allclose(
        base_log_prob + forward_log_det, log_prob, atol=1e-8
    )
    single_latent, single_log_det = flow.forward(samples[0])
    single_sample, single_inverse_log_det = flow.inverse(single_latent)
    assert single_latent.shape == single_sample.shape == (2,)
    assert single_log_det.shape == single_inverse_log_det.shape == ()
    np.testing.assert_allclose(single_sample, samples[0], atol=1e-8)
    np.testing.assert_allclose(
        single_log_det, -single_inverse_log_det, atol=1e-8
    )
    np.testing.assert_allclose(
        flow.log_prob(samples[0]), log_prob[0], atol=1e-8
    )
    assert jax.config.jax_enable_x64 == initial_x64
