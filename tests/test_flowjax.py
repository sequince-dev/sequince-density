from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flowjax")

from sequince.transforms import AffineTransform  # noqa: E402

from sequince_density.flows.flowjax import FlowJax  # noqa: E402


def test_flowjax_fits_samples_and_round_trips() -> None:
    key = jax.random.key(12)
    particles = jax.random.normal(key, (64, 2))
    flow = FlowJax(
        dims=2,
        key=key,
        particle_transform=AffineTransform(),
        flow_layers=1,
        nn_width=8,
        nn_depth=1,
    )

    returned = flow.fit(
        particles,
        max_epochs=1,
        batch_size=16,
        show_progress=False,
    )
    latent, forward_log_det = flow.forward(particles[:4])
    restored, inverse_log_det = flow.inverse(latent)
    samples, log_prob = flow.sample_and_log_prob(5)

    assert returned is flow
    assert isinstance(samples, jax.Array)
    assert samples.shape == (5, 2)
    assert log_prob.shape == (5,)
    np.testing.assert_allclose(restored, particles[:4], atol=1e-5)
    np.testing.assert_allclose(
        forward_log_det + inverse_log_det,
        0.0,
        atol=1e-5,
    )
