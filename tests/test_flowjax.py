from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from orng import RandomGenerator

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flowjax")

from sequince.transforms import AffineTransform  # noqa: E402

from sequince_density.flows.flowjax import FlowJax  # noqa: E402


@pytest.fixture
def fitted_flow() -> FlowJax:
    particles = np.random.default_rng(21).normal(size=(64, 2))
    particles = particles * [2.0, 3.0] + [4.0, -2.0]
    flow = FlowJax(
        dims=2,
        particle_transform=AffineTransform(),
        seed=22,
        flow_layers=1,
        nn_width=8,
        nn_depth=1,
    )
    return flow.fit(
        particles,
        rng=23,
        max_epochs=1,
        batch_size=16,
        show_progress=False,
    )


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
        rng=RandomGenerator(seed=19),
        max_epochs=1,
        batch_size=16,
        show_progress=False,
    )
    latent, forward_log_det = flow.forward(particles[:4])
    restored, inverse_log_det = flow.inverse(latent)
    samples, log_prob = flow.sample_and_log_prob(
        5, rng=RandomGenerator(seed=20)
    )

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


@pytest.mark.parametrize("n_samples", [1, 8])
def test_sample_and_log_prob_matches_log_prob(
    fitted_flow: FlowJax, n_samples: int
) -> None:
    samples, log_prob = fitted_flow.sample_and_log_prob(n_samples, rng=24)

    assert samples.shape == (n_samples, 2)
    assert log_prob.shape == (n_samples,)
    assert np.all(np.isfinite(samples))
    assert np.all(np.isfinite(log_prob))
    # Inverting the float32 flow introduces round-off in density evaluation.
    np.testing.assert_allclose(
        log_prob, fitted_flow.log_prob(samples), rtol=1e-5, atol=1e-5
    )


def test_checkpoint_restores_fitted_density_and_rng_sequence(
    fitted_flow: FlowJax, tmp_path: Path
) -> None:
    # Save an advanced internal key so restoring constructor state is insufficient.
    fitted_flow.sample(3)
    checkpoint = tmp_path / "flow.pkl"
    checkpoint.write_bytes(pickle.dumps(fitted_flow.state_dict()))
    restored = FlowJax(
        dims=2,
        seed=99,
        flow_layers=1,
        nn_width=8,
        nn_depth=1,
    )
    restored.load_state_dict(pickle.loads(checkpoint.read_bytes()))

    points = jnp.array([[4.0, -2.0], [6.0, 1.0]], dtype=jnp.float32)
    np.testing.assert_array_equal(
        restored.log_prob(points), fitted_flow.log_prob(points)
    )
    for _ in range(2):
        np.testing.assert_array_equal(
            restored.sample(8), fitted_flow.sample(8)
        )
    assert fitted_flow.fit_history["training_loss"]
    assert fitted_flow.fit_history["validation_loss"]
    assert restored.fit_history == fitted_flow.fit_history
