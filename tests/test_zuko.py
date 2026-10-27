from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from orng import RandomGenerator
from sequince.transforms import (
    AffineTransform,
    CompositeTransform,
    FlowTransform,
    ParticleTransform,
)

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")

from sequince_density.flows.zuko import ZukoFlow  # noqa: E402


@pytest.fixture
def fitted_flow() -> ZukoFlow:
    particles = np.random.default_rng(21).normal(size=(64, 2))
    particles = particles * [2.0, 3.0] + [4.0, -2.0]
    flow = ZukoFlow(
        dims=2,
        hidden_features=[8, 8],
        particle_transform=AffineTransform(),
        seed=22,
        device="cpu",
        dtype=torch.float32,
    )
    return flow.fit(particles, n_epochs=2, batch_size=16, rng=23)


def test_zuko_flow_transform_fits_and_round_trips() -> None:
    rng = np.random.default_rng(1234)
    particles = rng.normal(size=(64, 2))
    flow = ZukoFlow(
        dims=2,
        hidden_features=[8, 8],
        device="cpu",
    )

    transform = CompositeTransform(
        [
            AffineTransform(),
            FlowTransform(
                flow,
                fit_kwargs={
                    "n_epochs": 2,
                    "batch_size": 16,
                    "validation_fraction": 0.25,
                },
            ),
        ]
    )
    transform_rng = RandomGenerator(seed=19)
    transform = transform.fit(particles, rng=transform_rng)
    latent, log_det, auxiliary = transform.forward(
        particles, rng=transform_rng
    )
    restored, inverse_log_det = transform.inverse(
        latent, auxiliary=auxiliary, rng=transform_rng
    )

    assert isinstance(latent, np.ndarray)
    assert latent.shape == particles.shape
    assert log_det.shape == (particles.shape[0],)
    np.testing.assert_allclose(restored, particles, atol=1e-5)
    np.testing.assert_allclose(log_det + inverse_log_det, 0.0, atol=1e-5)


@pytest.mark.parametrize("n_samples", [1, 8])
def test_sample_and_log_prob_matches_log_prob(
    fitted_flow: ZukoFlow, n_samples: int
) -> None:
    samples, log_prob = fitted_flow.sample_and_log_prob(n_samples, rng=24)

    assert samples.shape == (n_samples, 2)
    assert log_prob.shape == (n_samples,)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(log_prob).all()
    # Inverting the float32 flow introduces round-off in density evaluation.
    torch.testing.assert_close(
        log_prob, fitted_flow.log_prob(samples), rtol=1e-5, atol=1e-5
    )


def test_checkpoint_restores_fitted_density_and_seeded_samples(
    fitted_flow: ZukoFlow, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "flow.pkl"
    checkpoint.write_bytes(pickle.dumps(fitted_flow.state_dict()))
    restored = ZukoFlow(
        dims=2,
        hidden_features=[8, 8],
        seed=99,
        device="cpu",
        dtype=torch.float32,
    )
    restored.load_state_dict(pickle.loads(checkpoint.read_bytes()))

    points = torch.tensor([[4.0, -2.0], [6.0, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(
        restored.log_prob(points), fitted_flow.log_prob(points), rtol=0, atol=0
    )
    for seed in (25, 26):
        torch.testing.assert_close(
            restored.sample(8, rng=seed),
            fitted_flow.sample(8, rng=seed),
            rtol=0,
            atol=0,
        )
    assert fitted_flow.fit_history["training_loss"]
    assert fitted_flow.fit_history["validation_loss"]
    assert restored.fit_history == fitted_flow.fit_history


def test_rejects_periodic_particle_transform(
    periodic_transform: ParticleTransform,
) -> None:
    with pytest.raises(ValueError, match="PeriodicBoundaryTransform"):
        ZukoFlow(dims=2, particle_transform=periodic_transform)


def test_checkpoint_rejects_periodic_particle_transform(
    periodic_transform: ParticleTransform,
) -> None:
    source = ZukoFlow(dims=2, hidden_features=[8, 8], seed=22)
    flow = ZukoFlow(dims=2, hidden_features=[8, 8], seed=99)
    original_transform = flow.particle_transform
    points = torch.tensor([[0.5, -0.2], [4.0, 1.0]])
    original_log_prob = flow.log_prob(points).detach().clone()
    state = source.state_dict()
    state["particle_transform"] = periodic_transform

    with pytest.raises(ValueError, match="PeriodicBoundaryTransform"):
        flow.load_state_dict(state)

    assert flow.particle_transform is original_transform
    torch.testing.assert_close(
        flow.log_prob(points), original_log_prob, rtol=0, atol=0
    )
