from __future__ import annotations

import numpy as np
import pytest
from orng import RandomGenerator
from sequince.transforms import (
    AffineTransform,
    CompositeTransform,
    FlowTransform,
)

pytest.importorskip("torch")
pytest.importorskip("zuko")

from sequince_density.flows.zuko import ZukoFlow  # noqa: E402


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
