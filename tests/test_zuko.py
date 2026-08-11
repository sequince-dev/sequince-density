from __future__ import annotations

import numpy as np
from sequince.transforms import (
    AffineTransform,
    CompositeTransform,
    FlowTransform,
)

from sequince_proposals import ZukoFlow


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
    transform = transform.fit(particles)
    latent, log_det = transform.forward(particles)
    restored, inverse_log_det = transform.inverse(latent)

    assert isinstance(latent, np.ndarray)
    assert latent.shape == particles.shape
    assert log_det.shape == (particles.shape[0],)
    np.testing.assert_allclose(restored, particles, atol=1e-5)
    np.testing.assert_allclose(log_det + inverse_log_det, 0.0, atol=1e-5)
