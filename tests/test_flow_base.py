from __future__ import annotations

import numpy as np
from orng import RandomGenerator
from sequince.transforms import ParticleTransform

from sequince_density.flows.base import Flow


class RecordingTransform(ParticleTransform):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def fit(self, particles, context=None, *, rng=None):
        self.calls.append(("fit", context, rng, None))
        return self

    def forward(self, particles, context=None, *, rng=None):
        auxiliary = {"mode": np.ones_like(particles)}
        self.calls.append(("forward", context, rng, auxiliary))
        return particles + 1.0, np.zeros(particles.shape[0]), auxiliary

    def inverse(self, values, context=None, *, auxiliary=None, rng=None):
        self.calls.append(("inverse", context, rng, auxiliary))
        return values - 1.0, np.zeros(values.shape[0])


def test_particle_transform_helpers_preserve_auxiliary_and_rng() -> None:
    particles = np.zeros((3, 2))
    context = object()
    rng = RandomGenerator(seed=12)
    transform = RecordingTransform()
    flow = Flow(dims=2, particle_transform=transform)

    fitted = flow.fit_particle_transform(particles, context=context, rng=rng)
    transformed, log_det, auxiliary = flow.forward_particle_transform(
        particles, context=context, rng=rng
    )
    restored, inverse_log_det = flow.inverse_particle_transform(
        transformed,
        context=context,
        auxiliary=auxiliary,
        rng=rng,
    )

    np.testing.assert_array_equal(fitted, particles + 1.0)
    np.testing.assert_array_equal(restored, particles)
    np.testing.assert_array_equal(log_det, 0.0)
    np.testing.assert_array_equal(inverse_log_det, 0.0)
    assert [call[0] for call in transform.calls] == [
        "fit",
        "forward",
        "forward",
        "inverse",
    ]
    assert all(call[1] is context for call in transform.calls)
    assert all(call[2] is rng for call in transform.calls)
    assert transform.calls[-1][3] is auxiliary
