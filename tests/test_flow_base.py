from __future__ import annotations

import numpy as np
import pytest
from orng import RandomGenerator
from sequince.transforms import (
    AffineTransform,
    CompositeTransform,
    IdentityTransform,
    ParticleTransform,
)

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


def test_rejects_periodic_particle_transform(
    periodic_transform: ParticleTransform,
) -> None:
    with pytest.raises(ValueError, match="PeriodicBoundaryTransform"):
        Flow(dims=2, particle_transform=periodic_transform)


def test_rejects_periodic_particle_transform_assignment(
    periodic_transform: ParticleTransform,
) -> None:
    flow = Flow(dims=2)
    original_transform = flow.particle_transform

    with pytest.raises(ValueError, match="PeriodicBoundaryTransform"):
        flow.particle_transform = periodic_transform

    assert flow.particle_transform is original_transform


def test_rejects_periodic_transform_returned_by_fit(
    periodic_transform: ParticleTransform,
) -> None:
    class FittingTransform(ParticleTransform):
        def fit(
            self,
            particles: np.ndarray,
            context: object | None = None,
            *,
            rng: RandomGenerator | int | None = None,
        ) -> ParticleTransform:
            return periodic_transform

    transform = FittingTransform()
    flow = Flow(dims=2, particle_transform=transform)

    with pytest.raises(ValueError, match="PeriodicBoundaryTransform"):
        flow.fit_particle_transform(np.ones((4, 2)))

    assert flow.particle_transform is transform


@pytest.mark.parametrize(
    "transform",
    [
        None,
        IdentityTransform(),
        AffineTransform(),
        CompositeTransform([AffineTransform(), IdentityTransform()]),
    ],
    ids=["default", "identity", "affine", "composite"],
)
def test_nonperiodic_particle_transform_round_trip(
    transform: ParticleTransform | None,
) -> None:
    particles = np.array([[0.0, 1.0], [2.0, 4.0], [3.0, 6.0], [4.0, 9.0]])
    flow = Flow(dims=2, particle_transform=transform)

    flow.fit_particle_transform(particles)
    values, forward_log_det, auxiliary = flow.forward_particle_transform(
        particles
    )
    restored, inverse_log_det = flow.inverse_particle_transform(
        values, auxiliary=auxiliary
    )

    np.testing.assert_allclose(restored, particles, atol=1e-12)
    np.testing.assert_allclose(forward_log_det + inverse_log_det, 0.0)
