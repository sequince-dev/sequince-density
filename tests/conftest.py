from __future__ import annotations

import numpy as np
import pytest
from sequince.transforms import (
    AffineTransform,
    CompositeTransform,
    IdentityTransform,
    ParticleTransform,
    PeriodicBoundaryTransform,
)


@pytest.fixture(params=["direct", "composite", "nested"])
def periodic_transform(request: pytest.FixtureRequest) -> ParticleTransform:
    transform: ParticleTransform = PeriodicBoundaryTransform(
        bounds={0: (0.0, 2.0 * np.pi)}
    )
    if request.param == "nested":
        transform = CompositeTransform([IdentityTransform(), transform])
    if request.param != "direct":
        transform = CompositeTransform([AffineTransform(), transform])
    return transform
