from __future__ import annotations

import numpy as np
import pytest

from sequince_density.utils import validate_training_data


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax.numpy"])
def test_validate_training_data_across_backends(backend: str) -> None:
    xp = pytest.importorskip(backend)
    values = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    particles = xp.asarray(values)
    validate_training_data(particles, dims=3)
    np.testing.assert_array_equal(np.asarray(particles), values)

    invalid = values.copy()
    invalid[0, 0] = np.nan
    invalid[1, 2] = np.inf
    with pytest.raises(
        ValueError, match=r"Transformed.*non-finite.*dimensions: \[0, 2\]"
    ):
        validate_training_data(
            xp.asarray(invalid), dims=3, stage="Transformed"
        )

    with pytest.raises(ValueError, match="Raw training data.*shape"):
        validate_training_data(particles, dims=2)
