"""Utility helpers shared by density implementations."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from orng import RandomGenerator
from sequince.array_api import namespace, to_numpy
from sequince.typing import Array


def validate_training_data(
    samples: Array, dims: int, stage: str = "Raw"
) -> None:
    """Validate training particles using their native array backend.

    Parameters
    ----------
    samples : array, shape (n_samples, dims)
        Nonempty, finite particle coordinates. Validation does not modify them.
    dims : int
        Expected number of dimensions.
    stage : str, default="Raw"
        Label for the data in error messages, for example "Transformed".

    Raises
    ------
    ValueError
        If the shape is invalid or any coordinates are non-finite. Non-finite
        errors report the affected dimension indices (zero-based).
    """
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != dims:
        raise ValueError(
            f"{stage} training data must have nonempty shape "
            f"(n_samples, {dims})."
        )
    xp = namespace(samples)
    invalid = ~xp.isfinite(samples)
    if xp.any(invalid):
        dimensions = [
            i
            for i, affected in enumerate(
                to_numpy(xp.any(invalid, axis=0)).tolist()
            )
            if affected
        ]
        raise ValueError(
            f"{stage} training data contains non-finite values "
            f"in dimensions: {dimensions}"
        )


def draw_seed(rng: RandomGenerator | int | None) -> int | None:
    """Consume one deterministic integer seed from an optional RNG."""
    if rng is None or isinstance(rng, int):
        return rng
    if not isinstance(rng, RandomGenerator):
        raise TypeError(
            "rng must be an orng.RandomGenerator, an int, or None."
        )
    value = rng.uniform(0.0, float(2**31 - 1))
    return int(to_numpy(value).item())


@contextmanager
def torch_random_context(
    rng: RandomGenerator | int | None = None,
    *,
    device: Any = None,
) -> Generator[None, None, None]:
    """Temporarily seed Torch and restore its global RNG state on exit."""
    import torch

    seed = draw_seed(rng)
    if seed is None:
        yield
        return

    resolved_device = None if device is None else torch.device(device)
    devices = (
        [resolved_device]
        if resolved_device is not None and resolved_device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


__all__ = ["draw_seed", "torch_random_context", "validate_training_data"]
