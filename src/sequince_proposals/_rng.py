"""Random-state helpers shared by backend implementations."""

from __future__ import annotations

from orng import RandomGenerator

from ._array import to_numpy


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


__all__ = ["draw_seed"]
