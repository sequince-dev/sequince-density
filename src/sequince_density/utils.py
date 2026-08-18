"""Utility helpers shared by density implementations."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from orng import RandomGenerator
from sequince.array_api import to_numpy


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


__all__ = ["draw_seed", "torch_random_context"]
