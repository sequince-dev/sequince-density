"""Shared helpers for Torch-backed density implementations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from ._rng import draw_seed


@contextmanager
def torch_random_context(
    rng: Any = None,
    *,
    device: torch.device | str | None = None,
) -> Iterator[None]:
    """Temporarily seed Torch from an optional Sequince RNG.

    The caller's global Torch RNG state is restored on exit. CUDA state is
    included when a CUDA device is supplied.
    """
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


__all__ = ["torch_random_context"]
