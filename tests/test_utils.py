from __future__ import annotations

import pytest
from orng import RandomGenerator

from sequince_density.utils import (  # noqa: E402
    draw_seed,
    torch_random_context,
)


def test_draw_seed_is_reproducible() -> None:
    first = draw_seed(RandomGenerator(seed=123))
    second = draw_seed(RandomGenerator(seed=123))

    assert first == second
    assert draw_seed(42) == 42
    assert draw_seed(None) is None


def test_draw_seed_rejects_unknown_rng() -> None:
    with pytest.raises(TypeError, match="orng.RandomGenerator"):
        draw_seed(object())


def test_torch_random_context_is_reproducible_and_restores_state() -> None:
    torch = pytest.importorskip("torch")
    initial_state = torch.random.get_rng_state().clone()

    with torch_random_context(123):
        first = torch.rand(5)
    with torch_random_context(123):
        second = torch.rand(5)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(torch.random.get_rng_state(), initial_state)
