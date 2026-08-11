"""Small cross-backend conversion helpers used by proposal implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
from array_api_compat import array_namespace, device, is_torch_array, to_device


def namespace(array: Any):
    return array_namespace(array)


def to_numpy(array: Any, *, dtype: Any | None = None) -> np.ndarray:
    if is_torch_array(array):
        array = array.detach().cpu().numpy()
    else:
        try:
            array = to_device(array, "cpu")
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            if hasattr(array, "get"):
                array = array.get()
            elif hasattr(array, "cpu"):
                array = array.cpu()
        if hasattr(array, "numpy"):
            array = array.numpy()
    return np.asarray(array, dtype=dtype)


def asarray_like(array: Any, like: Any):
    xp = namespace(like)
    try:
        result = xp.asarray(array, dtype=like.dtype)
    except (TypeError, ValueError):
        result = xp.asarray(to_numpy(array), dtype=like.dtype)
    return to_device(result, device(like))
