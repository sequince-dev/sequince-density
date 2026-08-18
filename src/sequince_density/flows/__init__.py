"""Normalizing-flow backend adapters."""

from .base import Flow

try:
    from .flowjax import FlowJax
except ImportError:  # pragma: no cover - optional backend
    FlowJax = None  # type: ignore[assignment,misc]

try:
    from .zuko import ZukoFlow
except ImportError:  # pragma: no cover - optional backend
    ZukoFlow = None  # type: ignore[assignment,misc]

__all__ = ["Flow", "FlowJax", "ZukoFlow"]
