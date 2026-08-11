import importlib.metadata

try:
    __version__ = importlib.metadata.version("sequince-proposals")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .base import Flow
from .gmm import GaussianMixtureProposal

try:
    from .zuko import ZukoFlow
except ImportError:  # pragma: no cover - optional backend
    ZukoFlow = None  # type: ignore[assignment,misc]

__all__ = [
    "Flow",
    "GaussianMixtureProposal",
    "ZukoFlow",
]
