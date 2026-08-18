import importlib.metadata

try:
    __version__ = importlib.metadata.version("sequince-density")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .flows import Flow, FlowJax, ZukoFlow
from .gmm import GaussianMixtureProposal

__all__ = [
    "Flow",
    "FlowJax",
    "GaussianMixtureProposal",
    "ZukoFlow",
]
