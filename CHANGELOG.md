# Changelog

Notable changes to sequince-density are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Documentation sidebar links to the main Sequince documentation and
  sequince-examples under Related Packages.

## [0.1.0b1] - 2026-10-27

First public beta release.

### Added

- Normalizing-flow adapters for Zuko (PyTorch) and FlowJAX (JAX), supporting
  fitting, sampling, density evaluation, and forward/inverse transformations.
- Gaussian-mixture proposals fitted with scikit-learn, using full covariance
  matrices and equal-weight training populations. Fitting and density evaluation
  use NumPy on the CPU.
- Integration with Sequince's `FlowProposal` and `FlowTransform` for fitted
  starting proposals and flow-preconditioned mutation.
- Explicit RNG arguments for fitting and sampling, and checkpoint restoration of
  fitted models, particle transforms, and training histories. FlowJAX checkpoints
  also preserve the internal random key.

[Unreleased]: https://github.com/sequince-dev/sequince-density/compare/v0.1.0b1...HEAD
[0.1.0b1]: https://github.com/sequince-dev/sequince-density/releases/tag/v0.1.0b1
