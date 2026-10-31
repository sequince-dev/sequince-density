# Changelog

Notable changes to sequince-density are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Optional Zuko gradient clipping, cosine learning-rate annealing, early stopping
  and progress display, available through flow adapters' `fit_kwargs`.
- Zuko training-data checks before and after particle transforms, reporting
  affected dimensions for non-finite values.
- Software archive DOI and method references for flow preconditioning and
  masked autoregressive flows.
- Documentation sidebar links to the main Sequince documentation and
  sequince-examples under Related Packages.

### Changed

- Zuko's `validation_fraction=0` skips validation and keeps final parameters,
  with an empty validation-loss history. Combining it with `patience` raises
  `ValueError`.

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
