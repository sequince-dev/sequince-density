# sequince-density

`sequince-density` provides density estimators for use with the `sequince` SMC
framework, including fitted proposal distributions and normalizing flows for
preconditioned sampling.

## Installation

`sequince-density` can be installed directly from PyPI:

```bash
pip install sequince-density
```

However, we recommend also installing the relevant dependencies for the features
you plan to use. See below for more details.

## Features

- Normalizing-flow adapters for Zuko (PyTorch) and FlowJAX (JAX).
- Gaussian-mixture proposals fitted with scikit-learn.
- Integration with Sequince's `FlowTransform` for mutation in a learned latent
  space and `FlowProposal` for fitted proposal distributions.

## Quick start

### Normalizing flows

Install the optional dependencies for your chosen backend:

```bash
pip install "sequince-density[zuko]"
# Alternatively, for the JAX backend:
pip install "sequince-density[flowjax]"
```

Fit a Zuko flow to a two-dimensional sample population, then draw samples and
evaluate their log densities:

```python
import numpy as np

from sequince_density import ZukoFlow

rng = np.random.default_rng(42)
particles = rng.normal(size=(2000, 2))

flow = ZukoFlow(dims=2, flow_class="MAF", hidden_features=[32, 32])
flow.fit(particles, n_epochs=100, batch_size=500, rng=42)
samples, log_q = flow.sample_and_log_prob(256, rng=43)
```

`ZukoFlow` returns PyTorch tensors; `FlowJax` returns JAX arrays.

For preconditioned SMC, pass an unfitted flow to `FlowTransform` and use the
transform as the `particle_transform` argument to
`make_standard_bayesian_smc`. The mutation step fits it to the current particle
population:

```python
from sequince.transforms import FlowTransform

transform = FlowTransform(
    ZukoFlow(dims=2, flow_class="MAF", hidden_features=[32, 32]),
    fit_kwargs={"n_epochs": 100, "batch_size": 500},
)
```

### Gaussian mixture models

GMM support is provided via scikit-learn and can be installed with:

```bash
pip install "sequince-density[gmm]"
```

```python
import numpy as np

from sequince_density import GaussianMixtureProposal

particles = np.random.default_rng(42).normal(size=(2000, 2))
proposal = GaussianMixtureProposal(n_components=4).fit(particles, rng=42)
samples, log_q = proposal.sample_and_log_prob(256, rng=43)
```

Fitting expects particles shaped `(n_particles, n_dimensions)`. Nonuniform
weights are not supported; use an equal-weight population, such as particles
after resampling.

## Proposal distributions

Both Gaussian mixtures and normalizing flows can provide a fitted starting
distribution for SMC. `GaussianMixtureProposal` implements Sequince's proposal
interface directly. Wrap a flow in `FlowProposal` to use the same interface,
including conversion to the SMC particle backend:

```python
from sequince.proposals import FlowProposal

proposal = FlowProposal(
    ZukoFlow(dims=2, flow_class="MAF", hidden_features=[32, 32]),
    fit_kwargs={"n_epochs": 100, "batch_size": 500},
).fit(particles, rng=42)
```

For either fitted proposal type, pass `proposal` as `log_proposal` to
`make_standard_bayesian_smc` and draw the initial particles from that same
proposal. SMC then bridges from the proposal to the target posterior.

In an ASPIRE-style analysis, a flow is fitted to samples from an initial
analysis and used as the starting proposal for a new SMC run with an updated
likelihood.

Full proposal examples will be linked here once published in
[`sequince-examples`](https://github.com/sequince-dev/sequince-examples).

## Documentation

See the [sequince density documentation](https://sequince.readthedocs.io/projects/sequince-density/en/latest/) for installation,
density fitting, and integration with Sequince proposals and transforms.

See the [changelog](CHANGELOG.md) for release notes and unreleased changes.

## Examples

The [`sequince-examples`](https://github.com/sequince-dev/sequince-examples)
repository and corresponding documentation contain examples highlighting
different features in `sequince`.

## Contributing

Bug reports, feature requests, and pull requests are welcome via the GitHub
repository. Please see the [contributing guidelines](CONTRIBUTING.md)
and [AI policy](https://github.com/sequince-dev/sequince/blob/main/AI_POLICY.md).


## Related repositories

The following repositories are part of the `sequince` ecosystem:

- [`sequince`](https://github.com/sequince-dev/sequince): the main `sequince` repository
- [`sequince-bilby`](https://github.com/sequince-dev/sequince-bilby): provides an interface for using prebuilt `sequince` samplers with `bilby`
- [`orng`](https://github.com/sequince-dev/orng): provides cross-backend random number generation
- [`acai`](https://github.com/sequince-dev/acai): provides array-API compatible control flow and iteration


## License

Licensed under the [MIT License](LICENSE).
