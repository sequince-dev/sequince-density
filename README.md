# sequince-density

Fitted proposal distributions and optional normalizing-flow backends for
`sequince`.

## Installation

```bash
pip install sequince-density
```

## Gaussian-mixture proposal

Install the scikit-learn integration with:

```bash
pip install "sequince-density[gmm]"
```

```python
from orng import RandomGenerator
from sequince_density import GaussianMixtureProposal

proposal = GaussianMixtureProposal(n_components=4).fit(
    particles,
    rng=RandomGenerator(seed=122),
)
samples, log_q = proposal.sample_and_log_prob(
    256,
    rng=RandomGenerator(seed=123),
)
```

Fitting uses scikit-learn's `GaussianMixture`. Inputs may use NumPy, PyTorch,
JAX, or CuPy arrays; samples and densities are restored to the input backend
and device unless an explicit `xp` is requested when sampling. Full covariance
matrices, regularization, and checkpoint state are supported. Weighted fitting
is not currently supported; uniform post-resampling populations are the
intended input.

## Sequince rejuvenation

`sequince-density` depends on `sequince`; Sequince does not depend on this
package. Inject the proposal into Sequince's backend-agnostic rejuvenation
step and place it explicitly after resampling:

```python
from sequince.steps import ProposalRejuvenation
from sequince_density import GaussianMixtureProposal

rejuvenation = ProposalRejuvenation(
    posterior,
    GaussianMixtureProposal(n_components=8),
)
```

This direct redraw is approximate and deliberately has no Metropolis-Hastings
correction. Sequince fits it after resampling in the standard step order.

## Available flow backends

Install the Zuko integration with `pip install "sequince-density[zuko]"` or
the FlowJAX integration with `pip install "sequince-density[flowjax]"`.

```python
from sequince_density import FlowJax, ZukoFlow
```

`ZukoFlow` wraps flows from `zuko.flows`, while `FlowJax` wraps constructors
from `flowjax.flows`. Both satisfy the `sequince.flows.NormalizingFlow`
protocol and return arrays in their native backend; `FlowProposal` handles any
conversion to the SMC particle namespace.

## Flow as a mutation transform

Use `FlowTransform` when you want Sequince mutation steps to operate in a
learned latent space.

```python
import numpy as np

from sequince import ParticleState, make_standard_bayesian_smc
from sequince.transforms import AffineTransform, FlowTransform
from sequince_density import ZukoFlow

rng = np.random.default_rng(123)
dims = 2
n_particles = 256

initial_particles = rng.normal(size=(n_particles, dims))
initial_state = ParticleState(particles=initial_particles)

flow = ZukoFlow(
    dims=dims,
    flow_class="MAF",
    hidden_features=[64, 64],
    particle_transform=AffineTransform(),
)

smc, posterior, metadata = make_standard_bayesian_smc(
    log_prior=lambda x, context=None: -0.5 * np.sum(x**2, axis=1),
    log_likelihood=lambda x, context=None: np.zeros(x.shape[0]),
    particle_transform=FlowTransform(
        flow,
        fit_kwargs={
            "n_epochs": 100,
            "lr": 1e-3,
            "batch_size": 500,
        },
    ),
    mutation_kwargs={"transform_refit": "per-context"},
)

result = smc.run(initial_state, rng=rng, metadata=metadata)
```

A `FlowTransform` is fitted on the current particle cloud when the mutation
step prepares its transform, so you normally pass an unfitted flow adapter.
The mutation transform fit policy controls its lifetime:

- `always` fits before every mutation call.
- `per-context` fits once per outer SMC iteration and reuses the flow across
  repeated or alternating mutation steps.
- `once` fits once for the lifetime of the mutation step.

## Flow as a proposal distribution

Use `FlowProposal` when you want a trained flow to provide `log q(x)` for a
bridged target, and optionally to draw proposal samples.

```python
import numpy as np

from sequince import TemperedPosterior
from sequince.proposals import FlowProposal
from sequince_density import ZukoFlow

training_particles = np.random.default_rng(123).normal(size=(2000, 2))

proposal = FlowProposal(
    ZukoFlow(
        dims=2,
        flow_class="MAF",
        hidden_features=[64, 64],
    ),
    fit_kwargs={"n_epochs": 100, "lr": 1e-3, "batch_size": 500},
).fit(training_particles, rng=123)

posterior = TemperedPosterior(
    log_prior=lambda x, context=None: -0.5 * np.sum(x**2, axis=1),
    log_likelihood=lambda x, context=None: np.zeros(x.shape[0]),
    log_proposal=proposal,
)

samples, log_q = proposal.sample_and_log_prob(
    128,
    rng=123,
    xp=np,
)
values = posterior.log_prob(samples, context=None)
```

`FlowProposal.sample_and_log_prob()` uses the flow backend's native
implementation when available and otherwise falls back to `sample()` plus
`log_prob()`. Both `GaussianMixtureProposal` and a flow wrapped by
`FlowProposal` satisfy Sequince's `TrainableProposal` protocol, with the same
`fit`, `sample`, `log_prob`, and `sample_and_log_prob` surface. Stochastic
fitting and sampling both accept an explicit `rng`.
