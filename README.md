# sequince-proposals

Fitted proposal distributions and optional normalizing-flow backends for
`sequince`.

The core package provides a scikit-learn-based
`GaussianMixtureProposal`. Neural-flow integrations are optional.

- `sequince.transforms.FlowTransform` for using a flow as a mutation-space transform
- `sequince.proposals.FlowProposal` for using a flow as a proposal distribution

## Installation

```bash
pip install sequince-proposals
```

## Gaussian-mixture proposal

```python
from orng import RandomGenerator
from sequince_proposals import GaussianMixtureProposal

proposal = GaussianMixtureProposal(n_components=4).fit(particles)
samples, log_q = proposal.sample_and_log_prob(
    256,
    rng=RandomGenerator(seed=123),
)
```

Fitting uses scikit-learn's `GaussianMixture`. Inputs may use
NumPy, PyTorch, JAX, or CuPy arrays; samples and densities are restored to the
input backend and device. Full covariance matrices, regularization, and
checkpoint state are supported. Weighted fitting is not currently supported;
uniform post-resampling populations are the intended input.

## Sequince rejuvenation

`sequince-proposals` depends on `sequince`; Sequince does not depend on this
package. Inject the proposal into Sequince's backend-agnostic rejuvenation
step and place it explicitly after resampling:

```python
from sequince.steps import ProposalRejuvenation
from sequince_proposals import GaussianMixtureProposal

rejuvenation = ProposalRejuvenation(
    posterior,
    GaussianMixtureProposal(n_components=8),
)
```

This direct redraw is approximate and deliberately has no Metropolis-Hastings
correction. Sequince fits it after resampling in the standard step order.

## Available flow backends

Install the Zuko integration with `pip install "sequince-proposals[zuko]"`.

```python
from sequince_proposals.zuko import ZukoFlow
```

`ZukoFlow` wraps flows from `zuko.flows` and satisfies the
`sequince.flows.NormalizingFlow` protocol.

## Flow as a mutation transform

Use `FlowTransform` when you want Sequince mutation steps to operate in a
learned latent space.

```python
import numpy as np

from sequince import ParticleState, make_standard_bayesian_smc
from sequince.transforms import AffineTransform, FlowTransform
from sequince_proposals import ZukoFlow

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
    particle_transform=FlowTransform(flow),
)

result = smc.run(initial_state, rng=rng, metadata=metadata)
```

A `FlowTransform` is re-fit on the current particle cloud when the mutation
step prepares its transform, so you normally pass an unfitted flow adapter.

## Flow as a proposal distribution

Use `FlowProposal` when you want a trained flow to provide `log q(x)` for a
bridged target, and optionally to draw proposal samples.

```python
import numpy as np

from sequince import TemperedPosterior
from sequince.proposals import FlowProposal
from sequince_proposals import ZukoFlow

training_particles = np.random.default_rng(123).normal(size=(2000, 2))

proposal = FlowProposal(
    ZukoFlow(
        dims=2,
        flow_class="MAF",
        hidden_features=[64, 64],
    )
).fit(training_particles)

posterior = TemperedPosterior(
    log_prior=lambda x, context=None: -0.5 * np.sum(x**2, axis=1),
    log_likelihood=lambda x, context=None: np.zeros(x.shape[0]),
    log_proposal=proposal,
)

samples, log_q = proposal.sample_and_log_prob(128)
values = posterior.log_prob(samples, context=None)
```

`FlowProposal.sample_and_log_prob()` uses the flow backend's native
implementation when available and otherwise falls back to `sample()` plus
`log_prob()`.
