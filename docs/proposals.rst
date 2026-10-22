Proposal distributions
======================

Both Gaussian mixtures and normalizing flows can provide a fitted starting
distribution for SMC. ``GaussianMixtureProposal`` implements Sequince's proposal
interface directly. Wrap a flow in ``FlowProposal`` to use the same interface,
including conversion to the SMC particle backend:

.. code-block:: python

    import numpy as np
    from sequince.proposals import FlowProposal

    from sequince_density import ZukoFlow

    particles = np.random.default_rng(42).normal(size=(2000, 2))
    proposal = FlowProposal(
        ZukoFlow(dims=2, flow_class="MAF", hidden_features=[32, 32]),
        fit_kwargs={"n_epochs": 100, "batch_size": 500},
    ).fit(particles, rng=42)

For either fitted proposal type, pass ``proposal`` as ``log_proposal`` to
``make_standard_bayesian_smc`` and draw the initial particles from that same
proposal. SMC then bridges from the proposal to the target posterior.

In an ASPIRE-style analysis, a flow is fitted to samples from an initial
analysis and used as the starting proposal for a new SMC run with an updated
likelihood.

See the `Sequince documentation <https://sequince.readthedocs.io/en/latest/>`_
for the sampler and its proposal interface, and :doc:`examples` for the
examples repository.
