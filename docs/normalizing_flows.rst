Normalizing flows
=================

Install the optional dependencies for your chosen backend:

.. code-block:: bash

    python -m pip install "sequince-density[zuko]"
    # Alternatively, for the JAX backend:
    python -m pip install "sequince-density[flowjax]"

Fit a Zuko flow to a two-dimensional sample population, then draw samples and
evaluate their log densities:

.. code-block:: python

    import numpy as np

    from sequince_density import ZukoFlow

    rng = np.random.default_rng(42)
    particles = rng.normal(size=(2000, 2))

    flow = ZukoFlow(dims=2, flow_class="MAF", hidden_features=[32, 32])
    flow.fit(particles, n_epochs=100, batch_size=500, rng=42)
    samples, log_q = flow.sample_and_log_prob(256, rng=43)

``ZukoFlow`` returns PyTorch tensors; ``FlowJax`` returns JAX arrays.

Particle transform restrictions
-------------------------------

The flow's ``particle_transform`` must not contain a
``PeriodicBoundaryTransform``, including inside nested ``CompositeTransform``
objects. Both adapters reject these transforms on construction, assignment,
fitting, and checkpoint restoration. Periodic wrapping is many-to-one: a
correct density would require summing over periodic images, which these
adapters do not implement. A Jacobian correction alone is insufficient.

Existing checkpoints containing periodic particle transforms are also rejected.
Nonperiodic transforms, such as ``AffineTransform``, remain supported.

Preconditioned sampling
-----------------------

For preconditioned SMC, pass an unfitted flow to ``FlowTransform`` and use the
transform as the ``particle_transform`` argument to
``make_standard_bayesian_smc``. The mutation step fits it to the current particle
population:

.. code-block:: python

    from sequince.transforms import FlowTransform

    transform = FlowTransform(
        ZukoFlow(dims=2, flow_class="MAF", hidden_features=[32, 32]),
        fit_kwargs={"n_epochs": 100, "batch_size": 500},
    )

See :doc:`proposals` to use a flow as a fitted starting distribution instead.
