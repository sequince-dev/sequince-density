Gaussian mixture models
=======================

GMM support is provided via
`scikit-learn <https://scikit-learn.org/stable/modules/mixture.html>`_.
Install it with:

.. code-block:: bash

    python -m pip install "sequince-density[gmm]"

.. warning::

    The GMM backend uses NumPy on the CPU. Fitting and density evaluation do not
    run natively on other array backends or GPUs; inputs from other backends
    are converted to NumPy for computation.

Fit a mixture to a sample population, then draw samples and evaluate their log
densities:

.. code-block:: python

    import numpy as np

    from sequince_density import GaussianMixtureProposal

    particles = np.random.default_rng(42).normal(size=(2000, 2))
    proposal = GaussianMixtureProposal(n_components=4).fit(particles, rng=42)
    samples, log_q = proposal.sample_and_log_prob(256, rng=43)

Fitting expects particles shaped ``(n_particles, n_dimensions)``. Nonuniform
weights are not supported; use an equal-weight population, such as particles
after resampling.

``GaussianMixtureProposal`` implements Sequince's proposal interface directly.
See :doc:`proposals` for using it as a fitted starting distribution for SMC.
