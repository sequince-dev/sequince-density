sequince-density documentation
==============================

``sequince-density`` provides density estimators for use with the
`Sequince <https://sequince.readthedocs.io/en/latest/>`_ SMC framework,
including fitted proposal distributions and normalizing flows for
preconditioned sampling.

Features
--------

* Normalizing-flow adapters for `Zuko <https://zuko.readthedocs.io/stable/>`_
  (PyTorch) and `FlowJAX <https://danielward27.github.io/flowjax/>`_ (JAX).
* Gaussian-mixture proposals fitted with scikit-learn.
* Integration with Sequince's ``FlowTransform`` for mutation in a learned latent
  space and ``FlowProposal`` for fitted proposal distributions.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation
   normalizing_flows
   gaussian_mixtures
   proposals
   examples
   changelog
   API reference <autoapi/sequince_density/index>
   developer_guide/index

.. toctree::
   :maxdepth: 1
   :caption: Related Packages
   :hidden:

   sequince <https://sequince.readthedocs.io/en/latest/>
   sequince-examples <https://sequince.readthedocs.io/projects/sequince-examples/en/latest/>

Related projects
----------------

* `Sequince repository <https://github.com/sequince-dev/sequince>`_: the main SMC framework.
* `sequince-bilby <https://github.com/sequince-dev/sequince-bilby>`_: an interface
  for using prebuilt Sequince samplers with Bilby.
* `orng <https://github.com/sequince-dev/orng>`_: cross-backend random number
  generation.
* `acai <https://github.com/sequince-dev/acai>`_: Array API compatible control
  flow and iteration.

``sequince-density`` is licensed under the
`MIT License <https://github.com/sequince-dev/sequince-density/blob/main/LICENSE>`_.
