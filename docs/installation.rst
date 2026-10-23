Installation
============

``sequince-density`` requires Python 3.11 or later. Install it from PyPI:

.. code-block:: bash

    python -m pip install sequince-density

Install the optional dependencies for the features you plan to use:

.. list-table::
   :header-rows: 1

   * - Extra
     - Provides
   * - ``gmm``
     - Gaussian-mixture proposals fitted with scikit-learn
   * - ``zuko``
     - Zuko normalizing flows using PyTorch
   * - ``flowjax``
     - FlowJAX normalizing flows using JAX

For example, install Zuko support with:

.. code-block:: bash

    python -m pip install "sequince-density[zuko]"

Extras can be combined:

.. code-block:: bash

    python -m pip install "sequince-density[gmm,zuko,flowjax]"
