Installation
============

Required dependencies
---------------------

- `xarray <http://xarray.pydata.org/>`__
- For :doc:`ncdiff`: one or more NetCDF engines;
  see :func:`xarray.open_dataset`


Installing with conda
---------------------

.. code-block:: bash

    conda install recursive_diff


Installing with pip
-------------------

.. code-block:: bash

    pip install recursive-diff


.. _mindeps_policy:

Minimum dependency versions
---------------------------

This project adopts a rolling policy based on `SPEC 0
<https://scientific-python.org/specs/spec-0000/>`_ regarding the minimum
supported version of its dependencies.

You can see the actual minimum tested versions in `pyproject.toml
<https://github.com/crusaderky/recursive_diff/blob/main/pyproject.toml>`_.
