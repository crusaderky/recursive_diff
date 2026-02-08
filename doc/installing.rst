Installation
============

Required dependencies
---------------------

- `Xarray <http://xarray.pydata.org/>`__


Optional dependencies
---------------------

recursive-diff supports Xarray objects backed by `Dask <https://dask.org/>`__.

Dependencies needed for the :doc:`cli`:

- `Dask <https://dask.org/>`__
- To read NetCDF v3: `netCDF4 <https://unidata.github.io/netcdf4-python/>`__
  or `scipy <https://www.scipy.org/>`__
- To read NetCDF v4: `netCDF4 <https://unidata.github.io/netcdf4-python/>`__
  or `h5netcdf <https://h5netcdf.org/>`__


Installing with conda
---------------------

.. code-block:: bash

    conda install -c conda-forge recursive_diff


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
