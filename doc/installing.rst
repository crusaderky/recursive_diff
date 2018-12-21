.. _installing:

Installation
============

Required dependencies
---------------------

- Python 3.5 or later
- `xarray <http://xarray.pydata.org/>`__
- For :doc:`ncdiff`: one or more NetCDF engines;
  see :func:`xarray.open_dataset`

Deployment
----------

- With pip: :command:`pip install recursive_diff`
- With `anaconda <https://www.anaconda.com/>`_:
  :command:`conda install -c conda-forge recursive_diff`

Testing
-------

To run the test suite after installing recursive_diff, first install (via pypi or conda)

- `py.test <https://pytest.org>`__: Simple unit testing library

and run
``py.test --pyargs recursive_diff``.

