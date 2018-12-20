recursive_diff: Compare two Python data structures
==================================================
JSON and YAML are two massively popular formats to represent nested data.
A problem arises when you want to compare two large JSON data structures,
because the `==` operator will tell you if the two structures differ somewhere,
but won't tell you *where*. Additionally, if the structures contain
floating-point numbers, it won't allow to set a tolerance: 1.00000000000001 is
different from 1.0, which is majorly problematic as floating point arithmetics
are naturally characterised by noise around the 15th decimal position (the
size of the mantissa).

A second problem that data scientists are well accustomed to is comparing
huge numpy-based data structures, such as :class:`pandas.DataFrame` objects
or data loaded from HDF5 datastores.
Again, it is needed to identify *where* differences are, and apply tolerance
to the comparison

This module offers the function :func:`~recursive_diff.recursive_diff`,
which crawls through two arbitrarily large nested JSON-like structures and
dumps out all the differences. Python-specific data types, such as
:class:`set` and :class:`tuple`, are also supported.
`numpy <http://www.numpy.org>`_, `pandas <https://pandas.pydata.org>`_, and
`xarray <http://xarray.pydata.org>`_ are supported and optimized for speed.

Another function, :func:`~recursive_diff.recursive_eq`, is designed to be used
in unit tests.

Finally, the command-line tool :doc:`ncdiff` allows comparing two NetCDF files,
or two directories full of NetCDF files, as long as they can be loaded with
:func:`xarray.open_dataset`.

Index
-----

.. toctree::

   installing
   whats-new
   api
   ncdiff


Credits
-------
- recursive_diff, recursive_eq and ncdiff were originally developed by
  Legal & General and released to the open source community in 2018.
- All boilerplate is from
  `python_project_template <https://github.com/crusaderky/python_project_template>`_,
  which in turn is from `xarray <http://xarray.pydata.org/>`_.

License
-------

recursive_diff is available under the open source `Apache License`__.

__ http://www.apache.org/licenses/LICENSE-2.0.html