recursive_diff: Compare two Python data structures
==================================================
.. currentmodule:: recursive_diff

**JSON**, **JSONL**, **YAML** and **MessagePack** are massively popular formats used to
represent nested data. A problem arises when you want to compare two large JSON-like
data structures, because the `==` operator will tell you if the two structures differ
*somewhere*, but won't tell you *where*. Additionally, if the structures contain
floating-point numbers, == won't allow to set a tolerance: 1.00000000000001 is different
from 1.0, which is majorly problematic as floating point arithmetic is naturally
characterised by noise around the 15th decimal position (the size of the
double-precision mantissa). Tests on floating point numbers are typically performed with
:func:`math.isclose` or :func:`numpy.isclose`, which however are not usable if the
numbers to be tested lie deep inside a nested structure.

A second problem that data scientists need to face routinely is comparing huge
NumPy-based data structures, such as :class:`pandas.DataFrame` objects or data loaded
from **Zarr**, **netCDF**, or **HDF5** datastores. Again, it is very frequently needed
to identify *where* differences are, and apply tolerance to the comparison.

This module offers the function :func:`recursive_diff`, which crawls through two
arbitrarily large nested JSON-like structures and dumps out all the differences.
Python-specific data types, such as :class:`set` and :class:`tuple`, are also supported.
`NumPy`_, `Pandas`_, and `Xarray`_ are supported and optimized for speed. Two variant
functions, :func:`display_diffs` and :func:`recursive_eq`, are designed to be used in
Jupyter Notebooks and unit tests respectively.

You can load a whole directory tree of JSON, JSONL, YAML, MessagePack, netCDF, or Zarr
files with one call to :func:`recursive_open` and then pass it to :func:`recursive_diff`
or :func:`recursive_eq` to compare it to another directory tree.

Finally, a :doc:`cli` allows comparing two files in any of the formats above, or two
directory trees full of files, as long as they can be loaded with
:func:`xarray.open_dataset`.

Index
-----

.. toctree::

   quickstart
   installing
   api
   extend
   cli
   develop
   whats-new


Credits
-------
- recursive_diff, recursive_eq and ncdiff were originally developed by
  Legal & General and released to the open source community in 2018.
- All boilerplate is from
  `python_project_template <https://github.com/crusaderky/python_project_template>`_,
  which in turn is from `Xarray`_.

License
-------

This software is available under the open source `Apache License`_.

.. _NumPy: http://www.numpy.org
.. _Pandas: https://pandas.pydata.org
.. _Xarray: https://docs.xarray.dev
.. _Apache License: http://www.apache.org/licenses/LICENSE-2.0.html
