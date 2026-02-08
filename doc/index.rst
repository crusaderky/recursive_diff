recursive_diff: Compare two Python data structures
==================================================
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

This module offers the function :func:`~recursive_diff.recursive_diff`, which crawls
through two arbitrarily large nested JSON-like structures and dumps out all the
differences. Python-specific data types, such as :class:`set` and :class:`tuple`, are
also supported. `NumPy`_, `Pandas`_, and `Xarray`_ are supported and optimized for
speed. A variant function, :func:`~recursive_diff.recursive_eq`, is designed to be used
in unit tests.

You can load a whole directory tree of JSON, JSONL, YAML, MessagePack, netCDF, or Zarr
files with one call to :func:`~recursive_diff.recursive_open` and then pass it to
:func:`~recursive_diff.recursive_diff` or :func:`~recursive_diff.recursive_eq` to
compare it to another directory tree.

Finally, a :doc:`cli` allows comparing two files in any of the formats above, or two
directory trees full of files, as long as they can be loaded with
:func:`xarray.open_dataset`.

Examples
--------

.. code::

    from recursive_diff import recursive_diff

    lhs = {
        'foo': [1, 2, ('one', 5.2), 4],
        'only_lhs': 1
    }
    rhs = {
        'foo': [1, 2, ['two', 5.200001, 3]],
        'only_rhs': 1
    }

    for diff in recursive_diff(lhs, rhs, abs_tol=.1):
        print(diff)

Output::

    Pair only_lhs:1 is in LHS only
    Pair only_rhs:1 is in RHS only
    [foo]: LHS has 1 more elements than RHS: [4]
    [foo][2]: object type differs: tuple != list
    [foo][2]: RHS has 1 more elements than LHS: [3]
    [foo][2][0]: one != two


Or as a unit test:

.. code::

    from recursive_diff import recursive_eq

    def test1():
        recursive_eq(lhs, rhs, abs_tol=.1)

py.test output::

    ==================== FAILURES ===================
    E       AssertionError: 6 differences found

    -------------- Captured stdout call --------------

    Pair only_lhs:1 is in LHS only
    Pair only_rhs:1 is in RHS only
    [foo]: LHS has 1 more elements than RHS: [4]
    [foo][2]: object type differs: tuple != list
    [foo][2]: RHS has 1 more elements than LHS: [3]
    [foo][2][0]: one != two


Compare two nested directory trees that contain ``.json``, ``.jsonl``, ``.yaml``,
``.msgpack``, ``.nc``, or ``.zarr`` files:

.. code::

    from recursive_diff import recursive_open, recursive_eq

    lhs = recursive_open("baseline")
    rhs = recursive_open("new_output")
    recursive_eq(lhs, rhs)


Same as above, but from the command line::

    $ recursive-diff -r baseline new_output

Index
-----

.. toctree::

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
