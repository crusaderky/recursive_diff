.. currentmodule:: recursive_diff

What's New
==========

v1.4.0 (unreleased)
-------------------

- Added **JSON**, **JSONL**, **MessagePack**, **YAML**, and **Zarr** support
- New functions :func:`open` and :func:`recursive_open` for opening files from the
  Python API
- Dropped support for pynio, cfgrib, and pseudonetcdf netCDF engines
- :class:`pandas.Index` diffs are much faster and retain the original order, instead of
  being sorted alphabetically
- :class:`pandas.Index` now compare dtypes
- :class:`pandas.MultiIndex` no longer compare names
- Added support for NumPy and Pandas datetime objects too large for ``M8[ns]``
  (before year 1677 or after 2262)

Breaking CLI changes
^^^^^^^^^^^^^^^^^^^^
- The ``ncdiff`` CLI tool has been deprecated in favor of the new ``recursive-diff``
- The ``recursive-diff`` CLI tool, in addition to netCDF, also supports and
  compares by default JSON, JSONL, MessagePack, YAML, and Zarr files
- The ``recursive-diff`` CLI tool supports multiple wildcard patterns, e.g.::

    recursive-diff -r -m "foo*.nc" "bar*.nc" -- dir1 dir2

  .. note::

     This new feature implies a syntax difference between the legacy ``ncdiff`` CLI tool
     and the new ``recursive-diff``::

       ncdiff -r -m "foo*.nc" dir1 dir2  # valid
       recursive-diff -r -m "foo*.nc" dir1 dir2  # NOT VALID
       recursive-diff -r -m "foo*.nc" "bar*.nc" -- dir1 dir2  # valid (note the --)
       recursive-diff -r dir1 dir2 -m "foo*.nc"  # valid

- The ``recursive-diff`` CLI tool no longer requires Dask to be installed.


v1.3.0 (2025-10-14)
-------------------
- Test against Python 3.13 and 3.14
- Test against recent Pandas versions (tested up to 3.0 beta)
- Detect and handle recursion in data structures (:issue:`24`)
- Fixed warnings in recent Pandas versions (:issue:`27`)
- Bumped up minimum versions for all dependencies:

  ========== ====== ========
  Dependency v1.2.0 v1.3.0
  ========== ====== ========
  python     3.8    3.9
  dask       2.0    2022.7.0
  numpy      1.16   1.23
  pandas     0.25   1.5
  xarray     0.12   2023.8.0
  ========== ====== ========


v1.2.0 (2024-03-16)
-------------------
- Added support for Python 3.11 and 3.12
- Added support for recent Pandas versions (tested up to 2.2)


v1.1.0 (2022-03-26)
-------------------
- Added support for Python 3.8, 3.9, and 3.10
- Type annotations
- Support for pandas 1.0
- This project now adheres to NEP-29; see :ref:`mindeps_policy`.
  Bumped up minimum versions for all dependencies:

  ========== ====== ======
  Dependency v1.0.0 v1.1.0
  ========== ====== ======
  python     3.5.0  3.8
  dask       0.19.0 2.0
  numpy      1.13   1.16
  pandas     0.21   0.25
  xarray     0.10.1 0.12
  ========== ====== ======

- Now using setuptools-scm for versioning
- Migrated CI from Travis + AppVeyor + coveralls to GitHub actions + codecov.io
- Added static code checkers (black, isort, absolufy_imports, flake8, mypy) to CI,
  wrapped by pre-commit


v1.0.0 (2019-01-02)
-------------------

Initial release, split out from xarray-extras v0.3.0.
