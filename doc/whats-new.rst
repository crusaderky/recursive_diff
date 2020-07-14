.. currentmodule:: recursive_diff

What's New
==========

.. _whats-new.1.1.0:

v1.1.0 (Unreleased)
-------------------
- Added explicit support for Python 3.8
- Type annotations
- Support for pandas 1.0
- This project now adheres to NEP-29; see :ref:`mindeps_policy`.
  Bumped up minimum versions for all dependencies:

  ========== ====== ======
  Dependency v1.1.0 v1.2.0
  ========== ====== ======
  python     3.5.0  3.6
  dask       0.19.0 2.0
  numpy      1.13   1.15
  pandas     0.21   0.25
  xarray     0.10.1 0.12
  ========== ====== ======

- Now using setuptools-scm for versioning
- Migrated CI from Travis + AppVeyor + coveralls to GitHub actions + codecov.io
- Added black, isort, flake8 and mypy to CI


.. _whats-new.1.0.0:

v1.0.0 (2019-01-02)
-------------------

Initial release, split out from xarray-extras v0.3.0.
