.. currentmodule:: recursive_diff

What's New
==========

.. _whats-new.1.2.0:

v1.2.0 (2024-03-16)
-------------------
- Added support for Python 3.11 and 3.12
- Added support for recent Pandas versions (tested up to 2.2)


.. _whats-new.1.1.0:

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


.. _whats-new.1.0.0:

v1.0.0 (2019-01-02)
-------------------

Initial release, split out from xarray-extras v0.3.0.
