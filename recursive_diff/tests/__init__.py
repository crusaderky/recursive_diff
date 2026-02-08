from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest
from packaging.version import Version


def _import_or_skip(modname: str) -> tuple:
    """Build skip markers for an optional module

    :param str modname:
        Name of the optional module
    :return:
        Tuple of

        has_module (bool)
            True if the module is available and >= minversion
        requires_module (pytest mark)
            Tests decorated with it will only run if the module is available
            and >= minversion
    """
    try:
        importlib.import_module(modname)
        has = True
    except ImportError:
        has = False

    mark = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, mark


has_dask, requires_dask = _import_or_skip("dask")
has_h5netcdf, requires_h5netcdf = _import_or_skip("h5netcdf")
has_scipy, requires_scipy = _import_or_skip("scipy")

# Suppress `numpy.ndarry size changed` warning emitted by netCDF4 on import
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    has_netcdf4, requires_netcdf4 = _import_or_skip("netCDF4")

has_netcdf = (has_h5netcdf or has_netcdf4 or has_scipy) and has_dask
requires_netcdf = pytest.mark.skipif(not has_netcdf, reason="No netCDF engine found")

NUMPY_GE_126 = Version(np.__version__) >= Version("1.26")
PANDAS_GE_200 = Version(pd.__version__) >= Version("2.0")


def filter_old_numpy_warnings(testfunc):
    if NUMPY_GE_126:
        return testfunc
    return pytest.mark.filterwarnings(
        "ignore:elementwise comparison failed:DeprecationWarning",
        "ignore:elementwise comparison failed:FutureWarning",
    )(testfunc)
