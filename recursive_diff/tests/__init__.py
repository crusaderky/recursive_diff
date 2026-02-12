from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray
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
has_zarr, requires_zarr = _import_or_skip("zarr")

# Suppress `numpy.ndarry size changed` warning emitted by netCDF4 on import
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    has_netcdf4, requires_netcdf4 = _import_or_skip("netCDF4")

has_netcdf = has_h5netcdf or has_netcdf4 or has_scipy
requires_netcdf = pytest.mark.skipif(not has_netcdf, reason="No netCDF engine found")

NUMPY_GE_126 = Version(np.__version__) >= Version("1.26")
PANDAS_GE_200 = Version(pd.__version__) >= Version("2.0")
PANDAS_GE_300 = Version(pd.__version__) >= Version("3.0")

XARRAY_GE_2024_9_1 = Version(xarray.__version__) >= Version("2024.9.1")
XARRAY_GE_2025_1_2 = Version(xarray.__version__) >= Version("2025.1.2")
if XARRAY_GE_2024_9_1:
    TO_ZARR_V2 = {"zarr_format": 2}
    TO_ZARR_V3 = {"zarr_format": 3}
    HAS_ZARR_V3 = True
else:
    TO_ZARR_V2 = {"zarr_version": 2}
    TO_ZARR_V3 = {}
    HAS_ZARR_V3 = False


def filter_old_numpy_warnings(testfunc):
    if NUMPY_GE_126:
        return testfunc
    return pytest.mark.filterwarnings(
        "ignore:elementwise comparison failed:DeprecationWarning",
        "ignore:elementwise comparison failed:FutureWarning",
    )(testfunc)
