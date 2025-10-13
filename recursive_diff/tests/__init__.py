from __future__ import annotations

import importlib

import numpy as np
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
        requires_module (decorator)
            Tests decorated with it will only run if the module is available
            and >= minversion
    """
    try:
        importlib.import_module(modname)
        has = True
    except ImportError:
        has = False

    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


has_dask, requires_dask = _import_or_skip("dask")
has_h5netcdf, requires_h5netcdf = _import_or_skip("h5netcdf")
has_netcdf4, requires_netcdf4 = _import_or_skip("netcdf4")
has_scipy, requires_scipy = _import_or_skip("scipy")

has_netcdf = has_h5netcdf or has_netcdf4 or has_scipy
requires_netcdf = pytest.mark.skipif(not has_netcdf, reason="No NetCDF engine found")

NP_GE_126 = Version(np.__version__) >= Version("1.26")


def filter_old_numpy_warnings(testfunc):
    if NP_GE_126:
        return testfunc
    return pytest.mark.filterwarnings(
        "ignore:elementwise comparison failed:DeprecationWarning",
        "ignore:elementwise comparison failed:FutureWarning",
    )(testfunc)
