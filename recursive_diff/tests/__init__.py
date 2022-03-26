from __future__ import annotations

import importlib.metadata

import pytest
from packaging.version import parse as parse_version


def _import_or_skip(modname: str, minversion: str | None = None) -> tuple:
    """Build skip markers for an optional module

    :param str modname:
        Name of the optional module
    :param str minversion:
        Minimum required version
    :return:
        Tuple of

        has_module (bool)
            True if the module is available and >= minversion
        requires_module (decorator)
            Tests decorated with it will only run if the module is available
            and >= minversion
    """
    reason = f"requires {modname}"
    if minversion:
        reason += f">={minversion}"

    try:
        version = importlib.metadata.version(modname)
        has = True
    except importlib.metadata.PackageNotFoundError:
        has = False
    if has and minversion and parse_version(version) < parse_version(minversion):
        has = False

    func = pytest.mark.skipif(not has, reason=reason)
    return has, func


has_dask, requires_dask = _import_or_skip("dask")
has_h5netcdf, requires_h5netcdf = _import_or_skip("h5netcdf")
has_netcdf4, requires_netcdf4 = _import_or_skip("netcdf4")
has_scipy, requires_scipy = _import_or_skip("scipy")

has_netcdf = has_h5netcdf or has_netcdf4 or has_scipy
requires_netcdf = pytest.mark.skipif(not has_netcdf, reason="No NetCDF engine found")
