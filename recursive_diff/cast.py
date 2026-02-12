from __future__ import annotations

from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd
import xarray


@singledispatch
def cast(obj: object) -> object:
    """Helper function of :func:`recursive_diff`.

    Cast objects into simpler object types:

    - Cast tuple to list
    - Cast frozenset to set
    - Cast all numpy-based objects to :class:`xarray.DataArray`, as it is the
      most generic format that can describe all use cases:

      - :class:`numpy.ndarray`
      - :class:`pandas.Series`
      - :class:`pandas.DataFrame`
      - :class:`pandas.Index`, except :class:`pandas.RangeIndex`, which is
        instead returned unaltered
      - :class:`xarray.Dataset`

    The data will be potentially wrapped by a dict to hold the various
    attributes and marked so that it doesn't trigger an infinite recursion.

    - Do nothing for any other object types.

    :param obj:
        complex object that must be simplified
    :returns:
        simpler object to compare
    """
    # This is a single dispatch function, defining the default for any
    # classes not explicitly registered below.
    return obj


@cast.register(tuple)
def cast_tuple(obj: tuple) -> list:
    """Single dispatch specialised variant of :func:`cast` for
    :class:`tuple`.

    Cast to a list.
    """
    return list(obj)


@cast.register(frozenset)
def cast_frozenset(obj: frozenset) -> set:
    """Single dispatch specialised variant of :func:`cast` for
    :class:`frozenset`.

    Cast to a set.
    """
    return set(obj)


@cast.register(np.integer)
def cast_npint(obj: np.integer) -> int:
    """Single dispatch specialised variant of :func:`cast` for all numpy scalar
    integers (not to be confused with numpy arrays of integers)
    """
    return int(obj)


@cast.register(np.floating)
def cast_npfloat(obj: np.floating) -> float:
    """Single dispatch specialised variant of :func:`cast` for all numpy scalar
    floats (not to be confused with numpy arrays of floats)
    """
    return float(obj)


@cast.register(np.ndarray)
def cast_nparray(obj: np.ndarray) -> dict[str, Any]:
    """Single dispatch specialised variant of :func:`cast` for
    :class:`numpy.ndarray`.

    Cast to a DataArray with dimensions dim_0, dim_1, ... and
    RangeIndex() as the coords.
    """
    out: dict[str, Any] = {
        "__index_dict": True,
        "data": _strip_dataarray(xarray.DataArray(obj)),
    }
    for i, size in enumerate(obj.shape):
        out[f"dim_{i}"] = pd.RangeIndex(size)
    return out


@cast.register(pd.Series)
def cast_series(obj: pd.Series) -> dict[str, Any]:
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.Series`.

    Cast to a DataArray.
    """
    return {
        "name": obj.name,
        "index": obj.index,
        "data": _strip_dataarray(xarray.DataArray(obj, dims=["index"])),
    }


@cast.register(pd.DataFrame)
def cast_dataframe(obj: pd.DataFrame) -> dict[str, Any]:
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.DataFrame`.

    Cast to a DataArray.

    TODO: proper support for columns with different dtypes. Right now
    they are cast to the closest common type by DataFrame.values.
    """
    return {
        "index": obj.index,
        "columns": obj.columns,
        "data": _strip_dataarray(xarray.DataArray(obj, dims=["index", "column"])),
    }


@cast.register(xarray.DataArray)
def cast_dataarray(obj: xarray.DataArray) -> xarray.DataArray | dict[str, Any]:
    """Single dispatch specialised variant of :func:`cast` for
    :class:`xarray.DataArray`.

    Cast to a simpler DataArray, with separate indices, non-index coords,
    name, and attributes.
    """
    # Prevent infinite recursion - see _strip_dataarray()
    if "__strip_dataarray__" in obj.attrs:
        return obj

    # Strip out the non-index coordinates and attributes
    out: dict[str, Any] = {
        "name": obj.name,
        "attrs": obj.attrs,
        # Index is handled separately, and created as a default
        # RangeIndex(shape[i]) if it doesn't exist
        "index": {k: obj.coords[k].to_index() for k in obj.dims},
        "coords": {
            k: _strip_dataarray(v)
            for k, v in obj.coords.items()
            if not isinstance(v.variable, xarray.IndexVariable)
        },
        "data": _strip_dataarray(obj),
    }
    out["index"]["__index_dict"] = True
    return out


@cast.register(xarray.Dataset)
def cast_dataset(obj: xarray.Dataset) -> dict[str, Any]:
    """Single dispatch specialised variant of :func:`cast` for
    :class:`xarray.Dataset`.

    Cast to a dict of DataArrays.
    """
    out: dict[str, Any] = {
        "attrs": obj.attrs,
        # There may be coords, index or not, that are not
        # used in any data variable.
        # See above on why indices are handled separately
        "index": {k: obj.coords[k].to_index() for k in obj.dims},
        "coords": {
            k: _strip_dataarray(v)
            for k, v in obj.coords.items()
            if not isinstance(v.variable, xarray.IndexVariable)
        },
        "data_vars": {k: _strip_dataarray(v) for k, v in obj.data_vars.items()},
    }
    out["index"]["__index_dict"] = True
    return out


def _strip_dataarray(obj: xarray.DataArray) -> xarray.DataArray:
    """Helper function of :func:`recursive_diff`.

    Return a shallow copy of a :class:`xarray.DataArray` with:

    - no non-index coordinates (including scalar coords)
    - all indices have a coordinate
    - dimensions sorted alphabetically

    :param obj:
        any xarray.DataArray
    :returns:
        a stripped-down shallow copy of obj
    """
    res = obj.copy(deep=False)

    # Remove non-index coordinates
    for k, v in obj.coords.items():
        if not isinstance(v.variable, xarray.IndexVariable):
            del res[k]

    # Add missing index coordinates
    # This is needed to align e.g. two numpy arrays of different sizes
    for k, size in zip(obj.dims, obj.shape):
        if k not in res.coords:
            res.coords[k] = pd.RangeIndex(size)

    # Transpose to ignore dimensions order
    res = res.transpose(*sorted(res.dims, key=str))

    # Prevent infinite recursion - see cast_dataarray()
    res.attrs["__strip_dataarray__"] = True
    return res
