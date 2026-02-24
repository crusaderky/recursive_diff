"""Recursively compare Python objects.

See also its most commonly used wrapper:
:func:`~recursive_diff.testing.recursive_eq`
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Generator, Hashable
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray

from recursive_diff.cast import MissingKeys, cast
from recursive_diff.dask_compat import Delayed

PANDAS_GE_200 = int(pd.__version__.split(".")[0]) >= 2


def are_instances(lhs: object, rhs: object, cls: type | tuple[type, ...]) -> bool:
    """Return True if both lhs and rhs are instances of cls; False otherwise"""
    return isinstance(lhs, cls) and isinstance(rhs, cls)


def is_array(dtype: str) -> bool:
    return any(
        dtype.startswith(t) for t in ("ndarray", "DataArray", "Series", "DataFrame")
    )


def is_array_like(dtype: str) -> bool:
    return dtype in {"int", "float", "complex", "bool", "str", "list", "tuple"}


def is_basic_noncontainer(x: object) -> bool:
    # Beware: np.float64 is a subclass of float;
    # np.complex128 is a subclass of complex.
    return type(x) in {bool, int, float, complex, type(None), str, bytes}


DO_NOT_CAST_TYPES = {bool, int, float, complex, str, bytes, list, dict, set, type(None)}


def recursive_diff(
    lhs: Any,
    rhs: Any,
    *,
    rel_tol: float = 1e-09,
    abs_tol: float = 0.0,
    brief_dims: Collection[Hashable] | Literal["all"] = (),
) -> Generator[str]:
    """Compare two objects and yield all differences.
    The two objects must any of:

    - basic types (int, float, complex, bool, str, bytes)
    - basic collections (list, tuple, dict, set, frozenset)
    - numpy scalar types
    - :class:`numpy.ndarray`
    - :class:`pandas.Series`
    - :class:`pandas.DataFrame`
    - :class:`pandas.Index`
    - :class:`xarray.DataArray`
    - :class:`xarray.Dataset`
    - :class:`dask.delayed.Delayed`
    - any recursive combination of the above
    - any other object (compared with ==)

    Special treatment is reserved to different types:

    - floats and ints are compared with tolerance, using :func:`math.isclose`
    - complex numbers are compared with tolerance, using :func:`math.isclose`
      separately on the real and imaginary parts
    - NaN equals to NaN
    - floats without decimals compare as equal to ints
    - complex numbers without imaginary part DO NOT compare as equal to floats,
      as they have substantially different behaviour
    - bools are only equal to other bools
    - numpy arrays are compared elementwise and with tolerance,
      also testing the dtype, using :func:`numpy.isclose(lhs, rhs) <numpy.isclose>`
      for numeric arrays and equality for other dtypes.
    - pandas and Xarray objects are compared elementwise, with tolerance, and
      without order. Duplicate indices are not supported.
    - Xarray dimensions and variables are compared without order
    - collections (list, tuple, dict, set, frozenset) are recursively
      descended into
    - generic/unknown objects are compared with ==

    Custom classes can be registered to benefit from the above behaviour;
    see :func:`cast`.

    :param lhs:
        left-hand-side data structure
    :param rhs:
        right-hand-side data structure
    :param float rel_tol:
        relative tolerance when comparing numbers.
        Applies to floats, integers, and all numpy-based data.
    :param float abs_tol:
        absolute tolerance when comparing numbers.
        Applies to floats, integers, and all numpy-based data.
    :param brief_dims:
        One of:

        - collection of strings representing Xarray dimensions. If one or more
          differences are found along one of these dimensions, only one message
          will be reported, stating the differences count.
        - "all", to produce one line only for every Xarray variable that
          differs

        Omit to output a line for every single different cell.

    Yields strings containing difference messages, prepended by the path to
    the point that differs.
    """
    # For as long as we don't encounter any Delayed or dask-backed xarray objects in lhs
    # or rhs, yield diff messages directly from the recursive generator, without
    # accumulating them. This allows to start printing differences as soon as they are
    # found, without waiting for the whole recursion to finish. Once we encounter a
    # Delayed or dask-backed xarray object, we start accumulating all eager messages and
    # Delayed[list[str]] in a list and compute all the delayeds at once.
    diffs: list[str | Delayed] = []
    for diff in _recursive_diff(
        lhs,
        rhs,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        brief_dims=brief_dims,
        path=[],
        seen_lhs={},
        seen_rhs={},
    ):
        if diffs or isinstance(diff, Delayed):
            diffs.append(diff)
        else:
            yield diff

    if not diffs:
        return

    import dask

    # Override default scheduler for Delayed, which would be multiprocessing.
    # We use Delayed to post-process the comparison of Dask-backed Xarrays.

    # This has the disadvantage that JSON, JSONL, YAML, and MessagePack files
    # comparison contends for the GIL. Eventually this problem will go away once
    # free-threading becomes the norm.
    scheduler = dask.config.get("scheduler", "threads")
    computed_diff: str | list[str]
    (computed_diffs,) = dask.compute(diffs, scheduler=scheduler)
    for computed_diff in computed_diffs:
        if isinstance(computed_diff, list):
            # From a Delayed
            yield from computed_diff
        else:
            yield computed_diff


def _recursive_diff(
    lhs: Any,
    rhs: Any,
    *,
    rel_tol: float,
    abs_tol: float,
    brief_dims: Collection[Hashable] | Literal["all"],
    path: list[object],
    seen_lhs: dict[int, int],
    seen_rhs: dict[int, int],
) -> Generator[str | Delayed]:  # yields str | Delayed[list[str]]
    """Recursive implementation of :func:`recursive_diff`

    :param list path:
        list of nodes traversed so far, to be prepended to all error messages
    param dict[int, int] seen_lhs:
        map of {id(): path index} of all lhs objects traversed so far, to detect cycles
    param dict[int, int] seen_rhs:
        map of {id(): path index} of all rhs objects traversed so far, to detect cycles

    This function calls itself recursively for all elements of numpy-based
    data, list, tuple, and dict.values(). Every time, it appends to the
    path list one element.
    """

    # Fast path
    if lhs is rhs:
        return

    lhs_is_basic_noncontainer = is_basic_noncontainer(lhs)
    rhs_is_basic_noncontainer = is_basic_noncontainer(rhs)
    if (
        lhs_is_basic_noncontainer
        and rhs_is_basic_noncontainer
        and lhs == rhs
        # Do not compare complex vs. float and int vs. bool as equal
        and type(lhs) is type(rhs)
    ):
        return

    def diff(msg: str, print_path: list[object] = path) -> str:
        """Format diff message, prepending the formatted path"""
        path_prefix = "".join(f"[{elem}]" for elem in print_path)
        if path_prefix != "":
            path_prefix += ": "
        return path_prefix + msg

    # Detect recursion
    recursive_lhs = seen_lhs.get(id(lhs), -1)
    recursive_rhs = seen_rhs.get(id(rhs), -1)

    if recursive_lhs >= 0 or recursive_rhs >= 0:
        if recursive_lhs != recursive_rhs:
            if recursive_lhs == -1:
                msg_lhs = "is not recursive"
            else:
                msg_lhs = f"recurses to {path[: recursive_lhs + 1]}"
            if recursive_rhs == -1:
                msg_rhs = "is not recursive"
            else:
                msg_rhs = f"recurses to {path[: recursive_rhs + 1]}"
            yield diff(f"LHS {msg_lhs}; RHS {msg_rhs}")
        return

    # Don't add potentially internalized objects
    if not lhs_is_basic_noncontainer:
        seen_lhs = {**seen_lhs, id(lhs): len(path)}
    if not rhs_is_basic_noncontainer:
        seen_rhs = {**seen_rhs, id(rhs): len(path)}
    # End of recursion detection

    if isinstance(lhs, Delayed) or isinstance(rhs, Delayed):
        from dask import delayed

        @delayed
        def _recursive_diff_d(*args, **kwargs):  # type: ignore[no-untyped-def]
            return list(_recursive_diff(*args, **kwargs))

        yield _recursive_diff_d(
            lhs,
            rhs,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            brief_dims=brief_dims,
            path=path,
            seen_lhs={},
            seen_rhs={},
        )
        return

    # Build string representation of the two variables *before* casting
    lhs_repr = _str_trunc(lhs)
    rhs_repr = _str_trunc(rhs)

    # cast lhs and rhs to simpler data types; pretty-print data type
    dtype_lhs = _dtype_str(lhs)
    dtype_rhs = _dtype_str(rhs)
    # fast path: Skip single dispatch for basic types
    if type(lhs) not in DO_NOT_CAST_TYPES:
        lhs = cast(lhs)
    if type(rhs) not in DO_NOT_CAST_TYPES:
        rhs = cast(rhs)

    # 1.0 vs. 1 must not be treated as a difference
    if isinstance(lhs, int) and isinstance(rhs, float):
        dtype_lhs = "float"
    elif isinstance(rhs, int) and isinstance(lhs, float):
        dtype_rhs = "float"

    # When comparing an array vs. a plain python list or scalar, log an error
    # for the different dtype and then proceed to compare the contents
    if is_array(dtype_lhs) and is_array_like(dtype_rhs):
        rhs = cast(np.array(rhs))
    elif is_array(dtype_rhs) and is_array_like(dtype_lhs):
        lhs = cast(np.array(lhs))

    if dtype_lhs != dtype_rhs and not (
        are_instances(lhs, rhs, xarray.DataArray)
        and "__strip_dataarray__" in lhs.attrs
        and "__strip_dataarray__" in rhs.attrs
    ):
        yield diff(f"object type differs: {dtype_lhs} != {dtype_rhs}")

    # Continue even in case dtype doesn't match
    # This allows comparing e.g. a numpy array vs. a list or a tuple

    if are_instances(lhs, rhs, list):
        if len(lhs) > len(rhs):
            yield diff(
                f"LHS has {len(lhs) - len(rhs)} more elements than RHS: "
                + _str_trunc(lhs[len(rhs) :])
            )
        elif len(lhs) < len(rhs):
            yield diff(
                f"RHS has {len(rhs) - len(lhs)} more elements than LHS: "
                + _str_trunc(rhs[len(lhs) :])
            )
        for i, (lhs_i, rhs_i) in enumerate(zip(lhs, rhs)):
            yield from _recursive_diff(
                lhs_i,
                rhs_i,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                brief_dims=brief_dims,
                path=[*path, i],
                seen_lhs=seen_lhs,
                seen_rhs=seen_rhs,
            )

    elif are_instances(lhs, rhs, set):
        for x in sorted(lhs - rhs, key=repr):
            yield diff(f"{_str_trunc(x)} is in LHS only")
        for x in sorted(rhs - lhs, key=repr):
            yield diff(f"{_str_trunc(x)} is in RHS only")

    elif are_instances(lhs, rhs, pd.RangeIndex):
        # Pretty-print differences in size. This is used not only by
        # pd.Series and pd.DataFrame, but also by numpy arrays
        # and xarrays without coords
        if (
            lhs.start == rhs.start == 0
            and lhs.step == rhs.step == 1
            and lhs.name == rhs.name
        ):
            delta = rhs.stop - lhs.stop
            if delta < 0:
                yield diff(f"LHS has {-delta} more elements than RHS")
            elif delta > 0:
                yield diff(f"RHS has {delta} more elements than LHS")
        else:
            # General case
            # e.g. RangeIndex(start=1, stop=3, step=1, name='x')
            lhs, rhs = str(lhs), str(rhs)
            if lhs != rhs:
                yield diff(f"{lhs} != {rhs}")

    elif are_instances(lhs, rhs, pd.Index):
        # Note: this also covers RangeIndex vs. other types of Index
        if not lhs.equals(rhs):
            # Remove everything but the differences.
            # This can fail e.g. for int vs. object
            try:
                lhs_only = lhs[~lhs.isin(rhs)]
                rhs_only = rhs[~rhs.isin(lhs)]
            except ValueError:
                lhs_set = set(lhs)
                rhs_set = set(rhs)
                lhs_only = [x for x in lhs if x not in rhs_set]
                rhs_only = [x for x in rhs if x not in lhs_set]

            for x in lhs_only:
                yield diff(f"{_str_trunc(x)} is in LHS only")
            for x in rhs_only:
                yield diff(f"{_str_trunc(x)} is in RHS only")

    elif are_instances(lhs, rhs, dict):
        # Was this dict generated by cast() as a dict of indices?
        missing_keys = lhs.pop(MissingKeys, MissingKeys.PAIR)
        assert rhs.pop(MissingKeys, MissingKeys.PAIR) is missing_keys
        assert missing_keys in MissingKeys

        for key in sorted(lhs.keys() - rhs.keys(), key=repr):
            if missing_keys is MissingKeys.DIMENSION:
                yield diff(f"Dimension {key} is in LHS only")
            elif missing_keys is MissingKeys.PAIR:
                yield diff(f"Pair {key}:{_str_trunc(lhs[key])} is in LHS only")
        for key in sorted(rhs.keys() - lhs.keys(), key=repr):
            if missing_keys is MissingKeys.DIMENSION:
                yield diff(f"Dimension {key} is in RHS only")
            elif missing_keys is MissingKeys.PAIR:
                yield diff(f"Pair {key}:{_str_trunc(rhs[key])} is in RHS only")
        for key in sorted(lhs.keys() & rhs.keys(), key=repr):
            yield from _recursive_diff(
                lhs[key],
                rhs[key],
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                brief_dims=brief_dims,
                path=[*path, key],
                seen_lhs=seen_lhs,
                seen_rhs=seen_rhs,
            )

    elif are_instances(lhs, rhs, bool):
        if lhs != rhs:
            yield diff(f"{lhs} != {rhs}")
    elif are_instances(lhs, rhs, str):  # noqa: SIM114
        if lhs != rhs:
            yield diff(f"{lhs_repr} != {rhs_repr}")
    elif are_instances(lhs, rhs, bytes):
        if lhs != rhs:
            yield diff(f"{lhs_repr} != {rhs_repr}")
    elif are_instances(lhs, rhs, (int, float)):
        if (
            lhs != rhs  # Fast path
            and not math.isclose(lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol)
            and not (math.isnan(lhs) and math.isnan(rhs))
        ):
            abs_delta = rhs - lhs
            rel_delta = math.nan if lhs == 0 else rhs / lhs - 1
            yield diff(f"{lhs} != {rhs} (abs: {abs_delta:.1e}, rel: {rel_delta:.1e})")
    elif are_instances(lhs, rhs, (int, float, complex)):
        if (
            lhs != rhs  # Fast path
            and (
                (
                    not math.isclose(
                        lhs.real, rhs.real, rel_tol=rel_tol, abs_tol=abs_tol
                    )
                    and not (math.isnan(lhs.real) and math.isnan(rhs.real))
                )
                or (
                    not math.isclose(
                        lhs.imag, rhs.imag, rel_tol=rel_tol, abs_tol=abs_tol
                    )
                    and not (math.isnan(lhs.imag) and math.isnan(rhs.imag))
                )
            )
        ):
            # Print tweak for inf vs. inf
            abs_delta = complex(
                rhs.real - lhs.real if rhs.real != lhs.real else 0,
                rhs.imag - lhs.imag if rhs.imag != lhs.imag else 0,
            )
            try:
                rel_delta = complex(
                    rhs.real / lhs.real - 1 if rhs.real != lhs.real else 0,
                    rhs.imag / lhs.imag - 1 if rhs.imag != lhs.imag else 0,
                )
            except ZeroDivisionError:
                rel_delta = math.nan
            yield diff(f"{lhs} != {rhs} (abs: {abs_delta:.1e}, rel: {rel_delta:.1e})")

    elif are_instances(lhs, rhs, xarray.DataArray):
        yield from _diff_dataarrays(
            lhs,
            rhs,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            brief_dims=brief_dims,
            path=path,
        )

    else:
        # unknown objects
        try:
            if lhs != rhs:
                yield diff(f"{lhs_repr} != {rhs_repr}")
        except Exception:
            # e.g. bool(xarray.DataArray([1, 2]) == {1: 2}) will raise:
            #   ValueError: The truth value of an array with more than one
            #   element is ambiguous. Use a.any() or a.all()
            # Note special case of comparing an array vs. a list is handled
            # above in this function.
            # Custom classes which implement a duck-typed __eq__ will
            # possibly fail with AttributeError, IndexError, etc. instead.
            yield diff(f"Cannot compare objects: {lhs_repr}, {rhs_repr}")


def _str_trunc(x: object) -> str:
    """Helper function of :func:`recursive_diff`.

    Convert x to string. If it is longer than 80 characters, or spans
    multiple lines, truncate it
    """
    x = str(x)
    if len(x) <= 80 and "\n" not in x:
        return x
    return x.splitlines()[0][:76] + " ..."


def _diff_dataarrays(
    lhs: xarray.DataArray,
    rhs: xarray.DataArray,
    rel_tol: float,
    abs_tol: float,
    brief_dims: Collection[Hashable] | Literal["all"],
    path: list[object],
) -> Generator[str | Delayed]:  # str | Delayed[list[str]]
    """
    Compare two DataArrays, previously prepared by _strip_dataarray.

    This function is executed for all data that was originally:

    - numpy.ndarray
    - pandas.Series
    - pandas.DataFrame
    - xarray.DataArray
    - xarray.Dataset
    - any of the above, compared against a plain Python list
    """
    # Note: lhs.dims and rhs.dims were aligned by _strip_dataarray
    if lhs.dims != rhs.dims:
        # This is already reported elsewhere when comparing dicts
        # (Dimension x is in LHS only)
        return

    if not lhs.dims:
        # 0-dimensional arrays
        yield from _recursive_diff(
            _array0d_to_scalar(lhs),
            _array0d_to_scalar(rhs),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            brief_dims=(),
            path=path,
            seen_lhs={},
            seen_rhs={},
        )
        return

    # Align to guarantee that the index is identical on both sides.
    # Change the order as needed.
    lhs, rhs = xarray.align(lhs, rhs, join="inner")

    if lhs.chunks is not None or rhs.chunks is not None:
        import dask.array as da

        if lhs.chunks is None:
            # Fix for old versions of Dask, where
            # np.ndarray == da.Array returned a np.ndarray
            lhs = lhs.chunk(rhs.chunks)

        is_dask = True
        isclose = da.isclose
    else:
        is_dask = False
        isclose = np.isclose

    # Generate a bit-mask of the differences
    # For Dask-backed arrays, this operation is delayed.
    if lhs.dtype.kind in "iufc" and rhs.dtype.kind in "iufc":
        # Both arrays are numeric
        mask = ~isclose(lhs.data, rhs.data, rtol=rel_tol, atol=abs_tol, equal_nan=True)
    elif lhs.dtype.kind == "M" and rhs.dtype.kind == "M":
        # Both arrays are datetime64
        # Unlike with np.isclose(equal_nan=True), there is no
        # straightforward way to do a comparison of dates where
        # NaT == NaT returns True.
        mask = (lhs.data != rhs.data) & ~(np.isnat(lhs.data) & np.isnat(rhs.data))
    else:
        # At least one between lhs and rhs is non-numeric,
        # e.g. bool or str
        mask = lhs.data != rhs.data

        # NumPy <1.26:
        # `lhs != rhs` between two non-scalar, incomparable types
        # (like strings and numbers) returns True
        # FutureWarning: elementwise comparison failed; returning scalar
        # instead, but in the future will perform elementwise comparison
        if mask is True:
            mask = np.ones(lhs.shape, dtype=bool)

    if brief_dims == "all":
        brief_axes = set(range(lhs.ndim))
    else:
        brief_dims = set(brief_dims)
        brief_axes = {axis for axis, dim in enumerate(lhs.dims) if dim in brief_dims}
    del brief_dims

    full_coords = {
        dim: lhs.coords[dim]
        for axis, dim in enumerate(lhs.dims)
        if axis not in brief_axes
    }

    # Generate:
    # If brief_dims and lhs.dims are fully disjoint:
    #   - 1D NumPy array of only the elements in lhs that differ from rhs
    #   - 1D NumPy array of only the elements in rhs that differ from lhs
    # If at least one dimension is in brief_dims:
    #   - 1D NumPy array of the count of the differences along all brief_dims,
    #     with size equal to the flattened non-brief dims.
    # Plus:
    # - one 1D NumPy array of the indices of the elements that differ, one per
    #   non-brief dim, with potentially repeated indices
    # All of the arrays will have the same size, which is the number of differences.
    # For Dask-backed arrays, this whole operation is delayed.

    if brief_axes:
        diffs_count = mask.astype(int).sum(axis=tuple(brief_axes))
        mask = diffs_count > 0
        if mask.ndim:
            diffs_count = diffs_count[mask]
    else:
        diffs_lhs = lhs.data[mask]
        diffs_rhs = rhs.data[mask]

    diffs_idx = []
    for axis, size in enumerate(mask.shape):
        idx_shape = (1,) * axis + (-1,) + (1,) * (mask.ndim - axis - 1)
        if is_dask:
            import dask.array as da

            assert isinstance(mask, da.Array)
            idx = da.arange(size, chunks=mask.chunks[axis])
            idx = idx.reshape(idx_shape)
            idx = da.broadcast_to(idx, mask.shape, chunks=mask.chunks)
        else:
            idx = np.arange(size)
            idx = idx.reshape(idx_shape)
            idx = np.broadcast_to(idx, mask.shape)

        idx = idx[mask]
        diffs_idx.append(idx)

    msg_prefix = "".join(f"[{elem}]" for elem in path)

    args: tuple
    if brief_axes:
        pp_func = _diff_dataarrays_print_brief
        args = (diffs_count, diffs_idx, full_coords, msg_prefix)  # type: ignore[possibly-undefined]
    else:
        pp_func = _diff_dataarrays_print_full  # type: ignore[assignment]
        args = (diffs_lhs, diffs_rhs, diffs_idx, full_coords, msg_prefix)  # type: ignore[possibly-undefined]

    if is_dask:
        from dask import delayed

        yield delayed(pp_func)(*args)

    else:
        yield from pp_func(*args)


def _diff_dataarrays_print_full(
    diffs_lhs: np.ndarray,
    diffs_rhs: np.ndarray,
    diffs_idx: list[np.ndarray],
    coords: dict[Hashable, xarray.DataArray],
    msg_prefix: str,
) -> list[str]:
    """Final step of diffing two DataArrays, for when brief_dims is disjoint from
    the array's dims.

    The data has all been loaded to disk and compared by _diff_dataarrays.
    If there was a Dask backend, the Dask arrays have been computed by delayed() into
    NumPy arrays.

    All arrays are one-dimensional aligned and same-sized np.ndarray objects, one point
    per diff to be returned.

    :param diffs_lhs:
        Array of the values of the elements that differ in lhs from rhs.
    :param diffs_rhs:
        Array of the values of the elements that differ in rhs from lhs.
    :param diffs_idx:
        List of arrays of the indices of the elements that differ, one array per
        non-brief dimension, with potentially repeated indices.
    :param coords:
        dict of {dim: coord} of the non-brief indices, in the same order as diffs_idx
    :param msg_prefix:
        string to prepend to all messages
    """
    assert isinstance(diffs_lhs, np.ndarray)
    assert isinstance(diffs_rhs, np.ndarray)
    ndiff = diffs_lhs.size
    assert diffs_rhs.size == ndiff
    if not ndiff:
        return []

    if diffs_lhs.dtype.kind in "iufc" and diffs_rhs.dtype.kind in "iufc":
        with_tol = True
        diffs_abs = diffs_rhs - diffs_lhs
        with np.errstate(divide="ignore", invalid="ignore"):
            diffs_rel = diffs_rhs / diffs_lhs - 1
        # Replace inf with nan, in alignment with floats comparison
        diffs_rel = np.where(diffs_lhs == 0, np.nan, diffs_rel)
    else:
        with_tol = False
        diffs_abs = range(ndiff)  # dummy
        diffs_rel = range(ndiff)  # dummy

    assert len(diffs_idx) == len(coords)
    for idx in diffs_idx:
        assert idx.size == ndiff
    diffs_coords_iters = [
        coord[idx].values for coord, idx in zip(coords.values(), diffs_idx)
    ]
    dim_tags = [
        # Prettier output when there was no coord at the beginning,
        # e.g. with plain numpy arrays
        re.sub(r"^dim_\d=$", "", f"{dim}=")
        for dim in coords
    ]

    # Can't use a generator as this function may be delayed
    diff_msgs = []

    for lhs_value, rhs_value, abs_value, rel_value, *diff_coords in zip(
        diffs_lhs,
        diffs_rhs,
        diffs_abs,
        diffs_rel,
        *diffs_coords_iters,
    ):
        addr = ", ".join(f"{dim_tag}{i}" for dim_tag, i in zip(dim_tags, diff_coords))
        msg = f"{msg_prefix}[{addr}]: {lhs_value} != {rhs_value}"
        if with_tol:
            msg += f" (abs: {abs_value:.1e}, rel: {rel_value:.1e})"
        diff_msgs.append(msg)

    return diff_msgs


def _diff_dataarrays_print_brief(
    diffs_count: np.ndarray,
    diffs_idx: list[np.ndarray],
    coords: dict[Hashable, xarray.DataArray],
    msg_prefix: str,
) -> list[str]:
    """Variant of _diff_dataarrays_print_brief for when there is at least one brief dim.

    The data has all been loaded to disk and compared by _diff_dataarrays.
    If there was a Dask backend, the Dask arrays have been computed by delayed() into
    NumPy arrays.

    All arrays are one-dimensional aligned and same-sized np.ndarray objects, one point
    per diff to be returned.

    :param diffs_count:
        Array of the count of differences along all brief_dims, one point per non-brief
        dim, flattened.
    :param diffs_idx:
        List of arrays of the indices of the elements that differ, one array per
        non-brief dimension, with potentially repeated indices.
    :param coords:
        dict of {dim: coord} of the non-brief indices, in the same order as diffs_idx
    :param msg_prefix:
        string to prepend to all messages
    """
    ndiff = diffs_count.size

    for idx in diffs_idx:
        assert idx.size == ndiff

    if not diffs_count.ndim:
        # Scalar outcome of brief_dims collapsing all dimensions
        if diffs_count:
            return [f"{msg_prefix}: {diffs_count} differences"]
        return []

    diffs_coords = [coord[idx].values for coord, idx in zip(coords.values(), diffs_idx)]
    dim_tags = [
        # Prettier output when there was no coord at the beginning,
        # e.g. with plain numpy arrays
        re.sub(r"^dim_\d=$", "", f"{dim}=")
        for dim in coords
    ]

    # Can't use a generator as this function may be delayed
    diff_msgs = []

    for count, *diff_coords in zip(diffs_count, *diffs_coords):
        addr = ", ".join(f"{dim_tag}{i}" for dim_tag, i in zip(dim_tags, diff_coords))
        diff_msgs.append(f"{msg_prefix}[{addr}]: {count} differences")

    return diff_msgs


def _array0d_to_scalar(x: xarray.DataArray) -> Any:
    """Convert a 0-dimensional DataArray to either its item
    or a dask delayed that returns the item.
    """
    assert not x.dims
    if x.chunks is not None:
        from dask import delayed

        return delayed(lambda x_np: x_np.item())(x.data)
    return x.data.item()


def _dtype_str(obj: object) -> str:
    """Generate dtype information for object.
    For non-numpy objects, this is just the object class.
    Numpy-based objects also contain the data type (e.g. int32).

    Fixed-length numpy strings that differ only by length should be
    treated as identical, e.g. <U3 and <U6 will both return <U.
    Sub-types of datetime64 must also be discarded.

    :param obj:
        any object being compared
    :return:
        dtype string
    """
    try:
        dtype = type(obj).__name__
    except AttributeError:  # pragma: nocover
        # FIXME This used to be triggered in Python 2. Is it still possible to get here?
        # Maybe some poorly written C extensions?
        dtype = str(type(obj))

    if isinstance(obj, np.integer):
        dtype = "int"
    elif isinstance(obj, np.floating):
        dtype = "float"
    elif isinstance(obj, np.complexfloating):
        dtype = "complex"

    if isinstance(obj, (pd.MultiIndex, pd.RangeIndex)):
        np_dtype = None
    elif isinstance(obj, (np.ndarray, pd.Series, xarray.DataArray)) or (
        PANDAS_GE_200
        and isinstance(obj, pd.Index)
        and not isinstance(obj, (pd.MultiIndex, pd.RangeIndex))
    ):
        np_dtype = obj.dtype
    elif isinstance(obj, pd.DataFrame):
        dtypes = obj.dtypes.unique()
        np_dtype = dtypes[0] if len(dtypes) == 1 else None
    else:
        np_dtype = None

    if np_dtype:
        np_dtype = str(np_dtype)
        if np_dtype[:2] in {"<U", "|S"}:
            np_dtype = np_dtype[:2] + "..."
        if np_dtype.startswith("datetime64"):
            np_dtype = "datetime64"
        return f"{dtype}<{np_dtype}>"
    return dtype
