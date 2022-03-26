"""Recursively compare Python objects.

See also its most commonly used wrapper:
:func:`~recursive_diff.testing.recursive_eq`
"""
from __future__ import annotations

import math
import re
from collections.abc import Collection, Hashable, Iterator
from typing import Any

import numpy
import pandas
import xarray

from recursive_diff import dask_or_stub as dask
from recursive_diff.cast import cast


def are_instances(lhs, rhs, cls) -> bool:
    """Return True if both lhs and rhs are instances of cls; False otherwise"""
    return isinstance(lhs, cls) and isinstance(rhs, cls)


def is_array(dtype: str) -> bool:
    return any(
        dtype.startswith(t) for t in ("ndarray", "DataArray", "Series", "DataFrame")
    )


def is_array_like(dtype: str) -> bool:
    return dtype in {"int", "float", "complex", "bool", "str", "list", "tuple"}


def recursive_diff(
    lhs: Any,
    rhs: Any,
    *,
    rel_tol: float = 1e-09,
    abs_tol: float = 0.0,
    brief_dims: Collection[Hashable] | str = (),
) -> Iterator[str]:
    """Compare two objects and yield all differences.
    The two objects must any of:

    - basic types (str, int, float, bool)
    - basic collections (list, tuple, dict, set, frozenset)
    - numpy scalar types
    - :class:`numpy.ndarray`
    - :class:`pandas.Series`
    - :class:`pandas.DataFrame`
    - :class:`pandas.Index`
    - :class:`xarray.DataArray`
    - :class:`xarray.Dataset`
    - any recursive combination of the above
    - any other object (compared with ==)

    Special treatment is reserved to different types:

    - floats and ints are compared with tolerance, using :func:`math.isclose`
    - NaN equals to NaN
    - bools are only equal to other bools
    - numpy arrays are compared elementwise and with tolerance,
      also testing the dtype
    - pandas and xarray objects are compared elementwise, with tolerance, and
      without order. Duplicate indices are not supported.
    - xarray dimensions and variables are compared without order
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

        - collection of strings representing xarray dimensions. If one or more
          differences are found along one of these dimensions, only one message
          will be reported, stating the differences count.
        - "all", to produce one line only for every xarray variable that
          differs

        Omit to output a line for every single different cell.

    Yields strings containing difference messages, prepended by the path to
    the point that differs.
    """
    yield from _recursive_diff(
        lhs,
        rhs,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        brief_dims=brief_dims,
        path=[],
        suppress_type_diffs=False,
        join="inner",
    )


def _recursive_diff(
    lhs: Any,
    rhs: Any,
    *,
    rel_tol: float,
    abs_tol: float,
    brief_dims: Collection[Hashable] | str,
    path: list[object],
    suppress_type_diffs: bool,
    join: str,
) -> Iterator[str]:
    """Recursive implementation of :func:`recursive_diff`

    :param list path:
        list of nodes traversed so far, to be prepended to all error messages
    :param bool suppress_type_diffs:
        if True, don't print out messages about differeces in type
    :param str join:
        join type of numpy objects: 'inner' or 'outer'.
        Ignored for plain Python collections (set, dict, etc.) for which
        outer join is always applied.

    This function calls itself recursively for all elements of numpy-based
    data, list, tuple, and dict.values(). Every time, it appends to the
    path list one element.
    """

    def diff(msg: str, print_path: list[object] = path) -> str:
        """Format diff message, prepending the formatted path"""
        path_prefix = "".join(f"[{elem}]" for elem in print_path)
        if path_prefix != "":
            path_prefix += ": "
        return path_prefix + msg

    # Build string representation of the two variables *before* casting
    lhs_repr = _str_trunc(lhs)
    rhs_repr = _str_trunc(rhs)

    # Identify if the variables are indices that must go through outer join,
    # *before* casting. This will be propagated downwards into the recursion.
    if join == "inner" and are_instances(lhs, rhs, pandas.Index):
        join = "outer"

    if (
        are_instances(lhs, rhs, xarray.DataArray)
        and "__strip_dataarray__" in lhs.attrs
        and "__strip_dataarray__" in rhs.attrs
    ):
        # Don't repeat dtype comparisons
        suppress_type_diffs = True

    # cast lhs and rhs to simpler data types; pretty-print data type
    dtype_lhs = _dtype_str(lhs)
    dtype_rhs = _dtype_str(rhs)
    lhs = cast(lhs, brief_dims=brief_dims)
    rhs = cast(rhs, brief_dims=brief_dims)

    # 1.0 vs. 1 must not be treated as a difference
    if isinstance(lhs, int) and isinstance(rhs, float):
        # Cast lhs to float
        dtype_lhs = "float"
        lhs = float(lhs)
    elif isinstance(rhs, int) and isinstance(lhs, float):
        # Cast rhs to float
        dtype_rhs = "float"
        rhs = float(rhs)

    # When comparing an array vs. a plain python list or scalar, log an error
    # for the different dtype and then proceed to compare the contents
    if is_array(dtype_lhs) and is_array_like(dtype_rhs):
        rhs = cast(numpy.array(rhs), brief_dims=brief_dims)
    elif is_array(dtype_rhs) and is_array_like(dtype_lhs):
        lhs = cast(numpy.array(lhs), brief_dims=brief_dims)

    # Allow mismatched comparison of a RangeIndex vs. a regular index
    if isinstance(lhs, pandas.RangeIndex) and not isinstance(rhs, pandas.RangeIndex):
        lhs = cast(pandas.Index(lhs.values), brief_dims=brief_dims)
    if isinstance(rhs, pandas.RangeIndex) and not isinstance(lhs, pandas.RangeIndex):
        rhs = cast(pandas.Index(rhs.values), brief_dims=brief_dims)

    if dtype_lhs != dtype_rhs and not suppress_type_diffs:
        yield diff(f"object type differs: {dtype_lhs} != {dtype_rhs}")

    # Continue even in case dtype doesn't match
    # This allows comparing e.g. a numpy array vs. a list or a tuple

    if are_instances(lhs, rhs, list):
        if len(lhs) > len(rhs):
            yield diff(
                "LHS has %d more elements than RHS: %s"
                % (len(lhs) - len(rhs), _str_trunc(lhs[len(rhs) :]))
            )
        elif len(lhs) < len(rhs):
            yield diff(
                "RHS has %d more elements than LHS: %s"
                % (len(rhs) - len(lhs), _str_trunc(rhs[len(lhs) :]))
            )
        for i, (lhs_i, rhs_i) in enumerate(zip(lhs, rhs)):
            yield from _recursive_diff(
                lhs_i,
                rhs_i,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                brief_dims=brief_dims,
                path=path + [i],
                suppress_type_diffs=suppress_type_diffs,
                join=join,
            )

    elif are_instances(lhs, rhs, set):
        for x in sorted(lhs - rhs, key=repr):
            yield diff(f"{_str_trunc(x)} is in LHS only")
        for x in sorted(rhs - lhs, key=repr):
            yield diff(f"{_str_trunc(x)} is in RHS only")

    elif are_instances(lhs, rhs, pandas.RangeIndex):
        # Pretty-print differences in size. This is used not only by
        # pandas.Series and pandas.DataFrame, but also by numpy arrays
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

    elif are_instances(lhs, rhs, dict):
        for key in sorted(lhs.keys() - rhs.keys(), key=repr):
            if isinstance(lhs[key], pandas.Index):
                join = "outer"
            if join == "outer":
                # Comparing an index
                yield diff(f"Dimension {key} is in LHS only")
            else:
                yield diff(f"Pair {key}:{_str_trunc(lhs[key])} is in LHS only")
        for key in sorted(rhs.keys() - lhs.keys(), key=repr):
            if isinstance(rhs[key], pandas.Index):
                join = "outer"
            if join == "outer":
                # Comparing an index
                yield diff(f"Dimension {key} is in RHS only")
            else:
                yield diff(f"Pair {key}:{_str_trunc(rhs[key])} is in RHS only")
        for key in sorted(rhs.keys() & lhs.keys(), key=repr):
            yield from _recursive_diff(
                lhs[key],
                rhs[key],
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                brief_dims=brief_dims,
                path=path + [key],
                suppress_type_diffs=suppress_type_diffs,
                join=join,
            )

    elif are_instances(lhs, rhs, bool):
        if lhs != rhs:
            yield diff(f"{lhs} != {rhs}")
    elif are_instances(lhs, rhs, str):
        if lhs != rhs:
            yield diff(f"{lhs_repr} != {rhs_repr}")
    elif are_instances(lhs, rhs, bytes):
        if lhs != rhs:
            yield diff(f"{lhs_repr} != {rhs_repr}")
    elif are_instances(lhs, rhs, (int, float, complex)):
        if math.isnan(lhs) and math.isnan(rhs):
            pass
        elif not math.isclose(lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol):
            try:
                rel_delta = rhs / lhs - 1
            except ZeroDivisionError:
                rel_delta = math.nan
            yield diff(f"{lhs} != {rhs} (abs: {rhs - lhs:.1e}, rel: {rel_delta:.1e})")

    elif are_instances(lhs, rhs, xarray.DataArray):
        # This block is executed for all data that was originally:
        # - numpy.ndarray
        # - pandas.Series
        # - pandas.DataFrame
        # - pandas.Index (except RangeIndex)
        # - xarray.DataArray
        # - xarray.Dataset
        # - any of the above, compared against a plain Python list

        # Both DataArrays are guaranteed by _strip_dataarray to be either
        # ravelled on a single dim with a MultiIndex or 0-dimensional

        lhs_dims = _get_stripped_dims(lhs)
        rhs_dims = _get_stripped_dims(rhs)

        if lhs_dims != rhs_dims:
            # This is already reported elsewhere when comparing dicts
            # (Dimension x is in LHS only)
            pass

        elif lhs.dims:
            # Load the entire objects into RAM. When parsing huge disk-backed
            # datasets, e.g. with landg.bin.ncdiff, you want to do this at the
            # very last possible moment. After this, we'll do:
            # - alignment, which is potentially very expensive with dask
            # - Extract differences (simplified code):
            #     diff = lhs != rhs
            #     lhs = lhs[lhs != rhs].compute()
            #     rhs = rhs[lhs != rhs].compute()
            #   The above 3 lines, if lhs and rhs were dask-backed, would
            #   effectively load the arrays 3 times each.
            lhs, rhs = dask.compute(lhs, rhs)

            # Align to guarantee that the index is identical on both sides.
            # Change the order as needed. Fill the gaps with NaNs.

            # index variables go through an outer join, whereas data variables
            # and non-index coords use an inner join. This avoids creating
            # spurious NaNs in the data variable and only reporting missing
            # elements only once
            lhs, rhs = xarray.align(lhs, rhs, join=join)

            # Build array of bools that highlight all differences, use it to
            # filter the two inputs, and finally convert only the differences
            # to pure python. This is MUCH faster than iterating on all
            # elements in the case where most elements are identical.
            if lhs.dtype.kind in "iufc" and rhs.dtype.kind in "iufc":
                # Both arrays are numeric
                # i = int8, int16, int32, int64
                # u = uint8,uint16, uint32, uint64
                # f = float32, float64
                # c = complex64, complex128
                diffs = ~numpy.isclose(
                    lhs.values, rhs.values, rtol=rel_tol, atol=abs_tol, equal_nan=True
                )

            elif lhs.dtype.kind == "M" and rhs.dtype.kind == "M":
                # Both arrays are datetime64
                # Unlike with numpy.isclose(equal_nan=True), there is no
                # straightforward way to do a comparison of dates where
                # NaT == NaT returns True.
                # All datetime64's, including NaT, can be cast to milliseconds
                # since 1970-01-01 (NaT is a special harcoded value).
                # We must first normalise the subtype, so that you can
                # transparently compare e.g. <M8[ns] vs. <M8[D]
                diffs = lhs.astype("<M8[ns]").astype(int) != rhs.astype(
                    "<M8[ns]"
                ).astype(int)

            else:
                # At least one between lhs and rhs is non-numeric,
                # e.g. bool or str
                diffs = lhs.values != rhs.values

                # Comparison between two non-scalar, incomparable types
                # (like strings and numbers) will return True
                if diffs is True:
                    diffs = numpy.full(lhs.shape, dtype=bool, fill_value=True)

            if diffs.ndim > 1 and lhs.dims[-1] == "__stacked__":
                # N>0 original dimensions, some (but not all) of which are in
                # brief_dims
                assert brief_dims
                # Produce diffs count along brief_dims
                diffs = diffs.astype(int).sum(axis=tuple(range(diffs.ndim - 1)))
                # Reattach original coords
                diffs = xarray.DataArray(
                    diffs,
                    dims=["__stacked__"],
                    coords={"__stacked__": lhs.coords["__stacked__"]},
                )
                # Filter out identical elements
                diffs = diffs[diffs != 0]
                # Convert the diff count to plain dict with the original coords
                diffs = _dataarray_to_dict(diffs)
                for k, count in sorted(diffs.items()):
                    yield diff(f"{count} differences", print_path=path + [k])

            elif "__stacked__" not in lhs.dims:
                # N>0 original dimensions, all of which are in brief_dims

                # Produce diffs count along brief_dims
                count = diffs.astype(int).sum()
                if count:
                    yield diff(f"{count} differences")
            else:
                # N>0 original dimensions, none of which are in brief_dims

                # Filter out identical elements
                lhs = lhs[diffs]
                rhs = rhs[diffs]
                # Convert the original arrays to plain dict
                lhs = _dataarray_to_dict(lhs)
                rhs = _dataarray_to_dict(rhs)

                if join == "outer":
                    # We're here showing the differences of two non-range
                    # indices, aligned on themselves. All dict values are NaN
                    # by definition, so we can print a terser output by
                    # converting the dicts to sets.
                    lhs = {k for k, v in lhs.items() if not pandas.isnull(v)}
                    rhs = {k for k, v in rhs.items() if not pandas.isnull(v)}

                # Finally dump out all the differences
                yield from _recursive_diff(
                    lhs,
                    rhs,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                    brief_dims=brief_dims,
                    path=path,
                    suppress_type_diffs=True,
                    join=join,
                )

        else:
            # 0-dimensional arrays
            assert lhs.dims == ()
            assert rhs.dims == ()
            yield from _recursive_diff(
                lhs.values.tolist(),
                rhs.values.tolist(),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                brief_dims=brief_dims,
                path=path,
                suppress_type_diffs=True,
                join=join,
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


def _get_stripped_dims(a: xarray.DataArray) -> list[Hashable]:
    """Helper function of :func:`recursive_diff`.

    :param xarray.DataArray a:
        array that has been stripped with :func:`_strip_dataarray`
    :returns:
        list of original dims, sorted alphabetically
    """
    if "__stacked__" in a.dims:
        res = set(a.coords["__stacked__"].to_index().names)
        res |= set(a.dims) - {"__stacked__"}
        return sorted(res)
    return list(a.dims)


def _dtype_str(obj: Any) -> str:
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
    except AttributeError:
        # Base types don't have __name__
        dtype = str(type(obj))

    if isinstance(obj, numpy.integer):
        dtype = "int"
    elif isinstance(obj, numpy.floating):
        dtype = "float"

    if isinstance(obj, (numpy.ndarray, pandas.Series, xarray.DataArray)):
        np_dtype = obj.dtype
    elif isinstance(obj, pandas.DataFrame):
        # TODO: support for DataFrames with different dtypes on different
        # columns. See also cast(obj: pandas.DataFrame)
        np_dtype = obj.values.dtype
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


def _dataarray_to_dict(a: xarray.DataArray) -> dict[str, Any]:
    """Helper function of :func:`recursive_diff`.
    Convert a DataArray prepared by :func:`_strip_dataarray` to a plain
    Python dict.

    :param a:
        :class:`xarray.DataArray` which has exactly 1 dimension,
        no non-index coordinates, and a MultiIndex on its dimension.
    :returns:
        Plain python dict, where the keys are a string representation
        of the points of the MultiIndex.

    .. note::
       Order will be discarded. Duplicate coordinates are not supported.
    """
    assert a.dims == ("__stacked__",)
    dims = a.coords["__stacked__"].to_index().names
    res = {}
    for idx, val in a.to_pandas().iteritems():
        key = ", ".join(f"{d}={i}" for d, i in zip(dims, idx))
        # Prettier output when there was no coord at the beginning,
        # e.g. with plain numpy arrays
        key = re.sub(r"dim_\d+=", "", key)
        res[key] = val
    return res
