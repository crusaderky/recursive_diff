"""Public API functions"""

from __future__ import annotations

from collections.abc import Collection, Generator, Hashable
from typing import Any, Literal

import numpy as np
import pandas as pd

from recursive_diff.dask_compat import Array, Delayed, compute
from recursive_diff.recursive_diff import recursive_diff_impl


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
    diffs: list[list[str] | Array | Delayed] = []
    for diff in recursive_diff_impl(
        lhs,
        rhs,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        brief_dims=brief_dims,
        as_dataframes=False,
        path=[],
        seen_lhs={},
        seen_rhs={},
    ):
        if isinstance(diff, str):
            if diffs:
                diffs.append([diff])
            else:
                yield diff
        else:
            assert isinstance(diff, (Delayed, Array))
            # Comparison of Delayed objects or Dask-backed arrays
            diffs.append(diff)

    (computed_diffs,) = compute(diffs)
    for diff_batch in computed_diffs:
        yield from diff_batch


def recursive_eq(
    lhs: Any,
    rhs: Any,
    rel_tol: float = 1e-09,
    abs_tol: float = 0.0,
    *,  # TODO move before rel_tol (breaking change)
    brief_dims: Collection[Hashable] | Literal["all"] = (),
) -> None:
    """Wrapper around :func:`recursive_diff`.

    Print out all differences to stdout and finally assert that there are none.
    This is meant to be used inside pytest, where stdout is captured.
    """
    diffs_iter = recursive_diff(
        lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol, brief_dims=brief_dims
    )
    i = -1
    for i, diff in enumerate(diffs_iter):  # noqa: B007
        print(diff)
    i += 1
    if i == 0:
        return
    if brief_dims:
        msg = "Found differences; see stdout"
    else:
        msg = f"Found {i} differences; see stdout"
    raise AssertionError(msg)


def diff_arrays(
    lhs: Any,
    rhs: Any,
    *,
    rel_tol: float = 1e-09,
    abs_tol: float = 0.0,
    brief_dims: Collection[Hashable] | Literal["all"] = (),
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Compare two objects with :func:`recursive_diff`.

    Return tuple of:

    - {path: dataframe of differences} for all NumPy, Pandas, and Xarray objects found.
      Arrays with no differences won't be returned.
    - List of all other differences found. This includes differences in metadata,
      shape, dtype, and indices in NumPy, Pandas, and Xarray objects.
    """
    diffs = list(
        recursive_diff_impl(
            lhs,
            rhs,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            brief_dims=brief_dims,
            as_dataframes=True,
            path=[],
            seen_lhs={},
            seen_rhs={},
        )
    )
    (computed_diffs,) = compute(diffs)

    array_diffs = {}
    other_diffs = []

    for diff in computed_diffs:
        if isinstance(diff, str):
            # Eager comparison
            other_diffs.append(diff)
        elif isinstance(diff, tuple):
            # Comparison of arrays (both eager and dask-backed)
            path, constructor, args = diff
            if args[0].size:
                assert path not in array_diffs
                df = constructor(*args)
                array_diffs[path] = df
        else:
            # - Delayed objects
            # - 0-dimensional dask-backed arrays
            # - 1+dimensional dask-backed arrays with brief_dims squashing them to 0d
            assert isinstance(diff, (list, np.ndarray))
            other_diffs.extend(diff)

    return array_diffs, other_diffs


def display_diffs(
    lhs: Any,
    rhs: Any,
    *,
    rel_tol: float = 1e-09,
    abs_tol: float = 0.0,
    brief_dims: Collection[Hashable] | Literal["all"] = (),
) -> None:
    """Compare two objects with :func:`recursive_diff`.

    Display all differences in Jupyter notebook, with diffs in NumPy, Pandas, and Xarray
    objects displayed as tables.
    """
    from IPython.display import HTML, display

    array_diffs, other_diffs = diff_arrays(
        lhs,
        rhs,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        brief_dims=brief_dims,
    )

    for path, df in array_diffs.items():
        # Hide the title when comparing two np.ndarray, xarray.DataArray, or
        # pd.Series and there are no differences in metadata or coordinates
        if len(array_diffs) > 1 or other_diffs or path != "[data]":
            display(HTML(f"<h3>{path}</h3>"))
        display(df)

    if other_diffs:
        if array_diffs:
            display(HTML("<h3>Other differences</h3>"))

        html = (
            "<ul>\n" + "".join(f"\t<li>{diff}</li>\n" for diff in other_diffs) + "</ul>"
        )
        display(HTML(html))

    if not array_diffs and not other_diffs:
        display(HTML("No differences found"))
