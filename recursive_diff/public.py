"""Public API functions"""

from __future__ import annotations

from typing import Any, Collection, Hashable, Literal

import numpy as np
import pandas as pd

from recursive_diff.dask_compat import compute
from recursive_diff.recursive_diff import _recursive_diff


def diff_arrays(
    lhs: Any,
    rhs: Any,
    *,
    rel_tol: float = 1e-09,
    abs_tol: float = 0.0,
    brief_dims: Collection[Hashable] | Literal["all"] = (),
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Compare two objects with :func:`~recursive_diff.recursive_diff`.

    Return tuple of:

    - {path: dataframe of differences} for all array objects found.
      Arrays with no differences won't be returned.
    - List of all other differences found.
    """
    diffs = list(
        _recursive_diff(
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
    """Compare two objects with :func:`~recursive_diff.recursive_diff`.

    Display all differences in Jupyter notebook, with diffs in array objects
    displayed as tables.
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
