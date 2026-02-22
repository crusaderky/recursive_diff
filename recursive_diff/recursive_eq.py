"""Tools for unit testing"""

from __future__ import annotations

from typing import Any, Collection, Hashable, Literal

from recursive_diff.recursive_diff import recursive_diff


def recursive_eq(
    lhs: Any,
    rhs: Any,
    rel_tol: float = 1e-09,
    abs_tol: float = 0.0,
    *,
    brief_dims: Collection[Hashable] | Literal["all"] = (),
) -> None:
    """Wrapper around :func:`~recursive_diff.recursive_diff`.
    Print out all differences and finally assert that there are none.
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
