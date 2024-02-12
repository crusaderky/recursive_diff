"""Tools for unit testing
"""
from typing import Any

from recursive_diff.recursive_diff import recursive_diff


def recursive_eq(
    lhs: Any, rhs: Any, rel_tol: float = 1e-09, abs_tol: float = 0.0
) -> None:
    """Wrapper around :func:`~recursive_diff.recursive_diff`.
    Print out all differences and finally assert that there are none.
    """
    diffs_iter = recursive_diff(lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol)
    i = -1
    for i, diff in enumerate(diffs_iter):  # noqa: B007
        print(diff)
    i += 1
    assert i == 0, f"{i} differences found"
