import pytest
import xarray

from recursive_diff import recursive_eq


def test_recursive_eq_success(capsys):
    recursive_eq(0, 0)
    assert capsys.readouterr().out == ""


def test_recursive_eq_fail(capsys):
    # Test the actual log lines dumped out by recursive_eq
    with pytest.raises(AssertionError, match="Found 2 differences"):
        recursive_eq(("foo", ("bar", "baz")), ("foo", ("bar", "asd", "lol")))
    assert capsys.readouterr().out.splitlines() == [
        "[1]: RHS has 1 more elements than LHS: ['lol']",
        "[1][1]: baz != asd",
    ]


def test_recursive_eq_brief_dims(capsys):
    a = xarray.DataArray([[1, 2, 3], [4, 5, 6]], dims=["x", "y"])
    b = xarray.DataArray([[1, 3, 5], [4, 5, 7]], dims=["x", "y"])
    with pytest.raises(AssertionError, match="Found 3 differences"):
        recursive_eq(a, b)
    assert capsys.readouterr().out.splitlines() == [
        "[data][x=0, y=1]: 2 != 3 (abs: 1.0e+00, rel: 5.0e-01)",
        "[data][x=0, y=2]: 3 != 5 (abs: 2.0e+00, rel: 6.7e-01)",
        "[data][x=1, y=2]: 6 != 7 (abs: 1.0e+00, rel: 1.7e-01)",
    ]
    with pytest.raises(AssertionError, match="Found differences"):
        recursive_eq(a, b, brief_dims=["x"])
    assert capsys.readouterr().out.splitlines() == [
        "[data][y=1]: 1 differences",
        "[data][y=2]: 2 differences",
    ]
