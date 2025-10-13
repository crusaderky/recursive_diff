import pytest

from recursive_diff import recursive_eq


def test_recursive_eq_success(capsys):
    recursive_eq(0, 0)
    assert capsys.readouterr().out == ""


def test_recursive_eq_fail(capsys):
    # Test the actual log lines dumped out by recursive_eq
    with pytest.raises(AssertionError):
        recursive_eq(("foo", ("bar", "baz")), ("foo", ("bar", "asd", "lol")))
    assert capsys.readouterr().out.splitlines() == [
        "[1]: RHS has 1 more elements than LHS: ['lol']",
        "[1][1]: baz != asd",
    ]
