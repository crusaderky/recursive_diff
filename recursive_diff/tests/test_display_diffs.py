import sys
import types

import pandas as pd
import pytest
import xarray

from recursive_diff import display_diffs


@pytest.fixture
def display(monkeypatch):
    mod = types.ModuleType("IPython.display")

    out = []

    def display(x):
        out.append(type(x) if isinstance(x, pd.DataFrame) else x)

    def HTML(x):
        return x

    mod.display = display
    mod.HTML = HTML

    monkeypatch.setitem(sys.modules, "IPython.display", mod)
    return out


def test_single_array(display):
    """When there are differences in data and nothing else, hide the title"""
    a = xarray.DataArray([1, 2])
    b = xarray.DataArray([4, 2])
    display_diffs(a, b)
    assert display == [pd.DataFrame]


def test_single_array_with_metadata(display):
    a = xarray.DataArray([1, 2], name="a")
    b = xarray.DataArray([4, 2], name="b")
    display_diffs(a, b)
    assert display == [
        "<h3>[data]</h3>",
        pd.DataFrame,
        "<h3>Other differences</h3>",
        "<ul>\n\t<li>[name]: a != b</li>\n</ul>",
    ]


def test_dataset(display):
    a = xarray.Dataset({"v1": ("x", [1, 2]), "v2": ("x", [3, 4])}, attrs={"foo": 1})
    b = xarray.Dataset({"v1": ("x", [1, 5]), "v2": ("x", [3, 6]), "y": [1, 2]})
    display_diffs(a, b)
    assert display == [
        "<h3>[data_vars][v1]</h3>",
        pd.DataFrame,
        "<h3>[data_vars][v2]</h3>",
        pd.DataFrame,
        "<h3>Other differences</h3>",
        "<ul>\n"
        "\t<li>[attrs]: Pair foo:1 is in LHS only</li>\n"
        "\t<li>[index]: Dimension y is in RHS only</li>\n"
        "</ul>",
    ]


def test_no_diffs(display):
    display_diffs(1, 1)
    assert display == ["No differences found"]


def test_one_diff_many_arrays(display):
    """Don't hide the variable name when only one variable differs"""
    a = xarray.Dataset({"v1": ("x", [1, 2]), "v2": ("x", [3, 4])})
    b = xarray.Dataset({"v1": ("x", [1, 5]), "v2": ("x", [3, 4])})
    display_diffs(a, b)
    assert display == [
        "<h3>[data_vars][v1]</h3>",
        pd.DataFrame,
    ]


def test_nonindex_coords(display):
    a = xarray.DataArray([1, 2], dims=["x"], coords={"ni": ("x", ["a", "b"])})
    b = xarray.DataArray([4, 2], dims=["x"], coords={"ni": ("x", ["a", "c"])})
    display_diffs(a, b)
    assert display == [
        "<h3>[coords][ni]</h3>",
        pd.DataFrame,
        "<h3>[data]</h3>",
        pd.DataFrame,
    ]
