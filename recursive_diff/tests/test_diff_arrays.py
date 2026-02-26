import numpy as np
import pandas as pd
import xarray

from recursive_diff import diff_arrays
from recursive_diff.tests import requires_dask


def test_diff_arrays(chunk):
    # MultiIndex; numeric
    a = xarray.DataArray(
        [[1.0, 2.0], [3.0, 4.0]], dims=["x", "y"], coords={"x": ["a", "b"]}, name="a"
    )
    b = xarray.DataArray(
        [[1.0, 2.0], [3.0, 5.0]], dims=["x", "y"], coords={"x": ["a", "b"]}, name="b"
    )

    # Only metadata differs
    c = xarray.DataArray([1.0, 2.0], name="c")
    d = xarray.DataArray([1.0, 2.0], name="d")

    # Single index; not numeric
    e = xarray.DataArray(["foo", "bar"], name="e")
    f = xarray.DataArray(["foo", "baz"], name="f")

    if chunk:
        a = a.chunk()
        b = b.chunk()
        c = c.chunk()
        d = d.chunk()
        e = e.chunk()
        f = f.chunk()

    lhs = [a, 1, c, e]
    rhs = [b, 2, d, f]

    arrays, other = diff_arrays(lhs, rhs)
    assert arrays.keys() == {"[0][data]", "[3][data]"}
    assert other == [
        "[0][name]: a != b",
        "[1]: 1 != 2 (abs: 1.0e+00, rel: 1.0e+00)",
        "[2][name]: c != d",
        "[3][name]: e != f",
    ]

    pd.testing.assert_frame_equal(
        arrays["[0][data]"],
        pd.DataFrame(
            {
                "lhs": [4.0],
                "rhs": [5.0],
                "abs_delta": [1.0],
                "rel_delta": [0.25],
                "x": ["b"],
                "y": np.asarray([1]),
            }
        ).set_index(["x", "y"]),
    )

    pd.testing.assert_frame_equal(
        arrays["[3][data]"],
        pd.DataFrame(
            {"lhs": ["bar"], "rhs": ["baz"]},
            index=pd.Index(np.asarray([1]), name="dim_0"),
        ),
    )


def test_no_diffs(chunk):
    a = xarray.DataArray([1, 2])
    b = xarray.DataArray([1, 2])
    if chunk:
        a = a.chunk()
        b = b.chunk()
    assert diff_arrays(a, b) == ({}, [])


def test_no_arrays():
    actual = diff_arrays(["foo", 1], ["bar", 1])
    expect = {}, ["[0]: foo != bar"]
    assert actual == expect


def test_0d(chunk):
    """0d arrays produce text diffs"""
    if chunk:
        import dask.array as da

        a = xarray.DataArray(da.asarray(1.0), name="n")
        b = xarray.DataArray(da.asarray(2.0), name="n")
    else:
        a = xarray.DataArray(1.0)
        b = xarray.DataArray(2.0)

    expect = {}, ["[data]: 1.0 != 2.0 (abs: 1.0e+00, rel: 1.0e+00)"]
    actual = diff_arrays(a, b)
    assert actual == expect


def test_brief_dims(chunk):
    a = xarray.DataArray(
        [[1.0, 2.0], [3.0, 4.0]], dims=["x", "y"], coords={"x": ["a", "b"]}
    )
    b = xarray.DataArray(
        [[1.0, 2.0], [3.0, 5.0]], dims=["x", "y"], coords={"x": ["a", "b"]}
    )
    if chunk:
        a = a.chunk()
        b = b.chunk()

    arrays, other = diff_arrays(a, b, brief_dims=["x"])
    assert arrays.keys() == {"[data]"}
    assert other == []

    pd.testing.assert_frame_equal(
        arrays["[data]"],
        pd.DataFrame(
            {"diffs_count": np.asarray([1])},
            index=pd.Index(np.asarray([1]), name="y"),
        ),
    )

    arrays, other = diff_arrays(a, b, brief_dims=["y"])
    assert arrays.keys() == {"[data]"}
    assert other == []
    pd.testing.assert_frame_equal(
        arrays["[data]"],
        pd.DataFrame({"diffs_count": np.asarray([1])}, index=pd.Index(["b"], name="x")),
    )

    # When all dims are brief, print as text
    expect = {}, ["[data]: 1 differences"]
    actual = diff_arrays(a, b, brief_dims=["x", "y", "z"])
    assert actual == expect
    actual = diff_arrays(a, b, brief_dims="all")
    assert actual == expect


def test_name_collision(chunk):
    """Test that name collisions between the input dims and the generated dataframe
    columns are handled gracefully.
    """
    a = xarray.DataArray([1.0, 2.0], dims=["lhs"], coords={"lhs": ["a", "b"]})
    b = xarray.DataArray([1.0, 3.0], dims=["lhs"], coords={"lhs": ["a", "b"]})

    if chunk:
        a = a.chunk()
        b = b.chunk()

    arrays, other = diff_arrays(a, b)
    assert arrays.keys() == {"[data]"}
    assert other == []

    pd.testing.assert_frame_equal(
        arrays["[data]"],
        pd.DataFrame(
            {
                "lhs": [2.0],
                "rhs": [3.0],
                "abs_delta": [1.0],
                "rel_delta": [0.5],
            },
            index=pd.Index(["b"], name="lhs"),
        ),
    )


def test_preserve_int_dtype(chunk):
    a = xarray.DataArray(
        np.asarray([1, 2], dtype=np.int8),
        dims=["x"],
        coords={"x": np.asarray([10, 20], dtype=np.int32)},
    )
    b = xarray.DataArray(
        np.asarray([1, 3], dtype=np.int16),
        dims=["x"],
        coords={"x": np.asarray([10, 20], dtype=np.int32)},
    )
    if chunk:
        a = a.chunk()
        b = b.chunk()

    arrays, other = diff_arrays(a, b)
    assert arrays.keys() == {"[data]"}
    assert other == ["object type differs: DataArray<int8> != DataArray<int16>"]

    pd.testing.assert_frame_equal(
        arrays["[data]"],
        pd.DataFrame(
            {
                "lhs": np.asarray([2], dtype=np.int8),
                "rhs": np.asarray([3], dtype=np.int16),
                # automatic promotion between int8 and int16
                "abs_delta": np.asarray([1], dtype=np.int16),
                "rel_delta": np.asarray([0.5], dtype=np.float64),
            },
            index=pd.Index(np.asarray([20], dtype=np.int32), name="x"),
        ),
    )


def test_index_mismatch(chunk):
    a = xarray.DataArray([1.0, 2.0, 3.0], dims=["x"], coords={"x": ["a", "b", "c"]})
    b = xarray.DataArray([1.0, 3.0], dims=["x"], coords={"x": ["a", "b"]})
    if chunk:
        a = a.chunk()
        b = b.chunk()

    arrays, other = diff_arrays(a, b)
    assert arrays.keys() == {"[data]"}
    assert other == ["[index][x]: c is in LHS only"]

    pd.testing.assert_frame_equal(
        arrays["[data]"],
        pd.DataFrame(
            {
                "lhs": [2.0],
                "rhs": [3.0],
                "abs_delta": [1.0],
                "rel_delta": [0.5],
            },
            index=pd.Index(["b"], name="x"),
        ),
    )


@requires_dask
def test_delayed():
    from dask import delayed

    @delayed
    def f(x):
        return x

    actual = diff_arrays(f([1, 2]), f([1, 3]))
    expect = {}, ["[1]: 2 != 3 (abs: 1.0e+00, rel: 5.0e-01)"]
    assert actual == expect
