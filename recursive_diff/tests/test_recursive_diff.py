import math
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import xarray

from recursive_diff import cast, recursive_diff
from recursive_diff.tests import (
    PANDAS_GE_200,
    PANDAS_GE_300,
    TO_ZARR_V2,
    XARRAY_GE_2024_5_0,
    XARRAY_GE_2025_1_2,
    filter_old_numpy_warnings,
    requires_dask,
    requires_netcdf,
    requires_zarr,
)
from recursive_diff.tests.memory import MemoryMonitor


class Rectangle:
    """Sample class to test custom comparisons."""

    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __eq__(self, other):
        # Never invoked thanks to @cast.register
        raise AssertionError("__eq__ should not be called")  # pragma: nocover

    def __repr__(self):
        return f"Rectangle({self.w}, {self.h})"


class Drawing:
    """Another class that is not Rectangle but just happens to be cast to the
    same dict
    """

    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __eq__(self, other):
        # Never invoked thanks to @cast.register
        raise AssertionError("__eq__ should not be called")  # pragma: nocover


@cast.register(Rectangle)
@cast.register(Drawing)
def _(obj):
    return {"w": obj.w, "h": obj.h}


class Circle:
    """A class which that supports == but is not registered"""

    def __init__(self, radius):
        self.radius = radius

    def __eq__(self, other):
        return self.radius == other.radius

    def __repr__(self):
        return f"Circle({self.radius})"


class Square:
    """Another unregistered class"""

    def __init__(self, side):
        self.side = side

    def __eq__(self, other):
        return self.side == other.side

    def __repr__(self):
        return f"Square({self.side})"


def check(lhs, rhs, *expect, rel_tol=1e-09, abs_tol=0.0, brief_dims=()):
    expect = sorted(expect)
    actual = sorted(
        recursive_diff(
            lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol, brief_dims=brief_dims
        )
    )
    assert actual == expect


@pytest.mark.parametrize(
    "x",
    [
        123,
        123.0,
        12 + 3j,
        "blah",
        "a\nb",
        math.nan,
        np.nan,
        math.nan + 1j,
        1 + math.nan * 1j,
        math.inf,
        np.inf,
        math.inf + 1j,
        1 + math.inf * 1j,
        True,
        False,
        [1, 2],
        (1, 2),
        np.int8(1),
        np.uint8(1),
        np.int64(1),
        np.uint64(1),
        np.float32(1),
        np.float64(1),
        np.complex64(1 + 2j),
        np.complex128(1 + 2j),
        {1: 2, 3: 4},
        {1, 2},
        frozenset([1, 2]),
        np.arange(10),
        np.arange(10, dtype=np.float64),
        np.array([1.2j]),
        np.array([np.nan + 1j]),
        pytest.param(
            np.array([np.inf + 1j]),
            marks=[pytest.mark.filterwarnings("ignore:Invalid value")]
            if np.__version__ < "2"
            else [],
        ),
        pytest.param(
            np.array([np.nan + np.inf * 1j]),
            marks=[pytest.mark.filterwarnings("ignore:Invalid value")]
            if np.__version__ < "2"
            else [],
        ),
        np.array([np.nan + np.nan * 1j]),
        pd.Series([1, 2]),
        pd.Series([1, 2], index=[3, 4]),
        pd.RangeIndex(10),
        pd.RangeIndex(1, 10, 3),
        pd.Index([1, 2, 3]),
        pd.MultiIndex.from_tuples(
            [("bar", "one"), ("bar", "two"), ("baz", "one")], names=["l1", "l2"]
        ),
        pd.DataFrame([[1, 2], [3, 4]]),
        pd.DataFrame([[1, 2], [3, 4]], index=["i1", "i2"], columns=["c1", "c2"]),
        xarray.DataArray([1, 2]),
        xarray.DataArray([1, 2], dims=["x"], coords={"x": [3, 4]}),
        xarray.Dataset(data_vars={"v": ("x", [1, 2])}, coords={"x": [3, 4]}),
        Rectangle(1, 2),
        Circle(1),
    ],
)
def test_identical(x):
    assert not list(recursive_diff(x, deepcopy(x)))
    assert not list(recursive_diff(x, deepcopy(x), brief_dims="all"))


@requires_dask
@pytest.mark.parametrize(
    "x",
    [
        xarray.DataArray([1, 2]),
        xarray.DataArray([1, 2], dims=["x"], coords={"x": [3, 4]}),
        xarray.Dataset(data_vars={"v": ("x", [1, 2])}, coords={"x": [3, 4]}),
    ],
)
def test_identical_dask(x):
    check(x.chunk(), deepcopy(x).chunk())


def test_simple():
    check(1, 0, "1 != 0 (abs: -1.0e+00, rel: -1.0e+00)")
    check("asd", "lol", "asd != lol")
    check(b"asd", b"lol", "b'asd' != b'lol'")
    check(True, False, "True != False")


def test_object_type_differs():
    check(1, "1", "1 != 1", "object type differs: int != str")
    check(True, 1, "object type differs: bool != int")
    check(False, 0, "object type differs: bool != int")
    check([1, 2], (1, 2), "object type differs: list != tuple")
    check({1, 2}, frozenset([1, 2]), "object type differs: set != frozenset")


def test_collections():
    check([1, 2], [1, 2, 3], "RHS has 1 more elements than LHS: [3]")
    check({1, 2}, {1, 2, (3, 4)}, "(3, 4) is in RHS only")
    check({1, 2}, {1}, "2 is in LHS only")
    check(
        {"x": 10, "y": 20},
        {"x": 10, "y": 30},
        "[y]: 20 != 30 (abs: 1.0e+01, rel: 5.0e-01)",
    )
    check({2: 20}, {1: 10}, "Pair 1:10 is in RHS only", "Pair 2:20 is in LHS only")


def test_limit_str_length():
    """Long and multi-line strings are truncated"""
    check("a" * 100, "a" * 100)
    check("a" * 100, "a" * 101, "{} ... != {} ...".format("a" * 76, "a" * 76))
    check("a\nb", "a\nb")
    check("a\nb", "a\nc", "a ... != a ...")


@pytest.mark.parametrize("nan", [np.nan, math.nan])
def test_nan(nan):
    check(nan, nan)
    check(nan, math.nan)
    check(nan, np.nan)
    check(0.0, nan, "0.0 != nan (abs: nan, rel: nan)")
    check(nan, 0.0, "nan != 0.0 (abs: nan, rel: nan)")


def test_float():
    """Float comparison with tolerance"""
    # Test that floats are not accidentally rounded when printing
    check(
        123456.7890123456,
        123456.789,
        "123456.7890123456 != 123456.789 (abs: -1.2e-05, rel: -1.0e-10)",
        rel_tol=0,
        abs_tol=0,
    )

    check(123, 123.0)  # int vs. float
    check(123.0, 123)  # float vs. int
    check(123, 123.01, abs_tol=0.1)  # difference below tolerance
    check(123, 123.01, "123 != 123.01 (abs: 1.0e-02, rel: 8.1e-05)")

    check(
        123456.7890123456,
        123456.789,
        "123456.7890123456 != 123456.789 (abs: -1.2e-05, rel: -1.0e-10)",
        rel_tol=1e-11,
        abs_tol=0,
    )
    check(
        123456.7890123456,
        123456.789,
        "123456.7890123456 != 123456.789 (abs: -1.2e-05, rel: -1.0e-10)",
        rel_tol=0,
        abs_tol=1e-5,
    )

    check(123456.7890123456, 123456.789, rel_tol=0, abs_tol=1e-4)
    check(123456.7890123456, 123456.789, rel_tol=1e-7, abs_tol=0)

    # Abs tol is RHS - LHS; rel tol is RHS / LHS - 1
    check(80.0, 175.0, "80.0 != 175.0 (abs: 9.5e+01, rel: 1.2e+00)")

    # tolerance settings are retained when descending into containers
    check(
        [{"x": (1.0, 2.0)}],
        [{"x": (1.1, 2.01)}],
        "[0][x][0]: 1.0 != 1.1 (abs: 1.0e-01, rel: 1.0e-01)",
        rel_tol=0.05,
        abs_tol=0,
    )

    # tolerance > 1 in a comparison among int's
    # note how int's are not cast to float when both lhs and rhs are int
    check(1, 2, abs_tol=2)
    check(2, 5, "2 != 5 (abs: 3.0e+00, rel: 1.5e+00)", abs_tol=2)


def test_float_division_by_zero():
    # Division by zero in relative delta
    check(0.0, 0.1, "0.0 != 0.1 (abs: 1.0e-01, rel: nan)")
    check(0.1, 0.0, "0.1 != 0.0 (abs: -1.0e-01, rel: -1.0e+00)")


def test_complex_division_by_zero():
    # Division by zero in relative delta
    check(
        0.1 + 0j,
        0 + 0j,
        "(0.1+0j) != 0j (abs: -1.0e-01+0.0e+00j, rel: -1.0e+00+0.0e+00j)",
    )
    check(0 + 0.1j, 0 + 0j, "0.1j != 0j (abs: 0.0e+00-1.0e-01j, rel: 0.0e+00-1.0e+00j)")
    check(0 + 0j, 0.1 + 0j, "0j != (0.1+0j) (abs: 1.0e-01+0.0e+00j, rel: nan)")
    check(0 + 0j, 0 + 0.1j, "0j != 0.1j (abs: 0.0e+00+1.0e-01j, rel: nan)")


def test_int_vs_float():
    """ints are silently cast to float and do not cause an
    'object type differs' error.
    """
    check(123, 123.0)
    check(123, 123.0000000000001)  # difference is below rel_tol=1e-9
    check(1, 1.01, "1 != 1.01 (abs: 1.0e-02, rel: 1.0e-02)", abs_tol=0.001)
    check(1, 1.01, abs_tol=0.1)


def test_int_vs_complex():
    """ints are NOT silently cast to complex"""
    msg = "object type differs: int != complex"
    check(123, 123.0 + 0j, msg)
    check(123, 123.0000000000001 + 0j, msg)  # difference is below rel_tol=1e-9
    check(
        1,
        1.01 + 0j,
        msg,
        "1 != (1.01+0j) (abs: 1.0e-02+0.0e+00j, rel: 1.0e-02+0.0e+00j)",
        abs_tol=0.001,
    )
    check(1, 1.01 + 0j, msg, abs_tol=0.1)


def test_float_vs_complex():
    """floats are NOT silently cast to complex"""
    msg = "object type differs: float != complex"
    check(123.0, 123.0 + 0j, msg)
    check(123.0, 123.0000000000001 + 0j, msg)  # difference is below rel_tol=1e-9
    check(
        1.0,
        1.01 + 0j,
        msg,
        "1.0 != (1.01+0j) (abs: 1.0e-02+0.0e+00j, rel: 1.0e-02+0.0e+00j)",
        abs_tol=0.001,
    )
    check(1.0, 1.01 + 0j, msg, abs_tol=0.1)


def test_complex_nan():
    check(
        math.nan + 1j,
        math.nan + 1.01j,
        "(nan+1j) != (nan+1.01j) (abs: nan+1.0e-02j, rel: nan+1.0e-02j)",
    )
    check(
        math.nan + 1j,
        math.nan + 1.01j,
        abs_tol=0.1,
    )


def test_complex_infinities():
    check(
        math.inf + 1j,
        math.inf + 1.01j,
        "(inf+1j) != (inf+1.01j) (abs: 0.0e+00+1.0e-02j, rel: 0.0e+00+1.0e-02j)",
    )
    check(math.inf + 1j, math.inf + 1.01j, abs_tol=0.1)
    check(
        math.inf + 1j,
        -math.inf + 1.01j,
        "(inf+1j) != (-inf+1.01j) (abs: -inf+1.0e-02j, rel: nan+1.0e-02j)",
    )


def test_identity():
    """Test a fast path for identical objects. Note that this is also triggered
    by small integers, all strings hardcoded in a unit test, and other internalized
    objects.
    """
    lhs = [1, 123456, "foo", "bar"]  # All these objects are internalized!
    lhs[1] += 1  # Output of dynamic operation and no longer internalized
    lhs[3] += "baz"
    # Referencing objects directly from lhs is pointless in CPython due to
    # internalization. This is just for the sake of being implementation agnostic.
    rhs = [lhs[0], 123456, lhs[2], "bar"]
    rhs[1] += 1
    rhs[3] += "baz"

    check(lhs, rhs)


def test_numpy_types():
    """scalar numpy data types (not to be confused with numpy arrays)
    are silently cast to pure numpy types and do not cause an
    'object type differs' error. They're compared with tolerance.
    """
    check(123, np.int32(123))
    check(np.int64(123), np.int32(123))
    check(123, np.float32(123))
    check(123, np.float64(123))
    check(np.float32(123), np.float64(123))
    check(1 + 2j, np.complex64(1 + 2j))
    check(1 + 2j, np.complex128(1 + 2j))
    check(np.complex64(1 + 2j), np.complex128(1 + 2j))
    check(
        np.float64(1),
        np.float64(1.01),
        "1.0 != 1.01 (abs: 1.0e-02, rel: 1.0e-02)",
        abs_tol=0.001,
    )
    check(np.float32(1), np.float32(1.01), abs_tol=0.1)
    check(np.float64(1), np.float64(1.01), abs_tol=0.1)


@filter_old_numpy_warnings
def test_numpy():
    # test tolerance and comparison of float vs. int
    check(
        np.array([1.0, 2.0, 3.01, 4.0001, 5.0]),
        np.array([1, 4, 3, 4], dtype=np.int64),
        "[data][1]: 2.0 != 4 (abs: 2.0e+00, rel: 1.0e+00)",
        "[data][2]: 3.01 != 3 (abs: -1.0e-02, rel: -3.3e-03)",
        "[dim_0]: LHS has 1 more elements than RHS",
        "object type differs: ndarray<float64> != ndarray<int64>",
        abs_tol=0.001,
    )

    # Tolerance > 1 in a comparison among int's
    # Make sure that tolerance is not applied to RangeIndex comparison
    check(
        np.array([1, 2]),
        np.array([2, 20, 3, 4]),
        "[data][1]: 2 != 20 (abs: 1.8e+01, rel: 9.0e+00)",
        "[dim_0]: RHS has 2 more elements than LHS",
        abs_tol=10,
    )

    # array of numbers vs. dates; mismatched size
    subsecond = "000000" if PANDAS_GE_300 else "000000000"
    check(
        np.array([1, 2], dtype=np.int64),
        pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]).values,
        f"[data][0]: 1 != 2000-01-01T00:00:00.{subsecond}",
        f"[data][1]: 2 != 2000-01-02T00:00:00.{subsecond}",
        "[dim_0]: RHS has 1 more elements than LHS",
        "object type differs: ndarray<int64> != ndarray<datetime64>",
    )

    # array of numbers vs. strings; mismatched size
    check(
        np.array([1, 2, 3], dtype=np.int64),
        np.array(["foo", "bar"]),
        "[data][0]: 1 != foo",
        "[data][1]: 2 != bar",
        "[dim_0]: LHS has 1 more elements than RHS",
        "object type differs: ndarray<int64> != ndarray<<U...>",
    )

    # Mismatched dimensions
    check(
        np.array([1, 2, 3, 4]),
        np.array([[1, 2], [3, 4]]),
        "[dim_0]: LHS has 2 more elements than RHS",
        "Dimension dim_1 is in RHS only",
    )

    # numpy vs. list
    check(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        [[1, 4, 3], [4, 5, 6]],
        "object type differs: ndarray<int64> != list",
        "[data][0, 1]: 2 != 4 (abs: 2.0e+00, rel: 1.0e+00)",
    )

    # list vs. numpy
    check(
        [[1, 4, 3], [4, 5, 6]],
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        "object type differs: list != ndarray<int64>",
        "[data][0, 1]: 4 != 2 (abs: -2.0e+00, rel: -5.0e-01)",
    )

    # numpy vs. other object
    check(
        np.array([0, 0], dtype=np.int64),
        0,
        "Dimension dim_0 is in LHS only",
        "object type differs: ndarray<int64> != int",
    )


def test_numpy_strings():
    """Strings in numpy can be unicode (<U...), binary ascii (<S...)
    or Python variable-length (object).
    Test that these three types are not considered equivalent.
    """
    a = np.array(["foo"], dtype=object)
    b = np.array(["foo"], dtype="U")
    c = np.array(["foo"], dtype="S")
    check(a, b, "object type differs: ndarray<object> != ndarray<<U...>")
    check(
        a,
        c,
        "object type differs: ndarray<object> != ndarray<|S...>",
        "[data][0]: foo != b'foo'",
    )
    check(
        b,
        c,
        "object type differs: ndarray<<U...> != ndarray<|S...>",
        "[data][0]: foo != b'foo'",
    )


@pytest.mark.parametrize("x,y", [("foo", "barbaz"), (b"foo", b"babaz")])
def test_numpy_string_slice(x, y):
    """When slicing an array of strings, the output sub-dtype won't change.
    Test that string that differs only by dtype-length are considered
    equivalent.
    """
    a = np.array([x, y])  # dtype=<U6/<S6
    b = a[:1]  # dtype=<U6/<S6
    c = np.array([x])  # dtype=<U3/<S3
    assert a.dtype == b.dtype
    assert a.dtype != c.dtype
    check(b, c)


@pytest.mark.filterwarnings(  # xarray < 2025.1.2
    "ignore:Converting non-nanosecond precision datetime:UserWarning"
)
def test_numpy_dates():
    a = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03", "NaT"]).values.astype(
        "<M8[D]"
    )
    b = pd.to_datetime(
        [
            "2000-01-01",  # identical
            "2000-01-04",  # differs, both LHS and RHS are non-NaT
            "NaT",  # non-NaT vs. NaT
            "NaT",  # NaT == NaT
            # differences in sub-type must be ignored
        ]
    ).values.astype("<M8[ns]")
    subsecond = "" if PANDAS_GE_200 and XARRAY_GE_2025_1_2 else ".000000000"
    check(
        a,
        b,
        f"[data][1]: 2000-01-02T00:00:00{subsecond} != 2000-01-04T00:00:00.000000000",
        f"[data][2]: 2000-01-03T00:00:00{subsecond} != NaT",
    )


@pytest.mark.filterwarnings(  # xarray < 2025.1.2
    "ignore:Converting non-nanosecond precision datetime:UserWarning"
)
def test_numpy_dates_ns():
    """Test nanosecond accuracy of M8[ns]"""
    a = pd.to_datetime(
        ["2000-01-01T00:00:00.000000000", "2000-01-01T00:00:00.000000000"]
    ).values
    b = pd.to_datetime(
        ["2000-01-01T00:00:00.000000000", "2000-01-01T00:00:00.000000001"]
    ).values
    check(
        a,
        b,
        "[data][1]: 2000-01-01T00:00:00.000000000 != 2000-01-01T00:00:00.000000001",
    )


@pytest.mark.skipif(not XARRAY_GE_2025_1_2, reason="Requires xarray >= 2025.1.2")
def test_numpy_dates_beyond_ns():
    """M8[ns] supports dates from year 1677 to 2262. Test dates beyond this range."""
    a = pd.to_datetime(
        ["1500-01-01", "1500-01-02", "2300-01-01", "2300-01-02"]
    ).values.astype("<M8[s]")
    b = pd.to_datetime(
        ["1500-01-01", "1500-01-03", "2300-01-01", "2300-01-03"]
    ).values.astype("<M8[D]")
    subsecond = "" if PANDAS_GE_200 else ".000000000"
    check(
        a,
        b,
        f"[data][1]: 1500-01-02T00:00:00{subsecond} != 1500-01-03T00:00:00{subsecond}",
        f"[data][3]: 2300-01-02T00:00:00{subsecond} != 2300-01-03T00:00:00{subsecond}",
    )


def test_numpy_scalar():
    check(
        np.array(1, dtype=np.int64),
        np.array(2.5),
        "[data]: 1 != 2.5 (abs: 1.5e+00, rel: 1.5e+00)",
        "object type differs: ndarray<int64> != ndarray<float64>",
    )
    check(
        np.array(1, dtype=np.int64),
        2,
        "[data]: 1 != 2 (abs: 1.0e+00, rel: 1.0e+00)",
        "object type differs: ndarray<int64> != int",
    )
    check(np.array("foo"), np.array("bar"), "[data]: foo != bar")
    # Note: datetime64 are not 0-dimensional arrays
    check(
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        "2000-01-01 != 2000-01-02",
    )
    check(np.datetime64("2000-01-01"), np.datetime64("NaT"), "2000-01-01 != NaT")


def test_rtol_zerodivision_float():
    """test rtol calculation of floats where one of the two elements is 0"""
    check(
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        "[1]: 1.0 != 0.0 (abs: -1.0e+00, rel: -1.0e+00)",
        "[2]: 0.0 != 1.0 (abs: 1.0e+00, rel: nan)",
    )


def test_rtol_zerodivision_numpy():
    """test rtol calculation of numpy arrays where one of the two elements is 0"""
    check(
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        "[data][1]: 1.0 != 0.0 (abs: -1.0e+00, rel: -1.0e+00)",
        "[data][2]: 0.0 != 1.0 (abs: 1.0e+00, rel: nan)",
    )


def test_pandas_series():
    # pd.Series
    # Note that we're also testing that order is ignored
    check(
        pd.Series([1, 2, 3], index=["foo", "bar", "baz"], name="hello"),
        pd.Series([1, 3, 4], index=["foo", "baz", "bar"], name="world"),
        "[data][index=bar]: 2 != 4 (abs: 2.0e+00, rel: 1.0e+00)",
        "[name]: hello != world",
    )


def test_pandas_dataframe_one_dtype():
    df1 = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]], index=["x1", "x2"], columns=["y1", "y2", "y3"]
    )
    df2 = pd.DataFrame(
        [[1, 3, 2], [4, 7, 5]], index=["x1", "x2"], columns=["y1", "y3", "y4"]
    )

    check(
        df1,
        df2,
        "[data][column=y3, index=x2]: 6 != 7 (abs: 1.0e+00, rel: 1.7e-01)",
        "[columns]: y2 is in LHS only",
        "[columns]: y4 is in RHS only",
    )

    df3 = df1.astype(float)
    # Difference in dtype is only reported once
    check(
        df1,
        df3,
        "object type differs: DataFrame<int64> != DataFrame<float64>",
    )


def test_pandas_dataframe_many_dtypes():
    df1 = pd.DataFrame(
        {"x": [1, 2, 3, 4], "y": ["a", "b", "c", "d"]}, index=["i1", "i2", "i3", "i4"]
    )
    df2 = pd.DataFrame(
        {"x": [1.0, 3.001, 2.1], "y": ["b", "c", "b"], "z": [1, 2, 3]},
        index=["i1", "i3", "i2"],
    )

    check(
        df1,
        df2,
        "[columns]: z is in RHS only",
        "[index]: i4 is in LHS only",
        "[dtypes][x]: object type differs: Series<int64> != Series<float64>",
        "[data][x][index=i2]: 2 != 2.1 (abs: 1.0e-01, rel: 5.0e-02)",
        "[data][y][index=i1]: a != b",
        abs_tol=0.05,
    )


def test_pandas_index():
    # Regular index
    # Test that order is ignored
    # Use huge abs_tol and rel_tol to test that tolerance is ignored
    int_index = "Index<int64>" if PANDAS_GE_200 else "Int64Index"
    float_index = "Index<float64>" if PANDAS_GE_200 else "Float64Index"
    check(
        pd.Index([1, 2, 3, 4]),
        pd.Index([1, 3.000001, 2]),
        "3 is in LHS only",
        "3.000001 is in RHS only",
        "4 is in LHS only",
        f"object type differs: {int_index} != {float_index}",
        rel_tol=10,
        abs_tol=10,
    )

    check(pd.Index(["x", "y", "z"]), pd.Index(["y", "x"]), "z is in LHS only")


def test_pandas_index_numeric_vs_object():
    int_index = "Index<int64>" if PANDAS_GE_200 else "Int64Index"
    obj_index = "Index<object>" if PANDAS_GE_200 else "Index"
    check(
        pd.Index([1, 2, 3]),
        pd.Index([1, "3", 2]),
        "3 is in LHS only",
        "3 is in RHS only",
        f"object type differs: {int_index} != {obj_index}",
    )


def test_pandas_rangeindex():
    # RangeIndex(stop)
    check(pd.RangeIndex(10), pd.RangeIndex(10))
    check(pd.RangeIndex(8), pd.RangeIndex(10), "RHS has 2 more elements than LHS")
    check(pd.RangeIndex(10), pd.RangeIndex(8), "LHS has 2 more elements than RHS")

    # RangeIndex(start, stop, step, name)
    check(pd.RangeIndex(1, 2, 3, name="x"), pd.RangeIndex(1, 2, 3, name="x"))
    check(
        pd.RangeIndex(0, 4, 1),
        pd.RangeIndex(1, 4, 1),
        "RangeIndex(start=0, stop=4, step=1) != RangeIndex(start=1, stop=4, step=1)",
    )
    check(
        pd.RangeIndex(0, 4, 2),
        pd.RangeIndex(0, 5, 2),
        "RangeIndex(start=0, stop=4, step=2) != RangeIndex(start=0, stop=5, step=2)",
    )
    check(
        pd.RangeIndex(0, 4, 2),
        pd.RangeIndex(0, 4, 3),
        "RangeIndex(start=0, stop=4, step=2) != RangeIndex(start=0, stop=4, step=3)",
    )
    check(
        pd.RangeIndex(4, name="foo"),
        pd.RangeIndex(4, name="bar"),
        "RangeIndex(start=0, stop=4, step=1, name='foo') != "
        "RangeIndex(start=0, stop=4, step=1, name='bar')",
    )

    # RangeIndex vs regular index
    int_index = "Index<int64>" if PANDAS_GE_200 else "Int64Index"
    check(
        pd.RangeIndex(4),
        pd.Index([0, 1, 2]),
        "3 is in LHS only",
        f"object type differs: RangeIndex != {int_index}",
    )

    # Regular index vs RangeIndex
    check(
        pd.Index([0, 1, 2]),
        pd.RangeIndex(4),
        "3 is in RHS only",
        f"object type differs: {int_index} != RangeIndex",
    )


def test_pandas_multiindex():
    lhs = pd.MultiIndex.from_tuples(
        [("bar", "one"), ("bar", "two"), ("baz", "one")],
        names=["l1", "l2"],
    )
    rhs = pd.MultiIndex.from_tuples(
        [("baz", "one"), ("bar", "three"), ("bar", "one"), ("baz", "four")],
        names=["l1", "l3"],  # Differences in names are ignored
    )
    check(
        lhs,
        rhs,
        "('bar', 'three') is in RHS only",
        "('bar', 'two') is in LHS only",
        "('baz', 'four') is in RHS only",
    )

    # MultiIndex vs. regular index.
    # This goes through a special case path where pd.Index.isin raises.
    int_index = "Index<int64>" if PANDAS_GE_200 else "Int64Index"
    check(
        lhs,
        pd.Index([0, 1, 2]),
        "('bar', 'one') is in LHS only",
        "('bar', 'two') is in LHS only",
        "('baz', 'one') is in LHS only",
        "0 is in RHS only",
        "1 is in RHS only",
        "2 is in RHS only",
        f"object type differs: MultiIndex != {int_index}",
    )


@pytest.fixture(params=[False, pytest.param(True, marks=requires_dask)])
def chunk(request):
    return request.param


def test_xarray(chunk):
    # xarray.Dataset
    ds1 = xarray.Dataset(
        data_vars={"d1": ("x", [1, 2, 3]), "d2": (("y", "x"), [[4, 5, 6], [7, 8, 9]])},
        coords={
            "x": ("x", ["x1", "x2", "x3"]),
            "y": ("y", ["y1", "y2"]),
            "nonindex": ("x", ["ni1", "ni2", "ni3"]),
        },
        attrs={"some": "attr", "some2": 1},
    )

    ds2 = ds1.copy(deep=True)
    del ds2["d1"]
    ds2["d2"][0, 0] = 10
    ds2["nonindex"][1] = "ni4"
    ds2.attrs["some2"] = 2
    ds2.attrs["other"] = "someval"

    if chunk:
        ds1 = ds1.chunk()
        ds2 = ds2.chunk()

    # Older versions of Xarray don't have the 'Size 24B' bit
    d1_str = str(ds1["d1"]).splitlines()[0].strip()

    check(
        ds1,
        ds2,
        "[attrs]: Pair other:someval is in RHS only",
        "[attrs][some2]: 1 != 2 (abs: 1.0e+00, rel: 1.0e+00)",
        "[coords][nonindex][x=x2]: ni2 != ni4",
        f"[data_vars]: Pair d1:{d1_str} ... is in LHS only",
        "[data_vars][d2][x=x1, y=y1]: 4 != 10 (abs: 6.0e+00, rel: 1.5e+00)",
    )

    check(
        ds1,
        ds2,
        "[attrs]: Pair other:someval is in RHS only",
        "[coords][nonindex][x=x2]: ni2 != ni4",
        f"[data_vars]: Pair d1:{d1_str} ... is in LHS only",
        abs_tol=7,
    )

    # xarray.DataArray
    # Note: this sample has a non-index coordinate
    # In Linux, int maps to int64 while in Windows it maps to int32
    da1 = ds1["d2"].astype(np.int64)
    da1.name = "foo"
    da1.attrs["attr1"] = 1.0
    da1.attrs["attr2"] = 1.0

    # Test dimension order does not matter
    check(da1, da1.T)

    da2 = da1.copy(deep=True).astype(float)
    da2[0, 0] *= 1.0 + 1e-7
    da2[0, 1] *= 1.0 + 1e-10
    da2["nonindex"][1] = "ni4"
    da2.name = "bar"
    da2.attrs["attr1"] = 1.0 + 1e-7
    da2.attrs["attr2"] = 1.0 + 1e-10
    da2.attrs["attr3"] = "new"

    check(
        da1,
        da2,
        "[attrs]: Pair attr3:new is in RHS only",
        "[attrs][attr1]: 1.0 != 1.0000001 (abs: 1.0e-07, rel: 1.0e-07)",
        "[coords][nonindex][x=x2]: ni2 != ni4",
        "[data][x=x1, y=y1]: 4 != 4.0000004 (abs: 4.0e-07, rel: 1.0e-07)",
        "[name]: foo != bar",
        "object type differs: DataArray<int64> != DataArray<float64>",
    )


def test_xarray_scalar(chunk):
    da1 = xarray.DataArray(1.0)
    da2 = xarray.DataArray(1.0 + 1e-7)
    if chunk:
        da1 = da1.chunk()
        da2 = da2.chunk()

    check(da1, da2, "[data]: 1.0 != 1.0000001 (abs: 1.0e-07, rel: 1.0e-07)")
    da2 = xarray.DataArray(1.0 + 1e-10)
    check(da1, da2)


def test_xarray_no_coords(chunk):
    da1 = xarray.DataArray([0, 1])
    da2 = xarray.DataArray([0, 2])
    if chunk:
        da1 = da1.chunk()
        da2 = da2.chunk()

    check(
        da1,
        da2,
        "[data][1]: 1 != 2 (abs: 1.0e+00, rel: 1.0e+00)",
    )


def test_xarray_mismatched_dims_0d_1d(chunk):
    # 0-dimensional vs. 1+-dimensional
    da1 = xarray.DataArray(1.0)
    da2 = xarray.DataArray([0.0, 0.1])
    if chunk:
        da1 = da1.chunk()
        da2 = da2.chunk()

    check(da1, da2, "[index]: Dimension dim_0 is in RHS only")


def test_xarray_mismatched_dims_1dplus(chunk):
    # both arrays are 1+-dimensional
    da1 = xarray.DataArray([0, 1], dims=["x"])
    da2 = xarray.DataArray([[0, 1], [2, 3]], dims=["x", "y"])
    if chunk:
        da1 = da1.chunk()
        da2 = da2.chunk()

    check(da1, da2, "[index]: Dimension y is in RHS only")


def test_xarray_size0(chunk):
    da1 = xarray.DataArray([])
    da2 = xarray.DataArray([1.0])
    if chunk:
        da1 = da1.chunk()
        da2 = da2.chunk()

    check(da1, da2, "[index][dim_0]: RHS has 1 more elements than LHS")


def test_xarray_stacked(chunk):
    # Pre-stacked dims, mixed with non-stacked ones
    da1 = xarray.DataArray(
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        dims=["x", "y", "z"],
        coords={"x": ["x1", "x2"]},
    )
    if chunk:
        da1 = da1.chunk()

    # Stacked and unstacked dims are compared point by point,
    # while still pointing out the difference in stacking
    da2 = da1.copy(deep=True)
    da2[0, 0, 0] = 10
    da1 = da1.stack(s=["x", "y"])
    da2 = da2.stack(s=["x", "y"])
    check(
        da1,
        da2,
        "[data][s=('x1', 0), z=0]: 0 != 10 (abs: 1.0e+01, rel: nan)",
    )


def test_brief_dims_1d(chunk):
    # all dims are brief
    da1 = xarray.DataArray([1, 2, 3], dims=["x"])
    da2 = xarray.DataArray([1, 3, 4], dims=["x"])
    if chunk:
        da1 = da1.chunk()
        da2 = da2.chunk()

    check(
        da1,
        da2,
        "[data][x=1]: 2 != 3 (abs: 1.0e+00, rel: 5.0e-01)",
        "[data][x=2]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)",
    )
    check(da1, da2, "[data]: 2 differences", brief_dims=["x"])
    check(da1, da2, "[data]: 2 differences", brief_dims="all")

    check(da1, da1)
    check(da1, da1, brief_dims=["x"])
    check(da1, da1, brief_dims="all")


def test_brief_dims_nd(chunk):
    # some dims are brief
    da1 = xarray.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dims=["r", "c"])
    da2 = xarray.DataArray([[1, 5, 4], [4, 5, 6], [7, 8, 0]], dims=["r", "c"])
    if chunk:
        da1 = da1.chunk()
        da2 = da2.chunk()

    check(
        da1,
        da2,
        "[data][c=1, r=0]: 2 != 5 (abs: 3.0e+00, rel: 1.5e+00)",
        "[data][c=2, r=0]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)",
        "[data][c=2, r=2]: 9 != 0 (abs: -9.0e+00, rel: -1.0e+00)",
    )
    check(
        da1,
        da2,
        "[data][c=1]: 1 differences",
        "[data][c=2]: 2 differences",
        brief_dims=["r"],
    )
    check(
        da1,
        da2,
        "[data][r=0]: 2 differences",
        "[data][r=2]: 1 differences",
        brief_dims=["c"],
    )
    check(da1, da2, "[data]: 3 differences", brief_dims="all")

    check(da1, da1)
    check(da1, da1, brief_dims=["r"])
    check(da1, da1, brief_dims="all")


def test_brief_dims_nested(chunk):
    """Xarray object not at the first level, and not all variables have all
    brief_dims
    """
    lhs = {
        "foo": xarray.Dataset(
            data_vars={
                "x": (("r", "c"), [[1, 2, 3], [4, 5, 6]]),
                "y": ("c", [1, 2, 3]),
            }
        )
    }
    rhs = {
        "foo": xarray.Dataset(
            data_vars={
                "x": (("r", "c"), [[1, 2, 4], [4, 5, 6]]),
                "y": ("c", [1, 2, 4]),
            }
        )
    }
    if chunk:
        lhs["foo"] = lhs["foo"].chunk()
        rhs["foo"] = rhs["foo"].chunk()

    check(
        lhs,
        rhs,
        "[foo][data_vars][x][c=2, r=0]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)",
        "[foo][data_vars][y][c=2]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)",
    )
    check(
        lhs,
        rhs,
        "[foo][data_vars][x][c=2]: 1 differences",
        "[foo][data_vars][y][c=2]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)",
        brief_dims=["r"],
    )
    check(
        lhs,
        rhs,
        "[foo][data_vars][x]: 1 differences",
        "[foo][data_vars][y]: 1 differences",
        brief_dims="all",
    )


def test_nested1():
    # Subclasses of the supported types must only produce a type error
    class MyDict(dict):
        pass

    class MyList(list):
        pass

    class MyTuple(tuple):
        pass

    # Two complex arrays which are identical
    lhs = {
        "foo": [1, 2, (5.2, "asd")],
        "bar": None,
        "baz": np.array([1, 2, 3]),
        None: [np.array([1, 2, 3])],
    }
    rhs = MyDict(
        {
            "foo": MyList([1, 2, MyTuple((5.20000000001, "asd"))]),
            "bar": None,
            "baz": np.array([1, 2, 3]),
            None: [np.array([1, 2, 3])],
        }
    )
    check(
        lhs,
        rhs,
        "[foo]: object type differs: list != MyList",
        "[foo][2]: object type differs: tuple != MyTuple",
        "object type differs: dict != MyDict",
    )


def test_nested2():
    lhs = {
        "foo": [1, 2, ("asd", 5.2), 4],
        "bar": np.array([1, 2, 3, 4], dtype=np.int64),
        "baz": np.array([1, 2, 3], dtype=np.int64),
        "key_only_lhs": None,
    }
    rhs = {
        # type changed from tuple to list
        # a string content has changed
        # LHS outermost list is longer
        # RHS innermost list is longer
        "foo": [1, 2, ["lol", 5.2, 3]],
        # numpy dtype has changed
        # LHS is longer
        "bar": np.array([1, 2, 3], dtype=np.float64),
        # numpy vs. list
        "baz": [1, 2, 3],
        # Test string truncation
        "key_only_rhs": "a" * 200,
    }

    check(
        lhs,
        rhs,
        "[bar]: object type differs: ndarray<int64> != ndarray<float64>",
        "[bar][dim_0]: LHS has 1 more elements than RHS",
        "[baz]: object type differs: ndarray<int64> != list",
        "[foo]: LHS has 1 more elements than RHS: [4]",
        "[foo][2]: RHS has 1 more elements than LHS: [3]",
        "[foo][2]: object type differs: tuple != list",
        "[foo][2][0]: asd != lol",
        "Pair key_only_lhs:None is in LHS only",
        "Pair key_only_rhs:%s ... is in RHS only" % ("a" * 76),
    )


def test_custom_classes():
    check(
        Rectangle(1, 2),
        Rectangle(1.1, 2.7),
        "[h]: 2 != 2.7 (abs: 7.0e-01, rel: 3.5e-01)",
        abs_tol=0.5,
    )

    check(
        Rectangle(1, 2),
        Drawing(3, 2),
        "[w]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)",
        "object type differs: Rectangle != Drawing",
    )

    # Unregistered classes can still be compared but without
    # tolerance or recursion
    check(Circle(4), Circle(4.1), "Circle(4) != Circle(4.1)", abs_tol=0.5)

    check(
        Rectangle(4, 4),
        Square(4),
        "Cannot compare objects: Rectangle(4, 4), Square(4)",
        "object type differs: Rectangle != Square",
    )

    check(
        Circle(4),
        Square(4),
        "Cannot compare objects: Circle(4), Square(4)",
        "object type differs: Circle != Square",
    )


@requires_dask
@pytest.mark.parametrize(
    "chunk_lhs,chunk_rhs",
    [(None, None), (None, -1), (None, 2), ({"x": (1, 2)}, {"x": (2, 1)})],
)
def test_dask_dataarray(chunk_lhs, chunk_rhs):
    lhs = xarray.DataArray(["a", "b", "c"], dims=["x"])
    rhs = xarray.DataArray(["a", "b", "d"], dims=["x"])
    if chunk_lhs:
        lhs = lhs.chunk(chunk_lhs)
    if chunk_rhs:
        rhs = rhs.chunk(chunk_rhs)

    check(lhs, rhs, "[data][x=2]: c != d")


@requires_dask
def test_dask_dataarray_discards_data():
    """Test that chunked Dask datasets are loaded into memory and then
    discarded, without caching them in place with .load() or .persist()
    """
    import dask.array as da

    allow = True

    def f():
        assert allow
        return np.array([1, 2])

    a = xarray.DataArray(da.Array({("a", 0): (f,)}, "a", chunks=[(2,)], dtype="int"))
    b = xarray.DataArray([1, 2], name=a.name)
    check(a, b)

    # Test that the graph is computed again
    allow = False
    with pytest.raises(AssertionError):
        a.load()


@requires_dask
def test_dask_delayed():
    from dask import delayed

    a = delayed(lambda: [10, 20])()
    b = delayed(lambda: [10, 21])()
    c = delayed(lambda: (10, 20))()
    d = delayed([10, 20])

    check(a, b, "[1]: 20 != 21 (abs: 1.0e+00, rel: 5.0e-02)")
    check(a, c, "object type differs: list != tuple")
    check(a, d)
    check(a.compute(), d)
    check(a, d.compute())


def test_0d_arrays():
    a = xarray.DataArray(1)
    b = xarray.DataArray(2)
    check(a, b, "[data]: 1 != 2 (abs: 1.0e+00, rel: 1.0e+00)")


@requires_dask
def test_0d_arrays_dask():
    # Note: DataArray.chunk() does not convert 0d arrays to Dask
    import dask.array as da

    a = xarray.DataArray(da.asarray(1), name="foo")
    b = xarray.DataArray(da.asarray(2), name="foo")
    check(a, b, "[data]: 1 != 2 (abs: 1.0e+00, rel: 1.0e+00)")


def test_empty_arrays():
    a = xarray.DataArray(np.array([], dtype=np.int64))
    b = xarray.DataArray(np.array([], dtype=np.int32))
    check(a, b, "object type differs: DataArray<int64> != DataArray<int32>")


def test_recursion():
    lhs = []
    lhs.append(lhs)
    rhs = [1]
    check(lhs, rhs, "[0]: LHS recurses to [0]; RHS is not recursive")
    check(rhs, lhs, "[0]: LHS is not recursive; RHS recurses to [0]")

    rhs = []
    rhs.append(rhs)
    check(lhs, rhs)


def test_recursion_different_target_different():
    lhs = [[1, 2], [3, 4]]
    lhs.append(lhs[0])
    rhs = [[1, 2], [3, 4]]
    rhs.append(rhs[1])

    check(
        lhs,
        rhs,
        "[2][0]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)",
        "[2][1]: 2 != 4 (abs: 2.0e+00, rel: 1.0e+00)",
    )


def test_recursion_different_target_identical():
    lhs = [[1, 2], [1, 2]]
    lhs.append(lhs[0])
    rhs = [[1, 2], [1, 2]]
    rhs.append(rhs[1])
    check(lhs, rhs)


def test_repetition_is_not_recursion():
    class C:
        def __eq__(self, other):
            return isinstance(other, C)

    c1 = C()
    lhs = [c1, c1]
    rhs = [c1, C()]
    check(lhs, rhs)


@requires_netcdf
def test_lazy_datasets_without_dask(tmp_path):
    """Test that xarray datasets using internal lazy indices
    (e.g. xarray.open(..., chunks=None) ) are compared and are not permanently cached
    into memory afterwards.
    """
    a = xarray.Dataset({"v": ("x", [1, 2, 3])})
    b = xarray.Dataset({"v": ("x", [1, 2, 4])})
    a.to_netcdf(tmp_path / "a.nc")
    b.to_netcdf(tmp_path / "b.nc")

    a2 = xarray.open_dataset(tmp_path / "a.nc")
    b2 = xarray.open_dataset(tmp_path / "b.nc")

    check(a, b, "[data_vars][v][x=2]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)")

    # Check that the data is not cached in place
    assert not a2["v"]._in_memory
    assert not b2["v"]._in_memory
    a2.load()
    assert a2["v"]._in_memory


@pytest.mark.skipif(not XARRAY_GE_2024_5_0, reason="requires xarray>=2024.5.0")
@requires_dask
@requires_netcdf
@requires_zarr
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="process-wide memory readings")
@pytest.mark.parametrize(
    "chunks,max_peak",
    [pytest.param(None, 50, id="no dask"), pytest.param({}, 50, id="dask")],
)
@pytest.mark.parametrize("format", ["netcdf", "zarr"])
def test_lazy_datasets_huge(tmp_path, chunks, max_peak, format):
    import dask.array as da
    import dask.config

    # 320 MiB, 8 MiB per variable, 2 MiB per chunk, indices are pd.RangeIndex
    a = xarray.Dataset(
        {f"v{i}": ("x", da.random.random(1_000_000, chunks=250_000)) for i in range(40)}
    )
    if format == "netcdf":
        a.to_netcdf(tmp_path / "a.nc")
        b = xarray.open_dataset(tmp_path / "a.nc", chunks=chunks)
        c = xarray.open_dataset(tmp_path / "a.nc", chunks=chunks)
    else:
        a.to_zarr(tmp_path / "a.zarr", **TO_ZARR_V2)
        b = xarray.open_dataset(tmp_path / "a.zarr", engine="zarr", chunks=chunks)
        c = xarray.open_dataset(tmp_path / "a.zarr", engine="zarr", chunks=chunks)
    del a

    # Peak memory usage on Dask = thread count * thread heap size
    with dask.config.set({"num_workers": 1}), MemoryMonitor() as mm:
        check(b, c)

    mm.assert_peak(max_peak * 2**20)


@requires_dask
@pytest.mark.thread_unsafe(reason="process-wide dask config")
def test_dask_scheduler():
    """Test that recursive_diff respects the global dask scheduler config, e.g.
    one can define a process-wide distributed.Client, and doesn't always
    override it with the threaded scheduler.

    This test also checks that all dask objects are computed at once.
    """
    import dask.config
    import dask.delayed
    import dask.threaded

    seen = []

    def get(dsk, keys, **kwargs):
        seen.append((dsk, keys, kwargs))
        return dask.threaded.get(dsk, keys, **kwargs)

    a1 = xarray.DataArray([1, 2]).chunk()
    a2 = xarray.DataArray([1, 2]).chunk()
    b1 = xarray.DataArray([1, 3]).chunk()
    b2 = xarray.DataArray([1, 3]).chunk()
    c1 = dask.delayed(lambda: 1)()
    c2 = dask.delayed(lambda: 1)()

    with dask.config.set({"scheduler": get}):
        check([a1, b1, c1], [a2, b2, c2])
    assert len(seen) == 1


@requires_dask
@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "darwin", reason="Very slow and high memory usage")
@pytest.mark.thread_unsafe(reason="process-wide dask config")
def test_distributed_index_bloom():
    """Test against an issue where broadcasting the indices
    vs. the mask on dask.distributed can cause the worker memory
    to bloom.
    """
    distributed = pytest.importorskip("distributed")
    import dask.array as da
    import dask.config

    a = xarray.DataArray(
        # 781 MiB, 24 MiB per chunk
        # The issue of broadcasting the indices vs. the mask and accidentally
        # deep-copying afterwards grows with the number of dimensions.
        da.random.random((40,) * 5, chunks=(20,) * 5)
    )
    # Identical array but with different Dask keys
    b = xarray.where(a < 0, 1, a)

    with dask.config.set(
        {
            # P2P rechunk is slower and takes more memory!
            "array.rechunk.method": "tasks",
            "distributed.worker.memory.spill": False,
        }
    ), distributed.Client(
        n_workers=2,
        threads_per_worker=2,
        memory_limit="1 GiB",
    ):
        check(a, b)


@requires_dask
@pytest.mark.thread_unsafe(reason="spawns processes")
def test_p2p_rechunk():
    """Test support for p2p rechunk, which is the default when using
    dask.distributed.
    """
    distributed = pytest.importorskip("distributed")

    a = xarray.DataArray([1, 2, 3], dims=["x"], coords={"x": ["a", "b", "c"]})
    b = xarray.DataArray([1, 2, 4], dims=["x"], coords={"x": ["a", "b", "c"]})

    with distributed.Client(n_workers=2, threads_per_worker=1):
        check(a, b, "[data][x=c]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)")
