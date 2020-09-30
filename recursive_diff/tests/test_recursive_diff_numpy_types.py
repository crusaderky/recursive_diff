import datetime

import numpy as np

from recursive_diff import recursive_diff


def check(lhs, rhs, *expect, rel_tol=1e-09, abs_tol=0.0,
          brief_dims=()):
    expect = set(expect)
    actual = set(recursive_diff(
        lhs, rhs,
        rel_tol=rel_tol, abs_tol=abs_tol,
        brief_dims=brief_dims))
    assert actual == expect


def test_numpy_mixed_date_types():

    a = np.array(['2000-01-01', '2000-01-02'])
    b = np.array([datetime.date(2000, 1, 1), datetime.date(2000, 1, 2)])
    c = np.array([np.datetime64('2000-01-01'), np.datetime64('2000-01-02')])
    check(a, b,
          'object type differs: ndarray<<U...> != ndarray<object>',
          '[data][0]: 2000-01-01 != 2000-01-01',
          '[data][1]: 2000-01-02 != 2000-01-02')
    check(a, c,
          'object type differs: ndarray<<U...> != ndarray<datetime64>',
          '[data][0]: 2000-01-01 != 2000-01-01 00:00:00',
          '[data][1]: 2000-01-02 != 2000-01-02 00:00:00')
    check(b, c,
          'object type differs: ndarray<object> != ndarray<datetime64>',
          '[data][0]: 2000-01-01 != 2000-01-01 00:00:00',
          '[data][1]: 2000-01-02 != 2000-01-02 00:00:00')


def test_numpy_mixed_datetime_types():

    a = np.array(['2000-01-01 12:34:56', '2000-01-02 01:23:45'])
    b = np.array([datetime.datetime(2000, 1, 1, 12, 34, 56),
                  datetime.datetime(2000, 1, 2, 1, 23, 45)])
    c = np.array([np.datetime64('2000-01-01 12:34:56'),
                  np.datetime64('2000-01-02 01:23:45')])
    check(a, b,
          'object type differs: ndarray<<U...> != ndarray<object>',
          '[data][0]: 2000-01-01 12:34:56 != 2000-01-01 12:34:56',
          '[data][1]: 2000-01-02 01:23:45 != 2000-01-02 01:23:45')
    check(a, c,
          'object type differs: ndarray<<U...> != ndarray<datetime64>',
          '[data][0]: 2000-01-01 12:34:56 != 2000-01-01 12:34:56',
          '[data][1]: 2000-01-02 01:23:45 != 2000-01-02 01:23:45')
    check(b, c,
          'object type differs: ndarray<object> != ndarray<datetime64>')


def test_numpy_mixed_nat_types():

    n1 = np.array(['2000-01-01 12:34:56', 'NaT'])
    n2 = np.array([np.datetime64('2000-01-01 12:34:56'), np.datetime64('NaT')])
    check(n1, n2,
          'object type differs: ndarray<<U...> != ndarray<datetime64>',
          '[data][0]: 2000-01-01 12:34:56 != 2000-01-01 12:34:56',
          '[data][1]: NaT != NaT')


def test_numpy_same_type_string():

    s1 = np.array(['foo', 'bar', 'bar', 'foo'])
    s2 = np.array(['foo', 'foo', 'bar', 'bar'])

    check(s1, s2,
          "[data][1]: bar != foo",
          "[data][3]: foo != bar")


def test_numpy_mixed_string_types():

    so = np.array(['foo', 'bar'], dtype=object)
    su = np.array(['foo', 'bar'], dtype='U')
    ss = np.array(['foo', 'bar'], dtype='S')

    check(so, su,
          "object type differs: ndarray<object> != ndarray<<U...>")
    check(so, ss,
          "object type differs: ndarray<object> != ndarray<|S...>",
          "[data][0]: foo != b'foo'",
          "[data][1]: bar != b'bar'")
    check(su, ss,
          "object type differs: ndarray<<U...> != ndarray<|S...>",
          "[data][0]: foo != b'foo'",
          "[data][1]: bar != b'bar'")


def test_numpy_same_type_bool():

    b1 = np.array([True, True, False, False], dtype=bool)
    b2 = np.array([True, False, False, True], dtype=bool)

    check(b1, b2,
          "[data][1]: True != False",
          "[data][3]: False != True")


def test_numpy_mixed_bool_types():

    b = np.array([True, False], dtype=bool)
    bo = np.array([True, False], dtype=object)
    ib = np.array([1, 0], dtype=np.int64)

    check(b, bo,
          "object type differs: ndarray<bool> != ndarray<object>")
    check(b, ib,
          "object type differs: ndarray<bool> != ndarray<int64>")
    check(bo, ib,
          "object type differs: ndarray<object> != ndarray<int64>")


def test_numpy_mixed_between_bool_string_types():

    b = np.array([True, False], dtype=bool)
    bo = np.array([True, False], dtype=object)

    so = np.array(['True', 'False'], dtype=object)
    su = np.array(['True', 'False'], dtype='U')
    ss = np.array(['True', 'False'], dtype='S')

    check(b, so,
          "object type differs: ndarray<bool> != ndarray<object>",
          "[data][0]: True != True",
          "[data][1]: False != False")
    check(b, su,
          "object type differs: ndarray<bool> != ndarray<<U...>",
          "[data][0]: True != True",
          "[data][1]: False != False")
    check(b, ss,
          "object type differs: ndarray<bool> != ndarray<|S...>",
          "[data][0]: True != b'True'",
          "[data][1]: False != b'False'")
    check(bo, so,
          "[data][0]: True != True",
          "[data][1]: False != False")
    check(bo, su,
          "object type differs: ndarray<object> != ndarray<<U...>",
          "[data][0]: True != True",
          "[data][1]: False != False")
    check(bo, ss,
          "object type differs: ndarray<object> != ndarray<|S...>",
          "[data][0]: True != b'True'",
          "[data][1]: False != b'False'")
