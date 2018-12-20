Extending recursive_diff/recursive_eq
*************************************

Without any changes, :func:`~recursive_diff.recursive_diff` and
:func:`~recursive_diff.recursive_eq` can compare arbitrary objects using the
``==`` operator. Take for example this custom class::

    >>> class Rectangle:
    ...    def __init__(self, w, h):
    ...        self.w = w
    ...        self.h = h
    ...
    ...    def __eq__(self, other):
    ...        return self.w == other.w and self.h == other.h
    ...
    ...    def __repr__(self):
    ...        return 'Rectangle(%f, %f)' % (self.w, self.h)

The above can be processed by recursive_diff, because it supports the ==
operator against objects of the same type, and when converted to string
it conveys meaningful information::

    >>> list(recursive_diff(Rectangle(1, 2), Rectangle(3, 4)))
    ['Rectangle(1.000000, 2.000000) != Rectangle(2.000000, 3.000000)']

However, it doesn't support the more powerful features of recursive_diff,
namely recursion and tolerance:

    >>> list(recursive_diff(
    ...     Rectangle(1, 2), Rectangle(1.1, 2.2), abs_tol=.5))
    ['Rectangle(1.0000000, 2.0000000) != Rectangle(1.100000, 2.200000)']

This can be fixed by registering a custom :func:`~recursive_diff.cast`
function::

    >>> from recursive_diff import cast
    >>> @cast.register(Rectangle)
    ... def _(obj, brief_dims):
    ...     return {'w': obj.w, 'h': obj.h}

After doing so, w and h will be compared with tolerance and, if they are
collections, will be recursively descended into::

    >>> list(recursive_diff(
    ...     Rectangle(1, 2), Rectangle(1.1, 2.7), abs_tol=.5))
    ['[h]: 2.0 != 2.7 (abs: 7.0e-01, rel: 3.5e-01)']