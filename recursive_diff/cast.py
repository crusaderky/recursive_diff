from functools import singledispatch
import numpy
import pandas
import xarray
from .proper_unstack import proper_unstack


@singledispatch
def cast(obj, brief_dims):
    """Helper function of :func:`recursive_diff`.

    Cast objects into simpler object types:

    - Cast tuple to list
    - Cast frozenset to set
    - Cast all numpy-based objects to :class:`xarray.DataArray`, as it is the
      most generic format that can describe all use cases:

      - :class:`numpy.ndarray`
      - :class:`pandas.Series`
      - :class:`pandas.DataFrame`
      - :class:`pandas.Index`, except :class:`pandas.RangeIndex`, which is
        instead returned unaltered
      - :class:`xarray.Dataset`

    The data will be potentially wrapped by a dict to hold the various
    attributes and marked so that it doesn't trigger an infinite recursion.

    - Do nothing for any other object types.

    :param obj:
        complex object that must be simplified
    :param tuple brief_dims:
        sequence of xarray dimensions that must be compacted.
        See documentation on :func:`recursive_diff`.
    :returns:
        simpler object to compare
    """
    # This is a single dispatch function, defining the default for any
    # classes not explicitly registered below.
    return obj


@cast.register(numpy.integer)
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for all numpy scalar
    integers (not to be confused with numpy arrays of integers)
    """
    return int(obj)


@cast.register(numpy.floating)
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for all numpy scalar
    floats (not to be confused with numpy arrays of floats)
    """
    return float(obj)


@cast.register(numpy.ndarray)
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`numpy.ndarray`.

    Map to a DataArray with dimensions dim_0, dim_1, ... and
    RangeIndex() as the coords.
    """
    data = _strip_dataarray(xarray.DataArray(obj), brief_dims=brief_dims)
    obj = {
        'dim_%d' % i: pandas.RangeIndex(size)
        for i, size in enumerate(obj.shape)
    }
    obj['data'] = data
    return obj


@cast.register(pandas.Series)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.Series`.

    Map to a DataArray.
    """
    return {
        'name': obj.name,
        'data': _strip_dataarray(
            xarray.DataArray(obj, dims=['index']), brief_dims=brief_dims),
        'index': obj.index,
    }


@cast.register(pandas.DataFrame)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.DataFrame`.

    Map to a DataArray.

    TODO: proper support for columns with different dtypes. Right now
    they are cast to the closest common type by DataFrame.values.
    """
    return {
        'data': _strip_dataarray(
            xarray.DataArray(obj, dims=['index', 'column']),
            brief_dims=brief_dims),
        'index': obj.index,
        'columns': obj.columns,
    }


@cast.register(xarray.DataArray)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`xarray.DataArray`.

    Map to a simpler DataArray, with separate indices, non-index coords,
    name, and attributes.
    """
    # Prevent infinite recursion - see _strip_dataarray()
    if '__strip_dataarray__' in obj.attrs:
        return obj

    # Strip out the non-index coordinates and attributes
    return {
        'name': obj.name,
        'attrs': obj.attrs,
        # Index is handled separately, and created as a default
        # RangeIndex(shape[i]) if it doesn't exist, as it is compared
        # with outer join, whereas non-index coords and data are
        # compared with inner joinu
        'index': {
            k: obj.coords[k].to_index()
            for k in obj.dims
        },
        'coords': {
            k: _strip_dataarray(v, brief_dims=brief_dims)
            for k, v in obj.coords.items()
            if not isinstance(v.variable, xarray.IndexVariable)
        },
        'data': _strip_dataarray(obj, brief_dims=brief_dims)
    }


@cast.register(xarray.Dataset)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`xarray.Dataset`.

    Map to a dict of DataArrays.
    """
    return {
        'attrs': obj.attrs,
        # There may be coords, index or not, that are not
        # used in any data variable.
        # See above on why indices are handled separately
        'index': {
            k: obj.coords[k].to_index()
            for k in obj.dims
        },
        'coords': {
            k: _strip_dataarray(v, brief_dims=brief_dims)
            for k, v in obj.coords.items()
            if not isinstance(v.variable, xarray.IndexVariable)
        },
        'data_vars': {
            k: _strip_dataarray(v, brief_dims=brief_dims)
            for k, v in obj.data_vars.items()
        }
    }


@cast.register(pandas.MultiIndex)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.MultiIndex`.

    Map to a set of tuples. Note that this means that levels are
    positional. Using a set allows comparing the indices non-positionally.
    """
    return {
        'names': obj.names,
        'data': set(obj.tolist())
    }


@cast.register(pandas.RangeIndex)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.RangeIndex`.

    This function does nothing - RangeIndex objects are dealt with
    directly by :func:`_recursive_diff`. This function is defined
    to prevent RangeIndex objects to be processed by the more generic
    ``cast(obj: pandas.Index)`` below.
    """
    return obj


@cast.register(pandas.Index)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.Index`.

    Cast to a DataArray.

    .. note::
       :func:`~functools.singledispatch` always prefers a more specialised
       variant if available, so this function will not be called for
       :class:`pandas.MultiIndex` or :class:`pandas.RangeIndex`, as they have
       their own single dispatch variants.
    """
    return _strip_dataarray(xarray.DataArray(obj), brief_dims=brief_dims)


@cast.register(frozenset)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`frozenset`.

    Cast to a set.
    """
    return set(obj)


@cast.register(tuple)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`tuple`.

    Cast to a list.
    """
    return list(obj)


def _strip_dataarray(obj, brief_dims):
    """Helper function of :func:`recursive_diff`.

    Analyse a :class:`xarray.DataArray` and:

    - strip away any non-index coordinates (including scalar coords)
    - create stub coords for dimensions without coords
    - sort dimensions alphabetically
    - ravel the array to a 1D array with (potentially) a MultiIndex.
      brief_dims, if any, are excluded.

    :param obj:
        any xarray.DataArray
    :param brief_dims:
        sequence of dims, or "all"
    :returns:
        a stripped-down shallow copy of obj; otherwise None
    """
    res = obj.copy()

    # Remove non-index coordinates
    for k, v in obj.coords.items():
        if not isinstance(v.variable, xarray.IndexVariable):
            del res[k]

    # Ravel the array to make it become 1-dimensional.
    # To do this, we must first unstack any already stacked dimension.
    for dim in obj.dims:
        if isinstance(obj.get_index(dim), pandas.MultiIndex):
            res = proper_unstack(res, dim)

    # Transpose to ignore dimensions order
    res = res.transpose(*sorted(res.dims))

    # Finally stack everything back together
    if brief_dims != "all":
        stack_dims = sorted(set(res.dims) - set(brief_dims))
        if stack_dims:
            res = res.stack(__stacked__=stack_dims)

    # Prevent infinite recursion - see cast(obj: xarray.DataArray)
    res.attrs['__strip_dataarray__'] = True
    return res
