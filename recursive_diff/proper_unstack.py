"""Copy-pasted from xarray-extras
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from typing import overload

import pandas
import xarray


@overload
def proper_unstack(array: xarray.DataArray, dim: Hashable) -> xarray.DataArray:
    ...  # pragma: nocover


@overload
def proper_unstack(array: xarray.Dataset, dim: Hashable) -> xarray.Dataset:
    ...  # pragma: nocover


def proper_unstack(array, dim):
    """Work around an issue in xarray that causes the data to be sorted
    alphabetically by label on unstack():

    `<https://github.com/pydata/xarray/issues/906>`_

    Also work around issue that causes string labels to be converted to
    objects:

    `<https://github.com/pydata/xarray/issues/907>`_

    :param array:
        xarray.DataArray or xarray.Dataset to unstack
    :param str dim:
        Name of existing dimension to unstack
    :returns:
        xarray.DataArray / xarray.Dataset with unstacked dimension
    """
    # Regenerate Pandas multi-index to be ordered by first appearance
    mindex = array.coords[dim].to_pandas().index

    levels = []
    labels = []
    for levels_i, labels_i in zip(mindex.levels, mindex.codes):
        level_map = OrderedDict()

        for label in labels_i:
            if label not in level_map:
                level_map[label] = len(level_map)

        levels.append([levels_i[k] for k in level_map.keys()])
        labels.append([level_map[k] for k in labels_i])

    mindex = pandas.MultiIndex(levels, labels, names=mindex.names)
    array = array.copy()
    array.coords[dim] = mindex

    # Invoke builtin unstack
    array = array.unstack(dim)

    # Convert numpy arrays of Python objects to numpy arrays of C floats, ints,
    # strings, etc.
    for dim in mindex.names:
        if array.coords[dim].dtype == object:
            array.coords[dim] = array.coords[dim].values.tolist()

    return array
