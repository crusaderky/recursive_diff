"""Support dask-backed Xarray objects, if dask is installed"""

try:
    from dask.array import Array
    from dask.delayed import Delayed

except ImportError:

    class Array:  # type: ignore[no-redef]
        pass

    class Delayed:  # type: ignore[no-redef]
        pass
