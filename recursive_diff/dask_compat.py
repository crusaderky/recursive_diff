"""Support dask-backed Xarray objects, if dask is installed"""

try:
    from dask.delayed import Delayed

except ImportError:

    class Delayed:  # type: ignore[no-redef]
        pass
