"""Support dask-backed Xarray objects, if dask is installed"""

try:
    from dask.delayed import Delayed

    def is_dask_collection(x: object) -> bool:
        f = getattr(x, "__dask_keys__", None)
        return bool(f()) if f is not None else False

except ImportError:

    class Delayed:  # type: ignore[no-redef]
        pass

    def is_dask_collection(_: object) -> bool:  # type: ignore[misc]
        return False
