"""Support dask-backed Xarray objects, if dask is installed"""

try:
    from dask import compute
    from dask.delayed import Delayed

    def is_dask_delayed(x: object) -> bool:
        return isinstance(x, Delayed)

    def is_dask_collection(x: object) -> bool:
        f = getattr(x, "__dask_keys__", None)
        return bool(f()) if f is not None else False


except ImportError:

    def compute(*args: object) -> object:  # type: ignore[misc]
        return args

    def is_dask_delayed(_: object) -> bool:  # type: ignore[misc]
        return False

    def is_dask_collection(_: object) -> bool:  # type: ignore[misc]
        return False
