"""Support dask-backed Xarray objects, if dask is installed"""

try:
    from dask import compute
    from dask.delayed import Delayed

    def is_delayed(x: object) -> bool:
        return isinstance(x, Delayed)

except ImportError:

    def compute(*args: object) -> object:  # type: ignore[misc]
        return args

    def is_delayed(_: object) -> bool:  # type: ignore[misc]
        return False
