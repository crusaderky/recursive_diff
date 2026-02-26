"""Support dask-backed Xarray objects, if dask is installed"""

from typing import Any

try:
    import dask
    from dask.array import Array
    from dask.delayed import Delayed

    def compute(*args: Any) -> tuple[Any, ...]:
        """Override default scheduler for Delayed, which would be multiprocessing.
        We use Delayed to post-process the comparison of Dask-backed Xarrays.

        This has the disadvantage that JSON, JSONL, YAML, and MessagePack files
        comparison contends for the GIL. Eventually this problem will go away once
        free-threading becomes the norm.
        """
        scheduler = dask.config.get("scheduler", "threads")
        return dask.compute(*args, scheduler=scheduler)

except ModuleNotFoundError:

    class Array:  # type: ignore[no-redef]
        pass

    class Delayed:  # type: ignore[no-redef]
        pass

    def compute(*args: Any) -> tuple[Any, ...]:
        return args
