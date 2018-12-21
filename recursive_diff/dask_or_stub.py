"""Support dask-backed xarray objects, if dask is installed
"""

try:
    from dask import compute
except ImportError:
    def compute(*args):
        return args
