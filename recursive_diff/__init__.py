import pkg_resources

from .cast import cast
from .recursive_diff import recursive_diff
from .recursive_eq import recursive_eq

try:
    __version__ = pkg_resources.get_distribution("recursive_diff").version
except Exception:  # pragma: nocover
    # Local copy, not installed with setuptools
    __version__ = "999"


__all__ = ("__version__", "recursive_diff", "recursive_eq", "cast")
