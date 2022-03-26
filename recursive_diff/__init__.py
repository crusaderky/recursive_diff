import importlib.metadata

from recursive_diff.cast import cast
from recursive_diff.recursive_diff import recursive_diff
from recursive_diff.recursive_eq import recursive_eq

try:
    __version__ = importlib.metadata.version("recursive_diff")
except importlib.metadata.PackageNotFoundError:  # pragma: nocover
    # Local copy, not installed with pip
    __version__ = "999"

# Prevent Intersphinx from pointing to the implementation modules
for obj in (recursive_diff, recursive_eq, cast):
    obj.__module__ = "recursive_diff"
del obj

__all__ = ("__version__", "recursive_diff", "recursive_eq", "cast")
