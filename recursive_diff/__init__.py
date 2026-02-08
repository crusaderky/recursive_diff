import importlib.metadata

from recursive_diff.cast import cast
from recursive_diff.files import open, recursive_open
from recursive_diff.recursive_diff import recursive_diff
from recursive_diff.recursive_eq import recursive_eq

try:
    __version__ = importlib.metadata.version("recursive_diff")
except importlib.metadata.PackageNotFoundError:  # pragma: nocover
    # Local copy, not installed with pip
    __version__ = "9999"

# Prevent Intersphinx from pointing to the implementation modules
for obj in (open, recursive_open, recursive_diff, recursive_eq, cast):
    obj.__module__ = "recursive_diff"
del obj

__all__ = (
    "__version__",
    "cast",
    "open",
    "recursive_diff",
    "recursive_eq",
    "recursive_open",
)
