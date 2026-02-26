import importlib.metadata

from recursive_diff.cast import cast
from recursive_diff.files import open, recursive_open
from recursive_diff.public import (
    diff_arrays,
    display_diffs,
    recursive_diff,
    recursive_eq,
)

try:
    __version__ = importlib.metadata.version("recursive_diff")
except importlib.metadata.PackageNotFoundError:  # pragma: nocover
    # Local copy, not installed with pip
    __version__ = "9999"

__all__ = (
    "__version__",
    "cast",
    "diff_arrays",
    "display_diffs",
    "open",
    "recursive_diff",
    "recursive_eq",
    "recursive_open",
)

# Prevent Intersphinx from pointing to the implementation modules
for name in __all__:
    if name != "__version__":
        globals()[name].__module__ = "recursive_diff"
    del name
