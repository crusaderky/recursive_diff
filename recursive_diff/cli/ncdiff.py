"""Deprecated command-line tool. Please use recursive-diff instead."""

from __future__ import annotations

import sys
import warnings

from .recursive_diff import main as recursive_diff_main


def main(argv: list[str] | None = None) -> int:
    warnings.warn(
        "ncdiff is deprecated. Please use recursive-diff instead.",
        FutureWarning,
        stacklevel=2,
    )
    return recursive_diff_main(argv, cli_name="ncdiff")


if __name__ == "__main__":
    sys.exit(main())  # pragma: nocover
