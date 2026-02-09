"""Compare either two files or all files in two directories.

See :doc:`cli`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Literal

try:
    import dask
except ImportError:
    dask = None  # type: ignore[assignment]

from recursive_diff.files import (
    DEFAULT_GLOB_PATTERNS,
    FORMATS,
    logger,
    recursive_open,
)
from recursive_diff.files import (
    open as open_,
)
from recursive_diff.recursive_diff import recursive_diff

LOGFORMAT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"


def argparser(cli_name: Literal["recursive-diff", "ncdiff"]) -> argparse.ArgumentParser:
    """Return precompiled ArgumentParser"""
    parser = argparse.ArgumentParser(
        description="Compare either two data files or all data files in "
        "two directories.",
        epilog="Examples:\n\n"
        "Compare two files:\n"
        f"  {cli_name} a.json b.json\n"
        "Compare all files with identical names in two "
        "directories:\n"
        f"  {cli_name} -r dir1 dir2\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress logging")

    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Compare all files with matching names in two directories",
    )

    if cli_name == "ncdiff":
        parser.add_argument(
            "--match",
            "-m",
            dest="patterns",
            nargs=1,
            metavar="PATTERN",
            default=["**/*.nc"],
            help="Bash wildcard pattern for file names when using --recursive "
            "(default: **/*.nc)",
        )
        parser.add_argument(
            "--format",
            choices=["netcdf"],
            default=None,
            help="File format.",
        )
    else:
        assert cli_name == "recursive-diff"
        parser.add_argument(
            "--match",
            "-m",
            dest="patterns",
            nargs="+",
            metavar="PATTERN",
            default=DEFAULT_GLOB_PATTERNS,
            help="Bash wildcard patterns for file names when using --recursive "
            f"(default: {' '.join(DEFAULT_GLOB_PATTERNS)})",
        )
        parser.add_argument(
            "--format",
            choices=FORMATS,
            default=None,
            help="File format (default: infer from file extension)",
        )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="Relative comparison tolerance (default: 1e-9)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0,
        help="Absolute comparison tolerance (default: 0)",
    )

    brief = parser.add_mutually_exclusive_group()
    brief.add_argument(
        "--brief_dims",
        nargs="+",
        default=(),
        metavar="DIM",
        help="Just count differences along one or more dimensions instead of "
        "printing them out individually",
    )
    brief.add_argument(
        "--brief",
        "-b",
        action="store_true",
        help="Just count differences for every variable instead of printing "
        "them out individually",
    )

    parser.add_argument(
        "--engine",
        "-e",
        dest="netcdf_engine",
        help="netCDF engine (default: first available)",
        choices=[
            "netcdf4",
            "h5netcdf",
            "scipy",
            "pydap",
        ],
    )

    parser.add_argument("lhs", help="Left-hand-side file or (if --recursive) directory")
    parser.add_argument(
        "rhs", help="Right-hand-side file or (if --recursive) directory"
    )

    return parser


def main(
    argv: list[str] | None = None,
    *,
    cli_name: Literal["recursive-diff", "ncdiff"] = "recursive-diff",
) -> int:
    """Parse command-line arguments, load all files, and invoke recursive_diff

    :param list[str] argv:
        List of command-line arguments (excluding the script name). If None,
        the arguments will be taken from sys.argv.
    :param str cli_name:
        DEPRECATED. Name of the CLI tool to implement. The new recursive-diff and the
        deprecated ncdiff differ only by command-line arguments.

    :returns:
        exit code
    """
    # Parse command-line arguments and init logging
    args = argparser(cli_name=cli_name).parse_args(argv)
    if args.brief:
        args.brief_dims = "all"

    if args.quiet:
        loglevel = logging.WARNING
    else:
        loglevel = logging.INFO

    # Don't init logging when running inside unit tests
    if argv is None:
        logging.basicConfig(level=loglevel, format=LOGFORMAT)  # pragma: nocover

    # Load all files. For netCDF and Zarr, this only loads metadata into RAM, but not
    # the actual data, regardless if Dask is installed or not. For other file formats,
    # this returns dask futures if Dask is installed; otherwise it eagerly loads
    # everything into RAM.
    lhs: object
    rhs: object
    kwargs = {
        "format": args.format,
        "chunks": "auto" if dask is not None else None,
        "netcdf_engine": args.netcdf_engine,
    }
    if args.recursive:
        lhs = recursive_open(args.lhs, args.patterns, **kwargs)
        rhs = recursive_open(args.rhs, args.patterns, **kwargs)
    else:
        lhs = open_(args.lhs, **kwargs)
        rhs = open_(args.rhs, **kwargs)

    logger.info("Comparing...")
    # In case of netCDF or Zarr:
    # 1. Load a pair of variables from lhs and rhs fully into RAM
    #    TODO: We could compare them chunk by chunk instead.
    # 2. compare them
    # 3. print all differences
    # 4. free the RAM
    # 5. proceed to next pair of variables, or to the next file
    # For all other file formats, if Dask is installed do the same file per file;
    # otherwise, recursive_open already loaded everything eagerly into RAM.
    diff_iter = recursive_diff(
        lhs, rhs, abs_tol=args.atol, rel_tol=args.rtol, brief_dims=args.brief_dims
    )

    diff_count = 0
    for diff in diff_iter:
        diff_count += 1
        print(diff)

    print(f"Found {diff_count} differences")
    if diff_count:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: nocover
