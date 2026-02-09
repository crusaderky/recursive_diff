from __future__ import annotations

import builtins
import glob
import json
import logging
import os
import pathlib
import sys
from collections.abc import Collection
from typing import Any, Literal

import xarray

FORMATS = ("json", "jsonl", "msgpack", "yaml", "yml", "netcdf", "nc", "zarr")
Format = Literal["json", "jsonl", "msgpack", "yaml", "yml", "netcdf", "nc", "zarr"]
DEFAULT_GLOB_PATTERNS = (
    "**/*.json",
    "**/*.jsonl",
    "**/*.msgpack",
    "**/*.yaml",
    "**/*.yml",
    "**/*.nc",
    "**/*.zarr",
)

logger = logging.getLogger("recursive-diff")


def _infer_format_from_extension(fname: str | pathlib.Path) -> Format:
    ext = str(fname).rsplit(".", 1)[-1].lower()
    if ext in FORMATS:
        return ext  # type: ignore[return-value]
    raise ValueError(f"Could not infer file format from extension of {fname!r}")


def open(
    fname: str | pathlib.Path,
    *,
    format: Format | None = None,
    netcdf_engine: str | None = None,
) -> Any:
    """Open a single file from disk and return it as a recursively comparable object.

    Supported file formats:

    - JSON (.json)
    - JSON Lines (.jsonl)
    - MessagePack (.msgpack)
    - YAML (.yaml, .yml)
    - netCDF v3/v4 (.nc, .netcdf)
    - Zarr v2/v3 (.zarr)

    Different file formats require additional dependencies; see
    :ref:`optional_dependencies`.

    For netCDF and Zarr files, this function reads the metadata into RAM; loading the
    actual data is delayed until later with Dask (typically until you feed the output of
    this function to :func:`~recursive_diff.recursive_diff` or
    :func:`~recursive_diff.recursive_eq`). Other file formats are eagerly fully loaded
    into RAM.

    :param str | pathlib.Path fname:
        path to file
    :param str format:
        File format. Default: infer from file extension.
    :param str netcdf_engine:
        netCDF engine (see :func:`xarray.open_dataset`). Ignored for other file formats.
        Default: use Xarray default depending on what is available.
    :returns:
        :class:`xarray.Dataset` for netCDF and Zarr files; otherwise a plain-Python
        object, which can be passed as either the ``lhs`` or ``rhs`` argument to
        :func:`~recursive_diff.recursive_diff` or :func:`~recursive_diff.recursive_eq`.
    """
    logger.info("Opening %s", fname)
    if format is None:
        format = _infer_format_from_extension(fname)

    if format == "json":
        with builtins.open(fname) as f:
            return json.load(f)

    if format == "jsonl":
        with builtins.open(fname) as f:
            return [json.loads(line) for line in f]

    if format in ("yaml", "yml"):
        import yaml  # noqa: PLC0415

        with builtins.open(fname) as f:
            return yaml.safe_load(f)

    if format == "msgpack":
        import msgpack  # noqa: PLC0415

        with builtins.open(fname, "rb") as f:
            return msgpack.unpack(f)

    if format in ("nc", "netcdf"):
        return xarray.open_dataset(fname, engine=netcdf_engine, chunks={})

    if format == "zarr":
        return xarray.open_dataset(fname, engine="zarr", chunks={})

    raise ValueError(
        f"Unknown format={format}. "
        "Expected one of: json, jsonl, yaml, yml, msgpack, netcdf, nc, zarr"
    )


def recursive_open(
    path: str,
    patterns: str | Collection[str] = DEFAULT_GLOB_PATTERNS,
    *,
    format: Format | None = None,
    netcdf_engine: str | None = None,
) -> dict[str, Any]:
    """Recursively find and open all supported files that exist in any of
    the given local paths. See :func:`open` for supported file formats.

    :param str path:
        Root directory to search into
    :param str | list[str] patterns:
        One or more glob patterns relative to path
    :param str format:
        File format. Default: infer from file extension.
    :param str netcdf_engine:
        netCDF engine (see :func:`xarray.open_dataset`). Ignored for other file formats.
        Default: use Xarray default depending on what is available.
    :returns:
        dict of {file name relative to *path*: file contents}, which can be passed as
        either the ``lhs`` or ``rhs`` argument to :func:`~recursive_diff.recursive_diff`
        or :func:`~recursive_diff.recursive_eq`.

    **Thread-safety note:** this function is not thread-safe on Python 3.10.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    fnames = set()
    if sys.version_info >= (3, 10):
        # Thread-safe
        for pattern in patterns:
            fnames.update(glob.glob(pattern, root_dir=path, recursive=True))
    else:
        cwd = os.getcwd()
        os.chdir(path)
        try:
            for pattern in patterns:
                fnames.update(glob.glob(pattern, recursive=True))
        finally:
            os.chdir(cwd)

    logger.info("Opening %d files from %s", len(fnames), path)
    return {
        fname: open(
            os.path.join(path, fname), format=format, netcdf_engine=netcdf_engine
        )
        for fname in sorted(fnames)
    }
