CLI tool
========
Compare either two files or all files in two directories.
For supported file formats, see :func:`~recursive_diff.open`.

Usage
-----
::

  usage: recursive-diff [-h] [--quiet]
                        [--recursive] [--match PATTERN [PATTERN ...]]
                        [--format {json,jsonl,msgpack,yaml,yml,netcdf,nc,zarr}]
                        [--rtol RTOL] [--atol ATOL]
                        [--brief_dims DIM [DIM ...] | --brief]
                        [--engine {netcdf4,h5netcdf,scipy,pydap}]
                        lhs rhs

  Compare either two data files or all data files in two directories.

  positional arguments:
    lhs                   Left-hand-side file or (if --recursive) directory
    rhs                   Right-hand-side file or (if --recursive) directory

  options:
    -h, --help            show this help message and exit
    --quiet, -q           Suppress logging
    --recursive, -r       Compare all files with matching names in two directories
    --match, -m PATTERN [PATTERN ...]
                          Bash wildcard patterns for file names when using --recursive
                          (default: **/*.json **/*.jsonl **/*.msgpack **/*.yaml **/*.yml
                          **/*.nc **/*.zarr)
    --format {json,jsonl,msgpack,yaml,yml,netcdf,nc,zarr}
                          File format (default: infer from file extension)
    --rtol RTOL           Relative comparison tolerance (default: 1e-9)
    --atol ATOL           Absolute comparison tolerance (default: 0)
    --brief_dims DIM [DIM ...]
                          Just count differences along one or more dimensions instead of
                          printing them out individually
    --brief, -b           Just count differences for every variable instead of printing
                          them out individually
    --engine, -e {netcdf4,h5netcdf,scipy,pydap}
                          netCDF engine (default: first available)

  Examples:

  Compare two files:
    recursive-diff a.json b.json
  Compare all files with identical names in two directories:
    recursive-diff -r dir1 dir2

Chunking and RAM design
-----------------------
For netCDF and Zarr files, the tool completely loads one variable at a time into RAM,
compares it, and then discards it. Chunks are not used.
JSON, JSONL, YAML, and MessagePack files are loaded completely into RAM all at once.

This has the advantage of simplicity, but the disadvantage of potentially very intensive
RAM usage.

Further limitations
-------------------
- Doesn't compare netCDF/Zarr settings or metadata, e.g. store version, compression,
  chunking, etc.
- Doesn't support netCDF/Zarr indices with duplicate elements
- Treats chunked netCDF datasets (split across multiple files) as individual files. This
  can be slow, as there is no option to skip loading over and over again variables that
  don't sit on the ``concat_dim``. It also means that it can't compare two datasets that
  differ only by file chunking.
  See also `xarray#2039 <https://github.com/pydata/xarray/issues/2039>`_.
- Can't compare file sets not grouped by root directory, but by prefix
  (e.g. :file:`foo.*.json` vs. :file:`bar.*.json`).
