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

Memory design
-------------
If Dask is installed:
  For netCDF and Zarr files, the tool loads one pair of matching Dask chunks at a time
  into RAM, compares them, and then discards them. Dask chunks are automatically cut to
  128 MiB, or to the native netCDF/Zarr chunks on disk if they are larger than that.
  JSON, JSONL, YAML, and MessagePack files are loaded a pair of files at a time,
  compared, and then discarded. Chunking JSONL files is not supported. You may end up
  with as many pairs of chunks/files in RAM at once as there are CPUs available (or more
  if chunks are misaligned).

If Dask is not installed:
  The tool fully loads a pair of netCDF/Zarr variables into RAM at once, compares them,
  and then discards them. Native chunks are not used. JSON, JSONL, YAML, and MessagePack
  files are loaded all at once eagerly.

Limitations
-----------
- Doesn't compare netCDF/Zarr settings or metadata, e.g. store version, compression,
  chunking, etc.
- Doesn't support netCDF/Zarr indices with duplicate elements
- Treats netCDF datasets split across multiple files (typically created by Dask) as
  individual files. This can be slow, as there is no option to skip loading over and
  over again variables that don't sit on the ``concat_dim``. It also means that it can't
  compare two datasets that differ only by file chunking.
  See also `xarray#2039 <https://github.com/pydata/xarray/issues/2039>`_.
- Can't compare file sets not grouped by root directory, but by prefix
  (e.g. :file:`foo.*.json` vs. :file:`bar.*.json`).
