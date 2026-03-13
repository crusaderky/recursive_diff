This demonstrates a regression in libnetcdf which breaks coverage.

To run:
```bash
$ pixi run coverage
[...]
No source for code: '<[...]/recursive_diff>/src/netCDF4/_netCDF4.pyx'; see https://coverage.readthedocs.io/en/7.13.4/messages.html#error-no-source
$ echo $?
1
```

To make the issue disappear, uncomment `# libnetcdf = "<4.10.0"` in pixi.toml.
