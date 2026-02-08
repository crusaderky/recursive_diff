import os

import pytest
import xarray

from recursive_diff.ncdiff import main
from recursive_diff.tests import (
    has_netcdf,
    requires_h5netcdf,
    requires_netcdf,
    requires_netcdf4,
    requires_scipy,
)

# Suppress warning emitted by NetCDF4 for all versions of NumPy
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:numpy.ndarry size changed:RuntimeWarning:netCDF4"
    )
]

a = xarray.Dataset(
    data_vars={
        "d1": ("x", [1, 2]),
        "d2": (("x", "y"), [[1.0, 1.1], [1.2, 1.3]]),
        "d3": ("y", [3.0, 4.0]),
    },
    coords={"x": [10, 20], "x2": ("x", [100, 200]), "y": ["y1", "y2"]},
    attrs={"a1": 1, "a2": 2},
)


def assert_stdout(capsys, expect):
    actual = capsys.readouterr().out
    print("Expect:")
    print(expect)
    print("Actual:")
    print(actual)
    assert expect == actual
    # Discard the print output above
    capsys.readouterr()


@requires_netcdf
@pytest.mark.parametrize(
    "argv",
    [
        ["d1/a.nc", "d1/b.nc"],
        ["-q", "d1/a.nc", "d1/b.nc"],
        ["-b", "d1/a.nc", "d1/b.nc"],
        ["-r", "d1", "d2"],
        ["-b", "-r", "d1", "d2"],
        ["-r", "-m", "*/a.nc", "--", "d1", "d2"],
        ["-r", "d1", "d2", "-m", "*/a.nc"],
    ],
)
def test_identical(tmpdir, capsys, argv):
    os.chdir(str(tmpdir))
    os.mkdir("d1")
    os.mkdir("d2")
    a.to_netcdf("d1/a.nc")
    a.to_netcdf("d1/b.nc")
    a.to_netcdf("d2/a.nc")
    a.to_netcdf("d2/b.nc")

    exit_code = main(argv)
    assert exit_code == 0
    assert_stdout(capsys, "Found 0 differences\n")


@requires_netcdf
@pytest.mark.parametrize(
    "argv,out",
    [
        (
            [],
            "[attrs]: Pair a3:4 is in RHS only\n"
            "[attrs][a1]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)\n"
            "[coords][x2][x=10]: 100 != 110 (abs: 1.0e+01, rel: 1.0e-01)\n"
            "[data_vars][d1][x=10]: 1 != 11 (abs: 1.0e+01, rel: 1.0e+01)\n"
            "[data_vars][d3][y=y1]: 3.0 != 3.01 (abs: 1.0e-02, rel: 3.3e-03)\n"
            "Found 5 differences\n",
        ),
        (
            ["-b"],
            "[attrs]: Pair a3:4 is in RHS only\n"
            "[attrs][a1]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)\n"
            "[coords][x2]: 1 differences\n"
            "[data_vars][d1]: 1 differences\n"
            "[data_vars][d3]: 1 differences\n"
            "Found 5 differences\n",
        ),
        (
            ["--brief_dims", "x", "--"],
            "[attrs]: Pair a3:4 is in RHS only\n"
            "[attrs][a1]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)\n"
            "[coords][x2]: 1 differences\n"
            "[data_vars][d1]: 1 differences\n"
            "[data_vars][d3][y=y1]: 3.0 != 3.01 (abs: 1.0e-02, rel: 3.3e-03)\n"
            "Found 5 differences\n",
        ),
        (
            ["--atol", "5"],
            "[attrs]: Pair a3:4 is in RHS only\n"
            "[coords][x2][x=10]: 100 != 110 (abs: 1.0e+01, rel: 1.0e-01)\n"
            "[data_vars][d1][x=10]: 1 != 11 (abs: 1.0e+01, rel: 1.0e+01)\n"
            "Found 3 differences\n",
        ),
        (
            ["--rtol", "1e-1"],
            "[attrs]: Pair a3:4 is in RHS only\n"
            "[attrs][a1]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)\n"
            "[data_vars][d1][x=10]: 1 != 11 (abs: 1.0e+01, rel: 1.0e+01)\n"
            "Found 3 differences\n",
        ),
    ],
)
def test_singlefile(tmpdir, capsys, argv, out):
    b = a.copy(deep=True)
    b.d1[0] += 10
    b.d3[0] += 0.01
    b.attrs["a1"] = 3
    b.attrs["a3"] = 4
    b.x2[0] += 10
    a.to_netcdf(f"{tmpdir}/a.nc")
    b.to_netcdf(f"{tmpdir}/b.nc")

    exit_code = main([*argv, f"{tmpdir}/a.nc", f"{tmpdir}/b.nc"])
    assert exit_code == 1
    assert_stdout(capsys, out)


@requires_netcdf
@pytest.mark.parametrize(
    "argv,out",
    [
        # no differences
        (["-r", "lhs", "lhs"], "Found 0 differences\n"),
        # pattern matches nothing
        (["-r", "lhs", "rhs", "-m", "notexist"], "Found 0 differences\n"),
        (
            # default pattern
            ["-r", "lhs", "rhs"],
            "[b.nc][data_vars][d1][x=30]: 1 != -9 (abs: -1.0e+01, rel: -1.0e+01)\n"
            "[" + os.path.join("subdir", "a.nc") + "][data_vars][d1][x=10]: 1 != -9 "
            "(abs: -1.0e+01, rel: -1.0e+01)\n"
            "Found 2 differences\n",
        ),
        (
            # multiple patterns
            ["-r", "lhs", "rhs", "-m", "**/a.nc", "*.nc"],
            "[b.nc][data_vars][d1][x=30]: 1 != -9 (abs: -1.0e+01, rel: -1.0e+01)\n"
            "[" + os.path.join("subdir", "a.nc") + "][data_vars][d1][x=10]: 1 != -9 "
            "(abs: -1.0e+01, rel: -1.0e+01)\n"
            "Found 2 differences\n",
        ),
        (
            # multiple overlapping patterns
            ["-r", "lhs", "rhs", "-m", "**/*.nc", "*.nc"],
            "[b.nc][data_vars][d1][x=30]: 1 != -9 (abs: -1.0e+01, rel: -1.0e+01)\n"
            "[" + os.path.join("subdir", "a.nc") + "][data_vars][d1][x=10]: 1 != -9 "
            "(abs: -1.0e+01, rel: -1.0e+01)\n"
            "Found 2 differences\n",
        ),
        (
            # pattern matches only b
            ["-r", "lhs", "rhs", "-m", "*.nc"],
            "[b.nc][data_vars][d1][x=30]: 1 != -9 (abs: -1.0e+01, rel: -1.0e+01)\n"
            "Found 1 differences\n",
        ),
        (
            # pattern matches only a
            ["-r", "lhs", "rhs", "-m", "**/a.nc"],
            "[" + os.path.join("subdir", "a.nc") + "][data_vars][d1][x=10]: 1 != -9 "
            "(abs: -1.0e+01, rel: -1.0e+01)\n"
            "Found 1 differences\n",
        ),
    ],
)
def test_recursive(tmpdir, capsys, argv, out):
    os.chdir(str(tmpdir))

    a_lhs = a
    b_lhs = a_lhs.copy(deep=True)
    b_lhs.coords["x"] = [30, 40]
    os.makedirs(f"{tmpdir}/lhs/subdir")
    a_lhs.to_netcdf(f"{tmpdir}/lhs/subdir/a.nc")
    b_lhs.to_netcdf(f"{tmpdir}/lhs/b.nc")

    a_rhs = a.copy(deep=True)
    a_rhs.d1[0] -= 10
    b_rhs = a_rhs.copy(deep=True)
    b_rhs.coords["x"] = [30, 40]
    os.makedirs("rhs/subdir")
    a_rhs.to_netcdf("rhs/subdir/a.nc")
    b_rhs.to_netcdf("rhs/b.nc")

    exit_code = main(argv)
    if out == "Found 0 differences\n":
        assert exit_code == 0
    else:
        assert exit_code == 1
    assert_stdout(capsys, out)


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param("netcdf4", marks=[requires_netcdf4]),
        pytest.param("h5netcdf", marks=[requires_h5netcdf]),
        pytest.param("scipy", marks=[requires_scipy]),
    ],
)
def test_engine(tmpdir, capsys, engine):
    """Test the --engine parameter"""
    os.chdir(str(tmpdir))
    b = a.copy(deep=True)
    b.d1[0] += 10
    a.to_netcdf("a.nc", engine=engine)
    b.to_netcdf("b.nc", engine=engine)

    exit_code = main(["--engine", engine, "a.nc", "b.nc"])
    assert exit_code == 1
    assert_stdout(
        capsys,
        "[data_vars][d1][x=10]: 1 != 11 (abs: 1.0e+01, rel: 1.0e+01)\n"
        "Found 1 differences\n",
    )


@pytest.mark.parametrize(
    "w_engine,r_engine",
    [
        pytest.param("scipy", "netcdf4", marks=[requires_scipy, requires_netcdf4]),
        pytest.param(
            "h5netcdf", "netcdf4", marks=[requires_h5netcdf, requires_netcdf4]
        ),
        pytest.param(
            "netcdf4", "h5netcdf", marks=[requires_netcdf4, requires_h5netcdf]
        ),
    ],
)
def test_cross_engine(tmpdir, w_engine, r_engine, capsys):
    """Test the --engine parameter vs. files written by another engine"""
    os.chdir(str(tmpdir))
    b = a.copy(deep=True)
    b.d1[0] += 10
    a.to_netcdf("a.nc", engine=w_engine)
    b.to_netcdf("b.nc", engine=w_engine)

    exit_code = main(["--engine", r_engine, "a.nc", "b.nc"])
    assert exit_code == 1
    assert_stdout(
        capsys,
        "[data_vars][d1][x=10]: 1 != 11 (abs: 1.0e+01, rel: 1.0e+01)\n"
        "Found 1 differences\n",
    )


@pytest.mark.parametrize(
    "w_engine",
    [
        pytest.param("netcdf4", marks=[requires_netcdf4]),
        pytest.param("h5netcdf", marks=[requires_h5netcdf]),
    ],
)
@requires_scipy
def test_cross_engine_scipy(tmpdir, w_engine):
    """Test the --engine scipy parameter vs. files written by NetCDF4 engines"""
    os.chdir(str(tmpdir))
    b = a.copy(deep=True)
    b.d1[0] += 10
    a.to_netcdf("a.nc", engine=w_engine)
    b.to_netcdf("b.nc", engine=w_engine)

    with pytest.raises(TypeError, match="not a valid NetCDF 3 file"):
        main(["--engine", "scipy", "a.nc", "b.nc"])


@requires_h5netcdf
def test_compression(tmpdir, capsys):
    os.chdir(str(tmpdir))
    b = a.copy(deep=True)
    b.d1[0] += 10
    a.to_netcdf("a.nc", engine="h5netcdf", encoding={"d1": {"compression": "lzf"}})
    b.to_netcdf("b.nc", engine="h5netcdf")

    # Differences in compression are not picked up
    exit_code = main(["a.nc", "b.nc", "--engine", "h5netcdf"])
    assert exit_code == 1
    assert_stdout(
        capsys,
        "[data_vars][d1][x=10]: 1 != 11 (abs: 1.0e+01, rel: 1.0e+01)\n"
        "Found 1 differences\n",
    )


@pytest.mark.skipif(has_netcdf, reason="Found a NetCDF engine")
def test_no_engine(tmpdir):
    os.chdir(str(tmpdir))
    open("a.nc", "w").close()
    open("b.nc", "w").close()

    with pytest.raises(ValueError, match="no currently installed IO backends"):
        main(["a.nc", "b.nc"])
