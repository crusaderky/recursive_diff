import xarray
from xarray.testing import assert_equal


def test1(tmp_path):
    a = xarray.Dataset({"x": [1, 2]})
    fname = tmp_path / "test.nc"
    a.to_netcdf(fname)
    b = xarray.open_dataset(fname, engine="netcdf4", chunks=None)
    assert_equal(a, b)
