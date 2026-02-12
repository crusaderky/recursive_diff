import json
import os.path
from pathlib import Path

import pytest
import xarray
from xarray.testing import assert_equal

from recursive_diff import open as rdopen
from recursive_diff import recursive_open
from recursive_diff.files import _infer_format_from_extension
from recursive_diff.tests import (
    HAS_ZARR_V3,
    TO_ZARR_V2,
    TO_ZARR_V3,
    requires_dask,
    requires_h5netcdf,
    requires_netcdf,
    requires_scipy,
    requires_zarr,
)


@pytest.fixture(params=[str, Path])
def path_type(request):
    return request.param


@pytest.fixture(
    params=[
        None,
        pytest.param("auto", marks=requires_dask),
        pytest.param(-1, marks=requires_dask),
        pytest.param({}, marks=requires_dask),
    ]
)
def chunks(request):
    return request.param


def test_open_json(tmp_path, path_type, chunks):
    a = {"foo": "bar", "baz": [1, 2, 3]}
    fname = path_type(tmp_path / "test.json")
    with open(fname, "w") as f:
        json.dump(a, f)
    b = rdopen(fname, chunks=chunks)
    if chunks is not None:
        b = b.compute()
    assert b == a


def test_open_jsonl(tmp_path, path_type, chunks):
    a = [{"foo": "bar"}, {"baz": [1, 2, 3]}]
    fname = path_type(tmp_path / "test.jsonl")
    with open(fname, "w") as f:
        for line in a:
            f.write(json.dumps(line) + "\n")
    b = rdopen(fname, chunks=chunks)
    if chunks is not None:
        b = b.compute()
    assert b == a


def test_open_yaml(tmp_path, path_type, chunks):
    yaml = pytest.importorskip("yaml")

    a = {"foo": "bar", "baz": [1, 2, 3]}
    fname = path_type(tmp_path / "test.yaml")
    with open(fname, "w") as f:
        yaml.dump(a, f)
    b = rdopen(fname, chunks=chunks)
    if chunks is not None:
        b = b.compute()
    assert b == a


def test_open_msgpack(tmp_path, path_type, chunks):
    msgpack = pytest.importorskip("msgpack")

    a = {"foo": "bar", "baz": [1, 2, 3]}
    fname = path_type(tmp_path / "test.msgpack")
    with open(fname, "wb") as f:
        msgpack.dump(a, f)
    b = rdopen(fname, chunks=chunks)
    if chunks is not None:
        b = b.compute()
    assert b == a


@requires_netcdf
def test_open_netcdf(tmp_path, path_type, chunks):
    a = xarray.Dataset({"v1": (["x", "y"], [[1, 2], [3, 4]])}, coords={"x": ["a", "b"]})
    fname = path_type(tmp_path / "test.nc")
    a.to_netcdf(fname)
    b = rdopen(fname, chunks=chunks)
    assert (b.__dask_graph__() is None) is (chunks is None)
    assert_equal(a, b)


@requires_netcdf
@requires_scipy
@requires_h5netcdf
def test_open_netcdf_engine(tmp_path):
    a = xarray.Dataset({"v1": (["x", "y"], [[1, 2], [3, 4]])}, coords={"x": ["a", "b"]})
    a.to_netcdf(tmp_path / "v3.nc", engine="scipy")  # Save as NetCDF v3
    a.to_netcdf(tmp_path / "v4.nc", engine="h5netcdf")  # Save as NetCDF v4

    b = rdopen(tmp_path / "v3.nc", netcdf_engine="scipy")
    assert_equal(a, b)
    c = rdopen(tmp_path / "v4.nc", netcdf_engine="h5netcdf")
    assert_equal(a, c)
    with pytest.raises(TypeError, match="not a valid NetCDF 3 file"):
        rdopen(tmp_path / "v4.nc", netcdf_engine="scipy")


@requires_zarr
@pytest.mark.filterwarnings(
    "ignore:Consolidated metadata is currently not part in the "
    "Zarr format 3 specification"
)
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(TO_ZARR_V2, id="v2"),
        pytest.param(
            TO_ZARR_V3,
            id="v3",
            marks=pytest.mark.skipif(
                not HAS_ZARR_V3, reason="Requires recent Xarray version"
            ),
        ),
    ],
)
def test_open_zarr(tmp_path, path_type, kwargs, chunks):
    a = xarray.Dataset({"v1": (["x", "y"], [[1, 2], [3, 4]])}, coords={"x": [10, 20]})
    fname = path_type(tmp_path / "test.zarr")
    a.to_zarr(fname, **kwargs)
    b = rdopen(fname, chunks=chunks)
    assert (b.__dask_graph__() is None) is (chunks is None)
    assert_equal(a, b)
    # Quietly ignore netcdf_engine parameter
    # Do not pass it to xarray.open_dataset(..., engine=...)
    b = rdopen(fname, netcdf_engine="scipy")
    assert_equal(a, b)


def test_infer_format():
    assert _infer_format_from_extension("foo.json") == "json"
    assert _infer_format_from_extension("foo.JSON") == "json"
    assert _infer_format_from_extension("/a/b/../c/foo.bar.JsoN") == "json"
    assert _infer_format_from_extension("foo.jsonl") == "jsonl"
    assert _infer_format_from_extension("foo.yaml") == "yaml"
    assert _infer_format_from_extension("foo.yml") == "yml"
    assert _infer_format_from_extension("foo.msgpack") == "msgpack"
    assert _infer_format_from_extension("foo.nc") == "nc"
    assert _infer_format_from_extension("foo.netcdf") == "netcdf"
    assert _infer_format_from_extension("foo.zarr") == "zarr"

    with pytest.raises(ValueError, match="Could not infer file format"):
        _infer_format_from_extension("foo")
    with pytest.raises(ValueError, match="Could not infer file format"):
        _infer_format_from_extension("foo.json.gz")


def test_open_force_format(tmp_path):
    # File is jsonl, but extension is incorrectly .json
    a = [{"foo": "bar"}, {"baz": [1, 2, 3]}]
    fname = tmp_path / "test.json"
    with open(fname, "w") as f:
        for line in a:
            f.write(json.dumps(line) + "\n")
    with pytest.raises(ValueError, match="Extra data"):
        rdopen(fname)

    b = rdopen(fname, format="jsonl")
    assert b == a

    with pytest.raises(ValueError, match="Unknown format"):
        rdopen(fname, format="unk")


def test_recursive_open(tmp_path, path_type):
    a = {"foo": "bar", "baz": [1, 2, 3]}
    with open(tmp_path / "a.json", "w") as f:
        json.dump(a, f)

    (tmp_path / "subdir").mkdir()
    b = [{"foo": "bar"}, {"baz": [1, 2, 3]}]
    with open(tmp_path / "subdir" / "b.jsonl", "w") as f:
        for line in b:
            f.write(json.dumps(line) + "\n")

    # Unsupported file extensions are ignored by the default glob pattern
    with open(tmp_path / "c.txt", "w") as f:
        f.write("This file should be ignored")

    actual = recursive_open(path_type(tmp_path))
    assert actual == {"a.json": a, os.path.join("subdir", "b.jsonl"): b}

    actual = recursive_open(tmp_path, "*.json")
    assert actual == {"a.json": a}

    actual = recursive_open(tmp_path, ["*.json"])
    assert actual == {"a.json": a}

    actual = recursive_open(tmp_path, "sub*/*")
    assert actual == {os.path.join("subdir", "b.jsonl"): b}

    actual = recursive_open(tmp_path, ["a*", "**/b.*"])
    assert actual == {"a.json": a, os.path.join("subdir", "b.jsonl"): b}


def test_recursive_open_force_format(tmp_path):
    a = [{"foo": "bar"}, {"baz": [1, 2, 3]}]
    # actually a jsonl file, but with .json extension
    with open(tmp_path / "a.json", "w") as f:
        for line in a:
            f.write(json.dumps(line) + "\n")

    with pytest.raises(ValueError, match="Extra data"):
        recursive_open(tmp_path)
    actual = recursive_open(tmp_path, format="jsonl")
    assert actual == {"a.json": a}


def test_recursive_open_chunks(tmp_path, chunks):
    a = {"foo": "bar", "baz": [1, 2, 3]}
    with open(tmp_path / "a.json", "w") as f:
        json.dump(a, f)
    actual = recursive_open(tmp_path, chunks=chunks)
    assert actual.keys() == {"a.json"}
    b = actual["a.json"]
    if chunks is not None:
        b = b.compute()
    assert b == a


@requires_h5netcdf
@requires_scipy
@requires_zarr
def test_recursive_open_netcdf_engine(tmp_path):
    a = xarray.Dataset({"v1": (["x", "y"], [[1, 2], [3, 4]])}, coords={"x": ["a", "b"]})
    a.to_netcdf(tmp_path / "netcdf_v4.nc", engine="h5netcdf")  # Save as NetCDF v4
    a.to_zarr(tmp_path / "zarr_v2.zarr", **TO_ZARR_V2)

    # Test that netcdf_engine is ignored for zarr files
    actual = recursive_open(tmp_path, netcdf_engine="h5netcdf")
    assert actual.keys() == {"netcdf_v4.nc", "zarr_v2.zarr"}
    assert_equal(actual["netcdf_v4.nc"], a)
    assert_equal(actual["zarr_v2.zarr"], a)

    with pytest.raises(TypeError, match="not a valid NetCDF 3 file"):
        recursive_open(tmp_path, netcdf_engine="scipy")


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Thread-unsafe on Python <3.10")
def test_recursive_open_threadsafe(tmp_path):
    """Test that recursive_open does not tamper with the cwd.
    This test relies on pytest-run-parallel to detect race conditions.
    """
    cwd = os.getcwd()
    assert cwd != str(tmp_path)
    lhs = recursive_open(tmp_path)
    assert cwd == os.getcwd()

