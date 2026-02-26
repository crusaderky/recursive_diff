import pytest

from recursive_diff.tests import requires_dask


@pytest.fixture(params=[False, pytest.param(True, marks=requires_dask)])
def chunk(request):
    return request.param
