try:
    from .version import version as __version__  # noqa: F401
except ImportError:  # pragma: no cover
    raise ImportError('recursive_diff not properly installed. If you are '
                      'running from the source directory, please instead '
                      'create a new virtual environment (using conda or '
                      'virtualenv) and then install it in-place by running: '
                      'pip install -e .')


from .recursive_diff import recursive_diff  # noqa: F401
from .recursive_eq import recursive_eq  # noqa: F401
from .cast import cast  # noqa: F401
