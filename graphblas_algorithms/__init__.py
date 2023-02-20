import importlib.metadata

from .classes import *

from .algorithms import *  # isort:skip

try:
    __version__ = importlib.metadata.version("graphblas-algorithms")
except Exception as exc:  # pragma: no cover (safety)
    raise AttributeError(
        "`graphblas_algorithms.__version__` not available. This may mean "
        "graphblas-algorithms was incorrectly installed or not installed at all. "
        "For local development, you may want to do an editable install via "
        "`python -m pip install -e path/to/graphblas-algorithms`"
    ) from exc
del importlib
