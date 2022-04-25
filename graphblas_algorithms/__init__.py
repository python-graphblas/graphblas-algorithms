from . import _version
from .cluster import triangles  # noqa
from .link_analysis import pagerank  # noqa

__version__ = _version.get_versions()["version"]
