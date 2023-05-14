from ..exceptions import NoPath
from .unweighted import bidirectional_shortest_path_length

__all__ = ["has_path"]


def has_path(G, source, target):
    try:
        bidirectional_shortest_path_length(G, source, target)
    except NoPath:
        return False
    return True
