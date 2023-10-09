import warnings

from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["s_metric"]


def s_metric(G, **kwargs):
    if kwargs:
        if "normalized" in kwargs:
            warnings.warn(
                "\n\nThe `normalized` keyword is deprecated and will be removed\n"
                "in the future. To silence this warning, remove `normalized`\n"
                "when calling `s_metric`.\n\nThe value of `normalized` is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            raise TypeError(f"s_metric got an unexpected keyword argument '{kwargs.popitem()[0]}'")
    G = to_graph(G)
    return algorithms.s_metric(G)
