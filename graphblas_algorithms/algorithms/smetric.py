from graphblas import binary

__all__ = ["s_metric"]


def s_metric(G):
    if G.is_directed():
        degrees = G.get_property("total_degrees+")
    else:
        degrees = G.get_property("degrees+")
    return (binary.first(degrees & G._A) @ degrees).reduce().get(0) / 2
    # Alternatives
    # return (degrees @ binary.second(G._A & degrees)).reduce().get(0) / 2
    # return degrees.outer(degrees).new(mask=G._A.S).reduce_scalar().get(0) / 2
