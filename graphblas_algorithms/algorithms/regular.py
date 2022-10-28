from graphblas import monoid

__all__ = ["is_regular", "is_k_regular"]


def is_regular(G):
    if not G.is_directed():
        degrees = G.get_property("degrees+")
        if degrees.nvals != degrees.size:
            return False
        d = degrees.get(0)
        return (degrees == d).reduce(monoid.land).get(True)
    else:
        row_degrees = G.get_property("row_degrees+")
        if row_degrees.nvals != row_degrees.size:
            return False
        column_degrees = G.get_property("column_degrees+")
        if column_degrees.nvals != column_degrees.size:
            return False
        d = row_degrees.get(0)
        if not (row_degrees == d).reduce(monoid.land):
            return False
        d = column_degrees.get(0)
        return (column_degrees == d).reduce(monoid.land).get(True)


def is_k_regular(G, k):
    degrees = G.get_property("degrees+")
    if degrees.nvals != degrees.size:
        return False
    return (degrees == k).reduce(monoid.land).get(True)
