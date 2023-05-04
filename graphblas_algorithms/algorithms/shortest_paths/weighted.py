import numpy as np
from graphblas import Matrix, Vector, binary, indexunary, monoid, replace, select, unary
from graphblas.semiring import any_pair, min_plus

from .._bfs import _bfs_level, _bfs_levels, _bfs_parent, _bfs_plain
from ..exceptions import Unbounded

__all__ = [
    "single_source_bellman_ford_path_length",
    "bellman_ford_path",
    "bellman_ford_path_lengths",
    "negative_edge_cycle",
]


def single_source_bellman_ford_path_length(G, source, *, cutoff=None):
    # No need for `is_weighted=` keyword, b/c this is assumed to be weighted (I think)
    index = G._key_to_id[source]
    if G.get_property("is_iso"):
        # If the edges are iso-valued (and positive), then we can simply do level BFS
        is_negative, iso_value = G.get_properties("has_negative_edges+ iso_value")
        if not is_negative:
            if cutoff is not None:
                cutoff = int(cutoff // iso_value)
            d = _bfs_level(G, source, cutoff, dtype=iso_value.dtype)
            if iso_value != 1:
                d *= iso_value
            return d
        # It's difficult to detect negative cycles with BFS
        if G._A[index, index].get() is not None:
            raise Unbounded("Negative cycle detected.")
        if not G.is_directed() and G._A[index, :].nvals > 0:
            # For undirected graphs, any negative edge is a cycle
            raise Unbounded("Negative cycle detected.")

    # Use `offdiag` instead of `A`, b/c self-loops don't contribute to the result,
    # and negative self-loops are easy negative cycles to avoid.
    # We check if we hit a self-loop negative cycle at the end.
    A, has_negative_diagonal = G.get_properties("offdiag has_negative_diagonal")
    if A.dtype == bool:
        # Should we upcast e.g. INT8 to INT64 as well?
        dtype = int
    else:
        dtype = A.dtype
    n = A.nrows
    d = Vector(dtype, n, name="single_source_bellman_ford_path_length")
    d[index] = 0
    cur = d.dup(name="cur")
    mask = Vector(bool, n, name="mask")
    one = unary.one[bool]
    for _i in range(n - 1):
        # This is a slightly modified Bellman-Ford algorithm.
        # `cur` is the current frontier of values that improved in the previous iteration.
        # This means that in this iteration we drop values from `cur` that are not better.
        cur << min_plus(cur @ A)
        if cutoff is not None:
            cur << select.valuele(cur, cutoff)

        # Mask is True where cur not in d or cur < d
        mask << one(cur)
        mask(binary.second) << binary.lt(cur & d)

        # Drop values from `cur` that didn't improve
        cur(mask.V, replace) << cur
        if cur.nvals == 0:
            break
        # Update `d` with values that improved
        d(cur.S) << cur
    else:
        # Check for negative cycle when for loop completes without breaking
        cur << min_plus(cur @ A)
        if cutoff is not None:
            cur << select.valuele(cur, cutoff)
        mask << binary.lt(cur & d)
        if mask.reduce(monoid.lor):
            raise Unbounded("Negative cycle detected.")
    if has_negative_diagonal:
        # We removed diagonal entries above, so check if we visited one with a negative weight
        diag = G.get_property("diag")
        cur << select.valuelt(diag, 0)
        if any_pair(d @ cur):
            raise Unbounded("Negative cycle detected.")
    return d


def bellman_ford_path_lengths(G, nodes=None, *, expand_output=False):
    """Extra parameter: expand_output

    Parameters
    ----------
    expand_output : bool, default False
        When False, the returned Matrix has one row per node in nodes.
        When True, the returned Matrix has the same shape as the input Matrix.
    """
    # Same algorithms as in `single_source_bellman_ford_path_length`, but with
    # `Cur` as a Matrix with each row corresponding to a source node.
    if G.get_property("is_iso"):
        is_negative, iso_value = G.get_properties("has_negative_edges+ iso_value")
        if not is_negative:
            D = _bfs_levels(G, nodes, dtype=iso_value.dtype)
            if iso_value != 1:
                D *= iso_value
            if nodes is not None and expand_output and D.ncols != D.nrows:
                ids = G.list_to_ids(nodes)
                rv = Matrix(D.dtype, D.ncols, D.ncols, name=D.name)
                rv[ids, :] = D
                return rv
            return D
        if not G.is_directed():
            # For undirected graphs, any negative edge is a cycle
            if nodes is not None:
                ids = G.list_to_ids(nodes)
                if G._A[ids, :].nvals > 0:
                    raise Unbounded("Negative cycle detected.")
            elif G._A.nvals > 0:
                raise Unbounded("Negative cycle detected.")

    A, has_negative_diagonal = G.get_properties("offdiag has_negative_diagonal")
    if A.dtype == bool:
        dtype = int
    else:
        dtype = A.dtype
    n = A.nrows
    if nodes is None:
        # TODO: `D = Vector.from_scalar(0, n, dtype).diag()`
        D = Vector(dtype, n, name="bellman_ford_path_lengths_vector")
        D << 0
        D = D.diag(name="bellman_ford_path_lengths")
    else:
        ids = G.list_to_ids(nodes)
        D = Matrix.from_coo(
            np.arange(len(ids), dtype=np.uint64),
            ids,
            0,
            dtype,
            nrows=len(ids),
            ncols=n,
            name="bellman_ford_path_lengths",
        )
    Cur = D.dup(name="Cur")
    Mask = Matrix(bool, D.nrows, D.ncols, name="Mask")
    one = unary.one[bool]
    for _i in range(n - 1):
        Cur << min_plus(Cur @ A)
        Mask << one(Cur)
        Mask(binary.second) << binary.lt(Cur & D)
        Cur(Mask.V, replace) << Cur
        if Cur.nvals == 0:
            break
        D(Cur.S) << Cur
    else:
        Cur << min_plus(Cur @ A)
        Mask << binary.lt(Cur & D)
        if Mask.reduce_scalar(monoid.lor):
            raise Unbounded("Negative cycle detected.")
    if has_negative_diagonal:
        diag = G.get_property("diag")
        cur = select.valuelt(diag, 0)
        if any_pair(D @ cur).nvals > 0:
            raise Unbounded("Negative cycle detected.")
    if nodes is not None and expand_output and D.ncols != D.nrows:
        rv = Matrix(D.dtype, n, n, name=D.name)
        rv[ids, :] = D
        return rv
    return D


def _reconstruct_path_from_parents(G, parents, src, dst):
    indices, values = parents.to_coo(sort=False)
    d = dict(zip(indices.tolist(), values.tolist()))
    if dst not in d:
        return []
    cur = dst
    path = [cur]
    while cur != src:
        cur = d[cur]
        path.append(cur)
    return G.list_to_keys(reversed(path))


def bellman_ford_path(G, source, target):
    src_id = G._key_to_id[source]
    dst_id = G._key_to_id[target]
    if G.get_property("is_iso"):
        # If the edges are iso-valued (and positive), then we can simply do level BFS
        is_negative = G.get_property("has_negative_edges+")
        if not is_negative:
            p = _bfs_parent(G, source, target=target)
            return _reconstruct_path_from_parents(G, p, src_id, dst_id)
        raise Unbounded("Negative cycle detected.")
    A, is_negative, has_negative_diagonal = G.get_properties(
        "offdiag has_negative_edges- has_negative_diagonal"
    )
    if A.dtype == bool:
        # Should we upcast e.g. INT8 to INT64 as well?
        dtype = int
    else:
        dtype = A.dtype
    cutoff = None
    n = A.nrows
    d = Vector(dtype, n, name="bellman_ford_path_length")
    d[src_id] = 0
    p = Vector(int, n, name="bellman_ford_path_parent")
    p[src_id] = src_id

    prev = d.dup(name="prev")
    cur = Vector(dtype, n, name="cur")
    indices = Vector(int, n, name="indices")
    mask = Vector(bool, n, name="mask")
    B = Matrix(dtype, n, n, name="B")
    Indices = Matrix(int, n, n, name="Indices")
    cols = prev.to_coo(values=False)[0]
    one = unary.one[bool]
    for _i in range(n - 1):
        # This is a slightly modified Bellman-Ford algorithm.
        # `cur` is the current frontier of values that improved in the previous iteration.
        # This means that in this iteration we drop values from `cur` that are not better.
        cur << min_plus(prev @ A)
        if cutoff is not None:
            cur << select.valuele(cur, cutoff)

        # Mask is True where cur not in d or cur < d
        mask << one(cur)
        mask(binary.second) << binary.lt(cur & d)

        # Drop values from `cur` that didn't improve
        cur(mask.V, replace) << cur
        if cur.nvals == 0:
            break
        # Update `d` with values that improved
        d(cur.S) << cur
        if not is_negative:
            # Limit exploration if we have a target
            cutoff = cur.get(dst_id, cutoff)

        # Now try to find the parents!
        # This is also not standard. Typically, UDTs and UDFs are used to keep
        # track of both the minimum element and the parent id at the same time.
        # Only include rows and columns that were used this iteration.
        rows = cols
        cols = cur.to_coo(values=False)[0]
        B.clear()
        B[rows, cols] = A[rows, cols]

        # Reverse engineer to determine parent
        B << binary.plus(prev & B)
        B << binary.iseq(B & cur)
        B << select.valuene(B, False)
        Indices << indexunary.rowindex(B)
        indices << Indices.reduce_columnwise(monoid.min)
        p(indices.S) << indices
        prev, cur = cur, prev
    else:
        # Check for negative cycle when for loop completes without breaking
        cur << min_plus(prev @ A)
        if cutoff is not None:
            cur << select.valuele(cur, cutoff)
        mask << binary.lt(cur & d)
        if mask.get(dst_id):
            raise Unbounded("Negative cycle detected.")
    path = _reconstruct_path_from_parents(G, p, src_id, dst_id)
    if has_negative_diagonal and path:
        mask.clear()
        mask[G.list_to_ids(path)] = True
        diag = G.get_property("diag", mask=mask.S)
        if diag.nvals > 0:
            raise Unbounded("Negative cycle detected.")
        mask << binary.first(mask & cur)  # mask(cur.S, replace) << mask
        if mask.nvals > 0:
            # Is there a path from any visited node with negative self-loop to target?
            # We could actually stop as soon as any from `path` is visited
            indices, _ = mask.to_coo(values=False)[0]
            q = _bfs_plain(G, target=target, index=indices, cutoff=_i)
            if dst_id in q:
                raise Unbounded("Negative cycle detected.")
    return path


def negative_edge_cycle(G):
    # TODO: use a heuristic to try to stop early
    if G.is_directed():
        deg = "total_degrees-"
    else:
        deg = "degrees-"
    A, degrees, has_negative_diagonal, has_negative_edges = G.get_properties(
        f"offdiag {deg} has_negative_diagonal has_negative_edges-"
    )
    if has_negative_diagonal:
        return True
    if not has_negative_edges:
        return False
    if A.dtype == bool:
        # Should we upcast e.g. INT8 to INT64 as well?
        dtype = int
    else:
        dtype = A.dtype
    n = A.nrows
    # Begin from every node that has edges
    d = Vector(dtype, n, name="negative_edge_cycle")
    d(degrees.S) << 0
    cur = d.dup(name="cur")
    mask = Vector(bool, n, name="mask")
    one = unary.one[bool]
    for _i in range(n - 1):
        cur << min_plus(cur @ A)
        mask << one(cur)
        mask(binary.second) << binary.lt(cur & d)
        cur(mask.V, replace) << cur
        if cur.nvals == 0:
            return False
        d(cur.S) << cur
    cur << min_plus(cur @ A)
    mask << binary.lt(cur & d)
    if mask.reduce(monoid.lor):
        return True
    return False
