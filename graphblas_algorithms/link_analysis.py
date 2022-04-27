from warnings import warn

import networkx as nx
from graphblas import Vector, binary, unary
from graphblas.semiring import plus_first, plus_times

from ._utils import dict_to_vector, graph_to_adjacency, vector_to_dict


def pagerank_core(
    A,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1e-06,
    nstart=None,
    dangling=None,
    row_degrees=None,
    name="pagerank",
):
    N = A.nrows
    if A.nvals == 0:
        return Vector.new(float, N, name=name)

    # Initial vector
    x = Vector.new(float, N, name="x")
    if nstart is None:
        x[:] = 1.0 / N
    else:
        denom = nstart.reduce(allow_empty=False).value
        if denom == 0:
            raise ZeroDivisionError()
        x << nstart / denom

    # Personalization vector or scalar
    if personalization is None:
        p = 1.0 / N
    else:
        denom = personalization.reduce(allow_empty=False).value
        if denom == 0:
            raise ZeroDivisionError()
        p = (personalization / denom).new(name="p")

    # Inverse of row_degrees
    # Fold alpha constant into S
    if row_degrees is None:
        S = A.reduce_rowwise().new(float, name="S")  # XXX: What about self-edges
        S << alpha / S
    else:
        S = (alpha / row_degrees).new(name="S")

    if A.ss.is_iso:
        # Fold iso-value of A into S
        # This lets us use the plus_first semiring, which is faster
        iso_value = A.ss.iso_value
        if iso_value != 1:
            S *= iso_value
        semiring = plus_first[float]
    else:
        semiring = plus_times[float]

    is_dangling = S.nvals < N
    if is_dangling:
        dangling_mask = Vector.new(float, N, name="dangling_mask")
        dangling_mask(mask=~S.S) << 1.0
        # Fold alpha constant into dangling_weights (or dangling_mask)
        if dangling is not None:
            dangling_weights = (alpha / dangling.reduce(allow_empty=False).value * dangling).new(
                name="dangling_weights"
            )
        elif personalization is None:
            # Fast case (and common case); is iso-valued
            dangling_mask(mask=dangling_mask.S) << alpha * p
        else:
            dangling_weights = (alpha * p).new(name="dangling_weights")

    # Fold constant into p
    p *= 1 - alpha

    # Power iteration: make up to max_iter iterations
    xprev = Vector.new(float, N, name="x_prev")
    w = Vector.new(float, N, name="w")
    for _ in range(max_iter):
        xprev, x = x, xprev

        # x << alpha * ((xprev * S) @ A + "dangling_weights") + (1 - alpha) * p
        x << p
        if is_dangling:
            if dangling is None and personalization is None:
                # Fast case: add a scalar; x is still iso-valued (b/c p is also scalar)
                x += xprev @ dangling_mask
            else:
                # Add a vector
                x += plus_first(xprev @ dangling_mask) * dangling_weights
        w << xprev * S
        x += semiring(w @ A)  # plus_first if A.ss.is_iso else plus_times

        # Check convergence, l1 norm: err = sum(abs(xprev - x))
        xprev << binary.minus(xprev | x, require_monoid=False)
        xprev << unary.abs(xprev)
        err = xprev.reduce().value
        if err < N * tol:
            x.name = name
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)


def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1e-06,
    nstart=None,
    weight="weight",
    dangling=None,
):
    warn("", DeprecationWarning, stacklevel=2)
    N = len(G)
    if N == 0:
        return {}
    A, key_to_id = graph_to_adjacency(G, weight=weight, dtype=float)
    # We'll normalize initial, personalization, and dangling vectors later
    x = dict_to_vector(nstart, key_to_id, dtype=float, name="nstart")
    p = dict_to_vector(personalization, key_to_id, dtype=float, name="personalization")
    row_degrees = A.reduce_rowwise().new(name="row_degrees")  # XXX: What about self-edges?
    if dangling is not None and row_degrees.nvals < N:
        dangling_weights = dict_to_vector(dangling, key_to_id, dtype=float, name="dangling")
    else:
        dangling_weights = None
    result = pagerank_core(
        A,
        alpha=alpha,
        personalization=p,
        max_iter=max_iter,
        tol=tol,
        nstart=x,
        dangling=dangling_weights,
        row_degrees=row_degrees,
    )
    return vector_to_dict(result, key_to_id, fillvalue=0.0)
