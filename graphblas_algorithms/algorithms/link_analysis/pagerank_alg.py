import numpy as np
from graphblas import Matrix, Vector, binary, monoid
from graphblas.semiring import plus_first, plus_times

from graphblas_algorithms import Graph

from .._helpers import is_converged
from ..exceptions import ConvergenceFailure

__all__ = ["pagerank", "google_matrix"]


def pagerank(
    G: Graph,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1e-06,
    nstart=None,
    dangling=None,
    row_degrees=None,
    name="pagerank",
) -> Vector:
    A = G._A
    N = A.nrows
    if A.nvals == 0:
        return Vector(float, N, name=name)

    # Initial vector
    x = Vector(float, N, name="x")
    if nstart is None:
        x[:] = 1.0 / N
    else:
        denom = nstart.reduce().get(0)
        if denom == 0:
            raise ZeroDivisionError("nstart sums to 0")
        x << nstart / denom

    # Personalization vector or scalar
    if personalization is None:
        p = 1.0 / N
    else:
        denom = personalization.reduce().get(0)
        if denom == 0:
            raise ZeroDivisionError("personalization sums to 0")
        p = (personalization / denom).new(name="p")

    # Inverse of row_degrees
    # Fold alpha constant into S
    if row_degrees is None:
        row_degrees = G.get_property("plus_rowwise+")  # XXX: What about self-edges?
    S = (alpha / row_degrees).new(name="S")

    if (iso_value := G.get_property("iso_value")) is not None:
        # Fold iso-value of A into S
        # This lets us use the plus_first semiring, which is faster
        if iso_value.get(1) != 1:
            S *= iso_value
        semiring = plus_first[float]
    else:
        semiring = plus_times[float]

    is_dangling = S.nvals < N
    if is_dangling:
        dangling_mask = Vector(float, N, name="dangling_mask")
        dangling_mask(mask=~S.S) << 1.0
        # Fold alpha constant into dangling_weights (or dangling_mask)
        if dangling is not None:
            dangling_weights = (alpha / dangling.reduce().get(0) * dangling).new(
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
    xprev = Vector(float, N, name="x_prev")
    w = Vector(float, N, name="w")
    for _i in range(max_iter):
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

        if is_converged(xprev, x, tol):  # sum(abs(xprev - x)) < N * tol
            x.name = name
            return x
    raise ConvergenceFailure(max_iter)


def google_matrix(
    G: Graph,
    alpha=0.85,
    personalization=None,
    nodelist=None,
    dangling=None,
    name="google_matrix",
) -> Matrix:
    A = G._A
    ids = G.list_to_ids(nodelist)
    if ids is not None:
        ids = np.array(ids, np.uint64)
        A = A[ids, ids].new(float, name=name)
    else:
        A = A.dup(float, name=name)
    N = A.nrows
    if N == 0:
        return A

    # Personalization vector or scalar
    if personalization is None:
        p = 1.0 / N
    else:
        if ids is not None:
            personalization = personalization[ids].new(name="personalization")
        denom = personalization.reduce().get(0)
        if denom == 0:
            raise ZeroDivisionError("personalization sums to 0")
        p = (personalization / denom).new(mask=personalization.V, name="p")

    if ids is None or len(ids) == len(G):
        nonempty_rows = G.get_property("any_rowwise+")  # XXX: What about self-edges?
    else:
        nonempty_rows = A.reduce_rowwise(monoid.any).new(name="nonempty_rows")

    is_dangling = nonempty_rows.nvals < N
    if is_dangling:
        empty_rows = (~nonempty_rows.S).new(name="empty_rows")
        if dangling is not None:
            if ids is not None:
                dangling = dangling[ids].new(name="dangling")
            dangling_weights = (1.0 / dangling.reduce().get(0) * dangling).new(
                mask=dangling.V, name="dangling_weights"
            )
            A << binary.first(empty_rows.outer(dangling_weights) | A)
        elif personalization is None:
            A << binary.first((p * empty_rows) | A)
        else:
            A << binary.first(empty_rows.outer(p) | A)

    scale = A.reduce_rowwise(monoid.plus).new(float)
    scale << alpha / scale
    A << scale * A
    p *= 1 - alpha
    if personalization is None:
        # Add a scalar everywhere, which makes A dense
        A(binary.plus)[:, :] = p
    else:
        A << A + p
    return A
