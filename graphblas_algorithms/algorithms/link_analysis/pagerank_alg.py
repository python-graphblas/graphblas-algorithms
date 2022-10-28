from graphblas import Vector
from graphblas.semiring import plus_first, plus_times

from graphblas_algorithms import Graph
from graphblas_algorithms.algorithms._helpers import is_converged
from graphblas_algorithms.algorithms.exceptions import ConvergenceFailure

__all__ = ["pagerank"]


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

        if is_converged(xprev, x, tol):  # sum(abs(xprev - x)) < N * tol
            x.name = name
            return x
    raise ConvergenceFailure(max_iter)
