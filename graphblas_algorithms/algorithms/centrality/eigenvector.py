from graphblas import Vector, binary, unary

from graphblas_algorithms.algorithms.exceptions import (
    ConvergenceFailure,
    GraphBlasAlgorithmException,
    PointlessConcept,
)

__all__ = ["eigenvector_centrality"]


def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None, name="eigenvector_centrality"):
    N = len(G)
    if N == 0:
        raise PointlessConcept("cannot compute centrality for the null graph")
    x = Vector(float, N, name="x")
    if nstart is None:
        x << 1.0 / N
    else:
        x << nstart
        denom = x.reduce().get(0)  # why not use L2 norm?
        if denom == 0:
            raise GraphBlasAlgorithmException("initial vector cannot have all zero values")
        x *= 1.0 / denom

    # Power iteration: make up to max_iter iterations
    tol = tol * N
    A = G._A
    xprev = Vector(float, N, name="x_prev")
    for _ in range(max_iter):
        xprev << x
        x += x @ A

        # Normalize the vector
        try:
            x *= 1.0 / (x @ x).get(0) ** 0.5
        # this should never be zero?
        except ZeroDivisionError:  # pragma: no cover
            pass

        # Check convergence, l1 norm: err = sum(abs(xprev - x))
        xprev << binary.minus(xprev | x)
        xprev << unary.abs(xprev)
        err = xprev.reduce().value
        if err < tol:
            x.name = name
            return x
    raise ConvergenceFailure(max_iter)
