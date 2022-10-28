from graphblas import Scalar, Vector
from graphblas.core.utils import output_type
from graphblas.semiring import plus_first, plus_times

from graphblas_algorithms.algorithms._helpers import is_converged, normalize
from graphblas_algorithms.algorithms.exceptions import (
    ConvergenceFailure,
    GraphBlasAlgorithmException,
)

__all__ = ["katz_centrality"]


def katz_centrality(
    G,
    alpha=0.1,
    beta=1.0,
    max_iter=1000,
    tol=1.0e-6,
    nstart=None,
    normalized=True,
    name="katz_centrality",
):
    N = len(G)
    x = Vector(float, N, name="x")
    if nstart is None:
        x << 0.0
    else:
        x << nstart
    if output_type(beta) is not Vector:
        b = Scalar.from_value(beta, dtype=float, name="beta")
    else:
        b = beta
        if b.nvals != N:
            raise GraphBlasAlgorithmException("beta must have a value for every node")

    A = G._A
    if A.ss.is_iso:
        # Fold iso-value into alpha
        alpha *= A.ss.iso_value.get(1.0)
        semiring = plus_first[float]
    else:
        semiring = plus_times[float]

    # Power iteration: make up to max_iter iterations
    xprev = Vector(float, N, name="x_prev")
    for _ in range(max_iter):
        xprev, x = x, xprev
        # x << alpha * semiring(xprev @ A) + beta
        x << semiring(xprev @ A)
        x *= alpha
        x += b
        if is_converged(xprev, x, tol):  # sum(abs(xprev - x)) < N * tol
            x.name = name
            if normalized:
                normalize(x, "L2")
            return x
    raise ConvergenceFailure(max_iter)
