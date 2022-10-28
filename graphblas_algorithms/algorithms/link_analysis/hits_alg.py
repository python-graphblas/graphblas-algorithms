from graphblas import Vector

from graphblas_algorithms.algorithms._helpers import is_converged, normalize
from graphblas_algorithms.algorithms.exceptions import ConvergenceFailure

__all__ = ["hits"]


def hits(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True, *, with_authority=False):
    """HITS algorithms with additional parameter `with_authority`.

    When `with_authority` is True, the authority matrix, ``A.T @ A`` will be
    created and used. This may be faster, but requires more memory.
    """
    N = len(G)
    h = Vector(float, N, name="h")
    a = Vector(float, N, name="a")
    if N == 0:
        return h, a
    if nstart is None:
        h << 1.0 / N
    else:
        h << nstart
        denom = h.reduce().get(0)
        h *= 1.0 / denom

    # Power iteration: make up to max_iter iterations
    A = G._A
    if with_authority:
        a, h = h, a
        ATA = (A.T @ A).new(name="ATA")  # Authority matrix
        aprev = Vector(float, N, name="a_prev")
        for _ in range(max_iter):
            aprev, a = a, aprev
            a << ATA @ aprev
            normalize(a, "Linf")
            if is_converged(aprev, a, tol):
                h << A @ a
                break
        else:
            raise ConvergenceFailure(max_iter)
    else:
        hprev = Vector(float, N, name="h_prev")
        for _ in range(max_iter):
            hprev, h = h, hprev
            a << hprev @ A
            h << A @ a
            normalize(h, "Linf")
            if is_converged(hprev, h, tol):
                break
        else:
            raise ConvergenceFailure(max_iter)
    if normalized:
        normalize(h, "L1")
        normalize(a, "L1")
    elif with_authority:
        normalize(h, "Linf")
    else:
        normalize(a, "Linf")
    h.name = "hits_h"
    a.name = "hits_a"
    return h, a
