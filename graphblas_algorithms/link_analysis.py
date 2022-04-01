from warnings import warn

import grblas as gb
import networkx as nx
import numpy as np
from grblas.binary import plus
from grblas.semiring import plus_first


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
    nodelist = list(G)
    # grblas io functions should be able to do this
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    A = gb.Matrix.ss.import_csr(
        nrows=A.shape[0],
        ncols=A.shape[1],
        indptr=A.indptr,
        col_indices=A.indices,
        values=A.data,
        sorted_cols=A._has_sorted_indices,
        take_ownership=True,
        name="A",
    )
    S = A.reduce_rowwise().new(name="S")
    A << gb.semiring.any_truediv.commutes_to(S.diag() @ A)  # TODO: use any_rtruediv

    # initial vector
    if nstart is None:
        x = gb.Vector.new(float, N)
        x[:] = 1.0 / N
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x /= x.sum()
        x = gb.Vector.ss.import_full(x, take_ownership=True, name="x")

    # Personalization vector
    if personalization is None:
        p = gb.Vector.new(float, N, name="p")
        p[:] = 1.0 / N
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError()
        p /= p.sum()
        p = gb.Vector.ss.import_full(p, take_ownership=True, name="p")

    # Dangling nodes
    if dangling is None:
        dangling_weights = gb.Vector.new(float, N, name="dangling_weights")
        dangling_weights(mask=~S.S) << alpha * p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        weights /= weights.sum()
        weights *= alpha
        dangling_weights = gb.Vector.ss.import_full(weights, take_ownership=True, name="weights")
        dangling_mask = gb.Vector.new(float, N, name="dangling_mask")
        dangling_mask(mask=~S.S) << 1.0

    # Fold constants into objects
    A *= alpha
    p *= 1 - alpha
    is_dangling = dangling_weights.nvals > 0
    xnew = gb.Vector.new(float, N, name="xnew")
    for i in range(max_iter):
        # xnew << alpha * (x @ A + "dangling_weights") + (1 - alpha) * p
        xnew << x @ A
        xnew(plus) << p
        if is_dangling:
            if dangling is None:
                # Add a constant (dangling_weights is already masked)
                xnew += x @ dangling_weights
            else:
                # Add a vector
                xnew(plus) << plus_first(x @ dangling_mask) * dangling_weights

        err = gb.binary.minus(x & xnew).apply(gb.unary.abs).reduce().value
        if err < N * tol:
            return dict(zip(nodelist, xnew.to_values()[1]))
        x, xnew = xnew, x
    raise nx.PowerIterationFailedConvergence(max_iter)
