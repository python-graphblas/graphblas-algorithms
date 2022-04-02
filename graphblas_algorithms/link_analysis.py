from warnings import warn

import grblas as gb
import networkx as nx
import numpy as np
from grblas.semiring import any_times, plus_first


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
    # If we know A is purely structural (comprised of only 1 values),
    # we can structure the calculation to use the plus_first semiring).
    S = A.reduce_rowwise().new(name="S")
    # Fold alpha constant into A
    A << any_times((alpha / S).diag() @ A)

    # Initial vector
    if nstart is None:
        x = gb.Vector.new(float, N)
        x[:] = 1.0 / N
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x /= x.sum()
        x = gb.Vector.ss.import_full(x, take_ownership=True, name="x")

    # Personalization vector
    if personalization is None:
        p = 1.0 / N
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError()
        p /= p.sum()
        p = gb.Vector.ss.import_full(p, take_ownership=True, name="p")

    # Dangling nodes
    is_dangling = S.nvals < N
    if is_dangling:
        dangling_mask = gb.Vector.new(float, N, name="dangling_mask")
        dangling_mask(mask=~S.S) << 1.0
        # Fold alpha constant into dangling_weights (or dangling_mask)
        if dangling is not None:
            # Convert the dangling dictionary into an array in nodelist order
            weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
            weights *= alpha / weights.sum()
            dangling_weights = gb.Vector.ss.import_full(
                weights, take_ownership=True, name="dangling_weights"
            )
        elif personalization is None:
            # Fast case (and common case)
            dangling_mask(mask=dangling_mask.S) << alpha * p
        else:
            dangling_weights = (alpha * p).new(name="dangling_weights")

    # Fold decay constant into p
    p *= 1 - alpha
    xnew = gb.Vector.new(float, N, name="xnew")
    # Power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        # xnew << alpha * (x @ A + "dangling_weights") + (1 - alpha) * p
        xnew << p
        if is_dangling:
            if dangling is None and personalization is None:
                # Fast case: add a scalar; xnew is still iso-valued (b/c p is also scalar)
                xnew += x @ dangling_mask
            else:
                # Add a vector
                xnew += plus_first(x @ dangling_mask) * dangling_weights
        # We use plus_times semiring here b/c A may be weighted or from a multigraph.
        # If we're clever, we could figure out when we can use plus_first semiring.
        xnew += x @ A

        # Check convergence, l1 norm
        err = gb.binary.minus(x & xnew).apply(gb.unary.abs).reduce().value
        if err < N * tol:
            return dict(zip(nodelist, xnew.to_values()[1]))
        x, xnew = xnew, x
    raise nx.PowerIterationFailedConvergence(max_iter)
