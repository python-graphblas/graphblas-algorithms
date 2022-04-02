from warnings import warn

import networkx as nx
import numpy as np
from grblas import Matrix, Vector, binary, monoid, unary
from grblas.semiring import plus_first, plus_times


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
    # TODO: consider what happens and what we should do if personalization,
    # nstart, dangling, and row_degrees input vectors are not dense.
    N = A.nrows
    if A.nvals == 0:
        return Vector.new(float, N, name=name)

    # Initial vector
    x = Vector.new(float, N, name="x")
    if nstart is None:
        x[:] = 1.0 / N
    else:
        x << nstart / nstart.reduce()

    # Personalization vector or scalar
    if personalization is None:
        p = 1.0 / N
    else:
        denom = personalization.reduce().value
        if denom == 0:
            raise ZeroDivisionError()
        p = (personalization / denom).new(name="p")

    # Inverse of row_degrees
    # Fold alpha constant into S
    if row_degrees is None:
        S = A.reduce_rowwise().new(float, name="S")
        S << alpha / S
    else:
        S = (alpha / row_degrees).new(name="S")

    if A.ss.is_iso:
        # Fold iso-value of A into S
        # This lets us use the plus_first semiring, which is faster
        S *= A.reduce_scalar(monoid.any)
        # S *= A.ss.iso_value  # This would be nice to have
        semiring = plus_first
    else:
        semiring = plus_times

    is_dangling = S.nvals < N
    if is_dangling:
        dangling_mask = Vector.new(float, N, name="dangling_mask")
        dangling_mask(mask=~S.S) << 1.0
        # Fold alpha constant into dangling_weights (or dangling_mask)
        if dangling is not None:
            dangling_weights = (alpha / dangling.reduce().value * dangling).new(
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
    xnew = Vector.new(float, N, name="xnew")
    w = Vector.new(float, N, name="w")
    for _ in range(max_iter):
        # xnew << alpha * ((x * S) @ A + "dangling_weights") + (1 - alpha) * p
        xnew << p
        if is_dangling:
            if dangling is None and personalization is None:
                # Fast case: add a scalar; xnew is still iso-valued (b/c p is also scalar)
                xnew += x @ dangling_mask
            else:
                # Add a vector
                xnew += plus_first(x @ dangling_mask) * dangling_weights
        w << x * S
        xnew += semiring(w @ A)  # plus_first if A.ss.is_iso else plus_times

        # Check convergence, l1 norm: err = sum(abs(x - xnew))
        x << binary.minus(x & xnew)
        x << unary.abs(x)
        err = x.reduce().value
        if err < N * tol:
            xnew.name = name
            return xnew
        x, xnew = xnew, x
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
    nodelist = list(G)
    # grblas io functions should be able to do this more conveniently, and it would
    # be best if it could determine whether the output matrix is iso-valued or not.
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    A = Matrix.ss.import_csr(
        nrows=A.shape[0],
        ncols=A.shape[1],
        indptr=A.indptr,
        col_indices=A.indices,
        values=A.data,
        sorted_cols=A._has_sorted_indices,
        take_ownership=True,
        name="A",
    )
    x = p = dangling_weights = None
    # Initial vector (we'll normalize later)
    if nstart is not None:
        x = Vector.ss.import_full(
            np.array([nstart.get(n, 0) for n in nodelist], dtype=float),
            take_ownership=True,
            name="nstart",
        )
    # Personalization vector (we'll normalize later)
    if personalization is not None:
        p = Vector.ss.import_full(
            np.array([personalization.get(n, 0) for n in nodelist], dtype=float),
            take_ownership=True,
            name="personalization",
        )
    # Dangling nodes (we'll normalize later)
    row_degrees = A.reduce_rowwise().new(name="row_degrees")
    if dangling is not None:
        if row_degrees.nvals < N:  # is_dangling
            dangling_weights = Vector.ss.import_full(
                np.array([dangling.get(n, 0) for n in nodelist], dtype=float),
                take_ownership=True,
                name="dangling",
            )
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
    return dict(zip(nodelist, result.to_values()[1]))
