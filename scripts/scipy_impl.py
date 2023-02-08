import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse  # call as sp.sparse


def pagerank(
    A,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    N = A.shape[0]
    if A.nnz == 0:
        return {}

    # nodelist = list(G)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    # TODO: csr_array
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    # initial vector
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        raise NotImplementedError()
    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        raise NotImplementedError()
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        raise NotImplementedError()
    is_dangling = np.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x @ A + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < N * tol:
            return x
            # return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)
