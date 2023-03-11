def overall_reciprocity(G):
    return G.reciprocity()


def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1e-06,
    nstart=None,
    weight="weight",
    dangling=None,
    *,
    vertices=None,
    directed=True,
    arpack_options=None,
    implementation="prpack",
):
    if personalization is not None:
        raise NotImplementedError
    if nstart is not None:
        raise NotImplementedError
    if dangling is not None:
        raise NotImplementedError
    rv = G.pagerank(
        vertices=vertices,
        directed=directed,
        damping=alpha,
        weights=weight,
        arpack_options=arpack_options,
        implementation=implementation,
    )
    return rv


def transitivity(G):
    return G.transitivity_undirected()


def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    if nodes is not None:
        raise NotImplementedError
    # TODO: check results when `count_zeros=False`
    mode = "zero" if count_zeros else "nan"
    return G.transitivity_avglocal_undirected(mode=mode, weights=weight)
