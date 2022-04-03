import click
import networkx as nx


def best_units(num):
    """Returns scale factor and prefix such that 1 <= num*scale < 1000"""
    if num < 1e-12:
        return 1e15, "f"
    if num < 1e-9:
        return 1e12, "p"
    if num < 1e-6:
        return 1e9, "n"
    if num < 1e-3:
        return 1e6, "u"
    if num < 1:
        return 1e3, "m"
    if num < 1e3:
        return 1.0, ""
    if num < 1e6:
        return 1e-3, "k"
    if num < 1e9:
        return 1e-6, "M"
    if num < 1e12:
        return 1e-9, "G"
    return 1e-12, "T"


def stime(time):
    scale, units = best_units(time)
    return f"{time * scale:4.3g} {units}s"


# Copied and modified from networkx
def pagerank_scipy(
    A,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    import numpy as np
    import scipy as sp
    import scipy.sparse  # call as sp.sparse

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


@click.command()
@click.argument("filename")
@click.option(
    "-b",
    "--backend",
    default="graphblas",
    type=click.Choice(["graphblas", "networkx", "scipy", "gb", "nx", "sp"]),
)
@click.option(
    "-t",
    "--time",
    default=3,
    type=click.FloatRange(min=0),
)
@click.option(
    "-n",
    default=None,
    type=click.IntRange(min=0),
)
def main(filename, backend, time, n):
    import statistics
    import timeit

    import numpy as np

    backend = {
        "gb": "graphblas",
        "nx": "networkx",
        "sp": "scipy",
    }.get(backend, backend)
    print(f"Filename: {filename} ; backend: {backend}")

    if backend == "graphblas":
        import pandas as pd
        from grblas import Matrix

        from graphblas_algorithms.link_analysis import pagerank_core as pagerank

        start = timeit.default_timer()
        df = pd.read_csv(filename, delimiter="\t", names=["row", "col"])
        G = Matrix.from_values(df["row"].values, df["col"].values, 1)
        stop = timeit.default_timer()
        num_nodes = G.nrows
        num_edges = G.nvals
    elif backend == "networkx":
        from networkx import pagerank

        start = timeit.default_timer()
        G = nx.read_edgelist(filename, delimiter="\t", nodetype=int, create_using=nx.DiGraph)
        for i in range(max(G)):
            if i not in G:
                G.add_node(i)
        stop = timeit.default_timer()
        num_nodes = len(G.nodes)
        num_edges = len(G.edges)
    else:
        import pandas as pd
        import scipy.sparse

        start = timeit.default_timer()
        df = pd.read_csv(filename, delimiter="\t", names=["row", "col"])
        G = scipy.sparse.csr_array((np.repeat(1.0, len(df)), (df["row"].values, df["col"].values)))
        pagerank = pagerank_scipy
        stop = timeit.default_timer()
        num_nodes = G.shape[0]
        num_edges = G.nnz

    print("Num nodes:", num_nodes)
    print("Num edges:", num_edges)
    print("Load time:", stime(stop - start))
    timer = timeit.Timer("pagerank(G)", globals=dict(pagerank=pagerank, G=G))
    first_time = timer.timeit(1)
    if time == 0:
        n = 1
    elif n is None:
        n = 2 ** max(0, int(np.ceil(np.log2(time / first_time))))
    print("Number of runs:", n)
    print("first: ", stime(first_time))
    if n > 1:
        results = timer.repeat(n - 1, 1)
        results.append(first_time)
        print("median:", stime(statistics.median(results)))
        print("mean:  ", stime(statistics.mean(results)))
        # print("hmean: ", stime(statistics.harmonic_mean(results)))
        # print("gmean: ", stime(statistics.geometric_mean(results)))
        print("stdev: ", stime(statistics.stdev(results)))
        print("min:   ", stime(min(results)))
        print("max:   ", stime(max(results)))


if __name__ == "__main__":
    main()
