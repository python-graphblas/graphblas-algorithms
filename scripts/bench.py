#!/usr/bin/env python
import argparse
import json
import os
import statistics
import sys
import timeit

import download_data
import graphblas as gb
import networkx as nx
import numpy as np
import scipy.sparse

import graphblas_algorithms as ga
import scipy_impl
from graphblas_algorithms.interface import Dispatcher as ga_dispatcher

thisdir = os.path.dirname(__file__)
datapaths = [
    os.path.join(thisdir, "..", "data"),
    os.path.curdir,
]


def find_data(dataname):
    if os.path.exists(dataname):
        return os.path.relpath(dataname)
    for path in datapaths:
        path = os.path.join(path, dataname) + ".mtx"
        if os.path.exists(path):
            return os.path.relpath(path)
        path = path.removesuffix(".mtx")
        if os.path.exists(path):
            return os.path.relpath(path)
    if dataname in download_data.data_urls:
        return os.path.relpath(download_data.main([dataname])[0])
    raise FileNotFoundError(f"Unable to find data file for {dataname}")


def get_symmetry(file_or_mminfo):
    if not isinstance(file_or_mminfo, tuple):
        mminfo = scipy.io.mminfo(file_or_mminfo)
    else:
        mminfo = file_or_mminfo
    return mminfo[5]


def readfile(filename, is_symmetric, backend):
    name = filename.split(".", 1)[0].rsplit("/", 1)[0]
    if backend == "graphblas":
        A = gb.io.mmread(filename, name=name)
        A.wait()
        if is_symmetric:
            return ga.Graph(A)
        return ga.DiGraph(A)
    a = scipy.io.mmread(filename)
    if backend == "networkx":
        create_using = nx.Graph if is_symmetric else nx.DiGraph
        return nx.from_scipy_sparse_array(a, create_using=create_using)
    if backend == "scipy":
        return scipy.sparse.csr_array(a)
    raise ValueError(
        f"Backend {backend!r} not understood; must be 'graphblas', 'networkx', or 'scipy'"
    )


def best_units(num):
    """Returns scale factor and prefix such that 1 <= num*scale < 1000"""
    if num < 1e-12:
        return 1e15, "f"
    if num < 1e-9:
        return 1e12, "p"
    if num < 1e-6:
        return 1e9, "n"
    if num < 1e-3:
        return 1e6, "\N{MICRO SIGN}"
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


# Functions that aren't available in the main networkx namespace
functionpaths = {
    "inter_community_edges": "community.quality.inter_community_edges",
    "intra_community_edges": "community.quality.intra_community_edges",
    "is_tournament": "tournament.is_tournament",
    "mutual_weight": "structuralholes.mutual_weight",
    "score_sequence": "tournament.score_sequence",
    "tournament_matrix": "tournament.tournament_matrix",
}
functioncall = {
    "s_metric": "func(G, normalized=False)",
}
poweriteration = {"eigenvector_centrality", "katz_centrality", "pagerank"}
directed_only = {
    "in_degree_centrality",
    "is_tournament",
    "out_degree_centrality",
    "score_sequence",
    "tournament_matrix",
    "reciprocity",
    "overall_reciprocity",
}
# Is square_clustering undirected only? graphblas-algorthms doesn't implement it for directed
undirected_only = {"generalized_degree", "k_truss", "triangles", "square_clustering"}


def getfunction(functionname, backend):
    if backend == "graphblas":
        return getattr(ga_dispatcher, functionname)
    if backend == "scipy":
        return getattr(scipy_impl, functionname)
    if functionname in functionpaths:
        func = nx
        for attr in functionpaths[functionname].split("."):
            func = getattr(func, attr)
        return func
    return getattr(nx, functionname)


def main(dataname, backend, functionname, time=3.0, n=None, extra=None, display=True):
    filename = find_data(dataname)
    is_symmetric = get_symmetry(filename) == "symmetric"
    if not is_symmetric and functionname in undirected_only:
        # Should we automatically symmetrize?
        raise ValueError(
            f"Data {dataname!r} is not symmetric, but {functionname} only works on undirected"
        )
    if is_symmetric and functionname in directed_only:
        is_symmetric = False  # Make into directed graph
    G = readfile(filename, is_symmetric, backend)
    func = getfunction(functionname, backend)
    benchstring = functioncall.get(functionname, "func(G)")
    if extra is not None:
        benchstring = f"{benchstring[:-1]}, {extra})"
    globals = {"func": func, "G": G}
    if functionname in poweriteration:
        benchstring = f"try:\n    {benchstring}\nexcept exc:\n    pass"
        globals["exc"] = nx.PowerIterationFailedConvergence
    if backend == "graphblas":
        benchstring = f"G._cache.clear()\n{benchstring}"
    timer = timeit.Timer(benchstring, globals=globals)
    if display:
        line = f"Backend = {backend}, function = {functionname}, data = {dataname}"
        if extra is not None:
            line += f", extra = {extra}"
        print("=" * len(line))
        print(line)
        print("-" * len(line))
    info = {"backend": backend, "function": functionname, "data": dataname}
    if extra is not None:
        info["extra"] = extra
    try:
        first_time = timer.timeit(1)
    except Exception as exc:
        if display:
            print(f"EXCEPTION: {exc}")
            print("=" * len(line))
            raise
        info["exception"] = str(exc)
        return info
    if time == 0:
        n = 1
    elif n is None:
        n = 2 ** max(0, int(np.ceil(np.log2(time / first_time))))
    if display:
        print("Number of runs:", n)
        print("first: ", stime(first_time))
    info["n"] = n
    info["first"] = first_time
    if n > 1:
        results = timer.repeat(n - 1, 1)
        results.append(first_time)
        if display:
            print("median:", stime(statistics.median(results)))
            print("mean:  ", stime(statistics.mean(results)))
            print("stdev: ", stime(statistics.stdev(results)))
            print("min:   ", stime(min(results)))
            print("max:   ", stime(max(results)))
        info["median"] = statistics.median(results)
        info["mean"] = statistics.mean(results)
        info["stdev"] = statistics.stdev(results)
        info["min"] = min(results)
        info["max"] = max(results)
    if display:
        print("=" * len(line))
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Example usage: python {sys.argv[0]} -b graphblas -f pagerank -d amazon0302"
    )
    parser.add_argument(
        "-b", "--backend", choices=["graphblas", "networkx", "scipy"], default="graphblas"
    )
    parser.add_argument(
        "-t", "--time", type=float, default=3.0, help="Target minimum time to run benchmarks"
    )
    parser.add_argument(
        "-n",
        type=int,
        help="The number of times to run the benchmark (the default is to run according to time)",
    )
    parser.add_argument(
        "-d",
        "--data",
        required=True,
        help="The path to a mtx file or one of the following data names: {"
        + ", ".join(sorted(download_data.data_urls))
        + "}; data will be downloaded if necessary",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Print results as json instead of human-readable text",
    )
    parser.add_argument("-f", "--func", required=True, help="Which function to benchmark")
    parser.add_argument("--extra", help="Extra string to add to the function call")
    args = parser.parse_args()
    info = main(
        args.data,
        args.backend,
        args.func,
        time=args.time,
        n=args.n,
        extra=args.extra,
        display=not args.json,
    )
    if args.json:
        print(json.dumps(info))
