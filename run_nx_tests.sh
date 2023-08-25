#!/bin/bash
NETWORKX_GRAPH_CONVERT=graphblas \
NETWORKX_TEST_BACKEND=graphblas \
NETWORKX_FALLBACK_TO_NX=True \
    pytest --pyargs networkx "$@"
#    pytest --pyargs networkx --cov --cov-report term-missing "$@"
