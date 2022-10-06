#!/bin/bash
NETWORKX_GRAPH_CONVERT=graphblas pytest --pyargs networkx --cov --cov-report term-missing "$@"
