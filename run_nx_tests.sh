#!/bin/bash
NETWORKX_TEST_BACKEND=graphblas \
NETWORKX_FALLBACK_TO_NX=True \
    pytest --pyargs networkx "$@"
