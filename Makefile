.PHONY: all plugin-info readme check lint test test-nx clean

all: readme plugin-info

# The plugin info is read off the dispatch table that `readme` writes; a prerequisite, unlike
# the order in `all`, also holds under `make -j`.
plugin-info: readme
	python _nx_graphblas/__init__.py

# maketree writes both the README tree and interface.py's dispatch table.
readme:
	python scripts/maketree.py

check:
	pytest graphblas_algorithms/tests/test_match_nx.py -v

lint:
	pre-commit run --all-files

test:
	pytest --color=yes -v

test-nx:
	./run_nx_tests.sh --color=yes -n auto

# Named directories only, never the whole working copy.
clean:
	find _nx_graphblas graphblas_algorithms scripts -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find _nx_graphblas graphblas_algorithms scripts -name '*.pyc' -delete 2>/dev/null || true
	rm -rf build/ dist/ ./*.egg-info .pytest_cache/
