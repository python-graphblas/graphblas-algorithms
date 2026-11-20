"""What the `Dispatcher` offers, for the tests that walk every dispatched name."""


def dispatcher_functions():
    """The algorithms the `Dispatcher` offers NetworkX, without the protocol hooks."""
    from _nx_graphblas.core import get_functions

    return get_functions()
