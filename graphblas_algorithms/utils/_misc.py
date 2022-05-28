from importlib import import_module

__all__ = ["get_all"]


def get_all(name):
    this = import_module(name)
    that = import_module(name.replace("graphblas_algorithms", "networkx", 1))
    return [key for key in that.__all__ if key in this.__dict__]
