from graphblas import Matrix, binary

__all__ = ["is_simple_path"]


def is_simple_path(G, nodes):
    if len(nodes) == 0:
        return False
    if len(nodes) == 1:
        return nodes[0] in G
    A = G._A
    if A.nvals < len(nodes) - 1:
        return False
    key_to_id = G._key_to_id
    indices = [key_to_id[key] for key in nodes if key in key_to_id]
    if len(indices) != len(nodes) or len(indices) > len(set(indices)):
        return False
    # Check all steps in path at once
    P = Matrix.from_values(indices[:-1], indices[1:], True, nrows=A.nrows, ncols=A.ncols)
    P << binary.second(A & P)
    return P.nvals == len(indices) - 1
    # Alternative
    # it = iter(indices)
    # prev = next(it)
    # for cur in it:
    #     if (prev, cur) not in A:
    #         return False
    #     prev = cur
    # return True
