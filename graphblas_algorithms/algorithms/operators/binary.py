import numpy as np
from graphblas import Matrix, binary, dtypes, unary

from ..exceptions import GraphBlasAlgorithmException

__all__ = [
    "compose",
    "difference",
    "disjoint_union",
    "full_join",
    "intersection",
    "symmetric_difference",
    "union",
]


def union(G, H, rename=(), *, name="union"):
    if G.is_multigraph() != H.is_multigraph():
        raise GraphBlasAlgorithmException("All graphs must be graphs or multigraphs.")
    if G.is_multigraph():
        raise NotImplementedError("Not yet implemented for multigraphs")
    if rename:
        prefix = rename[0]
        if prefix is not None:
            G = type(G)(
                G._A, key_to_id={f"{prefix}{key}": val for key, val in G._key_to_id.items()}
            )
        if len(rename) > 1:
            prefix = rename[1]
            if prefix is not None:
                H = type(H)(
                    H._A, key_to_id={f"{prefix}{key}": val for key, val in H._key_to_id.items()}
                )
    A = G._A
    B = H._A
    if not G._key_to_id.keys().isdisjoint(H._key_to_id.keys()):
        raise GraphBlasAlgorithmException("The node sets of the graphs are not disjoint.")
    C = Matrix(dtypes.unify(A.dtype, B.dtype), A.nrows + B.nrows, A.ncols + B.ncols, name=name)
    C[: A.nrows, : A.ncols] = A
    C[A.nrows :, A.ncols :] = B
    offset = A.nrows
    key_to_id = {key: val + offset for key, val in H._key_to_id.items()}
    key_to_id.update(G._key_to_id)
    return type(G)(C, key_to_id=key_to_id)


def disjoint_union(G, H, *, name="disjoint_union"):
    if G.is_multigraph() != H.is_multigraph():
        raise GraphBlasAlgorithmException("All graphs must be graphs or multigraphs.")
    if G.is_multigraph():
        raise NotImplementedError("Not yet implemented for multigraphs")
    A = G._A
    B = H._A
    C = Matrix(dtypes.unify(A.dtype, B.dtype), A.nrows + B.nrows, A.ncols + B.ncols, name=name)
    C[: A.nrows, : A.ncols] = A
    C[A.nrows :, A.ncols :] = B
    return type(G)(C)


def intersection(G, H, *, name="intersection"):
    if G.is_multigraph() != H.is_multigraph():
        raise GraphBlasAlgorithmException("All graphs must be graphs or multigraphs.")
    if G.is_multigraph():
        raise NotImplementedError("Not yet implemented for multigraphs")
    keys = sorted(G._key_to_id.keys() & H._key_to_id.keys(), key=G._key_to_id.__getitem__)
    ids = np.array(G.list_to_ids(keys), np.uint64)
    A = G._A[ids, ids].new()
    ids = np.array(H.list_to_ids(keys), np.uint64)
    B = H._A[ids, ids].new(dtypes.unify(A.dtype, H._A.dtype), mask=A.S, name=name)
    B << unary.one(B)
    return type(G)(B, key_to_id=dict(zip(keys, range(len(keys)))))


def difference(G, H, *, name="difference"):
    if G.is_multigraph() != H.is_multigraph():
        raise GraphBlasAlgorithmException("All graphs must be graphs or multigraphs.")
    if G.is_multigraph():
        raise NotImplementedError("Not yet implemented for multigraphs")
    if G._key_to_id.keys() != H._key_to_id.keys():
        raise GraphBlasAlgorithmException("Node sets of graphs not equal")
    A = G._A
    if G._key_to_id == H._key_to_id:
        B = H._A
    else:
        # Need to perform a permutation
        keys = sorted(G._key_to_id, key=G._key_to_id.__getitem__)
        ids = np.array(H.list_to_ids(keys), np.uint64)
        B = H._A[ids, ids].new()
    C = unary.one(A).new(mask=~B.S, name=name)
    return type(G)(C, key_to_id=G._key_to_id)


def symmetric_difference(G, H, *, name="symmetric_difference"):
    if G.is_multigraph() != H.is_multigraph():
        raise GraphBlasAlgorithmException("All graphs must be graphs or multigraphs.")
    if G.is_multigraph():
        raise NotImplementedError("Not yet implemented for multigraphs")
    if G._key_to_id.keys() != H._key_to_id.keys():
        raise GraphBlasAlgorithmException("Node sets of graphs not equal")
    A = G._A
    if G._key_to_id == H._key_to_id:
        B = H._A
    else:
        # Need to perform a permutation
        keys = sorted(G._key_to_id, key=G._key_to_id.__getitem__)
        ids = np.array(H.list_to_ids(keys), np.uint64)
        B = H._A[ids, ids].new()
    Mask = binary.pair[bool](A & B).new(name="mask")
    C = binary.pair(A | B, left_default=True, right_default=True).new(mask=~Mask.S, name=name)
    return type(G)(C, key_to_id=G._key_to_id)


def compose(G, H, *, name="compose"):
    if G.is_multigraph() != H.is_multigraph():
        raise GraphBlasAlgorithmException("All graphs must be graphs or multigraphs.")
    if G.is_multigraph():
        raise NotImplementedError("Not yet implemented for multigraphs")
    A = G._A
    B = H._A
    if G._key_to_id.keys() == H._key_to_id.keys():
        if G._key_to_id != H._key_to_id:
            # Need to perform a permutation
            keys = sorted(G._key_to_id, key=G._key_to_id.__getitem__)
            ids = np.array(H.list_to_ids(keys), np.uint64)
            B = B[ids, ids].new()
        C = binary.second(A | B).new(name=name)
        key_to_id = G._key_to_id
    else:
        keys = sorted(G._key_to_id.keys() & H._key_to_id.keys(), key=G._key_to_id.__getitem__)
        B = H._A
        C = Matrix(
            dtypes.unify(A.dtype, B.dtype),
            A.nrows + B.nrows - len(keys),
            A.ncols + B.ncols - len(keys),
            name=name,
        )
        C[: A.nrows, : A.ncols] = A
        ids1 = np.array(G.list_to_ids(keys), np.uint64)
        ids2 = np.array(H.list_to_ids(keys), np.uint64)
        C[ids1, ids1] = B[ids2, ids2]
        newkeys = sorted(H._key_to_id.keys() - G._key_to_id.keys(), key=H._key_to_id.__getitem__)
        ids = np.array(H.list_to_ids(newkeys), np.uint64)
        C[A.nrows :, A.ncols :] = B[ids, ids]
        # Now make new `key_to_id`
        ids += A.nrows
        key_to_id = dict(zip(newkeys, ids.tolist()))
        key_to_id.update(G._key_to_id)
    return type(G)(C, key_to_id=key_to_id)


def full_join(G, H, rename=(), *, name="full_join"):
    rv = union(G, H, rename, name=name)
    nrows, ncols = G._A.shape
    rv._A[:nrows, ncols:] = True
    rv._A[nrows:, :ncols] = True
    return rv
