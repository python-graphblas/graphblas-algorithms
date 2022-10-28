from graphblas import op
from graphblas.core import operator


def get_reduce_to_vector(key, opname, methodname):
    op_ = op.from_string(opname)
    op_, opclass = operator.find_opclass(op_)
    keybase = key[:-1]
    if key[-1] == "-":

        def get_reduction(G, mask=None):
            cache = G._cache
            if mask is not None:
                if key in cache:
                    return cache[key].dup(mask=mask)
                elif cache.get("has_self_edges") is False and f"{keybase}+" in cache:
                    cache[key] = cache[f"{keybase}+"]
                    return cache[key].dup(mask=mask)
                elif "offdiag" in cache:
                    return getattr(cache["offdiag"], methodname)(op_).new(mask=mask, name=key)
                elif (
                    "L-" in cache
                    and "U-" in cache
                    and opclass in {"BinaryOp", "Monoid"}
                    and G.get_property("has_self_edges")
                ):
                    return op_(
                        getattr(cache["L-"], methodname)(op_).new(mask=mask)
                        | getattr(cache["U-"], methodname)(op_).new(mask=mask)
                    ).new(name=key)
                elif not G.get_property("has_self_edges"):
                    return G.get_property(f"{keybase}+", mask=mask)
                else:
                    return getattr(G.get_property("offdiag"), methodname)(op_).new(
                        mask=mask, name=key
                    )
            if key not in cache:
                if cache.get("has_self_edges") is False and f"{keybase}+" in cache:
                    cache[key] = cache[f"{keybase}+"]
                elif "offdiag" in cache:
                    cache[key] = getattr(cache["offdiag"], methodname)(op_).new(name=key)
                elif (
                    "L-" in cache
                    and "U-" in cache
                    and opclass in {"BinaryOp", "Monoid"}
                    and G.get_property("has_self_edges")
                ):
                    cache[key] = op_(
                        getattr(cache["L-"], methodname)(op_)
                        | getattr(cache["U-"], methodname)(op_)
                    ).new(name=key)
                elif not G.get_property("has_self_edges"):
                    cache[key] = G.get_property(f"{keybase}+")
                else:
                    cache[key] = getattr(G.get_property("offdiag"), methodname)(op_).new(name=key)
            if (
                "has_self_edges" not in cache
                and f"{keybase}+" in cache
                and cache[key].nvals != cache[f"{keybase}+"].nvals
            ):
                cache["has_self_edges"] = True
            elif cache.get("has_self_edges") is False:
                cache[f"{keybase}+"] = cache[key]
            return cache[key]

    else:

        def get_reduction(G, mask=None):
            A = G._A
            cache = G._cache
            if mask is not None:
                if key in cache:
                    return cache[key].dup(mask=mask)
                elif cache.get("has_self_edges") is False and f"{keybase}-" in cache:
                    cache[key] = cache[f"{keybase}-"]
                    return cache[key].dup(mask=mask)
                elif methodname == "reduce_columnwise" and "AT" in cache:
                    return cache["AT"].reduce_rowwise(op_).new(mask=mask, name=key)
                else:
                    return getattr(A, methodname)(op_).new(mask=mask, name=key)
            if key not in cache:
                if cache.get("has_self_edges") is False and f"{keybase}-" in cache:
                    cache[key] = cache[f"{keybase}-"]
                elif methodname == "reduce_columnwise" and "AT" in cache:
                    cache[key] = cache["AT"].reduce_rowwise(op_).new(name=key)
                else:
                    cache[key] = getattr(A, methodname)(op_).new(name=key)
            if (
                "has_self_edges" not in cache
                and f"{keybase}-" in cache
                and cache[key].nvals != cache[f"{keybase}-"].nvals
            ):
                cache["has_self_edges"] = True
            elif cache.get("has_self_edges") is False:
                cache[f"{keybase}-"] = cache[key]
            return cache[key]

    return get_reduction


def get_reduce_to_scalar(key, opname):
    op_ = op.from_string(opname)
    op_, opclass = operator.find_opclass(op_)
    keybase = key[:-1]
    if key[-1] == "-":

        def get_reduction(G, mask=None):
            cache = G._cache
            if key not in cache:
                if cache.get("has_self_edges") is False and f"{keybase}+" in cache:
                    cache[key] = cache[f"{keybase}+"]
                elif f"{opname}_rowwise-" in cache:
                    cache[key] = cache[f"{opname}_rowwise-"].reduce(op_).new(name=key)
                elif f"{opname}_columnwise-" in cache:
                    cache[key] = cache[f"{opname}_columnwise-"].reduce(op_).new(name=key)
                elif cache.get("has_self_edges") is False and f"{opname}_rowwise+" in cache:
                    cache[key] = cache[f"{opname}_rowwise+"].reduce(op_).new(name=key)
                elif cache.get("has_self_edges") is False and f"{opname}_columnwise+" in cache:
                    cache[key] = cache[f"{opname}_columnwise+"].reduce(op_).new(name=key)
                elif "offdiag" in cache:
                    cache[key] = cache["offdiag"].reduce_scalar(op_).new(name=key)
                elif (
                    "L-" in cache
                    and "U-" in cache
                    and opclass in {"BinaryOp", "Monoid"}
                    and G.get_property("has_self_edges")
                ):
                    return op_(
                        cache["L-"].reduce(op_)._as_vector() | cache["U-"].reduce(op_)._as_vector()
                    )[0].new(name=key)
                elif not G.get_property("has_self_edges"):
                    cache[key] = G.get_property(f"{keybase}+")
                else:
                    cache[key] = G.get_property("offdiag").reduce_scalar(op_).new(name=key)
            if (
                "has_self_edges" not in cache
                and f"{keybase}+" in cache
                and cache[key] != cache[f"{keybase}+"]
            ):
                cache["has_self_edges"] = True
            elif cache.get("has_self_edges") is False:
                cache[f"{keybase}+"] = cache[key]
            return cache[key]

    else:

        def get_reduction(G, mask=None):
            A = G._A
            cache = G._cache
            if key not in cache:
                if cache.get("has_self_edges") is False and f"{keybase}-" in cache:
                    cache[key] = cache[f"{keybase}-"]
                elif f"{opname}_rowwise+" in cache:
                    cache[key] = cache[f"{opname}_rowwise+"].reduce(op_).new(name=key)
                elif f"{opname}_columnwise+" in cache:
                    cache[key] = cache[f"{opname}_columnwise+"].reduce(op_).new(name=key)
                elif cache.get("has_self_edges") is False and f"{opname}_rowwise-" in cache:
                    cache[key] = cache[f"{opname}_rowwise-"].reduce(op_).new(name=key)
                elif cache.get("has_self_edges") is False and f"{opname}_columnwise-" in cache:
                    cache[key] = cache[f"{opname}_columnwise-"].reduce(op_).new(name=key)
                else:
                    cache[key] = A.reduce_scalar(op_).new(name=key)
            if (
                "has_self_edges" not in cache
                and f"{keybase}-" in cache
                and cache[key] != cache[f"{keybase}-"]
            ):
                cache["has_self_edges"] = True
            elif cache.get("has_self_edges") is False:
                cache[f"{keybase}-"] = cache[key]
            return cache[key]

    return get_reduction
