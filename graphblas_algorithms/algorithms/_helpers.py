from graphblas import binary, monoid, unary


def normalize(x, how="L1"):
    if how == "L1":
        denom = x.reduce().get(0)
    elif how == "L2":
        denom = (x @ x).get(0) ** 0.5
    elif how == "Linf":
        denom = x.reduce(monoid.max).get(0)
    else:
        raise ValueError(f"Unknown normalization method: {how}")
    try:
        x *= 1.0 / denom
    except ZeroDivisionError:  # pragma: no cover
        pass
    return x


def is_converged(xprev, x, tol):
    """Check convergence, L1 norm: err = sum(abs(xprev - x)); err < N * tol

    This modifies `xprev`.
    """
    xprev << binary.minus(xprev | x)
    xprev << unary.abs(xprev)
    err = xprev.reduce().get(0)
    return err < xprev.size * tol
