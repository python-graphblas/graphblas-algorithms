from math import ceil
from numbers import Number

try:
    from itertools import pairwise  # Added in Python 3.10
except ImportError:

    def pairwise(it):
        it = iter(it)
        for prev in it:
            for cur in it:
                yield (prev, cur)
                prev = cur


BYTES_UNITS = {
    "": 1,
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "pb": 1000**5,
    "eb": 1000**6,
    "zb": 1000**7,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
    "pib": 1024**5,
    "eib": 1024**6,
    "zib": 1024**7,
}


def normalize_chunksize(chunksize, itemsize=1, N=None):
    if chunksize is None:
        return None
    if isinstance(chunksize, Number):
        rv = int(chunksize)
        if rv <= 0 or N is not None and rv >= N:
            return None
        return rv
    if not isinstance(chunksize, str):
        raise TypeError(f"chunksize must be a number or a string; got {type(chunksize)}")
    chunkstring = chunksize.replace(" ", "").replace("_", "").lower()
    if not chunkstring or chunkstring == "all":
        return None
    for i, c in enumerate(reversed(chunkstring)):
        if c.isdigit():
            index = len(chunkstring) - i
            break
    else:
        chunkstring = f"1{chunkstring}"
        index = 1

    prefix = chunkstring[:index]
    suffix = chunkstring[index:]

    try:
        number = float(prefix)
    except ValueError as exc:
        raise ValueError(
            f"Bad chunksize: {chunksize!r}. Could not interpret {prefix!r} as a number."
        ) from exc

    if suffix in {"chunk", "chunks"}:
        if number <= 1:
            return None
        if N is None:
            raise TypeError(
                f"N argument is required to determine chunksize to split into {int(number)} chunks"
            )
        rv = ceil(N / number)
    else:
        scale = BYTES_UNITS.get(suffix)
        if scale is None:
            raise ValueError(
                f"Bad chunksize: {chunksize!r}. Could not interpret {suffix!r} as a bytes unit."
            )
        number *= scale
        if chunkstring[-1] == "b":
            number = max(1, number / itemsize)
        rv = int(round(number))
    if rv <= 0 or N is not None and rv >= N:
        return None
    return rv


def partition(chunksize, L, *, evenly=True):
    """Partition a list into chunks"""
    N = len(L)
    if N == 0:
        return
    chunksize = int(chunksize)
    if chunksize <= 0 or chunksize >= N:
        yield L
        return
    if chunksize == 1:
        yield from L
        return
    if evenly:
        k = ceil(L / chunksize)
        if k * chunksize != N:
            yield from split_evenly(k, L)
            return
    for start, stop in pairwise(range(0, N + chunksize, chunksize)):
        yield L[start:stop]


def split_evenly(k, L):
    """Split a list into approximately-equal parts"""
    N = len(L)
    if N == 0:
        return
    k = int(k)
    if k <= 1:
        yield L
        return
    start = 0
    for i in range(1, k):
        stop = (N * i + k - 1) // k
        if stop != start:
            yield L[start:stop]
            start = stop
    if stop != N:
        yield L[stop:]
