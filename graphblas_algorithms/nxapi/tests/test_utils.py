import pytest

from graphblas_algorithms.nxapi._utils import normalize_chunksize


def test_normalize_chunksize():
    assert normalize_chunksize(None) is None
    assert normalize_chunksize("all") is None
    assert normalize_chunksize("") is None
    assert normalize_chunksize(-1) is None
    assert normalize_chunksize("-1") is None
    assert normalize_chunksize(10, N=10) is None
    assert normalize_chunksize("1 MB", N=100) is None
    assert normalize_chunksize("1 chunk") is None
    assert normalize_chunksize("2 chunks", N=20) == 10
    assert normalize_chunksize(10) == 10
    assert normalize_chunksize(10.0) == 10
    assert normalize_chunksize("10") == 10
    assert normalize_chunksize("10.0") == 10
    assert normalize_chunksize("1_0 B") == 10
    assert normalize_chunksize("1e1") == 10
    assert normalize_chunksize("1e-2 kb") == 10
    assert normalize_chunksize("Mb") == 1000**2
    assert normalize_chunksize(" mb") == 1000**2
    assert normalize_chunksize("gib") == 1024**3
    with pytest.raises(TypeError, match="chunksize must be"):
        normalize_chunksize(object())
    with pytest.raises(ValueError, match="as a bytes"):
        normalize_chunksize("10 badbytes")
    with pytest.raises(ValueError, match="as a number"):
        normalize_chunksize("1bad0 TB")
    with pytest.raises(TypeError, match="N argument is required"):
        normalize_chunksize("10 chunks")
