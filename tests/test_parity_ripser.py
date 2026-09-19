import numpy as np
import pytest
from ripser import ripser
from scipy.sparse.csgraph import minimum_spanning_tree

import persist0


def _clouds(rng, b, n, d=8, metric="euclidean"):
    x = rng.standard_normal((b, n, d)).astype(np.float32)
    if metric == "cosine":
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        D = 1.0 - np.einsum("bnd,bmd->bnm", x, x)
    else:
        diff = x[:, :, None, :] - x[:, None, :, :]
        D = np.sqrt((diff**2).sum(-1))
    D = ((D + D.transpose(0, 2, 1)) / 2).astype(np.float32)
    for k in range(b):
        np.fill_diagonal(D[k], 0.0)
    return np.ascontiguousarray(D)


def _deaths(D, idx):
    b = np.arange(D.shape[0])[:, None]
    return D[b, idx[..., 0], idx[..., 1]]


@pytest.mark.parametrize("n", [8, 32, 64, 128])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_h0_matches_ripser(n, metric):
    rng = np.random.default_rng(0)
    D = _clouds(rng, 25, n, metric=metric)
    idx = persist0.h0_batched(D)
    assert idx.shape == (25, n - 1, 2)

    ours = np.sort(_deaths(D, idx), axis=1)
    for k in range(D.shape[0]):
        dgm = ripser(D[k], maxdim=0, distance_matrix=True)["dgms"][0]
        ref = np.sort(dgm[np.isfinite(dgm[:, 1]), 1])
        np.testing.assert_allclose(ours[k], ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("n", [16, 64])
def test_mst_weight_matches_scipy(n):
    rng = np.random.default_rng(1)
    D = _clouds(rng, 10, n)
    idx = persist0.h0_batched(D)
    ours = _deaths(D, idx).sum(axis=1)
    ref = np.array([minimum_spanning_tree(D[k]).sum() for k in range(D.shape[0])])
    np.testing.assert_allclose(ours, ref, rtol=1e-5)


def test_batched_equals_unbatched():
    rng = np.random.default_rng(2)
    D = _clouds(rng, 6, 40)
    full = persist0.h0_batched(D)
    for k in range(D.shape[0]):
        single = persist0.h0_batched(D[k : k + 1])
        np.testing.assert_array_equal(full[k], single[0])


def test_rejects_non_square():
    with pytest.raises(ValueError):
        persist0.h0_batched(np.zeros((1, 4, 5), dtype=np.float32))
