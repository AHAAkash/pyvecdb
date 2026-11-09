import numpy as np
import pytest


def clustered(n, d, clusters=40, seed=0):
    """Gaussian mixture: the data shape that breaks naive HNSW neighbour selection."""
    r = np.random.default_rng(seed)
    centers = r.normal(0, 3, (clusters, d))
    X = centers[r.integers(0, clusters, n)] + r.normal(0, 1, (n, d))
    Q = centers[r.integers(0, clusters, 60)] + r.normal(0, 1, (60, d))
    return X.astype(np.float32), Q.astype(np.float32)


def recall(index, truth_index, Q, k=10, **params):
    hits = 0
    for q in Q:
        want = set(truth_index.search(q, k)[0].tolist())
        got = set(index.search(q, k, **params)[0].tolist())
        hits += len(want & got) / max(1, len(want))
    return hits / len(Q)


@pytest.fixture(scope="session")
def data():
    return clustered(1500, 24)
