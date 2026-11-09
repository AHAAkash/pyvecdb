import numpy as np
import pytest
from conftest import recall

from pyvecdb.index import HNSW, BruteForce, IVFFlat, index_from_state, make_index
from pyvecdb.index.ivf import kmeans
from pyvecdb.metrics import Metric


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_bruteforce_matches_numpy(metric):
    r = np.random.default_rng(1)
    X, q = r.normal(size=(300, 8)).astype(np.float32), r.normal(size=8).astype(np.float32)
    idx = BruteForce(8, metric)
    idx.add(X)
    rows, d = idx.search(q, 5)
    m = Metric(metric)
    expect = np.argsort(m.dist(m.prepare(q), m.prepare(X)))[:5]
    assert rows.tolist() == expect.tolist()
    assert np.all(np.diff(d) >= -1e-6)


def test_cosine_ignores_vector_length():
    r = np.random.default_rng(2)
    X = r.normal(size=(200, 12)).astype(np.float32)
    idx = HNSW(12, "cosine")
    idx.add(X)
    q = r.normal(size=12).astype(np.float32)
    assert idx.search(q, 5)[0].tolist() == idx.search(q * 37.5, 5)[0].tolist()


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_hnsw_recall_on_clustered_data(data, metric):
    X, Q = data
    truth, h = BruteForce(24, metric), HNSW(24, metric, M=16, ef_construction=100)
    truth.add(X)
    h.add(X)
    assert recall(h, truth, Q, ef=64) >= 0.97


def test_neighbour_heuristic_beats_naive_selection(data):
    """The ablation behind the README: nearest-M selection fragments clusters."""
    X, Q = data
    truth = BruteForce(24, "l2")
    truth.add(X)
    scores = {}
    for heuristic in (False, True):
        h = HNSW(24, "l2", M=8, ef_construction=40, heuristic=heuristic)
        h.add(X)
        scores[heuristic] = recall(h, truth, Q, ef=10)
    assert scores[True] >= 0.97
    assert scores[True] >= scores[False]


def test_hnsw_deleted_rows_are_never_returned(data):
    X, Q = data
    h, truth = HNSW(24, "cosine", M=16, ef_construction=100), BruteForce(24, "cosine")
    h.add(X)
    truth.add(X)
    dead = np.random.default_rng(3).choice(len(X), size=len(X) * 3 // 10, replace=False)
    h.delete(dead)
    truth.delete(dead)
    dead_set = set(dead.tolist())
    for q in Q:
        assert not dead_set & set(h.search(q, 10, ef=64)[0].tolist())
    assert recall(h, truth, Q, ef=64) >= 0.95


def test_search_returns_everything_alive_when_k_exceeds_size():
    idx = HNSW(4, "l2")
    idx.add(np.eye(4, dtype=np.float32))
    idx.delete([0])
    rows, _ = idx.search(np.ones(4), 10)
    assert sorted(rows.tolist()) == [1, 2, 3]
    assert idx.search(np.ones(4), 0)[0].size == 0


def test_empty_index_search_is_empty():
    for kind in ("bruteforce", "hnsw", "ivf"):
        assert make_index(kind, 4).search(np.ones(4), 3)[0].size == 0


def test_ivf_exact_before_training_then_accurate_after(data):
    X, Q = data
    truth, ivf = BruteForce(24, "cosine"), IVFFlat(24, "cosine", nlist=20, nprobe=6)
    ivf.add(X[:100])  # < nlist * 8: not trained yet
    truth.add(X[:100])
    assert ivf._centroids is None and recall(ivf, truth, Q) == 1.0
    ivf.add(X[100:])
    truth.add(X[100:])
    assert ivf._centroids is not None and recall(ivf, truth, Q) >= 0.9


def test_ivf_probes_past_nprobe_when_lists_are_sparse(data):
    X, _ = data
    ivf = IVFFlat(24, "l2", nlist=30, nprobe=1)
    ivf.add(X)
    assert len(ivf.search(X[0], 50)[0]) == 50


def test_kmeans_returns_k_distinct_centroids(data):
    X, _ = data
    C = kmeans(X, 12, Metric("l2"))
    assert C.shape == (12, 24) and len({tuple(np.round(c, 3)) for c in C}) == 12


@pytest.mark.parametrize("kind,params", [("bruteforce", {}), ("hnsw", {"M": 8, "ef_construction": 50}),
                                         ("ivf", {"nlist": 10, "nprobe": 3})])
def test_state_roundtrip_gives_identical_results(kind, params, data):
    X, Q = data
    idx = make_index(kind, 24, "cosine", **params)
    idx.add(X[:600])
    idx.delete([5, 17, 99])
    cfg, arrays = idx.state()
    clone = index_from_state(cfg, {k: np.array(v) for k, v in arrays.items()})
    assert len(clone) == len(idx)
    for q in Q[:20]:
        a, b = idx.search(q, 8), clone.search(q, 8)
        assert a[0].tolist() == b[0].tolist() and np.allclose(a[1], b[1])
    clone.add(X[600:700])  # a restored index must keep accepting writes
    assert len(clone) == len(idx) + 100


def test_input_validation():
    idx = HNSW(4)
    with pytest.raises(ValueError):
        idx.add(np.ones((2, 5)))
    with pytest.raises(ValueError):
        idx.add([[1, 2, np.nan, 4]])
    idx.add(np.ones((1, 4)))
    with pytest.raises(ValueError):
        idx.search(np.ones(7), 1)
    with pytest.raises(ValueError):
        make_index("annoy", 4)
    with pytest.raises(ValueError):
        Metric("hamming")
