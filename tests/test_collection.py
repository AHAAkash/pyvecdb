import threading

import numpy as np
import pytest

from pyvecdb import Collection


def make(n=50, index="hnsw", **kw):
    r = np.random.default_rng(0)
    col = Collection(8, "cosine", index, **kw)
    col.add([f"d{i}" for i in range(n)], r.normal(size=(n, 8)),
            [{"parity": "even" if i % 2 == 0 else "odd", "i": i} for i in range(n)])
    return col, r


def test_add_search_get_delete():
    col, r = make()
    v = col.vector("d7")
    assert col.search(v, 1)[0].id == "d7"
    assert col.get("d7").payload["i"] == 7
    assert col.delete(["d7", "nope"]) == 1
    assert "d7" not in col and len(col) == 49
    assert all(h.id != "d7" for h in col.search(v, 10))


def test_upsert_replaces_payload_and_vector():
    col, _ = make(10)
    col.add(["d3"], [[1, 0, 0, 0, 0, 0, 0, 0]], [{"i": 999}])
    assert len(col) == 10 and col.get("d3").payload["i"] == 999
    assert col.search([1, 0, 0, 0, 0, 0, 0, 0], 1)[0].id == "d3"


def test_where_filter_widens_search_until_k_matches():
    col, r = make(200)
    hits = col.search(r.normal(size=8), k=10, where={"parity": "odd"})
    assert len(hits) == 10 and all(h.payload["parity"] == "odd" for h in hits)
    hits = col.search(r.normal(size=8), k=5, where={"i": [3, 4, 5]})
    assert sorted(h.payload["i"] for h in hits) == [3, 4, 5]
    assert col.search(r.normal(size=8), k=5, where={"parity": "none"}) == []


def test_compact_reclaims_tombstones_and_keeps_results():
    col, _ = make(60, M=8, ef_construction=50)
    col.delete([f"d{i}" for i in range(0, 60, 3)])
    probe = col.vector("d1")
    before = [h.id for h in col.search(probe, 5)]
    assert col.compact() == 20
    assert col.stats()["tombstones"] == 0 and len(col) == 40
    assert [h.id for h in col.search(probe, 5)] == before
    assert col.compact() == 0


@pytest.mark.parametrize("index", ["bruteforce", "hnsw", "ivf"])
def test_save_load_roundtrip(tmp_path, index):
    kw = {"nlist": 4, "nprobe": 4} if index == "ivf" else {}
    col, r = make(80, index, **kw)
    col.delete(["d2"])
    col.extra["note"] = {"hello": "world"}
    path = tmp_path / "snap.npz"
    col.save(path)
    assert not (tmp_path / "snap.npz.tmp").exists()
    back = Collection.load(path)
    q = r.normal(size=8)
    assert [h.id for h in col.search(q, 10)] == [h.id for h in back.search(q, 10)]
    assert back.extra == {"note": {"hello": "world"}} and "d2" not in back
    assert back.get("d5").payload == col.get("d5").payload
    back.add(["new"], r.normal(size=(1, 8)))
    assert "new" in back


def test_rejects_bad_input():
    col, r = make(5)
    with pytest.raises(ValueError):
        col.add(["a", "a"], r.normal(size=(2, 8)))
    with pytest.raises(ValueError):
        col.add(["a"], r.normal(size=(2, 8)))


def test_concurrent_writers_and_readers_do_not_corrupt_state():
    col = Collection(8, "cosine", "hnsw", M=8, ef_construction=40)
    errors = []

    def writer(w):
        try:
            r = np.random.default_rng(w)
            for i in range(40):
                col.add([f"{w}-{i}"], r.normal(size=(1, 8)))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def reader():
        try:
            r = np.random.default_rng(99)
            for _ in range(60):
                col.search(r.normal(size=8), 5)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(3)] + \
              [threading.Thread(target=reader) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(col) == 120
