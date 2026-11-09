"""``Collection``: ids + payloads + metadata filtering + snapshots on top of an index."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field

import numpy as np

from .index import Index, index_from_state, make_index

FORMAT = 1


@dataclass
class Hit:
    id: str
    distance: float
    payload: dict = field(default_factory=dict)


def _matches(payload: dict, where: dict) -> bool:
    """Equality filter. A list value means "any of"."""
    for key, want in where.items():
        have = payload.get(key)
        if isinstance(want, (list, tuple, set)):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


class Collection:
    """Thread-safe vector collection.

    >>> col = Collection(dim=3, metric="cosine", index="hnsw")
    >>> _ = col.add(["a", "b"], [[1, 0, 0], [0, 1, 0]], [{"tag": "x"}, {"tag": "y"}])
    >>> col.search([1, 0.1, 0], k=1)[0].id
    'a'

    ``extra`` is a free-form JSON-serialisable dict that is saved with the snapshot.
    """

    def __init__(self, dim: int, metric: str = "cosine", index: str = "hnsw", **index_params):
        self.index: Index = make_index(index, dim, metric, **index_params)
        self.extra: dict = {}
        self.version = 0
        self._ids: list[str] = []
        self._payloads: list[dict | None] = []
        self._row_of: dict[str, int] = {}
        self._lock = threading.RLock()

    # ---- basic properties -----------------------------------------------------
    @property
    def dim(self) -> int:
        return self.index.dim

    def __len__(self) -> int:
        return len(self._row_of)

    def __contains__(self, id_) -> bool:
        return str(id_) in self._row_of

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._row_of)

    def stats(self) -> dict:
        with self._lock:
            return {"dim": self.dim, "metric": self.index.metric.name, "index": self.index.kind,
                    "live": len(self), "tombstones": self.index.n_total - len(self),
                    "version": self.version, "params": self.index.params()}

    # ---- writes ---------------------------------------------------------------
    def add(self, ids, vectors, payloads=None) -> list[str]:
        """Insert vectors; an existing id is replaced (upsert)."""
        ids = [str(i) for i in ids]
        V = np.asarray(vectors, dtype=np.float32)
        if V.ndim == 1:
            V = V[None, :]
        if len(ids) != len(V):
            raise ValueError("ids and vectors must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in one call")
        payloads = [dict(p) for p in payloads] if payloads is not None else [{} for _ in ids]
        if len(payloads) != len(ids):
            raise ValueError("payloads must match ids")
        with self._lock:
            stale = [self._row_of[i] for i in ids if i in self._row_of]
            rows = self.index.add(V)  # may raise: nothing else has changed yet
            if stale:
                self.index.delete(stale)
                for r in stale:
                    self._payloads[r] = None
            for i, r, p in zip(ids, rows.tolist(), payloads):
                self._ids.append(i)
                self._payloads.append(p)
                self._row_of[i] = r
            self.version += 1
        return ids

    def delete(self, ids) -> int:
        with self._lock:
            rows = [self._row_of.pop(str(i)) for i in ids if str(i) in self._row_of]
            self.index.delete(rows)
            for r in rows:
                self._payloads[r] = None
            if rows:
                self.version += 1
            return len(rows)

    # ---- reads ----------------------------------------------------------------
    def get(self, id_) -> Hit | None:
        with self._lock:
            r = self._row_of.get(str(id_))
            return None if r is None else Hit(self._ids[r], 0.0, dict(self._payloads[r]))

    def vector(self, id_) -> np.ndarray:
        """The stored (normalised, for cosine) vector."""
        with self._lock:
            return self.index.vectors[self._row_of[str(id_)]].copy()

    def live_vectors(self) -> tuple[list[str], np.ndarray]:
        with self._lock:
            rows = np.flatnonzero(~self.index.deleted_mask)
            return [self._ids[r] for r in rows], self.index.vectors[rows].copy()

    def search(self, vector, k: int = 10, where: dict | None = None, **params) -> list[Hit]:
        """Nearest neighbours. ``where`` filters on payload equality (post-filter: the
        search is widened until ``k`` matches are found or the collection is exhausted)."""
        with self._lock:
            live = len(self)
            if live == 0 or k <= 0:
                return []
            fetch = k if not where else max(4 * k, 32)
            while True:
                rows, dists = self.index.search(vector, min(fetch, live), **params)
                hits = [Hit(self._ids[r], float(d), dict(self._payloads[r]))
                        for r, d in zip(rows.tolist(), dists.tolist())]
                if where:
                    hits = [h for h in hits if _matches(h.payload, where)]
                if len(hits) >= k or fetch >= live:
                    return hits[:k]
                fetch *= 4

    # ---- maintenance ----------------------------------------------------------
    def compact(self) -> int:
        """Rebuild the index without tombstones. Returns the number of rows reclaimed."""
        with self._lock:
            reclaimed = self.index.n_total - len(self)
            if reclaimed == 0:
                return 0
            keep = np.flatnonzero(~self.index.deleted_mask)
            fresh = make_index(self.index.kind, self.dim, self.index.metric.name,
                               **self.index.params())
            fresh.add(self.index.vectors[keep])
            self._ids = [self._ids[r] for r in keep]
            self._payloads = [self._payloads[r] for r in keep]
            self._row_of = {i: n for n, i in enumerate(self._ids)}
            self.index = fresh
            self.version += 1
            return reclaimed

    # ---- persistence ----------------------------------------------------------
    def save(self, path: str | os.PathLike) -> None:
        """Atomically write one self-contained ``.npz`` snapshot (no pickle)."""
        path = os.fspath(path)
        with self._lock:
            cfg, arrays = self.index.state()
            meta = {"format": FORMAT, "index": cfg, "ids": self._ids, "payloads": self._payloads,
                    "version": self.version, "extra": self.extra}
            blob = np.frombuffer(json.dumps(meta, separators=(",", ":")).encode(), dtype=np.uint8)
            out = {f"idx__{k}": v for k, v in arrays.items()}
            out["meta"] = blob
            tmp = f"{path}.tmp"
            with open(tmp, "wb") as f:
                np.savez(f, **out)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Collection":
        with np.load(os.fspath(path), allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            if meta.get("format") != FORMAT:
                raise ValueError(f"unsupported snapshot format {meta.get('format')!r}")
            arrays = {k[5:]: z[k] for k in z.files if k.startswith("idx__")}
        col = cls.__new__(cls)
        col.index = index_from_state(meta["index"], arrays)
        col.extra = meta.get("extra", {})
        col.version = meta.get("version", 0)
        col._ids = list(meta["ids"])
        col._payloads = list(meta["payloads"])
        col._row_of = {i: r for r, i in enumerate(col._ids) if col._payloads[r] is not None}
        col._lock = threading.RLock()
        return col
