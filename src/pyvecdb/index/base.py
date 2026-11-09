"""Shared storage, deletion and (de)serialisation logic for every index."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..metrics import Metric

_EMPTY = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))


class Index(ABC):
    """An index owns its vectors. Rows are dense integers 0..n-1 and never move;
    deletes are tombstones (see ``Collection.compact`` to reclaim space).

    Instances are not thread-safe; ``Collection`` serialises access.
    """

    kind: str = ""

    def __init__(self, dim: int, metric: str = "cosine"):
        if int(dim) < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)
        self.metric = Metric(metric)
        self._X = np.empty((0, self.dim), dtype=np.float32)
        self._deleted = np.zeros(0, dtype=bool)
        self._n = 0
        self._n_deleted = 0

    # ---- storage ---------------------------------------------------------
    def _reserve(self, extra: int) -> None:
        need, cap = self._n + extra, self._X.shape[0]
        if need <= cap:
            return
        new_cap = max(need, cap * 2, 1024)
        X = np.empty((new_cap, self.dim), dtype=np.float32)
        X[: self._n] = self._X[: self._n]
        D = np.zeros(new_cap, dtype=bool)
        D[: self._n] = self._deleted[: self._n]
        self._X, self._deleted = X, D

    @property
    def vectors(self) -> np.ndarray:
        return self._X[: self._n]

    @property
    def deleted_mask(self) -> np.ndarray:
        return self._deleted[: self._n]

    @property
    def n_total(self) -> int:
        return self._n

    def __len__(self) -> int:
        return self._n - self._n_deleted

    # ---- public API ------------------------------------------------------
    def add(self, vectors) -> np.ndarray:
        V = np.asarray(vectors, dtype=np.float32)
        if V.ndim == 1:
            V = V[None, :]
        if V.ndim != 2 or V.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {V.shape}")
        if not np.isfinite(V).all():
            raise ValueError("vectors must not contain NaN or inf")
        V = self.metric.prepare(V)
        self._reserve(len(V))
        start = self._n
        self._X[start : start + len(V)] = V
        self._n += len(V)
        rows = np.arange(start, self._n, dtype=np.int64)
        self._insert(rows)
        return rows

    def delete(self, rows) -> int:
        removed = 0
        for r in np.asarray(rows, dtype=np.int64).reshape(-1):
            if 0 <= r < self._n and not self._deleted[r]:
                self._deleted[r] = True
                removed += 1
        self._n_deleted += removed
        return removed

    def search(self, q, k: int = 10, **params):
        """Return ``(rows, distances)`` sorted by increasing distance."""
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query has dim {q.shape[0]}, index has dim {self.dim}")
        if not np.isfinite(q).all():
            raise ValueError("query must not contain NaN or inf")
        k = min(int(k), len(self))
        if k <= 0:
            return _EMPTY
        rows, d = self._search(self.metric.prepare(q), k, **params)
        return np.asarray(rows, dtype=np.int64), np.asarray(d, dtype=np.float32)

    def _top_k(self, q: np.ndarray, rows: np.ndarray, k: int):
        """Exact top-k among candidate ``rows``, skipping tombstones."""
        rows = rows[~self._deleted[rows]]
        if rows.size == 0:
            return _EMPTY
        d = self.metric.dist(q, self._X[rows])
        if k < d.size:
            part = np.argpartition(d, k - 1)[:k]
        else:
            part = np.arange(d.size)
        order = part[np.argsort(d[part], kind="stable")]
        return rows[order], d[order]

    # ---- to be provided by subclasses -----------------------------------
    @abstractmethod
    def _insert(self, rows: np.ndarray) -> None: ...

    @abstractmethod
    def _search(self, q: np.ndarray, k: int, **params): ...

    def params(self) -> dict:
        """Constructor keyword arguments (JSON-serialisable), minus dim/metric."""
        return {}

    def _extra_state(self) -> tuple[dict, dict]:
        return {}, {}

    def _load_extra(self, cfg: dict, arrays: dict) -> None:
        pass

    # ---- persistence -----------------------------------------------------
    def state(self) -> tuple[dict, dict]:
        extra_cfg, extra_arrays = self._extra_state()
        cfg = {"kind": self.kind, "dim": self.dim, "metric": self.metric.name,
               "params": self.params(), **extra_cfg}
        arrays = {"vectors": self.vectors, "deleted": self.deleted_mask, **extra_arrays}
        return cfg, arrays

    @classmethod
    def from_state(cls, cfg: dict, arrays: dict) -> "Index":
        idx = cls(cfg["dim"], cfg["metric"], **cfg.get("params", {}))
        V = np.asarray(arrays["vectors"], dtype=np.float32)
        idx._reserve(len(V))
        idx._X[: len(V)] = V
        idx._deleted[: len(V)] = np.asarray(arrays["deleted"], dtype=bool)
        idx._n = len(V)
        idx._n_deleted = int(idx._deleted[: idx._n].sum())
        idx._load_extra(cfg, arrays)
        return idx
