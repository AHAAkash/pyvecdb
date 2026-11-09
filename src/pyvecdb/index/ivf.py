from __future__ import annotations

import numpy as np

from ..metrics import Metric
from .base import Index


def _nearest(X: np.ndarray, C: np.ndarray, metric: Metric, chunk: int = 8192) -> np.ndarray:
    out = np.empty(len(X), dtype=np.int32)
    for s in range(0, len(X), chunk):
        out[s : s + chunk] = metric.pairwise(X[s : s + chunk], C).argmin(axis=1)
    return out


def kmeans(X: np.ndarray, k: int, metric: Metric, iters: int = 15, seed: int = 0) -> np.ndarray:
    """Lloyd's algorithm (spherical for cosine). Empty clusters are re-seeded."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(iters):
        assign = _nearest(X, C, metric)
        counts = np.bincount(assign, minlength=k)
        order = np.argsort(assign, kind="stable")
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        nz = counts > 0
        sums = np.add.reduceat(X[order], starts[nz], axis=0)
        C[nz] = sums / counts[nz, None]
        if (~nz).any():
            C[~nz] = X[rng.choice(len(X), size=int((~nz).sum()), replace=False)]
        C = metric.prepare(C) if metric.name == "cosine" else C
    return C.astype(np.float32)


class IVFFlat(Index):
    """Inverted-file index: k-means partitions, then exact search inside the
    ``nprobe`` closest partitions. Falls back to exact search until it has
    enough vectors (>= 8 per list) to train on."""

    kind = "ivf"

    def __init__(self, dim, metric="cosine", nlist=64, nprobe=8, seed=0):
        super().__init__(dim, metric)
        self.nlist, self.nprobe, self.seed = int(nlist), int(nprobe), int(seed)
        self._centroids: np.ndarray | None = None
        self._lists: list[list[int]] = []

    def params(self):
        return {"nlist": self.nlist, "nprobe": self.nprobe, "seed": self.seed}

    def train(self, max_sample: int = 100_000) -> None:
        live = np.flatnonzero(~self.deleted_mask)
        if live.size == 0:
            raise ValueError("cannot train an empty index")
        rng = np.random.default_rng(self.seed)
        if live.size > max_sample:
            live = rng.choice(live, size=max_sample, replace=False)
        k = min(self.nlist, live.size)
        self._centroids = kmeans(self._X[live], k, self.metric, seed=self.seed)
        self._lists = [[] for _ in range(k)]
        self._assign_rows(np.arange(self._n))

    def _assign_rows(self, rows: np.ndarray) -> None:
        cells = _nearest(self._X[rows], self._centroids, self.metric)
        for r, c in zip(rows.tolist(), cells.tolist()):
            self._lists[c].append(r)

    def _insert(self, rows):
        if self._centroids is not None:
            self._assign_rows(rows)
        elif self._n >= self.nlist * 8:
            self.train()

    def _search(self, q, k, nprobe=None, **_):
        if self._centroids is None:
            return self._top_k(q, np.arange(self._n), k)
        nprobe = max(1, int(nprobe or self.nprobe))
        order = np.argsort(self.metric.dist(q, self._centroids))
        picked, live = [], 0
        for c in order:  # keep probing past nprobe if the lists are too sparse
            lst = np.asarray(self._lists[c], dtype=np.int64)
            picked.append(lst)
            live += int((~self._deleted[lst]).sum())
            if len(picked) >= nprobe and live >= k:
                break
        return self._top_k(q, np.concatenate(picked), k)

    def _extra_state(self):
        if self._centroids is None:
            return {"trained": False}, {}
        assign = np.full(self._n, -1, dtype=np.int32)
        for c, lst in enumerate(self._lists):
            assign[lst] = c
        return {"trained": True}, {"centroids": self._centroids, "assign": assign}

    def _load_extra(self, cfg, arrays):
        if not cfg.get("trained"):
            return
        self._centroids = np.asarray(arrays["centroids"], dtype=np.float32)
        self._lists = [[] for _ in range(len(self._centroids))]
        for r, c in enumerate(np.asarray(arrays["assign"]).tolist()):
            if c >= 0:
                self._lists[c].append(r)
