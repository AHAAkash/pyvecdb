"""Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2018).

Two details matter for recall and are easy to get wrong:

* **Neighbour selection.** Linking a node to its M *nearest* candidates is not
  enough: on clustered data every node links inside its own cluster and the
  clusters become islands. The paper's heuristic (Algorithm 4) only keeps a
  candidate if it is closer to the new node than to every neighbour already
  chosen, which forces some links to point *across* clusters.
  ``heuristic=False`` restores the naive rule so the effect can be measured.
* **Deletes.** Removing a node from the graph can disconnect it, so deletes are
  tombstones: dead nodes still route traffic but are never returned.
"""
from __future__ import annotations

import math
from heapq import heappop, heappush

import numpy as np

from .base import Index


class HNSW(Index):
    kind = "hnsw"

    def __init__(self, dim, metric="cosine", M=16, ef_construction=200, ef_search=64,
                 heuristic=True, seed=42):
        super().__init__(dim, metric)
        if M < 2:
            raise ValueError("M must be >= 2")
        self.M, self.M0 = int(M), 2 * int(M)
        self.ef_construction, self.ef_search = int(ef_construction), int(ef_search)
        self.heuristic, self.seed = bool(heuristic), int(seed)
        self._mL = 1.0 / math.log(self.M)
        self._rng = np.random.default_rng(self.seed)
        self._levels: list[int] = []
        self._nbrs: list[list[list[int]]] = []  # _nbrs[node][level] -> neighbour ids
        self._entry, self._max_level = -1, -1

    def params(self):
        return {"M": self.M, "ef_construction": self.ef_construction,
                "ef_search": self.ef_search, "heuristic": self.heuristic, "seed": self.seed}

    # ---- graph search ------------------------------------------------------
    def _search_layer(self, q, eps, ef, level):
        """Best-first search on one layer. ``eps`` is a list of (dist, id).
        Returns up to ``ef`` (dist, id) pairs sorted by increasing distance."""
        X, dist = self._X, self.metric.dist
        seen = np.zeros(self._n, dtype=bool)
        cand, res = [], []
        for d, e in eps:
            seen[e] = True
            heappush(cand, (d, e))
            heappush(res, (-d, e))
        while len(res) > ef:
            heappop(res)
        while cand:
            d, c = heappop(cand)
            if len(res) >= ef and d > -res[0][0]:
                break
            nb = self._nbrs[c][level]
            if not nb:
                continue
            nb = np.asarray(nb, dtype=np.int64)
            nb = nb[~seen[nb]]
            if nb.size == 0:
                continue
            seen[nb] = True
            for dd, nn in zip(dist(q, X[nb]).tolist(), nb.tolist()):
                if len(res) < ef or dd < -res[0][0]:
                    heappush(cand, (dd, nn))
                    heappush(res, (-dd, nn))
                    if len(res) > ef:
                        heappop(res)
        return sorted((-nd, i) for nd, i in res)

    def _descend(self, q, down_to):
        """Greedy walk from the entry point to layer ``down_to`` (exclusive)."""
        ep = self._entry
        eps = [(float(self.metric.dist(q, self._X[ep : ep + 1])[0]), ep)]
        for lc in range(self._max_level, down_to, -1):
            eps = self._search_layer(q, eps, 1, lc)[:1]
        return eps

    def _select(self, cands, m):
        """Choose up to ``m`` neighbours from ``cands`` [(dist_to_query, id)] sorted
        ascending (Algorithm 4 with keepPrunedConnections)."""
        if len(cands) <= m:
            return [i for _, i in cands]
        if not self.heuristic:
            return [i for _, i in cands[:m]]
        ids = [i for _, i in cands]
        ds = [d for d, _ in cands]
        Xc = self._X[ids]
        P = self.metric.pairwise(Xc, Xc)  # one matmul instead of a call per candidate
        chosen, pruned = [], []
        for j in range(len(ids)):
            if len(chosen) >= m:
                break
            if chosen and (P[j, chosen] <= ds[j]).any():
                pruned.append(j)  # an already-chosen neighbour covers this one
            else:
                chosen.append(j)
        for j in pruned:  # top up with the closest pruned candidates
            if len(chosen) >= m:
                break
            chosen.append(j)
        return [ids[j] for j in chosen]

    # ---- insertion ----------------------------------------------------------
    def _insert(self, rows):
        for r in rows.tolist():
            self._insert_one(r)

    def _insert_one(self, node):
        level = int(-math.log(1.0 - self._rng.random()) * self._mL)
        self._levels.append(level)
        self._nbrs.append([[] for _ in range(level + 1)])
        if self._entry < 0:
            self._entry, self._max_level = node, level
            return
        q = self._X[node]
        eps = self._descend(q, level)
        for lc in range(min(level, self._max_level), -1, -1):
            W = self._search_layer(q, eps, self.ef_construction, lc)
            cap = self.M0 if lc == 0 else self.M
            neigh = self._select(W, self.M)
            self._nbrs[node][lc] = neigh
            for nb in neigh:
                lst = self._nbrs[nb][lc]
                lst.append(node)
                if len(lst) > cap:
                    dn = self.metric.dist(self._X[nb], self._X[lst])
                    order = np.argsort(dn, kind="stable")
                    self._nbrs[nb][lc] = self._select([(float(dn[i]), lst[i]) for i in order], cap)
            eps = W
        if level > self._max_level:
            self._entry, self._max_level = node, level

    # ---- query ---------------------------------------------------------------
    def _search(self, q, k, ef=None, **_):
        ef = max(int(ef or self.ef_search), k)
        while True:
            W = self._search_layer(q, self._descend(q, 0), ef, 0)
            hits = [(d, i) for d, i in W if not self._deleted[i]]
            # tombstones can crowd out live results: widen the beam and retry
            if len(hits) >= k or ef >= self._n:
                break
            ef = min(self._n, ef * 2)
        hits = hits[:k]
        return np.array([i for _, i in hits], dtype=np.int64), np.array([d for d, _ in hits])

    # ---- persistence ---------------------------------------------------------
    def _extra_state(self):
        n = len(self._nbrs)
        arrays = {"levels": np.asarray(self._levels, dtype=np.int32)}
        for L in range(self._max_level + 1):
            lens = np.fromiter((len(nb[L]) if len(nb) > L else 0 for nb in self._nbrs),
                               dtype=np.int64, count=n)
            ptr = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(lens, out=ptr[1:])
            arrays[f"ptr_{L}"] = ptr
            arrays[f"data_{L}"] = np.fromiter(
                (x for nb in self._nbrs if len(nb) > L for x in nb[L]),
                dtype=np.int32, count=int(ptr[-1]))
        return {"entry": self._entry, "max_level": self._max_level}, arrays

    def _load_extra(self, cfg, arrays):
        self._levels = np.asarray(arrays["levels"]).tolist()
        self._nbrs = [[[] for _ in range(lv + 1)] for lv in self._levels]
        self._entry, self._max_level = int(cfg["entry"]), int(cfg["max_level"])
        for L in range(self._max_level + 1):
            ptr, data = np.asarray(arrays[f"ptr_{L}"]), np.asarray(arrays[f"data_{L}"]).tolist()
            for node, lv in enumerate(self._levels):
                if lv >= L:
                    self._nbrs[node][L] = data[ptr[node] : ptr[node + 1]]

    def stats(self) -> dict:
        per_level = [sum(1 for lv in self._levels if lv >= L) for L in range(self._max_level + 1)]
        edges0 = sum(len(nb[0]) for nb in self._nbrs)
        return {"nodes_per_level": per_level, "avg_degree_l0": edges0 / max(1, len(self._nbrs))}
