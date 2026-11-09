from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from ..db import Collection
from ..embed import Embedder
from .bm25 import BM25

MODES = ("dense", "bm25", "hybrid")


@dataclass
class Retrieved:
    chunk_id: str
    doc_id: str
    title: str
    chunk_index: int
    text: str
    score: float                    # what the ranking used: RRF, similarity or BM25
    similarity: float | None = None  # cosine similarity if the chunk was a dense hit
    bm25: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def rrf(rank_lists: Sequence[Sequence[str]], k: int = 60,
        weights: Sequence[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): ``sum_i w_i / (k + rank_i)``.
    Rank-based, so it needs no score calibration between BM25 and cosine."""
    weights = weights or [1.0] * len(rank_lists)
    fused: dict[str, float] = {}
    for ranks, w in zip(rank_lists, weights):
        for pos, id_ in enumerate(ranks, start=1):
            fused[id_] = fused.get(id_, 0.0) + w / (k + pos)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def mmr_select(query: np.ndarray, vectors: np.ndarray, k: int, lam: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance over unit vectors: trade relevance for diversity.
    ``lam=1`` is plain relevance ranking. Returns indices into ``vectors``."""
    if len(vectors) == 0:
        return []
    rel = vectors @ query
    chosen: list[int] = []
    remaining = list(range(len(vectors)))
    while remaining and len(chosen) < k:
        if chosen:
            red = (vectors[remaining] @ vectors[chosen].T).max(axis=1)
        else:
            red = np.zeros(len(remaining))
        score = lam * rel[remaining] - (1 - lam) * red
        chosen.append(remaining.pop(int(np.argmax(score))))
    return chosen


class HybridRetriever:
    def __init__(self, collection: Callable[[], Collection | None], bm25: BM25, embedder: Embedder):
        self._collection, self.bm25, self.embedder = collection, bm25, embedder

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid", candidates: int = 30,
                 min_similarity: float = 0.0, mmr_lambda: float | None = None,
                 reranker: Callable[[str, list[str]], Sequence[float]] | None = None,
                 doc_ids: Sequence[str] | None = None) -> list[Retrieved]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        col = self._collection()
        if col is None or len(col) == 0 or not query.strip():
            return []
        allowed = set(doc_ids) if doc_ids else None
        qv = self.embedder.embed_query(query) if (mode != "bm25" or mmr_lambda) else None

        dense: list[tuple[str, float]] = []
        if mode in ("dense", "hybrid"):
            where = {"doc_id": sorted(allowed)} if allowed else None
            for h in col.search(qv, k=candidates, where=where):
                if 1.0 - h.distance >= min_similarity:
                    dense.append((h.id, 1.0 - h.distance))
        sparse: list[tuple[str, float]] = []
        if mode in ("bm25", "hybrid"):
            pred = (lambda cid: cid.split(":")[0] in allowed) if allowed else None
            sparse = self.bm25.search(query, candidates, allowed=pred)

        sim, bm = dict(dense), dict(sparse)
        if mode == "dense":
            ranked = dense
        elif mode == "bm25":
            ranked = sparse
        else:
            ranked = rrf([[i for i, _ in dense], [i for i, _ in sparse]])
        ids = [i for i, _ in ranked][:candidates]

        if reranker and ids:
            texts = [col.get(i).payload["text"] for i in ids]
            order = np.argsort(-np.asarray(reranker(query, texts), dtype=float), kind="stable")
            ids = [ids[j] for j in order]
        if mmr_lambda and ids:
            order = mmr_select(qv, np.stack([col.vector(i) for i in ids]), k, mmr_lambda)
            ids = [ids[j] for j in order]

        score = dict(ranked)
        out = []
        for cid in ids[:k]:
            p = col.get(cid).payload
            out.append(Retrieved(cid, p["doc_id"], p["title"], p["chunk_index"], p["text"],
                                 float(score.get(cid, 0.0)), sim.get(cid), bm.get(cid)))
        return out
