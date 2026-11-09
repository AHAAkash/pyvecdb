from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..db import Collection
from ..embed import Embedder
from .bm25 import BM25
from .chunking import chunk_text
from .generate import ABSTAIN, ExtractiveGenerator, extract_citations
from .retriever import HybridRetriever, Retrieved

_DOC_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class _QueryCache:
    """Embeds each distinct query once; lets slow network embedding happen outside locks."""

    def __init__(self, inner: Embedder, size: int = 256):
        self.inner, self._size = inner, size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def embed_documents(self, texts):
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
        vec = self.inner.embed_query(text)
        with self._lock:
            self._cache[text] = vec
            while len(self._cache) > self._size:
                self._cache.popitem(last=False)
        return vec


class RAGPipeline:
    """Ingest -> chunk -> embed -> (dense + BM25) -> fuse -> grounded answer.

    With ``path`` set, every change is saved atomically to that single file and the
    pipeline resumes from it on the next start.
    """

    def __init__(self, embedder: Embedder, generator=None, path: str | Path | None = None,
                 chunk_words: int = 180, chunk_overlap: int = 30, index: str = "hnsw",
                 index_params: dict | None = None, min_similarity: float = 0.0):
        self.embedder = embedder
        self.generator = generator or ExtractiveGenerator()
        self.path = Path(path) if path else None
        self.chunk_words, self.chunk_overlap = chunk_words, chunk_overlap
        self.index_kind, self.index_params = index, dict(index_params or {})
        self.min_similarity = min_similarity
        self.collection: Collection | None = None
        self.docs: dict[str, dict] = {}
        self.bm25 = BM25()
        self._qcache = _QueryCache(embedder)
        self.retriever = HybridRetriever(lambda: self.collection, self.bm25, self._qcache)
        self._lock = threading.RLock()
        self._proj: tuple | None = None
        if self.path and self.path.exists():
            self._load()

    # ---- persistence ----------------------------------------------------------
    def _load(self) -> None:
        col = Collection.load(self.path)
        saved = col.extra.get("embedder")
        if saved and saved != self.embedder.name:
            raise ValueError(f"{self.path} was built with embedder {saved!r} but {self.embedder.name!r} "
                             "is configured; use the same embedder or a new data path")
        self.collection, self.docs = col, col.extra.get("docs", {})
        col.extra["docs"] = self.docs
        for cid in col.ids():
            p = col.get(cid).payload
            self.bm25.add(cid, f"{p['title']}\n{p['text']}")

    def save(self) -> None:
        if self.path and self.collection is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.collection.save(self.path)

    # ---- documents ------------------------------------------------------------
    def ingest(self, title: str, text: str, source: str | None = None,
               doc_id: str | None = None) -> dict:
        title, text = title.strip(), text.strip()
        chunks = chunk_text(text, self.chunk_words, self.chunk_overlap)
        if not title or not chunks:
            raise ValueError("a document needs a title and some text")
        doc_id = doc_id or hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()[:12]
        if not _DOC_ID.match(doc_id):
            raise ValueError("doc_id may only contain letters, digits, '-' and '_' (max 64)")
        embed_texts = [f"{title}\n{c.text}" for c in chunks]
        vectors = self.embedder.embed_documents(embed_texts)  # slow; keep outside the lock
        with self._lock:
            if self.collection is None:
                self.collection = Collection(vectors.shape[1], "cosine", self.index_kind,
                                             **self.index_params)
                self.collection.extra.update(
                    {"embedder": self.embedder.name, "docs": self.docs})
            if doc_id in self.docs:
                self._remove(doc_id)
            ids = [f"{doc_id}:{c.index}" for c in chunks]
            payloads = [{"doc_id": doc_id, "title": title, "chunk_index": c.index,
                         "text": c.text, "source": source} for c in chunks]
            self.collection.add(ids, vectors, payloads)
            for cid, t in zip(ids, embed_texts):
                self.bm25.add(cid, t)
            self.docs[doc_id] = {"doc_id": doc_id, "title": title, "source": source,
                                 "chunks": len(chunks), "words": sum(c.words for c in chunks),
                                 "created": time.time()}
            self.save()
            return dict(self.docs[doc_id])

    def _remove(self, doc_id: str) -> None:
        ids = [f"{doc_id}:{i}" for i in range(self.docs[doc_id]["chunks"])]
        self.collection.delete(ids)
        for cid in ids:
            self.bm25.remove(cid)
        del self.docs[doc_id]

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.docs:
                return False
            self._remove(doc_id)
            stats = self.collection.stats()
            if stats["tombstones"] > 100 and stats["tombstones"] > stats["live"] // 4:
                self.collection.compact()
            self.save()
            return True

    def list_docs(self) -> list[dict]:
        with self._lock:
            return sorted((dict(d) for d in self.docs.values()), key=lambda d: d["created"])

    # ---- retrieval and answering ---------------------------------------------------
    def search(self, query: str, k: int = 5, mode: str = "hybrid", **kw) -> list[Retrieved]:
        kw.setdefault("min_similarity", self.min_similarity)
        if (mode != "bm25" or kw.get("mmr_lambda")) and query.strip():
            self._qcache.embed_query(query)  # warm the cache before taking the lock
        with self._lock:
            return self.retriever.retrieve(query, k=k, mode=mode, **kw)

    def ask(self, question: str, k: int = 4, mode: str = "hybrid", **kw) -> dict:
        t0 = time.perf_counter()
        contexts = self.search(question, k=k, mode=mode, **kw)
        t1 = time.perf_counter()
        if not contexts:
            answer, abstained, valid, invalid = ABSTAIN, True, [], []
        else:
            answer = self.generator.generate(question, contexts)
            abstained = answer.strip() == ABSTAIN
            valid, invalid = extract_citations(answer, len(contexts))
        t2 = time.perf_counter()
        return {"answer": answer, "abstained": abstained, "citations": valid,
                "invalid_citations": invalid, "contexts": [c.to_dict() for c in contexts],
                "generator": getattr(self.generator, "name", "custom"), "mode": mode,
                "retrieval_ms": round((t1 - t0) * 1000, 2), "generation_ms": round((t2 - t1) * 1000, 2)}

    # ---- 2-D map of the embedding space ----------------------------------------------
    def _projection(self):
        col = self.collection
        if self._proj and self._proj[0] == col.version:
            return self._proj
        ids, V = col.live_vectors()
        mean = V.mean(axis=0)
        Xc = V - mean
        if len(V) >= 3:
            _, U = np.linalg.eigh(Xc.T @ Xc)  # covariance eigenvectors, ascending
            comps = U[:, -2:][:, ::-1].T
        else:
            comps = np.eye(2, V.shape[1], dtype=np.float32)
        xy = Xc @ comps.T
        scale = float(np.abs(xy).max()) or 1.0
        self._proj = (col.version, ids, xy / scale, mean, comps, scale)
        return self._proj

    def project(self, text: str) -> list[float] | None:
        with self._lock:
            if self.collection is None or len(self.collection) == 0:
                return None
            _, _, _, mean, comps, scale = self._projection()
        q = self._qcache.embed_query(text)
        return ((q - mean) @ comps.T / scale).tolist()

    def map_points(self) -> list[dict]:
        with self._lock:
            if self.collection is None or len(self.collection) == 0:
                return []
            _, ids, xy, *_ = self._projection()
            pts = []
            for cid, (x, y) in zip(ids, xy.tolist()):
                p = self.collection.get(cid).payload
                pts.append({"id": cid, "doc_id": p["doc_id"], "title": p["title"],
                            "chunk_index": p["chunk_index"], "x": x, "y": y,
                            "snippet": p["text"][:160]})
            return pts

    def stats(self) -> dict:
        with self._lock:
            col = self.collection.stats() if self.collection else None
            return {"embedder": self.embedder.name, "generator": getattr(self.generator, "name", "custom"),
                    "documents": len(self.docs), "chunks": len(self.bm25), "collection": col,
                    "chunk_words": self.chunk_words, "chunk_overlap": self.chunk_overlap}
