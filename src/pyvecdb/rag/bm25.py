from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

from ..embed.hashing import STOPWORDS

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude plural folding; good enough for ranking
        out.append(w)
    return out


class BM25:
    """Okapi BM25 over an incrementally updatable inverted index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._postings: dict[str, dict[str, int]] = {}
        self._len: dict[str, int] = {}
        self._total = 0

    def __len__(self) -> int:
        return len(self._len)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._len:
            self.remove(doc_id)
        tf = Counter(tokenize(text))
        for term, n in tf.items():
            self._postings.setdefault(term, {})[doc_id] = n
        self._len[doc_id] = sum(tf.values())
        self._total += self._len[doc_id]

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._len:
            return
        for term in [t for t, p in self._postings.items() if doc_id in p]:
            del self._postings[term][doc_id]
            if not self._postings[term]:
                del self._postings[term]
        self._total -= self._len.pop(doc_id)

    def search(self, query: str, k: int = 10,
               allowed: Callable[[str], bool] | None = None) -> list[tuple[str, float]]:
        n = len(self._len)
        if n == 0:
            return []
        avg = self._total / n or 1.0
        scores: dict[str, float] = {}
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = math.log(1.0 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for doc, tf in posting.items():
                if allowed and not allowed(doc):
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self._len[doc] / avg)
                scores[doc] = scores.get(doc, 0.0) + idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
