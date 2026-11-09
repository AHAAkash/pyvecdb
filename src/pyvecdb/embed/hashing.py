from __future__ import annotations

import math
import re
import zlib
from collections import Counter

import numpy as np

from .base import Embedder

_WORD = re.compile(r"\w+", re.UNICODE)
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to "
    "was were what when where which who why will with you your do does did can could should "
    "would about into than then there their they them we our not no".split())


class HashingEmbedder(Embedder):
    """Dependency-free, deterministic, offline embedder (signed feature hashing of
    words and character n-grams).

    It captures *lexical* similarity (shared words, word stems, typos), not meaning, so
    it is meant for tests, demos and CI. Use ``OllamaEmbedder`` or
    ``SentenceTransformerEmbedder`` when semantic similarity matters.
    """

    def __init__(self, dim: int = 512, ngrams: tuple[int, ...] = (3, 4, 5)):
        if dim < 16:
            raise ValueError("dim must be >= 16")
        self.dim, self.ngrams = int(dim), tuple(ngrams)
        self.name = f"hashing:{self.dim}"

    def _features(self, text: str) -> Counter:
        feats: Counter = Counter()
        for w in _WORD.findall(text.lower()):
            if w in STOPWORDS:
                continue
            feats["w:" + w] += 2
            padded = f"<{w}>"
            for n in self.ngrams:
                for i in range(len(padded) - n + 1):
                    feats[padded[i : i + n]] += 1
        return feats

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, count in self._features(text).items():
                h = zlib.crc32(feat.encode())
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[row, h % self.dim] += sign * (1.0 + math.log(count))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)
