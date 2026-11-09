"""Distance metrics. For every metric, a smaller distance means "closer".

Cosine vectors are L2-normalised once, when they enter the index, so cosine
distance is simply ``1 - dot`` and every search runs on plain matrix products.
"""
from __future__ import annotations

import numpy as np

SUPPORTED = ("cosine", "l2")


class Metric:
    __slots__ = ("name",)

    def __init__(self, name: str):
        if name not in SUPPORTED:
            raise ValueError(f"metric must be one of {SUPPORTED}, got {name!r}")
        self.name = name

    def prepare(self, x) -> np.ndarray:
        """Return float32, C-contiguous data (unit length for cosine)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.name == "cosine":
            norm = np.linalg.norm(x, axis=-1, keepdims=True)
            x = x / np.maximum(norm, 1e-12)
        return x

    def dist(self, q: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Distances from one query ``q`` (d,) to every row of ``X`` (n, d).

        ``l2`` returns the *squared* Euclidean distance (same ranking, cheaper).
        """
        if self.name == "l2":
            diff = X - q
            return np.einsum("ij,ij->i", diff, diff)
        return 1.0 - X @ q

    def pairwise(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.name == "l2":
            d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
            return np.maximum(d, 0.0)
        return 1.0 - A @ B.T
