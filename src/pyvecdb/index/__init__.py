from __future__ import annotations

from .base import Index
from .bruteforce import BruteForce
from .hnsw import HNSW
from .ivf import IVFFlat

INDEXES: dict[str, type[Index]] = {"bruteforce": BruteForce, "hnsw": HNSW, "ivf": IVFFlat}


def make_index(kind: str, dim: int, metric: str = "cosine", **params) -> Index:
    try:
        cls = INDEXES[kind]
    except KeyError:
        raise ValueError(f"unknown index {kind!r}; choose from {sorted(INDEXES)}") from None
    return cls(dim, metric, **params)


def index_from_state(cfg: dict, arrays: dict) -> Index:
    return INDEXES[cfg["kind"]].from_state(cfg, arrays)


__all__ = ["Index", "BruteForce", "HNSW", "IVFFlat", "INDEXES", "make_index", "index_from_state"]
