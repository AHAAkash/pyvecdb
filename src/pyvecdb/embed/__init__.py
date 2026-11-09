from __future__ import annotations

from .base import BackendError, Embedder, EmbeddingError
from .hashing import HashingEmbedder
from .ollama import OllamaEmbedder
from .st import SentenceTransformerEmbedder


def get_embedder(spec: str = "hashing") -> Embedder:
    """``hashing`` | ``hashing:1024`` | ``ollama:<model>`` | ``st:<model>``."""
    kind, _, arg = spec.partition(":")
    if kind == "hashing":
        return HashingEmbedder(int(arg)) if arg else HashingEmbedder()
    if kind == "ollama":
        return OllamaEmbedder(arg or "nomic-embed-text")
    if kind in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder(arg or "all-MiniLM-L6-v2")
    raise ValueError(f"unknown embedder {spec!r}; use hashing, ollama:<model> or st:<model>")


__all__ = ["BackendError", "Embedder", "EmbeddingError", "HashingEmbedder", "OllamaEmbedder",
           "SentenceTransformerEmbedder", "get_embedder"]
