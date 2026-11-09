from __future__ import annotations

import numpy as np


class BackendError(RuntimeError):
    """An external model server (embedding or generation) failed."""


class EmbeddingError(BackendError):
    """Raised when an embedding backend is unreachable or returns bad data."""


class Embedder:
    """Turns text into unit-length float32 vectors."""

    name = "embedder"
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
