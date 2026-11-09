from __future__ import annotations

import numpy as np

from .base import Embedder, EmbeddingError


class SentenceTransformerEmbedder(Embedder):
    """Local sentence-transformers model. Install with ``pip install pyvecdb[st]``."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError("sentence-transformers is not installed: "
                                 "pip install 'pyvecdb[st]'") from e
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{model}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True,
                                             show_progress_bar=False), dtype=np.float32)
