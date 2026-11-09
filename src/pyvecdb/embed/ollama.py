from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from .base import BackendError, Embedder, EmbeddingError

# Retrieval models trained with task prefixes (documented by their authors).
_PREFIXES = {"nomic-embed-text": ("search_document: ", "search_query: ")}


def post_json(url: str, body: dict, timeout: float,
              error: type[BackendError] = EmbeddingError) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise error(f"{url} returned HTTP {e.code}: {e.read()[:200]!r}") from e
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise error(f"cannot reach {url}: {e}") from e


def default_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    return host if host.startswith("http") else f"http://{host}"


class OllamaEmbedder(Embedder):
    """Embeddings from a local Ollama server (``ollama pull nomic-embed-text``)."""

    def __init__(self, model: str = "nomic-embed-text", host: str | None = None,
                 batch_size: int = 16, timeout: float = 120.0):
        self.model, self.host = model, (host or default_host()).rstrip("/")
        self.batch_size, self.timeout = batch_size, timeout
        self.name = f"ollama:{model}"
        base = model.split(":")[0]
        self._doc_prefix, self._query_prefix = _PREFIXES.get(base, ("", ""))
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._embed(["dimension probe"]).shape[1])
        return self._dim

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:  # current API: batched
            data = post_json(f"{self.host}/api/embed", {"model": self.model, "input": texts},
                             self.timeout)
            vecs = data["embeddings"]
        except EmbeddingError as e:
            if "HTTP 404" not in str(e):
                raise
            vecs = []  # older servers only have the one-prompt-at-a-time endpoint
            for t in texts:
                data = post_json(f"{self.host}/api/embeddings",
                                 {"model": self.model, "prompt": t}, self.timeout)
                vecs.append(data["embedding"])
        except KeyError as e:
            raise EmbeddingError(f"unexpected response from Ollama: missing {e}") from e
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim != 2 or len(arr) != len(texts):
            raise EmbeddingError("Ollama returned a malformed embedding batch")
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [self._embed([self._doc_prefix + t for t in texts[i : i + self.batch_size]])
                 for i in range(0, len(texts), self.batch_size)]
        out = np.vstack(parts)
        self._dim = out.shape[1]
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self._query_prefix + text])[0]
