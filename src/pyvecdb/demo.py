from __future__ import annotations

from copy import deepcopy

from ._demo_corpus import CORPUS


def corpus() -> dict:
    """The bundled mini corpus: documents plus labelled questions (see benchmarks/eval_rag.py)."""
    return deepcopy(CORPUS)


def load_demo(pipeline) -> int:
    docs = corpus()["docs"]
    for d in docs:
        pipeline.ingest(d["title"], d["text"], source="demo corpus", doc_id=d["id"])
    return len(docs)
